"""把用户输入的名字变成行情代码：先查内置名单，再查缓存，都没有才问模型，最后用行情接口核实。

为什么不全部交给模型：内置名单和缓存免费、瞬间返回、不会出错；
模型能认识名单外的公司（三星、Shopify、任何一家），但可能记错代码，所以它给出的代码必须查得到数据才算数。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field
from sqlalchemy import select

from . import db
from .symbols import UNLISTED, UnknownSymbol, check_known, normalize


class Resolved(BaseModel):
    """模型识别的结果。"""

    listed: bool = Field(description="这家公司或这个品种有没有公开上市交易")
    symbol: str = Field(default="", description="Yahoo Finance 的代码，例如 005930.KS、7203.T、SHOP、600519.SS；没上市就留空")
    name: str = Field(description="公司或品种的正式中文名")
    exchange: str = Field(default="", description="交易所，例如 韩国交易所、纳斯达克")
    note: str = Field(default="", description="一句话说明，比如没上市的原因、或有多个上市地时选了哪个")


RESOLVE_PROMPT = """用户在股票查询框里输入了「{text}」。判断它指的是哪家上市公司或哪个可交易品种，给出 Yahoo Finance 的代码。
规则：
- A 股用 .SS（上海）或 .SZ（深圳），港股用四位数字加 .HK，韩股 .KS，日股 .T，澳股 .AX，美股直接写代码
- 同一家公司在多个地方上市时，优先选它的主要上市地，并在 note 里说明
- 没有上市（比如华为）就把 listed 设为 false，symbol 留空
- 不确定就把 listed 设为 false，在 note 里说明不确定，不要猜"""


@dataclass
class Result:
    symbol: str
    name: str = ""
    source: str = "名单"  # 名单 / 缓存 / 模型
    note: str = ""


Asker = Callable[[str], Resolved]


def llm_asker() -> Asker | None:
    """用配置里的模型做识别。演示模式没有真模型，返回 None。"""
    from .config import get_settings
    from .llm import get_model

    if get_settings().demo_mode:
        return None
    # 用函数调用的方式拿结构化结果：DeepSeek 等兼容接口都支持
    structured = get_model().with_structured_output(Resolved, method="function_calling")
    return lambda text: structured.invoke(RESOLVE_PROMPT.format(text=text))


def resolve(text: str, ask: Asker | None = None, verify: Callable[[str], object] | None = None) -> Result:
    """返回核实过的代码。认不出、没上市、模型给的代码查不到数据时，抛出 UnknownSymbol（信息直接给用户看）。"""
    text = text.strip()
    sym = normalize(text)
    if sym in UNLISTED:
        check_known(sym)  # 抛出「没有上市」
    try:
        check_known(sym)
        return Result(sym)
    except UnknownSymbol as not_a_code:
        cached = _cached(text)
        if cached:
            return cached
        if ask is None:
            raise not_a_code
    r = ask(text)
    if not r.listed or not r.symbol:
        raise UnknownSymbol(f"{r.name or text}：{r.note or '没有找到上市的股票'}")
    code = normalize(r.symbol)
    try:
        check_known(code)
        if verify:
            verify(code)
    except Exception as e:
        raise UnknownSymbol(f"模型认为「{text}」是 {code}（{r.name}），但行情接口查不到这个代码：{e}")
    with db.session() as s:
        s.add(db.SymbolAlias(alias=text, symbol=code, name=r.name))
        s.commit()
    return Result(code, r.name, "模型", r.note)


def _cached(text: str) -> Result | None:
    with db.session() as s:
        row = s.scalars(select(db.SymbolAlias).where(db.SymbolAlias.alias == text)).first()
    return Result(row.symbol, row.name, "缓存") if row else None


def to_code(text: str) -> str:
    """工具里用：名单和缓存里有就换成代码，没有就原样返回（交给行情接口判断）。不调用模型。"""
    sym = normalize(text)
    try:
        check_known(sym)
        return sym
    except UnknownSymbol:
        cached = _cached(text.strip())
        return cached.symbol if cached else sym
