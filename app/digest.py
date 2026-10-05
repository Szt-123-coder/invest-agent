"""每日新闻摘要：给关注的每个标的搜新闻，让模型提炼「发生了什么、影响谁、偏涨还是偏跌」，核对后推送到微信。

这里没有用 agent，而是固定流程（workflow）：搜新闻 → 模型提炼 → 代码核对 → 推送。
步骤每天都一样，不需要模型自己决定下一步，固定流程更便宜、更稳定，也更好评测。
模型只做它擅长的一步：读新闻、归类、判断方向。

运行一次：python -m app.digest
服务里定时运行：设环境变量 DIGEST_TIMES=08:00,23:00（按 TIMEZONE 的当地时间）
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from sqlalchemy import select

from . import db
from .config import get_settings
from .symbols import CURRENCIES, display_name, is_pair
from .tools.market import fetch_series
from .tools.news import fetch_news

log = logging.getLogger(__name__)
DEFAULT_SYMBOLS = ["USD/CNY", "AUD/CNY"]  # 关注列表为空时用
DIRECTIONS = ("偏涨", "偏跌", "看不出")


class NewsImpact(BaseModel):
    news_id: int = Field(description="新闻编号，必须是上面列出的编号之一")
    symbol: str = Field(description="受影响的标的代码，必须是关注列表里的代码，原样照抄")
    what: str = Field(description="一句话：发生了什么，只写新闻里有的事实")
    direction: Literal["偏涨", "偏跌", "看不出"] = Field(
        description="这条新闻让这个标的更可能涨还是跌。货币对 AUD/CNY 偏涨表示澳元相对人民币走强。拿不准就填看不出")
    reason: str = Field(description="一句话说明为什么是这个方向")


class Extraction(BaseModel):
    items: list[NewsImpact] = Field(default_factory=list, description="每条有用的新闻一项；和关注标的无关的新闻不要填")
    overview: str = Field(description="两三句话的今日总览，只根据下面的新闻和价格")


EXTRACT_PROMPT = """你在给用户写每日新闻摘要。用户关注：{symbols}。

今天的价格：
{prices}

搜到的新闻（编号：标题｜摘要）：
{news}

请逐条判断：这条新闻和哪个关注标的有关、发生了什么、让它偏涨还是偏跌。
规则：
- news_id 只能用上面的编号，symbol 只能用关注列表里的代码
- 一条新闻影响多个标的，就分成多项
- 和关注标的无关、或者只是重复别的新闻的，不要填
- 只写新闻里有的事实，不要补充新闻里没有的数字
- 方向拿不准就填「看不出」，不要硬猜"""

Extractor = Callable[[str], Extraction]


def query_for(symbol: str) -> str:
    """标的 → 搜新闻用的关键词：AUD/CNY → 澳元 人民币 汇率；NVDA → 英伟达 股价。"""
    if is_pair(symbol):
        names = {v: k for k, v in reversed(list(CURRENCIES.items()))}  # 每个代码取第一个中文名
        base, quote = symbol.split("/")
        return f"{names.get(base, base)} {names.get(quote, quote)} 汇率"
    name = display_name(symbol).split("（")[0]
    return f"{name} 股价"


def llm_extractor() -> Extractor | None:
    if get_settings().demo_mode:
        return None
    from .llm import get_model

    structured = get_model().with_structured_output(Extraction, method="function_calling")
    return lambda prompt: structured.invoke(prompt)


DEMO_HINTS = {"AUD": ("澳", "铁矿石"), "USD": ("美联储", "美元"), "CNY": ("中国", "人民币"), "EUR": ("欧洲", "欧元")}


def demo_extract(news: list[dict], symbols: list[str]) -> Extraction:
    """演示模式没有真模型：按关键词粗略归类，只为让页面和推送能跑通。"""
    up, down = ("加息", "上涨", "走强", "升值"), ("下跌", "放缓", "承压", "贬值")
    items = []
    for n in news:
        text = n["title"] + n["snippet"]
        for sym in symbols:
            base, _, quote = sym.partition("/")
            if any(w in text for w in DEMO_HINTS.get(base, (base,))):
                direction = "偏涨" if any(w in text for w in up) else "偏跌" if any(w in text for w in down) else "看不出"
            elif quote and any(w in text for w in DEMO_HINTS.get(quote, ())):
                direction = "看不出"  # 只和人民币有关的新闻，对各个货币对的方向不好说
            else:
                continue
            items.append(NewsImpact(news_id=n["id"], symbol=sym, what=n["title"], direction=direction,
                                    reason="演示模式：按关键词粗略判断"))
    return Extraction(items=items, overview=f"演示模式：示例新闻 {len(news)} 条，按关键词归类，不代表真实判断。")


def check(ex: Extraction, news: list[dict], symbols: list[str]) -> tuple[list[dict], list[str]]:
    """核对模型的提炼结果。新闻编号对不上、标的不在关注列表、重复的项都丢掉，并记下原因。

    链接和标题由代码按编号填进去，模型没法编出一条不存在的新闻。
    """
    by_id = {n["id"]: n for n in news}
    kept, dropped, seen = [], [], set()
    for it in ex.items:
        if it.news_id not in by_id:
            dropped.append(f"新闻编号 {it.news_id} 不存在")
        elif it.symbol not in symbols:
            dropped.append(f"新闻 {it.news_id} 填的标的 {it.symbol} 不在关注列表")
        elif (it.news_id, it.symbol) in seen:
            dropped.append(f"新闻 {it.news_id} 对 {it.symbol} 重复")
        elif not it.what.strip():
            dropped.append(f"新闻 {it.news_id} 没写发生了什么")
        else:
            seen.add((it.news_id, it.symbol))
            n = by_id[it.news_id]
            kept.append({**it.model_dump(), "title": n["title"], "url": n["source"]})
    return kept, dropped


def _price(symbol: str, fetch) -> dict:
    try:
        pts = fetch(symbol, 5)
        p0, (d1, p1) = pts[-2][1] if len(pts) > 1 else pts[-1][1], pts[-1]
        return {"symbol": symbol, "price": p1, "date": d1, "change_pct": round((p1 / p0 - 1) * 100, 2)}
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def watched_symbols() -> list[str]:
    with db.session() as s:
        return list(s.scalars(select(db.WatchItem.symbol).order_by(db.WatchItem.id))) or DEFAULT_SYMBOLS


def build(symbols: list[str] | None = None, extract: Extractor | None = None,
          news_fn=fetch_news, fetch=fetch_series, per_symbol: int = 4) -> dict:
    """生成一份摘要（不推送）。extract 为 None 时：有真模型就用模型，否则用演示规则。"""
    symbols = symbols or watched_symbols()
    prices = [_price(sym, fetch) for sym in symbols]
    news, urls, errors = [], set(), []
    for sym in symbols:
        try:
            found = news_fn(query_for(sym), per_symbol, days=2)
        except Exception as e:
            errors.append(f"{sym} 新闻搜索失败：{e}")
            continue
        for n in found:
            key = n["source"] if n.get("source", "").startswith("http") else n.get("title")
            if key in urls:  # 不同标的搜到同一条新闻，只留一份，让模型自己判断影响谁
                continue
            urls.add(key)
            news.append({**n, "id": len(news) + 1, "symbol": sym})
    extract = extract or llm_extractor()
    if not news:
        ex = Extraction(overview="今天没有搜到相关新闻。")
    elif extract is None:
        ex = demo_extract(news, symbols)
    else:
        price_text = "\n".join(f"- {p['symbol']}：{p['price']}（{p['change_pct']:+}%）" if "price" in p
                               else f"- {p['symbol']}：查不到" for p in prices)
        news_text = "\n".join(f"{n['id']}：{n['title']}｜{n['snippet']}" for n in news)
        ex = extract(EXTRACT_PROMPT.format(symbols="、".join(symbols), prices=price_text, news=news_text))
    items, dropped = check(ex, news, symbols)
    tz = ZoneInfo(get_settings().timezone)
    return {"date": datetime.now(tz).strftime("%Y-%m-%d %H:%M"), "symbols": symbols, "prices": prices,
            "overview": ex.overview, "items": items, "dropped": dropped, "errors": errors,
            "news_count": len(news), "demo": extract is None, "default_symbols": symbols == DEFAULT_SYMBOLS}


ARROW = {"偏涨": "↑", "偏跌": "↓", "看不出": "·"}


def to_markdown(d: dict) -> str:
    """推送到微信的内容（PushPlus 的 markdown 模板）。"""
    lines = [f"**{d['overview']}**", ""]
    for p in d["prices"]:
        lines.append(f"- {display_name(p['symbol'])}：{p['price']}（{p['change_pct']:+}%）" if "price" in p
                     else f"- {display_name(p['symbol'])}：暂时查不到价格")
    for sym in d["symbols"]:
        its = [i for i in d["items"] if i["symbol"] == sym]
        lines += ["", f"### {display_name(sym)}"]
        lines += [f"- {ARROW[i['direction']]} {i['direction']}｜{i['what']}（{i['reason']}）[原文]({i['url']})"
                  if i["url"].startswith("http") else f"- {ARROW[i['direction']]} {i['direction']}｜{i['what']}（{i['reason']}）"
                  for i in its] or ["- 没有相关新闻"]
    lines += ["", "仅供学习参考，不构成投资建议。"]
    return "\n".join(lines)


def run_once(extract: Extractor | None = None, pusher=None, **kwargs) -> dict:
    """生成、保存、推送一份摘要。"""
    from .jobs import push

    d = build(extract=extract, **kwargs)
    with db.session() as s:
        s.add(db.Digest(data=d))
        s.commit()
    title = f"每日新闻摘要 {d['date'][:10]}"
    d["pushed"] = (pusher or push)(title, to_markdown(d), "markdown")
    return d


def due(now: datetime, times: list[str], last: datetime | None) -> bool:
    """现在是否该发：已经过了今天某个设定时间，而且那之后还没发过。"""
    slots = [now.replace(hour=int(t[:2]), minute=int(t[3:5]), second=0, microsecond=0) for t in times]
    passed = [x for x in slots if x <= now]
    # 超过 1 小时就不补发：服务停了几个小时再启动时，不发一份过时的摘要
    return bool(passed) and (last is None or last < max(passed)) and now - max(passed) < timedelta(hours=1)


def last_sent() -> datetime | None:
    with db.session() as s:
        t = s.scalar(select(db.Digest.created_at).order_by(db.Digest.id.desc()))
    if t and t.tzinfo is None:  # SQLite 读出来没有时区，存的是 UTC
        t = t.replace(tzinfo=timezone.utc)
    return t


def start_background(times: list[str]) -> None:
    tz = ZoneInfo(get_settings().timezone)

    def loop():
        while True:
            try:
                last = last_sent()
                if due(datetime.now(tz), times, last.astimezone(tz) if last else None):
                    run_once()
            except Exception:
                log.exception("每日摘要失败")
            time.sleep(60)

    threading.Thread(target=loop, daemon=True, name="digest").start()


def digest_times() -> list[str]:
    return [t.strip() for t in os.getenv("DIGEST_TIMES", "").split(",") if t.strip()]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    result = run_once()
    print(to_markdown(result))
    if result["dropped"]:
        print("\n核对时丢掉：\n" + "\n".join(result["dropped"]))
