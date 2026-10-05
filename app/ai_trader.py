"""AI 账户：每天跟着早上的新闻摘要跑一次，自己决定买、卖还是不动。

分工：
- 模型（agent）负责研究和决定：可以查价格、走势、历史相似情形、和大盘比较、搜新闻，但不能直接下单
- 代码负责执行和把关：检查现金够不够、标的查不查得到、单个标的不超过账户的 40%、一天最多 3 笔
  模型的每个决定都记下理由；被代码拦下的也记下原因，页面上能看到

为什么不让模型直接调用下单工具：下单是改数据的操作，放在代码里统一检查，规则一处写清楚，也好测试。
这和提醒工具「工具返回 ok 才算做成」是同一个思路，只是这里连调用都不交给模型。

运行一次：python -m app.ai_trader
"""

from __future__ import annotations

import json
import logging
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field

from . import db
from . import portfolio as pf
from .config import get_settings
from .resolve import to_code
from .tools.market import compare_with_index, fetch_series, find_similar_history, get_history, get_quote
from .tools.news import search_news

log = logging.getLogger(__name__)
MAX_WEIGHT = 0.4  # 单个标的最多占账户总值的 40%
MAX_ORDERS = 3


class Order(BaseModel):
    action: Literal["buy", "sell"]
    symbol: str = Field(description="代码，例如 NVDA、600519.SS、AUD/CNY")
    amount_cny: float = Field(default=0, description="买入时：花多少元人民币（含手续费）")
    fraction: float = Field(default=1.0, description="卖出时：卖掉持仓的比例，0.5 是一半，1 是全部")
    reason: str = Field(description="一两句话的理由，引用你查到的具体数字或新闻")


class Decision(BaseModel):
    """今天的决定。查完需要的数据后，调用它交出决定。"""

    orders: list[Order] = Field(default_factory=list, description="今天要下的单，最多 3 笔；不动就留空")
    summary: str = Field(description="一两句话：今天怎么看、为什么这样操作（或不操作）")


TRADER_PROMPT = """你在管理一个模拟投资账户，目标是长期跑赢「沪深300买了不动」和「标普500买了不动」。这是学习用的模拟盘，不是真钱。

规则：
- 先用工具查数据再决定：价格、近 30 天走势、历史相似情形、和大盘比较、新闻。不要凭记忆编数字
- 每笔买入股票收 0.1% 手续费，外币资产还要再收 0.3% 换汇费；频繁买卖会被手续费吃掉，没有把握就不动
- 单个标的买完后不能超过账户总值的 40%，一天最多 3 笔
- 可以买关注列表里的标的，也可以买你研究过、觉得更好的标的
- 理由要具体，写出你依据的数字或新闻；之后会有人对照结果复盘你的理由
- 最后调用 Decision 交出决定"""


def _state_text(state: dict, watch: list[str], digest: dict | None) -> str:
    holdings = "\n".join(f"- {h['symbol']}：{h['quantity']} 份，市值 {h['value_cny']} 元，盈亏 {h.get('pnl_pct')}%"
                         for h in state["holdings"]) or "（空仓）"
    bench = "、".join(f"{b['name']} {b['return_pct']:+}%" for b in state["benchmarks"])
    parts = [f"账户：现金 {state['cash']} 元，总值 {state['total']} 元，起始 {state['initial_cash']} 元，"
             f"收益 {state['return_pct']:+}%（同期 {bench}）", f"持仓：\n{holdings}", f"关注列表：{'、'.join(watch)}"]
    if digest:
        lines = [f"- {i['symbol']} {i['direction']}：{i['what']}（{i['reason']}）" for i in digest.get("items", [])]
        parts.append(f"今天的新闻摘要：{digest.get('overview', '')}\n" + "\n".join(lines))
    return "\n\n".join(parts)


def llm_decide(state_text: str) -> tuple[Decision | None, list[dict]]:
    """让 agent 研究并给出决定。返回（决定，调用过的工具）。"""
    from .llm import get_model

    agent = create_agent(get_model(), tools=[get_quote, get_history, find_similar_history, compare_with_index,
                                             search_news],
                         system_prompt=TRADER_PROMPT, response_format=ToolStrategy(Decision))
    result = agent.invoke({"messages": [{"role": "user", "content": state_text}]})
    calls = [{"name": c["name"], "args": c["args"]} for m in result["messages"]
             for c in getattr(m, "tool_calls", []) or [] if c["name"] != Decision.__name__]
    return result.get("structured_response"), calls


def demo_decide(state: dict, digest: dict | None) -> Decision:
    """演示模式：按新闻摘要的方向机械操作，只为让流程能跑通，不是真的判断。"""
    orders = []
    held = {h["symbol"] for h in state["holdings"]}
    for i in (digest or {}).get("items", []):
        if i["direction"] == "偏跌" and i["symbol"] in held:
            orders.append(Order(action="sell", symbol=i["symbol"], fraction=0.5, reason=f"演示规则：新闻偏跌（{i['what']}）"))
            held.discard(i["symbol"])
        elif i["direction"] == "偏涨" and state["cash"] > 100:
            orders.append(Order(action="buy", symbol=i["symbol"], amount_cny=round(state["cash"] * 0.2, 2),
                                reason=f"演示规则：新闻偏涨（{i['what']}）"))
    return Decision(orders=orders[:MAX_ORDERS], summary="演示模式：按新闻方向机械买卖，不代表真实判断。")


def execute(decision: Decision, fetch=fetch_series) -> list[dict]:
    """按代码规则检查并执行决定。每笔都返回结果：做成了，或者为什么没做。"""
    results = []
    for i, o in enumerate(decision.orders):
        row = o.model_dump()
        if i >= MAX_ORDERS:
            results.append({**row, "ok": False, "error": f"一天最多 {MAX_ORDERS} 笔，这笔没做"})
            continue
        try:
            sym = to_code(o.symbol)
            t = (pf.buy("ai", sym, o.amount_cny, o.reason, fetch, max_weight=MAX_WEIGHT) if o.action == "buy"
                 else pf.sell("ai", sym, o.fraction, o.reason, fetch))
            results.append({**row, "symbol": sym, "ok": True, "trade_id": t.id, "filled_cny": t.amount_cny,
                            "fee_cny": t.fee_cny})
        except Exception as e:  # 现金不够、超过 40%、查不到价格……都记下来
            results.append({**row, "ok": False, "error": str(e)})
    return results


def run_once(digest: dict | None = None, decide=None, fetch=fetch_series) -> dict:
    """AI 账户的一天：读账户状态 → 决定 → 执行 → 存下这次运行。AI 账户没开就什么都不做。"""
    from .digest import watched_symbols

    state = pf.summary("ai", fetch)
    if not state:
        return {"ok": False, "error": "AI 账户还没开（用户开模拟账户时会一起开）"}
    text = _state_text(state, watched_symbols(), digest)
    calls: list[dict] = []
    if decide:
        decision = decide(text)
    elif get_settings().demo_mode:
        decision = demo_decide(state, digest)
    else:
        decision, calls = llm_decide(text)
    if decision is None:
        decision = Decision(summary="模型这次没有交出决定，今天不操作。")
    results = execute(decision, fetch)
    after = pf.summary("ai", fetch, trades=0)
    pf.summary("user", fetch, trades=0)  # 顺便给用户账户记下今天的市值，曲线每天都有点
    record = {"summary": decision.summary, "orders": results, "research": calls,
              "total": after["total"], "return_pct": after["return_pct"]}
    with db.session() as s:
        s.add(db.AiDecision(data=record))
        s.commit()
    return {"ok": True, **record}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    print(json.dumps(run_once(), ensure_ascii=False, indent=2))
