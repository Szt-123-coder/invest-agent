"""用户模拟账户的工具：开户、买、卖、看账户。只操作用户自己的账户，AI 账户由 app/ai_trader.py 管。

和提醒工具一样，返回 ok=false 就是没做成，模型必须如实告诉用户。
"""

from __future__ import annotations

from langchain_core.tools import tool

from .. import portfolio as pf
from ..db import dumps
from ..resolve import to_code


def _fail(msg: str) -> str:
    return dumps({"ok": False, "error": msg})


@tool
def open_paper_account(amount_cny: float) -> str:
    """给用户开一个模拟投资账户，起始资金 amount_cny 元人民币（用户自己定）。AI 账户还没开时，会用同样的金额一起开，方便比较。"""
    try:
        a = pf.open_account("user", float(amount_cny))
    except pf.TradeError as e:
        return _fail(str(e))
    ai_opened = False
    try:
        pf.open_account("ai", float(amount_cny))
        ai_opened = True
    except pf.TradeError:
        pass  # AI 账户已经有了
    return dumps({"ok": True, "account": "user", "initial_cash": a.initial_cash, "ai_account_opened": ai_opened})


@tool
def paper_buy(symbol: str, amount_cny: float) -> str:
    """在用户的模拟账户里花 amount_cny 元人民币（含手续费）买入 symbol。只有用户明确说要买、并给了金额时才调用。
    symbol 可以是股票（NVDA、600519.SS、茅台）或外币（澳元、AUD/CNY）。股票手续费 0.1%，换汇 0.3%，可以买零股。"""
    sym = to_code(symbol)
    try:
        t = pf.buy("user", sym, float(amount_cny))
    except pf.TradeError as e:
        return _fail(str(e))
    except Exception as e:
        return _fail(f"查不到 {sym} 的价格，没有买：{e}")
    return dumps({"ok": True, "trade_id": t.id, "symbol": sym, "side": "buy", "quantity": round(t.quantity, 6),
                  "price": t.price, "fx": round(t.fx, 4), "amount_cny": t.amount_cny, "fee_cny": t.fee_cny})


@tool
def paper_sell(symbol: str, fraction: float = 1.0) -> str:
    """在用户的模拟账户里卖出 symbol 持仓的一部分：fraction=1 全部卖掉，0.5 卖一半。只有用户明确说要卖时才调用。"""
    sym = to_code(symbol)
    try:
        t = pf.sell("user", sym, float(fraction))
    except pf.TradeError as e:
        return _fail(str(e))
    except Exception as e:
        return _fail(f"查不到 {sym} 的价格，没有卖：{e}")
    return dumps({"ok": True, "trade_id": t.id, "symbol": sym, "side": "sell", "quantity": round(t.quantity, 6),
                  "price": t.price, "fx": round(t.fx, 4), "amount_cny": t.amount_cny, "fee_cny": t.fee_cny})


@tool
def paper_account() -> str:
    """查看用户模拟账户：现金、持仓、总值、收益率、和沪深300 / 标普500「买了不动」的比较，以及 AI 账户的总收益。"""
    user = pf.summary("user", trades=5)
    if not user:
        return _fail("还没开模拟账户，先说一个起始金额，比如「开一个 10000 元的模拟账户」")
    ai = pf.summary("ai", trades=0)
    keep = ("initial_cash", "cash", "holdings", "total", "return_pct", "benchmarks", "trades")
    return dumps({"ok": True, "user": {k: user[k] for k in keep},
                  "ai": {k: ai[k] for k in ("total", "return_pct")} if ai else None})
