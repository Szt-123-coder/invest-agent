"""模拟投资的记账：开户、买、卖、算市值和收益。用户账户和 AI 账户用的是同一套代码。

规则（sand 定的）：
- 现金是人民币，起始金额开户时自己填；AI 账户默认用同样的金额
- 汇率和股票都能买，外币资产按当时的真实汇率换算成人民币；允许买零股
- 手续费：股票每笔 0.1%；换外汇 0.3%（相当于银行买卖价差）。买外国股票要先换汇，所以两样都收
- 基准：开户当天用同样的钱买沪深300、标普500，之后一直不动

这里只做记账和检查，不做决定。谁来决定（用户下单还是 AI）在别的地方。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import select

from . import db
from .config import get_settings
from .symbols import is_pair
from .tools.market import fetch_series

STOCK_FEE = 0.001
FX_FEE = 0.003
BENCHMARKS = {"000300.SS": "沪深300", "^GSPC": "标普500"}
BENCHMARK_FALLBACK = {"000300.SS": "000001.SS"}  # Yahoo 的沪深300 偶尔查不到，就用上证指数代替
SUFFIX_CURRENCY = {".SS": "CNY", ".SZ": "CNY", ".HK": "HKD", ".AX": "AUD", ".KS": "KRW", ".KQ": "KRW", ".T": "JPY",
                   ".L": "GBP", ".PA": "EUR", ".DE": "EUR"}
OWNERS = {"user": "你的账户", "ai": "AI 账户"}

Fetch = Callable[[str, int], list]


class TradeError(ValueError):
    """下单没成功，message 直接给用户看。"""


def currency_of(symbol: str) -> str:
    """标的用什么货币计价。货币对 AUD/CNY 的价格本身就是人民币；股票看交易所后缀，没后缀的是美股。"""
    if is_pair(symbol):
        return symbol.split("/")[1]
    if symbol.startswith("^"):
        return {"^GSPC": "USD", "^HSI": "HKD", "^AXJO": "AUD", "^KS11": "KRW", "^N225": "JPY"}.get(symbol, "USD")
    return next((c for suf, c in SUFFIX_CURRENCY.items() if symbol.endswith(suf)), "USD")


def fee_rate(symbol: str) -> float:
    if is_pair(symbol):
        return FX_FEE
    return STOCK_FEE + (FX_FEE if currency_of(symbol) != "CNY" else 0)


class Market:
    """一次操作里查过的价格记下来，同一个标的不重复查。"""

    def __init__(self, fetch: Fetch = fetch_series):
        self.fetch = fetch
        self.cache: dict[str, tuple[str, float]] = {}

    def last(self, symbol: str) -> tuple[str, float]:
        if symbol not in self.cache:
            pts = self.fetch(symbol, 5)
            if not pts:
                raise TradeError(f"{symbol} 查不到价格")
            self.cache[symbol] = pts[-1]
        return self.cache[symbol]

    def fx(self, currency: str) -> float:
        return 1.0 if currency == "CNY" else self.last(f"{currency}/CNY")[1]

    def quote(self, symbol: str) -> tuple[float, float]:
        """返回（价格，汇率）：价格是计价货币，汇率是 1 单位计价货币值多少人民币。"""
        return self.last(symbol)[1], self.fx(currency_of(symbol))


def today() -> str:
    return datetime.now(ZoneInfo(get_settings().timezone)).date().isoformat()


def get_account(s, owner: str) -> db.Account | None:
    return s.scalar(select(db.Account).where(db.Account.owner == owner))


def _benchmark_units(amount: float, m: Market) -> dict:
    units = {}
    for code in BENCHMARKS:
        for c in [code, BENCHMARK_FALLBACK.get(code)]:
            if not c:
                continue
            try:
                price, fx = m.quote(c)
            except Exception:
                continue
            units[code] = {"symbol": c, "units": amount / (price * fx)}
            break
    return units


def open_account(owner: str, amount: float, fetch: Fetch = fetch_series) -> db.Account:
    if owner not in OWNERS:
        raise TradeError(f"不认识的账户 {owner}")
    if not 100 <= amount <= 100_000_000:
        raise TradeError("起始金额要在 100 到 1 亿元之间")
    m = Market(fetch)
    with db.session() as s:
        if get_account(s, owner):
            raise TradeError(f"{OWNERS[owner]}已经开过了")
        a = db.Account(owner=owner, initial_cash=round(amount, 2), cash=round(amount, 2),
                       benchmarks=_benchmark_units(amount, m))
        s.add(a)
        s.commit()
        snapshot(s, a, m)
        return a


def positions(s, account: db.Account) -> dict[str, float]:
    """每个标的现在持有多少份（买的减去卖的）。"""
    out: dict[str, float] = {}
    for t in s.scalars(select(db.Trade).where(db.Trade.account_id == account.id).order_by(db.Trade.id)):
        out[t.symbol] = out.get(t.symbol, 0) + (t.quantity if t.side == "buy" else -t.quantity)
    return {k: v for k, v in out.items() if v > 1e-9}


def _cost(s, account: db.Account, symbol: str) -> float:
    """还拿着的这部分花了多少人民币（含手续费），按平均成本算。"""
    qty = cost = 0.0
    for t in s.scalars(select(db.Trade).where(db.Trade.account_id == account.id, db.Trade.symbol == symbol)
                       .order_by(db.Trade.id)):
        if t.side == "buy":
            qty, cost = qty + t.quantity, cost + t.amount_cny + t.fee_cny
        elif qty:
            cost *= (qty - t.quantity) / qty
            qty -= t.quantity
    return cost


def buy(owner: str, symbol: str, amount_cny: float, reason: str = "", fetch: Fetch = fetch_series,
        max_weight: float | None = None) -> db.Trade:
    """花 amount_cny 元（含手续费）买入。max_weight：买完后这个标的最多占账户总值的比例（给 AI 用的风控）。"""
    if amount_cny <= 0:
        raise TradeError("买入金额必须大于 0")
    m = Market(fetch)
    with db.session() as s:
        a = get_account(s, owner)
        if not a:
            raise TradeError(f"{OWNERS.get(owner, owner)}还没开户。请先问用户想用多少起始资金开户，不要自己替用户开")
        if amount_cny > a.cash + 1e-6:
            raise TradeError(f"现金不够：想买 {amount_cny:.2f} 元，账户里只有 {a.cash:.2f} 元")
        price, fx = m.quote(symbol)
        if max_weight is not None:
            total = value(s, a, m)["total"]
            held = positions(s, a).get(symbol, 0) * price * fx
            if held + amount_cny > total * max_weight + 1e-6:
                raise TradeError(f"买完后 {symbol} 会超过账户总值的 {max_weight:.0%}，单个标的不能押太多")
        fee = round(amount_cny * fee_rate(symbol), 2)
        gross = amount_cny - fee
        t = db.Trade(account_id=a.id, symbol=symbol, side="buy", quantity=gross / (price * fx), price=price, fx=fx,
                     amount_cny=round(gross, 2), fee_cny=fee, reason=reason)
        a.cash = round(a.cash - amount_cny, 2)
        s.add(t)
        s.commit()
        snapshot(s, a, m)
        return t


def sell(owner: str, symbol: str, fraction: float = 1.0, reason: str = "", fetch: Fetch = fetch_series) -> db.Trade:
    """卖掉持仓的 fraction（0.5 = 一半，1 = 全部），钱换回人民币。"""
    if not 0 < fraction <= 1:
        raise TradeError("卖出比例要在 0 到 1 之间，比如 0.5 是卖一半")
    m = Market(fetch)
    with db.session() as s:
        a = get_account(s, owner)
        if not a:
            raise TradeError(f"{OWNERS.get(owner, owner)}还没开户")
        held = positions(s, a).get(symbol, 0)
        if held <= 0:
            raise TradeError(f"账户里没有 {symbol}，没法卖")
        price, fx = m.quote(symbol)
        qty = held * fraction
        gross = qty * price * fx
        fee = round(gross * fee_rate(symbol), 2)
        t = db.Trade(account_id=a.id, symbol=symbol, side="sell", quantity=qty, price=price, fx=fx,
                     amount_cny=round(gross, 2), fee_cny=fee, reason=reason)
        a.cash = round(a.cash + gross - fee, 2)
        s.add(t)
        s.commit()
        snapshot(s, a, m)
        return t


def value(s, a: db.Account, m: Market) -> dict:
    """账户现在值多少钱：现金 + 每个持仓按最新价和汇率折成人民币。"""
    holdings = []
    for sym, qty in positions(s, a).items():
        try:
            price, fx = m.quote(sym)
        except Exception:  # 查不到最新价就按成本算，并标出来
            cost = _cost(s, a, sym)
            holdings.append({"symbol": sym, "quantity": qty, "value_cny": round(cost, 2), "cost_cny": round(cost, 2),
                             "stale": True})
            continue
        v = qty * price * fx
        cost = _cost(s, a, sym)
        holdings.append({"symbol": sym, "quantity": round(qty, 6), "price": price, "currency": currency_of(sym),
                         "fx": round(fx, 4), "value_cny": round(v, 2), "cost_cny": round(cost, 2),
                         "pnl_pct": round((v / cost - 1) * 100, 2) if cost else None})
    total = a.cash + sum(h["value_cny"] for h in holdings)
    return {"cash": round(a.cash, 2), "holdings": holdings, "total": round(total, 2),
            "return_pct": round((total / a.initial_cash - 1) * 100, 2)}


def benchmark_values(a: db.Account, m: Market) -> dict:
    out = {}
    for code, b in (a.benchmarks or {}).items():
        try:
            price, fx = m.quote(b["symbol"])
        except Exception:
            continue
        out[code] = round(b["units"] * price * fx, 2)
    return out


def snapshot(s, a: db.Account, m: Market) -> None:
    """记下今天的市值（同一天覆盖），画曲线用。"""
    day = today()
    v = value(s, a, m)["total"]
    bench = benchmark_values(a, m)
    row = s.scalar(select(db.Snapshot).where(db.Snapshot.account_id == a.id, db.Snapshot.day == day))
    if row:
        row.value_cny, row.benchmarks = v, bench
    else:
        s.add(db.Snapshot(account_id=a.id, day=day, value_cny=v, benchmarks=bench))
    s.commit()


def summary(owner: str, fetch: Fetch = fetch_series, trades: int = 20) -> dict | None:
    """一个账户的全部信息：市值、持仓、和基准比、最近的交易、收益曲线。"""
    m = Market(fetch)
    with db.session() as s:
        a = get_account(s, owner)
        if not a:
            return None
        snapshot(s, a, m)
        v = value(s, a, m)
        bench = benchmark_values(a, m)
        rows = s.scalars(select(db.Trade).where(db.Trade.account_id == a.id).order_by(db.Trade.id.desc())
                         .limit(trades)).all()
        curve = s.scalars(select(db.Snapshot).where(db.Snapshot.account_id == a.id).order_by(db.Snapshot.day)).all()
        return {"owner": owner, "name": OWNERS[owner], "initial_cash": a.initial_cash,
                "opened": a.created_at.date().isoformat(), **v,
                "benchmarks": [{"code": c, "name": BENCHMARKS[c], "symbol": a.benchmarks[c]["symbol"], "value_cny": x,
                                "return_pct": round((x / a.initial_cash - 1) * 100, 2)} for c, x in bench.items()],
                "trades": [{"id": t.id, "symbol": t.symbol, "side": t.side, "quantity": round(t.quantity, 6),
                            "price": t.price, "fx": round(t.fx, 4), "amount_cny": t.amount_cny, "fee_cny": t.fee_cny,
                            "reason": t.reason, "at": t.created_at.isoformat()} for t in rows],
                "curve": [{"day": c.day, "return_pct": round((c.value_cny / a.initial_cash - 1) * 100, 2),
                           "benchmarks": {k: round((x / a.initial_cash - 1) * 100, 2) for k, x in c.benchmarks.items()}}
                          for c in curve]}
