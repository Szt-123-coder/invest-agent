"""行情工具：实时价和历史走势。数据来自 Yahoo Finance；演示模式用固定的假数据。"""

from __future__ import annotations

import hashlib
import math
import statistics
from datetime import date, timedelta

import httpx
from langchain_core.tools import tool

from ..config import get_settings
from ..db import dumps
from ..symbols import normalize, yahoo_ticker

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _fake_series(symbol: str, days: int) -> list[tuple[str, float]]:
    """按代码生成一条稳定的假走势，同一个代码每次都一样，方便演示和测试。"""
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    base = {"AUD/CNY": 4.62, "USD/CNY": 7.12, "EUR/CNY": 7.75}.get(symbol, 50 + seed % 300)
    today = date(2026, 10, 2)
    total = 365  # 先生成一整年，再取最后 days 天，这样不同查询看到的是同一条走势
    out = []
    for i in range(total):
        d = today - timedelta(days=total - 1 - i)
        wave = math.sin((i + seed % 7) / 4) * 0.012 + (i - total) * 0.00004
        out.append((d.isoformat(), round(base * (1 + wave), 4)))
    return out[-days:]


def fetch_series(symbol: str, days: int = 30) -> list[tuple[str, float]]:
    if get_settings().demo_mode:
        return _fake_series(symbol, days)
    rng = "1mo" if days <= 31 else "3mo" if days <= 92 else "1y"
    r = httpx.get(YAHOO.format(ticker=yahoo_ticker(symbol)), params={"range": rng, "interval": "1d"},
                  headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    pts = [(date.fromtimestamp(t).isoformat(), round(c, 4)) for t, c in zip(res["timestamp"], closes) if c is not None]
    return pts[-days:]


@tool
def get_quote(symbol: str) -> str:
    """查一个标的的最新价格。symbol 可以是货币对（AUD/CNY）、股票代码（AAPL、600519.SS）或中文名（澳元、茅台）。"""
    sym = normalize(symbol)
    try:
        pts = fetch_series(sym, 5)
    except Exception as e:  # 网络或代码错误都如实告诉模型
        return dumps({"ok": False, "error": f"查不到 {sym} 的行情：{e}"})
    if not pts:
        return dumps({"ok": False, "error": f"{sym} 没有数据"})
    p0 = pts[-2][1] if len(pts) > 1 else pts[-1][1]
    d1, p1 = pts[-1]
    return dumps({"ok": True, "symbol": sym, "price": p1, "date": d1,
                  "change_pct": round((p1 / p0 - 1) * 100, 2)})


@tool
def get_history(symbol: str, days: int = 30) -> str:
    """查一个标的最近 days 天（默认 30，最多 365）的走势统计：最高、最低、平均、区间涨跌和波动。"""
    sym = normalize(symbol)
    days = max(5, min(int(days), 365))
    try:
        pts = fetch_series(sym, days)
    except Exception as e:
        return dumps({"ok": False, "error": f"查不到 {sym} 的历史数据：{e}"})
    if len(pts) < 2:
        return dumps({"ok": False, "error": f"{sym} 数据不足"})
    prices = [p for _, p in pts]
    rets = [b / a - 1 for a, b in zip(prices, prices[1:])]
    last = prices[-1]
    return dumps({
        "ok": True, "symbol": sym, "days": len(pts), "from": pts[0][0], "to": pts[-1][0],
        "last": last, "high": max(prices), "low": min(prices), "mean": round(statistics.mean(prices), 4),
        "change_pct": round((last / prices[0] - 1) * 100, 2),
        "daily_volatility_pct": round(statistics.pstdev(rets) * 100, 3),
        "position_in_range_pct": round((last - min(prices)) / ((max(prices) - min(prices)) or 1) * 100),
    })
