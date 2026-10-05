"""行情工具：实时价和历史走势。数据来自 Yahoo Finance；演示模式用固定的假数据。"""

from __future__ import annotations

import hashlib
import math
import statistics
from datetime import date, datetime, timedelta, timezone

import httpx
from langchain_core.tools import tool

from ..config import get_settings
from ..db import dumps
from ..resolve import to_code
from ..symbols import INDEXES, UnknownSymbol, benchmark, check_known, yahoo_ticker

YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _fake_series(symbol: str, days: int) -> list[tuple[str, float]]:
    """按代码生成一条稳定的假走势，同一个代码每次都一样，方便演示和测试。

    第 k 天前的价格只和 k 有关，所以查 30 天和查 3 年看到的是同一条走势的不同长度。
    """
    seed = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
    base = {"AUD/CNY": 4.62, "USD/CNY": 7.12, "EUR/CNY": 7.75, "^GSPC": 5800, "000300.SS": 3900, "000001.SS": 3300, "^HSI": 21000,
            "^AXJO": 8200, "^KS11": 2600, "^N225": 38000}.get(symbol, 50 + seed % 300)
    today = date(2026, 10, 2)
    out = []
    for k in range(days - 1, -1, -1):
        i = 364 - k
        wave = math.sin((i + seed % 7) / 4) * 0.012 + math.sin(i / 37) * 0.01 + (i - 365) * 0.00004
        if "/" not in symbol:  # 股票和指数波动比汇率大
            wave = wave * 4 + math.sin((i + seed % 11) / 9) * 0.03
        out.append(((today - timedelta(days=k)).isoformat(), round(base * (1 + wave), 4)))
    return out


def fetch_series(symbol: str, days: int = 30) -> list[tuple[str, float]]:
    check_known(symbol)
    if get_settings().demo_mode:
        return _fake_series(symbol, days)
    rng = next(r for n, r in ((31, "1mo"), (92, "3mo"), (366, "1y"), (731, "2y"), (10**9, "5y")) if days <= n)
    r = httpx.get(YAHOO.format(ticker=yahoo_ticker(symbol)), params={"range": rng, "interval": "1d"},
                  headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
    if r.status_code == 404:
        raise UnknownSymbol(f"行情接口里没有 {symbol}：可能没有上市、已经退市，或者代码写错了")
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    closes = res["indicators"]["quote"][0]["close"]
    # 按交易所当地时间算日期，不按运行这台电脑的时区，否则不同市场对不上日子
    offset = res.get("meta", {}).get("gmtoffset", 0)
    pts = [(datetime.fromtimestamp(t + offset, timezone.utc).date().isoformat(), round(c, 4))
           for t, c in zip(res["timestamp"], closes) if c is not None]
    return pts[-days:]


INDEX_FALLBACK = {"000300.SS": ["000001.SS"]}  # Yahoo 的沪深300 数据有时缺很多天，缺了就换上证指数


def _series_or_fallback(sym: str, days: int, need: int) -> tuple[str, list[tuple[str, float]], str | None]:
    """查走势；指数数据不到 need 天时换备用指数（沪深300 → 上证指数）。返回（实际用的代码, 数据, 给模型的说明）。"""
    try:
        pts = fetch_series(sym, days)
    except Exception:
        if not INDEX_FALLBACK.get(sym):
            raise
        pts = []
    if len(pts) >= need:
        return sym, pts, None
    for alt in INDEX_FALLBACK.get(sym, []):
        try:
            alt_pts = fetch_series(alt, days)
        except Exception:
            continue
        if len(alt_pts) > len(pts):
            return alt, alt_pts, (f"{sym} 只查到 {len(pts)} 个交易日，改用走势相近的 {alt}（{INDEXES.get(alt, alt)}）的数据；"
                                  f"买卖时仍可用 {sym}")
    return sym, pts, None


@tool
def get_quote(symbol: str) -> str:
    """查一个标的的最新价格。symbol 可以是货币对（AUD/CNY）、Yahoo 股票代码（AAPL、600519.SS、005930.KS）或常见中文名（澳元、茅台）。
    名字查不到时，如果你知道它的 Yahoo 代码，就用代码再查一次。"""
    sym = to_code(symbol)
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
    sym = to_code(symbol)
    days = max(5, min(int(days), 365))
    try:
        sym, pts, note = _series_or_fallback(sym, days, need=days // 2)
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
    } | ({"note": note} if note else {}))


@tool
def find_similar_history(symbol: str, lookback_days: int = 30, horizon_days: int = 30, years: int = 3) -> str:
    """在过去几年的真实走势里，找和现在相似的时候，统计之后 horizon_days 天怎么走。

    「相似」指：最近 lookback_days 个交易日的涨跌幅接近（相差不超过历史上这个涨跌幅波动的 1/4），
    而且当前价在这段区间里的位置接近（相差不超过 20 个百分点）。
    适合回答「要不要换」「会不会继续跌」这类问题时，用历史数据当依据。样本少时结论不可靠。
    """
    sym = to_code(symbol)
    w = max(5, min(int(lookback_days), 120))
    h = max(5, min(int(horizon_days), 120))
    years = max(1, min(int(years), 5))
    try:
        sym, pts, swap = _series_or_fallback(sym, years * 365, need=2 * w + h + 10)
    except Exception as e:
        return dumps({"ok": False, "error": f"查不到 {sym} 的历史数据：{e}"})
    p = [x for _, x in pts]
    if len(p) < 2 * w + h + 10:
        return dumps({"ok": False, "error": f"{sym} 历史数据不够长，只有 {len(p)} 个交易日"})

    def features(t: int) -> tuple[float, float]:
        win = p[t - w:t + 1]
        lo, hi = min(win), max(win)
        return p[t] / p[t - w] - 1, (p[t] - lo) / ((hi - lo) or 1)

    now = len(p) - 1
    chg_now, pos_now = features(now)
    candidates = range(w, now - h + 1)
    tol = statistics.pstdev([features(t)[0] for t in candidates]) / 4
    matches, t = [], w
    while t <= now - h:
        chg, pos = features(t)
        if abs(chg - chg_now) <= tol and abs(pos - pos_now) <= 0.2:
            matches.append((pts[t][0], round((p[t + h] / p[t] - 1) * 100, 2)))
            t += h  # 跳过这一段，避免把同一次行情重复算好几次
        else:
            t += 1
    fwd = [f for _, f in matches]
    out = {"ok": True, "symbol": sym, "from": pts[0][0], "to": pts[-1][0], "lookback_days": w, "horizon_days": h,
           "now_change_pct": round(chg_now * 100, 2), "now_position_in_range_pct": round(pos_now * 100),
           "matches": len(matches)}
    if fwd:
        up = sum(f > 0 for f in fwd)
        out |= {"up_count": up, "up_ratio_pct": round(up / len(fwd) * 100), "avg_forward_pct": round(statistics.mean(fwd), 2),
                "median_forward_pct": round(statistics.median(fwd), 2), "worst_forward_pct": min(fwd),
                "best_forward_pct": max(fwd), "recent_examples": [{"date": d, "forward_pct": f} for d, f in matches[-5:]]}
    out["note"] = ((swap + "。") if swap else "") + ("样本太少，结论不可靠。" if len(fwd) < 8 else "") + "过去的走势不代表未来。"
    return dumps(out)


@tool
def compare_with_index(symbol: str, days: int = 30) -> str:
    """把一只股票最近 days 天（默认 30）的涨跌和它所在市场的大盘指数比较。

    A 股比沪深300，港股比恒生指数，澳股比 ASX 200，韩股比 KOSPI，日股比日经225，其他比标普500。
    返回两者的涨跌幅、超额收益（股票减大盘），以及每日涨跌的相关系数（越接近 1 越是跟着大盘走）。
    """
    sym = to_code(symbol)
    idx = benchmark(sym)
    if not idx:
        return dumps({"ok": False, "error": f"{sym} 不是股票，没有对应的大盘指数"})
    days = max(5, min(int(days), 365))
    try:
        a = dict(fetch_series(sym, days))
        idx, b = _index_series(idx, days, set(a))
    except Exception as e:
        return dumps({"ok": False, "error": f"查不到 {sym} 或大盘的数据：{e}"})
    dates = sorted(set(a) & set(b))  # 两个市场休市日不同，只比较都开盘的日子
    if len(dates) < 5:
        return dumps({"ok": False, "error": f"{sym} 和大盘共同的交易日太少：股票 {len(a)} 天，{INDEXES[idx]} {len(b)} 天，"
                                            f"重合 {len(dates)} 天"})
    pa, pb = [a[d] for d in dates], [b[d] for d in dates]
    ra = [y / x - 1 for x, y in zip(pa, pa[1:])]
    rb = [y / x - 1 for x, y in zip(pb, pb[1:])]
    sa, sb = (pa[-1] / pa[0] - 1) * 100, (pb[-1] / pb[0] - 1) * 100
    try:
        corr = round(statistics.correlation(ra, rb), 2)
    except statistics.StatisticsError:
        corr = None
    return dumps({"ok": True, "symbol": sym, "index": idx, "index_name": INDEXES[idx], "from": dates[0], "to": dates[-1],
                  "days": len(dates), "stock_change_pct": round(sa, 2), "index_change_pct": round(sb, 2),
                  "excess_pct": round(sa - sb, 2), "daily_correlation": corr})


def _index_series(idx: str, days: int, stock_dates: set[str]) -> tuple[str, dict[str, float]]:
    """查大盘；和股票重合的交易日不到一半时，换备用指数。返回（实际用的指数, 日期→点位）。"""
    best = (idx, {})
    for code in [idx, *INDEX_FALLBACK.get(idx, [])]:
        try:
            b = dict(fetch_series(code, days))
        except Exception:
            continue
        if len(stock_dates & set(b)) * 2 >= len(stock_dates):
            return code, b
        if len(b) > len(best[1]):
            best = (code, b)
    return best


def series_overview(symbol: str, days: int = 30) -> dict:
    """股票页面用：一段走势的价格点和关键数字，另附大盘走势（用来画对比线）。不经过模型。"""
    sym = to_code(symbol)
    days = max(5, min(int(days), 365))
    year = fetch_series(sym, 365)
    if len(year) < 2:
        raise ValueError(f"{sym} 没有数据")
    pts = year[-days:]
    prices = [p for _, p in pts]
    rets = [y / x - 1 for x, y in zip(prices, prices[1:])]
    out = {"symbol": sym, "days": len(pts), "points": pts,
           "stats": {"last": prices[-1], "date": pts[-1][0], "change_pct": round((prices[-1] / prices[0] - 1) * 100, 2),
                     "high": max(prices), "low": min(prices),
                     "high_52w": max(p for _, p in year), "low_52w": min(p for _, p in year),
                     "daily_volatility_pct": round(statistics.pstdev(rets) * 100, 3)}}
    idx = benchmark(sym)
    if idx:
        code, b = _index_series(idx, days, {d for d, _ in pts})
        if b:  # 大盘查不到不影响股票本身的展示
            out |= {"index": code, "index_name": INDEXES[code], "index_points": sorted(b.items())}
    return out
