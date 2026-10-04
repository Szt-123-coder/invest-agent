"""把用户说的名字（澳元、茅台）变成标准代码（AUD/CNY、600519.SS）。"""

from __future__ import annotations

import re

CURRENCIES = {
    "人民币": "CNY", "美元": "USD", "美金": "USD", "澳元": "AUD", "澳币": "AUD", "欧元": "EUR",
    "英镑": "GBP", "日元": "JPY", "港币": "HKD", "港元": "HKD", "加元": "CAD", "新西兰元": "NZD",
    "新加坡元": "SGD", "韩元": "KRW", "瑞士法郎": "CHF",
}
STOCKS = {
    "茅台": "600519.SS", "贵州茅台": "600519.SS", "腾讯": "0700.HK", "阿里巴巴": "BABA", "阿里": "BABA",
    "苹果": "AAPL", "英伟达": "NVDA", "特斯拉": "TSLA", "微软": "MSFT", "必和必拓": "BHP.AX",
}
CODES = set(CURRENCIES.values())


def normalize(text: str) -> str:
    """返回标准代码。货币对一律写成 XXX/YYY；单个外币默认兑人民币；股票保持交易所代码。"""
    t = text.strip()
    if t in STOCKS:
        return STOCKS[t]
    if t in CURRENCIES:
        code = CURRENCIES[t]
        return f"{code}/CNY" if code != "CNY" else "USD/CNY"
    m = re.fullmatch(r"([A-Za-z]{3})\s*[/兑-]?\s*([A-Za-z]{3})", t)
    if m and m.group(1).upper() in CODES and m.group(2).upper() in CODES:
        return f"{m.group(1).upper()}/{m.group(2).upper()}"
    if t.upper() in CODES:
        return f"{t.upper()}/CNY"
    return t.upper()


def find_in_text(text: str) -> list[str]:
    """从一句话里找出提到的标的，按出现顺序去重。"""
    names = sorted({**CURRENCIES, **STOCKS}, key=len, reverse=True)
    hits: list[tuple[int, str]] = []
    taken = [False] * len(text)
    for name in names:
        for m in re.finditer(re.escape(name), text):
            if not any(taken[m.start():m.end()]):
                hits.append((m.start(), normalize(name)))
                for i in range(m.start(), m.end()):
                    taken[i] = True
    for m in re.finditer(r"\b[A-Z]{3}/[A-Z]{3}\b|\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b|\b\d{4,6}\.(?:SS|SZ|HK)\b", text):
        hits.append((m.start(), normalize(m.group())))
    seen, out = set(), []
    for _, sym in sorted(hits):
        if sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def is_pair(symbol: str) -> bool:
    return "/" in symbol


def yahoo_ticker(symbol: str) -> str:
    return symbol.replace("/", "") + "=X" if is_pair(symbol) else symbol
