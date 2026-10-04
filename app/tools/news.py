"""新闻工具：用 Tavily 搜索最近的新闻；演示模式返回固定的示例新闻。"""

from __future__ import annotations

import httpx
from langchain_core.tools import tool

from ..config import get_settings
from ..db import dumps

TAVILY = "https://api.tavily.com/search"

DEMO_NEWS = [
    {"title": "澳洲联储维持利率不变，暗示年内或再加息", "source": "示例新闻",
     "snippet": "澳洲联储本次会议维持现金利率不变，声明称通胀仍高于目标区间。"},
    {"title": "中国 9 月出口增速放缓", "source": "示例新闻",
     "snippet": "海关数据显示 9 月出口同比增速较上月回落，市场关注人民币汇率走向。"},
    {"title": "铁矿石价格本周下跌", "source": "示例新闻",
     "snippet": "铁矿石期货本周累计下跌，澳洲资源股承压。"},
]


@tool
def search_news(query: str, max_results: int = 3) -> str:
    """搜索和 query 相关的最近财经新闻，返回标题、来源和摘要。query 用中文或英文关键词，例如「澳元 汇率」。"""
    s = get_settings()
    n = max(1, min(int(max_results), 5))
    if s.demo_mode or not s.tavily_api_key:
        return dumps({"ok": True, "demo": True, "query": query, "results": DEMO_NEWS[:n]})
    try:
        r = httpx.post(TAVILY, json={"api_key": s.tavily_api_key, "query": query, "topic": "news",
                                      "max_results": n, "days": 7}, timeout=20)
        r.raise_for_status()
    except Exception as e:
        return dumps({"ok": False, "error": f"新闻搜索失败：{e}"})
    results = [{"title": x.get("title", ""), "source": x.get("url", ""), "snippet": (x.get("content") or "")[:300]}
               for x in r.json().get("results", [])]
    return dumps({"ok": True, "query": query, "results": results})
