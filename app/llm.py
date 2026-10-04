"""模型入口。代码其他地方只调用 get_model()，换模型只改环境变量。

没有密钥时用 DemoModel：一个按关键词决定调哪个工具的「假模型」。
它让演示页面不花钱也能跑起来，也让测试不依赖网络。它不是真 AI，回答是拼出来的。
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .config import get_settings
from .symbols import find_in_text


def get_model(model: str | None = None) -> BaseChatModel:
    s = get_settings()
    if s.demo_mode:
        return DemoModel()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model or s.llm_model, api_key=s.llm_api_key, base_url=s.llm_base_url,
                      temperature=0, timeout=60)


def model_name(model: BaseChatModel) -> str:
    return getattr(model, "model_name", None) or getattr(model, "model", None) or model._llm_type


class DemoModel(BaseChatModel):
    """关键词规则版的假模型，只用于演示和测试。"""

    @property
    def _llm_type(self) -> str:
        return "demo-rules"

    def bind_tools(self, tools: Any, **kwargs: Any) -> DemoModel:
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        msg = self._decide(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _decide(self, messages: list[BaseMessage]) -> AIMessage:
        # 最后一条是工具结果：根据结果写最终回答
        tail = []
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                tail.append(m)
            else:
                break
        if tail:
            return AIMessage(content=self._answer(list(reversed(tail))))
        question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        calls = self._plan(str(question))
        if not calls:
            return AIMessage(content="（演示模式）我可以帮你查汇率和股价、看走势、读新闻、设提醒。试试问「澳元最近怎么样，要不要换」。")
        return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"}
                                                  for i, (n, a) in enumerate(calls)])

    @staticmethod
    def _plan(q: str) -> list[tuple[str, dict]]:
        syms = find_in_text(q) or (["AUD/CNY"] if re.search(r"汇率|换汇", q) else [])
        num = re.search(r"-?\d+(?:\.\d+)?", re.sub(r"\d+(?:\.\d+)?\s*%", "", q))
        pct = re.search(r"(\d+(?:\.\d+)?)\s*%", q)
        down = re.search(r"跌|降|低", q)
        up = re.search(r"涨|升|高", q)
        direction = "down" if down and not up else "up" if up and not down else "both"
        if re.search(r"删除提醒|取消提醒", q) and num:
            return [("delete_alert", {"alert_id": int(float(num.group()))})]
        if re.search(r"哪些提醒|列表|我的提醒", q):
            return [("list_alerts", {})]
        if q.startswith("记住"):
            return [("remember_preference", {"key": "note", "value": q.removeprefix("记住").strip("，,：: ")})]
        if re.search(r"提醒|告诉我|通知我", q) and syms:
            if num:
                d = {"down": "down", "up": "up"}.get(direction, "auto")
                return [("set_price_alert", {"symbol": syms[0], "target": float(num.group()), "direction": d})]
            return [("set_move_alert", {"symbol": syms[0], "direction": direction,
                                         "step_pct": float(pct.group(1)) if pct else 0.2})]
        if re.search(r"取消关注", q) and syms:
            return [("watch", {"symbol": syms[0], "remove": True})]
        if re.search(r"关注", q) and syms:
            return [("watch", {"symbol": syms[0]})]
        if not syms:
            return [("search_news", {"query": q})] if re.search(r"新闻|消息", q) else []
        calls: list[tuple[str, dict]] = []
        for sym in syms[:2]:
            calls.append(("get_quote", {"symbol": sym}))
            if re.search(r"最近|走势|怎么样|要不要|该不该|趋势|换", q):
                calls.append(("get_history", {"symbol": sym, "days": 30}))
        if re.search(r"新闻|为什么|要不要|该不该|消息|原因", q):
            calls.append(("search_news", {"query": f"{syms[0]} 汇率 新闻" if "/" in syms[0] else f"{syms[0]} 新闻"}))
        return calls

    @staticmethod
    def _answer(results: list[ToolMessage]) -> str:
        lines, failed = [], []
        for m in results:
            try:
                data = json.loads(m.content)
            except (TypeError, ValueError):
                data = {"ok": False, "error": str(m.content)}
            if not data.get("ok"):
                failed.append(data.get("error", "未知错误"))
                continue
            if "price" in data:
                lines.append(f"{data['symbol']} 最新 {data['price']}（{data['date']}，较前一日 {data['change_pct']:+}%）。")
            elif "high" in data:
                lines.append(f"近 {data['days']} 天在 {data['low']} 到 {data['high']} 之间，区间涨跌 {data['change_pct']:+}%，"
                             f"当前处在区间的 {data['position_in_range_pct']}% 位置。")
            elif "results" in data:
                titles = "；".join(r["title"] for r in data["results"])
                lines.append(f"相关新闻：{titles}。")
            elif "alert_id" in data:
                what = f"到 {data['target']}" if "target" in data else f"每变动 {data['step_pct']}%"
                lines.append(f"已设置提醒 #{data['alert_id']}：{data['symbol']} {what}时通知你。")
            elif "alerts" in data:
                lines.append(f"共有 {len(data['alerts'])} 个提醒，关注 {', '.join(data['watchlist']) or '无'}。")
            elif "deleted" in data:
                lines.append(f"已删除提醒 #{data['deleted']}。")
            elif "watching" in data:
                lines.append(f"已{'加入' if data['watching'] else '移出'}关注：{data['symbol']}。")
            else:
                lines.append("已记住。")
        if failed:
            lines.append("没有做成：" + "；".join(failed))
        return "（演示模式，数据为示例）" + "".join(lines) + "\n\n仅供学习参考，不构成投资建议。"
