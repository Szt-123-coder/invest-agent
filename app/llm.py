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

from .answer import ANSWER_TOOL
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
            return self._submit(self._answer(list(reversed(tail))))
        question = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
        calls = self._plan(str(question))
        if not calls:
            return self._submit({"conclusion": "（演示模式）我可以帮你查汇率和股价、看走势、读新闻、设提醒。试试问「澳元最近怎么样，要不要换」。",
                                 "confidence": "高", "confidence_reason": "这是功能介绍，不涉及数据"})
        return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"call_{i}", "type": "tool_call"}
                                                  for i, (n, a) in enumerate(calls)])

    @staticmethod
    def _submit(answer: dict) -> AIMessage:
        """交卷：和真模型一样，通过调用 Answer 这个「工具」交出结构化回答。"""
        return AIMessage(content="", tool_calls=[{"name": ANSWER_TOOL, "args": answer, "id": "call_answer",
                                                  "type": "tool_call"}])

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
        topic = re.search(r"(?:关注|留意)(?:一下)?(.+?)(?:的)?(?:新闻|话题|消息)", q)
        if topic and not syms:
            return [("watch_topic", {"topic": topic.group(1).strip("「」 "), "remove": "取消" in q})]
        if re.search(r"取消关注", q) and syms:
            return [("watch", {"symbol": syms[0], "remove": True})]
        if re.search(r"关注", q) and syms:
            return [("watch", {"symbol": syms[0]})]
        if not syms:
            return [("search_news", {"query": q})] if re.search(r"新闻|消息", q) else []
        calls: list[tuple[str, dict]] = []
        for sym in syms[:2]:
            calls.append(("get_quote", {"symbol": sym}))
            if re.search(r"最近|走势|怎么样|要不要|该不该|趋势|换|分析", q):
                calls.append(("get_history", {"symbol": sym, "days": 30}))
            if re.search(r"要不要|该不该|会不会|历史上", q):
                calls.append(("find_similar_history", {"symbol": sym}))
            if "/" not in sym and re.search(r"大盘|指数|跑赢|跑输|分析", q):
                calls.append(("compare_with_index", {"symbol": sym, "days": 30}))
        if re.search(r"新闻|为什么|要不要|该不该|消息|原因", q):
            calls.append(("search_news", {"query": f"{syms[0]} 汇率 新闻" if "/" in syms[0] else f"{syms[0]} 新闻"}))
        return calls

    @staticmethod
    def _answer(results: list[ToolMessage]) -> dict:
        """把工具结果拼成 Answer 的各个字段。数字都直接取自工具结果，不做任何编造。"""
        said, evidence, risks, actions = [], [], [], []
        for m in results:
            try:
                data = json.loads(m.content)
            except (TypeError, ValueError):
                data = {"ok": False, "error": str(m.content)}
            name = m.name or ""
            ok = bool(data.get("ok"))
            if name in WRITE_TOOLS:
                actions.append({"action": WRITE_TOOLS[name], "ok": ok, "detail": _write_detail(data)})
                said.append(f"{WRITE_TOOLS[name]}{'成功' if ok else '没有做成'}。")
                continue
            if not ok:
                risks.append(f"有一步没查到数据：{data.get('error', '未知错误')}")
                continue
            if "price" in data:
                said.append(f"{data['symbol']} 最新 {data['price']}。")
                evidence += [{"label": "最新价", "value": str(data["price"]), "source": name},
                             {"label": "较前一日", "value": f"{data['change_pct']:+}%", "source": name}]
            elif "high" in data:
                said.append(f"近 {data['days']} 个交易日处在区间的 {data['position_in_range_pct']}% 位置。")
                evidence += [{"label": f"近 {data['days']} 天区间", "value": f"{data['low']} 到 {data['high']}", "source": name},
                             {"label": f"近 {data['days']} 天涨跌", "value": f"{data['change_pct']:+}%", "source": name}]
            elif "excess_pct" in data:
                said.append(f"同期{data['index_name']} {data['index_change_pct']:+}%，"
                            f"{'跑赢' if data['excess_pct'] >= 0 else '跑输'}大盘 {abs(data['excess_pct'])} 个百分点。")
                evidence += [{"label": f"同期{data['index_name']}涨跌", "value": f"{data['index_change_pct']:+}%", "source": name},
                             {"label": "超额收益", "value": f"{data['excess_pct']:+}%", "source": name}]
                if data["daily_correlation"] is not None and data["daily_correlation"] > 0.7:
                    risks.append(f"和大盘的每日相关系数是 {data['daily_correlation']}，大盘下跌时它很可能跟着跌")
            elif "matches" in data:
                if data["matches"]:
                    evidence.append({"label": f"历史相似情形之后 {data['horizon_days']} 天上涨的比例",
                                     "value": f"{data['up_count']}/{data['matches']}", "source": name})
                    evidence.append({"label": "相似情形之后的平均涨跌", "value": f"{data['avg_forward_pct']:+}%", "source": name})
                risks.append(f"历史相似情形只有 {data['matches']} 次，{data['note']}")
            elif "results" in data:
                said.append("相关新闻：" + "；".join(r["title"] for r in data["results"]) + "。")
                risks.append("新闻只看了标题，可能遗漏重要背景")
            elif "alerts" in data:
                said.append(f"共有 {len(data['alerts'])} 个提醒，关注 {', '.join(data['watchlist']) or '无'}。")
        failed = any(not a["ok"] for a in actions) or any(r.startswith("有一步没查到") for r in risks)
        return {"conclusion": "（演示模式，数据为示例）" + "".join(said), "evidence": evidence, "risks": risks,
                "confidence": "低" if failed or not evidence else "中",
                "confidence_reason": "演示模式按关键词规则拼出回答，不是真的分析", "actions": actions}


WRITE_TOOLS = {"set_price_alert": "设置价位提醒", "set_move_alert": "设置波动提醒", "delete_alert": "删除提醒",
               "watch": "修改关注", "watch_topic": "修改关注话题", "remember_preference": "记住偏好"}


def _write_detail(data: dict) -> str:
    if not data.get("ok"):
        return data.get("error", "未知错误")
    if "alert_id" in data:
        what = f"到 {data['target']}" if "target" in data else f"每变动 {data['step_pct']}%"
        return f"提醒 #{data['alert_id']}：{data['symbol']} {what}时通知你"
    if "deleted" in data:
        return f"提醒 #{data['deleted']} 已删除"
    if "topic" in data:
        return f"话题「{data['topic']}」已{'加入' if data['watching'] else '移出'}关注"
    if "watching" in data:
        return f"{data['symbol']} 已{'加入' if data['watching'] else '移出'}关注"
    return "已记住"
