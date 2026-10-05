"""Agent 本体：用 LangChain 的 create_agent（底层是 LangGraph 的 ReAct 循环）。

run_stream() 每发生一步就产出一个事件：模型决定调工具、工具返回结果、最终回答。
网页、评测脚本、定时任务都通过它运行 agent，所以三处看到的行为完全一样。
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from sqlalchemy import select

from . import db
from .answer import ANSWER_TOOL, Answer, render
from .config import get_settings
from .llm import get_model, model_name
from .tools import ALL_TOOLS
from .tools.alerts import preferences_text

HISTORY_TURNS = 6  # 短期记忆：带上最近几轮对话

SYSTEM_PROMPT = """你是一个投资学习助手，帮用户看汇率和股票、读新闻、设提醒，并解释背后的原因。今天是 {today}。

做事规则：
1. 需要数据就调用工具，不要凭记忆编价格、日期或新闻。可以一次调用多个工具。之前对话里出现过的价格和新闻可能已经过时，用户再问时要重新调用工具查。
2. 用户问「要不要换」「怎么样」这类问题时，先查实时价，再查近 30 天走势；问「要不要换」「会不会继续跌」时，再用 find_similar_history 看历史上相似的时候之后怎么走；必要时再搜新闻。分析股票时，用 compare_with_index 和大盘比较。
3. 设提醒、改关注、记偏好：只有工具返回 "ok": true 时，才能说「已设置」。返回 "ok": false 时，必须如实告诉用户没做成和原因。
4. 用户说「跌了就告诉我」这类没有具体价位的话，用 set_move_alert；有具体价位用 set_price_alert。用户给的数字原样传给工具，不要自己改（比如把 -3 改成 3）；数字不合理时，工具会返回原因，你如实转告并请用户确认。
5. 最后调用 Answer 交出回答，用中文，简洁。evidence 里的每个数字都必须原样抄自工具结果，并在 source 写上工具名；工具没给的数字不要写。历史相似情形样本少时，要写进 risks。actions 只填修改类操作（设提醒、删提醒、改关注、记偏好），查询不算；ok 必须和工具返回的 ok 一致。
6. 你的建议只供学习参考，页面会自动加上「不构成投资建议」。
{prefs}"""


def system_prompt() -> str:
    today = datetime.now(ZoneInfo(get_settings().timezone)).strftime("%Y-%m-%d")
    prefs = preferences_text()
    return SYSTEM_PROMPT.format(today=today, prefs=f"\n用户的长期偏好：\n{prefs}" if prefs else "")


def build_agent(model: BaseChatModel | None = None):
    # ToolStrategy：把 Answer 当成一个「交卷工具」，模型调用它就是交答案，格式不对会自动重填。
    # 不用模型厂商自带的 JSON 模式，是因为 DeepSeek 等很多兼容接口不支持，工具调用它们都支持。
    return create_agent(model or get_model(), tools=ALL_TOOLS, system_prompt=system_prompt(),
                        response_format=ToolStrategy(Answer))


def _history(session_id: str, model: str) -> list:
    """只带同一个模型的历史：否则演示模式的假数据回答会被真模型当成事实照抄。"""
    with db.session() as s:
        runs = s.scalars(select(db.Run).where(db.Run.session_id == session_id, db.Run.model == model,
                                              db.Run.answer != "")
                         .order_by(db.Run.id.desc()).limit(HISTORY_TURNS)).all()
    return [m for r in reversed(runs) for m in (HumanMessage(r.question), AIMessage(r.answer))]


def _ok(content: str) -> bool:
    try:
        return bool(json.loads(content).get("ok", True))
    except (TypeError, ValueError, AttributeError):
        return True


def run_stream(question: str, session_id: str = "default", model: BaseChatModel | None = None) -> Iterator[dict]:
    """运行一次 agent，按发生顺序产出事件，同时把每一步存进数据库。"""
    model = model or get_model()
    agent = build_agent(model)
    started = time.monotonic()
    names: dict[str, str] = {}
    answer = ""
    with db.session() as s:
        messages = _history(session_id, model_name(model)) + [HumanMessage(question)]
        run = db.Run(session_id=session_id, question=question, model=model_name(model))
        s.add(run)
        s.commit()
        yield {"type": "start", "run_id": run.id, "model": run.model}
        for update in agent.stream({"messages": messages}, stream_mode="updates"):
            for node in update.values():
                node = node or {}
                events = []
                for m in node.get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for c in m.tool_calls:
                            if c["name"] == ANSWER_TOOL:  # 交卷，不算一个步骤
                                continue
                            names[c["id"]] = c["name"]
                            events.append({"type": "tool_call", "id": c["id"], "name": c["name"], "args": c["args"]})
                    elif isinstance(m, AIMessage) and m.content:  # 模型没按格式交卷时的兜底
                        answer = m.content if isinstance(m.content, str) else str(m.content)
                        events.append({"type": "answer", "content": answer, "structured": None})
                    elif isinstance(m, ToolMessage) and m.name != ANSWER_TOOL:
                        name = m.name or names.get(m.tool_call_id, "")
                        events.append({"type": "tool_result", "id": m.tool_call_id, "name": name,
                                       "content": m.content, "ok": _ok(m.content)})
                if isinstance(node.get("structured_response"), Answer):
                    structured = node["structured_response"]
                    answer = render(structured)
                    events.append({"type": "answer", "content": answer, "structured": structured.model_dump()})
                for e in events:
                    s.add(db.Step(run_id=run.id, kind=e["type"], name=e.get("name", ""), payload=e))
                    yield e
        run.answer = answer
        run.duration_ms = int((time.monotonic() - started) * 1000)
        s.add_all([db.ChatMessage(session_id=session_id, role="user", content=question),
                   db.ChatMessage(session_id=session_id, role="assistant", content=answer)])
        s.commit()
        yield {"type": "done", "run_id": run.id, "duration_ms": run.duration_ms}


def ask(question: str, session_id: str = "default", model: BaseChatModel | None = None) -> dict:
    """不需要流式时用：返回最终回答和所有步骤。"""
    events = list(run_stream(question, session_id, model))
    final = next((e for e in reversed(events) if e["type"] == "answer"), {})
    return {"answer": final.get("content", ""), "structured": final.get("structured"),
            "steps": [e for e in events if e["type"] in ("tool_call", "tool_result")],
            "run_id": events[0]["run_id"]}
