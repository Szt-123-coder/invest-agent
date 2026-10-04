import json

from sqlalchemy import select

from app import db
from app.agent import ask
from app.symbols import find_in_text, normalize


def tool_names(result):
    return [s["name"] for s in result["steps"] if s["type"] == "tool_call"]


def test_symbols():
    assert normalize("澳元") == "AUD/CNY"
    assert normalize("aud/cny") == "AUD/CNY"
    assert normalize("茅台") == "600519.SS"
    assert find_in_text("英伟达和特斯拉哪个涨得多") == ["NVDA", "TSLA"]


def test_multi_step_question_calls_several_tools_and_saves_steps():
    r = ask("澳元最近怎么样，要不要换？")
    assert tool_names(r) == ["get_quote", "get_history", "find_similar_history", "search_news"]
    assert "AUD/CNY" in r["answer"] and "不构成投资建议" in r["answer"]
    assert r["structured"]["confidence"] in ("高", "中", "低")
    assert {e["source"] for e in r["structured"]["evidence"]} >= {"get_quote", "get_history", "find_similar_history"}
    with db.session() as s:
        run = s.get(db.Run, r["run_id"])
        assert [st.kind for st in run.steps].count("tool_result") == 4
        assert run.answer == r["answer"]


def test_alert_is_really_saved():
    r = ask("澳元跌了就提醒我")
    assert tool_names(r) == ["set_move_alert"]
    with db.session() as s:
        a = s.scalars(select(db.Alert)).one()
        assert (a.symbol, a.kind, a.direction, a.step) == ("AUD/CNY", "move", "down", 0.2)


def test_failed_action_is_reported_not_claimed():
    r = ask("删除提醒 999")
    result = next(s for s in r["steps"] if s["type"] == "tool_result")
    assert json.loads(result["content"])["ok"] is False
    assert "已删除" not in r["answer"] and "没有做成" in r["answer"]


def test_invalid_price_is_rejected():
    r = ask("澳元到 -3 提醒我")
    assert "已设置" not in r["answer"]
    with db.session() as s:
        assert s.scalars(select(db.Alert)).first() is None


def test_short_term_memory_keeps_history():
    ask("美元现在多少", session_id="s1")
    ask("那澳元呢", session_id="s1")
    with db.session() as s:
        rows = s.scalars(select(db.ChatMessage).where(db.ChatMessage.session_id == "s1")).all()
    assert [m.role for m in rows] == ["user", "assistant", "user", "assistant"]


def test_history_only_from_same_model():
    """演示模式的假数据回答不能带进真模型的上下文，否则真模型会照抄。"""
    from app.agent import _history
    ask("美元现在多少", session_id="s2")
    assert len(_history("s2", "demo-rules")) == 2
    assert _history("s2", "deepseek-chat") == []
