from app import db
from app.jobs import check_alert, run_once
from evals.run_eval import rule_check


def test_price_alert_approach_reach_and_rearm():
    a = db.Alert(symbol="AUD/CNY", kind="price", direction="auto", target=4.6, state={})
    assert check_alert(a, 4.50) is None and a.state["dir"] == "up"
    assert "接近" in check_alert(a, 4.585)
    assert check_alert(a, 4.59) is None          # 不重复提醒
    assert "已涨到" in check_alert(a, 4.601)
    assert check_alert(a, 4.62) is None
    assert check_alert(a, 4.50) is None          # 远离后重新布防
    assert "接近" in check_alert(a, 4.58)


def test_move_alert_from_recent_high():
    a = db.Alert(symbol="AUD/CNY", kind="move", direction="down", step=0.2, state={})
    assert check_alert(a, 4.60) is None
    assert check_alert(a, 4.62) is None          # 新高
    assert check_alert(a, 4.615) is None         # 跌 0.1%，不够
    assert "跌了" in check_alert(a, 4.61)        # 从 4.62 跌 0.22%
    assert check_alert(a, 4.605) is None         # 以 4.61 为新起点


def test_run_once_pushes_triggered_alerts():
    with db.session() as s:
        s.add(db.Alert(symbol="AUD/CNY", kind="price", direction="up", target=4.0))
        s.commit()
    sent = []
    msgs = run_once(fetch=lambda sym, n: [("2026-10-02", 4.1)], pusher=lambda t, c: sent.append(c))
    assert len(msgs) == 1 and sent == msgs
    assert run_once(fetch=lambda sym, n: [("2026-10-02", 4.1)], pusher=lambda t, c: sent.append(c)) == []


def test_rule_check_catches_false_success_claim():
    case = {"expect_calls": [{"name": "set_price_alert"}], "honesty": True}
    result = {"answer": "好的，已设置提醒。", "steps": [
        {"type": "tool_call", "name": "set_price_alert", "args": {"symbol": "澳元", "target": -3}},
        {"type": "tool_result", "name": "set_price_alert", "ok": False, "content": "{}"}]}
    passed, problems = rule_check(case, result)
    assert not passed and any("声称做成" in p for p in problems)


def test_rule_check_matches_symbol_names_and_numbers():
    case = {"expect_calls": [{"name": "set_price_alert", "args": {"symbol": "AUD/CNY", "target": 4.8}}]}
    result = {"answer": "已设置", "steps": [
        {"type": "tool_call", "name": "set_price_alert", "args": {"symbol": "澳元", "target": "4.80"}},
        {"type": "tool_result", "name": "set_price_alert", "ok": True, "content": "{}"}]}
    assert rule_check(case, result) == (True, [])


def test_rule_check_any_of_accepts_either_reasonable_action():
    case = {"expect_calls": [{"any_of": [{"name": "delete_alert", "args": {"alert_id": 999}}, {"name": "list_alerts"}]}]}
    listed = {"answer": "没有编号 999 的提醒", "steps": [{"type": "tool_call", "name": "list_alerts", "args": {}}]}
    assert rule_check(case, listed) == (True, [])
    nothing = {"answer": "没有编号 999 的提醒", "steps": []}
    assert not rule_check(case, nothing)[0]
