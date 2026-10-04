import json

from fastapi.testclient import TestClient

from app import db
from app.main import app


def events(resp):
    return [json.loads(line[6:]) for line in resp.text.split("\n\n") if line.startswith("data: ")]


def test_ask_streams_steps_in_order():
    with TestClient(app) as c:
        r = c.post("/api/ask", json={"question": "苹果股价最近走势如何？"})
    assert r.headers["content-type"].startswith("text/event-stream")
    kinds = [e["type"] for e in events(r)]
    assert kinds[0] == "start" and kinds[-1] == "done"
    assert kinds.index("tool_call") < kinds.index("tool_result") < kinds.index("answer")


def test_visitors_without_password_get_demo_model(monkeypatch):
    monkeypatch.setenv("ACCESS_PASSWORD", "secret")
    monkeypatch.setenv("DEMO_MODE", "")
    monkeypatch.setenv("LLM_API_KEY", "would-cost-money")
    with TestClient(app) as c:
        r = c.post("/api/ask", json={"question": "美元多少"})
    assert events(r)[0]["model"] == "demo-rules"


def test_eval_api_summarises_latest_batch():
    with db.session() as s:
        s.add_all([db.EvalResult(batch="b1", model="m", case_id="a", rule_pass=True, judge_score=4),
                   db.EvalResult(batch="b1", model="m", case_id="b", rule_pass=False, judge_score=2)])
        s.commit()
    with TestClient(app) as c:
        d = c.get("/api/eval").json()
    assert d["summary"][0] | {"at": None} == {"batch": "b1", "model": "m", "cases": 2, "rule_pass": 1,
                                             "judge_avg": 3.0, "at": None}
    assert [r["case_id"] for r in d["latest"]] == ["a", "b"]


def test_pages_load():
    with TestClient(app) as c:
        assert "投资学习 Agent" in c.get("/").text
        assert "评测面板" in c.get("/eval").text
