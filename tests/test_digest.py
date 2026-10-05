from datetime import datetime, timedelta

from fastapi.testclient import TestClient

from app import digest
from app.digest import Extraction, NewsImpact
from app.main import app

NEWS = [{"title": "澳洲联储加息", "source": "https://a.example/1", "snippet": "现金利率上调"},
        {"title": "美联储按兵不动", "source": "https://a.example/2", "snippet": "利率不变"}]


def fake_news(query, n, days=7):
    return NEWS  # 两个标的搜到同一批新闻：要去重


def test_query_for():
    assert digest.query_for("AUD/CNY") == "澳元 人民币 汇率"
    assert digest.query_for("NVDA") == "英伟达 股价"


def test_check_drops_made_up_news_wrong_symbols_and_duplicates():
    def extract(prompt):
        assert "1：澳洲联储加息" in prompt and "2：美联储按兵不动" in prompt
        return Extraction(overview="今天澳元偏强", items=[
            NewsImpact(news_id=1, symbol="AUD/CNY", what="澳洲联储加息", direction="偏涨", reason="利差扩大"),
            NewsImpact(news_id=1, symbol="AUD/CNY", what="重复", direction="偏涨", reason="重复"),
            NewsImpact(news_id=9, symbol="AUD/CNY", what="编的新闻", direction="偏跌", reason="编的"),
            NewsImpact(news_id=2, symbol="EUR/CNY", what="不在关注列表", direction="看不出", reason="-"),
        ])

    d = digest.build(["USD/CNY", "AUD/CNY"], extract=extract, news_fn=fake_news)
    assert d["news_count"] == 2
    assert [(i["news_id"], i["symbol"]) for i in d["items"]] == [(1, "AUD/CNY")]
    assert d["items"][0]["url"] == "https://a.example/1"  # 链接由代码按编号填，不是模型写的
    assert len(d["dropped"]) == 3
    text = digest.to_markdown(d)
    assert "偏涨｜澳洲联储加息" in text and "没有相关新闻" in text and "不构成投资建议" in text


def test_news_failure_is_reported_not_hidden():
    def broken(query, n, days=7):
        raise RuntimeError("网络断了")

    d = digest.build(["AUD/CNY"], extract=lambda p: Extraction(overview="x"), news_fn=broken)
    assert d["errors"] and "网络断了" in d["errors"][0]
    assert d["items"] == [] and d["overview"] == "今天没有搜到相关新闻。"


def test_run_once_saves_and_pushes_markdown():
    sent = []
    d = digest.run_once(pusher=lambda title, content, template: sent.append(template) or True)
    assert d["pushed"] and sent == ["markdown"]
    assert d["symbols"] == digest.DEFAULT_SYMBOLS and d["demo"]
    assert digest.last_sent() is not None


def test_due_only_once_per_slot_and_not_stale():
    times = ["08:00", "23:00"]
    morning = datetime(2026, 10, 5, 8, 1)
    assert digest.due(morning, times, None)
    assert not digest.due(morning, times, morning - timedelta(seconds=30))  # 这一档已经发过
    assert not digest.due(datetime(2026, 10, 5, 7, 59), times, None)  # 还没到
    assert not digest.due(datetime(2026, 10, 5, 12, 0), times, None)  # 过了太久，不补发
    assert digest.due(datetime(2026, 10, 5, 23, 5), times, morning)


def test_digest_api(monkeypatch):
    c = TestClient(app)
    assert c.get("/api/digest").json() == []
    monkeypatch.setenv("ACCESS_PASSWORD", "pw")
    assert c.post("/api/digest/run").status_code == 401  # 访客不能触发（会花钱）
    r = c.post("/api/digest/run", headers={"Authorization": "Bearer pw"})
    assert r.status_code == 200 and r.json()["pushed"] is False
    assert len(c.get("/api/digest").json()) == 1
    assert c.get("/digest").status_code == 200
