import pytest
from fastapi.testclient import TestClient

from app import ai_trader, db
from app import portfolio as pf
from app.ai_trader import Decision, Order
from app.main import app

PRICES = {"NVDA": 100.0, "USD/CNY": 7.0, "600519.SS": 1000.0, "AUD/CNY": 4.6, "000300.SS": 4000.0, "^GSPC": 5000.0}


def fetch(symbol, days):
    return [("2026-10-02", PRICES[symbol])]


def test_fees_and_fx():
    assert pf.fee_rate("600519.SS") == pytest.approx(0.001)          # A 股：只有手续费
    assert pf.fee_rate("NVDA") == pytest.approx(0.004)               # 美股：手续费 + 换汇
    assert pf.fee_rate("AUD/CNY") == pytest.approx(0.003)            # 换外汇
    assert pf.currency_of("0700.HK") == "HKD" and pf.currency_of("7203.T") == "JPY"


def test_buy_sell_and_value():
    pf.open_account("user", 10000, fetch)
    t = pf.buy("user", "NVDA", 1000, fetch=fetch)
    assert t.fee_cny == 4.0 and t.quantity == pytest.approx(996 / 700)  # 1 股 = 100 美元 × 7 = 700 元
    with db.session() as s:
        a = pf.get_account(s, "user")
        assert a.cash == 9000
    PRICES["NVDA"] = 110.0
    v = pf.summary("user", fetch)
    assert v["holdings"][0]["value_cny"] == pytest.approx(996 / 700 * 770, abs=0.01)
    assert {b["code"] for b in v["benchmarks"]} == {"000300.SS", "^GSPC"}
    s = pf.sell("user", "NVDA", 0.5, fetch=fetch)
    assert s.amount_cny == pytest.approx(996 / 700 * 770 / 2, abs=0.01)
    PRICES["NVDA"] = 100.0


def test_trade_errors_are_readable():
    with pytest.raises(pf.TradeError, match="还没开户"):
        pf.buy("user", "NVDA", 100, fetch=fetch)
    pf.open_account("user", 1000, fetch)
    with pytest.raises(pf.TradeError, match="现金不够"):
        pf.buy("user", "NVDA", 5000, fetch=fetch)
    with pytest.raises(pf.TradeError, match="没有 600519.SS"):
        pf.sell("user", "600519.SS", fetch=fetch)
    with pytest.raises(pf.TradeError, match="已经开过"):
        pf.open_account("user", 1000, fetch)


def test_ai_orders_are_checked_by_code():
    pf.open_account("ai", 10000, fetch)
    decision = Decision(summary="试试", orders=[
        Order(action="buy", symbol="NVDA", amount_cny=3000, reason="ok"),
        Order(action="buy", symbol="NVDA", amount_cny=3000, reason="超过 40%"),
        Order(action="sell", symbol="茅台", reason="没持有"),
        Order(action="buy", symbol="AUD/CNY", amount_cny=100, reason="第 4 笔"),
    ])
    r = ai_trader.run_once(decide=lambda text: decision, fetch=fetch)
    assert [o["ok"] for o in r["orders"]] == [True, False, False, False]
    assert "40%" in r["orders"][1]["error"] and "没有 600519.SS" in r["orders"][2]["error"]
    assert "最多 3 笔" in r["orders"][3]["error"]
    with db.session() as s:
        assert s.query(db.AiDecision).count() == 1
        assert pf.get_account(s, "user") is None  # AI 的决定不会碰用户账户


def test_ai_does_nothing_without_account():
    assert not ai_trader.run_once(decide=lambda t: Decision(summary="x"), fetch=fetch)["ok"]


def test_chat_tools_via_demo_agent():
    from app.agent import ask

    def calls(r):
        return [(s["name"], s["args"]) for s in r["steps"] if s["type"] == "tool_call"]

    assert calls(ask("开一个 10000 元的模拟账户")) == [("open_paper_account", {"amount_cny": 10000.0})]
    r = ask("用 3000 元买英伟达")
    assert calls(r) == [("paper_buy", {"symbol": "NVDA", "amount_cny": 3000.0})] and r["structured"]["actions"][0]["ok"]
    r = ask("卖掉茅台")
    assert not r["structured"]["actions"][0]["ok"] and "没法卖" in r["answer"]
    assert calls(ask("我的模拟账户怎么样")) == [("paper_account", {})]
    with db.session() as s:
        assert pf.get_account(s, "ai").initial_cash == 10000  # AI 账户跟着用同样金额开


def test_portfolio_api(monkeypatch):
    c = TestClient(app)
    assert c.get("/api/portfolio").json()["user"] is None
    monkeypatch.setenv("ACCESS_PASSWORD", "pw")
    h = {"Authorization": "Bearer pw"}
    assert c.post("/api/portfolio/open", json={"amount_cny": 5000}).status_code == 401
    d = c.post("/api/portfolio/open", json={"amount_cny": 5000}, headers=h).json()
    assert d["user"]["total"] == 5000 and d["ai"]["initial_cash"] == 5000
    d = c.post("/api/portfolio/trade", json={"action": "buy", "symbol": "苹果", "amount_cny": 1000}, headers=h).json()
    assert d["user"]["holdings"][0]["symbol"] == "AAPL"
    bad = c.post("/api/portfolio/trade", json={"action": "buy", "symbol": "苹果", "amount_cny": 99999}, headers=h)
    assert bad.status_code == 400 and "现金不够" in bad.json()["detail"]
    r = c.post("/api/portfolio/ai-run", headers=h).json()
    assert r["ok"] and "演示模式" in r["summary"]
    assert len(c.get("/api/portfolio").json()["decisions"]) == 1
    assert c.get("/portfolio").status_code == 200


def test_ai_candidates_include_benchmarks_watchlist_and_holdings():
    state = {"holdings": [{"symbol": "TSLA"}]}
    assert ai_trader.candidates(state, ["USD/CNY", "NVDA"]) == ["000300.SS", "^GSPC", "USD/CNY", "NVDA", "TSLA"]
    pf.open_account("ai", 10000, fetch)
    t = pf.buy("ai", "^GSPC", 1000, fetch=fetch)  # 指数本身也能买，按美元计价收换汇费
    assert t.fee_cny == 4.0


def test_research_budget_stops_extra_calls():
    from langchain_core.tools import tool

    @tool
    def ping(x: int) -> str:
        """测试用。"""
        return f"pong {x}"

    [t] = ai_trader.with_budget([ping], limit=2)
    assert t.invoke({"x": 1}) == "pong 1" and t.invoke({"x": 2}) == "pong 2"
    assert "已经用完" in t.invoke({"x": 3})


def test_index_tools_fall_back_to_shanghai(monkeypatch):
    import json
    from app.tools import market

    def fake(sym, days):
        return [("2026-09-30", 3900.0)] if sym == "000300.SS" else market._fake_series(sym, days)

    monkeypatch.setattr(market, "fetch_series", fake)
    out = json.loads(market.get_history.invoke({"symbol": "000300.SS", "days": 30}))
    assert out["ok"] and out["symbol"] == "000001.SS" and "上证" in out["note"]
