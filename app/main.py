"""FastAPI 服务：网页、评测脚本、定时任务都通过这里的接口使用 agent。

启动：uvicorn app.main:app --reload
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from . import db
from . import ai_trader, digest
from . import portfolio as pf
from .agent import run_stream
from .config import get_settings
from .db import dumps
from .jobs import start_background
from .llm import DemoModel
from .resolve import llm_asker, resolve
from .symbols import UnknownSymbol, display_name
from .tools.market import fetch_series, series_overview

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not db.is_ready():
        db.init_db()
    minutes = int(os.getenv("SCHEDULER_MINUTES", "0") or 0)
    if minutes > 0:
        start_background(minutes)
    if digest.digest_times():
        digest.start_background(digest.digest_times())
    yield


app = FastAPI(title="invest-agent", lifespan=lifespan)


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    session_id: str = Field(default="web", max_length=64)


def _authorized(token: str | None) -> bool:
    """设了 ACCESS_PASSWORD 时，只有带对密码的请求才用真模型（花钱），其他访客自动走演示模式（免费）。"""
    password = os.getenv("ACCESS_PASSWORD", "")
    if not password:
        return True
    return bool(token) and hmac.compare_digest(token.removeprefix("Bearer "), password)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/eval")
def eval_page() -> FileResponse:
    return FileResponse(STATIC / "eval.html")


@app.get("/stock")
def stock_page() -> FileResponse:
    return FileResponse(STATIC / "stock.html")


@app.get("/portfolio")
def portfolio_page() -> FileResponse:
    return FileResponse(STATIC / "portfolio.html")


@app.get("/digest")
def digest_page() -> FileResponse:
    return FileResponse(STATIC / "digest.html")


@app.get("/static/{name}")
def static_file(name: str) -> FileResponse:
    path = STATIC / name
    if path.parent != STATIC or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/series")
def api_series(symbol: str, days: int = 22, authorization: str | None = Header(default=None)) -> dict:
    """股票页面画图用：价格点、关键数字和大盘走势。

    名字先查内置名单和缓存；都没有时，有密码的请求会让模型识别代码，并用行情接口核实（访客不调用模型，不花钱）。
    """
    if not symbol.strip():
        raise HTTPException(404, "请输入股票名或代码")
    ask = llm_asker() if _authorized(authorization) else None
    try:
        r = resolve(symbol, ask=ask, verify=lambda code: fetch_series(code, 5))
        data = series_overview(r.symbol, days)
    except UnknownSymbol as e:
        no_model_for_visitor = ask is None and not get_settings().demo_mode
        hint = "（在问答页填了访问密码后，可以让模型识别名单外的名字）" if no_model_for_visitor else ""
        raise HTTPException(404, str(e) + hint)
    except Exception as e:  # 网络等其他问题
        raise HTTPException(502, f"暂时查不到 {symbol} 的数据：{e}")
    known = display_name(r.symbol)
    data["name"] = f"{r.name}（{r.symbol}）" if known == r.symbol and r.name else known
    data["resolved_by"] = r.source
    data["resolve_note"] = r.note
    return data


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True, "demo_mode": get_settings().demo_mode}


@app.post("/api/ask")
def api_ask(body: AskBody, authorization: str | None = Header(default=None)) -> StreamingResponse:
    """流式返回 agent 的每一步，格式是 Server-Sent Events：每条 `data: {json}`。"""
    model = None if _authorized(authorization) else DemoModel()

    def events():
        try:
            for e in run_stream(body.question, body.session_id, model):
                yield f"data: {dumps(e)}\n\n"
        except Exception as e:  # 模型或网络出错：告诉页面，而不是让连接悄悄断掉
            yield f"data: {dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/digest")
def api_digest(limit: int = 7) -> list[dict]:
    """最近几份新闻摘要，新的在前。"""
    with db.session() as s:
        rows = s.scalars(select(db.Digest).order_by(db.Digest.id.desc()).limit(min(limit, 30))).all()
        return [{"id": r.id, **r.data} for r in rows]


@app.post("/api/digest/run")
def api_digest_run(push: bool = False, authorization: str | None = Header(default=None)) -> dict:
    """马上生成一份摘要。会搜新闻、调模型（花钱），所以要访问密码；push=true 时同时推送到微信。"""
    if not _authorized(authorization):
        raise HTTPException(401, "生成摘要需要访问密码，访客可以看已经生成的摘要")
    try:
        return digest.run_once(pusher=None if push else (lambda *a: False))
    except Exception as e:
        raise HTTPException(502, f"生成摘要失败：{e}")


class OpenBody(BaseModel):
    amount_cny: float


class TradeBody(BaseModel):
    action: str = Field(pattern="^(buy|sell)$")
    symbol: str = Field(min_length=1, max_length=40)
    amount_cny: float = 0
    fraction: float = 1.0


def _need_password(authorization: str | None) -> None:
    if not _authorized(authorization):
        raise HTTPException(401, "这个操作需要访问密码（在问答页底部填）")


@app.get("/api/portfolio")
def api_portfolio() -> dict:
    """两个模拟账户的全部信息，加上 AI 最近几天的决定。"""
    with db.session() as s:
        rows = s.scalars(select(db.AiDecision).order_by(db.AiDecision.id.desc()).limit(10)).all()
        decisions = [{"id": r.id, "at": r.created_at.isoformat(), **r.data} for r in rows]
    try:
        return {"user": pf.summary("user"), "ai": pf.summary("ai"), "decisions": decisions}
    except Exception as e:
        raise HTTPException(502, f"暂时查不到行情：{e}")


@app.post("/api/portfolio/open")
def api_portfolio_open(body: OpenBody, authorization: str | None = Header(default=None)) -> dict:
    _need_password(authorization)
    try:
        pf.open_account("user", body.amount_cny)
    except pf.TradeError as e:
        raise HTTPException(400, str(e))
    try:
        pf.open_account("ai", body.amount_cny)  # AI 账户默认用同样的金额，已经开过就不动
    except pf.TradeError:
        pass
    return api_portfolio()


@app.post("/api/portfolio/trade")
def api_portfolio_trade(body: TradeBody, authorization: str | None = Header(default=None)) -> dict:
    """页面上的下单表单：只操作用户自己的账户。"""
    _need_password(authorization)
    try:
        sym = resolve(body.symbol, ask=llm_asker(), verify=lambda code: fetch_series(code, 5)).symbol
        if body.action == "buy":
            pf.buy("user", sym, body.amount_cny)
        else:
            pf.sell("user", sym, body.fraction)
    except (pf.TradeError, UnknownSymbol) as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(502, f"下单失败：{e}")
    return api_portfolio()


@app.post("/api/portfolio/ai-run")
def api_portfolio_ai_run(authorization: str | None = Header(default=None)) -> dict:
    """马上让 AI 账户做一次决定（平时每天早上自动跑）。用最近一份新闻摘要。"""
    _need_password(authorization)
    with db.session() as s:
        latest = s.scalar(select(db.Digest).order_by(db.Digest.id.desc()))
        d = latest.data if latest else None
    try:
        return ai_trader.run_once(d)
    except Exception as e:
        raise HTTPException(502, f"AI 决定失败：{e}")


@app.get("/api/runs")
def api_runs(limit: int = 20) -> list[dict]:
    with db.session() as s:
        runs = s.scalars(select(db.Run).order_by(db.Run.id.desc()).limit(min(limit, 100))).all()
        return [{"id": r.id, "question": r.question, "answer": r.answer, "model": r.model,
                 "duration_ms": r.duration_ms, "steps": [st.payload for st in r.steps]} for r in runs]


@app.get("/api/eval")
def api_eval() -> dict:
    """每次评测一行汇总（按时间倒序），加上最近一次的逐题结果。"""
    with db.session() as s:
        R = db.EvalResult
        batches = s.execute(
            select(R.batch, R.model, func.count(), func.sum(R.rule_pass.cast(db.Integer)), func.avg(R.judge_score),
                   func.min(R.created_at))
            .group_by(R.batch, R.model).order_by(func.min(R.created_at).desc()).limit(20)).all()
        summary = [{"batch": b, "model": m, "cases": n, "rule_pass": int(p or 0),
                    "judge_avg": round(j, 2) if j is not None else None, "at": t.isoformat()}
                   for b, m, n, p, j, t in batches]
        latest = []
        if summary:
            rows = s.scalars(select(R).where(R.batch == summary[0]["batch"]).order_by(R.id)).all()
            latest = [{"case_id": r.case_id, "rule_pass": r.rule_pass, "judge_score": r.judge_score,
                       "detail": r.detail} for r in rows]
        return {"summary": summary, "latest": latest}
