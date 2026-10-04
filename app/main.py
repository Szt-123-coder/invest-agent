"""FastAPI 服务：网页、评测脚本、定时任务都通过这里的接口使用 agent。

启动：uvicorn app.main:app --reload
"""

from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from . import db
from .agent import run_stream
from .config import get_settings
from .db import dumps
from .jobs import start_background
from .llm import DemoModel

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    if not db.is_ready():
        db.init_db()
    minutes = int(os.getenv("SCHEDULER_MINUTES", "0") or 0)
    if minutes > 0:
        start_background(minutes)
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
