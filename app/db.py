"""数据库：用 SQLAlchemy 访问 SQLite。以后换 PostgreSQL 只需改 DATABASE_URL。"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from .config import get_settings


def now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


# ---------- 长期记忆：跨对话也要记住的东西 ----------

class WatchItem(Base):
    """关注的标的，比如 AUD/CNY、AAPL。"""
    __tablename__ = "watch_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Alert(Base):
    """提醒。kind=price：到某个价位；kind=move：从高点/低点变动超过 step 百分比。"""
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(32))
    kind: Mapped[str] = mapped_column(String(8))
    direction: Mapped[str] = mapped_column(String(8))  # up / down / both
    target: Mapped[float | None] = mapped_column(Float, nullable=True)
    step: Mapped[float | None] = mapped_column(Float, nullable=True)
    # 检查提醒时用的状态：价位提醒记「已接近 / 已到达」，波动提醒记最近的高点和低点
    state: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class Preference(Base):
    """用户偏好，比如「只关心换汇」。会被放进系统提示词里。"""
    __tablename__ = "preferences"
    id: Mapped[int] = mapped_column(primary_key=True)
    key: Mapped[str] = mapped_column(String(64), unique=True)
    value: Mapped[str] = mapped_column(Text)


# ---------- 短期记忆：这次对话说过什么 ----------

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


# ---------- 可观测：agent 每次运行做了哪几步 ----------

class Run(Base):
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(64), default="")
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    steps: Mapped[list[Step]] = relationship(back_populates="run", order_by="Step.id")


class Step(Base):
    __tablename__ = "steps"
    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id"))
    kind: Mapped[str] = mapped_column(String(16))  # tool_call / tool_result / answer
    name: Mapped[str] = mapped_column(String(64), default="")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    run: Mapped[Run] = relationship(back_populates="steps")


# ---------- 评测结果 ----------

class EvalResult(Base):
    __tablename__ = "eval_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    batch: Mapped[str] = mapped_column(String(64), index=True)  # 一次评测的编号
    model: Mapped[str] = mapped_column(String(64))
    case_id: Mapped[str] = mapped_column(String(64))
    rule_pass: Mapped[bool] = mapped_column(default=False)
    judge_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    detail: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


_engine = None
_Session: sessionmaker | None = None


def init_db(url: str | None = None) -> None:
    """建表。测试时传 sqlite:// 用内存数据库。"""
    global _engine, _Session
    url = url or get_settings().database_url
    if url.startswith("sqlite:///") and not url.startswith("sqlite:////"):
        os.makedirs(os.path.dirname(url.removeprefix("sqlite:///")) or ".", exist_ok=True)
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {}
    if url == "sqlite://":
        from sqlalchemy.pool import StaticPool
        kwargs["poolclass"] = StaticPool
    _engine = create_engine(url, **kwargs)
    Base.metadata.create_all(_engine)
    _Session = sessionmaker(_engine, expire_on_commit=False)


def is_ready() -> bool:
    return _Session is not None


def session() -> Session:
    if _Session is None:
        init_db()
    return _Session()


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
