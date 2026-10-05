import pytest

from app import db


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    """每个测试用全新的内存数据库，并强制演示模式：不联网、不花钱。"""
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DATABASE_URL", "sqlite://")
    monkeypatch.delenv("ACCESS_PASSWORD", raising=False)
    monkeypatch.delenv("SCHEDULER_MINUTES", raising=False)
    monkeypatch.delenv("DIGEST_TIMES", raising=False)
    db.init_db("sqlite://")
    yield
