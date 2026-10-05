"""提醒、关注和偏好工具。这些工具会真的写数据库，返回值里 ok=false 就代表没做成。

系统提示词要求模型：只有工具返回 ok=true，才能对用户说「已设置」。
这是 fx-bot 里「AI 说设好了其实没设」那个问题的修复方式。
"""

from __future__ import annotations

from langchain_core.tools import tool
from sqlalchemy import select

from .. import db
from ..db import dumps
from ..resolve import to_code
from .market import fetch_series


def _fail(msg: str) -> str:
    return dumps({"ok": False, "error": msg})


@tool
def set_price_alert(symbol: str, target: float, direction: str = "auto") -> str:
    """设一个价位提醒：价格到达 target 时通知。direction 是 up（涨到）、down（跌到）或 auto（按当前价自动判断）。"""
    sym = to_code(symbol)
    if direction not in ("up", "down", "auto"):
        return _fail("direction 只能是 up、down 或 auto")
    try:
        target = float(target)
    except (TypeError, ValueError):
        return _fail(f"价位 {target!r} 不是数字")
    if target <= 0:
        return _fail("价位必须大于 0")
    try:
        price = fetch_series(sym, 5)[-1][1]
    except Exception:  # 查不到现价就不检查，照常设置
        price = None
    if price is not None and ((direction == "down" and price <= target) or (direction == "up" and price >= target)):
        where = "低于" if direction == "down" else "高于"
        return _fail(f"{sym} 现在是 {price}，已经{where}目标 {target}，这个提醒会立刻触发。"
                     f"请和用户确认：是想等它{'跌破更低' if direction == 'down' else '涨过更高'}的价位，"
                     f"还是等它{'涨回' if direction == 'down' else '跌回'} {target}")
    with db.session() as s:
        a = db.Alert(symbol=sym, kind="price", direction=direction, target=target)
        s.add(a)
        s.commit()
        return dumps({"ok": True, "alert_id": a.id, "symbol": sym, "target": target, "direction": direction})


@tool
def set_move_alert(symbol: str, direction: str = "down", step_pct: float = 0.2) -> str:
    """设一个波动提醒：从最近高点每跌（或从低点每涨）step_pct 百分比就通知一次。
    用户说「跌了就告诉我」这类没有具体价位的话时用它。direction 是 up、down 或 both。"""
    sym = to_code(symbol)
    if direction not in ("up", "down", "both"):
        return _fail("direction 只能是 up、down 或 both")
    try:
        step = float(step_pct)
    except (TypeError, ValueError):
        return _fail(f"幅度 {step_pct!r} 不是数字")
    if not 0 < step <= 20:
        return _fail("幅度要在 0 到 20 之间（单位是百分比）")
    with db.session() as s:
        a = db.Alert(symbol=sym, kind="move", direction=direction, step=step)
        s.add(a)
        s.commit()
        return dumps({"ok": True, "alert_id": a.id, "symbol": sym, "direction": direction, "step_pct": step})


@tool
def list_alerts() -> str:
    """列出所有提醒和关注的标的。"""
    with db.session() as s:
        alerts = s.scalars(select(db.Alert).order_by(db.Alert.id)).all()
        watch = s.scalars(select(db.WatchItem).order_by(db.WatchItem.id)).all()
        return dumps({"ok": True,
                      "alerts": [{"id": a.id, "symbol": a.symbol, "kind": a.kind, "direction": a.direction,
                                  "target": a.target, "step_pct": a.step} for a in alerts],
                      "watchlist": [w.symbol for w in watch]})


@tool
def delete_alert(alert_id: int) -> str:
    """按编号删除一个提醒。编号先用 list_alerts 查。"""
    with db.session() as s:
        a = s.get(db.Alert, int(alert_id))
        if not a:
            return _fail(f"没有编号为 {alert_id} 的提醒")
        s.delete(a)
        s.commit()
        return dumps({"ok": True, "deleted": int(alert_id)})


@tool
def watch(symbol: str, remove: bool = False) -> str:
    """把一个标的加入（remove=false）或移出（remove=true）关注列表。关注的标的会出现在每日总结里。"""
    sym = to_code(symbol)
    with db.session() as s:
        item = s.scalar(select(db.WatchItem).where(db.WatchItem.symbol == sym))
        if remove:
            if not item:
                return _fail(f"{sym} 本来就不在关注列表里")
            s.delete(item)
        elif not item:
            s.add(db.WatchItem(symbol=sym))
        s.commit()
        return dumps({"ok": True, "symbol": sym, "watching": not remove})


@tool
def remember_preference(key: str, value: str) -> str:
    """记住用户的一个长期偏好，下次对话也会记得。key 是简短的英文名（如 focus、risk），value 是内容（如「只关心换汇」）。"""
    if not key.strip() or not value.strip():
        return _fail("key 和 value 都不能为空")
    with db.session() as s:
        p = s.scalar(select(db.Preference).where(db.Preference.key == key))
        if p:
            p.value = value
        else:
            s.add(db.Preference(key=key, value=value))
        s.commit()
    return dumps({"ok": True, "key": key, "value": value})


def preferences_text() -> str:
    with db.session() as s:
        prefs = s.scalars(select(db.Preference)).all()
    return "\n".join(f"- {p.key}: {p.value}" for p in prefs)
