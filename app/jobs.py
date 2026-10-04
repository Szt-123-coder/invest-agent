"""定时任务：检查提醒，触发时推送到微信（PushPlus）。

逻辑和 fx-bot 一致：
- 价位提醒：离目标 0.5% 以内提醒一次「接近」，到达再提醒一次「已到」，远离后重新布防
- 波动提醒：从最近高点跌了 step%（或从低点涨了 step%）提醒一次，然后以新价格为起点继续盯

运行一次：python -m app.jobs
服务里自动运行：设环境变量 SCHEDULER_MINUTES=15
"""

from __future__ import annotations

import logging
import os
import threading
import time

import httpx
from sqlalchemy import select

from . import db
from .tools.market import fetch_series

log = logging.getLogger(__name__)
APPROACH_PCT = 0.5


def check_alert(a: db.Alert, price: float) -> str | None:
    """返回要推送的消息；没有就返回 None。会更新 a.state。"""
    st = dict(a.state or {})
    msg = None
    if a.kind == "price":
        direction = a.direction
        if direction == "auto":
            direction = st.get("dir") or ("up" if price < a.target else "down")
            st["dir"] = direction
        reached = price >= a.target if direction == "up" else price <= a.target
        gap = abs(price / a.target - 1) * 100
        if reached and not st.get("reached"):
            msg = f"🎯 {a.symbol} 已{'涨' if direction == 'up' else '跌'}到 {price}（提醒线 {a.target}）"
            st.update(reached=True, near=True)
        elif not reached and gap <= APPROACH_PCT and not st.get("near"):
            msg = f"👀 {a.symbol} 接近提醒线 {a.target}，现在 {price}"
            st["near"] = True
        elif not reached and gap > APPROACH_PCT * 2:
            st.update(near=False, reached=False)  # 远离后重新布防
    else:
        hi, lo = max(st.get("high", price), price), min(st.get("low", price), price)
        if "high" in st and a.direction != "up" and (price / hi - 1) * 100 <= -a.step:
            msg = f"📉 {a.symbol} 从 {hi} 跌了 {round((1 - price / hi) * 100, 2)}%，现在 {price}"
            hi = lo = price
        elif "high" in st and a.direction != "down" and (price / lo - 1) * 100 >= a.step:
            msg = f"📈 {a.symbol} 从 {lo} 涨了 {round((price / lo - 1) * 100, 2)}%，现在 {price}"
            hi = lo = price
        st.update(high=hi, low=lo)
    a.state = st
    return msg


def push(title: str, content: str) -> bool:
    token = os.getenv("PUSHPLUS_TOKEN", "")
    if not token:
        log.info("没有 PUSHPLUS_TOKEN，只打印：%s %s", title, content)
        return False
    r = httpx.post("http://www.pushplus.plus/send", json={"token": token, "title": title, "content": content}, timeout=15)
    return r.json().get("code") == 200


def run_once(fetch=fetch_series, pusher=push) -> list[str]:
    msgs = []
    with db.session() as s:
        alerts = s.scalars(select(db.Alert)).all()
        prices: dict[str, float] = {}
        for a in alerts:
            if a.symbol not in prices:
                try:
                    prices[a.symbol] = fetch(a.symbol, 5)[-1][1]
                except Exception as e:
                    log.warning("查不到 %s：%s", a.symbol, e)
                    continue
            m = check_alert(a, prices[a.symbol])
            if m:
                msgs.append(m)
        s.commit()
    for m in msgs:
        pusher("汇率提醒", m)
    return msgs


def start_background(minutes: int) -> None:
    def loop():
        while True:
            try:
                run_once()
            except Exception:
                log.exception("检查提醒失败")
            time.sleep(minutes * 60)

    threading.Thread(target=loop, daemon=True, name="alert-checker").start()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db.init_db()
    print("\n".join(run_once()) or "没有触发的提醒")
