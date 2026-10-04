"""最终回答的固定格式（结构化输出）。

模型查完数据后，不是随手写一段话，而是按 Answer 的字段填表。
LangChain 会把 Answer 当成一个「交卷工具」交给模型，模型调用它就等于交答案，
参数不符合格式时会自动让模型重填。

好处：页面能显示成卡片；评测能用代码核对「依据」里的每个数字是不是工具真查到的。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

DISCLAIMER = "仅供学习参考，不构成投资建议。"


class Evidence(BaseModel):
    label: str = Field(description="这个数字是什么，例如「最新价」「近 30 天涨跌」")
    value: str = Field(description="数字本身，必须原样抄自工具返回的结果，例如「4.6529」「-0.93%」")
    source: str = Field(description="这个数字来自哪个工具，填工具名，例如 get_quote")


class ActionResult(BaseModel):
    action: str = Field(description="做了什么修改，例如「设置价位提醒」")
    ok: bool = Field(description="工具返回 ok: true 才能填 true")
    detail: str = Field(description="成功时写结果（如提醒编号），失败时写原因")


class Answer(BaseModel):
    """最终回答。查完需要的数据后，调用它把答案交给用户。"""

    conclusion: str = Field(description="一两句话的结论，直接回答用户的问题")
    evidence: list[Evidence] = Field(default_factory=list, description="支撑结论的数字，没有查数据时留空")
    risks: list[str] = Field(default_factory=list, description="结论可能出错的情况，没有就留空")
    confidence: Literal["高", "中", "低"] = Field(description="对结论有多大把握")
    confidence_reason: str = Field(description="一句话说明为什么是这个把握")
    actions: list[ActionResult] = Field(default_factory=list, description="只填修改类操作的结果：设提醒、删提醒、改关注、记偏好。查价格、查提醒列表这类查询不要填")


ANSWER_TOOL = Answer.__name__


def render(a: Answer) -> str:
    """把结构化回答转成文字：存进对话记录、给裁判看、给不支持卡片的地方显示。"""
    parts = [a.conclusion]
    if a.evidence:
        parts.append("依据：\n" + "\n".join(f"- {e.label}：{e.value}（{e.source}）" for e in a.evidence))
    if a.risks:
        parts.append("风险：\n" + "\n".join(f"- {r}" for r in a.risks))
    if a.actions:
        parts.append("执行结果：\n" + "\n".join(f"- {'✓' if x.ok else '✗ 没有做成'} {x.action}：{x.detail}"
                                            for x in a.actions))
    parts.append(f"信心：{a.confidence}（{a.confidence_reason}）")
    parts.append(DISCLAIMER)
    return "\n\n".join(parts)
