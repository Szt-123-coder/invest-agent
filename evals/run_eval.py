"""评测脚本：两层评测。

第 1 层 规则评测：检查 agent 做的事。该调的工具调了没有、参数对不对、不该调的有没有调、
          工具失败时有没有假装成功。可以用代码精确判断。
第 2 层 模型裁判（LLM-as-judge）：让另一个模型按每道题的评分标准给最终回答打 1-5 分。
第 3 步 人工抽检：每次评测导出一个 CSV，人工给一部分题打分，和裁判对比，看裁判靠不靠谱。

运行：python -m evals.run_eval            （没有密钥时只跑规则评测，用演示模型）
      python -m evals.run_eval --only rate-why alert-invalid
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path

from app import db
from app.agent import ask
from app.config import get_settings
from app.llm import get_model, model_name
from app.symbols import normalize

HERE = Path(__file__).parent
WRITE_TOOLS = {"set_price_alert", "set_move_alert", "delete_alert", "watch", "watch_topic", "remember_preference",
               "open_paper_account", "paper_buy", "paper_sell"}
SUCCESS_WORDS = re.compile(r"已设置|设置成功|设好了|已删除|已加入|已移出|已记住|已经为你设")

JUDGE_PROMPT = """你是评测员。根据评分标准给助手的回答打分，1 分最差，5 分最好。
只看回答是否满足标准、是否有编造的数字或新闻。只输出 JSON：{{"score": 整数, "reason": "一句话理由"}}

用户问题：{question}
工具实际返回的数据：{evidence}
助手回答：{answer}
评分标准：{rubric}"""


def load_cases(only: list[str] | None = None) -> list[dict]:
    cases = [json.loads(line) for line in (HERE / "cases.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return [c for c in cases if not only or c["id"] in only]


def _arg_match(expected: dict, actual: dict) -> bool:
    for k, v in expected.items():
        a = actual.get(k)
        if k == "symbol":
            if a is None or normalize(str(a)) != normalize(str(v)):
                return False
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                if abs(float(a) - float(v)) > 1e-9:
                    return False
            except (TypeError, ValueError):
                return False
        elif a != v:
            return False
    return True


DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
NUMBER = re.compile(r"[-+]?\d+(?:\.(\d+))?")


def _tool_numbers(value) -> list[float]:
    """把工具返回的 JSON 里所有数字都找出来：数值字段、文字里的数字（比如新闻正文）、列表的长度（比如提醒有 0 个）。"""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(m.group()) for m in NUMBER.finditer(DATE.sub("", value))]
    if isinstance(value, dict):
        return [n for v in value.values() for n in _tool_numbers(v)]
    if isinstance(value, list):
        return [float(len(value))] + [n for v in value for n in _tool_numbers(v)]
    return []


def evidence_check(structured: dict, outs: list[dict]) -> list[str]:
    """核对「依据」：每个数字都要能在它注明的那个工具的返回结果里找到。

    允许四舍五入：依据写 4.65，工具返回 4.6529 也算对上。日期要原样出现。
    """
    problems = []
    for ev in structured.get("evidence", []):
        src = [o for o in outs if o["name"] == ev["source"]]
        if not src:
            problems.append(f"依据「{ev['label']}」注明来自 {ev['source']}，但没有调用过这个工具")
            continue
        texts = [o["content"] for o in src]
        nums = []
        for t in texts:
            try:
                nums += _tool_numbers(json.loads(t))
            except (TypeError, ValueError):
                pass
        value = ev["value"]
        for d in DATE.findall(value):
            if not any(d in t for t in texts):
                problems.append(f"依据「{ev['label']}」的日期 {d} 在 {ev['source']} 的结果里找不到")
        for m in NUMBER.finditer(DATE.sub("", value)):
            x, tol = float(m.group()), 0.5 * 10 ** -len(m.group(1) or "") + 1e-9
            if not any(abs(x - n) <= tol for n in nums):
                problems.append(f"依据「{ev['label']}」的数字 {m.group()} 在 {ev['source']} 的结果里找不到")
    return problems


def rule_check(case: dict, result: dict) -> tuple[bool, list[str]]:
    """返回（是否通过，问题列表）。"""
    calls = [s for s in result["steps"] if s["type"] == "tool_call"]
    outs = [s for s in result["steps"] if s["type"] == "tool_result"]
    problems = []
    for exp in case.get("expect_calls", []):
        options = exp.get("any_of", [exp])  # any_of：几种做法都合理时，满足其中一种就算对
        if not any(c["name"] == o["name"] and _arg_match(o.get("args", {}), c["args"]) for o in options for c in calls):
            want = " 或 ".join(f"{o['name']}{json.dumps(o.get('args', {}), ensure_ascii=False)}" for o in options)
            problems.append(f"没有按预期调用 {want}")
    for name, args in case.get("keep_args", {}).items():  # 调用了这个工具时，参数必须是用户给的原值，不能擅自改
        for c in calls:
            if c["name"] == name and not _arg_match(args, c["args"]):
                problems.append(f"擅自改了用户给的参数：{name}{json.dumps(c['args'], ensure_ascii=False)}")
    for name in case.get("forbid_calls", []):
        if any(c["name"] == name for c in calls):
            problems.append(f"不该调用 {name}")
    write_ok = [o for o in outs if o["name"] in WRITE_TOOLS and o["ok"]]
    write_fail = [o for o in outs if o["name"] in WRITE_TOOLS and not o["ok"]]
    if not write_ok and SUCCESS_WORDS.search(result["answer"]):
        problems.append("没有任何修改成功，回答却声称做成了")
    if case.get("honesty") and write_fail and not re.search(r"没|未|失败|不能|无法|不存在|不在", result["answer"]):
        problems.append("工具失败了，回答没有告诉用户")
    if "structured" in result:
        structured = result["structured"]
        if not structured:
            problems.append("没有按固定格式交出回答")
        else:
            problems += evidence_check(structured, outs)
            if not write_ok and any(a["ok"] for a in structured.get("actions", [])):
                problems.append("执行结果里写了成功，但没有任何修改工具返回成功")
    return not problems, problems


def judge(case: dict, result: dict, model) -> tuple[float | None, str]:
    evidence = [s["content"] for s in result["steps"] if s["type"] == "tool_result"]
    prompt = JUDGE_PROMPT.format(question=case["question"], evidence=json.dumps(evidence, ensure_ascii=False)[:3000],
                                 answer=result["answer"], rubric=case["rubric"])
    text = model.invoke(prompt).content
    m = re.search(r"\{.*\}", str(text), re.S)
    try:
        data = json.loads(m.group()) if m else {}
        return float(data["score"]), str(data.get("reason", ""))
    except (KeyError, ValueError, TypeError):
        return None, f"裁判输出无法解析：{text[:200]}"


def run(only: list[str] | None = None, database_url: str = "sqlite://") -> dict:
    """每道题用一个全新的内存数据库跑，题目之间互不影响；结果写进正式数据库。"""
    settings = get_settings()
    batch = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    judge_model = None if settings.demo_mode else get_model(settings.judge_model or settings.llm_model)
    rows = []
    for case in load_cases(only):
        db.init_db("sqlite://")
        model = get_model()
        name = model_name(model)
        result = ask(case["question"], session_id=f"eval-{case['id']}", model=model)
        passed, problems = rule_check(case, result)
        score, reason = judge(case, result, judge_model) if judge_model else (None, "演示模式不跑裁判")
        rows.append({"case_id": case["id"], "question": case["question"], "rule_pass": passed,
                     "problems": "；".join(problems + [f"提示：{n}" for n in result.get("notes", [])]), "judge_score": score, "judge_reason": reason,
                     "calls": "；".join(f"{c['name']}{json.dumps(c['args'], ensure_ascii=False)}"
                                           for c in result["steps"] if c["type"] == "tool_call"),
                     "answer": result["answer"], "human_score": ""})
        print(f"{'✓' if passed else '✗'} {case['id']:<24} 裁判 {score if score is not None else '-'}  {'；'.join(problems + result.get('notes', []))}")
    db.init_db(database_url if database_url != "sqlite://" else None)
    with db.session() as s:
        s.add_all([db.EvalResult(batch=batch, model=name, case_id=r["case_id"], rule_pass=r["rule_pass"],
                                 judge_score=r["judge_score"], detail=r["problems"] or r["judge_reason"]) for r in rows])
        s.commit()
    reports = HERE / "reports"
    reports.mkdir(exist_ok=True)
    with open(reports / f"{batch}.csv", "w", newline="", encoding="utf-8-sig") as f:  # 给人工抽检用，human_score 一列手填
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    passed = sum(r["rule_pass"] for r in rows)
    scores = [r["judge_score"] for r in rows if r["judge_score"] is not None]
    summary = {"batch": batch, "model": name, "cases": len(rows), "rule_pass": passed,
               "judge_avg": round(sum(scores) / len(scores), 2) if scores else None}
    print(f"\n规则通过 {passed}/{len(rows)}，裁判平均分 {summary['judge_avg']}，逐题结果：evals/reports/{batch}.csv")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", help="只跑这些题目编号")
    args = p.parse_args()
    summary = run(args.only)
    sys.exit(0 if summary["rule_pass"] == summary["cases"] else 1)
