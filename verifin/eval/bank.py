"""题库的读写与统计。

题库用 **JSONL**（每行一个 JSON 对象）而不是一个大 JSON：
逐行追加、逐行 diff，改一道题在 git 里就是一行变更，评审时看得清。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from verifin.eval.schema import (
    QUESTION_TYPE_LABELS,
    EvalItem,
    validate_bank,
)


def load_bank(path: str | Path) -> list[EvalItem]:
    """读题库。空行与以 `#` 开头的行会被跳过，方便在题库里写注释。"""
    p = Path(path)
    items: list[EvalItem] = []
    for lineno, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{p.name} 第 {lineno} 行不是合法 JSON：{exc}") from exc
        items.append(EvalItem.from_dict(payload))
    return items


def save_bank(items: Iterable[EvalItem], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(
        json.dumps(item.to_dict(), ensure_ascii=False) for item in items
    )
    p.write_text(body + "\n", encoding="utf-8")


def bank_stats(items: list[EvalItem]) -> dict[str, Any]:
    """题库构成统计。评测报告里要能直接引用这组数字。"""
    by_type = Counter(i.question_type for i in items)
    by_scope = Counter(i.scope for i in items)
    by_doc = Counter(i.doc for i in items)
    by_difficulty = Counter(i.difficulty for i in items)
    answerable = sum(1 for i in items if i.expected_outcome == "ANSWER")
    return {
        "总数": len(items),
        "可答": answerable,
        "应拒答": len(items) - answerable,
        "按题型": {QUESTION_TYPE_LABELS.get(k, k): by_type[k] for k in QUESTION_TYPE_LABELS if by_type[k]},
        "按口径": dict(sorted(by_scope.items())),
        "按文档": dict(sorted(by_doc.items())),
        "按难度": dict(sorted(by_difficulty.items())),
        "公式覆盖": sorted({i.formula for i in items if i.formula}),
    }


def render_stats(stats: dict[str, Any]) -> str:
    lines = [
        f"题库总数 {stats['总数']} 题（可答 {stats['可答']} / 应拒答 {stats['应拒答']}）",
        "题型：" + "  ".join(f"{k} {v}" for k, v in stats["按题型"].items()),
        "口径：" + "  ".join(f"{k} {v}" for k, v in stats["按口径"].items()),
        "难度：" + "  ".join(f"{k} {v}" for k, v in stats["按难度"].items()),
        "文档：" + "  ".join(f"{k} {v}" for k, v in stats["按文档"].items()),
        "公式覆盖：" + ("、".join(stats["公式覆盖"]) or "（无）"),
    ]
    return "\n".join(lines)


def report_problems(report: dict[str, list[str]], limit: int = 10) -> str:
    if not report:
        return "题库格式校验：全部通过"
    lines = [f"题库格式校验：{len(report)} 条不通过"]
    for i, (item_id, problems) in enumerate(report.items()):
        if i >= limit:
            lines.append(f"  … 另有 {len(report) - limit} 条")
            break
        lines.append(f"  [{item_id}]")
        lines.extend(f"    - {p}" for p in problems)
    return "\n".join(lines)


def validate_bank_or_raise(items: list[EvalItem]) -> None:
    report = validate_bank(items)
    if report:
        raise ValueError(report_problems(report))


__all__ = [
    "load_bank",
    "save_bank",
    "bank_stats",
    "render_stats",
    "report_problems",
    "validate_bank_or_raise",
]
