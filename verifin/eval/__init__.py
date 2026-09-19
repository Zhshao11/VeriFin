"""评测层：题库模型、题库自检、文档登记。

设计意图见 `docs/评测方案-v1.0.md`。三层职责：

- `schema.py`     一条题长什么样，以及**程序级**的格式校验
- `bank.py`       题库（JSONL）的读写与构成统计
- `gold_check.py` gold 标注自检（三道独立闸门，其中一道只看原始 PDF）
"""

from verifin.eval.bank import (
    bank_stats,
    load_bank,
    render_stats,
    report_problems,
    save_bank,
    validate_bank_or_raise,
)
from verifin.eval.docs import DOC_REGISTRY, DocSpec, get_spec
from verifin.eval.schema import (
    QUESTION_TYPE_LABELS,
    QUESTION_TYPES,
    EvalItem,
    formula_kind,
    is_verdict_formula,
    validate_bank,
    validate_item,
)

#: `gold_check` 用**惰性**方式再导出（PEP 562）。
#:
#: 原因不是洁癖：`gold_check.py` 自带命令行入口，
#: 而本 `__init__` 若在包导入期就把它拉进来，`python -m verifin.eval.gold_check`
#: 会打印一条 "found in sys.modules after import of package" 的 RuntimeWarning，
#: 把自检命令的干净输出弄脏。惰性再导出让 `from verifin.eval import check_bank`
#: 与 `python -m verifin.eval.gold_check` 两者都成立。
_LAZY_GOLD_CHECK = {
    "GoldCheck",
    "check_item",
    "check_bank",
    "load_index_pages",
    "pdf_page_text",
    "render_report",
    "summarize",
}


def __getattr__(name: str):
    if name in _LAZY_GOLD_CHECK:
        from verifin.eval import gold_check

        return getattr(gold_check, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DOC_REGISTRY",
    "DocSpec",
    "get_spec",
    "EvalItem",
    "QUESTION_TYPES",
    "QUESTION_TYPE_LABELS",
    "formula_kind",
    "is_verdict_formula",
    "validate_item",
    "validate_bank",
    "load_bank",
    "save_bank",
    "bank_stats",
    "render_stats",
    "report_problems",
    "validate_bank_or_raise",
    # gold_check（惰性）
    "GoldCheck",
    "check_item",
    "check_bank",
    "load_index_pages",
    "pdf_page_text",
    "render_report",
    "summarize",
]
