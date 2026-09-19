"""财报核验 Agent —— 核心领域层。

本包承载项目的三条红线，其余模块（解析、检索、编排）都建立在这三层之上：

1. :mod:`fin_verify.normalize` —— 数字与文本归一化。
   让 `12,345.67`、`12345.67`、`1.23亿元` 归到同一个规范形式，
   这是「数字精确匹配」的地基。

2. :mod:`fin_verify.span` —— span 硬校验。
   模型报数必须同时回吐原文片段，由程序做字符串级校验。
   把「模型是否撒谎」这个不可验证的问题，转成「字符串在不在原文里」这个可判定的问题。

3. :mod:`fin_verify.formulas` —— 勾稽公式注册表。
   算术全部走 :class:`decimal.Decimal`，模型只选公式不算数；
   容差由报表披露单位推导，核验结论分 PASS / WARN / FAIL 三级。

这三个模块**只依赖 Python 标准库**，不引入任何第三方包。
这样核验逻辑可以脱离 LLM、脱离解析层单独测试，
也保证了即使模型服务或解析器不可用，这套判定规则依然可以被验证。
依赖第三方库的部分（Pydantic 数据模型、LLM 客户端）分别放在
:mod:`fin_verify.models` 与 :mod:`fin_verify.llm`，需要时再显式导入。
"""

from fin_verify.compute import (
    ZERO,
    pct_change,
    quantize_amount,
    quantize_pct,
    safe_div,
    to_yuan,
)
from fin_verify.formulas import (
    FORMULA_REGISTRY,
    Formula,
    FormulaKind,
    Outcome,
    Verdict,
    check_cross_currency,
    check_gross_margin,
    check_sum_identity,
    check_yoy_reasonableness,
    derive_tolerance,
    evaluate_formula,
    suspect_operands,
)
from fin_verify.normalize import (
    ParsedAmount,
    find_missing_fields,
    normalize_text,
    parse_amounts,
    parse_single_amount,
    unit_scale,
)
from fin_verify.span import (
    EvidenceVerdict,
    MatchMode,
    SpanHit,
    locate_span,
    value_matches_span,
    verify_evidence,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # normalize
    "ParsedAmount",
    "normalize_text",
    "parse_amounts",
    "parse_single_amount",
    "find_missing_fields",
    "unit_scale",
    # span
    "MatchMode",
    "SpanHit",
    "EvidenceVerdict",
    "locate_span",
    "value_matches_span",
    "verify_evidence",
    # compute
    "ZERO",
    "to_yuan",
    "quantize_pct",
    "quantize_amount",
    "safe_div",
    "pct_change",
    # formulas
    "FormulaKind",
    "Verdict",
    "Formula",
    "Outcome",
    "FORMULA_REGISTRY",
    "derive_tolerance",
    "evaluate_formula",
    "suspect_operands",
    "check_sum_identity",
    "check_gross_margin",
    "check_cross_currency",
    "check_yoy_reasonableness",
]
