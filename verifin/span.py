"""span 硬校验：把「模型可能编造」转成「字符串在不在原文里」。

这是整个系统防幻觉的核心机制，也是本项目最值得展开讲的设计点。

朴素做法是在提示词里反复叮嘱模型「不要编造」，但这不可验证——
你无法判断模型到底有没有照做。本模块换一个思路：
要求模型在报出数值的同时，**必须回吐它认为的原文片段**，
然后由程序做字符串级校验。这样「模型是否撒谎」就变成了
「这个字符串在不在原文里」这个可以判定的问题。

校验分两层，缺一不可：

1. **片段命中**（:func:`locate_span`）：声称的片段必须能在原文中找到。
2. **数值一致**（:func:`value_matches_span`）：声称的数值必须能从该片段中解析出来。

只做第 1 层是不够的。模型完全可以引用一段真实存在的原文，
却报出一个错误的数字——第 1 层会放行。财务场景下这种错误最致命，
因为它读起来「有据可查」。
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum

from verifin.normalize import (
    UNIT_MULTIPLIER,
    normalize_text,
    parse_amounts,
)

__all__ = [
    "MatchMode",
    "SpanHit",
    "EvidenceVerdict",
    "locate_span",
    "value_matches_span",
    "verify_evidence",
]


class MatchMode(str, Enum):
    """片段是怎么命中的。区分这两者是为了让报告能如实反映降级情况。"""

    EXACT = "exact"
    """逐字命中。最可信，位置可直接用于 PDF 高亮。"""

    NORMALIZED = "normalized"
    """归一化后命中。差异仅来自空白或全角半角，不涉及任何字符改动。"""


@dataclass(frozen=True)
class SpanHit:
    """片段命中的位置信息。

    Attributes:
        mode: 命中方式。
        start: 在**归一化文本**中的起始下标。
        end: 结束下标（不含）。
        original_text: 命中处的原始文本。若 `mode` 为 ``EXACT``，
            该值等于声称的片段；若为 ``NORMALIZED``，则取自原文的对应区间，
            可能与声称的片段在空白上略有出入。
    """

    mode: MatchMode
    start: int
    end: int
    original_text: str

    @property
    def length(self) -> int:
        return self.end - self.start


def locate_span(claimed_span: str, chunk_text: str) -> SpanHit | None:
    """在原文中定位声称的片段。

    先做逐字匹配；失败则退到归一化匹配（抹平空白与全角半角差异）。

    Args:
        claimed_span: 模型声称的原文片段。
        chunk_text: 检索到的原文块（未归一化）。

    Returns:
        命中信息；完全找不到时返回 `None`。

    Note:
        `NORMALIZED` 模式下返回的下标是**归一化文本**的下标，
        不能直接用于原始文本切片。做 PDF 高亮时应改用 `original_text`
        重新在原文中定位，或改用解析阶段保留的 bbox。
    """
    if not claimed_span or not claimed_span.strip() or not chunk_text:
        return None

    # 第一层：逐字匹配。位置可直接使用。
    idx = chunk_text.find(claimed_span)
    if idx >= 0:
        return SpanHit(
            mode=MatchMode.EXACT,
            start=idx,
            end=idx + len(claimed_span),
            original_text=claimed_span,
        )

    # 第二层：归一化匹配。只抹平形式差异，不改动任何字符。
    n_claim = normalize_text(claimed_span)
    if not n_claim:
        return None
    n_chunk = normalize_text(chunk_text)
    idx = n_chunk.find(n_claim)
    if idx < 0:
        return None

    return SpanHit(
        mode=MatchMode.NORMALIZED,
        start=idx,
        end=idx + len(n_claim),
        original_text=n_claim,
    )


def _to_yuan(value: Decimal, unit: str | None) -> Decimal:
    return value * UNIT_MULTIPLIER.get(unit or "", Decimal(1))


def value_matches_span(
    span_text: str,
    claimed_value: Decimal,
    claimed_unit: str | None,
    *,
    rel_tolerance: Decimal = Decimal(0),
) -> bool:
    """校验声称的数值能否从片段中解析出来。

    把片段里出现的每个金额统一换算到「元」，与声称值比对。
    只要有一个命中即算通过——财务表格一行常有多列数字（本期/上期），
    片段里存在多个数字是正常的。

    Args:
        span_text: 已命中的原文片段。
        claimed_value: 模型声称的数值。
        claimed_unit: 模型声称的单位。`None` 表示文本未写单位。
        rel_tolerance: 相对容差。默认 0，即要求精确一致。
            若允许模型做四舍五入（如把 `1,234,567,890.12 元` 报成 `12.35 亿元`），
            可设为 `Decimal("0.001")`。**默认关闭是有意的**：
            本项目优先保证「不出现未经确认的数字」，而不是「少报几个数」。

    Returns:
        是否存在匹配的金额。

    Known limitation:
        本函数按「片段里出现过这个数字」判定，**不区分数字的语义角色**。
        若片段含日期（如 `2023-12-31`），年份 `2023` 会被当成一个金额，
        因此声称值为 `2023` 时会误判为通过。
        这是刻意接受的取舍：真正的语义归属由表格结构层负责，
        此处只做一道廉价的合理性闸门。
        因此在计数「引用支持度」时，本函数应作为必要条件而非充分条件。
    """
    amounts = parse_amounts(span_text)
    if not amounts:
        return False

    expected = _to_yuan(claimed_value, claimed_unit)

    for amount in amounts:
        if amount.normalized == expected:
            return True
        if rel_tolerance > 0 and expected != 0:
            deviation = abs(amount.normalized - expected) / abs(expected)
            if deviation <= rel_tolerance:
                return True

    return False


@dataclass(frozen=True)
class EvidenceVerdict:
    """单条证据的校验结论。

    Attributes:
        accepted: 是否通过全部校验。False 表示该条证据必须被丢弃。
        span_hit: 片段命中信息；未命中为 `None`。
        value_ok: 数值一致性是否通过。
        reason: 未通过时的原因码，通过时为 `None`。
        detail: 面向日志与失败案例归档的可读说明。
    """

    accepted: bool
    span_hit: SpanHit | None
    value_ok: bool
    reason: str | None
    detail: str

    @property
    def match_mode(self) -> MatchMode | None:
        return self.span_hit.mode if self.span_hit else None


#: 未通过的原因码。用常量而非裸字符串，便于评测脚本按类别统计。
REASON_EMPTY_SPAN = "EMPTY_SPAN"
REASON_SPAN_NOT_FOUND = "SPAN_NOT_FOUND"
REASON_VALUE_NOT_IN_SPAN = "VALUE_NOT_IN_SPAN"


def verify_evidence(
    *,
    claimed_span: str,
    chunk_text: str,
    claimed_value: Decimal,
    claimed_unit: str | None,
    rel_tolerance: Decimal = Decimal(0),
) -> EvidenceVerdict:
    """对一条抽取证据执行完整的两层校验。

    这是工具层 `extract_evidence` 返回前的最后一道闸门。
    未通过的证据**不得**进入计算层，也不得出现在最终答案里。
    """
    if not claimed_span or not claimed_span.strip():
        return EvidenceVerdict(
            accepted=False,
            span_hit=None,
            value_ok=False,
            reason=REASON_EMPTY_SPAN,
            detail="模型未提供原文片段，无法追溯来源。",
        )

    hit = locate_span(claimed_span, chunk_text)
    if hit is None:
        return EvidenceVerdict(
            accepted=False,
            span_hit=None,
            value_ok=False,
            reason=REASON_SPAN_NOT_FOUND,
            detail=f"片段在原文中不存在：{claimed_span[:60]!r}",
        )

    value_ok = value_matches_span(
        hit.original_text,
        claimed_value,
        claimed_unit,
        rel_tolerance=rel_tolerance,
    )
    if not value_ok:
        return EvidenceVerdict(
            accepted=False,
            span_hit=hit,
            value_ok=False,
            reason=REASON_VALUE_NOT_IN_SPAN,
            detail=(
                f"片段命中，但数值 {claimed_value} {claimed_unit or ''} "
                f"无法从该片段解析出来：{hit.original_text[:60]!r}"
            ),
        )

    return EvidenceVerdict(
        accepted=True,
        span_hit=hit,
        value_ok=True,
        reason=None,
        detail=f"校验通过（{hit.mode.value} 命中，长度 {hit.length}）。",
    )
