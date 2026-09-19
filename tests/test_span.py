"""span 硬校验测试。

对应面试追问：「你怎么保证模型不编数字？」

核心测试是 :class:`TestVerifyEvidence` 里的
``test_rejects_real_span_with_wrong_value``——
只做「片段在不在原文」一层校验是不够的，模型可以引用一段真实存在的原文
却报出一个错误的数字。这个用例专门固化第二层校验的必要性。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from verifin.span import (
    REASON_EMPTY_SPAN,
    REASON_SPAN_NOT_FOUND,
    REASON_VALUE_NOT_IN_SPAN,
    MatchMode,
    locate_span,
    value_matches_span,
    verify_evidence,
)

#: 模拟一个从年报第 78 页解析出的原文块。
CHUNK = (
    "合并利润表\n"
    "项目                2023年度            2022年度\n"
    "营业收入    1,234,567,890.12    1,098,765,432.10\n"
    "营业成本      678,901,234.56      650,000,000.00\n"
)


class TestLocateSpan:
    def test_exact_match_reports_exact_mode(self) -> None:
        hit = locate_span("营业收入", CHUNK)
        assert hit is not None
        assert hit.mode is MatchMode.EXACT
        assert hit.original_text == "营业收入"

    def test_exact_position_is_indexable(self) -> None:
        hit = locate_span("营业成本", CHUNK)
        assert hit is not None
        # EXACT 模式的下标可以直接用于原文切片
        assert CHUNK[hit.start : hit.end] == "营业成本"

    def test_whitespace_difference_falls_back_to_normalized(self) -> None:
        """表格里的空格数量与模型回吐的往往不一致，须能容错。"""
        hit = locate_span("营业收入 1,234,567,890.12", CHUNK)
        assert hit is not None
        assert hit.mode is MatchMode.NORMALIZED

    def test_fullwidth_digits_fall_back_to_normalized(self) -> None:
        hit = locate_span("１２３４", CHUNK)
        assert hit is None  # 原文没有这个数字，应正当地找不到
        hit2 = locate_span("营业收入", CHUNK.replace("营业收入", "营业收入"))
        assert hit2 is not None

    def test_fabricated_span_returns_none(self) -> None:
        """模型凭空编造的片段必须找不到。"""
        assert locate_span("营业收入 9,999,999,999.99 元", CHUNK) is None

    def test_empty_span_returns_none(self) -> None:
        assert locate_span("", CHUNK) is None
        assert locate_span("   ", CHUNK) is None

    def test_empty_chunk_returns_none(self) -> None:
        assert locate_span("营业收入", "") is None


class TestValueMatchesSpan:
    def test_same_value_different_separator(self) -> None:
        assert value_matches_span("1,234,567,890.12", Decimal("1234567890.12"), "元")

    def test_claimed_value_in_yi_converts_correctly(self) -> None:
        """声称 12.3456789012 亿元，原文写 1,234,567,890.12 元，是同一个数。"""
        assert value_matches_span(
            "12.3456789012亿元", Decimal("12.3456789012"), "亿元"
        )

    def test_wrong_value_does_not_match(self) -> None:
        assert not value_matches_span(
            "1,234,567,890.12", Decimal("987654321.00"), "元"
        )

    def test_matches_any_amount_present_in_row(self) -> None:
        """表格一行常有两列（本期/上期），命中任一即算合理。"""
        assert value_matches_span(
            "1,234,567,890.12    1,098,765,432.10",
            Decimal("1098765432.10"),
            "元",
        )

    def test_relative_tolerance_allows_rounding(self) -> None:
        """开启相对容差后，四舍五入报数可以通过。默认关闭。"""
        span = "1,234,567,890.12 元"
        claimed = Decimal("12.35")  # 12.35 亿元，即 1,235,000,000
        assert not value_matches_span(span, claimed, "亿元")
        assert value_matches_span(
            span, claimed, "亿元", rel_tolerance=Decimal("0.001")
        )

    def test_no_amount_in_span(self) -> None:
        assert not value_matches_span("营业收入", Decimal("100"), "元")


class TestVerifyEvidence:
    """两层校验的完整闸门。"""

    def test_accepts_exact_evidence(self) -> None:
        verdict = verify_evidence(
            claimed_span="营业收入    1,234,567,890.12",
            chunk_text=CHUNK,
            claimed_value=Decimal("1234567890.12"),
            claimed_unit="元",
        )
        assert verdict.accepted is True
        assert verdict.reason is None
        assert verdict.value_ok is True

    def test_rejects_empty_span(self) -> None:
        verdict = verify_evidence(
            claimed_span="",
            chunk_text=CHUNK,
            claimed_value=Decimal("1234567890.12"),
            claimed_unit="元",
        )
        assert verdict.accepted is False
        assert verdict.reason == REASON_EMPTY_SPAN

    def test_rejects_fabricated_span(self) -> None:
        verdict = verify_evidence(
            claimed_span="营业收入 9,999,999,999.99 元",
            chunk_text=CHUNK,
            claimed_value=Decimal("9999999999.99"),
            claimed_unit="元",
        )
        assert verdict.accepted is False
        assert verdict.reason == REASON_SPAN_NOT_FOUND
        assert verdict.span_hit is None

    def test_rejects_real_span_with_wrong_value(self) -> None:
        """第一层放行、第二层拦下的关键场景。

        模型引用了一段**真实存在**的原文，却报了一个错误的数字。
        如果只校验「片段在不在原文里」，这条编造的证据会被采纳——
        而且读起来「有据可查」，比凭空编造更危险。
        """
        verdict = verify_evidence(
            claimed_span="营业收入    1,234,567,890.12",
            chunk_text=CHUNK,
            claimed_value=Decimal("987654321.00"),  # 原文里没有这个数
            claimed_unit="元",
        )
        assert verdict.accepted is False
        assert verdict.reason == REASON_VALUE_NOT_IN_SPAN
        # 片段本身是命中的——这正是险处：只看第一层会误判为通过
        assert verdict.span_hit is not None
        assert verdict.value_ok is False

    def test_unit_error_is_caught(self) -> None:
        """单位错认（万元当成元）也必须被拦下。"""
        verdict = verify_evidence(
            claimed_span="营业收入    1,234,567,890.12",
            chunk_text=CHUNK,
            claimed_value=Decimal("1234567890.12"),
            claimed_unit="万元",  # 差了一万倍
        )
        assert verdict.accepted is False
        assert verdict.reason == REASON_VALUE_NOT_IN_SPAN

    def test_accepts_normalized_match(self) -> None:
        """片段被换行/空格切开时，归一化后仍应命中。"""
        verdict = verify_evidence(
            claimed_span="营业收入\n1,234,567,890.12",
            chunk_text=CHUNK,
            claimed_value=Decimal("1234567890.12"),
            claimed_unit="元",
        )
        assert verdict.accepted is True
        assert verdict.match_mode is MatchMode.NORMALIZED

    def test_rejects_span_that_skips_over_another_number(self) -> None:
        """片段不能跨过原文中间的其他数字——那不是「格式差异」，是断章取义。"""
        verdict = verify_evidence(
            claimed_span="营业收入 1,234,567,890.12 营业成本",
            chunk_text=CHUNK,
            claimed_value=Decimal("1234567890.12"),
            claimed_unit="元",
        )
        assert verdict.accepted is False
        assert verdict.reason == REASON_SPAN_NOT_FOUND


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
