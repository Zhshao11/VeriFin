"""归一化层测试。

对应关键问题：「财务数字有千分位、全角、中文单位，怎么做精确匹配？」
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from verifin.normalize import (
    find_missing_fields,
    is_known_unit,
    normalize_text,
    parse_amounts,
    parse_single_amount,
    unit_scale,
)


class TestNormalizeText:
    """只改变形式，不改动任何字符。"""

    def test_fullwidth_digits_become_halfwidth(self) -> None:
        assert normalize_text("１２３") == "123"

    def test_fullwidth_comma_and_period_become_ascii(self) -> None:
        assert normalize_text("１，２３４．５６") == "1,234.56"

    def test_all_whitespace_removed(self) -> None:
        assert normalize_text(" 营业 收入\n1,234 ") == "营业收入1,234"

    def test_dash_variants_unified(self) -> None:
        # 破折号、连接号、减号在 PDF 抽取中形态不一，须统一成 ASCII 减号
        assert normalize_text("A\u2014B\u2013C\u2212D") == "A-B-C-D"

    def test_quotes_unified(self) -> None:
        assert normalize_text("\u300c引用\u300d") == '"引用"'

    def test_empty_input(self) -> None:
        assert normalize_text("") == ""


class TestParseAmounts:
    """金额解析：要能正确处理千分位、中文单位、会计负数写法。"""

    def test_plain_thousands_separator(self) -> None:
        (amount,) = parse_amounts("1,234.56")
        assert amount.value == Decimal("1234.56")
        assert amount.unit is None
        assert amount.unit_assumed is True

    def test_yi_unit_scales_to_100_million(self) -> None:
        (amount,) = parse_amounts("1.23亿元")
        assert amount.value == Decimal("1.23")
        assert amount.unit == "亿元"
        assert amount.normalized == Decimal("123000000.00")

    def test_wan_unit_is_not_confused_with_plain_yuan(self) -> None:
        """万元必须以 10^4 处理。这是最容易写错的地方。"""
        (amount,) = parse_amounts("1,234万元")
        assert amount.unit == "万元"
        assert amount.normalized == Decimal("12340000")

    def test_baiwan_unit_scales_to_million(self) -> None:
        """百万元是 10^6。若按后缀包含「万」判断会错成 10^4。"""
        (amount,) = parse_amounts("5百万元")
        assert amount.normalized == Decimal("5000000")

    def test_parentheses_mean_negative(self) -> None:
        """会计惯例：括号表示负数。"""
        (amount,) = parse_amounts("(1,234.56)")
        assert amount.value == Decimal("-1234.56")
        assert amount.normalized == Decimal("-1234.56")

    def test_multiple_amounts_in_one_row(self) -> None:
        amounts = parse_amounts("营业收入 100 万元 营业成本 60 万元")
        assert len(amounts) == 2
        assert amounts[0].normalized == Decimal("1000000")
        assert amounts[1].normalized == Decimal("600000")

    def test_amounts_carrying_different_units_compare_equal(self) -> None:
        """归一化的目的：等价写法必须归到同一个值。"""
        forms = ["12345.67", "12,345.67", "1.234567万元"]
        normalized = [parse_amounts(f)[0].normalized for f in forms]
        assert len(set(normalized)) == 1

    def test_percentage_has_no_unit_assumed(self) -> None:
        (amount,) = parse_amounts("12.3")
        assert amount.unit is None
        assert amount.unit_assumed is True

    def test_empty_text_yields_nothing(self) -> None:
        assert parse_amounts("") == []

    def test_known_limitation_date_yields_year_as_amount(self) -> None:
        """已知局限：日期里的年份会被当成一个金额。

        这是刻意的取舍——语义归属交给表格结构层，
        本层只做形式归一化。此测试用于固化该行为，避免被误当作 bug「修掉」。
        """
        amounts = parse_amounts("2023-12-31")
        assert Decimal("2023") in [a.value for a in amounts]


class TestParseSingleAmount:
    """有歧义时必须返回 None，不许猜。"""

    def test_unambiguous_returns_value(self) -> None:
        amount = parse_single_amount("营业收入 1,234.56 元")
        assert amount is not None
        assert amount.normalized == Decimal("1234.56")

    def test_multiple_amounts_return_none(self) -> None:
        assert parse_single_amount("本期 100 上期 90") is None

    def test_no_amount_returns_none(self) -> None:
        assert parse_single_amount("营业收入") is None


class TestDisclosureUnitScale:
    def test_wan_yuan_scale(self) -> None:
        assert unit_scale("万元") == Decimal(10) ** 4

    def test_unknown_unit_defaults_to_one(self) -> None:
        assert unit_scale("美元") == Decimal(1)

    def test_known_unit_check(self) -> None:
        assert is_known_unit("万元") is True
        assert is_known_unit("美元") is False
        assert is_known_unit(None) is False
        assert is_known_unit("  ") is False


class TestFindMissingFields:
    """六元组缺字段即拒答，因此「什么算缺失」必须钉死。"""

    @staticmethod
    def _complete(**overrides: object) -> list[str]:
        base: dict[str, object] = {
            "company": "示例公司",
            "period": "2023年度",
            "metric": "营业收入",
            "value": Decimal("1234.56"),
            "unit": "元",
            "source": object(),
        }
        base.update(overrides)
        return find_missing_fields(**base)  # type: ignore[arg-type]

    def test_complete_evidence_has_no_missing(self) -> None:
        assert self._complete() == []

    def test_zero_value_is_not_missing(self) -> None:
        """数值 0 是合法值，不能因为它「假值」就判缺失。"""
        assert self._complete(value=Decimal(0)) == []

    def test_blank_string_is_missing(self) -> None:
        assert self._complete(company="   ") == ["company"]

    def test_none_is_missing(self) -> None:
        assert self._complete(unit=None) == ["unit"]

    def test_multiple_missing_reported_in_field_order(self) -> None:
        missing = self._complete(company=None, unit="")
        assert missing == ["company", "unit"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
