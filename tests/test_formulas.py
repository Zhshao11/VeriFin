"""勾稽公式测试。

对应两个关键问题：「容差怎么定的？」「凭什么说 300% 变动就是错的？」
这两个问题的答案分别固化为 :class:`TestDeriveTolerance` 与
:class:`TestYoyReasonableness`。
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from verifin.formulas import (
    FORMULA_REGISTRY,
    FormulaKind,
    Verdict,
    check_cross_currency,
    check_gross_margin,
    check_sum_identity,
    check_yoy_reasonableness,
    derive_tolerance,
    evaluate_formula,
    suspect_operands,
)


class TestDeriveTolerance:
    """容差必须由报表披露单位推导，不能硬编码。"""

    def test_three_operands_in_wan_yuan(self) -> None:
        # 每个科目最多偏 0.5 万元，三个科目合计最多 1.5 万元
        assert derive_tolerance(3, "万元") == Decimal("15000.0")

    def test_single_operand_in_yuan(self) -> None:
        assert derive_tolerance(1, "元") == Decimal("0.5")

    def test_unknown_unit_falls_back_to_yuan(self) -> None:
        assert derive_tolerance(2, None) == Decimal("1.0")

    def test_more_operands_means_looser_tolerance(self) -> None:
        # F3a 涉及 5 个科目，累积误差必然比 3 个科目的 F1 大
        assert derive_tolerance(5, "万元") > derive_tolerance(3, "万元")

    def test_rejects_non_positive_operand_count(self) -> None:
        with pytest.raises(ValueError):
            derive_tolerance(0, "元")


class TestSumIdentity:
    """形如「左值 = Σ(符号 × 右项)」的恒等式。"""

    #: 资产 10 亿 = 负债 4 亿 + 权益 6 亿
    BALANCED = {
        "资产总计": Decimal("1000000000"),
        "负债合计": Decimal("400000000"),
        "所有者权益合计": Decimal("600000000"),
    }

    def test_balanced_passes(self) -> None:
        outcome = check_sum_identity(
            FORMULA_REGISTRY["F1"], self.BALANCED, disclosure_unit="元"
        )
        assert outcome.verdict is Verdict.PASS
        assert outcome.diff == 0

    def test_small_rounding_difference_still_passes(self) -> None:
        """以万元披露时，5000 元的差额属于舍入误差，不应报错。"""
        operands = {**self.BALANCED, "所有者权益合计": Decimal("600005000")}
        outcome = check_sum_identity(
            FORMULA_REGISTRY["F1"], operands, disclosure_unit="万元"
        )
        assert outcome.verdict is Verdict.PASS
        assert outcome.tolerance == Decimal("15000.0")

    def test_same_difference_fails_when_unit_is_yuan(self) -> None:
        """同样的 5000 元差额，以元披露时就是真实的不平衡。

        这个对照说明「容差由披露单位推导」不是形式主义：
        同一个数字，单位不同，结论必须不同。
        """
        operands = {**self.BALANCED, "所有者权益合计": Decimal("600005000")}
        outcome = check_sum_identity(
            FORMULA_REGISTRY["F1"], operands, disclosure_unit="元"
        )
        assert outcome.verdict is Verdict.FAIL

    def test_negative_sign_in_rhs_is_honoured(self) -> None:
        """净利润 = 利润总额 − 所得税费用。符号写错会直接导致误报。"""
        outcome = check_sum_identity(
            FORMULA_REGISTRY["F2a"],
            {
                "净利润": Decimal("750000000"),
                "利润总额": Decimal("1000000000"),
                "所得税费用": Decimal("250000000"),
            },
            disclosure_unit="元",
        )
        assert outcome.verdict is Verdict.PASS

    def test_missing_operand_yields_warn_not_fail(self) -> None:
        """「无法核验」与「核验不通过」必须分开。"""
        outcome = check_sum_identity(
            FORMULA_REGISTRY["F1"],
            {"资产总计": Decimal("1000"), "负债合计": Decimal("400")},
            disclosure_unit="元",
        )
        assert outcome.verdict is Verdict.WARN
        assert outcome.missing_operands == ("所有者权益合计",)
        assert outcome.diff is None

    def test_fail_detail_carries_common_pitfall(self) -> None:
        """不平衡时要给出领域线索，而不只是报一个差额。"""
        outcome = check_sum_identity(
            FORMULA_REGISTRY["F1"],
            {
                "资产总计": Decimal("1000000000"),
                "负债合计": Decimal("400000000"),
                "所有者权益合计": Decimal("550000000"),
            },
            disclosure_unit="元",
        )
        assert outcome.verdict is Verdict.FAIL
        assert "少数股东权益" in outcome.detail


class TestSuspectOperands:
    def test_no_suspect_when_no_operand_explains_the_gap(self) -> None:
        """没有任何科目量级能解释差额时，应如实返回空，而不是随便挑一个。"""
        suspects = suspect_operands(
            {
                "资产总计": Decimal("1000000000"),
                "负债合计": Decimal("400000000"),
                "所有者权益合计": Decimal("550000000"),
            },
            Decimal("50000000"),
        )
        assert suspects == ()

    def test_detects_omitted_line_by_magnitude(self) -> None:
        """差额恰好等于某个科目的量级 —— 该科目可能被整体漏记。"""
        suspects = suspect_operands(
            {
                "资产总计": Decimal("1000000000"),
                "负债合计": Decimal("400000000"),
                "所有者权益合计": Decimal("600000000"),
            },
            Decimal("600000000"),
        )
        assert suspects[0] == "所有者权益合计"

    def test_detects_decimal_shift(self) -> None:
        """差额约为某科目量级的 9 倍 —— 疑似万/亿单位错认。"""
        suspects = suspect_operands(
            {"资产总计": Decimal("1000"), "负债合计": Decimal("100")},
            Decimal("900"),
        )
        assert suspects[0] == "负债合计"

    def test_zero_diff_yields_nothing(self) -> None:
        assert suspect_operands({"A": Decimal("1")}, Decimal(0)) == ()


class TestGrossMargin:
    def test_computes_expected_margin(self) -> None:
        outcome = check_gross_margin(
            revenue=Decimal("1000"), cost=Decimal("600")
        )
        assert outcome.verdict is Verdict.PASS
        assert outcome.lhs == Decimal("40.00")

    def test_matching_claim_passes(self) -> None:
        outcome = check_gross_margin(
            revenue=Decimal("1000"), cost=Decimal("600"), claimed_pct=Decimal("40")
        )
        assert outcome.verdict is Verdict.PASS

    def test_mismatched_claim_fails(self) -> None:
        outcome = check_gross_margin(
            revenue=Decimal("1000"), cost=Decimal("600"), claimed_pct=Decimal("45")
        )
        assert outcome.verdict is Verdict.FAIL
        assert outcome.diff == Decimal("-5.00")

    def test_zero_revenue_yields_warn(self) -> None:
        outcome = check_gross_margin(revenue=Decimal(0), cost=Decimal("600"))
        assert outcome.verdict is Verdict.WARN


class TestCrossCurrency:
    def test_consistent_conversion_passes(self) -> None:
        outcome = check_cross_currency(
            foreign_amount=Decimal("100"),
            local_amount=Decimal("700"),
            rate=Decimal("7"),
        )
        assert outcome.verdict is Verdict.PASS

    def test_inconsistent_conversion_fails(self) -> None:
        outcome = check_cross_currency(
            foreign_amount=Decimal("100"),
            local_amount=Decimal("800"),
            rate=Decimal("7"),
        )
        assert outcome.verdict is Verdict.FAIL


class TestYoyReasonableness:
    """启发式检查：永远只给 PASS 或 WARN，绝不 FAIL。"""

    def test_normal_change_passes(self) -> None:
        outcome = check_yoy_reasonableness(
            current=Decimal("112"), prior=Decimal("100")
        )
        assert outcome.verdict is Verdict.PASS
        assert outcome.diff == Decimal("12.00")

    def test_huge_change_warns_but_never_fails(self) -> None:
        """变动 300% 可能是并购重组造成的真实变化，判成 FAIL 就是误报。"""
        outcome = check_yoy_reasonableness(
            current=Decimal("500"), prior=Decimal("100")
        )
        assert outcome.verdict is Verdict.WARN
        assert outcome.kind is FormulaKind.HEURISTIC

    def test_sign_flip_warns(self) -> None:
        outcome = check_yoy_reasonableness(
            current=Decimal("-50"), prior=Decimal("100")
        )
        assert outcome.verdict is Verdict.WARN
        assert "翻转" in outcome.detail

    def test_non_positive_base_warns_instead_of_computing(self) -> None:
        """基期为负时百分比变化率没有财务意义，不能硬算一个数字出来。"""
        outcome = check_yoy_reasonableness(
            current=Decimal("50"), prior=Decimal("-100")
        )
        assert outcome.verdict is Verdict.WARN
        assert outcome.diff is None


class TestEvaluateFormula:
    """统一入口的调用约定。"""

    def test_dispatches_sum_identity(self) -> None:
        outcome = evaluate_formula(
            "F2b",
            {
                "归属于母公司股东的净利润": Decimal("700"),
                "净利润": Decimal("1000"),
                "少数股东损益": Decimal("300"),
            },
            disclosure_unit="元",
        )
        assert outcome.verdict is Verdict.PASS

    def test_translates_chinese_operand_names_for_evaluator(self) -> None:
        """F4 走自定义求值器，调用方仍只用中文科目名。"""
        outcome = evaluate_formula(
            "F4",
            {"营业收入": Decimal("1000"), "营业成本": Decimal("600")},
            claimed_pct=Decimal("40"),
        )
        assert outcome.verdict is Verdict.PASS

    def test_translates_for_cross_currency(self) -> None:
        outcome = evaluate_formula(
            "F5",
            {
                "外币金额": Decimal("100"),
                "本币列示金额": Decimal("700"),
                "折算汇率": Decimal("7"),
            },
        )
        assert outcome.verdict is Verdict.PASS

    def test_missing_operand_for_evaluator_formula_yields_warn(self) -> None:
        outcome = evaluate_formula("F4", {"营业收入": Decimal("1000")})
        assert outcome.verdict is Verdict.WARN
        assert outcome.missing_operands == ("营业成本",)

    def test_float_operand_is_rejected(self) -> None:
        """浮点污染是红线，宁可显式报错。"""
        with pytest.raises(TypeError, match="Decimal"):
            evaluate_formula("F1", {"资产总计": 1000.0})  # type: ignore[dict-item]

    def test_unknown_formula_id_raises(self) -> None:
        with pytest.raises(KeyError):
            evaluate_formula("F99", {})


class TestRegistry:
    def test_required_formulas_are_registered(self) -> None:
        """需求里点名的 5 类核验都必须存在。"""
        for fid in ("F1", "F2a", "F2b", "F3a", "F3b", "F4", "F5", "F6"):
            assert fid in FORMULA_REGISTRY

    def test_only_heuristic_formula_is_f6(self) -> None:
        """除 F6 外不得有启发式公式，否则 WARN/FAIL 的边界会被悄悄放宽。"""
        heuristics = [
            f.id
            for f in FORMULA_REGISTRY.values()
            if f.kind is FormulaKind.HEURISTIC
        ]
        assert heuristics == ["F6"]

    def test_identity_formulas_declare_pitfalls(self) -> None:
        """恒等式不平是 FAIL，必须能给出成因线索。"""
        for formula in FORMULA_REGISTRY.values():
            if formula.kind is FormulaKind.IDENTITY:
                assert formula.common_pitfalls, f"{formula.id} 缺少常见成因"

    def test_omitted_line_hint_when_no_operand_explains_the_gap(self) -> None:
        """漏项不在操作数里时，靠公式自带的领域线索给出核查方向。

        这是 ``suspect_operands`` 的能力边界：它只能从现有操作数的量级
        去猜，猜不到「本次根本没被抽取的项目」。所以每个恒等式还要声明
        自己常见的漏项，供人工核查。
        """
        outcome = check_sum_identity(
            FORMULA_REGISTRY["F1"],
            {
                "资产总计": Decimal("4000000000"),
                "负债合计": Decimal("2000000000"),
                "所有者权益合计": Decimal("1700000000"),  # 漏加了少数股东权益
            },
            disclosure_unit="元",
        )
        assert outcome.verdict is Verdict.FAIL
        assert outcome.suspects == ()
        assert "少数股东权益" in outcome.detail

    def test_fail_wording_is_not_repeated(self) -> None:
        """成因提示只应出现一次前缀，避免「常见成因：常见成因：」这类拼接错误。"""
        outcome = check_sum_identity(
            FORMULA_REGISTRY["F1"],
            {
                "资产总计": Decimal("4000000000"),
                "负债合计": Decimal("2000000000"),
                "所有者权益合计": Decimal("1700000000"),
            },
            disclosure_unit="元",
        )
        assert outcome.detail.count("常见成因：") == 1


class TestRequiredOperands:
    """`required_operands`：哪些科目必须**从报表里取到**。

    这一层的存在理由（D4 建题库时暴露）：`operand_names` 把左值也算进来，
    而派生量的左值是我们要**求出来**的东西（F4 的「毛利率」），
    报表上根本没有这一行。照 `operand_names` 取数，毛利率永远算不出来，
    报错还说「缺科目：毛利率」—— 把「报表没有派生量行」说成了「数据缺失」。
    """

    def test_identity_formula_needs_every_row(self) -> None:
        assert FORMULA_REGISTRY["F1"].required_operands == (
            "资产总计",
            "负债合计",
            "所有者权益合计",
        )

    def test_derived_formula_excludes_the_derived_lhs(self) -> None:
        required = FORMULA_REGISTRY["F4"].required_operands
        assert required == ("营业收入", "营业成本")
        assert "毛利率" not in required
        assert "毛利率" in FORMULA_REGISTRY["F4"].operand_names

    def test_gross_margin_computable_from_table_rows(self) -> None:
        """端到端：只用报表上真实存在的两行，就能算出毛利率。"""
        outcome = evaluate_formula(
            "F4",
            {
                "营业收入": Decimal("170899152276.34"),
                "营业成本": Decimal("13789482367.98"),
            },
            disclosure_unit="元",
        )
        assert outcome.verdict is Verdict.PASS
        assert str(outcome.lhs) == "91.93"

    def test_extra_operands_do_not_blow_up_the_evaluator(self) -> None:
        """调用方多带一个键不得抛 `KeyError`。

        早先 `evaluate_formula` 对 `operands` 里每个键都做 `param_map[name]`，
        多一个键就抛异常 —— 把「多给了个操作数」炸成未捕获异常。
        """
        outcome = evaluate_formula(
            "F4",
            {
                "毛利率": Decimal("0"),  # 非 param_map 里的键，应被忽略
                "营业收入": Decimal("1000"),
                "营业成本": Decimal("400"),
            },
            disclosure_unit="元",
        )
        assert outcome.verdict is Verdict.PASS
        assert str(outcome.lhs) == "60.00"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
