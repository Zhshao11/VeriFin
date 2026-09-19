"""评测层（D4）单元测试：题库校验、gold 自检闸门、判分口径。

三条纪律：

1. **判分口径的测试用构造出来的 `RunResult`**，不跑真实 Agent ——
   要验的是比对逻辑，不是系统能力。把两者混在一起测，失败时无法归因。
2. **gold 闸门的测试自造一份单页 PDF**，让 G3 真的被执行 ——
   G3 是三道闸门里唯一独立于本项目解析产物的，用假数据把它跳过等于没测。
3. **不依赖 `data/` 下的真实产物**（除了明确标注 `needs_product` 的跨文档一致性测试），
   合成数据即可覆盖规则本身。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifin.agent.graph import RunResult, Step  # noqa: E402
from verifin.eval import (  # noqa: E402
    DocSpec,
    EvalItem,
    bank_stats,
    check_item,
    load_bank,
    render_stats,
    save_bank,
    validate_bank,
    validate_item,
)
from verifin.eval.schema import is_verdict_formula  # noqa: E402
from verifin.eval.run_eval import compare  # noqa: E402

# 沙箱里 `tmp_path` 会抛 `PermissionError: EEXIST`（与 pip 无法解包 sdist 同源），
# 改用项目内目录；`.tmp/` 已在 `.gitignore` 里。
TMP_DIR = ROOT / ".tmp" / "eval_test"


# ---------------------------------------------------------------------------
# 构造样例
# ---------------------------------------------------------------------------


def l1_item(**over) -> EvalItem:
    base = dict(
        item_id="B-TEST-001",
        track="B",
        doc="T",
        company="某公司",
        period="2024 年度",
        question="2024年合并资产负债表的资产总计是多少？",
        question_type="L1",
        expected_outcome="ANSWER",
        scope="合并",
        gold_value="298,944,579,918.70",
        gold_unit="元",
        gold_pages=(59,),
        gold_evidence=("资产总计 | 非流动资产: | 298,944,579,918.70",),
        source="测试",
    )
    base.update(over)
    return EvalItem(**base)


def verdict_item(**over) -> EvalItem:
    base = dict(
        item_id="B-TEST-101",
        track="B",
        doc="T",
        company="某公司",
        period="2024 年度",
        question="2024年合并资产负债表的资产总计是否等于负债合计加所有者权益合计？",
        question_type="L2",
        expected_outcome="ANSWER",
        scope="合并",
        gold_verdict="PASS",
        gold_operands={
            "资产总计": "298,944,579,918.70",
            "负债合计": "56,933,264,798.10",
            "所有者权益合计": "242,011,315,120.60",
        },
        gold_pages=(59, 60, 61),
        gold_evidence=("资产总计 | 非流动资产: | 298,944,579,918.70",),
        formula="F1",
        source="测试",
    )
    base.update(over)
    return EvalItem(**base)


def refusal_item(**over) -> EvalItem:
    base = dict(
        item_id="B-TEST-201",
        track="B",
        doc="T",
        company="某公司",
        period="2024 年度",
        question="火星基地2024年的折旧年限是多少年？",
        question_type="R",
        expected_outcome="REFUSE",
        scope="不适用",
        refusal_reason="no_recall：文档中不存在该主体",
        source="测试",
    )
    base.update(over)
    return EvalItem(**base)


def run_result(
    decision: str,
    *,
    route: str = "LOOKUP",
    answer: dict | None = None,
    refusal: dict | None = None,
    steps: list[Step] | None = None,
    tool_calls: int = 0,
) -> RunResult:
    return RunResult(
        run_id="r1",
        question="q",
        decision=decision,
        answer=answer,
        refusal=refusal,
        route=route,
        steps=steps or [],
        tool_calls=tool_calls,
    )


def lookup_answer(value: str, page: int = 59) -> dict:
    return {
        "six_tuple": {
            "公司": "某公司", "期间": "2024 年度", "指标": "资产总计",
            "数值": value, "单位": "元", "来源": f"第 {page} 页",
        },
        "fragment": value,
    }


def verify_answer(verdict: str, operands: dict[str, str]) -> dict:
    return {
        "结论": verdict,
        "左值": operands.get("资产总计", "0"),
        "操作数": operands,
        "来源": [{"科目": k, "页码": 59} for k in operands],
    }


# ---------------------------------------------------------------------------
# 题库格式校验
# ---------------------------------------------------------------------------


def test_valid_l1_item_has_no_problem() -> None:
    assert validate_item(l1_item()) == []


def test_valid_refusal_item_has_no_problem() -> None:
    assert validate_item(refusal_item()) == []


def test_refusal_item_must_not_carry_a_gold_answer() -> None:
    """「本该拒答」被标成一个具体答案，拒答率的分母就失去意义。"""
    problems = validate_item(refusal_item(gold_value="1.00"))
    assert any("gold_value" in p for p in problems)


def test_refusal_item_requires_reason() -> None:
    problems = validate_item(refusal_item(refusal_reason=None))
    assert any("refusal_reason" in p for p in problems)


def test_answerable_item_may_not_be_marked_refuse() -> None:
    problems = validate_item(l1_item(expected_outcome="REFUSE"))
    assert any("应拒答的题必须归入 R 类" in p for p in problems)


def test_question_naming_parent_scope_must_match_field() -> None:
    problems = validate_item(
        l1_item(question="2024年母公司资产负债表的资产总计是多少？", scope="合并")
    )
    assert any("母公司口径" in p for p in problems)


def test_attributable_to_parent_is_not_a_scope_mention() -> None:
    """「归属于母公司股东的净利润」是**合并报表科目**，名字里带「母公司」但不是口径。

    这条是防误报的回归守卫：朴素的子串判断会把这条好题判死，
    而误报比漏报更消耗信任。
    """
    problems = validate_item(
        l1_item(
            question="2024年合并利润表中归属于母公司股东的净利润是多少？",
            scope="合并",
        )
    )
    assert problems == []


def test_question_naming_both_scopes_requires_cross_scope() -> None:
    problems = validate_item(
        l1_item(
            question="2024年母公司口径与合并口径的资产总计分别是多少？",
            scope="合并",
        )
    )
    assert any("跨口径" in p for p in problems)


def test_verdict_item_requires_operands() -> None:
    """只标判定不标操作数，会让「用母公司数据算合并恒等式」被判成对。"""
    problems = validate_item(verdict_item(gold_operands={}))
    assert any("gold_operands" in p for p in problems)


def test_verdict_item_must_not_carry_a_value() -> None:
    problems = validate_item(verdict_item(gold_value="1.00", gold_unit="元"))
    assert any("判定型题不应带 gold_value" in p for p in problems)


def test_l1_item_must_not_carry_operands() -> None:
    problems = validate_item(l1_item(gold_operands={"资产总计": "1"}))
    assert any("只有 L2 / L3" in p for p in problems)


def test_l2_item_requires_a_known_formula() -> None:
    assert any("缺 formula" in p for p in validate_item(verdict_item(formula=None)))
    assert any(
        "不在注册表里" in p for p in validate_item(verdict_item(formula="F99"))
    )


def test_verdict_formula_classification() -> None:
    """恒等式与启发式的答案是判定；派生量的答案是数值。"""

    def kind_holder(formula: str):
        return EvalItem.from_dict({**verdict_item(formula=formula).to_dict()})

    assert is_verdict_formula(kind_holder("F1")) is True
    assert is_verdict_formula(kind_holder("F6")) is True
    assert is_verdict_formula(kind_holder("F4")) is False


def test_bank_detects_duplicate_ids_and_questions() -> None:
    a = l1_item()

    # 题号重复
    report = validate_bank([a, l1_item()])
    assert any("item_id 重复" in p for p in report[a.item_id])

    # 题号不同但问句相同：同一道题被写了两遍，指标会被重复计权
    dup_q = l1_item(item_id="B-TEST-999")
    report = validate_bank([a, dup_q])
    assert any("问句重复" in p for p in report["B-TEST-999"])


# ---------------------------------------------------------------------------
# 题库读写
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_skips_comments() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    path = TMP_DIR / "bank.jsonl"
    save_bank([l1_item(), verdict_item(), refusal_item()], path)

    text = path.read_text(encoding="utf-8")
    path.write_text("# 注释行\n\n" + text, encoding="utf-8")

    items = load_bank(path)
    assert [i.item_id for i in items] == ["B-TEST-001", "B-TEST-101", "B-TEST-201"]
    assert items[1].gold_operands["负债合计"] == "56,933,264,798.10"
    assert items[2].is_refusal


def test_bank_stats_reports_composition() -> None:
    stats = bank_stats([l1_item(), verdict_item(), refusal_item()])
    assert stats["总数"] == 3
    assert stats["可答"] == 2
    assert stats["应拒答"] == 1
    assert stats["公式覆盖"] == ["F1"]
    assert "题库总数 3 题" in render_stats(stats)


# ---------------------------------------------------------------------------
# gold 自检闸门
# ---------------------------------------------------------------------------


def _make_spec(row_text: str) -> tuple[DocSpec, str]:
    """造一份单页 PDF，内容就是一行报表行文本。返回 (spec, 行文本)。"""
    import pymupdf  # PyMuPDF

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    pdf = TMP_DIR / "page59.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), row_text, fontsize=10, fontname="china-s")
    doc.save(str(pdf))
    doc.close()
    return (
        DocSpec(
            doc="T", label="测试文档", pdf=pdf, index_db=None, parse_pages=(1, 1)
        ),
        row_text,
    )


def _short_item(**over) -> EvalItem:
    """一页一行的最小可测题目。"""
    row = "资产总计 | 非流动资产: | 298,944,579,918.70"
    base = dict(
        gold_value="298,944,579,918.70",
        gold_unit="元",
        gold_pages=(1,),
        gold_evidence=(row,),
    )
    base.update(over)
    return l1_item(**base)


def test_gold_gates_pass_on_faithful_annotation() -> None:
    spec, row = _make_spec("资产总计 | 非流动资产: | 298,944,579,918.70")
    check = check_item(_short_item(), spec, index_pages={1: row})
    assert check.ok, check.problems
    assert check.checked
    assert "命中" in check.g1_index
    assert "命中" in check.g2_evidence
    assert "命中" in check.g3_pdf
    assert check.value_pages == (1,)


def test_gold_gate_g3_catches_value_not_in_original_pdf() -> None:
    """标注值与原件不符时，唯一独立于本项目解析产物的那道闸门必须拦下来。"""
    spec, row = _make_spec("资产总计 | 非流动资产: | 298,944,579,918.70")
    item = _short_item(gold_value="999,999,999,999.99", gold_evidence=(row,))
    check = check_item(item, spec, index_pages={1: row})
    # 索引里也没有这个数 → G1 与 G3 同时失败
    assert not check.ok
    assert any("G3" in p for p in check.problems)


def test_gold_gate_g1_uses_index_pages_and_is_fail_closed() -> None:
    """索引缺失时不能读成「通过」——「无法验证」不等于「已验证」。"""
    spec, _row = _make_spec("资产总计 | 非流动资产: | 298,944,579,918.70")
    check = check_item(_short_item(), spec, index_pages=None)
    assert not check.g1_index.startswith("命中")
    assert not check.ok


def test_gold_check_skips_refusal_items_by_design() -> None:
    spec, _row = _make_spec("资产总计 | 非流动资产: | 298,944,579,918.70")
    check = check_item(refusal_item(), spec, index_pages={})
    assert check.ok and check.checked


def test_gold_gate_uses_operands_when_answer_is_derived() -> None:
    """派生量的答案是算出来的，原文里没有这个串，闸门必须查操作数。"""
    spec, _row = _make_spec("营业收入 | 170,899,152,276.34")
    item = EvalItem.from_dict(
        {
            **l1_item().to_dict(),
            "item_id": "B-TEST-401",
            "question": "2024年合并口径的毛利率是多少？",
            "question_type": "L2",
            "scope": "合并",
            "gold_value": "91.93",
            "gold_unit": "%",
            "gold_verdict": None,
            "gold_operands": {"营业收入": "170,899,152,276.34"},
            "gold_pages": [1],
            "gold_evidence": ["营业收入 | 170,899,152,276.34"],
            "formula": "F4",
        }
    )
    check = check_item(item, spec, index_pages={1: "营业收入 | 170,899,152,276.34"})
    assert check.ok, check.problems
    assert "操作数" in check.g3_pdf


# ---------------------------------------------------------------------------
# 判分口径
# ---------------------------------------------------------------------------


def test_l1_correct_value_and_page_passes() -> None:
    res = compare(l1_item(), run_result("ANSWER", answer=lookup_answer("298,944,579,918.70", 59)))
    assert res.passed
    assert res.value_ok and res.page_ok


def test_l1_wrong_value_is_classified_as_value_error() -> None:
    res = compare(l1_item(), run_result("ANSWER", answer=lookup_answer("1.00", 59)))
    assert not res.passed
    assert res.failure == "数值错"


def test_l1_right_value_wrong_page_is_a_separate_failure() -> None:
    """数值对、页码错是证据链的问题，不是取数的问题，必须分开归类。"""
    res = compare(l1_item(), run_result("ANSWER", answer=lookup_answer("298,944,579,918.70", 99)))
    assert not res.passed
    assert res.failure == "页码不符"
    assert res.value_ok and not res.page_ok


def test_verdict_item_checks_operands_before_verdict() -> None:
    """**核心口径测试**：两套口径的资产恒等式都能平衡。

    只比 `PASS`，那么用母公司数据算合并恒等式也会被判成对 ——
    所以判分必须先比操作数。
    """
    item = verdict_item()
    wrong_scope = verify_answer(
        "PASS",
        {
            "资产总计": "180,236,524,477.01",
            "负债合计": "15,750,242,139.09",
            "所有者权益合计": "164,486,282,337.92",
        },
    )
    res = compare(item, run_result("ANSWER", route="VERIFY", answer=wrong_scope))
    assert not res.passed
    assert res.failure == "操作数取错"

    right = verify_answer(
        "PASS",
        {
            "资产总计": "298,944,579,918.70",
            "负债合计": "56,933,264,798.10",
            "所有者权益合计": "242,011,315,120.60",
        },
    )
    assert compare(item, run_result("ANSWER", route="VERIFY", answer=right)).passed


def test_verdict_item_detects_wrong_verdict_with_correct_operands() -> None:
    item = verdict_item()
    answer = verify_answer(
        "FAIL",
        {
            "资产总计": "298,944,579,918.70",
            "负债合计": "56,933,264,798.10",
            "所有者权益合计": "242,011,315,120.60",
        },
    )
    res = compare(item, run_result("ANSWER", route="VERIFY", answer=answer))
    assert res.failure == "判定错"


def test_refusal_expected_and_refused_passes() -> None:
    res = compare(refusal_item(), run_result("REFUSE", refusal={"reason": "NO_RECALL"}))
    assert res.passed


def test_abort_does_not_count_as_a_correct_refusal() -> None:
    """`ABORT` 是「没算完」，`REFUSE` 是「证据不足」。混为一谈会把规划器缺陷
    伪装成护栏生效。"""
    res = compare(refusal_item(), run_result("ABORT", refusal={"reason": "BUDGET_EXCEEDED"}))
    assert not res.passed
    assert res.failure == "ABORT"


def test_answering_a_refusal_question_is_a_false_answer() -> None:
    res = compare(refusal_item(), run_result("ANSWER", answer=lookup_answer("1.00")))
    assert not res.passed
    assert res.failure.startswith("误答")


def test_expected_answer_but_refused_is_a_miss() -> None:
    res = compare(l1_item(), run_result("REFUSE", refusal={"reason": "NO_RECALL"}))
    assert not res.passed
    assert res.failure == "未作答（REFUSE）"


def test_degradation_is_read_from_step_sources_not_assumed() -> None:
    """降级率必须从轨迹的 `source` 字段读出来，不能凭印象。"""
    steps = [
        Step(0, "INTENT", None, {}, True, "", "policy", None),
        Step(1, "SEARCH", "search_statement", {}, True, "", "tool", None),
        Step(2, "EVIDENCE", "get_row_evidence", {}, True, "", "llm", None),
    ]
    res = compare(
        l1_item(),
        run_result("ANSWER", answer=lookup_answer("298,944,579,918.70"), steps=steps, tool_calls=2),
    )
    assert res.steps == 3
    assert res.llm_steps == 1
    assert res.degraded  # 有步是兜底决定的


# ---------------------------------------------------------------------------
# gold 自检的命令行入口
# ---------------------------------------------------------------------------

_BANK = ROOT / "data" / "eval" / "b_moutai_2024.jsonl"
_INDEX = ROOT / "data" / "index" / "moutai2024.db"

needs_fixtures = pytest.mark.skipif(
    not (_BANK.exists() and _INDEX.exists() and (ROOT / "data" / "pdfs" / "MOUTAI_2024_ANNUAL.pdf").exists()),
    reason="缺少题库 / 索引 / 原始 PDF，跳过命令行集成测试",
)


def test_gold_check_symbols_are_reexported_lazily() -> None:
    """`from verifin.eval import check_bank` 必须成立（PEP 562 惰性再导出）。

    惰性不是为了洁癖：`gold_check.py` 自带命令行入口，
    包导入期就拉它会让 `python -m verifin.eval.gold_check` 多打一条 RuntimeWarning。
    """
    import verifin.eval as pkg

    assert pkg.summarize is not None
    assert pkg.check_bank is not None
    with pytest.raises(AttributeError):
        _ = pkg.this_name_does_not_exist


@needs_fixtures
def test_gold_check_cli_runs_and_writes_json() -> None:
    """命令行入口能在真实产物上跑通，并把结果落成机器可读 JSON。

    这是报告里「gold 自检 29/29 通过」这一句的可复现依据。
    """
    import json

    from verifin.eval.gold_check import main

    out = TMP_DIR / "gold_cli.json"
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    code = main(["--json", str(out)])
    assert code == 0, "gold 自检不应有失败项"

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["summary"]["失败"] == 0
    assert payload["summary"]["总数"] == len(payload["checks"])


@needs_fixtures
def test_gold_check_flags_a_wrong_page() -> None:
    """把 gold 页码改错，自检必须报失败 —— 否则三道闸门等于没有。

    这是对「自检是不是真在检」的守卫：如果它恒返回通过，这条会失败。
    """
    from verifin.eval.gold_check import check_bank
    from verifin.eval.docs import get_spec

    items = load_bank(_BANK)
    # 挑一条 L1 题，把 gold 页挪到一页不可能有该数值的地方。
    target = next(i for i in items if i.question_type == "L1" and i.gold_value)
    broken = EvalItem.from_dict({**target.to_dict(), "gold_pages": [1]})

    checks = {c.item_id: c for c in check_bank([broken], get_spec("MOUTAI_2024"))}
    assert broken.item_id in checks
    # 第 1 页是封面，不含报表数值；G1/G3 至少有一道必须拦下。
    assert not checks[broken.item_id].ok
