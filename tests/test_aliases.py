"""财务口语简称归一（P-030）测试。

背景
----
现金流量表里大量行名都含「现金流」，口语简称「经营现金流」在词法上同时沾到
9 行中的若干行，RRF 不保证取到「净额」那一行。实测把「支付其他与经营活动
有关的现金」（9.0B）当成「经营活动产生的现金流量净额」（92.5B）答出来 ——
数值真实、页码正确、span 通过，**全部护栏放行**。

本文件守住的正是这类「静默错误」：它比拒答危险，因为用户无从分辨。

刻意不依赖真实 PDF 与 LLM：这些是纯字符串规则，能在内存里被穷尽验证。
唯一需要真实年报的用例（`TestAgainstRealReport`）在数据缺失时**跳过而非通过**
—— 跳过会被报告出来，假装通过不会。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifin.aliases import (  # noqa: E402
    ABBREVIATION_ALIASES,
    AMBIGUOUS_ABBREVIATIONS,
    detect_ambiguous,
    expand_question,
    normalize_question,
)


class TestExpandDeterministicAbbreviations:
    """确定性别名：展开后必须得到报表里真实存在的规范名。"""

    @pytest.mark.parametrize(
        "question,expected",
        [
            ("2024年经营现金流是多少", "2024年经营活动产生的现金流量净额是多少"),
            ("2024年经营活动现金流是多少", "2024年经营活动产生的现金流量净额是多少"),
            ("2024年投资现金流净额", "2024年投资活动产生的现金流量净额"),
            ("2024年筹资现金流是多少", "2024年筹资活动产生的现金流量净额是多少"),
            ("2024年归母净利润是多少", "2024年归属于母公司股东的净利润是多少"),
        ],
    )
    def test_expands_to_canonical_name(self, question: str, expected: str) -> None:
        assert expand_question(question) == expected

    def test_question_without_abbreviation_is_unchanged(self) -> None:
        """没有简称时**必须原样返回** —— 擅自改写会污染后续的期间/口径解析。"""
        q = "2024年合并资产负债表的货币资金是多少？"
        assert expand_question(q) == q

    def test_period_and_scope_qualifiers_are_preserved(self) -> None:
        """只替换科目名，期间与口径限定词原样保留。"""
        out = expand_question("2024年母公司经营现金流是多少")
        assert "2024年" in out
        assert "母公司" in out
        assert "经营活动产生的现金流量净额" in out

    def test_each_alias_target_is_a_nonempty_string(self) -> None:
        for key, target in ABBREVIATION_ALIASES.items():
            assert key and target, f"别名表存在空键值：{key!r} -> {target!r}"

    def test_alias_and_ambiguous_tables_do_not_overlap(self) -> None:
        """同一简称不能既当确定映射又当歧义映射 —— 语义矛盾且行为取决于顺序。"""
        overlap = set(ABBREVIATION_ALIASES) & set(AMBIGUOUS_ABBREVIATIONS)
        assert not overlap, f"简称同时登记为两类，语义矛盾：{sorted(overlap)}"


class TestLongestMatchWins:
    """最长优先：短别名不得咬掉长别名的前半截。

    早先若不做最长优先，「经营活动现金流」会被「经营现金流」先匹配，
    在问句里留下「活动」这种无意义残渣，污染后续检索。
    """

    def test_longer_alias_beats_shorter_prefix(self) -> None:
        out = expand_question("2024年经营活动现金流是多少")
        assert out == "2024年经营活动产生的现金流量净额是多少"
        assert "活动产生的现金流量净额" in out
        # 若短别名抢先匹配，会残留一个孤立的「活动」
        assert "净额活动" not in out


class TestTailGuardAgainstFalseExpansion:
    """词尾保护：别名不得作为**更长科目名**的前缀被误替换。

    这是一个真实回归，值得单独守：
    第一版没有 :data:`_TAIL_GUARD`，于是问句
    「投资活动现金流入小计是多少」里的「投资活动现金流」被匹配并替换，
    产出 **「投资活动产生的现金流量净额入小计」这个根本不存在的科目名**。
    检索必然落空，而用户写的是完全正确的科目全名 —— 比原缺陷更糟，
    因为它把**正确输入**变成了失败。当时是一条既有用例红掉才暴露出来。
    """

    @pytest.mark.parametrize(
        "question",
        [
            "投资活动现金流入小计是多少",
            "投资活动现金流出小计是多少",
            "经营活动现金流入小计是多少",
            "经营活动现金流出小计是多少",
            "筹资活动现金流入小计是多少",
            "现金及现金等价物净增加额 与 投资活动现金流入小计 分别是多少",
        ],
    )
    def test_standard_subtotal_rows_are_not_rewritten(self, question: str) -> None:
        assert expand_question(question) == question

    def test_no_nonexistent_label_is_ever_produced(self) -> None:
        """替换结果里不得出现「……净额入小计 / 净额出小计」这类拼接残渣。"""
        for q in (
            "投资活动现金流入小计",
            "经营活动现金流出小计",
            "筹资活动现金流入小计",
        ):
            out = expand_question(q)
            assert "净额入" not in out
            assert "净额出" not in out


class TestTailGuardAgainstReportingTableName:
    """词尾保护的另一半：高频**报表名**「现金流量表」不得被当成歧义简称。

    这条比上面那条更隐蔽，因为它不产出错误的科目名，**只是拒答** ——
    一个完全正常的问句被拦掉，而用户根本没写任何简称。

    根因：歧义键「现金流」是「**现金流量表**」的前缀。web 端的歧义拒答一接上，
    问「2024年合并现金流量表的货币资金是多少」就会返回 `AMBIGUOUS_ABBREVIATION`，
    而「现金流量表」是这份文档里到处都在出现的词。
    判据是「加不加 `量表` 到 :data:`_TAIL_GUARD`」——不加必红。
    """

    @pytest.mark.parametrize(
        "question",
        [
            "2024年合并现金流量表的货币资金是多少？",
            "2024年合并现金流量表的净利润是多少？",
            "2024年现金流量表的资产总计是多少？",
            "2024年合并现金流量表的经营活动现金流入小计是多少？",
        ],
    )
    def test_balance_report_name_is_not_ambiguous(self, question: str) -> None:
        norm = normalize_question(question)
        assert not norm.has_ambiguous, (
            f"报表名「现金流量表」被误判成歧义简称：{question} → {norm.ambiguous}"
        )
        assert norm.expanded == question, "正常问句不应被改写"

    def test_real_abbreviation_still_detected(self) -> None:
        """反向对照：加了词尾保护**不得**把真正的歧义简称一起放过。

        若这条也绿，说明上面那个保护做过头了 —— 它会把
        「现金流是多少」这种真该拒答的问题放行。
        """
        assert normalize_question("2024年现金流是多少？").has_ambiguous


class TestCashFlowVariantWithLiangIsNotCorrupted:
    """「现金流量净额」变体：长别名先咬掉前半截会拼出不存在的科目名。

    「现金流量净额」不是「现金流净额」——中间多一个「量」。
    若只登记了短的「经营现金流」，最长优先会让它在「2024年经营现金流量净额」
    里先匹配，剩下「量净额」拼进规范名，
    得到 **「……现金流量净额量净额」这个报表里根本不存在的名字**。
    实测确认过这一条（当时逐条探测才发现，不是先想到的）。
    """

    @pytest.mark.parametrize(
        "question",
        [
            "2024年经营活动现金流量净额是多少？",
            "2024年经营现金流量净额是多少？",
            "2024年投资现金流量净额是多少？",
            "2024年筹资现金流量净额是多少？",
        ],
    )
    def test_liang_variant_expands_cleanly(self, question: str) -> None:
        out = expand_question(question)
        # 不得出现「净额量净额」这类叠字残渣
        assert "净额量" not in out, f"拼出残渣：{question} → {out}"
        assert "现金流量净额量" not in out
        # 且必须落到真实的规范科目名
        assert "活动产生的现金流量净额" in out

    def test_every_alias_target_belongs_to_the_cashflow_family(self) -> None:
        """别名表里所有「现金流」族的目标，必须含「活动产生的现金流量净额」。"""
        for key, target in ABBREVIATION_ALIASES.items():
            if "现金流" in key:
                assert target.endswith("活动产生的现金流量净额"), (
                    f"{key} → {target}：现金流族的目标名不合法"
                )


class TestAmbiguousAbbreviationsAreNotGuessed:
    """歧义简称：**刻意不映射**，交上层给出明确拒答原因。

    「营收」对应「营业收入」与「营业总收入」两行且数值不同
    （茅台 2024：170,899,152,276.34 vs 174,144,069,958.25）。
    凭语感二选一，等于用猜测换掉一次拒答。
    """

    @pytest.mark.parametrize("key", ["营收", "净资产", "现金流"])
    def test_ambiguous_key_is_left_untouched(self, key: str) -> None:
        q = f"2024年{key}是多少"
        assert expand_question(q) == q, "歧义简称被替换，等于替用户做了猜测"

    @pytest.mark.parametrize(
        "question,key",
        [
            ("2024年营收是多少", "营收"),
            ("2024年净资产是多少", "净资产"),
            ("2024年现金流是多少", "现金流"),
        ],
    )
    def test_ambiguous_key_is_detected(self, question: str, key: str) -> None:
        hit = detect_ambiguous(question)
        assert hit is not None, f"未检出歧义简称：{question}"
        assert hit[0] == key
        assert hit[1] == AMBIGUOUS_ABBREVIATIONS[key]
        assert len(hit[1]) >= 2, "歧义简称必须给出多个候选，否则不成其为歧义"

    def test_clean_question_reports_no_ambiguity(self) -> None:
        assert detect_ambiguous("2024年合并资产负债表的营业收入是多少") is None


class TestCanonicalNameIsNotSelfFlagged:
    """规范全名不得被自己的组成部分判成歧义（实测假阳性）。

    「经营活动产生的现金流量净额」含「现金流」三字。第一版把「展开」与
    「歧义检测」拆成两趟扫描，于是用户**直接写规范全名**提问时，
    第二趟把刚展开干净的结果又判成有歧义 —— 假阳性，把本来能答对的问题拦掉。
    修法是让两者在同一趟扫描里完成，且规范名本身也作为更长分支参与匹配。
    """

    @pytest.mark.parametrize(
        "question",
        [
            "2024年经营活动产生的现金流量净额是多少",
            "2024年归属于母公司股东的净利润是多少",
            "2024年所有者权益（或股东权益）合计是多少",
            "2024年投资活动产生的现金流量净额",
        ],
    )
    def test_canonical_name_has_no_ambiguity(self, question: str) -> None:
        norm = normalize_question(question)
        assert norm.expanded == question
        assert not norm.has_ambiguous, (
            f"规范全名被误判为歧义：{question} -> {norm.ambiguous}"
        )


class TestNormalizeQuestionShape:
    """单趟归一的产出结构。"""

    def test_returns_original_expanded_and_ambiguous(self) -> None:
        norm = normalize_question("2024年营收和经营现金流是多少")
        assert norm.original == "2024年营收和经营现金流是多少"
        # 确定别名被展开；歧义别名原样保留在同一句里
        assert "经营活动产生的现金流量净额" in norm.expanded
        assert "营收" in norm.expanded
        assert [k for k, _ in norm.ambiguous] == ["营收"]

    def test_multiple_ambiguous_keys_are_all_collected(self) -> None:
        norm = normalize_question("2024年营收和净资产分别是多少")
        keys = sorted(k for k, _ in norm.ambiguous)
        assert keys == ["净资产", "营收"]

    def test_empty_question_is_safe(self) -> None:
        for q in ("", None):  # type: ignore[arg-type]
            norm = normalize_question(q)  # type: ignore[arg-type]
            assert norm.expanded == (q or "")
            assert not norm.has_ambiguous


class TestAgainstRealReport:
    """需要真实索引的端到端用例：展开后检索必须落到正确的行。

    数据缺失时 **skip**，不 skip 成 pass —— 跳过会被报告出来，假装通过不会。
    """

    @staticmethod
    def _runtime():
        from verifin.runtime import build_document_runtime

        product = ROOT / "data" / "parsed" / "moutai2024_fs"
        index_db = ROOT / "data" / "index" / "moutai2024.db"
        pdf = ROOT / "data" / "pdfs" / "MOUTAI_2024_ANNUAL.pdf"
        if not (product.exists() and index_db.exists()):
            pytest.skip("缺少茅台 2024 解析产物或索引，跳过端到端用例")
        return build_document_runtime(
            doc_id="MOUTAI_2024",
            product=product,
            index_db=index_db,
            pdf=pdf if pdf.exists() else None,
            stitched=ROOT / "data" / "parsed" / "moutai2024_fs_stitched.md",
        )

    @pytest.mark.parametrize(
        "question,expected_label",
        [
            ("2024年经营现金流是多少", "经营活动产生的现金流量净额"),
            ("2024年投资现金流净额", "投资活动产生的现金流量净额"),
            ("2024年筹资现金流是多少", "筹资活动产生的现金流量净额"),
            ("2024年归母净利润是多少", "归属于母公司股东的净利润"),
        ],
    )
    def test_retrieval_top1_is_the_canonical_row(
        self, question: str, expected_label: str
    ) -> None:
        rt = self._runtime()
        result = rt.index.retrieve(question, top_k=3)
        assert not result.is_empty, f"简称展开后仍无召回：{question}"
        assert result.hits[0].chunk.label == expected_label, (
            f"「{question}」的 top-1 应为「{expected_label}」，"
            f"实际是「{result.hits[0].chunk.label}」"
        )

    def test_guard_accepts_the_expanded_answer(self) -> None:
        """护栏必须与检索层用同一条归一规则。

        两处不归一，等于亲手把修好的链路再堵上一次：检索层用展开后的问句
        取到了规范行，护栏却拿未展开的原句去比对，于是判成"没点名任何科目" → 误拒。
        """
        from verifin.guards import label_consistency

        rt = self._runtime()
        tr = rt.tool_runtime(company="x", period="y")
        ok, why = label_consistency(
            "2024年经营现金流是多少",
            "经营活动产生的现金流量净额",
            tr.known_labels,
        )
        assert ok, f"护栏误拒了简称提问：{why}"

    def test_guard_still_rejects_a_wrong_row(self) -> None:
        """归一不得把护栏放宽：取到**另一个**科目仍然必须拒答。"""
        from verifin.guards import label_consistency

        rt = self._runtime()
        tr = rt.tool_runtime(company="x", period="y")
        ok, _why = label_consistency(
            "2024年经营现金流是多少",
            "支付其他与经营活动有关的现金",
            tr.known_labels,
        )
        assert not ok, "护栏被归一放宽了 —— 取到别的科目竟然放行"