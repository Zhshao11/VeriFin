"""检索层与财务词典的单元测试。

合成数据用于验证规则本身（快、与 PDF 无关）；
真实年报用于验证规则在实际产物上确实成立（含 P-003 的回归守卫）。
"""

from __future__ import annotations

import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest

from verifin.lexicon import clean_label
from verifin.retrieval import (
    DEFAULT_RRF_K,
    Chunk,
    build_chunks,
    create_index,
    load_index,
    rrf_fuse,
    tokenize,
)
from verifin.tables import stitch_file

PRODUCT = Path("data/parsed/moutai2024_fs")
needs_product = pytest.mark.skipif(
    not PRODUCT.exists(), reason="缺少真实解析产物，跳过"
)


def make_chunk(cid: str, label: str, values: tuple[str, ...], page: int = 1) -> Chunk:
    return Chunk(
        chunk_id=cid, doc_id="T", table_index=0, page=page, label=label,
        section=None, values=values, text=" | ".join((label, *values)),
    )


#: 测试索引落地目录。
#: 不用 pytest 的 `tmp_path` —— 本环境的沙箱 shim 在创建临时目录时会抛
#: `PermissionError: EEXIST`（与 pip 无法解包 sdist 同源），改用项目内目录绕行。
#: `.tmp/` 已加入 `.gitignore`。
TEST_INDEX_DIR = Path(__file__).resolve().parent.parent / ".tmp" / "test_index"
_counter = 0


def build_test_index(chunks: list[Chunk]):
    global _counter
    _counter += 1
    TEST_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    db = TEST_INDEX_DIR / f"idx_{_counter:03d}.db"
    con = create_index(db, chunks)
    con.close()
    return load_index(db)


# --------------------------------------------------------------------------
# 财务词典与清洗
# --------------------------------------------------------------------------


def test_lexicon_keeps_financial_terms_whole() -> None:
    """jieba 默认词典会把「营业收入」切成 营业+收入，注入词典后必须是整词。

    这是「2024年营业收入是多少」能被正确召回的前提。
    """
    assert tokenize("营业收入") == ["营业收入"]
    assert tokenize("营业总收入")[0] == "营业总收入"


def test_lexicon_does_not_break_general_text() -> None:
    """注入词典不能影响通用文本：只丢标点与空白，词元本身保留。"""
    tokens = tokenize("资产的构成")
    assert "资产" in tokens and "构成" in tokens
    assert "" not in tokens


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("一、营业总收入", "营业总收入"),
        ("其中:营业收入", "营业收入"),
        ("加:营业外收入", "营业外收入"),
        ("减:营业外支出", "营业外支出"),
        ("货币资金", "货币资金"),
    ],
)
def test_clean_label_strips_presentation_prefixes(raw: str, expected: str) -> None:
    """序号与归属标记不携带财务语义，留着会让精确匹配通道无故失配。"""
    assert clean_label(raw) == expected


def test_clean_label_strips_footnote_suffix() -> None:
    """「（净亏损以"-"号填列）」是填列说明，不是指标名。"""
    assert clean_label('净利润（净亏损以"-"号填列）') == "净利润"


def test_clean_label_preserves_meaningful_parenthesis() -> None:
    """反例守卫：`所有者权益（或股东权益）合计` 的括号是名字的一部分，不能被摘。"""
    assert clean_label("所有者权益（或股东权益）合计") == "所有者权益（或股东权益）合计"


# --------------------------------------------------------------------------
# 数值列推导
# --------------------------------------------------------------------------


@needs_product
def test_build_chunks_excludes_note_number_column() -> None:
    """「附注」列是编号不是金额。

    若把它当数值，会得到「未分配利润 = 39」这种荒谬结果——
    而且它能通过 span 校验（39 确实在原文里）一路进入六元组。
    只能靠读表头结构性地防。
    """
    report = stitch_file(PRODUCT)
    chunks = build_chunks(report.tables, "MOUTAI_2024")
    target = next(c for c in chunks if c.label == "未分配利润")
    assert "39" not in target.values, f"附注编号混进了数值：{target.values}"
    assert target.values[0] == "182,787,415,205.05"


# --------------------------------------------------------------------------
# 通道 A：词法（P-003 回归守卫）
# --------------------------------------------------------------------------


@needs_product
def test_lexical_search_finds_multi_char_chinese_term() -> None:
    """P-003 的回归守卫：多字中文词必须能被检索到。

    若查询侧不做 jieba 切词，`MATCH '资产总计'` 会静默 0 命中。
    """
    report = stitch_file(PRODUCT)
    idx = build_test_index(build_chunks(report.tables, "MOUTAI_2024"))
    try:
        assert idx.lexical_search("资产总计"), "词法通道对多字中文词 0 命中（P-003 复发）"
        assert idx.lexical_search("负债合计")
    finally:
        idx.close()


def test_lexical_search_excludes_pure_numeric_queries() -> None:
    """纯数字查询交给数值通道：`298,944,579,918.70` 进 BM25 会被切成一堆碎片。"""
    idx = build_test_index([
        make_chunk("a", "资产总计", ("298,944,579,918.70",)),
        make_chunk("b", "存货", ("54,343,285,157.47",)),
    ])
    try:
        assert idx.lexical_search("298,944,579,918.70") == []
    finally:
        idx.close()


def test_lexical_search_handles_natural_language_question() -> None:
    """自然语言提问不能 0 命中——AND 语义在这里必然失败，故用 OR + BM25 排序。"""
    idx = build_test_index([
        make_chunk("a", "营业收入", ("170,899,152,276.34",)),
        make_chunk("b", "营业外收入", ("68,500,621.72",)),
        make_chunk("c", "手续费及佣金收入", ("1,000.00",)),
    ])
    try:
        hits = idx.lexical_search("2024年营业收入是多少")
        assert hits, "自然语言提问 0 命中"
        assert hits[0] == "a", f"营业收入 应排第一，实际 {hits}"
    finally:
        idx.close()


# --------------------------------------------------------------------------
# 通道 B：数值精确
# --------------------------------------------------------------------------


def test_numeric_search_normalizes_separators() -> None:
    """千分位与全角逗号必须被归一化后再比较。"""
    idx = build_test_index([
        make_chunk("a", "资产总计", ("298,944,579,918.70",)),
        make_chunk("b", "负债合计", ("56,933,264,798.10",)),
    ])
    try:
        assert idx.numeric_search("298944579918.70") == ["a"]
        assert idx.numeric_search("298,944,579,918.70") == ["a"]
    finally:
        idx.close()


def test_numeric_search_is_exact_not_approximate() -> None:
    """刻意不做数值近似：近似会让相近金额互相召回，导致用 A 行的数核验 B 行。"""
    idx = build_test_index([
        make_chunk("a", "资产总计", ("298,944,579,918.70",)),
        make_chunk("b", "负债合计", ("298,944,579,918.71",)),
    ])
    try:
        assert idx.numeric_search("298,944,579,918.70") == ["a"]
    finally:
        idx.close()


# --------------------------------------------------------------------------
# 通道 C：精确科目名
# --------------------------------------------------------------------------


def test_label_search_prefers_exact_over_substring() -> None:
    """BM25 不区分完整匹配与子串匹配，`负债合计` 必须压过 `非流动负债合计`。"""
    idx = build_test_index([
        make_chunk("a", "非流动负债合计", ("417,274,179.14",)),
        make_chunk("b", "流动负债合计", ("56,515,990,618.96",)),
        make_chunk("c", "负债合计", ("56,933,264,798.10",)),
    ])
    try:
        assert idx.label_search("负债合计") == ["c"]
        result = idx.retrieve("负债合计", top_k=3)
        assert result.hits[0].chunk.label == "负债合计"
    finally:
        idx.close()


# --------------------------------------------------------------------------
# 融合
# --------------------------------------------------------------------------


def test_rrf_fuse_formula() -> None:
    ranked = {"a": ["x", "y"], "b": ["y"]}
    fused = rrf_fuse(ranked, k=DEFAULT_RRF_K)
    scores = {cid: score for cid, score, _, _ in fused}
    # x 只在 a 通道第 1 名；y 在 a 通道第 2 名 + b 通道第 1 名
    assert scores["y"] > scores["x"]
    assert scores["x"] == pytest.approx(1.0 / (DEFAULT_RRF_K + 1))
    assert scores["y"] == pytest.approx(
        1.0 / (DEFAULT_RRF_K + 2) + 1.0 / (DEFAULT_RRF_K + 1)
    )


def test_rrf_records_which_channels_hit() -> None:
    """保留 via 是为了溯源：融合后如果黑盒化，就解释不了「为什么排在前面」。"""
    idx = build_test_index([
        make_chunk("a", "资产总计", ("298,944,579,918.70",)),
    ])
    try:
        hit = idx.retrieve("资产总计").hits[0]
        assert "label" in hit.via
        assert "lexical" in hit.via
        assert set(hit.ranks) >= {"label", "lexical"}
    finally:
        idx.close()


def test_retrieve_returns_empty_without_crashing() -> None:
    idx = build_test_index([make_chunk("a", "资产总计", ("1.00",))])
    try:
        res = idx.retrieve("这个查询串在库里什么也匹配不到")
        assert res.is_empty is True
    finally:
        idx.close()


@needs_product
def test_end_to_end_retrieval_on_real_annual_report() -> None:
    """端到端：真实年报上，四类查询都要能把目标行排到第一。"""
    report = stitch_file(PRODUCT)
    idx = build_test_index(build_chunks(report.tables, "MOUTAI_2024"))
    try:
        cases = {
            "负债合计": "负债合计",
            "营业总收入": "营业总收入",
            "2024年营业收入是多少": "营业收入",
            "242,011,315,120.60": "所有者权益（或股东权益）合计",
        }
        for query, expected_label in cases.items():
            res = idx.retrieve(query, top_k=3)
            assert not res.is_empty, f"{query!r} 空结果"
            assert res.hits[0].chunk.label == expected_label, (
                f"{query!r} 首位应为 {expected_label}，实际 {res.hits[0].chunk.label}"
            )
    finally:
        idx.close()


# --------------------------------------------------------------------------
# 科目名通道：整段出现（D4 建题库时补上）
# --------------------------------------------------------------------------


def test_label_channel_matches_account_name_inside_a_question() -> None:
    """真实问句里科目名是**整段出现**的，不是整串相等。

    原先该通道只做整串相等，于是只对「用户只打一个科目名」有效；
    问句一旦成句，这一路等于关闭 —— 实测表现为「一个又长又错的科目
    靠 TF-IDF 相似度排到第一位」（题面写着货币资金，答的却是别的科目）。
    """
    idx = build_test_index(
        [
            make_chunk("a", "负债合计", ("100.00",), page=60),
            make_chunk("b", "非流动负债合计", ("20.00",), page=60),
        ]
    )
    try:
        res = idx.retrieve("2024年合并资产负债表的负债合计是多少？", top_k=2)
        assert not res.is_empty
        assert res.hits[0].chunk.label == "负债合计"
        assert "label" in res.hits[0].via
    finally:
        idx.close()


def test_label_channel_prefers_the_longest_contained_label() -> None:
    """问句里同时出现短科目名与长科目名时，用户指的是长的那一个。"""
    idx = build_test_index(
        [
            make_chunk("a", "其他", ("1.00",), page=1),
            make_chunk("b", "其他应收款", ("2.00",), page=1),
        ]
    )
    try:
        res = idx.retrieve("其他应收款是多少？", top_k=2)
        assert res.hits[0].chunk.label == "其他应收款"
    finally:
        idx.close()


# --------------------------------------------------------------------------
# 通道强弱：向量只重排，不产生召回
# --------------------------------------------------------------------------


def test_tokenize_drops_function_words() -> None:
    """功能词必须在切词阶段丢掉：留在向量里会制造出「似曾相识」的相似度。"""
    assert "的" not in tokenize("资产的构成")
    assert "多少" not in tokenize("营业收入是多少")


def test_stopword_only_query_cannot_produce_recall() -> None:
    """回归守卫：问句里只剩功能词时，不得产生任何召回。

    实测事故：问句「火星基地2024年的折旧年限是多少年？」
    里只有「的」在词表内，查询向量退化成「的」这一个方向的单位向量，
    与库中含「的」最多的那一行余弦相似度高达 0.40 —— **比正确问句（0.26）还高**。
    于是四路都"有召回"，`NO_RECALL` 拒答护栏被架空，无关问题也会拿到一个数。
    """
    idx = build_test_index(
        [make_chunk("a", "偿还债务支付的现金", ("1.00",), page=67)]
    )
    try:
        assert idx.retrieve("火星基地2024年的折旧年限是多少年？").is_empty
    finally:
        idx.close()


def test_vector_channel_alone_cannot_produce_recall() -> None:
    """弱通道只做重排：只开向量通道时，召回必须为空。

    这是一条**结构性保证**，不是当前数据的巧合：本地向量是 TF-IDF，
    对没有共同词的问句也会给正分。将来若换成真 embedding，
    相似度只会更"平滑"，更需要这条规则兜住拒答护栏。
    """
    idx = build_test_index([make_chunk("a", "资产总计", ("298,944,579,918.70",))])
    try:
        res = idx.retrieve("资产总计", channels=("vector",))
        assert res.per_channel["vector"] >= 1, "向量通道本身应当命中，否则这条测试没意义"
        assert res.is_empty
    finally:
        idx.close()


def test_numeric_channel_is_strong() -> None:
    """纯数字查询不靠词法，但数字精确相等是硬匹配，属于强通道。"""
    idx = build_test_index([make_chunk("a", "资产总计", ("298,944,579,918.70",))])
    try:
        res = idx.retrieve("298,944,579,918.70")
        assert not res.is_empty
        assert res.hits[0].chunk.label == "资产总计"
    finally:
        idx.close()
