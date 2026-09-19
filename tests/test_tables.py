"""跨页表格拼接的单元测试。

命名约定：`test_<行为>_<预期>`，断言写明"为什么"，便于以后改规则时知道哪条约定被破坏。
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from fin_verify.tables import (
    Block,
    BlockRow,
    Cell,
    _is_header_row,
    _looks_like_amount,
    parse_blocks,
    stitch,
    stitch_file,
)

REAL_PRODUCT = Path("data/parsed/moutai2024_fs")


# --------------------------------------------------------------------------
# 单元格级：金额判定必须排除「附注」列的小整数
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "59,295,822,956.89",
        "248,513,280.00",
        "1,210,959,803.42",
        "298944579918.70",
        "-1,234.56",
        "12345",          # 5 位以上无分隔符也算（金额可能是整数）
    ],
)
def test_looks_like_amount_accepts_real_amounts(text: str) -> None:
    assert _looks_like_amount(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "项目",
        "附注",
        "货币资金",
        "1",              # 附注编号，绝不能当成金额
        "39",
        "2024年12月31日",
        "流动资产:",
    ],
)
def test_looks_like_amount_rejects_non_amounts(text: str) -> None:
    """附注列的 `1` / `39` 若被当成金额，会污染表头判定与后续数值提取。"""
    assert _looks_like_amount(text) is False


# --------------------------------------------------------------------------
# 表头识别
# --------------------------------------------------------------------------


def test_header_row_recognized_by_project_keyword() -> None:
    row = BlockRow(
        cells=(
            Cell("项目"),
            Cell("附注"),
            Cell("2024年12月31日"),
            Cell("2023年12月31日"),
        ),
        page=58,
    )
    assert _is_header_row(row) is True


def test_header_row_recognized_by_dates_without_amounts() -> None:
    row = BlockRow(
        cells=(Cell(""), Cell("本期"), Cell("2024年12月31日"), Cell("2023年12月31日")),
        page=58,
    )
    assert _is_header_row(row) is True


def test_data_row_is_not_header() -> None:
    """数据行里也含日期以外的数字，绝不能被误判成表头。"""
    row = BlockRow(
        cells=(
            Cell("货币资金"),
            Cell("1"),
            Cell("59,295,822,956.89"),
            Cell("69,070,136,376.12"),
        ),
        page=58,
    )
    assert _is_header_row(row) is False


def test_continuation_first_row_is_not_header() -> None:
    """续页首行是数据行——这正是"该块属于上一张表"的信号。"""
    row = BlockRow(
        cells=(
            Cell("拆出资金"),
            Cell("2"),
            Cell("127,187,293,298.17"),
            Cell("105,553,836,462.58"),
        ),
        page=59,
    )
    assert _is_header_row(row) is False


# --------------------------------------------------------------------------
# 解析：两种序列化统一
# --------------------------------------------------------------------------


def test_parse_html_table_keeps_colspan_and_empty_cells() -> None:
    text = (
        "<!-- page 1 of 2 -->\n"
        "<table><tbody>"
        '<tr><td>项目</td><td>附注</td><td>2024年12月31日</td><td>2023年12月31日</td></tr>'
        '<tr><td colspan="4">流动资产:</td></tr>'
        "<tr><td>货币资金</td><td>1</td><td>59,295,822,956.89</td><td>69,070,136,376.12</td></tr>"
        "<tr><td>结算备付金</td><td></td><td></td><td></td></tr>"
        "</tbody></table>\n"
        "<!-- page 2 of 2 -->\n"
    )
    _, blocks = parse_blocks(text)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.serialization == "html"
    assert len(block.rows) == 4
    # 空单元格保留为 ""，不是 None —— 「空」与「没有这一列」必须可区分
    assert block.rows[3].cells[2].text == ""
    assert block.rows[1].cells[0].colspan == 4
    assert block.rows[1].is_section is True


def test_parse_markdown_table_skips_separator_row() -> None:
    text = (
        "<!-- page 1 of 1 -->\n"
        "| 未分配利润 | 39 | 182,787,415,205.05 | 172,983,178,300.09 |\n"
        "| --- | --- | --- | --- |\n"
        "| 少数股东权益 |  | 8,905,330,721.13 | 7,987,897,687.39 |\n"
    )
    _, blocks = parse_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].serialization == "markdown"
    # 分隔行被跳过，只剩 2 条数据行
    assert len(blocks[0].rows) == 2


def test_parse_mixed_serialization_in_one_document() -> None:
    """MinerU 会在同一份文档里混用两种序列化（已实测），必须都能解析。"""
    text = (
        "<!-- page 1 of 2 -->\n"
        "<table><tbody>"
        "<tr><td>项目</td><td>附注</td><td>2024年12月31日</td><td>2023年12月31日</td></tr>"
        "<tr><td>未分配利润</td><td>39</td><td>182,787,415,205.05</td><td>172,983,178,300.09</td></tr>"
        "</tbody></table>\n"
        "<!-- page 2 of 2 -->\n"
        "| 所有者权益合计 |  | 242,011,315,120.60 | 223,656,469,294.82 |\n"
    )
    report = stitch(*reversed(parse_blocks(text)))
    serializations = {b.serialization for b in report.blocks}
    assert serializations == {"html", "markdown"}
    # 两种序列化仍然被拼成同一张逻辑表
    assert report.table_count == 1
    assert len(report.tables[0].rows) == 2


# --------------------------------------------------------------------------
# 拼接：核心行为
# --------------------------------------------------------------------------


def _doc(page1_body: str, page2_body: str) -> str:
    return (
        f"<!-- page 58 of 143 -->\n{page1_body}\n"
        f"<!-- page 59 of 143 -->\n{page2_body}\n"
    )


HEADER_P1 = (
    "<table><tbody>"
    "<tr><td>项目</td><td>附注</td><td>2024年12月31日</td><td>2023年12月31日</td></tr>"
    "<tr><td>货币资金</td><td>1</td><td>59,295,822,956.89</td><td>69,070,136,376.12</td></tr>"
    "</tbody></table>"
)
CONT_P2 = (
    "<table><tbody>"
    "<tr><td>拆出资金</td><td>2</td><td>127,187,293,298.17</td><td>105,553,836,462.58</td></tr>"
    "<tr><td>资产总计</td><td></td><td>298,944,579,918.70</td><td>272,699,660,092.25</td></tr>"
    "</tbody></table>"
)


def test_stitch_propagates_header_to_continuation_page() -> None:
    report = stitch(*reversed(parse_blocks(_doc(HEADER_P1, CONT_P2))))
    assert report.table_count == 1
    table = report.tables[0]
    assert table.header == ("项目", "附注", "2024年12月31日", "2023年12月31日")
    assert table.page_span == "p58-p59"
    assert table.header_is_propagated is True
    assert [r.label for r in table.rows] == ["货币资金", "拆出资金", "资产总计"]


def test_stitch_preserves_row_source_page() -> None:
    """六元组页码溯源的前提：拼接后每行仍能指回真实页码。"""
    report = stitch(*reversed(parse_blocks(_doc(HEADER_P1, CONT_P2))))
    pages = {r.label: r.page for r in report.tables[0].rows}
    assert pages == {"货币资金": 58, "拆出资金": 59, "资产总计": 59}


def test_stitch_marks_continuation_rows() -> None:
    report = stitch(*reversed(parse_blocks(_doc(HEADER_P1, CONT_P2))))
    flags = {r.label: r.from_continuation for r in report.tables[0].rows}
    assert flags == {"货币资金": False, "拆出资金": True, "资产总计": True}


def test_second_header_starts_new_table() -> None:
    """翻页后若又出现表头，说明是新的一张表，不能并进上一张。"""
    text = _doc(HEADER_P1, HEADER_P1)
    report = stitch(*reversed(parse_blocks(text)))
    assert report.table_count == 2


def test_non_adjacent_pages_are_not_merged() -> None:
    """页码不紧邻 → 中间隔着别的页，不能续接（否则会把两张表错并）。"""
    text = (
        "<!-- page 58 of 143 -->\n" + HEADER_P1 + "\n"
        "<!-- page 62 of 143 -->\n" + CONT_P2 + "\n"
    )
    report = stitch(*reversed(parse_blocks(text)))
    assert report.table_count == 2
    assert any("未能续接" in w for w in report.warnings)


def test_column_count_mismatch_is_not_merged() -> None:
    """列数不一致 → 拒绝续接并告警，绝不猜。"""
    mismatched = (
        "<table><tbody>"
        "<tr><td>拆出资金</td><td>2</td><td>127,187,293,298.17</td></tr>"
        "</tbody></table>"
    )
    report = stitch(*reversed(parse_blocks(_doc(HEADER_P1, mismatched))))
    assert report.table_count == 2
    assert any("列数" in w for w in report.warnings)


def test_table_without_header_is_flagged() -> None:
    """无表头的孤立表必须打标，不能静默通过——列语义未知等于不可用。"""
    report = stitch(*reversed(parse_blocks(_doc("", CONT_P2))))
    assert report.table_count == 1
    assert report.tables[0].header is None
    assert report.warnings


def test_markdown_roundtrip_carries_page_annotation() -> None:
    report = stitch(*reversed(parse_blocks(_doc(HEADER_P1, CONT_P2))))
    md = report.tables[0].to_markdown()
    assert "<!-- p58 -->" in md
    assert "<!-- p59 -->" in md


def test_colspan_row_expands_to_full_width() -> None:
    """分区行（colspan=4）必须展开成 4 格，否则会与数据行错位。"""
    text = (
        "<!-- page 1 of 1 -->\n"
        "<table><tbody>"
        "<tr><td>项目</td><td>附注</td><td>A</td><td>B</td></tr>"
        '<tr><td colspan="4">流动资产:</td></tr>'
        "<tr><td>货币资金</td><td>1</td><td>1,000.00</td><td>2,000.00</td></tr>"
        "</tbody></table>"
    )
    report = stitch(*reversed(parse_blocks(text)))
    rows = report.tables[0].rows
    assert len(rows[0].cells) == 4
    assert rows[0].is_section is True
    assert rows[0].cells == ("流动资产:", "", "", "")


# --------------------------------------------------------------------------
# 真实产物回归：这是本模块存在的理由
# --------------------------------------------------------------------------


@pytest.mark.skipif(not REAL_PRODUCT.exists(), reason="缺少真实解析产物，跳过")
def test_real_balance_sheet_is_stitched_across_four_pages() -> None:
    """合并资产负债表横跨 p58–p61，必须拼成一张表，且每行页号正确。"""
    report = stitch_file(REAL_PRODUCT)
    target = next(
        (t for t in report.tables if t.find("资产总计")),
        None,
    )
    assert target is not None, "未能定位包含「资产总计」的逻辑表"
    assert target.page_start == 58
    assert target.page_end == 61
    assert target.page_span == "p58-p61"
    assert target.header_is_propagated is True
    assert target.serializations == {"html", "markdown"}, (
        "该表应同时包含两种序列化（p58-p60 为 HTML，p61 为 markdown）"
    )
    # 页码溯源：关键科目的页号必须与 PDF 一致
    assert target.find("资产总计")[0].page == 59
    assert target.find("负债合计")[0].page == 60
    assert target.find("所有者权益（或股东权益）合计")[0].page == 61


@pytest.mark.skipif(not REAL_PRODUCT.exists(), reason="缺少真实解析产物，跳过")
def test_real_balance_sheet_identity_passes_after_stitching() -> None:
    """拼接后仍能跑通会计恒等式——证明拼接没有把数字弄错位。"""
    report = stitch_file(REAL_PRODUCT)
    target = next(t for t in report.tables if t.find("资产总计"))

    def value(label: str, column: int) -> Decimal:
        row = target.find(label)[0]
        return Decimal(row.cell(column).replace(",", ""))

    assets = value("资产总计", 2)
    liabilities = value("负债合计", 2)
    equity = value("所有者权益（或股东权益）合计", 2)

    tolerance = Decimal(3) * Decimal("0.5") * Decimal(1)   # 三科目 × 0.5 × 报表单位「元」
    assert abs(assets - (liabilities + equity)) <= tolerance
    assert liabilities + equity == assets
