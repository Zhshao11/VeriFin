"""PyMuPDF 坐标层的单元测试。

纯几何部分不依赖任何 PDF，可直接测；真实 PDF 相关的用例用 `skipif` 保底。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fin_verify.geometry import (
    DEFAULT_ROW_TOLERANCE,
    BBox,
    PdfGeometry,
    open_pdf,
)

PDF = Path("data/pdfs/MOUTAI_2024_ANNUAL.pdf")
needs_pdf = pytest.mark.skipif(not PDF.exists(), reason="缺少真实 PDF，跳过")


def box(page: int, y0: float, y1: float, x0: float = 100.0, x1: float = 200.0) -> BBox:
    return BBox(page=page, x0=x0, y0=y0, x1=x1, y1=y1)


# --------------------------------------------------------------------------
# 纯几何
# --------------------------------------------------------------------------


def test_page_number_must_be_1_based() -> None:
    """页码必须是 1-based。0 页过去又一个键盘，会让整份报告的页码整体错位。"""
    with pytest.raises(ValueError):
        BBox(page=0, x0=0, y0=0, x1=1, y1=1)


def test_union_requires_same_page() -> None:
    """跨页取并集没有几何意义，必须显式报错而不是静默给出一个荒谬的矩形。"""
    a, b = box(1, 10, 20), box(2, 10, 20)
    with pytest.raises(ValueError):
        a.union(b)


def test_union_and_properties() -> None:
    a, b = box(1, 10, 20, x0=100, x1=200), box(1, 15, 25, x0=180, x1=260)
    u = a.union(b)
    assert u.astuple() == (100.0, 10.0, 260.0, 25.0)
    assert u.width == 160.0 and u.height == 15.0
    assert u.y_center == pytest.approx(17.5)


def test_same_row_uses_vertical_overlap_not_center() -> None:
    """判据是垂直重叠，不是中心点相等。

    原因：长科目名在 PDF 里会被折行，标签框跨两行，
    而数值对齐的是标签的**最后一行**——此时两者 y 中心并不重合，但确实重叠。
    """
    tall = box(1, 100.0, 120.0)          # 跨两行的标签，中心 110
    second_line = box(1, 112.0, 122.0)   # 数值对齐第二行，中心 117
    assert tall.y_center != pytest.approx(second_line.y_center)
    assert tall.same_row(second_line) is True


def test_adjacent_rows_are_not_same_row() -> None:
    """相邻表格行不能判为同一行——这是防「张冠李戴」的核心保证。

    间隙按年报实测取 3.2 point（见 `DEFAULT_ROW_TOLERANCE` 的注释）。
    """
    upper = box(1, 700.0, 710.6)
    lower = box(1, 713.8, 724.4)     # 间隙 3.2 point
    assert upper.same_row(lower) is False
    assert lower.same_row(upper) is False


def test_same_row_is_false_across_pages() -> None:
    assert box(1, 100, 110).same_row(box(2, 100, 110)) is False


def test_vertical_distance_overlapping_is_zero() -> None:
    assert box(1, 100, 110).vertical_distance(box(1, 105, 115)) == 0.0


def test_vertical_distance_separated() -> None:
    upper, lower = box(1, 100, 110), box(1, 130, 140)
    assert upper.vertical_distance(lower) == pytest.approx(20.0)


# --------------------------------------------------------------------------
# 页号转换（最容易出错的一处）
# --------------------------------------------------------------------------


@needs_pdf
def test_page_numbering_is_1_based() -> None:
    """对外一律 1-based：第 1 页取到的是 PDF 第一页，最后一页不能越界。"""
    with open_pdf(PDF) as geo:
        assert geo.page_count == 143
        # 1-based：第 1 页可访问
        assert "贵州茅台" in geo.extract_text(1)
        # 越界保护
        with pytest.raises(IndexError):
            geo.extract_text(0)
        with pytest.raises(IndexError):
            geo.extract_text(144)


@needs_pdf
def test_page_size_is_in_pdf_points() -> None:
    """A4 页面应为 595 x 842 point 量级——确认单位是物理坐标不是归一化 0-1000。"""
    with open_pdf(PDF) as geo:
        w, h = geo.page_size(1)
        assert 590 < w < 600 and 840 < h < 845


# --------------------------------------------------------------------------
# 定位（真实 PDF）
# --------------------------------------------------------------------------


@needs_pdf
def test_locate_returns_empty_for_absent_text() -> None:
    with open_pdf(PDF) as geo:
        assert geo.locate(58, "这个字符串绝不存在于报表里") == []


@needs_pdf
@pytest.mark.parametrize(
    "page,label,value",
    [
        (58, "货币资金", "59,295,822,956.89"),
        (59, "资产总计", "298,944,579,918.70"),
        (59, "存货", "54,343,285,157.47"),
        (60, "应付账款", "3,514,969,230.31"),
        (60, "租赁负债", "313,906,415.76"),
        (61, "负债和所有者权益（或股东权益）总计", "298,944,579,918.70"),
    ],
)
def test_locate_row_verifies_same_row(page: int, label: str, value: str) -> None:
    """科目名与数值必须落在同一行——这是「这个数确实属于这个科目」的确定性护栏。"""
    with open_pdf(PDF) as geo:
        loc = geo.locate_row(page, label, value)
        assert loc.found is True, f"p{page} {label} 未能同时定位到 label 与 value"
        assert loc.same_row_verified is True
        row_box = loc.row_box
        assert row_box is not None
        # 行框应同时覆盖 label 与 value
        assert row_box.x0 == pytest.approx(min(loc.label_box.x0, loc.value_box.x0))


@needs_pdf
def test_wrapped_label_is_located_across_two_lines() -> None:
    """长科目名被 PDF 折到下一行时仍要能定位到。

    实测：`所有者权益（或股东权益）合计` 在第 61 页被切成
    `所有者权益（或股东权` + `益）合计` 两断分布在相邻两行。
    """
    with open_pdf(PDF) as geo:
        loc = geo.locate_row(61, "所有者权益（或股东权益）合计", "242,011,315,120.60")
        assert loc.found is True
        assert loc.same_row_verified is True
        # 标签跨行，因此合并高度应接近但不超过 2 行（行高约 10.6）
        assert loc.label_box.height > 10.6
        assert loc.label_box.height < 10.6 * 2.6


@needs_pdf
def test_substring_collision_prefers_the_row_containing_the_value() -> None:
    """短标签会被包在更长的标签里命中，必须挑出与数值同行的那一个。

    `所有者权益（或股东权益）合计` 是
    `归属于母公司所有者权益（或股东权益）合计` 的子串，
    第 61 页上存在两处候选，分别在不同行。
    """
    with open_pdf(PDF) as geo:
        candidates = geo.locate(61, "所有者权益（或股东权益）合计")
        assert len(candidates) >= 2, "应识别到多处候选（这是本用例的前提）"

        short = geo.locate_row(61, "所有者权益（或股东权益）合计", "242,011,315,120.60")
        long_ = geo.locate_row(
            61, "归属于母公司所有者权益（或股东权益）合计", "233,105,984,399.47"
        )
        assert short.same_row_verified is True
        assert long_.same_row_verified is True
        # 两者必须落在**不同**的行上，否则说明挑错了对象
        assert short.label_box.y0 != pytest.approx(long_.label_box.y0)


@needs_pdf
def test_value_on_another_page_fails_row_verification() -> None:
    """数值不在该页时不能伪造通过同行校验——宁可返回 False 让上层拒答。"""
    with open_pdf(PDF) as geo:
        loc = geo.locate_row(59, "资产总计", "59,295,822,956.89")  # 该值在 p58
        assert loc.same_row_verified is False


@needs_pdf
def test_unique_flag_raises_on_ambiguous_hit() -> None:
    """一处文案在页面上出现多次时，`unique=True` 必须报错而不是任猜一个。"""
    with open_pdf(PDF) as geo:
        with pytest.raises(ValueError):
            geo.locate(61, "所有者权益（或股东权益）合计", unique=True)
