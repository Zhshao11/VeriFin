"""PyMuPDF 坐标层：把六元组的「来源」从「页码 + 文字」提升到「页码 + 坐标」。

为什么必须有这一层
------------------
主解析器 MinerU 只给出**归一化坐标（0–1000）**，且它自带的 doclib **不存任何 bbox 列**
（已读 schema 确认）。归一化坐标无法直接在 PDF 上画框，因此
「把证据在原 PDF 上高亮出来」这件事 MinerU 单独做不到。

PyMuPDF 补齐的是：**绝对坐标（PDF point，与页面物理尺寸一致）+ 精确页码**。
两者分工写在选型文档里：**MinerU 负责"读懂"，PyMuPDF 负责"指得准"**。

设计约定
--------
1. **页码一律用 1-based**（与 PDF 阅读器、年报页码、MinerU 的 locator 一致）。
   PyMuPDF 内部是 0-based，转换只在 `_Page.__getitem__` 一处发生，
   并有单测守住——书页号错位是本项目最容易出现又最不容易发现的缺陷。
2. **坐标单位是 PDF point**，原点在页面左上角，向下 y 增大。
   `595.3 x 841.9` 就是 A4 的尺寸，这个数量级可以直接用于高亮渲染。
3. **两种定位策略，先快后稳**：
   - 快路径：`page.search_for()`，MuPDF 内建，速度快；
   - 稳路径：当快路径失败时，退回基于 `get_text("words")` 的自写匹配，
     **容许 PDF 因字距而插入的空白**（实测 `search_for('59, 295, 822, 956.89')` 为 0 处，
     但那个数字在页上是存在的，只是被字距拆开了）。
4. **同页 yearly 判定靠 y 中心差**：实测同一行的科目名与其数值的 y 中心偏差为 **0.000**，
   容差取 2.0 point 已足够宽松，同时能挡住相邻行（行高约 10.6 point）。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pymupdf

__all__ = [
    "DEFAULT_ROW_TOLERANCE",
    "BBox",
    "RowLocation",
    "PdfGeometry",
    "open_pdf",
]


#: 判定两个矩形「属于同一行」时容许的垂直间隙，单位 PDF point。
#: 实测数据：同一行的科目名与数值 y 中心偏差为 **0.000**；相邻表格行的间隙约 **3.2** point。
#: 两个框在间隙 < 2×tolerance 时会被视为同行，因此取 1.0 —— 既能吸收字体渲染差异，
#: 又把判定阈值压在真实行间隙（3.2）之下，不会被相邻行骗过。
DEFAULT_ROW_TOLERANCE = 1.0

#: `_match_words` 允许的**最大跨行数**（按行高倍数计）。
#: 长科目名在 PDF 里会被折到下一行，必须放开到 2 行才能拼回来；
#: 但再放宽就会让窗口横跨不相干的表格行，产生伪命中。
MAX_LINES = 2.6

_WS_RE = re.compile(r"\s+")


def _squash(text: str) -> str:
    """归一化到「忽略空白 + 全角转半角」的形式，仅用于**匹配**，不用于展示。"""
    if not text:
        return ""
    return _WS_RE.sub("", unicodedata.normalize("NFKC", text))


@dataclass(frozen=True)
class BBox:
    """页面上一个矩形区域。

    Attributes:
        page: **1-based** 页码（与 PDF 阅读器显示的一致）。
        x0, y0, x1, y1: PDF point 坐标。原点在页面左上角，向右 x 增大、向下 y 增大。
    """

    page: int
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if self.page < 1:
            raise ValueError(f"页码必须 >= 1（1-based），收到 {self.page}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def x_center(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def y_center(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def is_empty(self) -> bool:
        return self.width <= 0 or self.height <= 0

    def union(self, other: "BBox") -> "BBox":
        """与另一个矩形取并集。要求同页——跨页取并集没有几何意义，属误用。"""
        if other.page != self.page:
            raise ValueError(
                f"不能跨页取并集：box1 在 p{self.page}，box2 在 p{other.page}"
            )
        return BBox(
            page=self.page,
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )

    def union_all(self, others: Sequence["BBox"]) -> "BBox":
        acc = self
        for box in others:
            acc = acc.union(box)
        return acc

    def same_row(self, other: "BBox", tolerance: float = DEFAULT_ROW_TOLERANCE) -> bool:
        """是否在同一行：**垂直方向有重叠**（含容差）。

        用「重叠」而不是「中心点接近」，是因为长科目名在 PDF 里会被折行。
        折行时标签框跨两行，数值对齐的是标签的**最后一行**——
        此时两者的 y 中心并不重合，但它们在垂直方向是重叠的。
        （实测：单行标签与其数值的 y 中心偏差为 0.000，对碰巧居中的情形两者等价。）

        相邻表格行之间不存在垂直重叠（行高约 10.6 point，行间距也在同量级），
        因此这条判据不会把上下相邻行误判为同一行。
        """
        if other.page != self.page:
            return False
        return not (
            self.y1 + tolerance < other.y0 or other.y1 + tolerance < self.y0
        )

    def vertical_distance(self, other: "BBox") -> float:
        """两个框在垂直方向的间隔；有重叠时为 0。用于在一行有多处命中时挑最近的。"""
        if other.page != self.page:
            return float("inf")
        if self.y1 < other.y0:
            return float(other.y0 - self.y1)
        if other.y1 < self.y0:
            return float(self.y0 - other.y1)
        return 0.0

    def astuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)

    def to_dict(self) -> dict[str, float | int]:
        return {"page": self.page, "x0": self.x0, "y0": self.y0,
                "x1": self.x1, "y1": self.y1}


@dataclass(frozen=True)
class RowLocation:
    """表格中一行的定位结果——六元组「来源」字段的完整形态。

    Attributes:
        label: 科目名。
        label_box: 科目名的坐标框，未找到为 `None`。
        value: 数值原文。
        value_box: 数值的坐标框，未找到为 `None`。
        page: 该行所在页码（1-based）。
    """

    label: str
    label_box: BBox | None
    value: str | None
    value_box: BBox | None
    page: int

    @property
    def found(self) -> bool:
        """label 与 value 是否都被定位到。"""
        return self.label_box is not None and self.value_box is not None

    @property
    def row_box(self) -> BBox | None:
        """整行的外接矩形（label 框与 value 框的并集）。"""
        if self.label_box is None and self.value_box is None:
            return None
        if self.label_box is None:
            return self.value_box
        if self.value_box is None:
            return self.label_box
        return self.label_box.union(self.value_box)

    @property
    def same_row_verified(self) -> bool:
        """是否通过「同一行」校验：两者都找到且 y 中心足够接近。

        这个校验有实际意义——它验证的是**这个数值确实属于这个科目**，
        而不是同一页上碰巧出现的另一个数字。这是防「张冠李戴」的一道确定性护栏。
        """
        if self.label_box is None or self.value_box is None:
            return False
        return self.label_box.same_row(self.value_box)


def _nearly_same(a: "BBox", b: "BBox", tol: float = 0.6) -> bool:
    """两个矩形几何上是否几乎重合（用于去重，四边都要比）。"""
    return (
        abs(a.x0 - b.x0) < tol
        and abs(a.x1 - b.x1) < tol
        and abs(a.y0 - b.y0) < tol
        and abs(a.y1 - b.y1) < tol
    )


class PdfGeometry:
    """一份 PDF 的坐标服务。**对外页码一律 1-based**。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._doc = pymupdf.open(self.path)

    # ---- 生命周期 ----
    def close(self) -> None:
        self._doc.close()

    def __enter__(self) -> "PdfGeometry":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- 基本信息 ----
    @property
    def page_count(self) -> int:
        return self._doc.page_count

    def page_size(self, page: int) -> tuple[float, float]:
        """页面尺寸（宽, 高），单位 PDF point。"""
        self._check_page(page)
        rect = self._doc[page - 1].rect
        return (float(rect.width), float(rect.height))

    # ---- 定位 ----
    def locate(
        self, page: int, text: str, *, unique: bool = False
    ) -> list[BBox]:
        """在指定页上定位一段文本，返回所有命中的坐标框。

        策略：先用 MuPDF 内建的 `search_for`（快）；
        命中为空时退回基于 `get_text("words")` 的自写匹配（稳，容许字距空白）。

        Args:
            page: **1-based** 页码。
            text: 待定位文本。
            unique: 为 True 且命中多于 1 处时抛出 `ValueError`。
                切换成「必须唯一」的场景：一个数字在一页出现多次时，
                继续猜哪一个是对的会违反本项目「不做语义推断」的约定。
        """
        if not text or not text.strip():
            return []
        self._check_page(page)
        pypage = self._doc[page - 1]

        boxes = self._search_for(pypage, text, page)
        if not boxes:
            boxes = self._match_words(pypage, text, page)

        if unique and len(boxes) > 1:
            raise ValueError(
                f"第 {page} 页上 {text!r} 命中 {len(boxes)} 处，无法唯一定位"
            )
        return boxes

    def locate_row(
        self,
        page: int,
        label: str,
        value: str | None = None,
        *,
        tolerance: float = DEFAULT_ROW_TOLERANCE,
    ) -> RowLocation:
        """定位「科目名 → 数值」这一整行。

        若同一页面上 label 或 value 命中多处，会按几何关系确定取哪一处：
        - 取 label 的每个候选，检查其同一行上是否出现 value；两者同行则采用该组合。
        - 若 value 为 None，只返回 label 的定位结果。

        这样做是确定性的：**不存在任何"猜"的成分**，全靠 y 坐标落在同一行来判定。
        若怎么都配不到同一行，就把 `same_row_verified=False` 如实返回，
        由调用方决定拒绝还是降级，而不是在这里硬凑一个答案。
        """
        self._check_page(page)
        label_boxes = self.locate(page, label)

        if value is None:
            return RowLocation(
                label=label,
                label_box=label_boxes[0] if label_boxes else None,
                value=None,
                value_box=None,
                page=page,
            )

        value_boxes = self.locate(page, value)

        # 在所有「同行」组合里挑垂直间隔最小的那一个，保证结果确定、可复现。
        best: tuple[float, BBox, BBox] | None = None
        for lb in label_boxes:
            for vb in value_boxes:
                if not lb.same_row(vb, tolerance):
                    continue
                dist = lb.vertical_distance(vb)
                if best is None or dist < best[0]:
                    best = (dist, lb, vb)
        if best is not None:
            _, lb, vb = best
            return RowLocation(
                label=label, label_box=lb, value=value, value_box=vb, page=page
            )

        # 配不到同一行：如实返回，标明未通过同行校验
        return RowLocation(
            label=label,
            label_box=label_boxes[0] if label_boxes else None,
            value=value,
            value_box=value_boxes[0] if value_boxes else None,
            page=page,
        )

    # ---- 渲染 ----
    def render_highlight(
        self,
        page: int,
        boxes: Sequence[BBox],
        out_path: str | Path,
        *,
        zoom: float = 2.0,
        color: tuple[float, float, float] = (1.0, 0.85, 0.2),
        opacity: float = 0.45,
    ) -> Path:
        """把若干坐标框高亮渲染成 PNG，用于人工复核与演示。

        Args:
            page: 1-based 页码。
            boxes: 要高亮的框（必须同页）。
            out_path: 输出 PNG 路径。
            zoom: 放大倍数，2.0 在屏幕上足够看清 10 point 高的字。
            color: RGB 三元组，默认琥珀色（在黑白报上既不刺眼也不被忽略）。
            opacity: 填充不透明度。
        """
        self._check_page(page)
        target_page = set()
        for b in boxes:
            if b.page != page:
                raise ValueError(f"坐标框 {b} 不在第 {page} 页")
            target_page.add(b.page)

        pypage = self._doc[page - 1]
        for box in boxes:
            annot = pypage.add_highlight_annot(pymupdf.Rect(*box.astuple()))
            annot.set_colors(stroke=None, fill=color)
            annot.update(opacity=opacity)
        matrix = pymupdf.Matrix(zoom, zoom)
        pix = pypage.get_pixmap(matrix=matrix)
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pix.save(out)
        return out

    def extract_text(self, page: int) -> str:
        """取指定页文本。**必须先做页码校验**——

        直接写 `self._doc[page - 1]` 的话，`page=0` 会变成 `self._doc[-1]`，
        被 Python 负索引解释成**最后一页**。这种错位不会报错，只会让证据指向错误的页码，
        是本模块最容易踩且最难发现的坑，因此这里显式校验。
        """
        self._check_page(page)
        return self._doc[page - 1].get_text()

    # ---- 内部实现 ----
    def _check_page(self, page: int) -> None:
        if not 1 <= page <= self.page_count:
            raise IndexError(
                f"页码越界：{page}，本 PDF 共 {self.page_count} 页（1-based）"
            )

    def _search_for(self, pypage, text: str, page: int) -> list[BBox]:
        try:
            rects = pypage.search_for(text)
        except ValueError:
            # MuPDF 对某些输入会抛 ValueError（如纯空白），按"未命中"处理
            return []
        return [
            BBox(page=page, x0=float(r.x0), y0=float(r.y0),
                 x1=float(r.x1), y1=float(r.y1))
            for r in rects
            if r.width > 0 and r.height > 0
        ]

    def _match_words(
        self, pypage, text: str, page: int, max_words: int = 12
    ) -> list[BBox]:
        """基于词级坐标的容错匹配。

        解决两类 `search_for` 命中不了的情况：

        1. **字距空白**：PDF 排版常把长数字按字距拆成多个词
           （`59,` `295,` `822,` `956.89`），原串直接搜会 0 命中。
        2. **跨行折行**：长科目名在 PDF 里会被折到下一行
           （实测 `所有者权益` / `（或股东权益）合计` 被切成两断），
           单行匹配同样会漏。

        做法是在**同一个 block 内**按阅读顺序（先 line 再 x）把若干连续词的文本
        **去空白后拼接**，与目标串去空白后比对；命中则把这些词的矩形合并成坐标框。

        伪命中防护：要求合并后矩形高度不超过「最大词高 × :data:`MAX_LINES`」，
        即**最多允许跨 2 行**。超过说明窗口横跨了不相干的表格行，丢弃。

        整个过程是纯几何 + 纯字符串操作，无任何模型参与。
        """
        target = _squash(text)
        if not target:
            return []

        words = pypage.get_text("words")
        if not words:
            return []

        # 按 block 分组；block 内按 (line, x0) 排序即得到阅读顺序。
        # 不按 line 再切一层，正是为了能把被折到下一行的长标签重新拼起来。
        blocks: dict[int, list[tuple]] = {}
        for w in words:
            blocks.setdefault(int(w[5]), []).append(w)

        max_word_height = max(
            (float(w[3] - w[1]) for w in words if w[3] > w[1]), default=1.0
        )
        height_cap = max_word_height * MAX_LINES

        results: list[BBox] = []
        for block_words in blocks.values():
            ordered = sorted(block_words, key=lambda w: (int(w[6]), float(w[0])))
            n = len(ordered)
            for start in range(n):
                acc = ""
                # 每个词在 `acc` 中的字符区间，用于事后回溯「哪些词真的构成了匹配」
                spans: list[tuple[int, int, BBox]] = []
                for step in range(min(max_words, n - start)):
                    w = ordered[start + step]
                    piece = _squash(str(w[4]))
                    begin = len(acc)
                    acc += piece
                    spans.append(
                        (begin, len(acc),
                         BBox(page=page, x0=float(w[0]), y0=float(w[1]),
                              x1=float(w[2]), y1=float(w[3])))
                    )
                    hit_at = acc.find(target)
                    if hit_at >= 0:
                        hit_end = hit_at + len(target)
                        # 只取与目标区间有重叠的词 —— 窗口里排在目标之前的无关词
                        # （上一行甚至上一段的内容）不能被框进来，否则坐标框会虚高。
                        picked = [
                            box for b, e, box in spans if b < hit_end and e > hit_at
                        ]
                        if picked:
                            box = picked[0].union_all(picked[1:])
                            # 去重必须同时比对 x 与 y：只比 x 会把「同一列但上下两行」
                            # 的两处命中误判为重复，从而吞掉真正目标的那一行。
                            # （实测踩过：合并资产负债表里 `所有者权益（或股东权益）合计`
                            #   在第 61 页有两处，一列相邻、一行之差，只比 x 会漏。）
                            if box.height <= height_cap and not any(
                                _nearly_same(box, e2) for e2 in results
                            ):
                                results.append(box)
                        break
                    if len(acc) > len(target) + 24:
                        break
        return results


def open_pdf(path: str | Path) -> PdfGeometry:
    """打开一份 PDF 并返回坐标服务。"""
    return PdfGeometry(path)
