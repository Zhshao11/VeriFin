"""把 MinerU 的按页输出，重建成「逻辑表格」——即跨页表格的自动拼接。

为什么要这个模块
----------------
MinerU 的 markdown 产物是**按页**组织的：一页一个 `<!-- page N of M -->` 段。
但财务报表不是按页组织的——合并资产负债表会横跨 4 页，而 MinerU 会把它们
切成 4 个互不相干的 `<table>`，并且**只有第一页带表头**，续页直接从数据行开始。

如果不做拼接：
- 续页的数字没有列语义（不知道第 3 列是"本期"还是"上期"），
- 六元组里的「指标」字段无从确定，
- 直接后果是本可核验的科目被判定为"证据不足"而拒答（假拒答）。

本模块做三件事，全部是确定性规则，不依赖任何模型推断：
1. **双序列化统一**：MinerU 对同一份文档会混用 HTML `<table>` 与 markdown 管道表，
   两种语法统一解析成同一内部结构（`Block`）。
2. **表头识别**：识别「项目 / 附注 / <日期> / <日期>」这类表头行。
3. **跨页拼接**：续页块若「无表头 + 页号紧邻 + 列数一致」，则并入上一逻辑表，
   并把首页表头**向下传播**。

设计约定
--------
- 所有解析函数均为纯函数，无 IO、无全局状态，可单独单测。
- **每一行都保留其来源页码**（:attr:`MergedRow.page`）。这是六元组页码溯源的前提，
  拼接后如果丢了页码，整个证据链就断了。
- 判不准时宁可"不合并 + 打告警标志"，也不做猜测式合并——猜错会把两个不同报表的
  数字混进同一张表，比漏合更严重。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .normalize import normalize_text

__all__ = [
    "Cell",
    "BlockRow",
    "Block",
    "MergedRow",
    "LogicalTable",
    "StitchReport",
    "parse_blocks",
    "stitch",
    "stitch_text",
    "stitch_file",
    "PAGE_MARKER_RE",
]


# --------------------------------------------------------------------------
# 正则
# --------------------------------------------------------------------------

#: MinerU 的页标记，形如 `<!-- page 58 of 143 -->`。
PAGE_MARKER_RE = re.compile(r"<!--\s*page\s+(\d+)\s+of\s+(\d+)\s*-->")

_HTML_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S)
_HTML_TR_RE = re.compile(r"<tr\b.*?</tr>", re.S)
_HTML_TD_RE = re.compile(r"<t[dh]\b([^>]*)>(.*?)</t[dh]>", re.S)
_COLSPAN_RE = re.compile(r'colspan\s*=\s*"?(\d+)"?', re.I)
_TAG_RE = re.compile(r"<[^>]+>")

#: 表头里的日期样式，如 `2024年12月31日` / `2024-12-31`。
_DATE_RE = re.compile(r"\d{4}\s*[年\-/]\s*\d{1,2}\s*[月\-/]\s*\d{1,2}")

#: 金额单元格。要求有位分隔符、或两位小数、或至少 5 位连续数字——
#: 这样能把「附注」列的 `1` / `39` 这种小整数排除掉。
_AMOUNT_CELL_RE = re.compile(r"^-?[\d,，]+(?:\.\d+)?$")


def _looks_like_amount(text: str) -> bool:
    t = normalize_text(text).replace("，", ",")
    if not t or not _AMOUNT_CELL_RE.match(t):
        return False
    if "," in t:
        return True
    if re.search(r"\.\d{2}$", t):
        return True
    return len(t.lstrip("-")) >= 5


def _looks_like_date(text: str) -> bool:
    return bool(_DATE_RE.search(normalize_text(text)))


# --------------------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Cell:
    """一个表格单元格。

    Attributes:
        text: 单元格文本，已去标签、已去首尾空白。空单元格为 `""`
            （**刻意不填 `None`**：空与"没有这一列"是两回事）。
        colspan: 该单元格横跨的列数，来自 HTML 的 `colspan`；markdown 表格恒为 1。
    """

    text: str
    colspan: int = 1


@dataclass(frozen=True)
class BlockRow:
    """表格块内的一行（尚未与其他页的表格发生关系）。"""

    cells: tuple[Cell, ...]
    page: int

    def flat(self, width: int | None = None) -> tuple[str, ...]:
        """把带 `colspan` 的行展开成定宽文本序列，用于列对齐判断。

        `colspan=4` 的一个单元格会展成 `(text, "", "", "")`。
        """
        out: list[str] = []
        for cell in self.cells:
            out.append(cell.text)
            out.extend([""] * max(0, cell.colspan - 1))
        if width is not None and len(out) < width:
            out.extend([""] * (width - len(out)))
        return tuple(out[:width] if width is not None else out)

    @property
    def is_section(self) -> bool:
        """是否为分区行（如 `流动资产:` 这种横跨整行的小标题）。

        判定依据：只有一个单元格且 `colspan > 1`。
        """
        return len(self.cells) == 1 and self.cells[0].colspan > 1


@dataclass(frozen=True)
class Block:
    """一页当中的一个表格块。同一逻辑表跨页时，会解析成多个 `Block`。

    Attributes:
        page: 来源页码（1-based）。
        serialization: `"html"` 或 `"markdown"`，记录 MinerU 用了哪种序列化。
            同一逻辑表可能两种都有——这正是必须统一解析的原因。
        rows: 块内所有行，保持原文顺序。
    """

    page: int
    serialization: str
    rows: tuple[BlockRow, ...]

    @property
    def width(self) -> int:
        """块的列数：取各行展开后长度的**众数**。

        用众数而不是最大值，是为了容忍偶发的错位行（不会因为某一行多一格
        就把整块判成更宽）。
        """
        if not self.rows:
            return 0
        counts: dict[int, int] = {}
        for row in self.rows:
            n = len(row.flat())
            counts[n] = counts.get(n, 0) + 1
        return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


@dataclass(frozen=True)
class MergedRow:
    """拼接完成后的一行，**保留来源页码**。

    Attributes:
        label: 第 0 列文本，即科目名。
        cells: 展开后的定宽文本序列。
        page: 该行**在 PDF 上的真实页码**——六元组「来源」字段取这里。
        is_section: 是否为分区行（无科目语义，仅用于分组显示）。
        from_continuation: 该行来自续页（即原表格块的首页不含它）。
    """

    label: str
    cells: tuple[str, ...]
    page: int
    is_section: bool = False
    from_continuation: bool = False

    def cell(self, index: int) -> str:
        """按列取文本，越界返回空串（不抛异常，便于"这列不存在"与"这列为空"同等处理）。"""
        return self.cells[index] if 0 <= index < len(self.cells) else ""


@dataclass
class LogicalTable:
    """跨页拼接后的逻辑表。行序与 PDF 阅读顺序一致。"""

    header: tuple[str, ...] | None
    rows: list[MergedRow] = field(default_factory=list)
    page_start: int = 0
    page_end: int = 0
    serializations: set[str] = field(default_factory=set)
    #: 该逻辑表并入过的页码，按出现顺序。用来看"它到底跨了几页"。
    pages: list[int] = field(default_factory=list)
    #: 是否因列数不一致等原因**未**续接上一张表，需要人工看一眼。
    flags: list[str] = field(default_factory=list)

    @property
    def width(self) -> int:
        return len(self.header) if self.header else (
            len(self.rows[0].cells) if self.rows else 0
        )

    @property
    def page_span(self) -> str:
        """页码范围的人类可读形式，如 `p58-p61`（单页则 `p58`）。"""
        if self.page_start == self.page_end:
            return f"p{self.page_start}"
        return f"p{self.page_start}-p{self.page_end}"

    @property
    def header_is_propagated(self) -> bool:
        """是否有续页行靠传播来的表头获得列语义。"""
        return len(self.pages) > 1

    def find(self, label: str) -> list[MergedRow]:
        """按科目名精确查找行。"""
        return [r for r in self.rows if r.label == label]

    def column(self, index: int) -> list[str]:
        return [r.cell(index) for r in self.rows]

    def to_markdown(self) -> str:
        """还原为一张 markdown 表，并在每行末标注来源页码（便于人工核对拼接是否正确）。"""
        width = self.width
        if width == 0:
            return ""
        header = list(self.header) if self.header else [""] * width
        header = (header + [""] * width)[:width]
        lines = ["| " + " | ".join(header) + " |",
                 "| " + " | ".join(["---"] * width) + " |"]
        for row in self.rows:
            cells = (list(row.cells) + [""] * width)[:width]
            lines.append("| " + " | ".join(cells) + f" |  <!-- p{row.page} -->")
        return "\n".join(lines)


@dataclass
class StitchReport:
    """一次拼接的汇总结果。"""

    total_pages: int
    blocks: list[Block]
    tables: list[LogicalTable]

    @property
    def table_count(self) -> int:
        return len(self.tables)

    @property
    def warnings(self) -> list[str]:
        out: list[str] = []
        for i, t in enumerate(self.tables):
            if t.header is None:
                out.append(f"表#{i} 无表头（{t.page_span}），列语义未知")
            for flag in t.flags:
                out.append(f"表#{i}（{t.page_span}）{flag}")
        return out

    def summary(self) -> str:
        lines = [
            f"页数 {self.total_pages}｜表格块 {len(self.blocks)}｜逻辑表 {self.table_count}"
        ]
        for i, t in enumerate(self.tables):
            head = " / ".join(t.header) if t.header else "(无表头)"
            mark = "跨页拼接" if t.header_is_propagated else "单页"
            lines.append(
                f"  表#{i}  {t.page_span:<12} 行数 {len(t.rows):<4} 列数 {t.width}  "
                f"[{mark}｜{'/'.join(sorted(t.serializations))}]  {head[:46]}"
            )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# 解析：MinerU markdown -> Block
# --------------------------------------------------------------------------


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text).replace("&nbsp;", " ").replace("\u00a0", " ").strip()


def _parse_html_table(table_html: str, page: int) -> Block:
    rows: list[BlockRow] = []
    for tr in _HTML_TR_RE.findall(table_html):
        cells: list[Cell] = []
        for attrs, inner in _HTML_TD_RE.findall(tr):
            m = _COLSPAN_RE.search(attrs)
            colspan = int(m.group(1)) if m else 1
            cells.append(Cell(text=_strip_tags(inner), colspan=max(1, colspan)))
        if cells:
            rows.append(BlockRow(cells=tuple(cells), page=page))
    return Block(page=page, serialization="html", rows=tuple(rows))


def _parse_markdown_table(lines: list[str], page: int) -> Block:
    rows: list[BlockRow] = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        raw_cells = stripped.strip("|").split("|")
        cells = [Cell(text=c.strip()) for c in raw_cells]
        # 跳过 markdown 表头的分隔行（`| --- | --- |`）
        if cells and all(
            c.text and set(c.text) <= {"-", ":", " "} for c in cells
        ):
            continue
        rows.append(BlockRow(cells=tuple(cells), page=page))
    return Block(page=page, serialization="markdown", rows=tuple(rows))


def parse_blocks(text: str) -> tuple[int, list[Block]]:
    """把 MinerU 的 markdown 产物切成按页的表格块。

    Returns:
        `(总页数, 按文档顺序排列的 Block 列表)`。
        同一页内若既有 HTML 表又有 markdown 表，按出现位置先后排列。
    """
    total_pages = 0
    blocks: list[Block] = []
    parts = PAGE_MARKER_RE.split(text)
    # split 结果形如：[前言, 页码, 总页数, 正文, 页码, 总页数, 正文, ...]
    for i in range(1, len(parts), 3):
        page = int(parts[i])
        total_pages = max(total_pages, int(parts[i + 1]))
        body = parts[i + 2]

        # 按出现位置收集两类表格，最后统一排序，保证与原文顺序一致
        positioned: list[tuple[int, Block]] = []
        for m in _HTML_TABLE_RE.finditer(body):
            positioned.append((m.start(), _parse_html_table(m.group(0), page)))
        for m in re.finditer(r"(?:^\|.*$\n?)+", body, re.M):
            positioned.append(
                (m.start(), _parse_markdown_table(m.group(0).splitlines(), page))
            )
        for _, block in sorted(positioned, key=lambda kv: kv[0]):
            if block.rows:
                blocks.append(block)
    return total_pages, blocks


# --------------------------------------------------------------------------
# 拼接：Block -> LogicalTable
# --------------------------------------------------------------------------


def _is_header_row(row: BlockRow) -> bool:
    """判断是否为表头行。

    两条判据（满足其一即可）：
    1. 第 0 列含「项目」——A 股报表的标准表头形态（`项目 | 附注 | 日期 | 日期`）。
    2. 至少一格是日期样式，且除第 0 列外没有任何金额格。
    """
    flat = row.flat()
    if not flat:
        return False
    if "项目" in flat[0]:
        return True
    if any(_looks_like_date(c) for c in flat):
        return not any(_looks_like_amount(c) for c in flat[1:])
    return False


def stitch(blocks: list[Block], total_pages: int = 0) -> StitchReport:
    """把按页的表格块拼接成逻辑表。

    续接规则（三条必须**同时**满足，避免把两张不同的表错误合并）：
    1. 当前块首页行**不是**表头行；
    2. 当前块的页码 == 上一逻辑表的 `page_end + 1`（紧邻下一页）；
    3. 当前块列数 == 上一逻辑表列数。

    第 1 条是关键：MinerU 对续页**不重复输出表头**（已实测），所以"没有表头"
    恰恰是续页的信号。若换成按表名或摘要判断，反而需要模型参与，违背本项目的
    「不做语义推断」约定。
    """
    tables: list[LogicalTable] = []
    for block in blocks:
        if not block.rows:
            continue
        width = block.width
        first = block.rows[0]
        head_like = _is_header_row(first)

        if head_like:
            tables.append(
                LogicalTable(
                    header=first.flat(width),
                    page_start=block.page,
                    page_end=block.page,
                    serializations={block.serialization},
                    pages=[block.page],
                )
            )
            body = block.rows[1:]
            current = tables[-1]
            continuation = False
        else:
            prev = tables[-1] if tables else None
            can_continue = (
                prev is not None
                and block.page == prev.page_end + 1
                and width == prev.width
            )
            if not can_continue:
                # 无法续接：要么这是第一块，要么列数/页码对不上。
                # 不自作聪明合并，新开一张表并置告警标志。
                reason = (
                    "文档首个表格块即无表头"
                    if prev is None
                    else (
                        f"未能续接上一表（上一表 {prev.page_span}，列数 {prev.width}；"
                        f"本块 p{block.page}，列数 {width}）"
                    )
                )
                tables.append(
                    LogicalTable(
                        header=None,
                        page_start=block.page,
                        page_end=block.page,
                        serializations={block.serialization},
                        pages=[block.page],
                        flags=[reason],
                    )
                )
                body = block.rows
                current = tables[-1]
                continuation = False
            else:
                current = prev
                current.page_end = block.page
                current.serializations.add(block.serialization)
                current.pages.append(block.page)
                body = block.rows
                continuation = True

        for row in body:
            current.rows.append(
                MergedRow(
                    label=row.flat(width)[0] if width else row.cells[0].text,
                    cells=row.flat(width),
                    page=row.page,
                    is_section=row.is_section,
                    from_continuation=continuation,
                )
            )

    return StitchReport(total_pages=total_pages, blocks=blocks, tables=tables)


def stitch_text(text: str) -> StitchReport:
    """便利函数：直接从 MinerU 产物文本拼接。"""
    total_pages, blocks = parse_blocks(text)
    return stitch(blocks, total_pages=total_pages)


def stitch_file(path: str | Path) -> StitchReport:
    """便利函数：从 MinerU 产物文件拼接。"""
    return stitch_text(Path(path).read_text(encoding="utf-8"))
