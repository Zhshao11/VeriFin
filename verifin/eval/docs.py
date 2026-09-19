"""评测集用到的文档登记表。

评测题只记 `doc` 键，具体路径从这里查。
把路径集中在一处的原因：gold 自检需要同时读**解析产物**、**检索索引**和**原始 PDF**，
三者的路径必须成组出现，散落在脚本里迟早会错配。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass(frozen=True)
class DocSpec:
    """一份评测文档的三件套路径。

    `parse_pages` 是 MinerU 标记的解析页范围；**它不是全文**。
    题库里凡声明了 `page_out_of_range` 的题，都是因为目标行落在这个窗口之外。
    """

    doc: str
    label: str
    pdf: Path
    index_db: Path | None
    parse_pages: tuple[int, int]
    note: str = ""


DOC_REGISTRY: dict[str, DocSpec] = {
    "MOUTAI_2024": DocSpec(
        doc="MOUTAI_2024",
        label="贵州茅台 2024 年年度报告",
        pdf=ROOT / "data" / "pdfs" / "MOUTAI_2024_ANNUAL.pdf",
        index_db=ROOT / "data" / "index" / "moutai2024.db",
        parse_pages=(58, 67),
        note="仅解析了财务报表章节（合并/母公司资产负债表、利润表、合并现金流量表的大部分）",
    ),
    "MOUTAI_2023": DocSpec(
        doc="MOUTAI_2023",
        label="贵州茅台 2023 年年度报告",
        pdf=ROOT / "data" / "pdfs" / "MOUTAI_2023_ANNUAL.pdf",
        index_db=None,
        parse_pages=(0, 0),
        note="PDF 已下载但未解析、未建索引，暂不可评测（跨年对比题依赖它）",
    ),
}


def get_spec(doc: str) -> DocSpec:
    if doc not in DOC_REGISTRY:
        known = "、".join(sorted(DOC_REGISTRY))
        raise KeyError(f"未知文档键 {doc!r}（已登记：{known}）")
    return DOC_REGISTRY[doc]
