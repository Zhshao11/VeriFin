"""把一个文档装配成可运行的运行时。

「解析产物 + 检索索引 + 坐标层 → ToolRuntime」这段装配原先在
`scripts/demo_agent.py` 与 `web/server.py` 里各写了一遍，D4 的评测执行器
还需要第三遍。三份拷贝的后果不是代码丑，而是**三份会各自漂移**：
单位识别、索引连接方式、按科目去重的规则只要有一处改了而另一处没改，
演示、网页与评测跑出来的就不是同一个系统 —— 而评测的意义正是「量的是同一个系统」。

所以装配收敛到这一处，三边都从这里取。凡是「装配参数」都集中在这个模块里、
不许散出去，理由与 :mod:`verifin.eval.docs` 把三条路径成组存放是同一个：
分散的路径迟早错配。
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from verifin.agent.tools import ToolRuntime
from verifin.retrieval import RetrievalIndex, build_chunks
from verifin.tables import stitch_file

__all__ = ["DocumentRuntime", "UNIT_RE", "detect_unit", "build_document_runtime"]


#: 单位识别必须用**精确字典匹配**，不能用后缀包含判断。
#: 原因（P-005）：报表里同时存在「编制单位:贵州茅台酒股份有限公司」与「单位:元 币种:人民币」，
#: 任何宽松匹配都会先命中前者，把公司名当成货币单位，
#: 而单位是容差推导的输入 —— 错了会让整张报表的核验结论失真。
UNIT_RE = re.compile(r"单位\s*[:：]\s*(万元|百万元|千元|亿元|元)")


def detect_unit(product: Path, stitched: Path | None = None) -> tuple[str, str]:
    """从解析产物里读报表披露单位。

    返回 `(单位, 来源说明)` —— 来源必须能被打印出来核对。

    单位是容差推导的输入（容差 = 科目数 × 0.5 × 披露单位），
    不能靠猜，也不能硬编码成「元」；来源说不清就等于没读到。
    """
    candidates: list[Path] = []
    if stitched is not None and stitched.exists():
        candidates.append(stitched)
    if product.is_dir():
        candidates += sorted(product.rglob("*.md"))[:20]
    elif product.exists():
        candidates.append(product)

    for path in candidates:
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hit = UNIT_RE.search(text)
        if hit:
            return hit.group(1), f"解析产物的 单位: 声明（{path.name}）"
    return "元", "解析产物里没有 单位: 声明，回退为「元」（偏保守，可能产生假阳性）"


@dataclass
class DocumentRuntime:
    """一个文档装载好之后的全部共享资源。"""

    doc_id: str
    report: Any
    chunks: list[Any]
    by_label: dict[str, Any]
    index: RetrievalIndex
    unit: str
    unit_source: str
    pdf: Path | None = None
    index_lock: threading.Lock = field(default_factory=threading.Lock)

    def tool_runtime(
        self,
        *,
        company: str = "",
        period: str = "",
        pdf_open: Callable[[], Any] | None = None,
    ) -> ToolRuntime:
        """打包成图编排层要用的 `ToolRuntime`。

        `company` / `period` 目前只有演示配置一个来源（封面结构化解析未接入），
        调用方必须自己说清这一点 —— 它们是六元组的前两个字段，
        来源是「配置」就得标「配置」。
        """
        return ToolRuntime(
            by_label=self.by_label,
            index=self.index,
            unit=self.unit,
            company=company,
            period=period,
            pdf_open=pdf_open,
            index_lock=self.index_lock,
        )


def build_document_runtime(
    *,
    doc_id: str,
    product: Path,
    index_db: Path,
    pdf: Path | None = None,
    stitched: Path | None = None,
) -> DocumentRuntime:
    """装配一个文档的运行时。

    Args:
        doc_id: 文档标识，写进每个检索块的 `doc_id`。
        product: MinerU 解析产物（目录或单个 .md）。
        index_db: 检索索引（SQLite）。
        pdf: 原始 PDF；给了才能做坐标定位与高亮。
        stitched: 跨页拼接后的 markdown（存在时优先用它识别披露单位）。

    Note:
        索引连接显式开 `check_same_thread=False`，并由 :attr:`DocumentRuntime.index_lock`
        串行化访问。SQLite 默认禁止跨线程，而演示服务是单进程多线程。
        「开了就不管」是错的 —— `check_same_thread=False` 只是**允许**跨线程，
        并发的 `execute` + `commit` 仍会事务交错，所以必须配一把锁。
    """
    report = stitch_file(product)
    chunks = build_chunks(report.tables, doc_id)
    by_label: dict[str, Any] = {}
    for chunk in chunks:
        # 同一科目名在年报里会出现多次（合并/母公司两套报表）。
        # 取首次出现是有意的：这样「不限定口径」的问句会稳定落在合并口径上，
        # 而不是随检索排名摇摆 —— 可复现比"偶尔取到母公司"重要。
        by_label.setdefault(chunk.label, chunk)

    index = RetrievalIndex(sqlite3.connect(str(index_db), check_same_thread=False))
    unit, unit_source = detect_unit(product, stitched)
    return DocumentRuntime(
        doc_id=doc_id,
        report=report,
        chunks=chunks,
        by_label=by_label,
        index=index,
        unit=unit,
        unit_source=unit_source,
        pdf=pdf,
    )
