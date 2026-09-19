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
from verifin.guards import DocConstraints
from verifin.lexicon import OTHER_ISSUERS
from verifin.retrieval import RetrievalIndex, build_chunks
from verifin.scope import SCOPE_CONSOLIDATED
from verifin.tables import stitch_file

__all__ = [
    "DocumentRuntime",
    "UNIT_RE",
    "ISSUER_RE",
    "detect_unit",
    "detect_issuer",
    "derive_company_aliases",
    "build_document_runtime",
]


#: 单位识别必须用**精确字典匹配**，不能用后缀包含判断。
#: 原因（P-005）：报表里同时存在「编制单位:贵州茅台酒股份有限公司」与「单位:元 币种:人民币」，
#: 任何宽松匹配都会先命中前者，把公司名当成货币单位，
#: 而单位是容差推导的输入 —— 错了会让整张报表的核验结论失真。
UNIT_RE = re.compile(r"单位\s*[:：]\s*(万元|百万元|千元|亿元|元)")

#: 发行人名称，取自报表的「编制单位:贵州茅台酒股份有限公司」。
#:
#: 刻意带上「编制」两字来匹配，而不是用 `单位\s*[:：]`：
#: 后者会与单位识别抢同一段文本（「编制单位:」里也含「单位:」），
#: 两者迟早有一个读到对方要的东西 —— 与 P-005 是同一个坑的两面。
ISSUER_RE = re.compile(r"编制单位\s*[:：]\s*([^\s（()）]+)")

#: 企业名称后缀。剥离它是为了得到"简称"，用于主体约束的自我识别。
_CORP_SUFFIX_RE = re.compile(
    r"(?:集团)?(?:股份)?(?:有限责任)?(?:公司)$|(?:集团)?公司$"
)


def detect_unit(product: Path, stitched: Path | None = None) -> tuple[str, str]:
    """从解析产物里读报表披露单位。

    返回 `(单位, 来源说明)` —— 来源必须能被打印出来核对。

    单位是容差推导的输入（容差 = 科目数 × 0.5 × 披露单位），
    不能靠猜，也不能硬编码成「元」；来源说不清就等于没读到。
    """
    for path in _product_files(product, stitched):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hit = UNIT_RE.search(text)
        if hit:
            return hit.group(1), f"解析产物的 单位: 声明（{path.name}）"
    return "元", "解析产物里没有 单位: 声明，回退为「元」（偏保守，可能产生假阳性）"


def _product_files(product: Path, stitched: Path | None = None) -> list[Path]:
    """按优先级列出可以读取的产物文件。"""
    candidates: list[Path] = []
    if stitched is not None and stitched.exists():
        candidates.append(stitched)
    if product.is_dir():
        candidates += sorted(product.rglob("*.md"))[:20]
    elif product.exists():
        candidates.append(product)
    return candidates


def detect_issuer(product: Path, stitched: Path | None = None) -> tuple[str, str]:
    """从解析产物里读发行人名称（「编制单位:」）。

    返回 `(公司名, 来源说明)`。读不到返回空串 —— `主体约束`此时退化为不检查，
    因为**猜一个公司名比不检查更危险**：猜错会把本可回答的问题全部拦掉。
    """
    for path in _product_files(product, stitched):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        hit = ISSUER_RE.search(text)
        if hit:
            return hit.group(1), f"解析产物的 编制单位: 声明（{path.name}）"
    return "", "解析产物里没有 编制单位: 声明，主体约束退化为不检查"


def derive_company_aliases(full_name: str) -> tuple[str, ...]:
    """从完整公司名推出可能的简称。

    这是**启发式**，所以它只用于「问句点的主体是不是本文档自己」这一件事，
    而且只在**别的主体已被识别出来**时才被咨询（见 `DocConstraints.owns`）。
    换句话说：它的误判不会凭空制造拒答，只会让一条本该拒答的题漏过去 ——
    这个方向的错误是可接受的，反向的不是。

    产出三类：
    1. 完整名称；
    2. 剥掉企业后缀（`…股份有限公司` → `…`）；
    3. 前 4 个字（A 股发行人简称多为 2–4 字，如「贵州茅台」「宁德时代」）。
    """
    name = (full_name or "").strip()
    if not name:
        return ()
    out: list[str] = [name]
    core = _CORP_SUFFIX_RE.sub("", name).strip()
    if len(core) >= 2:
        out.append(core)
    if len(name) >= 6:
        out.append(name[:4])
    seen: dict[str, None] = {}
    for item in out:
        if len(item) >= 2:
            seen.setdefault(item, None)
    return tuple(seen)


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
    by_label_scope: dict[tuple[str, str | None], Any] = field(default_factory=dict)
    """`(科目名, 报表口径)` → chunk。口径消歧的全部依据。"""
    constraints: DocConstraints = field(default_factory=DocConstraints)
    """主体 / 期间约束，全部从文档自身推导（编制单位 + 表头年份）。"""

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

        注意 `company` 只影响**六元组的展示**；**主体约束**用的是
        `self.constraints.company`（从 PDF 的编制单位读出来的），
        两者不是一回事：前者可能是演示配置里的花哨写法，后者必须是文档里的原文。

        Note:
            口径匹配与 `constraints.company` 的取值口径不同，容易混淆，故显式区分：
            - `constraints.company` 用于主体约束（判断"问的是不是这家公司"）；
            - `company` 用于六元组展示（可能带股票代码等修饰）。
        """
        return ToolRuntime(
            by_label=self.by_label,
            by_label_scope=self.by_label_scope,
            index=self.index,
            unit=self.unit,
            company=company,
            period=period,
            pdf_open=pdf_open,
            index_lock=self.index_lock,
            constraints=self.constraints,
            company_aliases=self.constraints.company_aliases,
            report_years=self.constraints.report_years,
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
    by_label_scope: dict[tuple[str, str | None], Any] = {}
    for chunk in chunks:
        by_label_scope.setdefault((chunk.label, chunk.scope), chunk)
        by_label.setdefault(chunk.label, chunk)

    # 主口径那一行：优先「合并」，其次未标注口径，最后「母公司」。
    #
    # 不能简单"取首次出现"：那只是碰巧对（合并表在文档里靠前）。
    # 写死优先级，才能在换一份年报、报表顺序不同时仍然稳定可复现。
    # 单口径科目（只在一个口径下存在）不受影响 —— 它们只有一个候选。
    for chunk in chunks:
        current = by_label.get(chunk.label)
        if current is None or chunk.scope == SCOPE_CONSOLIDATED:
            by_label[chunk.label] = chunk
        elif current.scope is None and chunk.scope is not None:
            # 未标注口径的那一行不该压过明确标注「合并」的
            continue

    index = RetrievalIndex(sqlite3.connect(str(index_db), check_same_thread=False))
    unit, unit_source = detect_unit(product, stitched)
    issuer, _issuer_source = detect_issuer(product, stitched)
    years = sorted({y for table in report.tables for y in table.report_years})
    constraints = DocConstraints(
        company=issuer,
        company_aliases=derive_company_aliases(issuer),
        other_issuers=OTHER_ISSUERS,
        report_years=tuple(years),
    )
    return DocumentRuntime(
        doc_id=doc_id,
        report=report,
        chunks=chunks,
        by_label=by_label,
        index=index,
        unit=unit,
        unit_source=unit_source,
        pdf=pdf,
        by_label_scope=by_label_scope,
        constraints=constraints,
    )
