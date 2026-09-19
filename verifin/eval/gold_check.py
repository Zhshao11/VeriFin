"""gold 标注自检：让「标注错了」变成一条可判定的失败。

题库最大的风险不是漏题，是**标错**——gold 页填错、gold 值抄错。
标错的题会把系统的正确行为判成错误（假阳性），比漏题危险。

所以这里用三道独立闸门校验每一条 gold，全部依赖**已有产物**，不引入新的人工核对：

| 闸门 | 检查什么 | 为什么不能省 |
|---|---|---|
| G1 索引页 | gold 值能在 `gold_pages` 里某一页的检索块中找到 | 证明「确实有一页承载这个数」 |
| G2 证据片段 | 每个 gold 片段能在**同一页**的原文里定位 | 证明片段属实，不是凭印象写的 |
| G3 PDF 原件 | gold 值能在**原始 PDF** 该页的文本层里找到 | 证明标注忠于原件，而不是只忠于我们的解析结果 |

**G3 是三道里唯一独立于本项目解析产物的**。只有它通过，
才能排除「解析器错了、标注跟着一起错」这种自洽型错误。

另外顺带记录一个信息量很大的字段：`value_pages` —— 该数值在**全文**出现的所有页。
若一个数值只出现在 gold 页，说明这道题能区分口径；若在多页出现，
说明光靠数值无法判对错，必须靠页码。
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verifin.eval.docs import DocSpec
from verifin.eval.schema import EvalItem
from verifin.span import MatchMode, locate_span


@dataclass(frozen=True)
class GoldCheck:
    """一条题目的 gold 自检结果。"""

    item_id: str
    checked: bool
    """是否真的执行了检查（产物缺失时为 False，不能当成通过）。"""
    ok: bool
    g1_index: str = ""
    g2_evidence: str = ""
    g3_pdf: str = ""
    value_pages: tuple[int, ...] = ()
    problems: tuple[str, ...] = ()


def load_index_pages(index_db: str | Path) -> dict[int, str]:
    """从检索索引还原「页 → 该页全部块文本」。"""
    con = sqlite3.connect(f"file:{index_db}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT page, text FROM chunks ORDER BY page, rowid").fetchall()
    finally:
        con.close()
    raw: dict[int, list[str]] = {}
    for page, text in rows:
        raw.setdefault(int(page), []).append(text or "")
    return {p: "\n".join(chunks) for p, chunks in raw.items()}


def pdf_page_text(pdf: str | Path, page: int) -> str | None:
    """取 PDF 某一页的文本层。取不到返回 None（页码越界或没有文本层）。"""
    try:
        import pymupdf  # PyMuPDF 1.x 的正式包名；`fitz` 是会被移除的旧别名
    except ImportError:  # pragma: no cover - 环境缺依赖
        return None

    if page < 1:
        return None
    with pymupdf.open(str(pdf)) as doc:
        if page > doc.page_count:
            return None
        return doc[page - 1].get_text()


def _locate_in_pages(
    targets: list[str],
    pages: list[int],
    lookup,
) -> tuple[int, int, str]:
    """在一组页里找一组目标。返回 (命中数, 总数, 命中方式的说明)。

    判定型答案（恒等式）没有 gold 数值，此时 `targets` 传的是操作数片段 ——
    操作数确实出现在原文里，是这类题唯一可机械校验的东西。
    """
    if not targets:
        return 0, 0, "-"
    hits = 0
    modes: set[MatchMode] = set()
    for target in targets:
        for page in pages:
            text = lookup(page)
            if text is None:
                continue
            hit = locate_span(target, text)
            if hit is not None:
                hits += 1
                modes.add(hit.mode)
                break
    if not hits:
        return 0, len(targets), "-"
    mode = "逐字" if modes == {MatchMode.EXACT} else "归一化"
    return hits, len(targets), mode


def check_item(
    item: EvalItem,
    spec: DocSpec,
    *,
    index_pages: dict[int, str] | None = None,
) -> GoldCheck:
    """对一条题目跑三道闸门。

    应拒答的题没有 gold 值可查，只校验「它凭什么不可答」——
    即 `refusal_reason` 必须非空，这条由 `schema.validate_item` 负责，
    自检阶段直接记为通过。

    判定型答案（恒等式 / 启发式题）没有 gold 数值，此时改查**操作数原值**：
    它们是报表上的真数字，能逐字在原文里找到，是这类题唯一可机械校验的部分。
    """
    if item.is_refusal:
        return GoldCheck(item.item_id, checked=True, ok=True, g2_evidence="应拒答题，无 gold 值需校验")

    problems: list[str] = []
    pages = list(item.gold_pages)
    answer = item.gold_value or ""
    #: G1 / G3 要校验的目标 —— 优先级是有讲究的：
    #:
    #: 1. 有 `gold_operands` 就查操作数。操作数是报表上的真数字，能逐字找到；
    #:    而派生量（如毛利率 91.93%）是算出来的，原文里根本没有这个串，
    #:    拿它去过闸门必然失败 —— 那是闸门设计错，不是标注错。
    #: 2. 没有操作数（L1 单值题）就查 `gold_value` 本身，此时它就是原文原值。
    targets = (
        list(item.gold_operands.values())
        if item.gold_operands
        else ([answer] if answer else [])
    )
    label = "操作数" if item.gold_operands else "答案"

    # ---------------------------------------------------------------- G1
    g1, g1_ok = "跳过（未建索引）", False
    if index_pages is not None:
        lookup_idx = lambda p: index_pages.get(p)  # noqa: E731
        hits, total, mode = _locate_in_pages(targets, pages, lookup_idx)
        g1 = f"{label} {hits}/{total} 命中" + (f"（{mode}）" if hits else "")
        g1_ok = hits == total and total > 0
        if not g1_ok:
            missing = [p for p in pages if p not in index_pages]
            extra = f"；其中 {missing} 页不在索引里" if missing else ""
            problems.append(f"G1 {label}在 gold_pages 任一页的检索块里找不到{extra}")

    # ---------------------------------------------------------------- G2
    g2, g2_ok = "跳过（未建索引）", False
    if index_pages is not None:
        wanted = list(item.gold_evidence)
        hits, total, _ = _locate_in_pages(wanted, pages, lambda p: index_pages.get(p))
        g2 = f"{hits}/{total} 片段命中"
        g2_ok = hits == total and total > 0
        if not g2_ok:
            problems.append("G2 有标注片段在 gold_pages 的原文里找不到")

    # ---------------------------------------------------------------- G3
    g3, g3_ok = "跳过（PDF 缺失）", False
    if spec.pdf.exists():
        lookup_pdf = lambda p: pdf_page_text(spec.pdf, p)  # noqa: E731
        hits, total, mode = _locate_in_pages(targets, pages, lookup_pdf)
        g3 = f"{label} {hits}/{total} 命中" + (f"（{mode}）" if hits else "")
        g3_ok = hits == total and total > 0
        if not g3_ok:
            problems.append(
                f"G3 {label}在原始 PDF 的 gold 页文本层里找不到（标注与原件不一致？）"
            )

    # ------------------------------------------------- 附加信息：数值唯一性
    #: 探针取「主校验目标」：L1 题是答案本身，L2 题是第一个操作数。
    #: 它在多少页出现，决定了这题能不能只靠数值判对错 —— 详见 summarize() 的说明。
    probe = answer or max(item.gold_operands.values(), key=len, default="")
    value_pages: tuple[int, ...] = ()
    if index_pages is not None and probe:
        value_pages = tuple(
            p for p, text in sorted(index_pages.items()) if locate_span(probe, text) is not None
        )

    return GoldCheck(
        item_id=item.item_id,
        checked=True,
        ok=g1_ok and g2_ok and g3_ok,
        g1_index=g1,
        g2_evidence=g2,
        g3_pdf=g3,
        value_pages=value_pages,
        problems=tuple(problems),
    )


def check_bank(items: list[EvalItem], spec: DocSpec) -> list[GoldCheck]:
    index_pages: dict[int, str] | None = None
    if spec.index_db is not None and Path(spec.index_db).exists():
        index_pages = load_index_pages(spec.index_db)
    return [check_item(i, spec, index_pages=index_pages) for i in items]


def summarize(checks: list[GoldCheck]) -> dict[str, Any]:
    failed = [c for c in checks if not c.ok]
    ambiguous = [c for c in checks if len(c.value_pages) > 1]
    return {
        "总数": len(checks),
        "通过": len(checks) - len(failed),
        "失败": len(failed),
        "失败题号": [c.item_id for c in failed],
        "数值在多页出现的题": [c.item_id for c in ambiguous],
    }


def render_report(checks: list[GoldCheck], *, show_all: bool = False) -> str:
    lines: list[str] = []
    for c in checks:
        flag = "✓" if c.ok else "✗"
        pages = ("、".join(f"p{p}" for p in c.value_pages)) if c.value_pages else "-"
        if not show_all and c.ok:
            continue
        lines.append(f"  [{flag}] {c.item_id}")
        lines.append(f"        G1 索引页：{c.g1_index}")
        lines.append(f"        G2 片段  ：{c.g2_evidence}")
        lines.append(f"        G3 PDF   ：{c.g3_pdf}")
        lines.append(f"        数值出现页：{pages}")
        for p in c.problems:
            lines.append(f"        ! {p}")
    if not lines:
        lines.append("  （全部通过，无非通过项可列）")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """命令行入口。评测报告里要能给出**可复现**的自检命令，不能只靠临时脚本。"""
    import argparse
    import sys

    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from verifin.eval.bank import load_bank
    from verifin.eval.docs import get_spec

    parser = argparse.ArgumentParser(description="VeriFin gold 标注自检（G1 索引页 / G2 片段 / G3 原件）")
    parser.add_argument("--bank", default=str(root / "data" / "eval" / "b_moutai_2024.jsonl"))
    parser.add_argument("--doc", default="MOUTAI_2024", help="文档键（查 docs.py 登记表）")
    parser.add_argument("--all", action="store_true", help="也列出通过项")
    parser.add_argument("--json", help="把结果写到这个 JSON 文件")
    args = parser.parse_args(argv)

    doc_spec = get_spec(args.doc)
    items = load_bank(args.bank)
    checks = check_bank(items, doc_spec)
    summary = summarize(checks)

    print("=" * 72)
    print("gold 自检汇总")
    print("=" * 72)
    print(f"文档：{doc_spec.label}（解析窗口 p{doc_spec.parse_pages[0]}–p{doc_spec.parse_pages[1]}）")
    print(
        f"通过 {summary['通过']}/{summary['总数']}"
        f"（失败 {summary['失败']}）"
    )
    if summary["失败题号"]:
        print("失败题号：" + "、".join(summary["失败题号"]))
    ambiguous = summary["数值在多页出现的题"]
    print(
        "数值在多页出现的题："
        + ("、".join(ambiguous) if ambiguous else "（无）")
        + "  ← 这些题光比数值不足以判对错，必须同时比页码"
    )
    print()
    print("=" * 72)
    print("逐条（只列未通过；--all 列出全部）")
    print("=" * 72)
    print(render_report(checks, show_all=args.all))

    if args.json:
        payload = {
            "summary": summary,
            "checks": [
                {
                    "item_id": c.item_id,
                    "ok": c.ok,
                    "g1_index": c.g1_index,
                    "g2_evidence": c.g2_evidence,
                    "g3_pdf": c.g3_pdf,
                    "value_pages": list(c.value_pages),
                    "problems": list(c.problems),
                }
                for c in checks
            ],
        }
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已写入 {args.json}")

    return 0 if summary["失败"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
