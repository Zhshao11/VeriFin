"""实测 A/B：财务词典注入前后，端到端检索 top-1 命中率变化。

为什么重新设计这个实验
----------------------
第一版只在**查询侧**关闭词典，但 FTS5 索引本身是用「带词典」的切词建的。
两侧分词粒度不一致，本身就是 P-003 那类静默 0 命中的成因 ——
这样做出来的对比不干净（差异里混进了"两侧不一致"这个无关变量）。
本版改成 **like-for-like**：建两个索引，各自用与自身一致的切词方式检索。

为什么必须在同一进程内、且顺序固定
----------------------------------
`verifin.retrieval.tokenize` 每次调用都会执行 `apply_lexicon()`，而它是
**幂等且全局**的（`_applied` 标志，注入后不可撤销）。所以正确顺序是：
  1) 进程启动后先建「无词典索引」，此时 jieba 是默认词典；
  2) 再 `apply_lexicon()` 注入，建「有词典索引」；
  3) 两个索引分别用各自对应的切词函数检索。
顺序一旦反过来，第二步拿到的仍是注入后的状态，A/B 恒等。

被测对象：**完整四路召回 + RRF 融合**（`RetrievalIndex.retrieve`）的 top-1，
不是单看词法通道 —— 要看的是"系统检索"整体变没变好，就必须测系统。

用法
----
    .venv/bin/python scripts/measure_lexicon_ablation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PARSED = ROOT / "data" / "parsed" / "moutai2024_fs"
# 中间产物写到临时目录，不污染 data/index/ 下的正式索引。
TMP_NOL_EX = ROOT / "data" / "index" / "_ablation_nolex.db"
TMP_LEX = ROOT / "data" / "index" / "_ablation_lex.db"

QUERY_LABELS: tuple[str, ...] = (
    "营业收入", "营业总收入", "营业成本", "净利润", "货币资金",
    "资产总计", "负债合计", "归属于母公司股东的净利润",
    "经营活动产生的现金流量净额", "未分配利润", "管理费用",
    "财务费用", "销售费用", "研发费用", "营业外收入",
)


def build_tokens_no_lexicon(text: str) -> list[str]:
    """复刻 tokenize 的过滤逻辑，但**不注入财务词典**。"""
    import jieba
    from verifin.retrieval import STOPWORDS, _CJK, _TOKEN

    out: list[str] = []
    for word in jieba.lcut(text):
        word = word.strip()
        if not word:
            continue
        if not (_CJK.match(word) or _TOKEN.fullmatch(word)):
            continue
        if word in STOPWORDS:
            continue
        out.append(word)
    return out


def main() -> None:
    if not PARSED.exists():
        raise SystemExit(f"解析产物不存在：{PARSED}")

    import verifin.retrieval as R
    from verifin.tables import stitch_file

    report = stitch_file(PARSED)
    chunks = R.build_chunks(report.tables, "MOUTAI_2024")

    # ---- 态 1：默认词典。先建 —— jieba 此刻尚未被注入 ----
    # 把 tokenize 临时替换成无词典版本，使 chunk 的 FTS 词表与查询侧一致。
    real_tokenize = R.tokenize
    R.tokenize = build_tokens_no_lexicon
    try:
        con_nolex = R.create_index(TMP_NOL_EX, chunks)
        con_nolex.close()
    finally:
        R.tokenize = real_tokenize

    # ---- 注入财务词典，态 2 ----
    from verifin.lexicon import apply_lexicon

    apply_lexicon()
    con_lex = R.create_index(TMP_LEX, chunks)
    con_lex.close()

    import sqlite3

    real_tokenize = R.tokenize  # 此刻已被 apply_lexicon 注入，等价于「有词典」切词

    def eval_index(db_path: Path, query_fn) -> tuple[int, list[tuple[str, str, bool]]]:
        """在给定索引上跑端到端 top-1。query_fn 必须与该索引建库时的切词一致。"""
        idx = R.RetrievalIndex(sqlite3.connect(str(db_path)))
        hits = 0
        rows = []
        saved = R.tokenize
        R.tokenize = query_fn  # lexical_search 读的是模块全局 tokenize
        try:
            for label in QUERY_LABELS:
                q = f"2024年合并资产负债表的{label}是多少？"
                res = idx.retrieve(q, top_k=1)
                top = res.hits[0].chunk.label if res.hits else "（无召回）"
                ok = top == label
                hits += int(ok)
                rows.append((label, top, ok))
        finally:
            R.tokenize = saved
            idx.close()
        return hits, rows

    base_hits, base_rows = eval_index(TMP_NOL_EX, build_tokens_no_lexicon)
    lex_hits, lex_rows = eval_index(TMP_LEX, real_tokenize)

    n = len(QUERY_LABELS)
    print(f"被测：{n} 个易碎科目名，问句「2024年合并资产负债表的<科目>是多少？」")
    print(f"检索：完整四路召回 + RRF 融合（RetrievalIndex.retrieve, top-1）")
    print()
    print(f"{'科目':<26}{'无词典 top-1':<28}{'有词典 top-1':<28}前后")
    print("-" * 106)
    for b, l in zip(base_rows, lex_rows):
        mark = "✓" if l[2] else ("✗" if b[2] else "·")
        print(f"{b[0]:<24}{b[1]:<26}{l[1]:<26}{mark}")
    print("-" * 106)
    print(f"无词典 top-1 命中：{base_hits}/{n}  ({base_hits / n * 100:.1f}%)")
    print(f"有词典 top-1 命中：{lex_hits}/{n}  ({lex_hits / n * 100:.1f}%)")
    print(f"净变化：{lex_hits - base_hits:+d} 条")


if __name__ == "__main__":
    main()