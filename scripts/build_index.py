"""重建检索索引（SQLite）。

**为什么要有这个独立脚本。**

索引不是"建一次就完事"的产物：chunk 的结构一变（例如新增报表口径字段），
旧库就与代码不再兼容，而**不兼容不会报错** —— 旧库照样能读，
只是读出来的 chunk 缺字段，于是口径消歧静默失效。
所以重建必须是一条显式、可复现的命令，写进 README，而不是靠"记得重跑某个演示脚本"。

用法::

    .venv/bin/python scripts/build_index.py
    .venv/bin/python scripts/build_index.py --product data/parsed/moutai2024_fs \\
        --db data/index/moutai2024.db --doc MOUTAI_2024
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifin.retrieval import build_chunks, create_index, load_index  # noqa: E402
from verifin.tables import stitch_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="重建 VeriFin 检索索引")
    parser.add_argument("--product", default=str(ROOT / "data/parsed/moutai2024_fs"))
    parser.add_argument("--db", default=str(ROOT / "data/index/moutai2024.db"))
    parser.add_argument("--doc", default="MOUTAI_2024")
    args = parser.parse_args()

    product = Path(args.product)
    db = Path(args.db)
    if not product.exists():
        raise SystemExit(f"解析产物不存在：{product}（先跑解析）")

    report = stitch_file(product)
    chunks = build_chunks(report.tables, args.doc)

    db.parent.mkdir(parents=True, exist_ok=True)
    con = create_index(db, chunks)
    con.close()

    print(f"索引已重建：{db}")
    print(f"  逻辑表 {len(report.tables)} 张，chunk {len(chunks)} 条")

    # 口径分布必须打出来核对。**它是本脚本存在的第二个理由**：
    # 口径若全部是 None（标题没被捕获），口径消歧会静默失效 ——
    # 不打印分布，这件事没人会发现。
    by_scope = Counter(c.scope for c in chunks)
    print("  口径分布：" + "  ".join(
        f"{scope or '未标注'} {n}" for scope, n in sorted(
            by_scope.items(), key=lambda kv: (kv[0] is None, kv[0] or "")
        )
    ))
    for table in report.tables:
        print(f"    {table.caption or '（无标题）':<20} 口径={table.scope or '未标注':<6} "
              f"行={len(table.rows):<4} 页={table.page_start}-{table.page_end}")

    idx = load_index(db)
    print(f"  向量实现：{idx.embedder_name}")
    idx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
