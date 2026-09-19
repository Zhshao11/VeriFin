"""解析 → 拼接 → 检索 → 坐标定位 全链路演示。

这一条链跑通，就意味着六元组的每一项都有了确定性来源：

    公司 / 期间  ← 报告标识与表头
    指标        ← 检索层的 label 通道（科目名精确匹配）
    数值 / 单位  ← 检索层的 numeric 通道 + 归一化
    来源        ← 页码（拼接模块保留）+ 坐标（PyMuPDF 坐标层）

用法:
    .venv/bin/python scripts/demo_evidence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifin.geometry import open_pdf  # noqa: E402
from verifin.retrieval import build_chunks, create_index, load_index  # noqa: E402
from verifin.tables import stitch_file  # noqa: E402

PDF = ROOT / "data/pdfs/MOUTAI_2024_ANNUAL.pdf"
PRODUCT = ROOT / "data/parsed/moutai2024_fs"
INDEX = ROOT / "data/index/moutai2024.db"
OUT_DIR = ROOT / "data/parsed"

QUERIES = [
    "2024年资产总计是多少",
    "负债合计",
    "2024年营业收入是多少",
    "242,011,315,120.60",
]


def main() -> int:
    if not PRODUCT.exists():
        print(f"缺解析产物：{PRODUCT}")
        return 1

    print("=== 一、解析与拼接 ===")
    report = stitch_file(PRODUCT)
    print(report.summary())

    print("\n=== 二、建检索索引 ===")
    chunks = build_chunks(report.tables, "MOUTAI_2024")
    con = create_index(INDEX, chunks)
    con.close()
    idx = load_index(INDEX)
    print(f"chunk 数 {len(chunks)}   向量实现 {idx.embedder_name}")

    print("\n=== 三、检索 → 坐标定位 ===")
    with open_pdf(PDF) as geo:
        boxes_by_page: dict[int, list] = {}
        for query in QUERIES:
            result = idx.retrieve(query, top_k=1)
            print(f"\n  查询：{query}")
            print(f"    各通道命中：{result.per_channel}")
            if result.is_empty:
                print("    → 无结果（本系统应在此拒答，不得编造）")
                continue
            hit = result.hits[0]
            value = hit.chunk.values[0] if hit.chunk.values else None
            print(f"    命中：p{hit.chunk.page}  {hit.chunk.label}  = {value}")
            print(f"    命中通道：{'+'.join(hit.via)}")

            if value:
                loc = geo.locate_row(hit.chunk.page, hit.chunk.label, value)
                if loc.same_row_verified:
                    row_box = loc.row_box
                    print(f"    坐标（PDF point）：{tuple(round(v, 1) for v in row_box.astuple())}")
                    boxes_by_page.setdefault(hit.chunk.page, []).append(row_box)
                else:
                    print("    坐标定位未通过同行校验 → 证据不采纳（拒答路径）")

        if boxes_by_page:
            print("\n=== 四、导出高亮图（人工复核用）===")
            for page, boxes in sorted(boxes_by_page.items()):
                out = geo.render_highlight(
                    page, boxes, OUT_DIR / f"evidence_p{page}.png"
                )
                print(f"    p{page} 高亮 {len(boxes)} 处 → {out.relative_to(ROOT)}")

    idx.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
