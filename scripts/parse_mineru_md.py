"""解析链路验证脚本：MinerU 产物 → 跨页拼接 → 会计恒等式核验。

用途不是做成正式解析器，而是回答三个问题：
  1. MinerU 的按页产物能不能稳定切成页？
  2. 跨页表格能不能被正确拼成一张逻辑表，且每行仍能指回真实页码？
  3. 拼接后的数字能不能跑通勾稽核验（证明拼接没把数字弄错位）？

用法:
    .venv/bin/python scripts/parse_mineru_md.py [产物路径]
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifin.tables import stitch_file  # noqa: E402

DEFAULT_PRODUCT = "data/parsed/moutai2024_fs"

#: 恒等式 F1 的参与科目：(报表科目名, 计算符号)
F1_TERMS: tuple[tuple[str, int], ...] = (
    ("资产总计", +1),
    ("负债合计", -1),
    ("所有者权益（或股东权益）合计", -1),
)


def verify_identity(table, column: int, label: str, unit_scale: Decimal) -> str:
    """在给定列上核验「资产 = 负债 + 所有者权益」。

    容差按工程约定推导：`科目数 × 0.5 × 报表披露单位`，**不硬编码 0.01**。
    报表以「元」披露时容差为 ±1.5 元。
    """
    values: dict[str, Decimal] = {}
    for name, _ in F1_TERMS:
        rows = table.find(name)
        if not rows:
            return f"  {label}: 缺科目「{name}」，无法核验"
        raw = rows[0].cell(column).replace(",", "").replace("，", "")
        if not raw:
            return f"  {label}: 「{name}」该列为空，无法核验"
        values[name] = Decimal(raw)

    diff = sum(values[name] * sign for name, sign in F1_TERMS)
    tolerance = Decimal(len(F1_TERMS)) * Decimal("0.5") * unit_scale
    verdict = "PASS" if abs(diff) <= tolerance else "FAIL"

    lhs = f"{values['资产总计']:,}"
    rhs = f"{values['负债合计'] + values['所有者权益（或股东权益）合计']:,}"
    return (
        f"  {label}: 资产总计 {lhs}  vs  负债+所有者权益 {rhs}\n"
        f"          差额 {diff:,}   容差 ±{tolerance}   → {verdict}"
    )


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PRODUCT)
    if not path.exists():
        print(f"产物不存在：{path}")
        print("先生成：.venv/bin/mineru parse data/pdfs/MOUTAI_2024_ANNUAL.pdf "
              "--tier standard --pages 58-67 -o " + str(path))
        return 1

    report = stitch_file(path)
    print(f"产物：{path}\n")
    print("=== 一、解析与拼接汇总 ===")
    print(report.summary())

    if report.warnings:
        print("\n--- 告警 ---")
        for w in report.warnings:
            print(f"  ! {w}")

    target = next((t for t in report.tables if t.find("资产总计")), None)
    if target is None:
        print("\n未找到含「资产总计」的逻辑表，终止。")
        return 1

    print(f"\n=== 二、合并资产负债表（{target.page_span}，已跨页拼接）===")
    print(f"表头：{' | '.join(target.header or [])}")
    print(f"行数：{len(target.rows)}   列数：{target.width}   "
          f"序列化：{'/'.join(sorted(target.serializations))}")

    print("\n--- 关键科目与来源页码（六元组的「来源」取这里）---")
    for name in ("流动资产合计", "非流动资产合计", "资产总计",
                 "流动负债合计", "负债合计",
                 "归属于母公司所有者权益（或股东权益）合计",
                 "所有者权益（或股东权益）合计",
                 "负债和所有者权益（或股东权益）总计"):
        rows = target.find(name)
        if not rows:
            print(f"  {'（未找到）':<24} {name}")
            continue
        r = rows[0]
        mark = "续页" if r.from_continuation else "首页"
        print(f"  p{r.page:<4}{mark}  {name:<24} 本期={r.cell(2):>22}  上期={r.cell(3):>22}")

    print("\n=== 三、会计恒等式核验（Decimal，容差由披露单位推导）===")
    # 报表表头声明「单位:元 币种:人民币」，故单位倍数 = 1
    for column, label in ((2, "本期"), (3, "上期")):
        print(verify_identity(target, column, label, Decimal(1)))

    out = path.parent / "moutai2024_fs_stitched.md"
    out.write_text(target.to_markdown(), encoding="utf-8")
    print(f"\n拼接结果已写出：{out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
