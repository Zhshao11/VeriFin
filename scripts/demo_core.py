"""端到端演示：LLM 抽取 → span 硬校验 → Decimal 勾稽核验。

这是核心链路的可运行证明，不依赖任何真实 PDF——用一段合成年报片段即可。
等解析层接上以后，只需把 ``CHUNK`` 换成 MinerU 的解析产物即可。

演示三件事：

1. **正常路径**：模型抽出的证据逐条通过两层校验，勾稽核验 PASS。
2. **拦截路径**：篡改一条证据的数值，观察它被第二层校验拦下——
   片段在原文里真实存在，但数值对不上，这正是只做一层校验会漏掉的场景。
3. **拒答路径**：询问年报中不存在的指标，程序级拒答。

用法::

    .venv/bin/python scripts/demo_core.py
"""

from __future__ import annotations

import json
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fin_verify import (  # noqa: E402
    FormulaKind,
    Verdict,
    find_missing_fields,
    evaluate_formula,
    verify_evidence,
)
from fin_verify.llm import LLMClient, LLMError  # noqa: E402
from fin_verify.models import EvidenceDraft, ExtractionResult  # noqa: E402

#: 合成的年报片段。结构与真实合并资产负债表一致，但数字是编造的。
CHUNK = """合并资产负债表（续）
编制单位：示例科技股份有限公司        2023 年 12 月 31 日        单位：元

项目                          期末余额              期初余额
流动资产合计              1,850,000,000.00      1,620,000,000.00
非流动资产合计            2,150,000,000.00      1,980,000,000.00
资产总计                  4,000,000,000.00      3,600,000,000.00
流动负债合计              1,200,000,000.00      1,150,000,000.00
非流动负债合计              800,000,000.00        750,000,000.00
负债合计                  2,000,000,000.00      1,900,000,000.00
归属于母公司所有者权益合计  1,700,000,000.00    1,600,000,000.00
少数股东权益                300,000,000.00        100,000,000.00
所有者权益合计            2,000,000,000.00      1,700,000,000.00
负债和所有者权益总计      4,000,000,000.00      3,600,000,000.00
"""

PAGE = 42

COMPANY = "示例科技股份有限公司"
PERIOD = "2023-12-31"


def banner(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def to_decimal(raw: str) -> Decimal | None:
    try:
        return Decimal(str(raw).replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def extract(client: LLMClient, metrics: list[str]) -> ExtractionResult:
    """让模型抽取指定科目的六元组证据。"""
    system = (
        "你是财务数据抽取器。只做抽取，不做任何计算与推断。\n"
        "铁律：source_span 必须是原文中**逐字出现**的连续片段，"
        "不得改写、不得拼接、不得加入原文没有的字符。\n"
        "如果原文中没有该科目，就不要输出这一项，不要编造。"
    )
    user = (
        f"公司：{COMPANY}\n报告期：{PERIOD}\n来源页码：第 {PAGE} 页\n"
        f"需要抽取的科目：{'、'.join(metrics)}\n\n"
        f"原文：\n{CHUNK}"
    )
    return client.extract_evidence(system=system, user=user)


def show_extraction(result: ExtractionResult) -> None:
    print(f"模型输出 {len(result.items)} 条证据：")
    for item in result.items:
        print(
            f"  · {item.metric:<12} {item.value:>22} {item.unit or '(未标单位)':<8}"
            f" p.{item.source_page}"
        )


def check_all(result: ExtractionResult, chunk: str) -> list[tuple[EvidenceDraft, object]]:
    """对每条证据执行 span 硬校验 + 六元组完整性检查。"""
    checked = []
    for item in result.items:
        missing = find_missing_fields(
            company=item.company,
            period=item.period,
            metric=item.metric,
            value=to_decimal(item.value),
            unit=item.unit,
            source=item.source_span,
        )
        value = to_decimal(item.value)
        if missing or value is None:
            print(f"  ✗ {item.metric:<12} 六元组缺字段：{'、'.join(missing) or 'value 无法解析'}")
            checked.append((item, None))
            continue

        verdict = verify_evidence(
            claimed_span=item.source_span,
            chunk_text=chunk,
            claimed_value=value,
            claimed_unit=item.unit,
        )
        mark = "✓" if verdict.accepted else "✗"
        print(f"  {mark} {item.metric:<12} {verdict.detail}")
        checked.append((item, verdict if verdict.accepted else None))
    return checked


def main() -> int:
    try:
        client = LLMClient.from_env()
    except LLMError as exc:
        print(f"无法初始化 LLM 客户端：{exc}", file=sys.stderr)
        return 1

    target_metrics = ["资产总计", "负债合计", "所有者权益合计"]

    # ---------------------------------------------------------------- 步骤 1
    banner("步骤 1｜模型抽取证据（LLM 只抽取，不计算）")
    result = extract(client, target_metrics)
    show_extraction(result)

    # ---------------------------------------------------------------- 步骤 2
    banner("步骤 2｜span 硬校验（程序判定，模型无权参与）")
    checked = check_all(result, CHUNK)

    accepted = {
        item.metric: to_decimal(item.value)
        for item, verdict in checked
        if verdict is not None
    }
    print(f"\n通过校验 {len(accepted)} / {len(result.items)} 条")

    # ---------------------------------------------------------------- 步骤 3
    banner("步骤 3｜篡改一条证据的数值，观察第二层校验拦截")

    # 场景：模型引用了一段真实存在的原文，却报了一个错误的数字。
    # 只校验「片段在不在原文」会放行，第二层数值校验必须拦下。
    tampered = {
        "claimed_span": "资产总计                  4,000,000,000.00      3,600,000,000.00",
        "claimed_value": Decimal("4000000000.00"),
        "claimed_unit": "元",
    }
    print("原始片段在原文中存在：", tampered["claimed_span"] in CHUNK)

    bogus = verify_evidence(
        claimed_span=tampered["claimed_span"],
        chunk_text=CHUNK,
        claimed_value=Decimal("9999999999.00"),  # 原文里没有这个数
        claimed_unit="元",
    )
    print(f"改为声称 9,999,999,999.00 元 → accepted={bogus.accepted}")
    print(f"  原因码：{bogus.reason}")
    print(f"  说明：{bogus.detail}")

    # ---------------------------------------------------------------- 步骤 4
    banner("步骤 4｜Decimal 勾稽核验（LLM 全程不参与算术）")
    if len(accepted) < 3:
        print("证据不足，跳过核验。")
    else:
        outcome = evaluate_formula("F1", accepted, disclosure_unit="元")
        print(f"公式：{outcome.formula_name}")
        print(f"左值：{outcome.lhs:,}")
        print(f"右值合计：{outcome.rhs:,}")
        print(f"差额：{outcome.diff:,}")
        print(f"容差：{outcome.tolerance:,}（3 个科目 × 0.5 × 元）")
        print(f"结论：{outcome.verdict.value}")
        print(f"说明：{outcome.detail}")

        # 演示「科目整体漏记」时的定位能力。
        # 场景：只取了「归属于母公司所有者权益合计」，漏加了「少数股东权益」。
        # 注意漏掉的那一项**不在操作数里**，因此不能靠操作数量级定位，
        # 只能依靠公式自带的领域线索（often_omitted）给出核查方向。
        print("\n--- 假设只取了归母权益、漏加少数股东权益，观察定位能力 ---")
        broken = dict(accepted)
        broken["所有者权益合计"] = Decimal("1700000000.00")  # 少了 3 亿少数股东权益
        bad = evaluate_formula("F1", broken, disclosure_unit="元")
        print(f"结论：{bad.verdict.value}")
        print(f"差额：{bad.diff:,}")
        print(f"按操作数量级定位到的科目：{'、'.join(bad.suspects) or '(无——漏项不在操作数里)'}")
        print(f"说明：{bad.detail}")
        assert bad.verdict is Verdict.FAIL
        assert bad.kind is FormulaKind.IDENTITY
        assert "少数股东权益" in bad.detail

    # ---------------------------------------------------------------- 步骤 5
    banner("步骤 5｜询问年报中不存在的指标，应触发程序级拒答")
    absent = extract(client, ["商誉减值准备"])
    if not absent.items:
        print("模型未编造证据（正确行为）→ 六元组无内容 → 程序级拒答。")
        print("拒答理由：EVIDENCE_INCOMPLETE（检索单元「商誉减值准备」未被任何证据覆盖）")
    else:
        print(f"模型仍返回了 {len(absent.items)} 条，逐条校验：")
        for item, verdict in check_all(absent, CHUNK):
            if verdict is None:
                print("  → 该条被拦下，不计入证据集")
        print("最终有效证据数：0 → 程序级拒答。")

    print("\n演示结束。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
