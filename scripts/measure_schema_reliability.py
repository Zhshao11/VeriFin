"""测量 LLM 结构化输出的字段可靠性。

为什么需要这个测量：
设计上我们依赖 ``json_schema`` 严格模式来拿到完整的六元组字段。
但实测中同一段输入、同一个模型，有时返回 7 个字段，有时只返回 5 个。
如果端点接受了 ``strict: true`` 却并未真正强制必填字段，
那「结构化输出」就不能作为字段完整性的保障，
必须由程序侧的 :func:`verifin.normalize.find_missing_fields` 兜底。

本脚本用同一份输入重复调用 N 次，统计：
- 返回全部 7 个字段的比例
- 各字段的缺失频次
- span 硬校验的通过率

这些数字直接支撑一个设计决策，也是评测报告里「结构化输出稳定性」的实测依据。

用法::

    .venv/bin/python scripts/measure_schema_reliability.py [重复次数]
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifin.llm import LLMClient, LLMError  # noqa: E402
from verifin.models import SIX_TUPLE_JSON_SCHEMA  # noqa: E402
from verifin.normalize import SIX_TUPLE_FIELDS  # noqa: E402
from verifin.span import verify_evidence  # noqa: E402

#: 与 demo_core.py 一致的合成片段，保证结果可复现。
CHUNK = """合并资产负债表（续）
编制单位：示例科技股份有限公司        2023 年 12 月 31 日        单位：元

项目                          期末余额              期初余额
资产总计                  4,000,000,000.00      3,600,000,000.00
负债合计                  2,000,000,000.00      1,900,000,000.00
所有者权益合计            2,000,000,000.00      1,700,000,000.00
"""

SYSTEM = (
    "你是财务数据抽取器。只做抽取，不做任何计算与推断。\n"
    "铁律：source_span 必须是原文中逐字出现的连续片段。\n"
    "输出必须包含全部字段，即使某个字段你无法确定，也要给出该字段。"
)

#: 六元组字段名到 JSON schema 字段名的映射。
SCHEMA_FIELDS = tuple(SIX_TUPLE_JSON_SCHEMA["required"])  # type: ignore[arg-type]

DEFAULT_ROUNDS = 12


def ask(client: LLMClient, metric: str) -> dict[str, object]:
    return client.chat_json(
        system=SYSTEM,
        user=(
            "公司：示例科技股份有限公司\n报告期：2023-12-31\n来源页码：42\n"
            f"需要抽取的科目：{metric}\n\n原文：\n{CHUNK}"
        ),
        schema={"type": "object", "additionalProperties": False,
                "required": ["items"],
                "properties": {"items": {"type": "array", "items": SIX_TUPLE_JSON_SCHEMA}}},
        schema_name="evidence_extraction",
        purpose="measure",
    )


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROUNDS
    try:
        client = LLMClient.from_env()
    except LLMError as exc:
        print(f"无法初始化 LLM 客户端：{exc}", file=sys.stderr)
        return 1

    print(f"模型：{client.model}   重复次数：{rounds}\n")

    complete = 0
    missing_counter: Counter[str] = Counter()
    span_ok = 0
    span_attempted = 0
    parse_failures = 0

    for i in range(1, rounds + 1):
        try:
            payload = ask(client, "资产总计、负债合计、所有者权益合计")
        except LLMError as exc:
            parse_failures += 1
            print(f"  第 {i:>2} 次：调用返回异常 —— {str(exc)[:70]}")
            continue

        items = payload.get("items") or []
        if not isinstance(items, list) or not items:
            parse_failures += 1
            print(f"  第 {i:>2} 次：items 为空或类型错误")
            continue

        absent = [f for f in SCHEMA_FIELDS if f not in items[0]]
        for f in absent:
            missing_counter[f] += 1
        if not absent:
            complete += 1

        # 对每条证据做 span 校验，统计通过率
        line_ok = 0
        for item in items:
            span = item.get("source_span")
            raw_value = item.get("value")
            if not span or raw_value is None:
                continue
            span_attempted += 1
            try:
                from decimal import Decimal
                value = Decimal(str(raw_value).replace(",", "").split()[0])
            except Exception:  # noqa: BLE001
                continue
            verdict = verify_evidence(
                claimed_span=span,
                chunk_text=CHUNK,
                claimed_value=value,
                claimed_unit=item.get("unit"),
            )
            if verdict.accepted:
                span_ok += 1
                line_ok += 1

        status = "字段齐全" if not absent else f"缺 {len(absent)} 项：{'、'.join(absent)}"
        print(f"  第 {i:>2} 次：{len(items)} 条证据，{line_ok} 条通过 span 校验，{status}")

    print("\n" + "=" * 64)
    print("测量结果")
    print("=" * 64)
    print(f"成功返回次数：{rounds - parse_failures} / {rounds}")
    print(f"字段齐全次数：{complete} / {rounds - parse_failures}")
    print(f"span 校验：{span_ok} 通过 / {span_attempted} 尝试")
    if missing_counter:
        print("\n各字段缺失频次：")
        for field, count in missing_counter.most_common():
            print(f"  {field:<14} 缺 {count} 次")
    else:
        print("\n无字段缺失。")

    print("\n结论：")
    if complete < rounds - parse_failures:
        print(
            "  端点接受了 json_schema 的 strict 参数，但**并未真正强制必填字段**。\n"
            "  因此「结构化输出」不能作为字段完整性的保障，\n"
            "  必须由程序侧的 find_missing_fields 兜底并在缺字段时拒答。"
        )
    else:
        print("  本批次未见字段缺失。样本量有限，不宜据此认定端点强制了 strict 模式。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
