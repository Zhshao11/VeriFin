"""Agent 编排（D3）演示：真实年报 + 真实检索层 + 真实坐标层。

用法:
    .venv/bin/python scripts/demo_agent.py            # 确定性策略（离线可跑）
    .venv/bin/python scripts/demo_agent.py --llm      # 接 LLM 调度（需 .env）

不加 `--llm` 时用的是确定性兜底策略 —— 这本身就是设计的一部分：
核心链路必须能在没有模型的情况下跑通，否则无法证明「数值不来自模型」。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from verifin.agent import (  # noqa: E402
    Budget,
    ToolRuntime,
    TraceStore,
    VeriFinAgent,
    render_graph_text,
)
from verifin.agent.planner import LLMPlanner  # noqa: E402
from verifin.geometry import open_pdf  # noqa: E402
from verifin.runtime import build_document_runtime  # noqa: E402

PDF = ROOT / "data/pdfs/MOUTAI_2024_ANNUAL.pdf"
PRODUCT = ROOT / "data/parsed/moutai2024_fs"
STITCHED = ROOT / "data/parsed/moutai2024_fs_stitched.md"
INDEX_DB = ROOT / "data/index/moutai2024.db"
TRACE_DB = ROOT / "data/index/agent_trace.db"

COMPANY = "贵州茅台酒股份有限公司（600519）"
PERIOD = "2024 年度（2024-01-01 ~ 2024-12-31）"
#: 公司/期间目前取自演示配置 —— 从 PDF 封面做结构化解析尚未接入。
#: 明写在这里是为了**不让它看起来像从文档里读出来的**：
#: 六元组的前两个字段必须有明确来源，来源是"配置"就得说"配置"。
IDENTITY_SOURCE = "演示配置（封面结构化解析未接入）"

QUESTIONS = [
    "2024年营业收入是多少",
    "负债合计",
    "242,011,315,120.60",
    "资产是否等于负债加所有者权益",
    "火星基地折旧年限",
]


def build_runtime() -> tuple[ToolRuntime, str]:
    """装配真实年报的运行时。

    装配本身在 `verifin.runtime` 里，与网页端、评测执行器共用同一份 ——
    否则三处各自漂移，评测量到的就不是演示里跑的那个系统。
    """
    doc = build_document_runtime(
        doc_id="MOUTAI_2024",
        product=PRODUCT,
        index_db=INDEX_DB,
        pdf=PDF,
        stitched=STITCHED,
    )
    runtime = doc.tool_runtime(
        company=COMPANY,
        period=PERIOD,
        pdf_open=lambda: open_pdf(PDF),
    )
    return runtime, doc.unit_source


def print_result(result, unit_source: str) -> None:
    print(f"\n  问题：{result.question}")
    print(f"  路线：{result.route}   预算：步 {len(result.steps)} / "
          f"工具 {result.tool_calls} / LLM {result.llm_calls}")
    for step in result.steps:
        mark = {True: "✓", False: "✗", None: "·"}[step.ok]
        tool = f" → {step.tool}" if step.tool else ""
        print(f"    {step.seq}. [{mark}] {step.node:<14}{tool:<20} {step.detail[:52]}")
    if result.decision == "ANSWER":
        print("  ── 结论：可答 ─────────────────────────")
        if result.route == "VERIFY":
            a = result.answer
            print(f"    {a['公式']}")
            print(f"    结论 {a['结论']}   差额 {a['差额']}   容差 ±{a['容差']}")
        else:
            for k, v in result.answer["six_tuple"].items():
                print(f"    {k:<6}{v}")
            print(f"    片段：{result.answer['fragment']}")
            print(f"    指标/数值/单位/来源 来源：检索层 + 表头推导（非模型）")
            print(f"    单位来源：{unit_source}")
            print(f"    公司/期间来源：{IDENTITY_SOURCE}")
    else:
        print(f"  ── 结论：{result.decision} ─────────────────────────")
        print(f"    原因 {result.refusal['reason']}")
        print(f"    {result.refusal['detail']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="VeriFin Agent 编排演示")
    parser.add_argument("--llm", action="store_true", help="启用 LLM 调度（需要 .env）")
    parser.add_argument("--question", action="append", help="追加自定义问题")
    args = parser.parse_args()

    print("=== 一、显式图结构 ===")
    print(render_graph_text())

    runtime, unit_source = build_runtime()
    print(f"\n索引装载完成：科目 {len(runtime.by_label)} 条，披露单位 {runtime.unit}"
          f"（来源：{unit_source}）")

    planner = None
    use_llm = False
    if args.llm:
        try:
            from verifin.llm import LLMClient

            planner = LLMPlanner(LLMClient.from_env())
            use_llm = True
            print("LLM 调度：已启用")
        except Exception as exc:  # noqa: BLE001
            print(f"LLM 调度：不可用（{type(exc).__name__}: {exc}）→ 使用确定性策略")
    else:
        print("LLM 调度：未启用（默认），走确定性策略")

    trace = TraceStore(TRACE_DB)
    agent = VeriFinAgent(
        runtime, budget=Budget(), planner=planner, trace=trace, use_llm=use_llm
    )

    questions = QUESTIONS + (args.question or [])
    print("\n=== 二、逐题运行 ===")
    for q in questions:
        print_result(agent.run(q), unit_source)

    print("\n=== 三、轨迹入库 ===")
    print(f"  库：{TRACE_DB.relative_to(ROOT)}   运行数：{len(trace.runs())}")
    latest = trace.runs()[0] if trace.runs() else None
    if latest is not None:
        print(f"  最近一次：{latest['run_id']}  决策 {latest['decision']}  "
              f"步数 {latest['steps']}  调度来源 {latest['planner_source']}")
        print(trace.abbreviated(latest["run_id"]))
    trace.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
