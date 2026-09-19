"""评测执行器：把题库逐题喂给 Agent，再与 gold 逐项比对。

用法::

    .venv/bin/python -m verifin.eval.run_eval                      # 全部题目，确定性策略
    .venv/bin/python -m verifin.eval.run_eval --tier L1            # 只看某一层
    .venv/bin/python -m verifin.eval.run_eval --planner llm        # 接 LLM 调度（需 .env）
    .venv/bin/python -m verifin.eval.run_eval --json out.json      # 同时落机器可读结果

**判分口径为什么这么细。**

1. **数值与页码分开报。** 数值对、页码错，和数值错是两种故障：
   前者是证据链的问题，后者是取数的问题。合成一个「正确/错误」会把故障类型丢掉。

2. **判定型题先比操作数、再比判定。** 合并与母公司两套报表的资产恒等式**都能平衡**
   （差额都是 0.00）。只比 `PASS` 的话，用母公司数据算合并恒等式也会被判成对 ——
   这正是本项目最想防的错误。

3. **`ABORT` 不算拒答成功。** `ABORT` 是「没算完」，`REFUSE` 是「证据不足」。
   把前者算成后者，会把规划器缺陷伪装成护栏生效。

4. **降级率必须从轨迹里读，不能凭印象。** 每一步都带 `source` 字段
   （`policy` / `llm`）。不读它就会重犯「看起来 LLM 没效果、其实它压根没被调用」的错。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifin.agent import Budget, ToolRuntime, VeriFinAgent  # noqa: E402
from verifin.agent.graph import RunResult  # noqa: E402
from verifin.eval.bank import load_bank  # noqa: E402
from verifin.eval.docs import get_spec  # noqa: E402
from verifin.eval.schema import QUESTION_TYPE_LABELS, EvalItem, is_verdict_formula  # noqa: E402
from verifin.geometry import open_pdf  # noqa: E402
from verifin.runtime import build_document_runtime  # noqa: E402

DEFAULT_BANK = ROOT / "data" / "eval" / "b_moutai_2024.jsonl"

#: 公司 / 期间目前只有演示配置一个来源（封面结构化解析未接入）。
#: 它们是六元组的前两个字段，来源是「配置」就得标「配置」。
DEMO_COMPANY = "贵州茅台酒股份有限公司（600519）"
DEMO_PERIOD = "2024 年度"
IDENTITY_SOURCE = "演示配置（封面结构化解析未接入）"


# ---------------------------------------------------------------------------
# 结果模型
# ---------------------------------------------------------------------------


@dataclass
class ItemResult:
    item_id: str
    question: str
    question_type: str
    scope: str
    expected: str
    decision: str
    route: str
    passed: bool
    failure: str = ""
    """未通过的原因分类（空字符串表示通过）。"""
    detail: str = ""
    got_value: str | None = None
    got_pages: tuple[int, ...] = ()
    value_ok: bool | None = None
    page_ok: bool | None = None
    llm_steps: int = 0
    tool_calls: int = 0
    steps: int = 0
    run_id: str = ""
    latency_ms: float = 0.0
    """端到端耗时（毫秒），含检索 / 校验 / 坐标全部动作。"""
    evidence_recalled: bool = False
    """检索层是否真的召回了候选证据（SEARCH 命中非空 / DIFF 取数成功）。"""

    @property
    def degraded(self) -> bool:
        """这次运行里有没有一步是兜底策略决定的（即 LLM 没参与那一步）。"""
        return self.steps > 0 and self.llm_steps < self.steps

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "question": self.question,
            "question_type": self.question_type,
            "scope": self.scope,
            "expected": self.expected,
            "decision": self.decision,
            "route": self.route,
            "passed": self.passed,
            "failure": self.failure,
            "detail": self.detail,
            "got_value": self.got_value,
            "got_pages": list(self.got_pages),
            "value_ok": self.value_ok,
            "page_ok": self.page_ok,
            "llm_steps": self.llm_steps,
            "tool_calls": self.tool_calls,
            "steps": self.steps,
            "run_id": self.run_id,
            "latency_ms": round(self.latency_ms, 1),
            "evidence_recalled": self.evidence_recalled,
        }


# ---------------------------------------------------------------------------
# 比对
# ---------------------------------------------------------------------------


def _to_decimal(raw: str | None) -> Decimal | None:
    """把披露原样的数字串转成 Decimal。**只做形式清洗，不做任何猜测。**"""
    if raw is None:
        return None
    cleaned = str(raw).replace(",", "").replace("，", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _same_number(a: str | None, b: str | None) -> bool:
    da, db = _to_decimal(a), _to_decimal(b)
    return da is not None and db is not None and da == db


def _system_value(result: RunResult) -> str | None:
    """从运行结果里取出「系统给出的那个数」。

    两条路线取的位置不同：核验路线的答案是判定的左值，查询路线的答案是六元组里的数值。
    这是件麻烦事，但必须在这里收口 —— 如果让每个调用方各取一次，
    早晚会有人从错的位置取到数，然后以为系统算错了。
    """
    if result.decision != "ANSWER" or not result.answer:
        return None
    answer = result.answer
    if result.route == "VERIFY":
        lhs = answer.get("左值")
        return None if lhs in (None, "None") else str(lhs)
    six = answer.get("six_tuple") or {}
    return six.get("数值")


def compare(item: EvalItem, result: RunResult) -> ItemResult:
    """把一次运行与 gold 比对，返回结构化的判定。"""
    llm_steps = sum(1 for s in result.steps if s.source == "llm")
    got_value = _system_value(result)

    res = ItemResult(
        item_id=item.item_id,
        question=item.question,
        question_type=item.question_type,
        scope=item.scope,
        expected=item.expected_outcome,
        decision=result.decision,
        route=result.route,
        passed=False,
        got_value=got_value,
        llm_steps=llm_steps,
        tool_calls=result.tool_calls,
        steps=len(result.steps),
        run_id=result.run_id,
    )

    # --------------------------------------------------- 应拒答题
    if item.expected_outcome == "REFUSE":
        if result.decision == "REFUSE":
            res.passed = True
            res.detail = (result.refusal or {}).get("reason", "")
        elif result.decision == "ABORT":
            # 「没算完」不等于「证据不足」，单独记，不算拒答成功。
            res.failure = "ABORT"
            res.detail = "预算耗尽，未跑到拒答 —— ABORT 不能算作正确拒答"
        else:
            res.failure = "误答（本该拒答）"
            res.detail = f"系统给出 {got_value!r}"
        return res

    # --------------------------------------------------- 可答题：先看结局
    if result.decision != "ANSWER":
        res.failure = f"未作答（{result.decision}）"
        res.detail = (result.refusal or {}).get("reason", "")
        return res

    # 判分优先级：判定型 → 操作数 + 判定；数值型 → 数值。
    if is_verdict_formula(item) and item.gold_verdict:
        got_verdict = str((result.answer or {}).get("结论"))
        operands = (result.answer or {}).get("操作数") or {}

        bad = [
            name
            for name, gold in item.gold_operands.items()
            if not _same_number(operands.get(name), gold)
        ]
        if bad:
            res.failure = "操作数取错"
            res.detail = "；".join(
                f"{n}: 期望 {item.gold_operands[n]}，实得 {operands.get(n)}" for n in bad
            )
            return res
        if got_verdict != item.gold_verdict:
            res.failure = "判定错"
            res.detail = f"期望 {item.gold_verdict}，实得 {got_verdict}"
            return res
        res.passed = True
        res.value_ok = True
        res.detail = f"{item.formula} {got_verdict}"
        return res

    # 数值型：先比数值，再（独立地）比页码。
    res.value_ok = _same_number(got_value, item.gold_value)
    got_pages = _pages_of(result, got_value)
    res.got_pages = got_pages
    res.page_ok = bool(set(got_pages) & set(item.gold_pages)) if got_pages else False

    if not res.value_ok:
        res.failure = "数值错"
        res.detail = f"期望 {item.gold_value}，实得 {got_value}"
        return res
    if not res.page_ok:
        # 数值对、页码错：证据链与取数是两回事，单独分类。
        res.failure = "页码不符"
        res.detail = f"期望 p{'、'.join(map(str, item.gold_pages))}，实得 p{'、'.join(map(str, got_pages))}"
        return res

    res.passed = True
    res.detail = f"{item.gold_value} {item.gold_unit} @ p{'、'.join(map(str, item.gold_pages))}"
    return res


def _pages_of(result: RunResult, got_value: str | None) -> tuple[int, ...]:
    """从运行结果里取出证据所在页码。

    查询路线从六元组的「来源」字段里解（形如 `第 61 页 · 坐标 (...)`），
    核验路线取各操作数来源页的并集。
    """
    if result.decision != "ANSWER" or not result.answer:
        return ()
    answer = result.answer
    if result.route == "VERIFY":
        pages: set[int] = set()
        for src in (answer.get("来源") or []):
            page = (src or {}).get("页码") if isinstance(src, dict) else None
            if isinstance(page, int) and page > 0:
                pages.add(page)
        return tuple(sorted(pages))

    if result.route == "DIFF":
        # 两行可能不在同一页，取证据所在页的并集（取数与溯源是两回事）。
        pages: set[int] = set()
        for op in (answer.get("operands") or []):
            page = (op or {}).get("页码")
            if isinstance(page, int) and page > 0:
                pages.add(page)
        return tuple(sorted(pages))

    source = ((answer.get("six_tuple") or {}).get("来源") or "")
    digits = ""
    out: list[int] = []
    for ch in str(source):
        if ch.isdigit():
            digits += ch
        elif digits:
            out.append(int(digits))
            digits = ""
    if digits:
        out.append(int(digits))
    return tuple(out[:1])


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------


@dataclass
class EvalReport:
    results: list[ItemResult] = field(default_factory=list)
    planner: str = "policy"

    def _group(self, key) -> dict[str, tuple[int, int]]:
        agg: dict[str, list[int]] = {}
        for r in self.results:
            k = key(r)
            agg.setdefault(k, [0, 0])
            agg[k][1] += 1
            agg[k][0] += int(r.passed)
        return {k: (v[0], v[1]) for k, v in agg.items()}

    @property
    def by_type(self) -> dict[str, tuple[int, int]]:
        return self._group(lambda r: r.question_type)

    @property
    def by_scope(self) -> dict[str, tuple[int, int]]:
        return self._group(lambda r: r.scope)

    def stats(self) -> dict[str, Any]:
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        answerable = [r for r in self.results if r.expected == "ANSWER"]
        refuse = [r for r in self.results if r.expected == "REFUSE"]
        aborts = [r for r in self.results if r.decision == "ABORT"]
        failures = Counter(r.failure for r in self.results if not r.passed)

        llm_steps = sum(r.llm_steps for r in self.results)
        all_steps = sum(r.steps for r in self.results)
        degraded = [r for r in self.results if r.degraded]

        latencies = sorted(r.latency_ms for r in self.results)
        recalled_answerable = [
            r for r in answerable if r.evidence_recalled
        ]

        def _pctl(xs: list[float], p: float) -> float:
            if not xs:
                return 0.0
            k = max(0, min(len(xs) - 1, int(round((len(xs) - 1) * p))))
            return round(xs[k], 1)

        return {
            "调度器": self.planner,
            "总题数": total,
            "通过": passed,
            "通过率": round(passed / total, 4) if total else 0.0,
            "可答题": len(answerable),
            "可答题通过": sum(1 for r in answerable if r.passed),
            "应拒答": len(refuse),
            "正确拒答": sum(1 for r in refuse if r.passed),
            "误答": sum(1 for r in refuse if r.failure.startswith("误答")),
            "ABORT": len(aborts),
            "按题型": {k: f"{v[0]}/{v[1]}" for k, v in sorted(self.by_type.items())},
            "按口径": {k: f"{v[0]}/{v[1]}" for k, v in sorted(self.by_scope.items())},
            "失败分类": dict(failures),
            "LLM 步数": llm_steps,
            "总步数": all_steps,
            "LLM 步占比": round(llm_steps / all_steps, 4) if all_steps else 0.0,
            "含兜底步的运行数": len(degraded),
            "端到端耗时P50_ms": _pctl(latencies, 0.50),
            "端到端耗时P95_ms": _pctl(latencies, 0.95),
            "证据召回率": (
                f"{len(recalled_answerable)}/{len(answerable)}"
                if answerable else "—"
            ),
        }

    def render(self, *, show_all: bool = False) -> str:
        lines = ["# 逐题结果", ""]
        for r in self.results:
            if not show_all and r.passed:
                continue
            flag = "✓" if r.passed else "✗"
            lines.append(
                f"  [{flag}] {r.item_id} {r.question_type} {r.scope} "
                f"→ {r.decision}/{r.route}"
            )
            if not r.passed:
                lines.append(f"        {r.failure}：{r.detail}")
            else:
                lines.append(f"        {r.detail}")
        if not show_all and all(r.passed for r in self.results):
            lines.append("  （全部通过）")
        return "\n".join(lines)

    def render_stats(self) -> str:
        s = self.stats()
        lines = [
            f"调度器：{s['调度器']}",
            f"通过 {s['通过']}/{s['总题数']}（{s['通过率']:.1%}）",
            f"可答题 {s['可答题通过']}/{s['可答题']}    "
            f"应拒答正确 {s['正确拒答']}/{s['应拒答']}（误答 {s['误答']}）    "
            f"ABORT {s['ABORT']}",
            "按题型：" + "  ".join(
                f"{QUESTION_TYPE_LABELS.get(k, k)} {v}" for k, v in s["按题型"].items()
            ),
            "按口径：" + "  ".join(f"{k} {v}" for k, v in s["按口径"].items()),
            "失败分类：" + ("；".join(f"{k} {v}" for k, v in s["失败分类"].items()) or "（无）"),
            f"LLM 步 {s['LLM 步数']}/{s['总步数']}（{s['LLM 步占比']:.1%}）"
            f"，含兜底步的运行 {s['含兜底步的运行数']} 次",
            f"端到端耗时 P50/P95：{s['端到端耗时P50_ms']} / {s['端到端耗时P95_ms']} ms",
            f"证据召回率：{s['证据召回率']}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 驱动
# ---------------------------------------------------------------------------


def run_bank(
    items: list[EvalItem],
    *,
    doc_key: str,
    planner: str = "policy",
    llm_planner: Any | None = None,
) -> EvalReport:
    """跑整个题库。

    `llm_planner` 为 None 时走确定性策略 —— **这不是降级，是设计**：
    核心链路必须能在没有模型的情况下跑通，否则「数值不来自模型」无法被证明。
    """
    spec = get_spec(doc_key)
    doc = build_document_runtime(
        doc_id=doc_key,
        product=ROOT / "data" / "parsed" / "moutai2024_fs",
        index_db=spec.index_db,  # type: ignore[arg-type]
        pdf=spec.pdf,
        stitched=ROOT / "data" / "parsed" / "moutai2024_fs_stitched.md",
    )
    runtime = doc.tool_runtime(
        company=DEMO_COMPANY,
        period=DEMO_PERIOD,
        pdf_open=lambda: open_pdf(spec.pdf),
    )
    agent = VeriFinAgent(
        runtime,
        budget=Budget(),
        planner=llm_planner,
        trace=None,
        use_llm=llm_planner is not None,
    )
    report = EvalReport(planner=planner)
    for item in items:
        import time

        t0 = time.perf_counter()
        result = agent.run(item.question)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        r = compare(item, result)
        r.latency_ms = dt_ms
        r.evidence_recalled = any(
            s.ok and s.node in (
                "SEARCH", "EVIDENCE", "VERIFY_SPAN", "LOCATE",
                "DIFF", "COMPUTE", "LIST_FORMULAS",
            )
            for s in result.steps
        )
        report.results.append(r)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="VeriFin 评测执行器")
    parser.add_argument("--bank", default=str(DEFAULT_BANK), help="题库 JSONL 路径")
    parser.add_argument("--doc", default="MOUTAI_2024", help="文档键")
    parser.add_argument("--tier", action="append", help="只跑指定题型（可重复）")
    parser.add_argument("--planner", choices=("policy", "llm"), default="policy")
    parser.add_argument("--json", help="把结果写到这个 JSON 文件")
    parser.add_argument("--all", action="store_true", help="逐题结果里也列出通过项")
    args = parser.parse_args(argv)

    items = load_bank(args.bank)
    if args.tier:
        wanted = {t.upper() for t in args.tier}
        items = [i for i in items if i.question_type in wanted]

    llm_planner = None
    planner_name = "policy"
    if args.planner == "llm":
        try:
            from verifin.agent.planner import LLMPlanner
            from verifin.llm import LLMClient

            llm_planner = LLMPlanner(LLMClient.from_env())
            planner_name = "llm"
            print("LLM 调度：已启用")
        except Exception as exc:  # noqa: BLE001
            print(f"LLM 调度：不可用（{type(exc).__name__}: {exc}）→ 回退确定性策略")

    report = run_bank(items, doc_key=args.doc, planner=planner_name, llm_planner=llm_planner)

    print()
    print("=" * 72)
    print("汇总")
    print("=" * 72)
    print(report.render_stats())
    print()
    print("=" * 72)
    print("逐题（只列未通过；--all 列出全部）")
    print("=" * 72)
    print(report.render(show_all=args.all))

    if args.json:
        payload = {"stats": report.stats(), "results": [r.to_dict() for r in report.results]}
        Path(args.json).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n结果已写入 {args.json}")

    return 0 if all(r.passed for r in report.results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
