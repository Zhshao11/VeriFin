"""显式图编排。

为什么自己写而不套现成 Agent 框架：

1. **要能画出图**。这是一份会摆在人面前的工程 —— 节点表 + 邻接表就是图本身，
   框架把控制流藏进内部循环里，讲解与复盘都无从下手。
2. **要能把 LLM 关掉**。核心链路必须能在没有 LLM 的情况下跑通并被测试，
   否则「LLM 不参与数值」这条红线无法验证。
3. **要在同一份代码里跑 V0/V1/V2 消融**。同一个图，换策略与组件即可，
   不需要维护三份彼此漂移的实现。

结构：

    INTENT ──┬─→ SEARCH → EVIDENCE → VERIFY_SPAN → LOCATE → ANSWER
             ├─→ LIST_FORMULAS → COMPUTE ─────────────────→ ANSWER
             └─→ REFUSE（任何时刻都可直达）

两条硬约束，都由**程序**执行，LLM 无法绕过：
- 每个节点的工具**入参由图从上一步状态里填**，LLM 只挑节点名，因此它没有编数字的机会。
- `ANSWER` 节点先查六元组是否齐全，不齐就照样转 `REFUSE`。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifin.agent.planner import Decision, LLMPlanner, policy_decide  # noqa: E402
from verifin.agent.tools import TOOL_IMPLS, ToolRuntime  # noqa: E402
from verifin.agent.trace import TraceStore, new_run_id  # noqa: E402

TERMINALS = ("ANSWER", "REFUSE", "ABORT")


# --------------------------------------------------------------------------
# 图的定义（声明式，可直接渲染成图）
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeSpec:
    """一个节点。工具可为空（如 INTENT 只做路由，不调用工具）。"""

    node: str
    tool: str | None
    legal: tuple[str, ...]
    terminal: bool = False
    description: str = ""


GRAPH: dict[str, NodeSpec] = {
    "INTENT": NodeSpec(
        "INTENT", None, ("SEARCH", "LIST_FORMULAS", "REFUSE"),
        description="判断问题是查某个科目，还是要核验一条勾稽关系",
    ),
    "SEARCH": NodeSpec(
        "SEARCH", "search_statement", ("SEARCH", "EVIDENCE", "REFUSE"),
        description="四路召回，拿到候选科目",
    ),
    "EVIDENCE": NodeSpec(
        "EVIDENCE", "get_row_evidence", ("EVIDENCE", "VERIFY_SPAN", "LOCATE", "REFUSE"),
        description="取候选科目的完整证据行（数值只能来自原文）",
    ),
    "VERIFY_SPAN": NodeSpec(
        "VERIFY_SPAN", "verify_span", ("LOCATE", "ANSWER", "REFUSE"),
        description="span 硬校验两道关卡",
    ),
    "LOCATE": NodeSpec(
        "LOCATE", "locate_in_pdf", ("ANSWER", "REFUSE"),
        description="定位到 PDF 坐标并做同行校验",
    ),
    "LIST_FORMULAS": NodeSpec(
        "LIST_FORMULAS", "list_formulas", ("LIST_FORMULAS", "COMPUTE", "REFUSE"),
        description="列出可用的勾稽公式",
    ),
    "COMPUTE": NodeSpec(
        "COMPUTE", "compute", ("ANSWER", "REFUSE"),
        description="Decimal 运算核验勾稽关系",
    ),
    "ANSWER": NodeSpec("ANSWER", None, (), terminal=True, description="装配六元组并判定是否可答"),
    "REFUSE": NodeSpec("REFUSE", None, (), terminal=True, description="程序级拒答"),
    "ABORT": NodeSpec("ABORT", None, (), terminal=True, description="预算耗尽，终止"),
}


@dataclass
class Budget:
    """预算三件套。任一耗尽即终止，而不是让 LLM 继续空转。"""

    max_steps: int = 12
    max_tool_calls: int = 10
    max_llm_calls: int = 8

    @property
    def exceeded_reason(self) -> str | None:
        return None


@dataclass
class Step:
    seq: int
    node: str
    tool: str | None
    args: dict[str, Any]
    ok: bool | None
    detail: str
    source: str
    result: dict[str, Any] | None = None


@dataclass
class RunResult:
    """一次运行的完整结果。"""

    run_id: str
    question: str
    decision: str
    """ANSWER / REFUSE / ABORT"""
    answer: dict[str, Any] | None = None
    refusal: dict[str, str] | None = None
    route: str = "LOOKUP"
    steps: list[Step] = field(default_factory=list)
    budget_exceeded: bool = False
    llm_calls: int = 0
    tool_calls: int = 0


# --------------------------------------------------------------------------
# 意图（确定性分类 —— 意图错判是可排查的程序行为，不是模型的自由发挥）
# --------------------------------------------------------------------------

VERIFY_KEYWORDS = ("平衡", "勾稽", "是否成立", "核验", "等于", "加总", "不平衡")
FORMULA_HINTS: dict[str, tuple[str, ...]] = {
    "F1": ("资产", "负债", "所有者权益"),
}


def classify_intent(question: str) -> tuple[str, str | None]:
    """返回 (route, formula_id)。"""
    q = question or ""
    if any(k in q for k in VERIFY_KEYWORDS) or "=" in q or "＋" in q or "+" in q:
        for fid, hints in FORMULA_HINTS.items():
            if sum(h in q for h in hints) >= 2:
                return "VERIFY", fid
        # 说了「平衡/勾稽」但没点名哪条公式，默认资产恒等式
        return "VERIFY", "F1"
    return "LOOKUP", None


# --------------------------------------------------------------------------
# 图执行器
# --------------------------------------------------------------------------


class VeriFinAgent:
    """把检索/校验/坐标层编排成一个会自己选工具的 Agent。"""

    def __init__(
        self,
        runtime: ToolRuntime,
        *,
        budget: Budget | None = None,
        planner: Any | None = None,
        trace: TraceStore | None = None,
        use_llm: bool = False,
    ) -> None:
        self.runtime = runtime
        self.budget = budget or Budget()
        self.trace = trace
        self.planner = planner
        self.use_llm = use_llm and planner is not None

    # ---------------------------------------------------------------- 主循环
    def run(self, question: str) -> RunResult:
        run_id = new_run_id()
        route, formula_id = classify_intent(question)
        if self.use_llm and self.planner is not None:
            reset = getattr(self.planner, "reset", None)
            if callable(reset):
                reset()  # 预算是每次运行的，不是全局的
        state: dict[str, Any] = {
            "question": question,
            "route": route,
            "formula_id": formula_id,
            "period": "current",
            "candidates": [],
            "tried": set(),
            "evidence": None,
            "span": None,
            "bbox": None,
            "outcome": None,
            "formulas": None,
            "visited": [],
            "node": "INTENT",
            "last_result": None,
            "last_status": None,
            # 公司名与期间属于六元组字段；缺失时照样拒答。
            # 目前来源是演示配置（封面结构化解析待接入），这一事实在演示输出里写明。
            "company": self.runtime.company,
            "period_label": self.runtime.period,
        }

        steps: list[Step] = []
        tool_calls = 0
        llm_calls = 0
        node = "INTENT"
        budget_exceeded = False
        decision_source = "policy"

        if self.trace is not None:
            self.trace.start_run(run_id, question, route=route)

        while True:
            spec = GRAPH[node]
            if spec.terminal:
                break
            if len(steps) >= self.budget.max_steps:
                node = self._handle_exhausted(steps, run_id, "max_steps")
                budget_exceeded = True
                break

            args = self._fill_args(node, state)
            step: Step
            if spec.tool is None:
                detail = f"route={state['route']}"
                if state["formula_id"]:
                    detail += f" formula={state['formula_id']}"
                step = Step(len(steps), node, None, {}, True, detail, "policy", None)
            else:
                if tool_calls >= self.budget.max_tool_calls:
                    node = self._handle_exhausted(steps, run_id, "max_tool_calls")
                    budget_exceeded = True
                    break
                impl = TOOL_IMPLS[spec.tool]
                result = impl(self.runtime, **args)
                tool_calls += 1
                ok = bool(result.get("ok"))
                detail = result.get("detail") or result.get("error") or ""
                if ok:
                    detail = self._summarize(spec.tool, result, detail)
                step = Step(len(steps), node, spec.tool, args, ok, detail, "tool", result)
                self._absorb(node, result, state)
                state["last_result"] = result
                state["last_status"] = "ok" if ok else "error"
                state["last_error"] = None if ok else result.get("error")

            steps.append(step)
            state["visited"].append(node)
            if self.trace is not None:
                self.trace.record_step(
                    run_id, step.seq, node=node, tool=step.tool, args=step.args,
                    ok=step.ok, detail=step.detail, source=step.source,
                )

            legal = spec.legal
            decision = self._decide_next(state, legal, llm_calls)
            if decision.source == "llm":
                llm_calls += 1
            decision_source = decision.source
            node = decision.next_node
            state["node"] = node

        result = self._finalize(run_id, question, node, state, steps,
                                tool_calls, llm_calls, budget_exceeded, decision_source)
        if self.trace is not None:
            self.trace.finish_run(
                run_id,
                decision=result.decision,
                reason=(result.refusal or {}).get("reason"),
                steps=len(steps),
                tool_calls=tool_calls,
                llm_calls=llm_calls,
                budget_exceeded=budget_exceeded,
                planner_source=decision_source,
            )
        result.steps = steps
        result.tool_calls = tool_calls
        result.llm_calls = llm_calls
        return result

    # ---------------------------------------------------------------- 内部
    def _decide_next(
        self, state: Mapping[str, Any], legal: Sequence[str], llm_calls: int = 0
    ) -> Decision:
        """挑下一个节点。LLM 预算用完就直接退到确定性策略。"""
        if not self.use_llm or self.planner is None:
            return policy_decide(state, legal)
        if llm_calls >= self.budget.max_llm_calls:
            return Decision(
                policy_decide(state, legal).next_node, "policy", "LLM 调用预算已用尽"
            )
        return self.planner.decide(state, legal)

    @staticmethod
    def _handle_exhausted(steps: list[Step], run_id: str, kind: str) -> str:
        steps.append(
            Step(len(steps), "ABORT", None, {}, False, f"预算耗尽：{kind}", "policy", None)
        )
        return "ABORT"

    @staticmethod
    def _fill_args(node: str, state: Mapping[str, Any]) -> dict[str, Any]:
        """工具入参由图从状态里填 —— LLM 没有编参数的机会。"""
        if node == "SEARCH":
            return {"query": state["question"]}
        if node == "EVIDENCE":
            label = VeriFinAgent._next_candidate(state)
            return {"label": label or "", "period": state.get("period", "current")}
        if node == "VERIFY_SPAN":
            ev = state.get("evidence") or {}
            return {
                "label": ev.get("label", ""),
                "claimed_span": ev.get("fragment", ""),
                "claimed_value": ev.get("value", ""),
            }
        if node == "LOCATE":
            ev = state.get("evidence") or {}
            return {"label": ev.get("label", ""), "value": ev.get("value", "")}
        if node == "LIST_FORMULAS":
            return {}
        if node == "COMPUTE":
            return {
                "formula_id": state.get("formula_id") or "F1",
                "period": state.get("period", "current"),
            }
        return {}

    @staticmethod
    def _next_candidate(state: Mapping[str, Any]) -> str | None:
        tried = state.get("tried") or set()
        for cand in state.get("candidates") or []:
            label = cand.get("label")
            if label and label not in tried:
                return label
        return None

    @staticmethod
    def _absorb(node: str, result: Mapping[str, Any], state: dict[str, Any]) -> None:
        """把工具返回值落到状态里。只存结论，不存大对象。"""
        if not result.get("ok"):
            return
        if node == "SEARCH":
            state["candidates"] = list(result.get("hits") or [])
            state["per_channel"] = result.get("per_channel") or {}
            state["has_more_candidates"] = bool(state["candidates"])
        elif node == "EVIDENCE":
            state["evidence"] = dict(result)
            tried = set(state.get("tried") or set())
            if result.get("requested_label"):
                tried.add(result["requested_label"])
            state["tried"] = tried
            state["has_more_candidates"] = (
                VeriFinAgent._next_candidate(state) is not None
            )
        elif node == "VERIFY_SPAN":
            state["span"] = dict(result)
        elif node == "LOCATE":
            state["bbox"] = result.get("bbox")
            state["located_page"] = result.get("page")
        elif node == "LIST_FORMULAS":
            state["formulas"] = list(result.get("formulas") or [])
        elif node == "COMPUTE":
            state["outcome"] = dict(result)

    @staticmethod
    def _summarize(tool: str, result: Mapping[str, Any], fallback: str) -> str:
        if tool == "search_statement":
            hits = result.get("hits") or []
            top = ", ".join(h.get("label", "?") for h in hits[:3])
            return f"命中 {len(hits)} 条候选：{top}"
        if tool == "get_row_evidence":
            return f"{result.get('label')} = {result.get('value')} {result.get('unit')} (p{result.get('page')})"
        if tool == "verify_span":
            accepted = result.get("accepted")
            return f"span 校验{'通过' if accepted else '未通过'}（{result.get('mode') or result.get('reason')}）"
        if tool == "locate_in_pdf":
            return f"p{result.get('page')} 坐标 {tuple(result.get('bbox') or [])}"
        if tool == "list_formulas":
            return "、".join(f["id"] for f in (result.get("formulas") or [])[:6])
        if tool == "compute":
            return f"{result.get('verdict')} 差额 {result.get('diff')} 容差 ±{result.get('tolerance')}"
        return fallback

    @staticmethod
    def _finalize(
        run_id: str,
        question: str,
        node: str,
        state: Mapping[str, Any],
        steps: list[Step],
        tool_calls: int,
        llm_calls: int,
        budget_exceeded: bool,
        decision_source: str,
    ) -> RunResult:
        if node == "ABORT" or budget_exceeded:
            return RunResult(
                run_id, question, "ABORT",
                refusal={"reason": "BUDGET_EXCEEDED",
                         "detail": "预算耗尽，未完成任务。宁可终止也不让 Agent 空转。"},
                route=state.get("route", "LOOKUP"), steps=steps,
                budget_exceeded=True, llm_calls=llm_calls, tool_calls=tool_calls,
            )

        if state.get("route") == "VERIFY":
            outcome = state.get("outcome")
            if not outcome:
                reason = VeriFinAgent._first_error_reason(steps) or "NO_OUTCOME"
                return RunResult(
                    run_id, question, "REFUSE",
                    refusal={"reason": reason,
                             "detail": "未拿到勾稽核验结论，不给结论。"},
                    route="VERIFY", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
                )
            answer = {
                "公式": f"{outcome.get('formula_id')} {outcome.get('formula_name')}",
                "结论": outcome.get("verdict"),
                "左值": outcome.get("lhs"),
                "右值合计": outcome.get("rhs"),
                "差额": outcome.get("diff"),
                "容差": outcome.get("tolerance"),
                "操作数": outcome.get("operands"),
                "来源": outcome.get("sources"),
            }
            return RunResult(run_id, question, "ANSWER", answer=answer, route="VERIFY",
                             steps=steps, llm_calls=llm_calls, tool_calls=tool_calls)

        # LOOKUP 路线：六元组齐全才给答案，缺一项就拒答
        ev = state.get("evidence") or {}
        span_state = state.get("span") or {}
        missing = [
            name
            for name, value in (
                ("指标", ev.get("label")),
                ("数值", ev.get("value")),
                ("单位", ev.get("unit")),
                ("来源页码", ev.get("page")),
                ("原文片段", ev.get("fragment")),
            )
            if not value
        ]
        if missing:
            reason = VeriFinAgent._first_error_reason(steps) or "SIX_TUPLE_INCOMPLETE"
            return RunResult(
                run_id, question, "REFUSE",
                refusal={"reason": reason,
                         "detail": f"六元组缺：{'、'.join(missing)}。程序级拒答，模型无否决权。"},
                route="LOOKUP", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
            )
        if not span_state.get("accepted"):
            return RunResult(
                run_id, question, "REFUSE",
                refusal={"reason": span_state.get("reason") or "SPAN_REJECTED",
                         "detail": span_state.get("detail") or "证据未通过 span 硬校验。"},
                route="LOOKUP", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
            )

        six = {
            "公司": state.get("company") or "",
            "期间": state.get("period_label") or "",
            "指标": ev.get("label"),
            "数值": ev.get("value"),
            "单位": ev.get("unit"),
            "来源": f"第 {ev.get('page')} 页"
            + (f" · 坐标 {tuple(state['bbox'])}" if state.get("bbox") else ""),
        }
        missing_meta = [k for k in ("公司", "期间") if not six[k]]
        if missing_meta:
            return RunResult(
                run_id, question, "REFUSE",
                refusal={"reason": "SIX_TUPLE_INCOMPLETE",
                         "detail": f"六元组缺：{'、'.join(missing_meta)}（封面解析未接入）。"},
                route="LOOKUP", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
            )
        return RunResult(
            run_id, question, "ANSWER",
            answer={"six_tuple": six, "fragment": ev.get("fragment"),
                    "bbox": state.get("bbox"), "span_mode": span_state.get("mode")},
            route="LOOKUP", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
        )

    @staticmethod
    def _first_error_reason(steps: Sequence[Step]) -> str | None:
        for step in reversed(steps):
            if step.ok is False and step.result and step.result.get("error"):
                return str(step.result["error"])
        return None


def render_graph_text() -> str:
    """把图渲染成文本，便于在演示与文档里直接展示结构（不依赖图形库）。"""
    lines = ["显式图结构（节点 → 合法后继）：", ""]
    for name, spec in GRAPH.items():
        if spec.terminal:
            lines.append(f"  {name:<16} ← 终态")
            continue
        tool = spec.tool or "（无工具）"
        lines.append(f"  {name:<16} {tool:<18} → {' / '.join(spec.legal)}")
    return "\n".join(lines)
