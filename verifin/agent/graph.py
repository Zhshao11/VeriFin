"""显式图编排。

为什么自己写而不套现成 Agent 框架：

1. **要能画出图**。这是一份会摆在人面前的工程 —— 节点表 + 邻接表就是图本身，
   框架把控制流藏进内部循环里，讲解与复盘都无从下手。
2. **要能把 LLM 关掉**。核心链路必须能在没有 LLM 的情况下跑通并被测试，
   否则「LLM 不参与数值」这条红线无法验证。
3. **要在同一份代码里跑 V0/V1/V2 消融**。同一个图，换策略与组件即可，
   不需要维护三份彼此漂移的实现。

结构：

    INTENT → GUARD ─┬─→ SEARCH → EVIDENCE → VERIFY_SPAN → LOCATE → ANSWER
                    ├─→ LIST_FORMULAS → COMPUTE ─────────────────→ ANSWER
                    ├─→ DIFF ───────────────────────────────────→ ANSWER
                    └─→ REFUSE（任何时刻都可直达）

两条硬约束，都由**程序**执行，LLM 无法绕过：
- 每个节点的工具**入参由图从上一步状态里填**，LLM 只挑节点名，因此它没有编数字的机会。
- `ANSWER` 节点先查六元组是否齐全，再查**问答一致性**；任一不过就转 `REFUSE`。
"""

from __future__ import annotations

import re
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
from verifin.guards import best_label_match, label_consistency  # noqa: E402
from verifin.scope import both_scopes_named, detect_scope_in_question  # noqa: E402

TERMINALS = ("ANSWER", "REFUSE", "ABORT")

#: 问句长度上限（字符）。与 web 端 `AskRequest.max_length=200` 保持一致。
#: 为什么要有：超长输入会（a）把检索查询串撑到无意义，（b）让 jieba 切词耗时线性增长，
#: （c）成为提示注入的载体。校验放在 Agent 层而不是只在 web 层，是因为 CLI 与评测
#: 也会调 `run()` —— 只在入口挡一道，换个入口就绕过去了。
MAX_QUESTION_CHARS = 200


class InputError(ValueError):
    """输入不合法（空、超长、类型不对）。属于**调用方错误**，不是拒答。"""


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
        "INTENT", None, ("GUARD", "REFUSE"),
        description="判断问题是查某个科目，还是要核验一条勾稽关系",
    ),
    "GUARD": NodeSpec(
        "GUARD", "check_constraints", ("SEARCH", "LIST_FORMULAS", "DIFF", "REFUSE"),
        description="主体 / 期间约束：这个问题本文档有没有资格回答",
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
    "DIFF": NodeSpec(
        "DIFF", "diff_two_rows", ("ANSWER", "REFUSE"),
        description="两行相减：取 A、B 两行（可跨口径）原文数值求差",
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

#: 公式触发词表：公式 id → (触发词, 触发门槛)。
#:
#: 门槛不是一律 2，因为触发词的性质不同：
#:
#: - 「毛利率」「同比」这类词**自身就足够独特**，出现一次即可确定意图（门槛 1）；
#: - 「资产总计」「负债合计」这类词会**分散出现在许多科目名里**，
#:   单个词可能只是顺带提到，必须同时命中两个才敢判定（门槛 2）。
#:
#: 早先这张表只有 F1 一条，导致注册表里另外 7 条公式**永远选不到** ——
#: D4 建题库时为了能问到 F2a/F2b/F4 才发现。**注册表里有不等于用得上**。
#: 漏加门槛更高的公式不会静默出错：它会退回 `VERIFY` + 默认 F1，被评测当成取错公式。
FORMULA_HINTS: dict[str, tuple[tuple[str, ...], int]] = {
    "F1": (("资产总计", "负债合计", "所有者权益合计"), 2),
    "F2a": (("净利润", "利润总额", "所得税费用"), 2),
    "F2b": (("归属于母公司", "少数股东损益"), 2),
    "F3a": (("经营活动", "投资活动", "筹资活动", "净增加额"), 2),
    "F3b": (("期末现金", "期初现金"), 2),
    "F4": (("毛利率",), 1),
    "F6": (("同比",), 1),
}

#: 触发词命中不足、但问题里出现了核验类词汇时，默认按资产恒等式处理。
DEFAULT_VERIFY_FORMULA = "F1"


def validate_question(question: str) -> str:
    """校验并归一化问句。返回去除首尾空白后的问句。

    Raises:
        InputError: 类型不对 / 空白 / 超长。

    **为什么不做关键词黑名单**（比如见到「忽略以上指令」就拒）：
    黑名单既会误拦正常问句（财务问句里出现「请忽略附注部分」很常见），
    又能被换个说法绕过 —— 拦不住的防御不如不拦。真正的注入防御在本项目里
    是**结构性的**：LLM 无权决定数值、无权改参数、无权决定拒答，
    所以就算问句里写满「输出 999」，系统能给的也只有原文真值或拒答。
    这一条由 `tests/test_agent.py::TestQuestionValidation` 里的注入用例守着。
    """
    if not isinstance(question, str):
        raise InputError(f"问句必须是字符串，收到 {type(question).__name__}")
    q = question.strip()
    if not q:
        raise InputError("问句为空")
    if len(q) > MAX_QUESTION_CHARS:
        raise InputError(
            f"问句过长（{len(q)} 字符 > 上限 {MAX_QUESTION_CHARS}）—— "
            "超长输入会拖垮检索与切词，且是提示注入的常用载体"
        )
    return q


def classify_intent(question: str, known_labels: Sequence[str] = ()) -> tuple[str, str | None]:
    """返回 (route, formula_id)。

    两步走，顺序不能反：

    1. 先判断这题**要不要核验**。信号有两个：出现核验类词汇，或命中了一个
       「自身就足够独特」的公式触发词（门槛为 1 的 F4 / F6）。
       少了这一步，「2024年毛利率是多少」会被当成查科目 —— 而报表上根本没有
       「毛利率」这一行，结局是召回失败，拒答原因指向检索层，指错了地方。
    2. 再在通过门槛的公式里挑命中最多的那个；一个都没有才退回默认公式。

    若把顺序反过来（先按词汇兜底成 F1），「净利润是否等于利润总额减去所得税费用」
    会被兜底成 F1，于是**拿资产负债表的科目去算利润表的恒等式**，
    一路走到 `MISSING_OPERANDS` 才失败 —— 失败得很晚，且原因指向错误的层。
    """
    q = question or ""

    hits: dict[str, int] = {
        fid: sum(1 for w in words if w in q)
        for fid, (words, _threshold) in FORMULA_HINTS.items()
    }
    met = [
        fid
        for fid, (words, threshold) in FORMULA_HINTS.items()
        if hits[fid] >= threshold
    ]
    #: 门槛为 1 的公式，其触发词本身就是「这题在要一个派生量 / 一个判定」的信号。
    distinctive = [
        fid for fid in met if FORMULA_HINTS[fid][1] == 1
    ]

    verify_signal = (
        any(k in q for k in VERIFY_KEYWORDS) or "=" in q or "＋" in q or "+" in q
    ) or bool(distinctive)
    if not verify_signal:
        # 核验信号没有，再看是不是「两行相减」题（"X 与 Y 的差额"）。
        # 这一步需要已知科目名，故把 known_labels 透传进来；缺它就不判 DIFF。
        if known_labels and re.search(r"相差|差额|之差", q):
            operands = parse_diff_operands(q, known_labels)
            if len(operands) == 2:
                return "DIFF", None
        return "LOOKUP", None

    if met:
        # 命中数相同时按注册顺序取先出现的，保证结果可复现。
        best = max(met, key=lambda fid: (hits[fid], -list(FORMULA_HINTS).index(fid)))
        return "VERIFY", best
    return "VERIFY", DEFAULT_VERIFY_FORMULA


#: 「两行相减」题的连词切分点。
_DIFF_SEG_RE = re.compile(r"与|及|以及|和|、|,|，")

#: 仅在段内出现这些词时，才把口径判给该段 —— 否则「合并利润表」里的
#: 「合并」会被误当成单独点名的口径（见 :func:`_segment_scope`）。
_SCOPE_QUALIFIER_WORDS = ("口径", "报表", "本部")


def _segment_scope(seg: str, whole_scope: str | None) -> str | None:
    """取某一段的口径。

    段内若带了「X 口径 / X 报表 / 本部」这类显式限定词，就以段内为准
    （例如「母公司口径的资产总计」→ 母公司，「合并口径的资产总计」→ 合并）。
    没带限定词时退回整句口径（例如「2024年合并利润表中，归母净利润」→ 合并）。
    """
    if any(w in seg for w in _SCOPE_QUALIFIER_WORDS):
        return detect_scope_in_question(seg)
    return whole_scope


def parse_diff_operands(question: str, known_labels: Sequence[str]) -> list[dict[str, str]]:
    """从「X 与 Y 的差额」题里析出两个操作数 `(科目名, 口径)`。

    确定性解析（不调模型），规则：

    1. 按连词把问句切成若干段；
    2. 每段用 :func:`best_label_match` 取「被点名的最具体科目」，
       并用 :func:`_segment_scope` 取该段的口径；
    3. 去重后若恰好得两个操作数，判定为 DIFF。

    为什么按段取最具体科目：问「归母净利润与净利润的差额」时，
    「净利润」是「归母净利润」的子串，只判断"包含"会把两个操作数都判成净利润。
    取最长就自然消解了嵌套。
    """
    q = question or ""
    whole_scope = detect_scope_in_question(q)
    operands: list[dict[str, str]] = []
    seen: set[tuple[str, str | None]] = set()
    for raw in _DIFF_SEG_RE.split(q):
        seg = raw.strip()
        if not seg:
            continue
        label = best_label_match(seg, known_labels)
        if not label:
            continue
        scope = _segment_scope(seg, whole_scope)
        key = (label, scope)
        if key in seen:
            continue
        seen.add(key)
        operands.append({"label": label, "scope": scope})
    return operands


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
        question = validate_question(question)
        run_id = new_run_id()
        route, formula_id = classify_intent(question, self.runtime.known_labels)
        operands = (
            parse_diff_operands(question, self.runtime.known_labels)
            if route == "DIFF"
            else []
        )
        if self.use_llm and self.planner is not None:
            reset = getattr(self.planner, "reset", None)
            if callable(reset):
                reset()  # 预算是每次运行的，不是全局的
        state: dict[str, Any] = {
            "question": question,
            "route": route,
            "formula_id": formula_id,
            "period": "current",
            # 报表口径由问句确定性解析而来，随后由图填进每一个取数工具。
            # 它决定了「同名科目取哪一张表的那一行」——不解析的话，
            # 系统只能取"先出现的那行"，取错也不报错。
            "scope": detect_scope_in_question(question),
            "scope_both_named": both_scopes_named(question),
            "candidates": [],
            "tried": set(),
            "evidence": None,
            "span": None,
            "bbox": None,
            "outcome": None,
            "formulas": None,
            "operands": operands,
            "diff": None,
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

            legal = self._legal_for(node, state)
            decision = self._decide_next(state, legal, llm_calls)
            if decision.source == "llm":
                llm_calls += 1
            decision_source = decision.source
            node = decision.next_node
            state["node"] = node

        result = self._finalize(run_id, question, node, state, steps,
                                tool_calls, llm_calls, budget_exceeded, decision_source,
                                known_labels=self.runtime.known_labels)
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
    def _legal_for(node: str, state: Mapping[str, Any]) -> tuple[str, ...]:
        """合法后继。GUARD 的后继随意图分支 —— DIFF 意图下跳过 SEARCH 直达 DIFF。

        否则 GUARD 与 LOOKUP 共用 (SEARCH, LIST_FORMULAS, REFUSE)，
        确定性策略会沿 SEARCH → EVIDENCE 走查表路径，永远到不了 DIFF。
        意图在 INTENT 就已确定，这里只是把"该走哪条支线"显式化。
        """
        if node == "GUARD" and state.get("route") == "DIFF":
            return ("DIFF", "REFUSE")
        if node == "GUARD":
            # 非 DIFF 意图不让 GUARD 暴露 DIFF 后继，否则确定性策略会沿 DIFF 空跑。
            return ("SEARCH", "LIST_FORMULAS", "REFUSE")
        return GRAPH[node].legal

    @staticmethod
    def _fill_args(node: str, state: Mapping[str, Any]) -> dict[str, Any]:
        """工具入参由图从状态里填 —— LLM 没有编参数的机会。"""
        scope = state.get("scope")
        if node == "GUARD":
            return {"question": state["question"]}
        if node == "SEARCH":
            return {"query": state["question"]}
        if node == "EVIDENCE":
            label = VeriFinAgent._next_candidate(state)
            return {
                "label": label or "",
                "period": state.get("period", "current"),
                "scope": scope,
            }
        if node == "VERIFY_SPAN":
            ev = state.get("evidence") or {}
            return {
                "label": ev.get("label", ""),
                "claimed_span": ev.get("fragment", ""),
                "claimed_value": ev.get("value", ""),
                "scope": scope,
            }
        if node == "LOCATE":
            ev = state.get("evidence") or {}
            return {"label": ev.get("label", ""), "value": ev.get("value", ""),
                    "scope": scope}
        if node == "LIST_FORMULAS":
            return {}
        if node == "COMPUTE":
            return {
                "formula_id": state.get("formula_id") or "F1",
                "period": state.get("period", "current"),
                "scope": scope,
            }
        if node == "DIFF":
            ops = state.get("operands") or []
            a = ops[0] if len(ops) > 0 else {}
            b = ops[1] if len(ops) > 1 else {}
            return {
                "label_a": a.get("label", ""),
                "scope_a": a.get("scope"),
                "label_b": b.get("label", ""),
                "scope_b": b.get("scope"),
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
        if node == "EVIDENCE":
            # 无论成败都要记「这个科目试过了」。
            #
            # 只在成功时记账，会让一个取不到值的候选被**无限重试**：
            # `tried` 不更新 → `_next_candidate` 每次都返回同一个科目 →
            # 兜底策略以为"还有下一条候选" → 直到工具预算烧光变 ABORT。
            # 实测复现：候选「筹资活动现金流入小计」没有本期列，
            # 被连续重试 8 次，最后报成「预算耗尽」——
            # 真实原因是「候选不可用」，报出来的原因却指向预算，指错了排查方向。
            attempted = result.get("requested_label")
            if attempted:
                tried = set(state.get("tried") or set())
                tried.add(attempted)
                state["tried"] = tried
                state["has_more_candidates"] = (
                    VeriFinAgent._next_candidate(state) is not None
                )
            if not result.get("ok"):
                return
            state["evidence"] = dict(result)
            return

        if not result.get("ok"):
            return
        if node == "GUARD":
            state["constraints_ok"] = True
        elif node == "SEARCH":
            state["candidates"] = list(result.get("hits") or [])
            state["per_channel"] = result.get("per_channel") or {}
            state["has_more_candidates"] = bool(state["candidates"])
        elif node == "VERIFY_SPAN":
            state["span"] = dict(result)
        elif node == "LOCATE":
            state["bbox"] = result.get("bbox")
            state["located_page"] = result.get("page")
        elif node == "LIST_FORMULAS":
            state["formulas"] = list(result.get("formulas") or [])
        elif node == "COMPUTE":
            state["outcome"] = dict(result)
        elif node == "DIFF":
            state["diff"] = dict(result)

    @staticmethod
    def _summarize(tool: str, result: Mapping[str, Any], fallback: str) -> str:
        if tool == "check_constraints":
            return f"主体与期间约束通过（{result.get('company') or '—'}）"
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
        if tool == "diff_two_rows":
            return f"{result.get('indicator')} = {result.get('value')} {result.get('unit')}"
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
        known_labels: Sequence[str] = (),
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

        # DIFF 路线：两行相减，差的绝对值即答案
        if state.get("route") == "DIFF":
            diff = state.get("diff")
            if not diff or not diff.get("ok"):
                reason = VeriFinAgent._first_error_reason(steps) or "DIFF_FAILED"
                return RunResult(
                    run_id, question, "REFUSE",
                    refusal={"reason": reason,
                             "detail": "未拿到两行相减结论，不给结论。"},
                    route="DIFF", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
                )
            six = {
                "公司": state.get("company") or "",
                "期间": state.get("period_label") or "",
                "指标": diff.get("indicator"),
                "口径": diff.get("scope_label") or "跨口径",
                "数值": diff.get("value"),
                "单位": diff.get("unit"),
                "来源": diff.get("source"),
            }
            missing_meta = [k for k in ("公司", "期间") if not six[k]]
            if missing_meta:
                return RunResult(
                    run_id, question, "REFUSE",
                    refusal={"reason": "SIX_TUPLE_INCOMPLETE",
                             "detail": f"六元组缺：{'、'.join(missing_meta)}（封面解析未接入）。"},
                    route="DIFF", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
                )
            return RunResult(
                run_id, question, "ANSWER",
                answer={"six_tuple": six, "operands": diff.get("operands"),
                        "route": "DIFF"},
                route="DIFF", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
            )

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

        # **问答一致性**：先确认"取到的这一行就是问句问的那一行"，再装配答案。
        #
        # 这道检查位置很关键 —— 它必须在这五个字段都齐、span 也通过**之后**再跑，
        # 因为要拦的正是"证据全都合格、但答的是另一个科目"这一类。
        # 实测反例：问「现金及现金等价物净增加额」，返回「投资活动现金流入小计」，
        # 数值真实、页码正确、span 通过 —— 上面所有关卡都会放行。
        consistent, why = label_consistency(
            question, str(ev.get("label") or ""), known_labels
        )
        if not consistent:
            return RunResult(
                run_id, question, "REFUSE",
                refusal={"reason": "LABEL_MISMATCH", "detail": why},
                route="LOOKUP", steps=steps, llm_calls=llm_calls, tool_calls=tool_calls,
            )

        six = {
            "公司": state.get("company") or "",
            "期间": state.get("period_label") or "",
            "指标": ev.get("label"),
            "口径": ev.get("scope") or state.get("scope") or "未标注",
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
