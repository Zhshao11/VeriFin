"""VeriFin Agent 编排层。

把已落地的解析层、检索层、坐标层、核验层编排成一个会自己选工具的 Agent。

三件事是这层的核心，也是「这算不算一个 Agent」的判据：

1. **自主决策**：下一步走哪个节点由调度器（LLM 或确定性策略）在图里选，
   不是写死的 if-else 流水线 —— 同样的问题在不同证据状态下可以走不同的路。
2. **受约束**：只能从当前节点的**合法后继**里选；LLM 给出非法值会被图纠正并记录。
3. **可复盘**：每一步的工具入参与返回值都落库，可回放。

三条红线在这一层的落实方式：

| 红线 | 落实位置 |
|---|---|
| LLM 不产生数字 | 工具**入参由图从状态里填**，LLM 只挑节点名（`graph._fill_args`） |
| LLM 不做算术 | `compute` 工具内部调 `formulas.evaluate_formula`（Decimal） |
| LLM 无权决定拒答 | `_finalize` 按六元组是否齐全判定，LLM 选了 ANSWER 也不算数 |
"""

from verifin.agent.graph import (
    GRAPH,
    MAX_QUESTION_CHARS,
    Budget,
    InputError,
    NodeSpec,
    RunResult,
    Step,
    VeriFinAgent,
    classify_intent,
    render_graph_text,
    validate_question,
)
from verifin.agent.planner import LLMPlanner, Decision, policy_decide
from verifin.agent.tools import (
    LABEL_ALIAS,
    TOOL_IMPLS,
    ToolRuntime,
    ToolSpec,
    tool_specs,
)
from verifin.agent.trace import TraceStore

__all__ = [
    "GRAPH",
    "MAX_QUESTION_CHARS",
    "Budget",
    "InputError",
    "NodeSpec",
    "RunResult",
    "Step",
    "VeriFinAgent",
    "classify_intent",
    "render_graph_text",
    "validate_question",
    "LLMPlanner",
    "Decision",
    "policy_decide",
    "LABEL_ALIAS",
    "TOOL_IMPLS",
    "ToolRuntime",
    "ToolSpec",
    "tool_specs",
    "TraceStore",
]
