"""下一步走哪个节点 —— LLM 选，图把关。

这里刻意**不让 LLM 直接写工具参数**：

    LLM 只在一个受限集合里挑「下一个节点」，
    参数由图从上一步的状态里填。

原因很直接：一旦让 LLM 自己编 `claimed_value` 之类的参数，
「LLM 不产生数字」这条红线就守不住了。
让 LLM 做选择、让程序填参数，既能体现自主决策，又不把数值生成权交出去。

LLM 不可用时（没配 `.env`、调用失败、返回非法后继），退到确定性策略，
并在返回结果里写明 `source="policy"` —— 降级事实不掩盖。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SYSTEM_PROMPT = """你是一个财报核验 Agent 的调度器。

你只做一件事：在当前允许的节点里，挑一个作为下一步。不要解释太多，不要编数据。

规则：
- 只能从给定的候选节点里选一个，不得自创节点名。
- 证据类节点（EVIDENCE / VERIFY_SPAN / LOCATE）应按顺序推进：先取证据，再校验片段，最后定位坐标。
- 若上一步返回错误且没有补救手段，直接选 REFUSE —— 拒答不是失败，是这个系统最重要的能力。
- 怀疑自己选错时，宁可选能补证据的节点，也不要急着选 ANSWER。
"""


@dataclass(frozen=True)
class Decision:
    """一次调度决策。"""

    next_node: str
    source: str
    """`llm` 或 `policy`。降级必须可查。"""
    reason: str = ""
    raw_choice: str | None = None
    """LLM 原样返回的选择。与最终选择不同时就说明它被图纠正了。"""


#: 图的合法后继之外的一切选择，都不会被采纳。
ILLEGAL_CHOICE_NOTE = "LLM 给出的节点不在当前合法后继集合内，已由兜底策略接管"


#: 这些错误不致命：坐标是证据的**增强项**，不是六元组的必需字段。
#: 没有挂 PDF 时拿不到坐标，但答案依然可给 —— 只是标注「未定位」。
NON_FATAL_ERRORS = frozenset({"NO_PDF"})


def policy_decide(state: Mapping[str, Any], legal: Sequence[str]) -> Decision:
    """确定性兜底策略。

    不追求聪明，追求**可复现**：同样的状态永远得到同样的下一步，
    这样评测时才能把「模型表现」和「图的表现」分开归因。

    两条必须写死的规则：
    1. **不走回头路**。图的合法后继里保留了 SEARCH / EVIDENCE 以允许重试，
       但兜底策略默认跳过已访问过的节点，否则会原地打转到预算耗尽
       （这个缺陷是被预算测试打出来的，不是先想到的）。
    2. **只有真的还有下一条候选时才重试**，否则错误会一路累积成拒答。
    """
    route = state.get("route")
    order = ("LIST_FORMULAS", "COMPUTE", "ANSWER") if route == "VERIFY" else (
        "DIFF", "SEARCH", "EVIDENCE", "VERIFY_SPAN", "LOCATE", "ANSWER",
    )
    # GUARD 永远排在流程最前面：主体 / 期间约束不过，后面做什么都是错的。
    order = ("GUARD", *order)
    visited = set(state.get("visited") or ())

    fatal = (
        state.get("last_status") == "error"
        and state.get("last_error") not in NON_FATAL_ERRORS
    )
    if fatal:
        if state.get("has_more_candidates") and "EVIDENCE" in legal and "EVIDENCE" in order:
            return Decision("EVIDENCE", "policy", "换下一条候选重试")
        if "REFUSE" in legal:
            return Decision("REFUSE", "policy", "上一步失败且无补救手段")

    for node in order:
        if node in legal and node not in visited:
            return Decision(node, "policy", "按流程推进（跳过已访问节点）")

    # 流程已走完：能给答案就给，给不出就拒答
    if "ANSWER" in legal:
        return Decision("ANSWER", "policy", "流程已走完，装配结论")
    if "REFUSE" in legal:
        return Decision("REFUSE", "policy", "流程已走完但证据不足")
    return Decision(legal[-1] if legal else "ABORT", "policy", "兜底：取最后一个合法后继")


class LLMPlanner:
    """用 LLM 选下一个节点，返回非法值时退回 :func:`policy_decide`。"""

    def __init__(self, client: Any, *, max_calls: int = 8) -> None:
        self._client = client
        self.max_calls = max_calls
        self.calls = 0

    def reset(self) -> None:
        """预算是**每次运行**的，不是全局的。

        调度器会被多个问题复用；不重置的话第二个问题起就永远走兜底策略，
        看起来像"LLM 调度没生效"，其实是预算被上一次运行吃掉了。
        """
        self.calls = 0

    def decide(self, state: Mapping[str, Any], legal: Sequence[str]) -> Decision:
        if self.calls >= self.max_calls:
            return Decision(
                policy_decide(state, legal).next_node,
                "policy",
                "LLM 调用预算已用尽，交给确定性策略",
            )

        schema = {
            "type": "object",
            "properties": {
                "next": {"type": "string", "enum": list(legal)},
                "reason": {"type": "string"},
            },
            "required": ["next", "reason"],
            "additionalProperties": False,
        }
        self.calls += 1
        try:
            payload = self._client.chat_json(
                system=SYSTEM_PROMPT,
                user=(
                    f"问题：{state.get('question', '')}\n"
                    f"路线：{state.get('route')}\n"
                    f"当前节点：{state.get('node')}\n"
                    f"已走过的节点：{state.get('visited')}\n"
                    f"上一步结果：{_brief(state.get('last_result'))}\n"
                    f"候选节点：{list(legal)}"
                ),
                schema=schema,
                schema_name="next_step",
                purpose="pick_next_node",
                max_tokens=256,
            )
        except Exception as exc:  # noqa: BLE001 —— 端点不可用不能让整个 Agent 崩
            fallback = policy_decide(state, legal)
            return Decision(
                fallback.next_node, "policy", f"LLM 调用失败（{type(exc).__name__}），降级"
            )

        choice = str(payload.get("next", "")).strip()
        reason = str(payload.get("reason", ""))[:200]
        if choice not in legal:
            fallback = policy_decide(state, legal)
            return Decision(
                fallback.next_node,
                "policy",
                ILLEGAL_CHOICE_NOTE,
                raw_choice=choice or None,
            )
        return Decision(choice, "llm", reason, raw_choice=choice)


def _brief(result: Any, limit: int = 220) -> str:
    if result is None:
        return "（无）"
    text = str(result)
    return text if len(text) <= limit else text[:limit] + "…"
