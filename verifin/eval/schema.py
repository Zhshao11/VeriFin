"""评测题的数据模型与**程序级**校验。

题库的质量不能靠"看起来对"。这里把每条题目必须满足的约束写成确定性规则，
不满足就是 FAIL —— 不是警告。

两条最关键的约束：

1. **标为「应拒答」的题不允许带 gold 数值。**
   否则「本该拒答」会被标注成一个具体答案，拒答率的分母失去意义。
2. **问题里点名了口径（合并 / 母公司）时，标注口径必须与之一致。**
   中文年报同一科目名有合并与母公司两套数值（例如「资产总计」在合并表是
   298,944,579,918.70、在母公司表是 180,236,524,477.01），
   标注与问句口径不一致时，评测会把「口径取错」记成「数值算错」。

3. **勾稽题的答案不能只有「成立 / 不成立」。**
   同一恒等式在两套口径下都能平衡（合并的资产恒等式与母公司的资产恒等式
   差额都是 0.00）。只看判定，取错口径的答案会被判成对。
   所以判定型题必须同时标注 `gold_operands`（参与运算的原值），
   判分时先比操作数、再比判定。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

#: 题型分层。L1 单值抽取 / L2 勾稽计算 / L3 综合判断 / R 应拒答。
QUESTION_TYPES: tuple[str, ...] = ("L1", "L2", "L3", "R")

QUESTION_TYPE_LABELS: dict[str, str] = {
    "L1": "单值抽取",
    "L2": "勾稽 / 计算",
    "L3": "综合判断",
    "R": "应拒答",
}

#: 期望结局。只有两种 —— 部分答案不进题库（见 docs/评测方案-v1.0.md §4）。
OUTCOMES: tuple[str, ...] = ("ANSWER", "REFUSE")

#: 报表口径。中文年报特有维度，缺失会导致「取到真数但取错表」。
#:
#: `跨口径` 单列而不是并进 `不适用`：像「合并与母公司口径的资产总计相差多少」
#: 这类题**有明确口径**（两个都要），把它标成「不适用」会让报告分不清
#: 「这题本来就不涉及口径」和「这题涉及两个口径」。
SCOPES: tuple[str, ...] = ("合并", "母公司", "跨口径", "不适用")

#: 勾稽题的判定型答案。恒等式的答案不是数值，是判定 + 差额。
VERDICTS: tuple[str, ...] = ("PASS", "WARN", "FAIL")

DIFICULTIES: tuple[str, ...] = ("easy", "medium", "hard")

#: 题号格式，例如 `B-MOUTAI_2024-001`。
ID_PATTERN = re.compile(r"^[A-Z]-[A-Z0-9_]+-\d{3}$")

#: 问句里的「母公司口径」提法。
#:
#: 必须排除「归属于母公司」—— 它是一个**合并报表科目名**（归属于母公司股东的净利润），
#: 名字里带「母公司」但和母公司报表毫无关系。早先用朴素的子串判断，
#: 结果这条规则把合并不等式题误判成「口径不一致」，误报比漏报更消耗信任。
_PARENT_SCOPE_RE = re.compile(r"(?<!归属于)母公司")


@dataclass(frozen=True)
class EvalItem:
    """一条评测题。

    字段分成四组：**问句**（问什么）、**期望**（正确结局是什么）、
    **gold**（正确结局对应的证据）、**元信息**（归类与溯源）。
    """

    # --- 问句 ---------------------------------------------------------
    item_id: str
    track: str
    doc: str
    company: str
    period: str
    question: str
    question_type: str

    # --- 期望 ---------------------------------------------------------
    expected_outcome: str
    scope: str
    """报表口径：`合并` / `母公司` / `不适用`（L3 与 R 题通常为后者）。"""

    # --- gold ---------------------------------------------------------
    gold_value: str | None = None
    """数值型答案。原样字符串（含千分位），便于与原文**逐字**核对，不做数值化。"""
    gold_unit: str | None = None
    gold_pages: tuple[int, ...] = ()
    gold_evidence: tuple[str, ...] = ()
    """标注的原文片段。每个片段都必须能在 `gold_pages` 里某一页的原文中找到。"""
    gold_verdict: str | None = None
    """判定型答案，仅恒等式 / 启发式类的 L2 题使用（`PASS` / `WARN` / `FAIL`）。

    恒等式题的答案不是某个数值，而是「成立 / 不成立」+ 差额。
    把它硬塞进 `gold_value` 会让判分口径变得含糊，故单列。
    """
    gold_operands: dict[str, str] = field(default_factory=dict)
    """参与运算的原值（科目名 → 披露原样字符串）。

    判定型题**必填**，理由见模块 docstring 第 3 条：同一公式在两套口径下
    都能平衡，只比判定会让「取错报表」被判成对。
    派生量题（如毛利率）选填：填了就能顺带校验操作数是否取对。
    """
    formula: str | None = None
    """L2 题必填：走哪个公式（必须是 `FORMULA_REGISTRY` 的 id）。"""
    refusal_reason: str | None = None
    """R 题必填：为什么它不可答（否则无法区分「正确拒答」与「恰好在题里没找到」）。

    注意这是**人的分类**（"这题为什么问不出来"），与系统的机器原因
    （`NO_RECALL` / `AMBIGUOUS_ABBREVIATION` / …）是两套词汇，多数情况下对得上，
    但对不上也正常：一条题可以因为"信息不在解析窗口内"（人的分类
    `out_of_parse_window`）而被系统以 `NO_RECALL` 或 `LABEL_MISMATCH` 拒掉 ——
    两者都对，只是看问题的角度不同。
    """
    expected_refusal_code: str | None = None
    """可选：**系统侧**的机器拒答原因，填了就参与判分。

    为什么需要它：R 题的判分原先只比"有没有拒答"，不比"为什么拒答"。
    而有些缺陷的表现恰恰是**该拒的拒了、真的该答的也拒了，原因却不是标注的那一个**——
    例如口语简称「现金流」在修 P-030 之前会以笼统的 `LABEL_MISMATCH` 拒答，
    修好之后才是明确的 `AMBIGUOUS_ABBREVIATION`。两者都"拒答了"，
    只看是否拒答的话，这条题对本缺陷**零约束力**（一个把所有问句都拒答的废系统
    也能过）。填上这个字段，评测才真正守住"拒得明白"而不仅是"拒了"。

    只在该原因本身是被测的修复点时填；不填则退回"只比是否拒答"。
    """

    # --- 元信息 -------------------------------------------------------
    difficulty: str = "medium"
    notes: str = ""
    source: str = ""

    # ------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "track": self.track,
            "doc": self.doc,
            "company": self.company,
            "period": self.period,
            "question": self.question,
            "question_type": self.question_type,
            "expected_outcome": self.expected_outcome,
            "scope": self.scope,
            "gold_value": self.gold_value,
            "gold_unit": self.gold_unit,
            "gold_pages": list(self.gold_pages),
            "gold_evidence": list(self.gold_evidence),
            "gold_verdict": self.gold_verdict,
            "gold_operands": dict(self.gold_operands),
            "formula": self.formula,
            "refusal_reason": self.refusal_reason,
            "expected_refusal_code": self.expected_refusal_code,
            "difficulty": self.difficulty,
            "notes": self.notes,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvalItem":
        return cls(
            item_id=str(raw.get("item_id", "")),
            track=str(raw.get("track", "")),
            doc=str(raw.get("doc", "")),
            company=str(raw.get("company", "")),
            period=str(raw.get("period", "")),
            question=str(raw.get("question", "")),
            question_type=str(raw.get("question_type", "")),
            expected_outcome=str(raw.get("expected_outcome", "")),
            scope=str(raw.get("scope", "")),
            gold_value=_opt_str(raw.get("gold_value")),
            gold_unit=_opt_str(raw.get("gold_unit")),
            gold_pages=_int_tuple(raw.get("gold_pages")),
            gold_evidence=_str_tuple(raw.get("gold_evidence")),
            gold_verdict=_opt_str(raw.get("gold_verdict")),
            gold_operands=_str_map(raw.get("gold_operands")),
            formula=_opt_str(raw.get("formula")),
            refusal_reason=_opt_str(raw.get("refusal_reason")),
            expected_refusal_code=_opt_str(raw.get("expected_refusal_code")),
            difficulty=str(raw.get("difficulty", "medium")),
            notes=str(raw.get("notes", "")),
            source=str(raw.get("source", "")),
        )

    @property
    def is_refusal(self) -> bool:
        return self.expected_outcome == "REFUSE"


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = [value]
    return tuple(str(v).strip() for v in value if str(v).strip())


def _int_tuple(value: Any) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, (int, str)):
        value = [value]
    out: list[int] = []
    for v in value:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(set(out)))


def _str_map(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    out: dict[str, str] = {}
    for k, v in value.items():
        key = str(k).strip()
        val = str(v).strip()
        if key and val:
            out[key] = val
    return out


def formula_kind(item: EvalItem) -> str | None:
    """这条 L2 题挂的公式属于哪一类（`identity` / `derived` / `heuristic`）。

    非 L2 题、或公式不在注册表里时返回 `None`。
    """
    if item.question_type != "L2" or not item.formula:
        return None
    from verifin.formulas import FORMULA_REGISTRY

    formula = FORMULA_REGISTRY.get(item.formula)
    return formula.kind.value if formula is not None else None


def is_verdict_formula(item: EvalItem) -> bool:
    """这条题的答案是「判定」还是「数值」。

    判定型：恒等式（`identity`）与启发式（`heuristic`）——答案都是
    PASS/WARN/FAIL，不产出数值，故要求 `gold_verdict` + `gold_operands`。
    数值型：派生量（`derived`，如毛利率）——答案是一个数，故要求 `gold_value`。
    """
    return formula_kind(item) in ("identity", "heuristic")


def validate_item(item: EvalItem) -> list[str]:
    """逐条检查一条题目。返回问题清单，空列表表示通过。

    刻意不用异常：一次要把全部问题都列出来，而不是遇到第一条就停，
    否则修标注要来回跑很多轮。
    """
    problems: list[str] = []

    # --- 基本字段 ---
    if not item.item_id:
        problems.append("item_id 为空")
    elif not ID_PATTERN.match(item.item_id):
        problems.append(f"item_id 格式不合法：{item.item_id!r}（应形如 B-MOUTAI_2024-001）")
    for name, value in (
        ("doc", item.doc),
        ("company", item.company),
        ("period", item.period),
        ("question", item.question),
        ("track", item.track),
        ("source", item.source),
    ):
        if not value:
            problems.append(f"{name} 为空")

    if item.question_type not in QUESTION_TYPES:
        problems.append(f"question_type 非法：{item.question_type!r}，可选 {QUESTION_TYPES}")
    if item.expected_outcome not in OUTCOMES:
        problems.append(f"expected_outcome 非法：{item.expected_outcome!r}")
    if item.scope not in SCOPES:
        problems.append(f"scope 非法：{item.scope!r}，可选 {SCOPES}")
    if item.difficulty not in DIFICULTIES:
        problems.append(f"difficulty 非法：{item.difficulty!r}")

    # --- 口径与问句一致（防标注错）---
    # 只在**可答题**上约束，且只在问句只点名了一个口径时约束：
    # 应拒答题没有 gold，口径标注错也伤不到判分，强行约束只会制造噪声。
    q = item.question
    if item.expected_outcome == "ANSWER":
        names_parent = bool(_PARENT_SCOPE_RE.search(q)) or ("本部" in q)
        names_consolidated = "合并" in q
        if names_parent and names_consolidated:
            if item.scope != "跨口径":
                problems.append(
                    f"问句同时点名了合并与母公司口径，scope 应为「跨口径」，当前 {item.scope!r}"
                )
        elif names_parent:
            if item.scope != "母公司":
                problems.append(f"问句点名了母公司口径，但 scope={item.scope!r}")
        elif names_consolidated:
            if item.scope != "合并":
                problems.append(f"问句点名了合并口径，但 scope={item.scope!r}")

    # --- 题型与期望结局必须一致 ---
    if item.question_type == "R" and item.expected_outcome != "REFUSE":
        problems.append("R 题的 expected_outcome 必须是 REFUSE")
    if item.expected_outcome == "REFUSE" and item.question_type != "R":
        problems.append(
            f"expected_outcome=REFUSE 但 question_type={item.question_type!r}"
            "（应拒答的题必须归入 R 类，否则拒答率的分母口径会乱）"
        )

    # --- gold 字段按结局分叉 ---
    if item.expected_outcome == "ANSWER":
        needs_verdict = is_verdict_formula(item)
        if needs_verdict:
            if not item.gold_verdict:
                problems.append(
                    f"判定型题（{item.formula}）缺 gold_verdict"
                    f"（可选 {'/'.join(VERDICTS)}；恒等式/启发式的答案不是数值）"
                )
            elif item.gold_verdict not in VERDICTS:
                problems.append(f"gold_verdict 非法：{item.gold_verdict!r}，可选 {VERDICTS}")
            if not item.gold_operands:
                problems.append(
                    f"判定型题（{item.formula}）缺 gold_operands —— "
                    "同一公式在合并与母公司两套口径下都能平衡，"
                    "只比判定会把「取错报表」判成对"
                )
        if not needs_verdict and not item.gold_value:
            problems.append(f"{item.question_type} 题缺 gold_value")
        if item.gold_value and not item.gold_unit:
            problems.append("给了 gold_value 就必须给 gold_unit")
        if needs_verdict and item.gold_value:
            problems.append("判定型题不应带 gold_value（判定与数值二选一）")
        if item.gold_verdict and item.gold_value:
            problems.append("gold_verdict 与 gold_value 不应同时出现（判定型与数值型二选一）")
        if not item.gold_pages:
            problems.append("可答题缺 gold_pages")
        if not item.gold_evidence:
            problems.append("可答题缺 gold_evidence")
        if item.scope == "不适用":
            problems.append(
                "可答题的 scope 不应为「不适用」"
                "（涉及两个口径的用「跨口径」，一个口径都不涉及的归入 R 类）"
            )
        if item.refusal_reason:
            problems.append("可答题不应带 refusal_reason")
        if item.expected_refusal_code:
            problems.append(
                "可答题不应带 expected_refusal_code（那是给应拒答题校验「拒得对不对」用的）"
            )
        for page in item.gold_pages:
            if page < 1:
                problems.append(f"gold_pages 含非法页码 {page}（页码一律 1-based）")
    else:
        if not item.refusal_reason:
            problems.append("应拒答题缺 refusal_reason（否则无法区分正确拒答与恰好没找到）")
        for name, value in (
            ("gold_value", item.gold_value),
            ("gold_unit", item.gold_unit),
            ("gold_verdict", item.gold_verdict),
            ("gold_operands", item.gold_operands),
        ):
            if value:
                problems.append(f"应拒答题不应带 {name}（会把「本该拒答」标成一个答案）")
        if item.gold_pages or item.gold_evidence:
            problems.append("应拒答题不应带 gold_pages / gold_evidence")

    # --- 公式题必须挂真实公式 ---
    if item.question_type == "L2":
        if not item.formula:
            problems.append("L2 题缺 formula")
        else:
            from verifin.formulas import FORMULA_REGISTRY

            if item.formula not in FORMULA_REGISTRY:
                known = "、".join(sorted(FORMULA_REGISTRY))
                problems.append(f"formula={item.formula!r} 不在注册表里（可用：{known}）")
    elif item.formula:
        problems.append(f"只有 L2 题才应带 formula，当前题型是 {item.question_type!r}")

    if item.gold_operands and item.question_type not in ("L2", "L3"):
        problems.append(
            f"只有 L2 / L3 题才应带 gold_operands，当前题型是 {item.question_type!r}"
            "（L1 单值抽取题的答案就是报表上那一行，不需要另标来源值）"
        )

    return problems


def validate_bank(items: list[EvalItem]) -> dict[str, list[str]]:
    """校验整个题库。除逐条校验外，还检查题号唯一性与问句重复。"""
    report: dict[str, list[str]] = {}
    seen_ids: dict[str, int] = {}
    seen_questions: dict[str, str] = {}

    for item in items:
        problems = validate_item(item)
        if item.item_id in seen_ids:
            problems.append(f"item_id 重复（与第 {seen_ids[item.item_id] + 1} 条相同）")
        seen_ids.setdefault(item.item_id, len(seen_ids))

        key = (item.doc, item.question)
        if key in seen_questions and seen_questions[key] != item.item_id:
            problems.append(f"同一文档下问句重复（与 {seen_questions[key]} 相同）")
        seen_questions.setdefault(key, item.item_id)

        if problems:
            report[item.item_id or f"<第 {len(seen_ids)} 条>"] = problems
    return report
