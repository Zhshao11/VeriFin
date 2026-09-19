"""Agent 工具集。

六个工具的设计约束来自项目的三条红线：

| 红线 | 在工具层怎么落实 |
|---|---|
| LLM 不产生数字 | 工具的返回值里，数值一律从 chunk（即原文行）读取，工具**不接受**数值入参来"填"答案 |
| LLM 不做算术 | `compute` 工具只是套壳，真正的运算在 `formulas.evaluate_formula`，走 Decimal |
| LLM 无权决定拒答 | 工具一律返回"成 / 不成 + 原因"，是否拒答由图里的 `ASSEMBLE` 节点按规则判定 |

工具本身不做语义推断：给什么参数就执行什么，拿不到就说拿不到。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifin import span as span_mod  # noqa: E402
from verifin.formulas import FORMULA_REGISTRY, evaluate_formula  # noqa: E402

DISCLOSURE_UNITS = ("元", "千元", "万元", "百万元", "亿元")

#: 公式规范科目名 → 报表披露科目名（见留档 P-012）。
#: 只做精确命中与显式别名，**不做模糊包含** ——
#: 模糊包含会让「负债合计」落到「非流动负债合计」上，那正是本项目要防的张冠李戴。
LABEL_ALIAS: dict[str, str] = {
    "所有者权益合计": "所有者权益（或股东权益）合计",
    "归属于母公司股东的净利润": "归属于母公司所有者的净利润",
}


# --------------------------------------------------------------------------
# 规格
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """工具的对外声明。给 LLM 看的名字、描述、入参 JSON Schema。"""

    name: str
    description: str
    parameters: dict[str, Any]


def tool_specs() -> list[ToolSpec]:
    return list(_SPECS)


_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="search_statement",
        description="在年报里检索科目，返回候选行（科目名、页码、数值、命中了哪几路召回收）。只做检索，不做判断。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "待检索的问题或科目名"}
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="get_row_evidence",
        description="取某个科目的完整证据行：数值、单位、页码、原文片段。数值取自原文，工具不接受外部传入的数值。",
        parameters={
            "type": "object",
            "properties": {
                "label": {"type": "string", "description": "科目名，须与检索返回的候选一致"},
                "period": {"type": "string", "enum": ["current", "prior"], "description": "本期或上期"},
            },
            "required": ["label"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="locate_in_pdf",
        description="把某个科目的数值定位到 PDF 物理坐标（PDF point），并做同行校验。未通过校验时返回不采纳。",
        parameters={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "value": {"type": "string", "description": "数值原文，须来自 get_row_evidence"},
            },
            "required": ["label", "value"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="verify_span",
        description="span 硬校验两道关卡：片段是否存在于原文，以及数值能否从该片段解析出来。两道都过才算采纳。",
        parameters={
            "type": "object",
            "properties": {
                "label": {"type": "string"},
                "claimed_span": {"type": "string"},
                "claimed_value": {"type": "string"},
            },
            "required": ["label", "claimed_span", "claimed_value"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="list_formulas",
        description="列出可用的勾稽公式（恒等式 / 派生量 / 启发式）及其操作数，供后续取数核验。",
        parameters={
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["identity", "derived", "heuristic"]}
            },
            "required": [],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="compute",
        description="按公式 id 执行勾稽核验（Decimal 运算，容差由披露单位推导）。工具自己按公式的操作数取数，不接受外部传入金额。",
        parameters={
            "type": "object",
            "properties": {
                "formula_id": {"type": "string", "description": "如 F1 / F2a / F4"},
                "period": {"type": "string", "enum": ["current", "prior"]},
            },
            "required": ["formula_id"],
            "additionalProperties": False,
        },
    ),
)


# --------------------------------------------------------------------------
# 运行时
# --------------------------------------------------------------------------


@dataclass
class ToolRuntime:
    """工具执行所需的全部资源。

    刻意做成一个可注入的容器而不是去 globals 里拿：
    测试可以塞一个只有三条 chunk、没有 PDF 的运行时，
    从而在不碰真实年报的前提下验证 Agent 的行为。
    """

    by_label: Mapping[str, Any] = field(default_factory=dict)
    """科目名 → chunk（具备 .label / .values / .page / .text）。"""
    index: Any = None
    """检索索引（RetrievalIndex）；为 None 时 `search_statement` 走离线降级路径。"""
    unit: str = "元"
    company: str = ""
    period: str = ""
    pdf_open: Callable[[], Any] | None = None
    """返回 `verifin.geometry.PdfGeometry` 的可调用对象（支持 with 协议）。"""
    index_lock: Any = None

    def resolve(self, name: str) -> str | None:
        """把公式规范科目名对齐到报表实际行名。"""
        if name in self.by_label:
            return name
        alias = LABEL_ALIAS.get(name)
        if alias and alias in self.by_label:
            return alias
        return None


def _fail(code: str, detail: str, **extra: Any) -> dict[str, Any]:
    """失败的返回值。`extra` 用来附上**调度器需要、但不能靠猜**的上下文。

    目前唯一的用法是 `requested_label`：取行证据失败时，调度器要靠它把这个
    科目记为「已尝试」，否则会反复重试同一个候选直到预算耗尽
    （该缺陷由 D4 评测打出来，详见 `_absorb`）。
    """
    return {"ok": False, "error": code, "detail": detail, **extra}


def _ok(**data: Any) -> dict[str, Any]:
    return {"ok": True, **data}


def _to_decimal(raw: str) -> Decimal | None:
    cleaned = (raw or "").replace(",", "").replace("，", "").strip()
    if not cleaned:
        return None
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def _fragment(chunk_text: str, raw_value: str, lead: int = 40) -> str:
    """从原文里截一段包含该数值的**真实子串**。

    不能按「科目 | 数值」去拼：资产负债表有缩进行名列
    （如「所有者权益（或股东权益）:」），拼出来的串原文里根本没有。
    """
    pos = chunk_text.find(raw_value)
    if pos < 0:
        return raw_value
    start = max(0, pos - lead)
    seps = [i for i, ch in enumerate(chunk_text) if ch == "|" and start <= i < pos]
    if len(seps) >= 2:
        start = seps[-2] + 1
    return chunk_text[start:pos + len(raw_value)].strip().lstrip("|").strip()


# --------------------------------------------------------------------------
# 六个工具
# --------------------------------------------------------------------------


def search_statement(rt: ToolRuntime, *, query: str, top_k: int = 5) -> dict[str, Any]:
    """工具 1：科目检索（委托给检索层，工具本身不排序不筛选）。"""
    from verifin.retrieval import RetrievalIndex  # 局部导入：避免循环依赖

    index = rt.index
    if not isinstance(index, RetrievalIndex):
        # 没有真实索引时，退化成按科目名包含与否在 by_label 里找。
        # 这条降级路径是为测试与离线场景准备的，裁决力量表记录在返回值里。
        hits = [
            {"label": lbl, "page": c.page, "via": ["label"],
             "value": c.values[0] if c.values else None}
            for lbl, c in rt.by_label.items()
            if query and (query in lbl or lbl in query)
        ][:top_k]
        if not hits:
            return _fail("NO_RECALL", f"未命中任何科目：{query!r}")
        return _ok(hits=hits, degraded="no_index")

    lock = rt.index_lock
    if lock is not None:
        with lock:
            result = index.retrieve(query, top_k=top_k)
    else:
        result = index.retrieve(query, top_k=top_k)

    if result.is_empty:
        return _fail("NO_RECALL", f"四路召回均无命中：{query!r}")

    hits = [
        {
            "label": h.chunk.label,
            "page": h.chunk.page,
            "value": h.chunk.values[0] if h.chunk.values else None,
            "via": list(h.via),
            "score": round(h.score, 6),
        }
        for h in result.hits
    ]
    return _ok(hits=hits, per_channel=dict(result.per_channel))


def get_row_evidence(
    rt: ToolRuntime, *, label: str, period: str = "current"
) -> dict[str, Any]:
    """工具 2：取整行证据。数值只能来自原文。"""
    column = 0 if period == "current" else 1
    actual = rt.resolve(label)
    if actual is None:
        return _fail(
            "LABEL_NOT_FOUND", f"报表里没有科目「{label}」", requested_label=label
        )
    chunk = rt.by_label[actual]
    if len(chunk.values) <= column:
        return _fail(
            "COLUMN_MISSING",
            f"「{actual}」没有{'本期' if column == 0 else '上期'}列",
            requested_label=label,
        )
    raw = chunk.values[column]
    value = _to_decimal(raw)
    if value is None:
        return _fail(
            "VALUE_UNPARSED",
            f"「{actual}」该列为空或无法解析：{raw!r}",
            requested_label=label,
        )
    return _ok(
        label=actual,
        requested_label=label,
        value=raw,
        decimal=str(value),
        unit=rt.unit,
        page=chunk.page,
        fragment=_fragment(chunk.text, raw),
        chunk_text=chunk.text,
    )


def locate_in_pdf(rt: ToolRuntime, *, label: str, value: str) -> dict[str, Any]:
    """工具 3：坐标定位 + 同行校验。"""
    actual = rt.resolve(label)
    if actual is None:
        return _fail("LABEL_NOT_FOUND", f"报表里没有科目「{label}」")
    if rt.pdf_open is None:
        return _fail("NO_PDF", "未挂载 PDF，无法做坐标定位")

    chunk = rt.by_label[actual]
    with rt.pdf_open() as geo:
        loc = geo.locate_row(chunk.page, actual, value)
        if not loc.same_row_verified or loc.row_box is None:
            return _fail(
                "SAME_ROW_FAILED",
                "数值与科目名不在同一行 —— 可能是张冠李戴，坐标不采纳",
            )
        box = loc.row_box
        return _ok(
            page=chunk.page,
            bbox=[round(v, 1) for v in box.astuple()],
            verified=True,
        )


def verify_span(
    rt: ToolRuntime, *, label: str, claimed_span: str, claimed_value: str
) -> dict[str, Any]:
    """工具 4：span 硬校验（两层关卡）。"""
    actual = rt.resolve(label)
    if actual is None:
        return _fail("LABEL_NOT_FOUND", f"报表里没有科目「{label}」")
    value = _to_decimal(claimed_value)
    if value is None:
        return _fail("VALUE_UNPARSED", f"声称的数值无法解析：{claimed_value!r}")

    chunk = rt.by_label[actual]
    verdict = span_mod.verify_evidence(
        claimed_span=claimed_span,
        chunk_text=chunk.text,
        claimed_value=value,
        claimed_unit=rt.unit,
    )
    return _ok(
        accepted=verdict.accepted,
        reason=verdict.reason,
        mode=verdict.match_mode.value if verdict.match_mode else None,
        detail=verdict.detail,
    )


def list_formulas(rt: ToolRuntime, *, kind: str | None = None) -> dict[str, Any]:
    """工具 5：列出公式注册表里的公式与它们需要的操作数。"""
    items = []
    for f in FORMULA_REGISTRY.values():
        if kind and f.kind.value != kind:
            continue
        items.append(
            {
                "id": f.id,
                "name": f.name,
                "kind": f.kind.value,
                "operands": list(f.operand_names),
                "description": f.description,
            }
        )
    if not items:
        return _fail("NO_FORMULA", f"没有类型为 {kind!r} 的公式")
    return _ok(formulas=items)


def compute(rt: ToolRuntime, *, formula_id: str, period: str = "current") -> dict[str, Any]:
    """工具 6：勾稽核验。取数与运算都在工具内部，LLM 不碰算术。"""
    if formula_id not in FORMULA_REGISTRY:
        return _fail("FORMULA_NOT_FOUND", f"注册表里没有公式 {formula_id!r}")
    formula = FORMULA_REGISTRY[formula_id]
    column = 0 if period == "current" else 1

    operands: dict[str, Decimal] = {}
    sources: list[dict[str, Any]] = []
    missing: list[str] = []
    # 用 required_operands 而不是 operand_names：后者含左值，
    # 对派生量公式（F4 毛利率）会去报表里找「毛利率」这一行，永远找不到。
    for name in formula.required_operands:
        actual = rt.resolve(name)
        chunk = rt.by_label.get(actual) if actual else None
        if chunk is None or len(chunk.values) <= column:
            missing.append(name)
            continue
        raw = chunk.values[column]
        val = _to_decimal(raw)
        if val is None:
            missing.append(name)
            continue
        operands[name] = val
        sources.append({"科目": name, "报表行名": actual, "值": raw, "页码": chunk.page})

    if missing:
        return _fail(
            "MISSING_OPERANDS",
            f"缺 {'、'.join(missing)}，无法核验（规范名与报表行名是否对齐？）",
        )

    outcome = evaluate_formula(formula_id, operands, disclosure_unit=rt.unit)
    return _ok(
        formula_id=outcome.formula_id,
        formula_name=outcome.formula_name,
        verdict=outcome.verdict.value,
        lhs=str(outcome.lhs),
        rhs=str(outcome.rhs),
        diff=str(outcome.diff),
        tolerance=str(outcome.tolerance),
        operands={k: str(v) for k, v in operands.items()},
        sources=sources,
        detail=outcome.detail,
    )


#: 名字 → 实现。图的 EXECUTE 节点按这个名字分发。
TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "search_statement": search_statement,
    "get_row_evidence": get_row_evidence,
    "locate_in_pdf": locate_in_pdf,
    "verify_span": verify_span,
    "list_formulas": list_formulas,
    "compute": compute,
}

#: 每个工具默认算几次预算。用于成本核算与预算控制。
TOOL_NAMES: tuple[str, ...] = tuple(TOOL_IMPLS)
