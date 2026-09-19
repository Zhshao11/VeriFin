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
from verifin.guards import DocConstraints, check_question_constraints  # noqa: E402

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
        name="check_constraints",
        description="在检索前核对问句点名的主体与期间是否属于本文档。不属于则拒答，不做后续召回。",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "用户原问句"}
            },
            "required": ["question"],
            "additionalProperties": False,
        },
    ),
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
    ToolSpec(
        name="diff_two_rows",
        description=(
            "两行相减：取 A、B 两行（可跨口径）的原文数值，用 Decimal 求差的绝对差。"
            "覆盖「X 与 Y 的差额是多少」这类综合题。数值只来自原文，不接受外部金额。"
        ),
        parameters={
            "type": "object",
            "properties": {
                "label_a": {"type": "string", "description": "第一个科目名"},
                "scope_a": {"type": "string", "enum": ["合并", "母公司", None],
                            "description": "第一个科目的口径"},
                "label_b": {"type": "string", "description": "第二个科目名"},
                "scope_b": {"type": "string", "enum": ["合并", "母公司", None],
                            "description": "第二个科目的口径"},
                "period": {"type": "string", "enum": ["current", "prior"]},
            },
            "required": ["label_a", "label_b"],
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
    """科目名 → chunk（具备 .label / .values / .page / .text）。

    同一科目名在合并与母公司两套报表里各有一行，这里是**主口径**那一行：
    优先 `合并`，其次未标注口径，最后 `母公司`。
    只用于「问句没点名口径且该科目只有一个口径」的情形；
    要按口径精确取数请用 :meth:`chunk_for`。
    """
    by_label_scope: Mapping[tuple[str, str | None], Any] = field(default_factory=dict)
    """`(科目名, 口径)` → chunk。口径消歧的落点：同名科目两套报表都在这里。"""
    index: Any = None
    """检索索引（RetrievalIndex）；为 None 时 `search_statement` 走离线降级路径。"""
    unit: str = "元"
    company: str = ""
    period: str = ""
    pdf_open: Callable[[], Any] | None = None
    """返回 `verifin.geometry.PdfGeometry` 的可调用对象（支持 with 协议）。"""
    index_lock: Any = None
    company_aliases: tuple[str, ...] = ()
    """本文档主体的别名（发行人简称等）。用于**主体约束**：问句点名了别家主体就拒答。"""
    report_years: tuple[int, ...] = ()
    """本文档财务报表覆盖的年份（由表头日期推导）。用于**期间约束**。"""
    constraints: DocConstraints | None = None
    """主体 / 期间约束的整体。给了它就以它为准（上面的两个字段只在没给时兜底）。"""

    #: 内部缓存：全部已知科目名（含各口径）。`known_labels` 用。
    _labels_cache: tuple[str, ...] | None = field(
        default=None, repr=False, compare=False
    )

    @property
    def known_labels(self) -> tuple[str, ...]:
        """报表里出现过的**全部**科目名（含合并与母公司两份）。

        问答一致性护栏要在它上面做「问句点名了哪个科目」的匹配，
        所以必须包含所有口径 —— 漏掉母公司那套，会使得
        「母公司资产总计是多少」被判成"没点名任何科目"而误拒。
        """
        if self._labels_cache is None:
            names = {label for (label, _scope) in self.by_label_scope}
            names |= set(self.by_label)
            self._labels_cache = tuple(sorted(names))
        return self._labels_cache

    def effective_constraints(self) -> DocConstraints:
        """取约束：优先用显式传入的 `constraints`，否则用零散字段拼一个。"""
        if self.constraints is not None:
            return self.constraints
        return DocConstraints(
            company=self.company,
            company_aliases=tuple(self.company_aliases),
            report_years=tuple(self.report_years),
        )

    def resolve(self, name: str, scope: str | None = None) -> str | None:
        """把公式规范科目名对齐到报表实际行名。

        `scope` 给定时只认该口径下的行 —— 这正是口径消歧的关键：
        「资产总计」在两套报表里都有，不指定口径就等于没指定取哪一张表。
        """
        for candidate in (name, LABEL_ALIAS.get(name)):
            if not candidate:
                continue
            if scope is None:
                if candidate in self.by_label:
                    return candidate
            elif (candidate, scope) in self.by_label_scope:
                return candidate
        return None

    def scopes_for(self, label: str) -> tuple[str | None, ...]:
        """该科目在哪些口径下存在（已排序，便于确定性输出）。

        返回 `("合并", "母公司")` 表示两套报表都有 → 问句不点名口径时必须拒答，
        而不是"用先出现的那一行"含糊过去。
        """
        found = {s for (lbl, s) in self.by_label_scope if lbl == label}
        if not found and label in self.by_label:
            # 兼容只塞了 by_label 的极简运行时（测试用）
            found = {None}
        return tuple(sorted(found, key=lambda s: (s is None, s or "")))

    def chunk_for(self, label: str, scope: str | None = None) -> Any | None:
        """按口径取 chunk。`scope=None` 时退回主口径那一行。"""
        if scope is None:
            return self.by_label.get(label)
        return self.by_label_scope.get((label, scope))


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


def check_constraints(rt: ToolRuntime, *, question: str) -> dict[str, Any]:
    """工具 0：主体 / 期间约束。

    在**检索之前**跑，因为它要回答的是「这个问题本文档有没有资格回答」——
    没资格的话，后面召回得再准也是错的（问宁德时代却拿茅台的值来答，
    数值是真的、页码是真的，但答的是另一家公司）。

    这是纯程序判断，与模型无关，也不看问句的措辞风格。
    """
    violation = check_question_constraints(question, rt.effective_constraints())
    if violation is not None:
        return _fail(violation["error"], violation["detail"])
    return _ok(checked=True, company=rt.company, report_years=list(rt.report_years))


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
            "scope": getattr(h.chunk, "scope", None),
            "via": list(h.via),
            "score": round(h.score, 6),
        }
        for h in result.hits
    ]
    return _ok(hits=hits, per_channel=dict(result.per_channel))


def _resolve_scoped(
    rt: ToolRuntime, label: str, scope: str | None
) -> tuple[str | None, dict[str, Any] | None]:
    """把科目名对齐到某个口径下的报表行。

    Returns:
        `(报表行名, 失败返回值)` —— 成功时第二个元素为 `None`。
        把「口径消歧」这一步单独抽出来，是因为取行证据与勾稽取数**必须用同一套规则**：
        两处各写一遍，迟早出现「证据取自母公司、勾稽取自合并」这种更隐蔽的错配。

    三种失败要分清，它们指向完全不同的修法：

    | 错误码 | 含义 |
    |---|---|
    | `LABEL_NOT_FOUND` | 报表里根本没有这个科目名 |
    | `AMBIGUOUS_SCOPE` | 科目在两套报表里都有，但问句没说取哪一套 → **必须拒答** |
    | `SCOPE_NOT_AVAILABLE` | 问句点了口径，但该科目不在这个口径下 |
    """
    # 先判口径歧义：**问句没点名口径**，但同名科目在多套报表里都有
    # （合并 / 母公司数值不同）→ 必须拒答，不能"用先出现的那一行"含糊过去。
    # 这一步必须在 `resolve(label, None)` 之前，否则主口径兜底会把歧义吞掉，
    # 「资产总计是多少」就会被错误地答成合并值。
    if scope is None:
        scopes = rt.scopes_for(label)
        named = tuple(s for s in scopes if s is not None)
        if len(named) > 1:
            return None, _fail(
                "AMBIGUOUS_SCOPE",
                f"「{label}」在 {'、'.join(named)} 两套报表里都有，数值不同；"
                "问句未限定口径 → 拒答（不替用户挑一个）",
                requested_label=label,
                scopes=list(named),
            )

    actual = rt.resolve(label, scope)
    if actual is not None:
        return actual, None

    # 没解析出来：分清是"名称不存在"还是"口径对不上"。
    alias = rt.resolve(label)
    probe = alias or label
    scopes = rt.scopes_for(probe)
    named = tuple(s for s in scopes if s is not None)

    if scope is not None and scopes:
        available = "、".join(str(s) for s in scopes) or "未标注口径"
        return None, _fail(
            "SCOPE_NOT_AVAILABLE",
            f"「{probe}」不在{scope}口径下（该科目只出现在：{available}）",
            requested_label=label,
        )
    return None, _fail(
        "LABEL_NOT_FOUND", f"报表里没有科目「{label}」", requested_label=label
    )


def get_row_evidence(
    rt: ToolRuntime, *, label: str, period: str = "current", scope: str | None = None
) -> dict[str, Any]:
    """工具 2：取整行证据。数值只能来自原文。

    `scope` 由问句解析而来。**同名科目在两套报表里数值不同**，
    所以取数前必须先过一遍口径消歧（见 :func:`_resolve_scoped`）。
    """
    column = 0 if period == "current" else 1
    actual, failure = _resolve_scoped(rt, label, scope)
    if failure is not None:
        return failure
    assert actual is not None
    chunk = rt.chunk_for(actual, scope)
    if chunk is None:
        return _fail(
            "LABEL_NOT_FOUND", f"报表里没有科目「{label}」", requested_label=label
        )
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
        scope=getattr(chunk, "scope", None),
        fragment=_fragment(chunk.text, raw),
        chunk_text=chunk.text,
    )


def locate_in_pdf(
    rt: ToolRuntime, *, label: str, value: str, scope: str | None = None
) -> dict[str, Any]:
    """工具 3：坐标定位 + 同行校验。"""
    actual, failure = _resolve_scoped(rt, label, scope)
    if failure is not None:
        return failure
    assert actual is not None
    if rt.pdf_open is None:
        return _fail("NO_PDF", "未挂载 PDF，无法做坐标定位")

    chunk = rt.chunk_for(actual, scope)
    if chunk is None:
        return _fail("LABEL_NOT_FOUND", f"报表里没有科目「{label}」")
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
    rt: ToolRuntime, *, label: str, claimed_span: str, claimed_value: str,
    scope: str | None = None,
) -> dict[str, Any]:
    """工具 4：span 硬校验（两层关卡）。"""
    actual, failure = _resolve_scoped(rt, label, scope)
    if failure is not None:
        return failure
    assert actual is not None
    value = _to_decimal(claimed_value)
    if value is None:
        return _fail("VALUE_UNPARSED", f"声称的数值无法解析：{claimed_value!r}")

    chunk = rt.chunk_for(actual, scope)
    if chunk is None:
        return _fail("LABEL_NOT_FOUND", f"报表里没有科目「{label}」")
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


def compute(
    rt: ToolRuntime, *, formula_id: str, period: str = "current",
    scope: str | None = None,
) -> dict[str, Any]:
    """工具 6：勾稽核验。取数与运算都在工具内部，LLM 不碰算术。

    `scope` 一定要传：**同一恒等式在合并与母公司两套报表下都能平衡**
    （差额都是 0.00）。不限定口径就取数，等于让"取到真数、答错报表"
    也能得到 PASS —— 那是本项目最想防的错误。
    """
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
        actual, failure = _resolve_scoped(rt, name, scope)
        if failure is not None:
            # 口径歧义/不存在要如实上报，不能记成"缺操作数" ——
            # 后者会把「没说清取哪张表」误导成「报表里没这个科目」。
            if failure.get("error") in ("AMBIGUOUS_SCOPE", "SCOPE_NOT_AVAILABLE"):
                return failure
            missing.append(name)
            continue
        assert actual is not None
        chunk = rt.chunk_for(actual, scope)
        if chunk is None or len(chunk.values) <= column:
            missing.append(name)
            continue
        raw = chunk.values[column]
        val = _to_decimal(raw)
        if val is None:
            missing.append(name)
            continue
        operands[name] = val
        sources.append(
            {
                "科目": name,
                "报表行名": actual,
                "口径": getattr(chunk, "scope", None),
                "值": raw,
                "页码": chunk.page,
            }
        )

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


def _scope_suffix(scope: str | None) -> str:
    """口径后缀：画进「指标」字段里，让人一眼看出取的是哪张表。"""
    return f"（{scope}）" if scope else ""


def diff_two_rows(
    rt: ToolRuntime, *, label_a: str, scope_a: str | None,
    label_b: str, scope_b: str | None, period: str = "current",
) -> dict[str, Any]:
    """两行相减：取 A、B 两行（可跨口径）的原文数值，用 Decimal 精确求差。

    这是 D5 新增的「综合判断」动作，覆盖题库里「X 与 Y 的差额是多少」这一类
    —— 它们既不能归为单值抽取（要取两行），也不能用恒等式核验（没有等号）。

    红线不变：数值只来自原文行（`get_row_evidence`），差只走 Decimal，
    工具不接受任何外部传入的金额。返回的「指标」同时标出两套口径，
    同源头的「口径歧义必须拒答」形成对照 —— 这里问句**主动点名了两套口径**，
    所以不是歧义，是要算两者的差。
    """
    ea = get_row_evidence(rt, label=label_a, period=period, scope=scope_a)
    if not ea.get("ok"):
        return ea
    eb = get_row_evidence(rt, label=label_b, period=period, scope=scope_b)
    if not eb.get("ok"):
        return eb
    va = _to_decimal(ea["value"])
    vb = _to_decimal(eb["value"])
    if va is None or vb is None:
        return _fail("NON_NUMERIC", "任一操作数无法解析为数字")
    diff = abs(va - vb)
    indicator = (
        f"{ea['label']}{_scope_suffix(ea.get('scope'))}"
        f" − {eb['label']}{_scope_suffix(eb.get('scope'))}"
    )
    return _ok(
        ok=True,
        indicator=indicator,
        value=str(diff),
        decimal=str(diff),
        unit=rt.unit,
        operands=[
            {"科目": ea["label"], "口径": ea.get("scope"), "值": ea["value"],
             "页码": ea["page"], "片段": ea.get("fragment", "")},
            {"科目": eb["label"], "口径": eb.get("scope"), "值": eb["value"],
             "页码": eb["page"], "片段": eb.get("fragment", "")},
        ],
        pages=[ea["page"], eb["page"]],
        source=f"第 {ea['page']} 页 · 第 {eb['page']} 页",
        scope_label="跨口径" if scope_a != scope_b else (scope_a or "未标注"),
    )


#: 名字 → 实现。图的 EXECUTE 节点按这个名字分发。
TOOL_IMPLS: dict[str, Callable[..., dict[str, Any]]] = {
    "check_constraints": check_constraints,
    "search_statement": search_statement,
    "get_row_evidence": get_row_evidence,
    "locate_in_pdf": locate_in_pdf,
    "verify_span": verify_span,
    "list_formulas": list_formulas,
    "compute": compute,
    "diff_two_rows": diff_two_rows,
}

#: 每个工具默认算几次预算。用于成本核算与预算控制。
TOOL_NAMES: tuple[str, ...] = tuple(TOOL_IMPLS)
