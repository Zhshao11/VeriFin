"""财务勾稽公式注册表。

设计要点（这几条都是最容易被追问的地方）：

1. **公式定义与计算分离。** 模型只输出「选哪个公式」和「操作数来自哪条证据」，
   算术全在本模块完成。操作数必须是已通过 span 校验的证据值。

2. **核验结论分三级**，而不是二值的通过/不通过：

   - `PASS`：恒等式成立，或派生量在容差内一致。
   - `WARN`：可疑但无法判定为错。用于启发式检查（如同比异常）。
   - `FAIL`：恒等式不平衡。这是确定性的错误。

   区分 `WARN` 与 `FAIL` 是有必要的：同比变动 300% 可能是并购重组导致的
   真实变化，把它判成 `FAIL` 就是误报。而资产 ≠ 负债 + 权益 无论什么原因
   都说明数据或抽取有问题。

3. **容差由报表披露单位推导，不硬编码。**
   报表以万元披露时，每个科目有 ±0.5 万元的舍入误差；恒等式涉及 n 个科目，
   误差可累积到 ±(n × 0.5) 万元。写死 `abs(diff) < 0.01` 会让所有恒等式误报。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum

from verifin.compute import ZERO, pct_change, quantize_pct, safe_div
from verifin.normalize import unit_scale

__all__ = [
    "FormulaKind",
    "Verdict",
    "Formula",
    "Outcome",
    "FORMULA_REGISTRY",
    "derive_tolerance",
    "check_sum_identity",
    "check_gross_margin",
    "check_cross_currency",
    "check_yoy_reasonableness",
    "evaluate_formula",
    "suspect_operands",
]

#: 报表数字四舍五入到披露单位时，单个科目的最大舍入误差占披露单位的比例。
#: 以万元披露时，每个科目最多偏离真实值 0.5 万元，故系数为 0.5。
ROUNDING_HALF = Decimal("0.5")

#: 同比合理性告警阈值（百分点）。超过则触发 WARN，不判 FAIL。
DEFAULT_YOY_THRESHOLD_PCT = Decimal("300")


class FormulaKind(str, Enum):
    """公式类型。决定异常时的结论等级。"""

    IDENTITY = "identity"
    """恒等式。左右不平衡即为 `FAIL`。"""

    DERIVED = "derived"
    """派生量。计算值与声称值不符即为 `FAIL`。"""

    HEURISTIC = "heuristic"
    """启发式检查。异常只判 `WARN`，不能判 `FAIL`。"""


class Verdict(str, Enum):
    """核验结论。三级而非二值。"""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass(frozen=True)
class Outcome:
    """一次公式核验的结论。

    Attributes:
        formula_id: 公式标识。
        formula_name: 公式中文名。
        kind: 公式类型。
        verdict: 三级结论。
        lhs: 左侧实际值；缺操作数时为 `None`。
        rhs: 右侧合计值；缺操作数时为 `None`。
        diff: `lhs - rhs`；缺操作数时为 `None`。
        tolerance: 本次核验采用的容差（绝对量）。
        missing_operands: 缺失的科目名。非空时结论为 `WARN`。
        suspects: 可能的矛盾字段（启发式排序，见 :func:`suspect_operands`）。
        detail: 可读说明，直接写入核验报告。
    """

    formula_id: str
    formula_name: str
    kind: FormulaKind
    verdict: Verdict
    lhs: Decimal | None
    rhs: Decimal | None
    diff: Decimal | None
    tolerance: Decimal
    missing_operands: tuple[str, ...] = ()
    suspects: tuple[str, ...] = ()
    detail: str = ""


@dataclass(frozen=True)
class Formula:
    """一条勾稽公式的定义。

    `lhs` 与 `rhs` 描述形如「左值 = Σ(符号 × 右项)」的恒等式。
    右侧每项带符号，以便表达减法（如 净利润 = 利润总额 − 所得税费用）。

    带自定义求值器的公式（F4/F5/F6）不是简单的求和恒等式，
    用 `param_map` 把注册表里的中文科目名映射到求值器的参数名，
    使 `evaluate_formula` 的调用方式对所有公式保持一致。
    """

    id: str
    name: str
    kind: FormulaKind
    lhs: str
    rhs: tuple[tuple[str, int], ...]
    description: str
    common_pitfalls: str = ""
    often_omitted: tuple[str, ...] = ()
    """本公式经常被漏取的关联行。这些行的值通常**不在操作数里**，
    因此无法靠 :func:`suspect_operands` 定位，只能作为人工核查线索给出。"""
    evaluator: Callable[..., Outcome] | None = field(default=None, repr=False)
    param_map: Mapping[str, str] | None = field(default=None, repr=False)

    @property
    def operand_names(self) -> tuple[str, ...]:
        """本公式需要的全部科目名，含左值。"""
        return (self.lhs, *(name for name, _ in self.rhs))

    @property
    def required_operands(self) -> tuple[str, ...]:
        """**必须从报表里取到**的科目名。

        带自定义求值器的公式以 `param_map` 为准 —— 那里的键才是求值器真正要的输入。

        为什么不能直接用 :attr:`operand_names`：它把左值也算进来，而派生量的左值
        是我们要求出来的东西（如 F4 的「毛利率」），报表上根本没有这一行。
        用 `operand_names` 去取数，结果是「毛利率永远算不出来」，
        且报错说「缺科目：毛利率」—— 把「报表没有派生量行」这个事实
        说成了「数据缺失」，指错了排查方向。
        """
        if self.evaluator is not None and self.param_map:
            return tuple(self.param_map)
        return self.operand_names

    @property
    def operand_count(self) -> int:
        """参与运算的科目数。容差按此推导。"""
        return 1 + len(self.rhs)


def derive_tolerance(operand_count: int, disclosure_unit: str | None) -> Decimal:
    """由报表披露单位推导容差。

    公式：`容差 = 参与科目数 × 0.5 × 披露单位倍数`

    Args:
        operand_count: 恒等式涉及几个科目。科目越多，舍入误差累积越大。
        disclosure_unit: 报表口径声明的单位，如 `"万元"`。未知时按「元」处理。

    Returns:
        绝对容差（单位：元）。

    Example:
        报表以万元披露，核验「资产 = 负债 + 所有者权益」共 3 个科目::

            derive_tolerance(3, "万元")   # -> Decimal('15000.0')

        即三个科目各最多偏 5000 元，合计最多偏 15000 元。
    """
    if operand_count <= 0:
        raise ValueError(f"科目数必须为正整数，收到 {operand_count}")
    return Decimal(operand_count) * ROUNDING_HALF * unit_scale(disclosure_unit)


#: 可疑科目的相对贴近阈值。差额与该科目量级相差在此比例以内才算「量级相当」。
#: 设得过松会把所有科目都报成可疑，等于没给线索；0.2 表示允许 20% 偏差。
SUSPECT_THRESHOLD = Decimal("0.2")


def suspect_operands(
    operands: Mapping[str, Decimal], diff: Decimal
) -> tuple[str, ...]:
    """定位可能的矛盾字段（启发式）。

    两个判据，都锚定在财报里真实高频的两类错误上：

    1. **整体漏记或错记**：某个科目没被读到，或读成了另一个科目的值。
       此时 `|diff|` 会接近该科目的量级。
    2. **十进制错位**：万元当成元、亿元当成万元这类单位错认。
       此时 `|diff|` 接近该科目量级的 9 倍（如 100 被读成 1000，差 900）。

    Args:
        operands: 参与运算的科目及其值。
        diff: 恒等式的左右差额。

    Returns:
        按可疑程度排序的科目名（最多 3 个）。**没有任何科目的量级能解释
        该差额时返回空元组**——这时应如实说明「无法从现有操作数定位」，
        而不是随便挑一个科目充数。

    Note:
        这是启发式排序，用于给人工核查提供线索，**不是归因结论**。
        报告里应标注为「待核查」，不能写成「该字段有误」。
    """
    if diff == 0 or not operands:
        return ()

    magnitude = abs(diff)
    scored: list[tuple[Decimal, str]] = []
    for name, value in operands.items():
        size = abs(value)
        if size == 0:
            continue
        # 判据 1：差额与该科目量级相当（该科目可能整体漏记或记错）。
        score_missing = abs(magnitude - size) / size
        # 判据 2：差额约为该科目量级的 9 倍（十进制错位）。
        score_shift = abs(magnitude - size * 9) / (size * 9)
        best = min(score_missing, score_shift)
        if best <= SUSPECT_THRESHOLD:
            scored.append((best, name))

    if not scored:
        return ()

    scored.sort()
    return tuple(name for _, name in scored[:3])


def check_sum_identity(
    formula: Formula,
    operands: Mapping[str, Decimal],
    *,
    disclosure_unit: str | None,
    tolerance: Decimal | None = None,
) -> Outcome:
    """核验形如「左值 = Σ(符号 × 右项)」的恒等式。

    Args:
        formula: 公式定义。
        operands: 科目名到值的映射（值已由证据层换算为「元」）。
        disclosure_unit: 报表披露单位，用于推导容差。
        tolerance: 显式指定容差。为 `None` 时按披露单位推导。

    缺失任一操作数时不做核验，直接返回 `WARN`——「无法核验」与
    「核验不通过」是两件事，不能混为一谈。
    """
    needed = formula.operand_names
    missing = tuple(name for name in needed if name not in operands)
    tol = (
        tolerance
        if tolerance is not None
        else derive_tolerance(formula.operand_count, disclosure_unit)
    )

    if missing:
        return Outcome(
            formula_id=formula.id,
            formula_name=formula.name,
            kind=formula.kind,
            verdict=Verdict.WARN,
            lhs=None,
            rhs=None,
            diff=None,
            tolerance=tol,
            missing_operands=missing,
            detail=f"缺少科目：{'、'.join(missing)}，无法核验。",
        )

    lhs = operands[formula.lhs]
    rhs = sum(
        (operands[name] * sign for name, sign in formula.rhs),
        ZERO,
    )
    diff = lhs - rhs

    if abs(diff) <= tol:
        return Outcome(
            formula_id=formula.id,
            formula_name=formula.name,
            kind=formula.kind,
            verdict=Verdict.PASS,
            lhs=lhs,
            rhs=rhs,
            diff=diff,
            tolerance=tol,
            detail=(
                f"平衡。差额 {diff} 在容差 {tol} 以内"
                f"（按披露单位「{disclosure_unit or '元'}」推导）。"
            ),
        )

    suspects = suspect_operands(operands, diff)
    hints: list[str] = []
    if suspects:
        hints.append(f"待核查科目：{'、'.join(suspects)}。")
    else:
        hints.append(
            f"现有操作数的量级均无法解释差额 {abs(diff)}，"
            "说明出错的可能是一个**不在本次核验操作数里**的项目。"
        )
    if formula.often_omitted:
        hints.append(
            f"请优先核对原文中的：{'、'.join(formula.often_omitted)}（本公式常见漏项）。"
        )
    if formula.common_pitfalls:
        hints.append(f"常见成因：{formula.common_pitfalls}")

    return Outcome(
        formula_id=formula.id,
        formula_name=formula.name,
        kind=formula.kind,
        verdict=Verdict.FAIL,
        lhs=lhs,
        rhs=rhs,
        diff=diff,
        tolerance=tol,
        suspects=suspects,
        detail=(
            f"不平衡。左值 {lhs}，右值合计 {rhs}，差额 {diff}，超出容差 {tol}。"
            + " ".join(hints)
        ),
    )


def check_gross_margin(
    *,
    revenue: Decimal,
    cost: Decimal,
    claimed_pct: Decimal | None = None,
    tolerance: Decimal = Decimal("0.01"),
) -> Outcome:
    """核验毛利率。

    毛利率 = (营业收入 − 营业成本) / 营业收入 × 100

    Args:
        revenue: 营业收入（元）。
        cost: 营业成本（元）。
        claimed_pct: 被核验的毛利率声称值（百分点）。为 `None` 时只做计算。
        tolerance: 比较容差，默认 0.01 个百分点（即两位小数）。
    """
    formula = FORMULA_REGISTRY["F4"]
    gross = revenue - cost
    ratio = safe_div(gross, revenue)

    if ratio is None:
        return Outcome(
            formula_id=formula.id,
            formula_name=formula.name,
            kind=formula.kind,
            verdict=Verdict.WARN,
            lhs=None,
            rhs=None,
            diff=None,
            tolerance=tolerance,
            missing_operands=("营业收入",),
            detail="营业收入为零，毛利率无定义。",
        )

    computed = quantize_pct(ratio * Decimal(100))

    if claimed_pct is None:
        return Outcome(
            formula_id=formula.id,
            formula_name=formula.name,
            kind=formula.kind,
            verdict=Verdict.PASS,
            lhs=computed,
            rhs=None,
            diff=None,
            tolerance=tolerance,
            detail=f"毛利率计算值 {computed}%。",
        )

    diff = computed - claimed_pct
    verdict = Verdict.PASS if abs(diff) <= tolerance else Verdict.FAIL
    return Outcome(
        formula_id=formula.id,
        formula_name=formula.name,
        kind=formula.kind,
        verdict=verdict,
        lhs=computed,
        rhs=claimed_pct,
        diff=diff,
        tolerance=tolerance,
        detail=(
            f"毛利率计算值 {computed}%，声称值 {claimed_pct}%，"
            f"差额 {diff}（容差 {tolerance}）。"
        ),
    )


def check_cross_currency(
    *,
    foreign_amount: Decimal,
    local_amount: Decimal,
    rate: Decimal,
    tolerance: Decimal | None = None,
) -> Outcome:
    """核验跨币种换算一致性。

    判据：`外币金额 × 汇率` 应与「按本币列示的金额」一致。

    Args:
        foreign_amount: 外币金额（外币单位）。
        local_amount: 同一数值在本币报表中的列示金额（元）。
        rate: 该期间的折算汇率。
        tolerance: 容差。为 `None` 时按 `max(1, local_amount) × 0.5%` 估算，
            因为汇率本身的精度与折算口径会带来小幅差异。

    Note:
        汇率必须来自本地汇率表，**不得写进提示词**——写进提示词意味着
        模型有办法影响它，等于把「数字可追溯」的红线开了个口子。
    """
    formula = FORMULA_REGISTRY["F5"]
    converted = quantize_pct(foreign_amount * rate)
    tol = (
        tolerance
        if tolerance is not None
        else (abs(local_amount) * Decimal("0.005")).quantize(Decimal("0.01"))
    )
    diff = converted - local_amount
    verdict = Verdict.PASS if abs(diff) <= tol else Verdict.FAIL

    return Outcome(
        formula_id=formula.id,
        formula_name=formula.name,
        kind=formula.kind,
        verdict=verdict,
        lhs=converted,
        rhs=local_amount,
        diff=diff,
        tolerance=tol,
        detail=(
            f"换算值 {converted}，本币列示 {local_amount}，"
            f"差额 {diff}（容差 {tol}，汇率 {rate}）。"
        ),
    )


def check_yoy_reasonableness(
    *,
    current: Decimal,
    prior: Decimal,
    threshold_pct: Decimal = DEFAULT_YOY_THRESHOLD_PCT,
) -> Outcome:
    """同比变动合理性检查（启发式）。

    触发 `WARN` 的两种情况：

    1. 变动幅度超过阈值。
    2. **符号翻转**：由盈转亏或由亏转盈。这种变化在财务上未必异常，
       但值得人工确认，属于典型的「需要解释」而非「需要报错」。

    Args:
        current: 本期值。
        prior: 上期值。
        threshold_pct: 告警阈值（百分点），默认 300。

    Important:
        本项**永远只返回 `PASS` 或 `WARN`，不会返回 `FAIL`**。
        同比变动不是恒等式，大幅变动可能是并购、重组、资产处置等
        真实经营事项所致，判定为错误即为误报。
    """
    formula = FORMULA_REGISTRY["F6"]
    change = pct_change(current, prior)

    if change is None:
        return Outcome(
            formula_id=formula.id,
            formula_name=formula.name,
            kind=formula.kind,
            verdict=Verdict.WARN,
            lhs=current,
            rhs=prior,
            diff=None,
            tolerance=threshold_pct,
            detail=f"基期值 {prior} 非正，百分比变化率不适用，需人工核对。",
        )

    sign_flipped = (current < 0) != (prior < 0)
    reasons: list[str] = []
    if abs(change) > threshold_pct:
        reasons.append(f"变动幅度 {change}% 超出阈值 {threshold_pct}%")
    if sign_flipped:
        reasons.append("发生盈亏方向翻转")

    if reasons:
        return Outcome(
            formula_id=formula.id,
            formula_name=formula.name,
            kind=formula.kind,
            verdict=Verdict.WARN,
            lhs=current,
            rhs=prior,
            diff=change,
            tolerance=threshold_pct,
            detail="；".join(reasons) + "，需人工确认是否由并购、重组等真实事项导致。",
        )

    return Outcome(
        formula_id=formula.id,
        formula_name=formula.name,
        kind=formula.kind,
        verdict=Verdict.PASS,
        lhs=current,
        rhs=prior,
        diff=change,
        tolerance=threshold_pct,
        detail=f"同比变动 {change}%，在合理范围内。",
    )


#: 公式注册表。模型的职责只是输出这里的 `id`，不参与任何计算。
FORMULA_REGISTRY: dict[str, Formula] = {
    "F1": Formula(
        id="F1",
        name="资产 = 负债 + 所有者权益",
        kind=FormulaKind.IDENTITY,
        lhs="资产总计",
        rhs=(("负债合计", 1), ("所有者权益合计", 1)),
        description=(
            "资产负债表基本恒等式。注意合并报表口径下，「所有者权益合计」"
            "应包含少数股东权益；若只取「归属于母公司所有者权益」需另加该项。"
        ),
        common_pitfalls=(
            "只取了「归属于母公司所有者权益合计」而漏加「少数股东权益」；"
            "或混用了合并报表与母公司报表两种口径。"
        ),
        often_omitted=("少数股东权益", "归属于母公司所有者权益合计"),
    ),
    "F2a": Formula(
        id="F2a",
        name="净利润 = 利润总额 − 所得税费用",
        kind=FormulaKind.IDENTITY,
        lhs="净利润",
        rhs=(("利润总额", 1), ("所得税费用", -1)),
        description="利润表内部恒等式，容差可设得很小。",
        common_pitfalls="把「所得税费用」误取为资产负债表上的「应交税费」；或误用母公司报表数据。该项目在利润表下方，容易与「净利润」行错位。",
    ),
    "F2b": Formula(
        id="F2b",
        name="归属于母公司股东的净利润 = 净利润 − 少数股东损益",
        kind=FormulaKind.IDENTITY,
        lhs="归属于母公司股东的净利润",
        rhs=(("净利润", 1), ("少数股东损益", -1)),
        description="合并报表口径拆分恒等式。",
        common_pitfalls="漏取「少数股东损益」。该项目可能为负数（少数股东分担亏损），取绝对值会直接导致不平衡。",
        often_omitted=("少数股东损益",),
    ),
    "F3a": Formula(
        id="F3a",
        name="现金及现金等价物净增加额 = 经营 + 投资 + 筹资 + 汇率影响",
        kind=FormulaKind.IDENTITY,
        lhs="现金及现金等价物净增加额",
        rhs=(
            ("经营活动产生的现金流量净额", 1),
            ("投资活动产生的现金流量净额", 1),
            ("筹资活动产生的现金流量净额", 1),
            ("汇率变动对现金及现金等价物的影响", 1),
        ),
        description="现金流量表分项加总恒等式。部分报表无「汇率影响」行，缺失时判 WARN。",
        common_pitfalls="部分报表没有「汇率变动对现金及现金等价物的影响」独立行，抽取时被当成 0，此时差额恰好等于该行金额。",
        often_omitted=("汇率变动对现金及现金等价物的影响",),
    ),
    "F3b": Formula(
        id="F3b",
        name="期末现金及现金等价物余额 = 期初余额 + 净增加额",
        kind=FormulaKind.IDENTITY,
        lhs="期末现金及现金等价物余额",
        rhs=(("期初现金及现金等价物余额", 1), ("现金及现金等价物净增加额", 1)),
        description="现金流量表期末结存恒等式。",
        common_pitfalls="期初余额误取了「年初余额」，两者在有会计政策变更或企业合并时会不同。",
    ),
    "F4": Formula(
        id="F4",
        name="毛利率 = (营业收入 − 营业成本) / 营业收入 × 100%",
        kind=FormulaKind.DERIVED,
        lhs="毛利率",
        rhs=(("营业收入", 1), ("营业成本", -1)),
        description="派生量核验。容差按百分点计，默认 0.01。",
        common_pitfalls="营业成本误取为「营业总成本」。「营业总成本」还包含税金及附加、销售/管理/研发/财务费用，用它算出的毛利率会明显偏低。",
        evaluator=check_gross_margin,
        param_map={"营业收入": "revenue", "营业成本": "cost"},
    ),
    "F5": Formula(
        id="F5",
        name="跨币种换算一致性",
        kind=FormulaKind.IDENTITY,
        lhs="本币列示金额",
        rhs=(("外币金额", 1), ("折算汇率", 1)),
        description="汇率来自本地汇率表，不进入提示词。",
        common_pitfalls=(
            "汇率错用了期末汇率，而报表折算用的是期间平均汇率；"
            "或把「原币金额」与「折算后本币金额」两列取混了。"
        ),
        evaluator=check_cross_currency,
        param_map={
            "本币列示金额": "local_amount",
            "外币金额": "foreign_amount",
            "折算汇率": "rate",
        },
    ),
    "F6": Formula(
        id="F6",
        name="同比变动合理性",
        kind=FormulaKind.HEURISTIC,
        lhs="本期值",
        rhs=(("上期值", 1),),
        description="启发式检查，异常只判 WARN，永不判 FAIL。",
        evaluator=check_yoy_reasonableness,
        param_map={"本期值": "current", "上期值": "prior"},
    ),
}


def evaluate_formula(
    formula_id: str,
    operands: Mapping[str, Decimal],
    *,
    disclosure_unit: str | None = None,
    **kwargs: object,
) -> Outcome:
    """按公式 id 执行核验。

    Args:
        formula_id: :data:`FORMULA_REGISTRY` 中的键。
        operands: 科目名到值的映射，值须为 :class:`~decimal.Decimal`。
        disclosure_unit: 报表披露单位，用于推导容差。
        **kwargs: 透传给自定义求值器的额外参数（如 F4 的 `claimed_pct`）。

    Raises:
        KeyError: 公式 id 不存在。
        TypeError: 操作数中出现 `float`。这是刻意的——
            宁可显式报错，也不允许浮点污染金额运算。

    Note:
        科目名一律使用注册表里的中文名（如 `"营业收入"`），
        由公式自带的 `param_map` 翻译成求值器的参数名，
        调用方不需要知道求值器的签名。
    """
    formula = FORMULA_REGISTRY[formula_id]

    for name, value in operands.items():
        if isinstance(value, float):
            raise TypeError(
                f"科目「{name}」使用了 float（{value!r}）。"
                "金额运算必须使用 Decimal。"
            )

    if formula.evaluator is None:
        return check_sum_identity(
            formula,
            operands,
            disclosure_unit=disclosure_unit,
            tolerance=kwargs.get("tolerance"),  # type: ignore[arg-type]
        )

    # 自定义求值器路径：先按 param_map 翻译，并在调用前检查操作数完整性。
    param_map = formula.param_map or {}
    missing = tuple(name for name in param_map if name not in operands)
    if missing:
        return Outcome(
            formula_id=formula.id,
            formula_name=formula.name,
            kind=formula.kind,
            verdict=Verdict.WARN,
            lhs=None,
            rhs=None,
            diff=None,
            tolerance=kwargs.get("tolerance", ZERO),  # type: ignore[arg-type]
            missing_operands=missing,
            detail=f"缺少科目：{'、'.join(missing)}，无法核验。",
        )

    # 只翻译 param_map 里列出的科目。多余的键（如调用方顺手带上的左值）
    # 直接忽略 —— 早先这里是 `param_map[name]`，多带一个键就 KeyError，
    # 把「调用方多给了个操作数」炸成「未捕获异常」。
    translated = {
        param_map[name]: value for name, value in operands.items() if name in param_map
    }
    translated.update(kwargs)
    return formula.evaluator(**translated)
