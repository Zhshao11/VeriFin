"""Decimal 计算原语。

本模块承载项目的一条硬性红线：**LLM 不做任何算术。**

模型只被允许「选择公式」和「指出操作数来自哪条证据」，
真正的加减乘除一律在这里完成。理由有二：

1. 浮点误差。`0.1 + 0.2 != 0.3`，而财报核验里差额 0.01 就是「不平」。
   全程使用 :class:`decimal.Decimal`，禁止 `float` 参与任何金额运算。
2. 可复现性。同样是 `Decimal`，同样的输入必然得到同样的输出；
   交给模型心算则每次结果可能不同，评测无从谈起。

所有函数均为纯函数，除零等异常情况显式返回 `None` 而非抛异常，
以免中断整条核验链路——「算不出来」应当降级为「该项无法核验」，
而不是让整个任务失败。
"""

from __future__ import annotations

from decimal import (
    ROUND_HALF_UP,
    Decimal,
    DivisionByZero,
    InvalidOperation,
)

from verifin.normalize import UNIT_MULTIPLIER

__all__ = [
    "ZERO",
    "PCT_QUANTUM",
    "AMOUNT_QUANTUM",
    "to_yuan",
    "quantize_pct",
    "quantize_amount",
    "safe_div",
    "pct_change",
    "signed_pct_change",
]

ZERO = Decimal(0)

#: 百分比的量化精度：保留两位小数（即 0.01 个百分点）。
PCT_QUANTUM = Decimal("0.01")

#: 金额的量化精度：保留两位小数。
AMOUNT_QUANTUM = Decimal("0.01")


def to_yuan(value: Decimal, unit: str | None) -> Decimal:
    """把带单位的金额换算成「元」。

    Args:
        value: 按 `unit` 计的值。
        unit: 单位写法，如 `"万元"`。无法识别时按「元」处理。

    Note:
        无法识别的单位会被静默按「元」处理。调用方若需要区分
        「单位已知」与「单位缺失」，应同时检查
        :func:`verifin.normalize.is_known_unit`。
    """
    if not isinstance(value, Decimal):
        raise TypeError(
            f"金额必须使用 Decimal，收到 {type(value).__name__}。"
            "float 会引入二进制浮点误差，禁止参与金额运算。"
        )
    return value * UNIT_MULTIPLIER.get(unit or "", Decimal(1))


def quantize_pct(value: Decimal) -> Decimal:
    """把百分比量化到两位小数。"""
    return value.quantize(PCT_QUANTUM, rounding=ROUND_HALF_UP)


def quantize_amount(value: Decimal) -> Decimal:
    """把金额量化到两位小数。"""
    return value.quantize(AMOUNT_QUANTUM, rounding=ROUND_HALF_UP)


def safe_div(numerator: Decimal, denominator: Decimal) -> Decimal | None:
    """除法，分母为零或结果非法时返回 `None`。

    不使用异常控制流程：核验一条公式失败不应中断整批核验。
    """
    if denominator == 0:
        return None
    try:
        return numerator / denominator
    except (DivisionByZero, InvalidOperation):
        return None


def pct_change(current: Decimal, prior: Decimal) -> Decimal | None:
    """同比变化率，以百分点表示（`12.34` 表示 12.34%）。

    基期为负值或零时，常规百分比变化率没有财务意义
    （例如「亏损转亏损，变化率 -50%」），此时返回 `None`，
    由调用方降级处理为「不适用」而非硬算一个数字。

    Args:
        current: 本期值。
        prior: 上期（基期）值。

    Returns:
        变化率（百分点）；基期 <= 0 时返回 `None`。
    """
    if prior <= 0:
        return None
    ratio = safe_div(current - prior, prior)
    if ratio is None:
        return None
    return quantize_pct(ratio * Decimal(100))


def signed_pct_change(current: Decimal, prior: Decimal) -> Decimal | None:
    """同比变化率，保留符号语义，基期为负时也能算。

    与 :func:`pct_change` 的区别：用基期的绝对值做分母，
    使「-100 到 -50」这类变动可以表达为「+50%」而不是被丢弃。
    适用于现金流、利润等允许为负的科目。

    Args:
        current: 本期值。
        prior: 上期（基期）值。

    Returns:
        变化率（百分点）；基期为零时返回 `None`。
    """
    if prior == 0:
        return None
    ratio = safe_div(current - prior, abs(prior))
    if ratio is None:
        return None
    return quantize_pct(ratio * Decimal(100))
