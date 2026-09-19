"""文本与数字归一化。

本模块是系统「数字精确匹配」能力的地基。

为什么必须做归一化：embedding 会把 `12,345.67`、`12345.67`、`1.23亿元`
当成三个互不相关的词串，因此纯向量检索在找财务数字时必然失败。
进入索引前必须先做确定性归一化，把等价写法映射到同一个规范形式。

设计约定：
- 所有函数都是纯函数：无副作用、无 IO、无全局可变状态，可单独单测。
- 归一化只做「形式统一」，不做「语义推断」。拿不准的一律返回 None 并显式标记，
  绝不做猜测——这个项目里猜错一个数字比找不到更严重。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

__all__ = [
    "UNIT_MULTIPLIER",
    "DISCLOSURE_UNIT_SCALE",
    "ParsedAmount",
    "SIX_TUPLE_FIELDS",
    "normalize_text",
    "parse_amounts",
    "parse_single_amount",
    "find_missing_fields",
    "unit_scale",
]


# --------------------------------------------------------------------------
# 单位换算表
# --------------------------------------------------------------------------

#: 文本中出现的数量单位写法 -> 换算到「元」的倍数。
#: 用精确匹配而非后缀匹配：`万元` 必须以 10^4 处理，若按字符包含判断
#: 会与 `百万元`(10^6) 混淆。
UNIT_MULTIPLIER: dict[str, Decimal] = {
    "元": Decimal(1),
    "千元": Decimal(10) ** 3,
    "万元": Decimal(10) ** 4,
    "百万元": Decimal(10) ** 6,
    "亿元": Decimal(10) ** 8,
    "千": Decimal(10) ** 3,
    "万": Decimal(10) ** 4,
    "亿": Decimal(10) ** 8,
}

#: 报表口径声明中出现的披露单位 -> 换算到「元」的倍数。
#: 用于推导勾稽容差：报表以万元披露时，每个科目有 ±0.5 万元的舍入误差。
DISCLOSURE_UNIT_SCALE: dict[str, Decimal] = {
    "元": Decimal(1),
    "千元": Decimal(10) ** 3,
    "万元": Decimal(10) ** 4,
    "百万元": Decimal(10) ** 6,
    "亿元": Decimal(10) ** 8,
}

#: 单位识别顺序：长的优先，否则 `亿元` 会被 `元` 先匹配掉。
_UNIT_ALTERNATION = "|".join(
    sorted(UNIT_MULTIPLIER, key=len, reverse=True)
)


def unit_scale(disclosure_unit: str | None) -> Decimal:
    """把报表披露单位换算成「元」的倍数。

    Args:
        disclosure_unit: 报表口径声明的单位，如 `"万元"`。`None` 表示未知。

    Returns:
        换算倍数。未知单位返回 `Decimal(1)`（即按「元」处理），
        调用方应同时检查 :func:`is_known_unit` 以便在报告中标注口径假设。
    """
    if not disclosure_unit:
        return Decimal(1)
    return DISCLOSURE_UNIT_SCALE.get(disclosure_unit.strip(), Decimal(1))


def is_known_unit(disclosure_unit: str | None) -> bool:
    """披露单位是否是已知口径。未知时容差推导会偏保守，需要显式告警。"""
    if not disclosure_unit or not disclosure_unit.strip():
        return False
    return disclosure_unit.strip() in DISCLOSURE_UNIT_SCALE


# --------------------------------------------------------------------------
# 文本归一化
# --------------------------------------------------------------------------

#: 匹配时需要抹平的空白字符。财务表格里同一数字常被空格或换行切开。
_WHITESPACE_RE = re.compile(r"\s+")

#: 各种破折号/连字符，统一成 ASCII 减号。PDF 抽取常把减号抽成全角或连接号。
_DASHES = "\u2010\u2011\u2012\u2013\u2014\u2212\uff0d"

#: 各种引号形态，统一成 ASCII 引号。
_QUOTES = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u300c": '"',
    "\u300d": '"',
    "\u300e": '"',
    "\u300f": '"',
}


def normalize_text(text: str) -> str:
    """把文本归一化成「适合做子串匹配」的规范形式。

    处理内容：全角转半角（NFKC）、去除**所有**空白、统一破折号与引号。

    注意：这是唯一允许用于 span 匹配的归一化。
    它只改变形式，绝不删除或改写数字与汉字本身。
    浏览器展示用的原文应始终取未归一化的原始串。
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFKC", text)
    out = _WHITESPACE_RE.sub("", out)
    for dash in _DASHES:
        out = out.replace(dash, "-")
    for src, dst in _QUOTES.items():
        out = out.replace(src, dst)
    return out


# --------------------------------------------------------------------------
# 金额解析
# --------------------------------------------------------------------------

#: 数字：允许千分位（半角或全角逗号），允许小数部分。
_NUMBER_PATTERN = r"[+-]?(?:\d{1,3}(?:[,，]\d{3})+|\d+)(?:\.\d+)?"

#: 会计惯例：括号表示负数，如 `(1,234.56)` 表示 -1234.56。
_PAREN_NEGATIVE_RE = re.compile(rf"[（(](?P<num>{_NUMBER_PATTERN})[)）]")

#: 带可选单位的金额。单位识别交给 :data:`_UNIT_ALTERNATION`，长的优先。
_AMOUNT_RE = re.compile(
    rf"(?P<num>{_NUMBER_PATTERN})\s*(?P<unit>{_UNIT_ALTERNATION})?"
)


@dataclass(frozen=True)
class ParsedAmount:
    """一个从文本中解析出来的金额。

    Attributes:
        raw: 原始匹配文本，保留用于人工核对。
        value: 按文本所示单位计的值。`1.23亿元` -> `1.23`。
        unit: 文本中出现的单位写法，未出现则为 `None`。
        normalized: 统一换算到「元」之后的值。`1.23亿元` -> `123000000`。
        unit_assumed: 文本未写单位、被按「元」处理时为 True。
            该标记会一路带到证据层——六元组里的「单位」字段是否可信依赖它。
    """

    raw: str
    value: Decimal
    unit: str | None
    normalized: Decimal
    unit_assumed: bool

    @property
    def multiplier(self) -> Decimal:
        """本金额使用的换算倍数。"""
        return UNIT_MULTIPLIER.get(self.unit or "", Decimal(1))


def _to_decimal(num_text: str) -> Decimal | None:
    cleaned = num_text.replace(",", "").replace("，", "")
    try:
        return Decimal(cleaned)
    except InvalidOperation:
        return None


def parse_amounts(text: str) -> list[ParsedAmount]:
    """解析文本中出现的所有金额，按出现顺序返回。

    不去重、不排序。调用方需要知道原文写了几个数字，就能拿到几个。
    括号负数会先被摘出来，不会与普通数字重复计入。
    """
    if not text:
        return []

    working = unicodedata.normalize("NFKC", text)
    results: list[ParsedAmount] = []

    # 第一步：摘出括号负数，并把它们从待扫描文本中抹掉，避免重复匹配。
    for match in _PAREN_NEGATIVE_RE.finditer(working):
        value = _to_decimal(match.group("num"))
        if value is None:
            continue
        results.append(
            ParsedAmount(
                raw=match.group(0),
                value=-value,
                unit=None,
                normalized=-value,
                unit_assumed=True,
            )
        )
    scanned = _PAREN_NEGATIVE_RE.sub(" ", working)

    # 第二步：扫描普通金额。
    for match in _AMOUNT_RE.finditer(scanned):
        value = _to_decimal(match.group("num"))
        if value is None:
            continue
        unit = match.group("unit")
        multiplier = UNIT_MULTIPLIER.get(unit or "", Decimal(1))
        results.append(
            ParsedAmount(
                raw=match.group(0).strip(),
                value=value,
                unit=unit,
                normalized=value * multiplier,
                unit_assumed=unit is None,
            )
        )

    return results


def parse_single_amount(text: str) -> ParsedAmount | None:
    """解析文本中唯一的金额。

    若文本中一个数字都没有、或出现多个数字（存在歧义），返回 `None`。

    这个保守策略是刻意的：宁可判定「无法确认」，也不猜哪一个才是目标值。
    财务表格一行常有多列数字，此时正确的做法是走表格定位而不是猜。
    """
    amounts = parse_amounts(text)
    if len(amounts) != 1:
        return None
    return amounts[0]


# --------------------------------------------------------------------------
# 六元组字段完整性
# --------------------------------------------------------------------------

#: 六元组的字段名。第 6 项「来源」是复合字段，展开为页码与原文片段两个属性。
SIX_TUPLE_FIELDS: tuple[str, ...] = (
    "company",   # 公司
    "period",    # 期间
    "metric",    # 指标
    "value",     # 数值
    "unit",      # 单位
    "source",    # 来源：页码 + 原文片段
)


def find_missing_fields(
    *,
    company: object,
    period: object,
    metric: object,
    value: object,
    unit: object,
    source: object,
) -> list[str]:
    """返回六元组中缺失的字段名列表。空列表表示证据完整。

    判定标准：字符串为「空或纯空白」，其他类型为 `None`。

    注意：**数值 0 是合法值**，不视为缺失——财务上零值常见且真实，
    因此这里刻意不使用 `if not raw` 这种会把 0 判掉的写法。
    """
    candidates: dict[str, object] = {
        "company": company,
        "period": period,
        "metric": metric,
        "value": value,
        "unit": unit,
        "source": source,
    }
    missing: list[str] = []
    for name, raw in candidates.items():
        if raw is None:
            missing.append(name)
        elif isinstance(raw, str) and not raw.strip():
            missing.append(name)
    return missing
