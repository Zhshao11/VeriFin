"""确定性护栏：在"给出答案"之前必须过的那几道程序级检查。

**为什么需要这一层。**

D4 的评测基线暴露出四道应拒答题被作答了，而它们其实是**四个不同的缺口**：

| 现象 | 缺的护栏 |
|---|---|
| 问「宁德时代 2024 年营业收入」→ 答了茅台的值 | **主体约束** |
| 问「2025 年第一季度营业收入」→ 答了 2024 年度值 | **期间约束** |
| 问「资产总计是多少」（未限口径）→ 默认给了合并值 | **口径歧义**（在 tools 层） |
| 问「现金及现金等价物净增加额」→ 答了「投资活动现金流入小计」 | **问答一致性** |

这四条的共性是：**证据是真的、页码是对的、span 校验也过了**，
所以 span 校验、Decimal、页码溯源**一道都拦不住**。它们只能被
「这个证据是不是**问句问的那件事**」这类检查挡住。

本模块只做**确定性判断**，不调用模型。

取舍原则分两种，不要混为一谈：

- **主体 / 期间约束**：判断不了就**放行**。误拦会把正确回答变成拒答，
  而"问句里没提年份"是很正常的事，不该因此拒答。
- **问答一致性**：判断不了就**拒答**。因为这一条要回答的问题是
  「凭什么说取到的这一行就是问的那一行」—— 答不上来就不该给答案。
  在一个输出财务数字的系统里，**拒答比给错答案便宜**。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "DocConstraints",
    "label_consistency",
    "check_question_constraints",
    "normalize_label_for_match",
    "best_label_match",
]

#: 问句里的年份。要求两侧都不是数字，避免把股票代码（600519）里的
#: 四位片段误认成"2005 年"，也避免把「200596」当成年份。
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

#: 匹配用的归一化：去掉括号内容与所有非文字字符。
#:
#: 「所有者权益（或股东权益）合计」与问句里的「所有者权益合计」必须能对上，
#: 否则一条本来正确的取数会被判成"答非所问"。
_BRACKET_RE = re.compile(r"[（(][^）)]*[）)]")
_NON_WORD_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")


@dataclass(frozen=True)
class DocConstraints:
    """本文档回答问题时必须遵守的约束。

    全部字段都**从文档自身推导**（表头日期、编制单位），不从配置文件硬编码 ——
    否则换一份年报就要改一次代码，而忘了改不会报错，只会静默放宽约束。
    """

    company: str = ""
    """本文档发行人的完整名称（取自「编制单位:」）。"""
    company_aliases: tuple[str, ...] = ()
    """本文档主体的别名（简称等），用于判断问句点的是不是本文档主体。"""
    other_issuers: tuple[str, ...] = ()
    """已知的**其他**发行人简称清单。

    这不是命名实体识别，是一份**显式清单** —— 这么做是有意的：
    确定、可测、可解释。代价是清单外的主体拦不住（残留限制已写入评测文档）。
    """
    report_years: tuple[int, ...] = ()
    """财务报表覆盖的年份。问句点名了这之外的年份，证据在物理上不可能存在。"""

    def owns(self, name: str) -> bool:
        """`name` 是不是本文档主体自己的名字。"""
        if not name:
            return False
        return name in self.company or name in self.company_aliases or any(
            a and a in name for a in self.company_aliases
        )


def check_question_constraints(
    question: str, constraints: DocConstraints
) -> dict[str, str] | None:
    """检查问句的主体与期间约束。通过返回 `None`，违反返回失败说明。

    顺序有意为之：**先查期间，再查主体**。期间是可从表头严格推导的硬事实，
    主体依赖显式清单（覆盖有限）。先报确定性更强的那一条，
    免得出现"两条都违反时报告挑了一条次要原因"。
    """
    q = question or ""

    # --- 期间约束 ---
    years = {int(y) for y in _YEAR_RE.findall(q)}
    if constraints.report_years and years:
        outside = sorted(y for y in years if y not in constraints.report_years)
        if outside:
            covered = "、".join(str(y) for y in constraints.report_years)
            return {
                "error": "OUT_OF_DOCUMENT_PERIOD",
                "detail": (
                    f"问句点名了 {'、'.join(str(y) for y in outside)} 年，"
                    f"而本文档只覆盖 {covered} 年 —— 证据在物理上不存在，拒答"
                ),
            }

    # --- 主体约束 ---
    for name in constraints.other_issuers:
        if name and name in q and not constraints.owns(name):
            return {
                "error": "WRONG_DOCUMENT",
                "detail": (
                    f"问句点名的主体是「{name}」，而本文档是「{constraints.company}」—— "
                    "不属于同一主体，拒答"
                ),
            }

    return None


def normalize_label_for_match(text: str) -> str:
    """把科目名/问句归一化到可比形态：去括号内容、去标点与空白。"""
    if not text:
        return ""
    out = _BRACKET_RE.sub("", text)
    return _NON_WORD_RE.sub("", out)


def best_label_match(question: str, known_labels: Iterable[str]) -> str | None:
    """在问句里找出**被提到的最具体的报表科目**。

    规则：取"归一化后是问句子串"的已知科目名里**最长的那一个**。

    为什么要取最长：「净利润」是「归属于母公司股东的净利润」的子串。
    只判断"包含"，问句问的是归母净利润时，「净利润」也会被判定为"被问到了"，
    于是取错行也放行。取最长就自然消解了这种嵌套。
    """
    qn = normalize_label_for_match(question)
    if not qn:
        return None
    best: str | None = None
    best_len = 0
    for label in known_labels:
        ln = normalize_label_for_match(label)
        if len(ln) < 2 or ln not in qn:
            continue
        if len(ln) > best_len:
            best, best_len = label, len(ln)
    return best


def label_consistency(
    question: str,
    chosen_label: str,
    known_labels: Iterable[str],
) -> tuple[bool, str]:
    """问答一致性：取到的行，是不是**问句问的**那一行。

    Returns:
        `(是否一致, 说明)`。

    这条护栏挡的是最阴的一类错误：数值真实、页码正确、span 校验通过，
    但**答的是另一个科目**。

    实测反例（D4 基线 B-028）：问「现金及现金等价物净增加额」，
    系统返回的六元组「指标」是「投资活动现金流入小计」——
    该科目在原文里确实存在、数值也确实是那一行的，可它就是不是被问的那个。
    现有全部护栏（span 校验 / Decimal / 页码溯源）都拦不住这一类，
    因为它们校验的是"这个数是不是原文里的"，而不是"是不是问句问的"。

    判据分两步，缺一不可：

    1. **点名**：取到的科目必须是问句里出现过的已知科目名。这是拦 B-028 的那一步
       —— 问句点了「现金及现金等价物净增加額」，取到「投资活动现金流入小计」，
       而后者**在问句里根本没出现**，所以直接不成立。
    2. **不被包含**：问句点名了更具体的科目时，取到它的一部分也算答非所问。
       「归属于母公司股东的净利润是多少」里也含有子串「净利润」，
       若第 1 步只做"是不是子串"，取到「净利润」会被放行 ——
       两者数值不同，正是必须拦住的情形。
    """
    if not chosen_label:
        return False, "没有取到任何科目"

    labels = list(known_labels)
    qn = normalize_label_for_match(question)
    chosen_n = normalize_label_for_match(chosen_label)

    named = [
        label
        for label in labels
        if len(normalize_label_for_match(label)) >= 2
        and normalize_label_for_match(label) in qn
    ]

    if not named:
        # 问句里**一个已知科目名都没出现**，却仍然取到了某一行 —— 这正是
        # 「答非所问」最典型的样子（实测 B-028：问「现金及现金等价物净增加额」，
        # 因为该科目不在解析窗口内，取到的是排序最靠前的「投资活动现金流入小计」）。
        #
        # 收紧到"没点名就不给答案"是刻意的取舍：它可能把「用简称提问」这类
        # 本来可答的问题也拦下（残留限制已记入评测文档）。理由是这个项目里
        # **拒答比给错答案便宜** —— 给错一个财务数字，后果不是"体验差"。
        return False, (
            "问句没有点名任何报表科目（可能是简称或不在解析范围内），"
            f"却取到了「{chosen_label}」—— 无法证明取的就是问的那一行，拒答"
        )

    named_n = {normalize_label_for_match(label) for label in named}

    if chosen_n not in named_n:
        # 归一化后仍对不上（例如括号内容被剥掉后长度/用字不同），
        # 说明取到的科目与问句点名的都不同名。
        expected = best_label_match(question, labels) or named[0]
        return False, (
            f"问句问的是「{expected}」，实际取到的是「{chosen_label}」—— "
            "答非所问，拒答（数值真实、页码正确也一律不采纳）"
        )

    # 第 2 步：问句点了更长的科目，取到的是它的真子串，仍是答非所问。
    for label in named:
        ln = normalize_label_for_match(label)
        if ln != chosen_n and chosen_n in ln:
            return False, (
                f"问句点名的科目是「{label}」，实际取到的是它的子科目"
                f"「{chosen_label}」—— 两者数值不同，答非所问，拒答"
            )

    return True, f"取到的「{chosen_label}」在问句点名之列"
