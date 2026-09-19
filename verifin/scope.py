"""报表口径（合并 / 母公司）的原语。

**为什么口径要单独立一个模块。**

中文年报里同一个科目名有**两套数值**：

    资产总计（合并）   = 298,944,579,918.70
    资产总计（母公司） = 180,236,524,477.01

两套都是**真实数字**，都在原文里，span 校验都能过。
所以「取错口径」是本项目最难自动发现的一类错误 ——
它不报错，只是安静地给出另一张表的数字。

口径这件事被四处用到：

| 用到的地方 | 用途 |
|---|---|
| 表格层 `tables.py` | 从表标题（`## 母公司资产负债表`）推导这张表的口径 |
| 检索层 `retrieval.py` | chunk 携带口径，召回时能按口径过滤 |
| 编排层 `agent/` | 从问句识别用户要哪个口径；两套都在而问句没限定 → 拒答 |
| 评测层 `eval/schema.py` | 校验标注口径与问句一致 |

四处若各写一份「母公司」的识别规则，**迟早漂移**，而漂移的后果是静默取错口径。
所以识别规则集中在这里，其余各处一律调用本模块。

`「归属于母公司」的坑`：它是**合并报表的科目名**（归属于母公司股东的净利润），
名字里带「母公司」却与母公司报表毫无关系。用朴素子串判断会把合并题
误判成「口径不一致」——**误报比漏报更消耗信任**，故必须用否定环视排除。
"""

from __future__ import annotations

import re

__all__ = [
    "SCOPE_CONSOLIDATED",
    "SCOPE_PARENT",
    "SCOPES",
    "PARENT_SCOPE_RE",
    "both_scopes_named",
    "detect_scope_in_question",
    "derive_scope_from_caption",
    "question_names_scope",
]

#: 合并报表口径。
SCOPE_CONSOLIDATED = "合并"
#: 母公司（本部）报表口径。
SCOPE_PARENT = "母公司"

SCOPES: tuple[str, ...] = (SCOPE_CONSOLIDATED, SCOPE_PARENT)

#: 问句里「母公司口径」的提法，但排除「归属于母公司」。
#:
#: 「归属于母公司股东的净利润」是**合并报表**的一行，带「母公司」三个字纯属巧合。
#: 早先用朴素子串判断，结果把合并口径的不等式题误判成「口径不一致」。
PARENT_SCOPE_RE = re.compile(r"(?<!归属于)母公司")

#: 母公司口径的其他常见叫法（中文年报里「本部」指母公司报表）。
_PARENT_ALIASES: tuple[str, ...] = ("本部", "母公司报表", "母公司口径")

#: 表标题里的口径词。标题形如 `## 合并资产负债表` / `## 母公司利润表`。
_CAPTION_PARENT_RE = re.compile(r"母公司|本部")
_CAPTION_CONSOLIDATED_RE = re.compile(r"合并")


def derive_scope_from_caption(caption: str | None) -> str | None:
    """从表标题推导这张表的口径。推不出来就返回 `None`（**不猜**）。

    `None` 的含义是「这张表的口径未标注」，与「合并」「母公司」并列为一个合法取值 ——
    有些附注表本来就不分口径。调用方必须显式处理 `None`，
    不能把它当成「合并」处理，否则等于把不确定悄悄变成确定。
    """
    if not caption:
        return None
    if _CAPTION_PARENT_RE.search(caption):
        return SCOPE_PARENT
    if _CAPTION_CONSOLIDATED_RE.search(caption):
        return SCOPE_CONSOLIDATED
    return None


def question_names_scope(question: str) -> bool:
    """问句有没有**点名**报表口径。"""
    return detect_scope_in_question(question) is not None


def detect_scope_in_question(question: str) -> str | None:
    """从问句里识别用户想要的口径。

    Returns:
        `"合并"` / `"母公司"`；**问句没有点名口径时返回 `None`**。

    Note:
        问句**同时**点名两个口径（如「合并与母公司口径的资产总计相差多少」）时
        也返回 `None` —— 这不是「没说」，而是「都要」。
        需要区分这两种情况的调用方用
        :func:`question_names_scope` 与 :func:`both_scopes_named`。
    """
    q = question or ""
    names_parent = bool(PARENT_SCOPE_RE.search(q)) or any(
        a in q for a in _PARENT_ALIASES
    )
    names_consolidated = SCOPE_CONSOLIDATED in q
    if names_parent and not names_consolidated:
        return SCOPE_PARENT
    if names_consolidated and not names_parent:
        return SCOPE_CONSOLIDATED
    return None


def both_scopes_named(question: str) -> bool:
    """问句是否**同时**点名了合并与母公司两套口径（即「跨口径」）。"""
    q = question or ""
    names_parent = bool(PARENT_SCOPE_RE.search(q)) or any(
        a in q for a in _PARENT_ALIASES
    )
    return names_parent and SCOPE_CONSOLIDATED in q
