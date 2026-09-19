"""证据数据模型。

设计要点：六元组的所有字段都是**可空**的。

这不是偷懒，而是刻意的：如果字段不可空，模型一旦漏抽某个字段就会触发
Pydantic 校验错误，整个抽取动作失败。而我们想要的语义是
「这条证据字段不全 → 该证据作废 → 不足以回答 → 拒答」，
是一等业务结果，不是异常。

所以这里让字段可空，由 :func:`fin_verify.normalize.find_missing_fields`
在业务层做完整性判定，把「缺字段」变成可统计、可报告的拒答原因，
而不是一个需要 try/except 的崩溃点。
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

from pydantic import BaseModel, Field

__all__ = [
    "EvidenceDraft",
    "ExtractionResult",
    "EVIDENCE_JSON_SCHEMA",
    "SIX_TUPLE_JSON_SCHEMA",
]


class EvidenceDraft(BaseModel):
    """模型抽取出的单条证据（尚未校验）。

    与最终的「证据」区别在于：本模型的字段未经 span 校验，
    数值尚未转成 :class:`~decimal.Decimal`，**不能直接进入计算层**。
    """

    company: str | None = Field(default=None, description="公司名称")
    period: str | None = Field(default=None, description="报告期")
    metric: str | None = Field(default=None, description="指标名称")
    value: str | None = Field(
        default=None,
        description="数值。用字符串承载，避免 JSON 浮点损失精度。",
    )
    unit: str | None = Field(default=None, description="单位，如 元 / 万元 / 亿元")
    source_page: int | None = Field(default=None, description="来源页码")
    source_span: str | None = Field(
        default=None,
        description="原文片段。必须是原文中逐字出现的连续文本。",
    )

    def parsed_value(self) -> Decimal | None:
        """把数值字符串转成 Decimal。无法解析时返回 `None`。

        刻意不使用 `float`：`Decimal("1.1")` 是精确的，
        而 `Decimal(1.1)` 会引入二进制浮点误差。
        """
        if self.value is None:
            return None
        cleaned = self.value.replace(",", "").replace("，", "").strip()
        if not cleaned:
            return None
        try:
            return Decimal(cleaned)
        except InvalidOperation:
            return None


class ExtractionResult(BaseModel):
    """一次抽取动作的完整结果。"""

    items: list[EvidenceDraft] = Field(default_factory=list)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


#: 发给模型的 JSON Schema。
#:
#: 手写而非由 Pydantic 自动生成，原因是 OpenAI 严格模式有额外要求：
#: 每个属性都必须列进 `required`，且必须显式 `additionalProperties: false`。
#: 手写可以保证这两点确定成立，避免在运行时才发现 schema 被拒。
#: 返回内容仍由 Pydantic 做二次校验。
SIX_TUPLE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "company",
        "period",
        "metric",
        "value",
        "unit",
        "source_page",
        "source_span",
    ],
    "properties": {
        "company": {"type": ["string", "null"], "description": "公司名称"},
        "period": {"type": ["string", "null"], "description": "报告期"},
        "metric": {"type": ["string", "null"], "description": "指标名称"},
        "value": {
            "type": ["string", "null"],
            "description": "数值，纯数字串，保留原文写法（可用千分位）",
        },
        "unit": {"type": ["string", "null"], "description": "单位"},
        "source_page": {"type": ["integer", "null"], "description": "来源页码"},
        "source_span": {
            "type": ["string", "null"],
            "description": "原文片段，必须是原文中逐字出现的连续文本",
        },
    },
}

EVIDENCE_JSON_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["items"],
    "properties": {
        "items": {
            "type": "array",
            "description": "抽取到的证据列表。原文中没有的科目不要出现在这里。",
            "items": SIX_TUPLE_JSON_SCHEMA,
        }
    },
}
