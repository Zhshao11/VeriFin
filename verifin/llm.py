"""LLM 客户端。

三条设计约定：

1. **凭证只从环境读取**，不从代码或参数传入。`.env` 已被 `.gitignore` 排除，
   日志里也不打印密钥。
2. **模型无关**。走 OpenAI 兼容协议，换 `base_url` + `model` 即可切换厂商，
   便于做「主模型 + 廉价档」的成本优化与消融实验。
3. **结构化输出带降级**。优先用 `json_schema` 严格模式；
   若该端点不支持，自动降级到 `json_object`，并如实记录降级事实——
   评测报告里要能看出哪次调用是降级跑的。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from verifin.models import EVIDENCE_JSON_SCHEMA, ExtractionResult

__all__ = ["LLMError", "CallRecord", "LLMClient"]


class LLMError(RuntimeError):
    """LLM 调用相关的错误。"""


@dataclass
class CallRecord:
    """单次调用的用量记录，用于「单次查询成本」指标。"""

    purpose: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    structured_mode: str = "json_schema"
    retried: bool = False

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMClient:
    """OpenAI 兼容协议的对话客户端。"""

    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout: float = 120.0
    #: 记录每一次调用，供成本统计与执行轨迹导出。
    calls: list[CallRecord] = field(default_factory=list)
    _client: Any = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError(
                "未安装 openai 包。请执行：pip install -e '.[dev]'"
            ) from exc
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
        )

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_env(cls, env_path: Path | None = None, **overrides: Any) -> LLMClient:
        """从 `.env` 构造客户端。

        Args:
            env_path: `.env` 路径，默认取项目根目录下的 `.env`。
            **overrides: 覆盖项，如 `model="qwen3.7-plus"`（用于跑廉价档消融）。

        Raises:
            LLMError: 缺少必需配置项。
        """
        path = env_path or Path(__file__).resolve().parent.parent / ".env"
        values: dict[str, str] = {}
        if path.exists():
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                values[key.strip()] = value.strip()

        # 环境变量优先于 .env 文件，便于在 CI 里注入。
        def pick(env_name: str, file_key: str) -> str | None:
            return os.environ.get(env_name) or values.get(file_key)

        base_url = pick("LLM_BASE_URL", "LLM_BASE_URL")
        api_key = pick("LLM_API_KEY", "LLM_API_KEY")
        model = pick("LLM_MODEL", "LLM_MODEL")

        missing = [
            name
            for name, value in (
                ("LLM_BASE_URL", base_url),
                ("LLM_API_KEY", api_key),
                ("LLM_MODEL", model),
            )
            if not value
        ]
        if missing:
            raise LLMError(
                f"缺少配置项：{'、'.join(missing)}。"
                f"请复制 .env.example 为 .env 并填写（查找路径：{path}）。"
            )

        return cls(
            base_url=base_url,  # type: ignore[arg-type]
            api_key=api_key,  # type: ignore[arg-type]
            model=model,  # type: ignore[arg-type]
            **overrides,
        )

    # ------------------------------------------------------------------ 调用
    def chat_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, object],
        schema_name: str,
        purpose: str,
        max_tokens: int = 4096,
    ) -> dict[str, Any]:
        """发起一次要求结构化输出的对话，返回解析后的字典。

        先试 `json_schema` 严格模式；端点不支持时降级到 `json_object`。
        降级事实记录在 :class:`CallRecord` 中，不会静默掩盖。
        """
        import openai

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        attempts: list[tuple[str, Any]] = [
            (
                "json_schema",
                {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "strict": True,
                        "schema": schema,
                    },
                },
            ),
            ("json_object", {"type": "json_object"}),
        ]

        last_error: Exception | None = None
        for mode, response_format in attempts:
            try:
                completion = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
            except openai.BadRequestError as exc:
                # 该端点不认识这种 response_format，试下一档。
                last_error = exc
                continue
            except Exception as exc:  # noqa: BLE001
                raise LLMError(f"调用失败：{type(exc).__name__}: {exc}") from exc

            usage = getattr(completion, "usage", None)
            self.calls.append(
                CallRecord(
                    purpose=purpose,
                    model=self.model,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    structured_mode=mode,
                )
            )

            content = completion.choices[0].message.content or ""
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                raise LLMError(
                    f"模型返回的不是合法 JSON（模式 {mode}）：{content[:200]!r}"
                ) from exc

            if not isinstance(parsed, dict):
                raise LLMError(f"期望 JSON 对象，实际是 {type(parsed).__name__}")
            return parsed

        raise LLMError(f"所有结构化输出模式均失败：{last_error}")

    # ------------------------------------------------------------- 业务封装
    def extract_evidence(self, *, system: str, user: str) -> ExtractionResult:
        """抽取六元组证据。

        校验失败时追加错误说明重试一次。重试仍失败则抛出——
        宁可让这一条任务失败并如实记录，也不返回半成品证据。
        """
        payload = self.chat_json(
            system=system,
            user=user,
            schema=EVIDENCE_JSON_SCHEMA,
            schema_name="evidence_extraction",
            purpose="extract_evidence",
        )
        try:
            return ExtractionResult.model_validate(payload)
        except ValidationError as exc:
            payload2 = self.chat_json(
                system=system,
                user=(
                    f"{user}\n\n上一次的输出不符合结构要求，请修正后重新输出。\n"
                    f"校验错误：{exc.errors()[:3]}"
                ),
                schema=EVIDENCE_JSON_SCHEMA,
                schema_name="evidence_extraction",
                purpose="extract_evidence_retry",
            )
            if self.calls:
                self.calls[-1].retried = True
            try:
                return ExtractionResult.model_validate(payload2)
            except ValidationError as exc2:
                raise LLMError(f"抽取结果两次校验均失败：{exc2.errors()[:3]}") from exc2

    # ------------------------------------------------------------------ 统计
    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self.calls)

    def usage_summary(self) -> dict[str, int | str]:
        """返回本轮用量汇总，供评测报告写入「单次查询成本」。"""
        return {
            "model": self.model,
            "calls": len(self.calls),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
        }
