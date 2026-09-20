"""LLM 客户端。

五条设计约定：

1. **凭证只从环境读取**，不从代码或参数传入。`.env` 已被 `.gitignore` 排除，
   日志里也不打印密钥。
2. **模型无关**。走 OpenAI 兼容协议，换 `base_url` + `model` 即可切换厂商，
   便于做「主模型 + 廉价档」的成本优化与消融实验。
3. **结构化输出带降级**。优先用 `json_schema` 严格模式；
   若该端点不支持，自动降级到 `json_object`，并如实记录降级事实——
   评测报告里要能看出哪次调用是降级跑的。
4. **网络层有界重试**。超时、连接错误、429、5xx 会退避重试；
   400/401/403/404 是**不重试**的（重试解决不了鉴权与格式问题，只会浪费预算）。
   `Retry-After` 头优先于自算退避。
5. **限流在客户端这一层**。同一进程内的所有请求共用一个令牌桶，
   避免并发打爆端点（本端点在实测中确实会返回 429，不是理论风险）。
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from verifin.models import EVIDENCE_JSON_SCHEMA, ExtractionResult

__all__ = ["LLMError", "CallRecord", "LLMClient", "RetryPolicy", "RateLimiter"]


class LLMError(RuntimeError):
    """LLM 调用相关的错误。"""


#: 值得重试的 HTTP 状态码。4xx 里只有 408/409/425/429 属于「再试可能就好了」。
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: 没有 status_code 时按异常类名判断（httpx 的连接层异常走这里）。
_RETRYABLE_NAMES = frozenset(
    {
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
        "RateLimitError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "RemoteProtocolError",
    }
)


def is_retryable(exc: BaseException) -> bool:
    """这个异常值得重试吗？

    判断顺序：先看 `status_code`（OpenAI SDK 的 APIStatusError 都带），
    再看类名（httpx 传输层异常没有 status_code）。
    显式的鉴权/权限/未找到类异常一律不重试 —— 重试它们只是把错误拖长。
    """
    name = type(exc).__name__
    if name in {
        "AuthenticationError",
        "PermissionDeniedError",
        "NotFoundError",
        "BadRequestError",
        "UnprocessableEntityError",
    }:
        return False
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS
    return name in _RETRYABLE_NAMES


def retry_after_seconds(exc: BaseException) -> float | None:
    """读取服务端 `Retry-After` 头（秒）。拿不到返回 None。

    服务端明确告诉你多久后再来，就不该用自算的退避覆盖它。
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    try:
        raw = headers.get("retry-after")
    except Exception:  # noqa: BLE001 —— headers 可能不是 Mapping，不容它把流程搞崩
        return None
    if not raw:
        return None
    try:
        return max(0.0, float(str(raw).strip()))
    except ValueError:
        return None


@dataclass(frozen=True)
class RetryPolicy:
    """有界重试策略：次数 + 指数退避 + 抖动。

    `max_attempts` 含首次尝试。抖动存在的意义：多个并发请求同时失败后
    如果按同一个节奏重试，会形成新的同步尖峰，把刚恢复的端点再打一遍。
    """

    max_attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    jitter_ratio: float = 0.3

    def delay_for(self, failed_attempt: int) -> float:
        """第 `failed_attempt` 次失败之后该等多久（failed_attempt 从 1 计）。"""
        raw = min(self.base_delay * (2 ** (failed_attempt - 1)), self.max_delay)
        return raw + raw * self.jitter_ratio * random.random()


class RateLimiter:
    """线程安全的令牌桶。

    单进程 CLI 用不到并发，但 web 端是多线程的（`web/server.py`），
    所以桶必须自己带锁 —— `check_same_thread=False` 那类"允许跨线程"
    不等于"跨线程安全"，这是本项目已经踩过一次的坑。
    """

    def __init__(self, *, rate_per_sec: float = 5.0, burst: int = 5) -> None:
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec 必须为正数")
        self.rate = float(rate_per_sec)
        self.burst = max(1, int(burst))
        self._tokens = float(self.burst)
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, timeout: float = 30.0) -> bool:
        """取一个令牌，必要时等待。超时返回 False（调用方决定怎么办）。"""
        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(
                    float(self.burst), self._tokens + (now - self._last) * self.rate
                )
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return True
                deficit = 1.0 - self._tokens
                wait = deficit / self.rate
            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 0.05) if wait > 0.05 else wait)

    @property
    def available_tokens(self) -> float:
        with self._lock:
            now = time.monotonic()
            self._tokens = min(
                float(self.burst), self._tokens + (now - self._last) * self.rate
            )
            self._last = now
            return self._tokens


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
    #: 有界重试：默认 3 次尝试（含首次）、0.5s 起指数退避、上限 8s、±30% 抖动。
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    #: 进程内共享的令牌桶。默认 5 req/s、突发 5；本机单跑时基本不触发。
    rate_limiter: RateLimiter | None = field(default=None)
    #: 记录每一次调用，供成本统计与执行轨迹导出。
    calls: list[CallRecord] = field(default_factory=list)
    #: 重试与限流的实际发生次数。评测报告要能看见"有没有被限过流"。
    retry_count: int = 0
    throttled_count: int = 0
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
        if self.rate_limiter is None:
            # 默认开启限流：端点在实测中确实会 429，并发时不加限流就是自找。
            self.rate_limiter = RateLimiter(rate_per_sec=5.0, burst=5)

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

        两层容错，**顺序不能反**：

        1. 内层：同一 `response_format` 下的**网络层有界重试**
           （超时 / 连接错误 / 429 / 5xx → 退避 + 抖动；`Retry-After` 优先）。
        2. 外层：`json_schema` 严格模式 → `json_object` **能力降级**。

        为什么不能反：如果外层先降级、内层才重试，一次网络抖动就会把
        `json_schema` 降成 `json_object`，之后即使端点明明支持严格模式，
        也永远用不上 —— 降级是能力问题，重试是稳定性问题，两回事。
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
            completion = self._call_with_retry(
                model=self.model,
                messages=messages,
                response_format=response_format,
                temperature=self.temperature,
                max_tokens=max_tokens,
                openai=openai,
            )
            if completion is None:
                # 该端点不认识这种 response_format，试下一档。
                last_error = LLMError(f"端点拒绝了 response_format={mode}")
                continue

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

    def _call_with_retry(
        self, *, openai: Any, **kwargs: Any
    ) -> Any | None:
        """带限流与有界重试地发一次请求。

        返回 `None` 表示**端点明确拒绝了这次 response_format**（BadRequest），
        应由调用方降级到下一档 —— 这不是失败，是能力探测的结果。
        其余情况要么返回 completion，要么抛 :class:`LLMError`。
        """
        assert self.rate_limiter is not None  # __post_init__ 已保证
        policy = self.retry_policy
        last_exc: BaseException | None = None

        for attempt in range(1, policy.max_attempts + 1):
            if not self.rate_limiter.acquire(timeout=max(1.0, self.timeout)):
                self.throttled_count += 1
                raise LLMError(
                    "本地限流等待超时（令牌桶），本次调用放弃 —— "
                    "端点压力过大，宁可不发也不要堆积请求"
                )
            try:
                return self._client.chat.completions.create(**kwargs)
            except openai.BadRequestError:
                # 端点不认这个 response_format，交给外层降级。
                return None
            except Exception as exc:  # noqa: BLE001
                if not is_retryable(exc) or attempt >= policy.max_attempts:
                    raise LLMError(
                        f"调用失败（{type(exc).__name__}，已尝试 {attempt} 次）：{exc}"
                    ) from exc
                last_exc = exc
                self.retry_count += 1
                wait = retry_after_seconds(exc)
                if wait is None:
                    wait = policy.delay_for(attempt)
                time.sleep(min(wait, policy.max_delay))

        raise LLMError(f"重试耗尽：{last_exc}")

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
        """返回本轮用量汇总，供评测报告写入「单次查询成本」。

        token 数是端点实测回报的（`usage` 字段），不是估算。
        价格的换算不在这里做 —— 价格随厂商与时间变动，
        凭记忆写价格等于编数字（本项目红线），由报告层带查询日期单独记录。
        """
        modes = [c.structured_mode for c in self.calls]
        degraded = [m for m in modes if m != "json_schema"]
        return {
            "model": self.model,
            "calls": len(self.calls),
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_prompt_tokens + self.total_completion_tokens,
            "retried_calls": sum(1 for c in self.calls if c.retried),
            "network_retries": self.retry_count,
            "throttled": self.throttled_count,
            "structured_degraded_calls": len(degraded),
        }
