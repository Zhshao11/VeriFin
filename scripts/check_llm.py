"""探测 LLM 通道的三项能力：基础对话 / 工具调用 / 结构化输出。

只读探测，不写任何数据。凭证从 .env 读取，不打印。
用法：python scripts/check_llm.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(f"未找到 {path}")
    env: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


def post(base_url: str, api_key: str, payload: dict) -> dict:
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    env = load_env(ENV_PATH)
    base_url = env["LLM_BASE_URL"]
    api_key = env["LLM_API_KEY"]
    model = env["LLM_MODEL"]
    print(f"通道: {base_url}  模型: {model}\n")

    # --- 测试 1：工具调用（function calling） ---
    tools = [
        {
            "type": "function",
            "function": {
                "name": "search_annual_report",
                "description": "在年报中检索指定指标与期间的证据",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "description": "财务指标名，如 营业收入"},
                        "period": {"type": "string", "description": "期间，如 2023-12-31"},
                    },
                    "required": ["metric", "period"],
                },
            },
        }
    ]
    print("=== 测试 1：工具调用 ===")
    try:
        r = post(
            base_url,
            api_key,
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": "帮我查一下贵州茅台 2023 年的营业收入"}
                ],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 512,
            },
        )
        msg = r["choices"][0]["message"]
        calls = msg.get("tool_calls")
        if calls:
            fn = calls[0]["function"]
            print(f"  结果: 支持。模型请求调用 {fn['name']}")
            print(f"  参数: {fn['arguments']}")
        else:
            print(f"  结果: 未触发工具调用。content={msg.get('content')!r}")
    except urllib.error.HTTPError as e:
        print(f"  结果: 失败 HTTP {e.code} -> {e.read().decode('utf-8', 'replace')[:300]}")
    except Exception as e:  # noqa: BLE001
        print(f"  结果: 失败 {type(e).__name__}: {e}")

    # --- 测试 2：结构化输出（json_schema 严格模式） ---
    schema = {
        "type": "object",
        "properties": {
            "company": {"type": "string"},
            "period": {"type": "string"},
            "metric": {"type": "string"},
            "value": {"type": "string"},
            "unit": {"type": "string"},
            "source_page": {"type": ["integer", "null"]},
            "source_span": {"type": ["string", "null"]},
        },
        "required": [
            "company",
            "period",
            "metric",
            "value",
            "unit",
            "source_page",
            "source_span",
        ],
        "additionalProperties": False,
    }
    print("\n=== 测试 2：结构化输出（json_schema） ===")
    prompt = (
        "从下面这段年报原文中抽取证据，输出 JSON。"
        "注意：source_span 必须是原文里逐字出现的片段，不得改写。\n\n"
        "原文（第 78 页）：合并利润表显示，公司 2023 年度实现营业收入 1,234,567,890.12 元，"
        "较上年同期增长 12.3%。\n\n"
        "公司=示例公司，期间=2023年度，指标=营业收入。"
    )
    try:
        r = post(
            base_url,
            api_key,
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "evidence", "strict": True, "schema": schema},
                },
                "max_tokens": 1024,
            },
        )
        msg = r["choices"][0]["message"]
        content = msg.get("content")
        print(f"  结果: 返回成功")
        try:
            parsed = json.loads(content)
            print(f"  解析: JSON 合法，{len(parsed)} 个字段")
            for k, v in parsed.items():
                print(f"    {k}: {v!r}")
            span = parsed.get("source_span")
            if span:
                hit = span in prompt
                print(f"  span 是否命中原文: {hit}")
        except json.JSONDecodeError:
            print(f"  ⚠ 内容不是合法 JSON: {content[:300]!r}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")[:400]
        print(f"  结果: json_schema 失败 HTTP {e.code}")
        print(f"  降级探测 json_object ...")
        try:
            r = post(
                base_url,
                api_key,
                {
                    "model": model,
                    "messages": [{"role": "user", "content": prompt + "\n\n只输出 JSON。"}],
                    "response_format": {"type": "json_object"},
                    "max_tokens": 1024,
                },
            )
            content = r["choices"][0]["message"].get("content")
            json.loads(content)
            print(f"  降级结果: json_object 可用，JSON 合法")
            print(f"  内容: {content[:300]}")
        except Exception as e2:  # noqa: BLE001
            print(f"  降级也失败: {type(e2).__name__} {str(e2)[:200]}")
    except Exception as e:  # noqa: BLE001
        print(f"  结果: 失败 {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
