"""LLM 共享纯函数和常量。"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from google.genai import types as gemini_types

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*\})\s*```", re.S)
_PROVIDER_ALIASES = {
    "openai": "openai",
    "openai_api": "openai",
    "openrouter": "openrouter",
    "gemini": "gemini",
    "claude": "claude",
}
_OPENAI_REASONING_PREFIXES = ("gpt-5", "o1", "o3", "o4")
_REQUEST_PARAM_MAPS = {
    "openai_responses": {
        "token_limit": "max_output_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "timeout": "timeout",
    },
    "openai_chat": {
        "token_limit": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "timeout": "timeout",
    },
    "openai_reasoning": {
        "token_limit": "max_completion_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "timeout": "timeout",
        "reasoning_effort": "reasoning_effort",
    },
    "openrouter_chat": {
        "token_limit": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "timeout": "timeout",
    },
    "openrouter_responses": {
        "token_limit": "max_output_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "timeout": "timeout",
    },
    "claude": {
        "token_limit": "max_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
        "timeout": "timeout",
    },
    "gemini": {
        "token_limit": "max_output_tokens",
        "temperature": "temperature",
        "top_p": "top_p",
    },
}


def normalize_provider(mode: str | None, config: dict[str, Any]) -> str:
    """把 mode / provider alias 归一化成统一 provider 名称。"""
    provider = config.get("provider") or mode or "openai"
    return _PROVIDER_ALIASES.get(provider, provider)


def resolve_api_key(provider: str, config: dict[str, Any]) -> str:
    """按 provider 和配置解析 API key。"""
    if config.get("api_key"):
        return config["api_key"]

    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    if provider == "gemini":
        return os.getenv("GEMINI_API_KEY", "")
    if provider == "claude":
        return os.getenv("ANTHROPIC_API_KEY", "")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY", "")
    return ""


def resolve_base_url(provider: str, config: dict[str, Any]) -> str | None:
    """按 provider 和配置解析 base_url。"""
    if config.get("base_url"):
        return config["base_url"]

    if provider in {"openai", "gemini", "claude"}:
        # 官方 API：base_url 留空，交由各 SDK 使用其默认端点。
        # 如需走自定义网关/代理，可在 settings.yaml 的对应 profile 里显式设置 base_url。
        return None
    if provider == "openrouter":
        return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return None


def resolve_model(provider: str, config: dict[str, Any]) -> str:
    """按 provider 和配置解析模型名。"""
    model = config.get("model")
    if model:
        return model

    model_env = config.get("model_env")
    if model_env:
        return os.getenv(model_env, "")

    return ""


def as_jsonable(value: Any) -> Any:
    """把 SDK 对象递归转成内建 Python 结构。"""
    if hasattr(value, "model_dump"):
        return {k: as_jsonable(v) for k, v in value.model_dump(exclude_none=True).items()}
    if isinstance(value, dict):
        return {k: as_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [as_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [as_jsonable(v) for v in value]
    return value


def try_load_json(text: str) -> Any:
    """尽量把字符串解析为 JSON；失败则原样返回。"""
    try:
        return json.loads(text)
    except Exception:
        return text


def get_value(value: Any, key: str, default: Any = None) -> Any:
    """兼容 dict / SDK 对象的统一字段访问。"""
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def normalize_token_usage(
    input_tokens: Any = 0, output_tokens: Any = 0, thinking_tokens: Any = 0,
) -> dict[str, int]:
    """统一 token usage 字段名，缺失时返回 0。

    output_tokens 是总输出（含 thinking），thinking_tokens 是其中的 thinking 子集。
    """
    return {
        "input_tokens": _safe_int(input_tokens),
        "thinking_tokens": _safe_int(thinking_tokens),
        "output_tokens": _safe_int(output_tokens),
    }


def normalize_openai_chat_usage(usage: Any) -> dict[str, int]:
    """归一化 OpenAI Chat Completions / OpenAI-compatible usage。"""
    details = get_value(usage, "completion_tokens_details")
    thinking = _safe_int(get_value(details, "reasoning_tokens")) if details else 0
    return normalize_token_usage(
        get_value(usage, "prompt_tokens"),
        get_value(usage, "completion_tokens"),
        thinking,
    )


def normalize_openai_responses_usage(usage: Any) -> dict[str, int]:
    """归一化 OpenAI Responses usage。"""
    details = get_value(usage, "output_tokens_details")
    thinking = _safe_int(get_value(details, "reasoning_tokens")) if details else 0
    return normalize_token_usage(
        get_value(usage, "input_tokens"),
        get_value(usage, "output_tokens"),
        thinking,
    )


def normalize_gemini_usage(usage: Any) -> dict[str, int]:
    """归一化 Gemini usage，output 包含 thinking tokens。"""
    input_tokens = _safe_int(get_value(usage, "prompt_token_count"))
    thinking_tokens = _safe_int(get_value(usage, "thoughts_token_count"))
    total_raw = get_value(usage, "total_token_count")
    if total_raw is not None:
        output_tokens = max(_safe_int(total_raw) - input_tokens, 0)
    else:
        output_tokens = (
            _safe_int(get_value(usage, "candidates_token_count"))
            + thinking_tokens
        )
    return normalize_token_usage(input_tokens, output_tokens, thinking_tokens)


def normalize_claude_usage(usage: Any) -> dict[str, int]:
    """归一化 Claude usage，cache input token 计入输入。"""
    input_tokens = (
        _safe_int(get_value(usage, "input_tokens"))
        + _safe_int(get_value(usage, "cache_creation_input_tokens"))
        + _safe_int(get_value(usage, "cache_read_input_tokens"))
    )
    return normalize_token_usage(input_tokens, get_value(usage, "output_tokens"))


def parse_json_text(text: str) -> dict[str, Any]:
    """从模型文本中稳健提取 JSON 对象。"""
    text = text.strip()
    if not text:
        raise ValueError("LLM returned empty JSON content")

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    fenced = _JSON_BLOCK_RE.search(text)
    if fenced:
        parsed = json.loads(fenced.group(1))
        if isinstance(parsed, dict):
            return parsed

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(text[start:end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError(f"Failed to parse JSON object from LLM output: {text[:200]}")


def augment_system_for_json(system_text: str | None) -> str:
    """在 JSON 模式下追加一条更强的格式约束。"""
    suffix = "Return only valid JSON. Do not add markdown fences or extra commentary."
    return f"{system_text}\n\n{suffix}" if system_text else suffix


def stringify_reasoning(value: Any) -> str:
    """把 reasoning 内容稳定转成字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(as_jsonable(value), ensure_ascii=False)
    except Exception:
        return str(value)


def apply_param_mapping(
    target: dict[str, Any],
    values: dict[str, Any],
    param_map: dict[str, str],
) -> None:
    """把统一参数名按映射写入请求参数。"""
    for internal_name, external_name in param_map.items():
        value = values.get(internal_name)
        if value is not None:
            target[external_name] = value


def merge_dict(base: dict[str, Any] | None, patch: dict[str, Any] | None) -> dict[str, Any] | None:
    """浅合并两个 dict，后者覆盖前者。"""
    merged = dict(base or {})
    for key, value in (patch or {}).items():
        merged[key] = value
    return merged or None


def extract_text_list(value: Any) -> list[str]:
    """从 SDK 对象 / dict / 字符串列表中稳健提取文本。"""
    items = value or []
    if isinstance(items, (str, bytes)):
        return [items.decode() if isinstance(items, bytes) else items]

    texts: list[str] = []
    for item in items:
        if isinstance(item, str):
            texts.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("text")
            if text:
                texts.append(str(text))
            continue
        text = getattr(item, "text", None)
        if text:
            texts.append(str(text))
    return texts


def normalize_reasoning_details(
    reasoning_details: list[Any] | None,
    *,
    provider: str,
    model: str,
    source: str,
) -> list[dict[str, Any]]:
    """把 OpenRouter reasoning_details 统一映射到项目的 reasoning_trace。"""
    trace: list[dict[str, Any]] = []
    for item in reasoning_details or []:
        item_type = get_value(item, "type")
        item_id = get_value(item, "id")
        item_format = get_value(item, "format")
        item_index = get_value(item, "index")

        summary_text = ""
        raw_reasoning = ""
        encrypted_content = None
        signatures: list[str] = []

        if item_type == "reasoning.summary":
            summary_text = str(get_value(item, "summary", "") or "")
        elif item_type == "reasoning.text":
            raw_reasoning = str(get_value(item, "text", "") or "")
            signature = get_value(item, "signature")
            if signature:
                signatures.append(str(signature))
        elif item_type == "reasoning.encrypted":
            encrypted_content = get_value(item, "data")
        else:
            continue

        trace.append({
            "provider": provider,
            "model": model,
            "summary_text": summary_text,
            "summary_parts": [summary_text] if summary_text else [],
            "raw_reasoning": raw_reasoning or None,
            "encrypted_content": encrypted_content,
            "thought_signatures": signatures,
            "source": source,
            "item_id": item_id,
            "format": item_format,
            "index": item_index,
            "type": item_type,
        })
    return trace


def gemini_schema_type(value: Any) -> Any:
    """Gemini 的 JSON schema 类型值通常使用大写，统一转换。"""
    if not isinstance(value, str):
        return value
    mapping = {
        "object": "OBJECT",
        "string": "STRING",
        "number": "NUMBER",
        "integer": "INTEGER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "null": "NULL",
    }
    return mapping.get(value.lower(), value)


def openai_schema_to_gemini(schema: Any) -> Any:
    """递归把 OpenAI JSON schema 转成 Gemini 可接受的格式。"""
    if isinstance(schema, dict):
        converted = {}
        for key, value in schema.items():
            if key == "type":
                converted[key] = gemini_schema_type(value)
            else:
                converted[key] = openai_schema_to_gemini(value)
        return converted
    if isinstance(schema, list):
        return [openai_schema_to_gemini(item) for item in schema]
    return schema


def openai_tools_to_anthropic(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """把 OpenAI function calling schema 转成 Anthropic tools 格式。"""
    converted = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        converted.append({
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
        })
    return converted


def openai_tools_to_responses(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """把 Chat Completions 风格 function tools 转成 Responses API tools。"""
    converted = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        converted.append({
            "type": "function",
            "name": func.get("name", ""),
            "description": func.get("description", ""),
            "parameters": func.get("parameters", {"type": "object", "properties": {}}),
            "strict": False,
        })
    return converted


def openai_tools_to_gemini(
    tools: list[dict[str, Any]] | None,
) -> list[gemini_types.Tool]:
    """把 OpenAI function calling schema 转成 Gemini tools。"""
    declarations = []
    for tool in tools or []:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        declarations.append(
            gemini_types.FunctionDeclaration(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=openai_schema_to_gemini(
                    func.get("parameters", {"type": "object", "properties": {}}),
                ),
            )
        )

    if not declarations:
        return []
    return [gemini_types.Tool(function_declarations=declarations)]


__all__ = [
    "_OPENAI_REASONING_PREFIXES",
    "_REQUEST_PARAM_MAPS",
    "apply_param_mapping",
    "as_jsonable",
    "augment_system_for_json",
    "extract_text_list",
    "get_value",
    "merge_dict",
    "normalize_claude_usage",
    "normalize_gemini_usage",
    "normalize_openai_chat_usage",
    "normalize_openai_responses_usage",
    "normalize_provider",
    "normalize_reasoning_details",
    "normalize_token_usage",
    "openai_schema_to_gemini",
    "openai_tools_to_anthropic",
    "openai_tools_to_gemini",
    "openai_tools_to_responses",
    "parse_json_text",
    "resolve_api_key",
    "resolve_base_url",
    "resolve_model",
    "stringify_reasoning",
    "try_load_json",
]
