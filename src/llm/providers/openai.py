"""OpenAI provider 实现。"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.llm.providers.base import BaseProviderClient
from src.llm.utils import (
    _OPENAI_REASONING_PREFIXES,
    as_jsonable,
    augment_system_for_json,
    extract_text_list,
    get_value,
    merge_dict,
    normalize_openai_chat_usage,
    normalize_openai_responses_usage,
    normalize_reasoning_details,
    openai_tools_to_responses,
    stringify_reasoning,
)


def _normalize_openai_tool_call(tool_call: Any, idx: int) -> dict[str, Any]:
    func = tool_call.function
    call_id = getattr(tool_call, "call_id", None) or tool_call.id or f"tool_call_{idx}"
    return {
        "id": call_id,
        "call_id": call_id,
        "type": "function",
        "function": {
            "name": func.name,
            "arguments": func.arguments,
        },
    }


def to_openai_responses_input(
    messages: list[dict[str, Any]],
    json_mode: bool = False,
) -> tuple[str | None, list[dict[str, Any]]]:
    """把 OpenAI 风格 messages 转成 Responses API input items。"""
    system_parts: list[str] = []
    input_items: list[dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role")
        if role == "system":
            if msg.get("content"):
                system_parts.append(str(msg["content"]))
            continue

        if role == "user":
            input_items.append({
                "type": "message",
                "role": "user",
                "content": str(msg.get("content", "")),
            })
            continue

        if role == "assistant":
            content = str(msg.get("content", ""))
            if content:
                input_items.append({
                    "type": "message",
                    "role": "assistant",
                    "content": content,
                })
            for reasoning_item in msg.get("_openrouter_responses_reasoning_items", []) or []:
                input_items.append(as_jsonable(reasoning_item))
            for tool_call in msg.get("tool_calls", []):
                call_id = tool_call.get("call_id") or tool_call.get("id") or "tool_call"
                item = {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": tool_call.get("function", {}).get("name", ""),
                    "arguments": tool_call.get("function", {}).get("arguments", "{}"),
                }
                item_id = tool_call.get("response_item_id") or tool_call.get("id")
                if isinstance(item_id, str) and item_id.startswith("fc"):
                    item["id"] = item_id
                input_items.append(item)
            continue

        if role == "tool":
            tool_call_id = msg.get("tool_call_id", "")
            input_items.append({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": str(msg.get("content", "")),
            })

    instructions = "\n\n".join(p for p in system_parts if p).strip() or None
    if json_mode:
        instructions = augment_system_for_json(instructions)
    return instructions, input_items


def normalize_openai_chat_completion_output(facade: Any, msg: Any, usage: Any = None) -> dict[str, Any]:
    """统一归一化 Chat Completions assistant message。"""
    result = {
        "role": "assistant",
        "content": msg.content or "",
    }
    result.update(normalize_openai_chat_usage(usage))
    if msg.tool_calls:
        result["tool_calls"] = [
            _normalize_openai_tool_call(tc, idx)
            for idx, tc in enumerate(msg.tool_calls)
        ]
    reasoning_details = getattr(msg, "reasoning_details", None)
    if reasoning_details:
        result["_openrouter_reasoning_details"] = as_jsonable(reasoning_details)
        reasoning_trace = normalize_reasoning_details(
            reasoning_details,
            provider=facade._trace_provider_name(),
            model=facade.model,
            source="chat.reasoning_details",
        )
        if reasoning_trace:
            result["reasoning_trace"] = reasoning_trace
    elif getattr(msg, "reasoning", None):
        reasoning_text = stringify_reasoning(getattr(msg, "reasoning", None)).strip()
        if reasoning_text:
            result["_openrouter_reasoning"] = reasoning_text
            result["reasoning"] = reasoning_text
            result["reasoning_trace"] = [{
                "provider": facade._trace_provider_name(),
                "model": facade.model,
                "summary_text": reasoning_text,
                "summary_parts": [reasoning_text],
                "raw_reasoning": reasoning_text,
                "encrypted_content": None,
                "thought_signatures": [],
                "source": "chat.reasoning",
            }]
    return result


def normalize_openai_responses_output(facade: Any, resp: Any) -> dict[str, Any]:
    """把 Responses API 返回归一化成项目统一消息结构。"""
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_trace: list[dict[str, Any]] = []
    raw_reasoning_items: list[dict[str, Any]] = []

    for idx, item in enumerate(getattr(resp, "output", []) or []):
        item_type = get_value(item, "type")
        if item_type == "message":
            for part in get_value(item, "content", []) or []:
                if get_value(part, "type") == "output_text" and get_value(part, "text"):
                    content_parts.append(str(get_value(part, "text")))
            continue

        if item_type == "function_call":
            response_item_id = get_value(item, "id") or f"fc_{idx}"
            call_id = get_value(item, "call_id") or f"call_{idx}"
            tool_calls.append({
                "id": call_id,
                "call_id": call_id,
                "response_item_id": response_item_id,
                "type": "function",
                "function": {
                    "name": get_value(item, "name", ""),
                    "arguments": get_value(item, "arguments", "{}"),
                },
            })
            continue

        if item_type == "reasoning":
            raw_reasoning_items.append(as_jsonable(item))
            summary_parts = extract_text_list(get_value(item, "summary"))
            raw_parts = extract_text_list(get_value(item, "content"))
            reasoning_trace.append({
                "provider": facade._trace_provider_name(),
                "model": facade.model,
                "summary_text": "\n".join(summary_parts).strip(),
                "summary_parts": summary_parts,
                "raw_reasoning": "\n".join(raw_parts).strip(),
                "encrypted_content": get_value(item, "encrypted_content"),
                "thought_signatures": [],
                "source": "responses.reasoning",
                "response_id": getattr(resp, "id", None),
                "item_id": get_value(item, "id"),
            })

    result = {
        "role": "assistant",
        "content": "".join(content_parts).strip() or getattr(resp, "output_text", "") or "",
    }
    result.update(normalize_openai_responses_usage(getattr(resp, "usage", None)))
    if tool_calls:
        result["tool_calls"] = tool_calls
    if reasoning_trace:
        result["reasoning_trace"] = reasoning_trace
    if raw_reasoning_items:
        result["_openrouter_responses_reasoning_items"] = raw_reasoning_items
    return result


class OpenAIProviderClient(BaseProviderClient):
    """OpenAI provider adapter。"""

    def __init__(self, config: dict[str, Any], facade: Any):
        super().__init__(config, facade)
        self.client = OpenAI(api_key=facade.api_key, base_url=facade.base_url)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if self.facade.use_responses_api:
            return self._chat_openai_responses(messages, tools=tools, json_mode=json_mode, max_tokens=max_tokens)
        return self._chat_openai_chat_completions(messages, tools=tools, json_mode=json_mode, max_tokens=max_tokens)

    def _chat_openai_responses(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        instructions, input_items = to_openai_responses_input(messages, json_mode=json_mode)
        kwargs: dict[str, Any] = {
            "model": self.facade.model,
            "input": input_items,
        }
        kwargs.update(self.facade._mapped_request_values_for_profile("openai_responses", max_tokens))
        if instructions:
            kwargs["instructions"] = instructions

        reasoning_kwargs: dict[str, Any] = {}
        if self.facade.reasoning_effort is not None:
            reasoning_kwargs["effort"] = self.facade.reasoning_effort
        if self.facade.capture_reasoning and self.facade.reasoning_summary:
            reasoning_kwargs["summary"] = self.facade.reasoning_summary
        if reasoning_kwargs:
            kwargs["reasoning"] = reasoning_kwargs

        responses_tools = openai_tools_to_responses(tools)
        if responses_tools:
            kwargs["tools"] = responses_tools
            kwargs["tool_choice"] = "auto"
        extra_body = merge_dict(self.facade.extra_body, None)
        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = self.client.responses.create(**kwargs)
        return normalize_openai_responses_output(self.facade, resp)

    def _chat_openai_chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.facade.model,
            "messages": messages,
        }
        model_name = (self.facade.model or "").lower()
        profile_name = "openai_reasoning" if model_name.startswith(_OPENAI_REASONING_PREFIXES) else "openai_chat"
        kwargs.update(
            self.facade._mapped_request_values_for_profile(
                profile_name,
                max_tokens,
                reasoning_effort=self.facade.reasoning_effort,
            )
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        extra_body = merge_dict(self.facade.extra_body, None)
        if extra_body:
            kwargs["extra_body"] = extra_body

        resp = self.client.chat.completions.create(**kwargs)
        return normalize_openai_chat_completion_output(
            self.facade,
            resp.choices[0].message,
            getattr(resp, "usage", None),
        )


__all__ = [
    "OpenAIProviderClient",
    "normalize_openai_chat_completion_output",
    "normalize_openai_responses_output",
    "to_openai_responses_input",
]
