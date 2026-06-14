"""Gemini provider 实现。"""

from __future__ import annotations

import base64
import json
from typing import Any

from google import genai
from google.genai import types as gemini_types

from src.llm.providers.base import BaseProviderClient
from src.llm.utils import (
    as_jsonable,
    augment_system_for_json,
    normalize_gemini_usage,
    openai_tools_to_gemini,
    try_load_json,
)


class GeminiProviderClient(BaseProviderClient):
    """Gemini provider adapter。"""

    def __init__(self, config: dict[str, Any], facade: Any):
        super().__init__(config, facade)
        http_kwargs: dict[str, Any] = {}
        if facade.api_version:
            http_kwargs["api_version"] = facade.api_version
        if facade.base_url:
            http_kwargs["base_url"] = facade.base_url
        if facade.timeout:
            http_kwargs["timeout"] = facade.timeout * 1000
        http_options = gemini_types.HttpOptions(**http_kwargs) if http_kwargs else None
        self.client = genai.Client(api_key=facade.api_key, http_options=http_options)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        system_text, contents = self._to_gemini_contents(messages)
        if json_mode:
            system_text = augment_system_for_json(system_text)

        config_kwargs: dict[str, Any] = self.facade._mapped_request_values(max_tokens)
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        gemini_tools = openai_tools_to_gemini(tools)
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools
            config_kwargs["tool_config"] = gemini_types.ToolConfig(
                function_calling_config=gemini_types.FunctionCallingConfig(
                    mode=gemini_types.FunctionCallingConfigMode.AUTO,
                )
            )

        thinking_kwargs = {}
        if self.facade.thinking_budget is not None:
            thinking_kwargs["thinking_budget"] = self.facade.thinking_budget
        if self.facade.include_thoughts:
            thinking_kwargs["include_thoughts"] = True
        if thinking_kwargs:
            config_kwargs["thinking_config"] = gemini_types.ThinkingConfig(**thinking_kwargs)

        resp = self.client.models.generate_content(
            model=self.facade.model,
            contents=contents,
            config=gemini_types.GenerateContentConfig(**config_kwargs),
        )

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        thought_parts: list[str] = []
        thought_signatures: list[str] = []
        candidates = getattr(resp, "candidates", []) or []
        if candidates:
            parts = getattr(candidates[0].content, "parts", []) or []
            for idx, part in enumerate(parts):
                thought_sig = getattr(part, "thought_signature", None) or getattr(part, "thoughtSignature", None)
                if thought_sig:
                    encoded_sig = base64.b64encode(thought_sig).decode() if isinstance(thought_sig, bytes) else thought_sig
                    thought_signatures.append(encoded_sig)
                if getattr(part, "text", None):
                    if getattr(part, "thought", False):
                        thought_parts.append(part.text)
                    else:
                        content_parts.append(part.text)
                function_call = getattr(part, "function_call", None)
                if function_call:
                    tc: dict[str, Any] = {
                        "id": getattr(function_call, "id", None) or f"tool_call_{idx}",
                        "call_id": getattr(function_call, "id", None) or f"tool_call_{idx}",
                        "type": "function",
                        "function": {
                            "name": getattr(function_call, "name", ""),
                            "arguments": json.dumps(
                                as_jsonable(getattr(function_call, "args", {})),
                                ensure_ascii=False,
                            ),
                        },
                    }
                    if thought_sig:
                        tc["thought_signature"] = encoded_sig
                    tool_calls.append(tc)

        content = "".join(content_parts).strip()
        if not content and json_mode and getattr(resp, "parsed", None) is not None:
            content = json.dumps(as_jsonable(resp.parsed), ensure_ascii=False)

        result = {
            "role": "assistant",
            "content": content,
        }
        result.update(normalize_gemini_usage(getattr(resp, "usage_metadata", None)))
        if tool_calls:
            result["tool_calls"] = tool_calls
        if thought_parts or thought_signatures:
            result["reasoning_trace"] = [{
                "provider": self.facade._trace_provider_name(),
                "model": self.facade.model,
                "summary_text": "\n".join(thought_parts).strip(),
                "summary_parts": thought_parts,
                "raw_reasoning": None,
                "encrypted_content": None,
                "thought_signatures": thought_signatures,
                "source": "gemini.thought_summary",
            }]
        return result

    def _to_gemini_contents(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[gemini_types.Content]]:
        system_parts: list[str] = []
        contents: list[gemini_types.Content] = []
        tool_name_by_id: dict[str, str] = {}

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                if msg.get("content"):
                    system_parts.append(str(msg["content"]))
                continue

            if role == "user":
                contents.append(
                    gemini_types.Content(
                        role="user",
                        parts=[gemini_types.Part.from_text(text=str(msg.get("content", "")))],
                    )
                )
                continue

            if role == "assistant":
                parts: list[gemini_types.Part] = []
                if msg.get("content"):
                    parts.append(gemini_types.Part.from_text(text=str(msg["content"])))
                for tool_call in msg.get("tool_calls", []):
                    call_id = tool_call.get("id", "")
                    call_name = tool_call.get("function", {}).get("name", "")
                    tool_name_by_id[call_id] = call_name
                    fc_part = gemini_types.Part.from_function_call(
                        name=call_name,
                        args=try_load_json(
                            tool_call.get("function", {}).get("arguments", "{}"),
                        ),
                    )
                    # Gemini 3 要求 function_call 带 id，并在 function_response 回传同一 id。
                    if call_id and fc_part.function_call is not None:
                        fc_part.function_call.id = call_id
                    thought_sig = tool_call.get("thought_signature")
                    if thought_sig:
                        fc_part.thought_signature = base64.b64decode(thought_sig) if isinstance(thought_sig, str) else thought_sig
                    parts.append(fc_part)
                if parts:
                    contents.append(gemini_types.Content(role="model", parts=parts))
                continue

            if role == "tool":
                call_id = msg.get("tool_call_id", "")
                tool_name = tool_name_by_id.get(call_id, "unknown_tool")
                tool_result = try_load_json(str(msg.get("content", "")))
                if not isinstance(tool_result, dict):
                    tool_result = {"result": tool_result}
                fr_part = gemini_types.Part.from_function_response(
                    name=tool_name,
                    response=tool_result,
                )
                # 回传与 function_call 相同的 id（Gemini 3 用于精确映射 request/response）。
                if call_id and fr_part.function_response is not None:
                    fr_part.function_response.id = call_id
                contents.append(
                    gemini_types.Content(role="user", parts=[fr_part])
                )

        system_text = "\n\n".join(p for p in system_parts if p).strip()
        return system_text or None, contents


__all__ = ["GeminiProviderClient"]
