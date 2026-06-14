"""Claude provider 实现。"""

from __future__ import annotations

import json
from typing import Any

import anthropic

from src.llm.providers.base import BaseProviderClient
from src.llm.utils import (
    as_jsonable,
    augment_system_for_json,
    normalize_claude_usage,
    openai_tools_to_anthropic,
    try_load_json,
)


class ClaudeProviderClient(BaseProviderClient):
    """Claude provider adapter。"""

    def __init__(self, config: dict[str, Any], facade: Any):
        super().__init__(config, facade)
        client_kwargs: dict[str, Any] = {"api_key": facade.api_key}
        if facade.base_url:
            client_kwargs["base_url"] = facade.base_url
        self.client = anthropic.Anthropic(**client_kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        system_text, anthropic_messages = self._to_claude_messages(messages)
        if json_mode:
            system_text = augment_system_for_json(system_text)

        kwargs: dict[str, Any] = {
            "model": self.facade.model,
            "messages": anthropic_messages,
        }
        kwargs.update(self.facade._mapped_request_values(max_tokens))
        if system_text:
            kwargs["system"] = system_text

        anthropic_tools = openai_tools_to_anthropic(tools)
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
            kwargs["tool_choice"] = {"type": "auto"}

        if self.facade.thinking_budget:
            thinking = {
                "type": "enabled",
                "budget_tokens": self.facade.thinking_budget,
            }
            if self.facade.thinking_display:
                thinking["display"] = self.facade.thinking_display
            kwargs["thinking"] = thinking

        resp = self.client.messages.create(**kwargs)

        content_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        reasoning_trace: list[dict[str, Any]] = []
        # 原样保留 thinking / redacted_thinking block（含 signature）：多轮 tool use 时
        # 官方 API 要求把它们回传在 tool_use 之前，否则会报错（见 _to_claude_messages）。
        raw_thinking_blocks: list[dict[str, Any]] = []
        for idx, block in enumerate(resp.content):
            block_type = getattr(block, "type", None)
            if block_type == "text":
                content_parts.append(getattr(block, "text", ""))
                continue
            if block_type in ("thinking", "redacted_thinking"):
                raw_thinking_blocks.append(as_jsonable(block))
                thinking_text = getattr(block, "thinking", "")
                reasoning_trace.append({
                    "provider": self.facade._trace_provider_name(),
                    "model": self.facade.model,
                    "summary_text": thinking_text,
                    "summary_parts": [thinking_text] if thinking_text else [],
                    "raw_reasoning": thinking_text,
                    "encrypted_content": getattr(block, "encrypted_content", None),
                    "thought_signatures": [],
                    "source": f"claude.{block_type}",
                })
                continue
            if block_type == "tool_use":
                tool_calls.append({
                    "id": getattr(block, "id", None) or f"tool_call_{idx}",
                    "type": "function",
                    "function": {
                        "name": getattr(block, "name", ""),
                        "arguments": json.dumps(
                            as_jsonable(getattr(block, "input", {})),
                            ensure_ascii=False,
                        ),
                    },
                })

        result = {
            "role": "assistant",
            "content": "".join(content_parts).strip(),
        }
        result.update(normalize_claude_usage(getattr(resp, "usage", None)))
        if tool_calls:
            result["tool_calls"] = tool_calls
        if reasoning_trace:
            result["reasoning_trace"] = reasoning_trace
        if raw_thinking_blocks:
            result["_claude_thinking_blocks"] = raw_thinking_blocks
        return result

    def _to_claude_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role")
            if role == "system":
                if msg.get("content"):
                    system_parts.append(str(msg["content"]))
                continue

            if role == "user":
                converted.append({"role": "user", "content": str(msg.get("content", ""))})
                continue

            if role == "assistant":
                content_blocks: list[dict[str, Any]] = []
                # thinking block 必须在 text / tool_use 之前、且原样（含 signature）回传。
                for thinking_block in msg.get("_claude_thinking_blocks", []):
                    content_blocks.append(thinking_block)
                if msg.get("content"):
                    content_blocks.append({"type": "text", "text": str(msg["content"])})
                for tool_call in msg.get("tool_calls", []):
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tool_call.get("id"),
                        "name": tool_call.get("function", {}).get("name", ""),
                        "input": try_load_json(
                            tool_call.get("function", {}).get("arguments", "{}"),
                        ),
                    })
                converted.append({
                    "role": "assistant",
                    "content": content_blocks or str(msg.get("content", "")),
                })
                continue

            if role == "tool":
                converted.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id", ""),
                        "content": str(msg.get("content", "")),
                    }],
                })

        system_text = "\n\n".join(p for p in system_parts if p).strip()
        return system_text or None, converted


__all__ = ["ClaudeProviderClient"]
