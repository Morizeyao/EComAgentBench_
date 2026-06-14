"""OpenRouter provider 实现。"""

from __future__ import annotations

from typing import Any

from openai import OpenAI

from src.llm.providers.base import BaseProviderClient
from src.llm.providers.openai import (
    normalize_openai_chat_completion_output,
    normalize_openai_responses_output,
    to_openai_responses_input,
)
from src.llm.utils import merge_dict, openai_tools_to_responses


class OpenRouterProviderClient(BaseProviderClient):
    """OpenRouter provider adapter。"""

    def __init__(self, config: dict[str, Any], facade: Any):
        super().__init__(config, facade)
        client_kwargs: dict[str, Any] = {
            "api_key": facade.api_key,
            "base_url": facade.base_url,
        }
        default_headers = {}
        http_referer = config.get("http_referer")
        app_title = config.get("app_title")
        if http_referer:
            default_headers["HTTP-Referer"] = http_referer
        if app_title:
            default_headers["X-Title"] = app_title
        if default_headers:
            client_kwargs["default_headers"] = default_headers
        self.client = OpenAI(**client_kwargs)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        if self.facade.use_responses_api:
            return self._chat_openrouter_responses(messages, tools=tools, json_mode=json_mode, max_tokens=max_tokens)
        return self._chat_openrouter_chat_completions(messages, tools=tools, json_mode=json_mode, max_tokens=max_tokens)

    def _build_openrouter_reasoning(self) -> dict[str, Any] | None:
        if isinstance(self.config.get("reasoning"), dict):
            reasoning = dict(self.config["reasoning"])
        else:
            reasoning = {"max_tokens": 2048}

        if self.facade.capture_reasoning:
            reasoning.setdefault("exclude", False)
        elif reasoning:
            reasoning.setdefault("exclude", True)

        return reasoning or None

    def _build_openrouter_extra_body(self, reasoning: dict[str, Any] | None = None) -> dict[str, Any]:
        extra_body = merge_dict(self.facade.extra_body, None) or {}
        plugins = [
            p for p in extra_body.get("plugins", [])
            if not (isinstance(p, dict) and p.get("id") == "context-compression")
        ]
        plugins.append({"id": "context-compression", "enabled": False})
        extra_body["plugins"] = plugins
        providers = self.config.get("providers") or []
        if providers:
            provider = dict(extra_body.get("provider") or {})
            provider["only"] = list(providers)
            extra_body["provider"] = provider
        if reasoning:
            extra_body["reasoning"] = reasoning
        return extra_body

    def _to_openrouter_chat_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for msg in messages:
            converted_msg = {
                key: value
                for key, value in msg.items()
                if not key.startswith("_openrouter_")
            }
            reasoning_text = msg.get("_openrouter_reasoning")
            if reasoning_text:
                converted_msg["reasoning"] = reasoning_text
            reasoning_details = msg.get("_openrouter_reasoning_details")
            if reasoning_details:
                converted_msg["reasoning_details"] = reasoning_details
            converted.append(converted_msg)
        return converted

    def _chat_openrouter_responses(
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
        kwargs.update(self.facade._mapped_request_values_for_profile("openrouter_responses", max_tokens))
        if instructions:
            kwargs["instructions"] = instructions

        reasoning = self._build_openrouter_reasoning()
        if reasoning:
            kwargs["reasoning"] = reasoning

        responses_tools = openai_tools_to_responses(tools)
        if responses_tools:
            kwargs["tools"] = responses_tools
            kwargs["tool_choice"] = "auto"
            if self.facade.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = bool(self.facade.parallel_tool_calls)
        kwargs["extra_body"] = self._build_openrouter_extra_body()

        resp = self.client.responses.create(**kwargs)
        return normalize_openai_responses_output(self.facade, resp)

    def _chat_openrouter_chat_completions(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.facade.model,
            "messages": self._to_openrouter_chat_messages(messages),
        }
        kwargs.update(self.facade._mapped_request_values_for_profile("openrouter_chat", max_tokens))
        reasoning = self._build_openrouter_reasoning()
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            if self.facade.parallel_tool_calls is not None:
                kwargs["parallel_tool_calls"] = bool(self.facade.parallel_tool_calls)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        kwargs["extra_body"] = self._build_openrouter_extra_body(reasoning)

        resp = self.client.chat.completions.create(**kwargs)
        if not resp.choices:
            raise RuntimeError("OpenRouter returned empty choices, temporarily unavailable")
        return normalize_openai_chat_completion_output(
            self.facade,
            resp.choices[0].message,
            getattr(resp, "usage", None),
        )


__all__ = ["OpenRouterProviderClient"]
