"""项目级 LLM 调用统一入口。"""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Any

from src.llm.providers import build_provider_client
from src.llm.utils import (
    _OPENAI_REASONING_PREFIXES,
    _REQUEST_PARAM_MAPS,
    apply_param_mapping,
    normalize_provider,
    normalize_token_usage,
    parse_json_text,
    resolve_api_key,
    resolve_base_url,
    resolve_model,
)

logger = logging.getLogger(__name__)


class LLMClient:
    """项目级统一 LLM 调用客户端。"""

    def __init__(self, mode: str = "openai", config: dict[str, Any] | None = None):
        config = config or {}

        self.mode = mode
        self.provider = normalize_provider(mode, config)
        self.config = config

        self.model = resolve_model(self.provider, config)
        self.base_url = resolve_base_url(self.provider, config)
        self.api_key = resolve_api_key(self.provider, config)

        self.temperature = config.get("temperature")
        self.max_tokens = config.get("max_tokens", 4096)
        self.top_p = config.get("top_p")
        self.timeout = config.get("timeout")

        self.reasoning_effort = config.get("reasoning_effort")
        self.use_responses_api = bool(config.get("use_responses_api", False))
        self.capture_reasoning = bool(config.get("capture_reasoning", True))
        self.reasoning_summary = config.get("reasoning_summary", "auto")
        self.parallel_tool_calls = config.get("parallel_tool_calls")

        self.thinking_budget = config.get("thinking_budget")
        self.include_thoughts = bool(config.get("include_thoughts", False))
        self.thinking_display = config.get("thinking_display")
        self.extra_body = config.get("extra_body")
        self.api_version = config.get("api_version", "v1")
        self.max_retries = max(1, int(config.get("max_retries", 5)))
        self.retry_base_delay = float(config.get("retry_base_delay", 5.0))

        self.total_input_tokens = 0
        self.total_thinking_tokens = 0
        self.total_output_tokens = 0
        self._token_usage_lock = Lock()

        self.provider_client = build_provider_client(self.provider, config, self)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return self._with_retries(
            lambda: self._chat_once(messages, tools=tools, json_mode=json_mode, max_tokens=max_tokens),
            op_name="chat",
        )

    def chat_json(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        return self._with_retries(
            lambda: self._chat_json_once(messages, max_tokens=max_tokens),
            op_name="chat_json",
        )

    def _chat_json_once(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int | None,
    ) -> dict[str, Any]:
        assistant_msg = self._chat_once(messages, tools=None, json_mode=True, max_tokens=max_tokens)
        if assistant_msg.get("tool_calls"):
            raise ValueError("JSON mode received unexpected tool calls")
        result = parse_json_text(assistant_msg.get("content", ""))
        result["input_tokens"] = assistant_msg["input_tokens"]
        result["thinking_tokens"] = assistant_msg["thinking_tokens"]
        result["output_tokens"] = assistant_msg["output_tokens"]
        return result

    def _chat_once(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        json_mode: bool,
        max_tokens: int | None,
    ) -> dict[str, Any]:
        assistant_msg = self.provider_client.chat(
            self._strip_token_usage_from_messages(messages),
            tools=tools,
            json_mode=json_mode,
            max_tokens=max_tokens,
        )
        assistant_msg = dict(assistant_msg)
        assistant_msg.update(
            normalize_token_usage(
                assistant_msg.get("input_tokens"),
                assistant_msg.get("output_tokens"),
                assistant_msg.get("thinking_tokens"),
            )
        )
        self._record_token_usage(assistant_msg)
        return assistant_msg

    def _record_token_usage(self, message: dict[str, Any]) -> None:
        with self._token_usage_lock:
            self.total_input_tokens += int(message.get("input_tokens", 0) or 0)
            self.total_thinking_tokens += int(message.get("thinking_tokens", 0) or 0)
            self.total_output_tokens += int(message.get("output_tokens", 0) or 0)

    def get_token_usage(self) -> dict[str, int]:
        with self._token_usage_lock:
            return {
                "input_tokens": self.total_input_tokens,
                "thinking_tokens": self.total_thinking_tokens,
                "output_tokens": self.total_output_tokens,
            }

    @staticmethod
    def _strip_token_usage_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned = []
        for msg in messages:
            cleaned_msg = dict(msg)
            cleaned_msg.pop("input_tokens", None)
            cleaned_msg.pop("thinking_tokens", None)
            cleaned_msg.pop("output_tokens", None)
            cleaned.append(cleaned_msg)
        return cleaned

    def _with_retries(self, operation, op_name: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return operation()
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                retryable = self._is_retryable_error(exc)
                if not retryable or attempt >= self.max_retries:
                    raise
                delay = self.retry_base_delay
                logger.warning(
                    "LLM %s failed (%s/%s): %s; retrying in %.1fs",
                    op_name, attempt, self.max_retries, exc, delay,
                )
                time.sleep(delay)
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"LLM {op_name} failed without a captured exception")

    def _is_retryable_error(self, exc: Exception) -> bool:
        import json as _json

        if isinstance(exc, _json.JSONDecodeError):
            return True
        status_code = getattr(exc, "status_code", None)
        if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
            return True
        response = getattr(exc, "response", None)
        response_status = getattr(response, "status_code", None)
        if isinstance(response_status, int) and (response_status == 429 or response_status >= 500):
            return True

        text = str(exc).lower()
        retry_terms = (
            "429",
            "resource_exhausted",
            "rate limit",
            "too many requests",
            "temporarily unavailable",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "server error",
            "internal error",
            "bad gateway",
            "service unavailable",
            "gateway timeout",
            "overloaded",
            "try again",
            "failed to parse json object",
            "empty json content",
            "unexpected tool calls",
            "provider returned error",
        )
        return any(term in text for term in retry_terms)

    def _request_profile_name(self) -> str:
        model_name = (self.model or "").lower()
        if self.provider == "openai":
            if self.use_responses_api:
                return "openai_responses"
            if model_name.startswith(_OPENAI_REASONING_PREFIXES):
                return "openai_reasoning"
            return "openai_chat"
        if self.provider == "openrouter":
            if self.use_responses_api:
                return "openrouter_responses"
            return "openrouter_chat"
        if self.provider == "gemini":
            return "gemini"
        if self.provider == "claude":
            return "claude"
        if model_name.startswith(_OPENAI_REASONING_PREFIXES):
            return "openai_reasoning"
        return "openai_chat"

    def _common_request_values(self, max_tokens: int | None) -> dict[str, Any]:
        return {
            "token_limit": max_tokens or self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }

    def _trace_provider_name(self) -> str:
        return self.provider

    def _mapped_request_values_for_profile(
        self,
        profile_name: str,
        max_tokens: int | None,
        **extra_values: Any,
    ) -> dict[str, Any]:
        values = self._common_request_values(max_tokens)
        values.update(extra_values)
        mapped: dict[str, Any] = {}
        apply_param_mapping(mapped, values, _REQUEST_PARAM_MAPS[profile_name])
        return mapped

    def _mapped_request_values(
        self,
        max_tokens: int | None,
        **extra_values: Any,
    ) -> dict[str, Any]:
        return self._mapped_request_values_for_profile(
            self._request_profile_name(),
            max_tokens,
            **extra_values,
        )
