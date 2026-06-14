"""LLM provider registry。"""

from __future__ import annotations

from typing import Any

from .claude import ClaudeProviderClient
from .gemini import GeminiProviderClient
from .openai import OpenAIProviderClient
from .openrouter import OpenRouterProviderClient

_PROVIDER_CLIENTS = {
    "openai": OpenAIProviderClient,
    "openrouter": OpenRouterProviderClient,
    "claude": ClaudeProviderClient,
    "gemini": GeminiProviderClient,
}


def build_provider_client(provider: str, config: dict[str, Any], facade: Any):
    client_cls = _PROVIDER_CLIENTS.get(provider)
    if client_cls is None:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return client_cls(config=config, facade=facade)


__all__ = [
    "build_provider_client",
    "ClaudeProviderClient",
    "GeminiProviderClient",
    "OpenAIProviderClient",
    "OpenRouterProviderClient",
]
