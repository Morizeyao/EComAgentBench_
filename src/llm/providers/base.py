"""LLM provider adapter 基类。"""

from __future__ import annotations

from typing import Any


class BaseProviderClient:
    """最小 provider adapter 接口。"""

    def __init__(self, config: dict[str, Any], facade: Any):
        self.config = config
        self.facade = facade

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        json_mode: bool = False,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        raise NotImplementedError
