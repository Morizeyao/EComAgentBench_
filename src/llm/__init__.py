"""项目级 LLM 统一入口。"""

from .client import LLMClient
from .config import resolve_llm_config, resolve_prediction_mode_config

__all__ = ["LLMClient", "resolve_llm_config", "resolve_prediction_mode_config"]
