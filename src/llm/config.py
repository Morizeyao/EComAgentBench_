"""LLM 配置解析工具。

把 `configs/settings.yaml` 里的公共默认项、profile 和阶段级覆盖项
合并成可直接传给 `LLMClient` 的最终配置。
"""

from __future__ import annotations

from typing import Any


def _merge_non_none(*configs: dict[str, Any]) -> dict[str, Any]:
    """按顺序合并配置，忽略值为 None 的项。"""
    merged: dict[str, Any] = {}
    for config in configs:
        for key, value in (config or {}).items():
            if value is not None:
                merged[key] = value
    return merged


def resolve_llm_config(
    config: dict[str, Any],
    section: str,
    mode: str | None = None,
) -> dict[str, Any]:
    """解析 generation / prediction / evaluation 任一阶段的最终 LLM 配置。"""
    defaults = config.get("llm_defaults", {})
    profiles = config.get("llm_profiles", {})

    if section == "generation":
        section_config = config["generation"]
        profile_name = section_config.get("llm_profile")
        overrides = section_config.get("llm", {})
    elif section == "prediction":
        prediction_config = config["prediction"]
        resolved_mode = mode or prediction_config.get("mode", "openai_api")
        mode_config = prediction_config.get("modes", {}).get(resolved_mode)
        if mode_config is None:
            raise ValueError(f"Unknown prediction mode: {resolved_mode}")
        profile_name = mode_config.get("llm_profile")
        overrides = mode_config.get("llm", {})
    elif section == "evaluation":
        section_config = config["evaluation"]
        profile_name = section_config.get("llm_profile")
        overrides = section_config.get("llm", {})
    else:
        raise ValueError(f"Unsupported LLM config section: {section}")

    if profile_name and profile_name not in profiles:
        raise ValueError(f"Unknown LLM profile: {profile_name}")

    profile = profiles.get(profile_name, {})
    resolved = _merge_non_none(defaults, profile, overrides)
    resolved.setdefault("provider", "openai")
    return resolved


def resolve_prediction_mode_config(
    config: dict[str, Any],
    mode: str | None = None,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    """解析 prediction 阶段的 mode 配置和最终 LLM 配置。"""
    prediction_config = config["prediction"]
    resolved_mode = mode or prediction_config.get("mode", "openai_api")
    mode_config = prediction_config.get("modes", {}).get(resolved_mode)
    if mode_config is None:
        raise ValueError(f"Unknown prediction mode: {resolved_mode}")
    llm_config = resolve_llm_config(config, section="prediction", mode=resolved_mode)
    return resolved_mode, mode_config, llm_config
