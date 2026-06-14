"""Persona 生成：Active Fields (rubric-driven) + Pure Noise (safe fields only)。"""

import json
import logging
import random

from src.generation.config import (
    PERSONA_FIELD_MAPPING,
    SAFE_NOISE_FIELDS,
)
from src.llm import LLMClient
from src.prompts.generation import PERSONA_NOISE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def build_persona(
    persona_rubrics: list[dict],
    product: dict,
    llm_client: LLMClient,
    rng: random.Random | None = None,
) -> dict:
    """构建 persona JSON。

    Args:
        persona_rubrics: 分配给 persona 的 rubrics（含 field, expected_value）
        product: target product（用于推断 category）
        llm_client: LLM client for noise generation
    """
    rng = rng or random.Random()
    user_id = f"U_{rng.randint(10000, 99999)}"

    # 1. 构建 Active Zone
    product_requirements = _build_active_fields(persona_rubrics)

    # 2. 构建 Noise Zone (LLM 生成)
    category = product.get("main_category", "General")
    noise = _generate_noise_fields(category, product_requirements, llm_client)

    persona = {
        "user_id": user_id,
        **noise,
        "product_requirements": product_requirements,
    }

    return persona


def _build_active_fields(persona_rubrics: list[dict]) -> dict:
    """从 persona-source rubrics 构建 product_requirements。"""
    requirements = {}
    for rubric in persona_rubrics:
        field = rubric["field"]
        expected = rubric["expected_value"]
        rubric_type = rubric.get("type", "attribute_match")

        persona_key = PERSONA_FIELD_MAPPING.get(field)
        if not persona_key:
            bare_field = field.split(".")[-1] if "." in field else field
            persona_key = bare_field.lower().replace(" ", "_")

        if rubric_type == "negative_attribute":
            requirements.setdefault("avoid_preferences", []).extend(
                expected if isinstance(expected, list) else [expected]
            )
        elif rubric_type == "numeric_range":
            if isinstance(expected, dict):
                requirements[persona_key] = expected
            else:
                requirements[persona_key] = {"max": expected} if field == "price" else expected
        else:
            requirements[persona_key] = expected

    return requirements


def _generate_noise_fields(category: str, product_requirements: dict,
                           llm_client: LLMClient) -> dict:
    """用 LLM 生成安全的 noise 字段。"""
    try:
        user_msg = f"Product category: {category}"
        if product_requirements:
            user_msg += (
                "\n\nActive product requirements (already filled, do NOT touch):\n"
                + json.dumps(product_requirements, ensure_ascii=False)
            )
        result = llm_client.chat_json(messages=[
            {"role": "system", "content": PERSONA_NOISE_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ])
        noise = {}
        for section in ("demographics", "lifestyle", "shopping_habits"):
            section_data = result.get(section, {})
            if isinstance(section_data, dict):
                noise[section] = section_data
        return noise
    except Exception as e:
        logger.warning("Failed to generate persona noise: %s, using defaults", e)
        return _default_noise()


def _default_noise() -> dict:
    return {
        "demographics": {
            "gender": "not specified",
            "location": "United States",
            "occupation": "professional",
        },
        "lifestyle": {
            "hobbies": ["reading", "cooking"],
            "daily_routine": "regular work schedule",
            "fitness_level": "moderate",
        },
        "shopping_habits": {
            "payment_method": "credit card",
            "shopping_frequency": "monthly",
            "preferred_device": "mobile",
            "preferred_platform": "Amazon",
        },
    }


def validate_persona(persona: dict, persona_rubrics: list[dict]) -> list[str]:
    """验证 persona 噪声字段安全性（whitelist 检查）。"""
    issues = []
    for section in ("demographics", "lifestyle", "shopping_habits"):
        section_data = persona.get(section, {})
        if not isinstance(section_data, dict):
            continue
        for key in section_data:
            full_path = f"{section}.{key}"
            if full_path not in SAFE_NOISE_FIELDS:
                issues.append(f"Persona noise field '{full_path}' not in SAFE_NOISE_FIELDS whitelist")
    return issues
