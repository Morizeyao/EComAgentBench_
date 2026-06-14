"""Query Builder：处理 query-source rubrics。"""

import random

from src.generation.config import get_min_implicit_count
from src.generation.composition.prompt_builder import (
    build_query_prompt,
    choose_opening_style,
)
from src.generation.composition.rubric_compiler import compile_final_rubrics
from src.llm import LLMClient
from src.prompts.generation import QUERY_SYSTEM_PROMPT


class QueryBuilder:
    """query 生成：处理 query-source rubrics。"""

    def __init__(self, llm_client: LLMClient, seed: int = 42):
        self.llm = llm_client
        self.rng = random.Random(seed)

    def build(self, intent: dict,
              product: dict,
              query_rubrics: list[dict],
              voucher: dict | None = None,
              excluded_values: list[str] | None = None,
              max_rubric_id: int = 0) -> dict:
        """调用 LLM 生成 query（只包含 query-source rubrics 信息）。"""
        min_implicit_count = get_min_implicit_count(intent["id"])

        implicit_eligible_count = sum(
            1 for r in query_rubrics if r.get("implicit_eligible")
        )
        effective_min_implicit = min(min_implicit_count, implicit_eligible_count)

        prompt = build_query_prompt(
            intent=intent,
            product=product,
            rubrics=query_rubrics,
            voucher=voucher,
            opening_style=choose_opening_style(intent["id"], self.rng),
            min_implicit_count=effective_min_implicit,
        )

        if excluded_values:
            items = "\n".join(f"- {v}" for v in excluded_values)
            prompt += (
                "\n\nIMPORTANT: The following values are handled by other information "
                "sources (user profile / follow-up questions) and must NOT appear in "
                "the query — do not mention, paraphrase, or hint at them:\n" + items
            )

        messages = [
            {"role": "system", "content": QUERY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        result = self.llm.chat_json(messages=messages)
        user_query = result.get("user_query", "")
        compiled = compile_final_rubrics(
            rubrics=query_rubrics,
            product=product,
            rubric_surfaces=result.get("rubric_surfaces", []),
            negative_constraints=result.get("negative_constraints", []),
            entity_plans=result.get("entity_plans", []),
            max_rubric_id=max_rubric_id,
            user_query=user_query,
        )
        implicit_rubric_ids = _normalize_implicit_rubric_ids(
            result.get("implicit_rubric_ids", [])
        )
        return {
            "user_query": user_query,
            "rubrics": compiled,
            "implicit_rubric_ids": implicit_rubric_ids,
            "min_implicit_count": effective_min_implicit,
        }


def _normalize_implicit_rubric_ids(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    normalized: list[str] = []
    for item in raw:
        rubric_id = str(item).strip()
        if rubric_id and rubric_id not in normalized:
            normalized.append(rubric_id)
    return normalized
