"""Judge：evaluation 用 RubricJudge，generation 用 GenerationJudge。"""

import json
import logging

from src.llm import LLMClient
from src.prompts.evaluation import JUDGE_SYSTEM_PROMPT, GENERATION_JUDGE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RubricJudge:
    """使用LLM判断推荐结果是否满足各rubric条件。"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.model = llm_client.model

    def judge_sample(self, product: dict | None,
                     rubrics: list[dict],
                     voucher: dict | None = None) -> dict:
        """评判一个样本：逐条 rubric 评测。

        Args:
            product: 模型推荐的商品（None 表示未推荐或 ID 无效）
            rubrics: rubric 列表

        Returns:
            {"results": list[dict]}
        """
        if not product:
            return {"results": self._build_fallback(rubrics, reason="No valid product recommended")}

        prompt_parts = [
            "=== RECOMMENDED PRODUCT ===",
        ]

        p_copy = dict(product)
        reviews = p_copy.pop("reviews", None)
        prompt_parts.append(json.dumps(p_copy, ensure_ascii=False, indent=2))
        if reviews:
            prompt_parts.append(f"\nReviews ({len(reviews)} reviews):")
            for ri, rev in enumerate(reviews[:20], 1):
                prompt_parts.append(
                    f"  Review {ri}: [{rev.get('rating', 'N/A')}*] "
                    f"{rev.get('title', '')} - {rev.get('text', '')[:500]}"
                )

        if voucher:
            prompt_parts.append("\n=== VOUCHER / BUDGET CONTEXT ===")
            prompt_parts.append(json.dumps(voucher, ensure_ascii=False, indent=2))

        prompt_parts.append("\n=== RUBRICS ===")
        for rb in rubrics:
            prompt_parts.append(
                f"  - [{rb['id']}] ({rb['type']}) {rb['description']}"
                + (f" | expected: {rb['expected_value']}" if rb.get("expected_value") else "")
            )

        result = self.llm.chat_json(
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(prompt_parts)},
            ]
        )
        results = result.get("results", [])

        # 兼容：如果 LLM 返回了嵌套结构，展平
        if results and isinstance(results[0], list):
            results = [r for group in results for r in group]

        if len(results) != len(rubrics):
            logger.warning(f"Judge returned {len(results)} results, expected {len(rubrics)}, building fallback")
            results = self._build_fallback(rubrics)

        return {"results": results}

    def _build_fallback(self, rubrics: list[dict], reason: str = "Judge response structure mismatch") -> list[dict]:
        return [
            {"rubric_id": rb["id"], "satisfied": False, "reasoning": reason}
            for rb in rubrics
        ]


class GenerationJudge:
    """Generation Judge：合并 query/persona/clarification 三个信息源验证全部 rubrics。"""

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
        self.model = llm_client.model

    def judge_generation_sample(
        self,
        *,
        user_query: str,
        user_persona: dict | None = None,
        clarification_script: dict | None = None,
        product: dict,
        rubrics: list[dict],
        voucher: dict | None = None,
        implicit_rubric_ids: list[str] | None = None,
    ) -> dict:
        p_copy = dict(product)
        reviews = p_copy.pop("reviews", None)
        reviews_section = ""
        if reviews:
            lines = [f"Reviews for target product ({len(reviews)} reviews):"]
            for ri, rev in enumerate(reviews[:20], 1):
                lines.append(
                    f"  Review {ri}: [{rev.get('rating', 'N/A')}*] "
                    f"{rev.get('title', '')} - {rev.get('text', '')[:500]}"
                )
            reviews_section = "\n".join(lines)

        cl_slots = []
        if clarification_script:
            for slot in clarification_script.get("clarification_slots", []):
                cl_slots.append({
                    "slot_id": slot.get("slot_id"),
                    "linked_rubric_ids": slot.get("linked_rubric_ids", []),
                    "trigger_keywords": slot.get("trigger_keywords", []),
                    "user_response": slot.get("user_response", ""),
                })

        payload = {
            "user_query": user_query,
            "user_persona": user_persona or {},
            "clarification_slots": cl_slots,
            "target_product": p_copy,
            "voucher": voucher,
            "rubrics": rubrics,
            "implicit_rubric_ids": implicit_rubric_ids or [],
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        if reviews_section:
            content += "\n\n" + reviews_section

        result = self.llm.chat_json(
            messages=[
                {"role": "system", "content": GENERATION_JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ]
        )
        passed = bool(result.get("passed"))
        issues = result.get("issues", [])
        rubric_results = result.get("rubric_results", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        if not isinstance(rubric_results, list):
            rubric_results = []
        return {
            "passed": passed,
            "issues": [str(issue) for issue in issues[:10]],
            "rubric_results": rubric_results,
        }


__all__ = ["RubricJudge", "GenerationJudge"]
