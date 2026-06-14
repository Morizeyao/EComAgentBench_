"""从正向 review 中抽取原子观点，用于 review_driven 意图。"""

import random

from src.llm import LLMClient

OPINION_EXTRACTION_PROMPT = """You are analyzing a positive product review to extract one atomic opinion.

Product title: {product_title}

Review:
Rating: {rating}
Title: {review_title}
Text: {review_text}

Extract exactly ONE concise opinion phrase (3-10 words) that represents a specific, verifiable claim about the product.
The opinion must be clearly supported by this review and should describe one atomic user-observable experience or property.

Rules:
- Do NOT extract opinions that merely restate the product title or catalog specifications. Focus on subjective user experiences that go beyond what the product listing already says.
- Do NOT extract overly subjective or unverifiable feelings (e.g. "love it", "great product").
- The opinion must be a single atomic claim, not a bundle of multiple claims.

Good examples: "perfect for short hair", "battery lasts all day", "adds noticeable volume", "lightweight and easy to carry", "smells nice without overpowering"
Bad examples: "great product", "I love it", "5 stars" (too vague), "the XYZ-2000 model" (too specific/named), "good quality and nice color" (multiple claims), "volumizing hair spray" (restates product title)

Respond in JSON:
{{"opinion": "concise opinion phrase"}}"""


def extract_review_opinions(
    reviews: list[dict],
    product: dict,
    llm_client: LLMClient,
    n_opinions: int = 2,
    rng: random.Random | None = None,
) -> list[dict]:
    """从高分 review 中抽取原子观点。

    仅从 rating >= 4.0 的 review 中抽取。
    优先选择 text 长度 >= 30 且 helpful_vote 高的 review。
    对每条 seed review 调用一次 LLM 抽取一个观点。

    Returns:
        [{"opinion": str, "source_review_id": int, "source_review_rating": float}]
    """
    rng = rng or random.Random()

    eligible = [
        r for r in reviews
        if float(r.get("rating") or 0) >= 4.0 and len(r.get("text", "")) >= 30
    ]
    if len(eligible) < n_opinions:
        eligible = [
            r for r in reviews
            if float(r.get("rating") or 0) >= 4.0
        ]
    if len(eligible) < n_opinions:
        return []

    eligible.sort(key=lambda r: r.get("helpful_vote", 0), reverse=True)
    top_pool = eligible[:min(len(eligible), max(n_opinions * 3, 6))]
    seed_reviews = rng.sample(top_pool, min(n_opinions, len(top_pool)))

    opinions = []
    for review in seed_reviews:
        prompt = OPINION_EXTRACTION_PROMPT.format(
            product_title=product.get("title", ""),
            rating=review.get("rating", "N/A"),
            review_title=review.get("title", ""),
            review_text=review.get("text", ""),
        )
        result = llm_client.chat_json(messages=[
            {"role": "user", "content": prompt},
        ])

        opinion_text = str(result.get("opinion", "")).strip()

        if opinion_text:
            opinions.append({
                "opinion": opinion_text,
                "source_review_id": review.get("review_id"),
                "source_review_rating": review.get("rating"),
            })

    return opinions
