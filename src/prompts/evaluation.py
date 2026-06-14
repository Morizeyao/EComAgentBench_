"""Evaluation prompts."""

RUBRIC_EVALUATION_RULES = """\
- attribute_match: The rubric's "field" indicates where to look (e.g. "details.Material"). Follow this lookup order:
  1. First check the structured "details" dict for the specified key. If the value matches, mark satisfied.
  2. If not found in details, check the "features" list for evidence. ONLY accept a features-based match if the feature text contains a PRECISE equivalent of the expected value, or the product's actual attribute is a strict SUBSET of what the user query requires (i.e., the product's value is MORE specific than or exactly equal to the requirement). NEVER broaden the requirement. For example: if the rubric expects "Laptop", a product stating "Personal Computer" does NOT satisfy it because "Personal Computer" is broader than "Laptop". If the rubric expects "Thermoplastic Polyurethane", a product feature saying "made of TPU" DOES satisfy it because TPU is the standard abbreviation.
  3. If neither source provides clear evidence, mark as NOT satisfied.
- negative_attribute: The rubric's "field" specifies the exact attribute path (e.g. "details.Mounting Type"). Look up the product's value at that specific field path. expected_value is a list of EXCLUDED values (e.g. ["blue", "green"]). Verify the product's actual value at that field does NOT match any item in the excluded list. Mark as satisfied only if the product's actual value is clearly different from every excluded value. If the product's value partially overlaps with an excluded value (e.g. "blue-green" vs excluded "blue"), mark as NOT satisfied.
- numeric_range: expected_value can be in several formats. Check accordingly:
  * A single number: exact match required.
  * {"min": X}: actual value must be >= X.
  * {"max": X}: actual value must be <= X.
  * {"min": X, "max": Y}: actual value must be between X and Y inclusive.
  * {"value": X, "op": ">"}: actual value must satisfy the comparison (supported ops: >, >=, <, <=, ==).
  * [X, Y]: actual value must be between X and Y inclusive.
- store_match: verify the product is from the specified store.
- budget_match: if voucher context is provided, apply the voucher rules and then verify the final price is within budget. Do not judge budget_match using raw list price when a voucher is available. Voucher discount_type variants:
  * "fixed" or "percentage" (simple): apply the single discount if product price >= threshold, then check final price <= expected_value.
  * "tiered": the voucher has a "tiers" list and an "active_tier" field. Use the active tier (always tier 1 by benchmark construction) to determine the applicable discount. Apply that tier's discount to the product price and verify final price <= expected_value.
  * "stacked": the voucher has a "coupons" list. Apply coupon 0 first to the full price, then apply coupon 1 to the intermediate price. Verify the cumulative final price <= expected_value.
  In all cases, the voucher's "price_after_voucher" field is the pre-computed correct final price for the target product — use it as the reference.
- entity_match: check if the product title contains the required core entity concept. The match should be based on whether the product IS the type of entity described. For example, if the entity is "screen protector", the title must clearly indicate this is a screen protector product. Semantic equivalences are acceptable only when they refer to the exact same product type (e.g. "phone case" matches "case for phone"), but different product categories must NOT match (e.g. "room divider" does NOT match "desk panel").
- review_opinion: expected_value contains {"opinion": "..."} and may also include source_review_id/source_review_rating metadata. Check if any review of the recommended product clearly expresses that same specific opinion. The review text need not match word-for-word, but it must convey the same concrete claim, not just a generic positive feeling. If review data is provided for the product, use it to verify. If no review data is available, mark as NOT satisfied.
- If information is ambiguous or missing, mark as NOT satisfied."""

JUDGE_SYSTEM_PROMPT = f"""You are an expert e-commerce evaluation judge.

You will receive:
1. The RECOMMENDED product (what the model actually recommended).
2. Optional voucher/budget context.
3. A list of evaluation rubrics for the required product.

Your task: Evaluate every rubric against the recommended product. Be strict but fair - only mark as satisfied if there is clear evidence.

Product data structure:
- The product has two data sources: "details" (structured key-value pairs like {{"Brand": "Sony", "Material": "TPU"}}) and "features" (a list of natural-language bullet-point descriptions). Check BOTH sources for evidence.

Rubric evaluation rules:
{RUBRIC_EVALUATION_RULES}

Respond in JSON format:
{{
  "results": [
    {{"rubric_id": "r1", "satisfied": true, "reasoning": "brief explanation"}},
    {{"rubric_id": "r2", "satisfied": false, "reasoning": "brief explanation"}}
  ]
}}

The "results" array must contain one entry per rubric, in the same order as the input rubrics."""

GENERATION_JUDGE_SYSTEM_PROMPT = f"""You are a strict benchmark-generation judge for a multi-source e-commerce benchmark.

You will receive:
1. user_query — the shopping query
2. user_persona — the user's profile; product_requirements contains rubric-driven fields
3. clarification_script — slots with hidden info the user reveals when asked
4. target_product — the product that should satisfy all rubrics
5. rubrics — each rubric has an "info_source" field: "query", "persona", or "clarification"
6. implicit_rubric_ids — rubrics whose expected value is hidden in the query via a reasoning clue
7. optional voucher/budget context

Decide whether this generated benchmark sample is FAIR and USABLE.

=== SOURCE-RUBRIC ALIGNMENT ===

A rubric must be supported by EXACTLY its designated info_source:

For info_source="query" rubrics:
- The rubric's requirement must be clearly expressed in user_query.
- The support must not be weak, arbitrary, or overly interpretive.
- For implicit rubric ids, the query must hide the literal expected value while making it reasonably inferable via a one-hop reasoning clue (not a synonym or paraphrase).

For info_source="persona" rubrics:
- The rubric's expected_value must be present in user_persona.product_requirements (under the appropriate key).
- The value in the persona must match or be a deterministic implicit expression of the expected_value.

For info_source="clarification" rubrics:
- The rubric must have a corresponding clarification slot (linked via linked_rubric_ids).
- The slot's user_response must contain the expected_value or a natural expression clearly conveying it.
- The slot must have reasonable trigger_keywords related to the topic.

=== CROSS-SOURCE ISOLATION ===

- user_query must NOT contain expected_values from persona-source or clarification-source rubrics. These values should only appear in their designated source.
- user_query must NOT introduce product requirements (specific values, constraints, brand names, numeric thresholds, etc.) that are not backed by any query-source rubric. Every product constraint expressed in user_query must correspond to exactly one query-source rubric.
- user_persona.product_requirements must NOT contain extra product-related fields that don't correspond to any persona-source rubric.
- clarification slots must NOT contain extra product requirements beyond their linked rubrics.

=== TARGET PRODUCT SATISFACTION ===

Every target product must clearly satisfy ALL rubrics regardless of info_source:
{RUBRIC_EVALUATION_RULES}

=== BOUNDARY EXAMPLES ===

- PASS: query says "withstands sharp objects without getting marked up", rubric expects "Scratch Resistant" (direct property implication)
- PASS: query says "older trapezoid-shaped USB plug", rubric expects "Micro USB" (unique physical description)
- PASS: persona has product_requirements.os_preference = "Windows 11", rubric expects Operating System = "Windows 11" (direct match)
- PASS: clarification slot user_response says "I need at least 16 GB of RAM", rubric expects RAM = "16 GB" (value in response)
- FAIL: query says "good quality material", rubric expects "Scratch Resistant" (too vague)
- FAIL: query mentions "16 GB RAM" but the rubric is persona-source (cross-source violation)
- FAIL: persona has product_requirements.brand_preference = "Samsung" but no persona-source rubric for Brand (extra requirement)
- FAIL: query mentions "free shipping" or a specific brand not covered by any query-source rubric (extra constraint in query)

Be conservative. If any source-to-rubric mapping feels weak, mark the sample as failed.

Respond in JSON:
{{
  "passed": true/false,
  "issues": ["short issue 1", "short issue 2"],
  "rubric_results": [
    {{"rubric_id": "r1", "info_source": "query", "supported": true, "reasoning": "brief reason"}}
  ]
}}"""

__all__ = [
    "RUBRIC_EVALUATION_RULES",
    "JUDGE_SYSTEM_PROMPT",
    "GENERATION_JUDGE_SYSTEM_PROMPT",
]
