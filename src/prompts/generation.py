"""Data generation prompts."""

FEWSHOT_EXAMPLES = [
    {
        "id": "beauty_hidden_clues",
        "title": "Beauty: knowledge reasoning with two hidden clues",
        "query": "I need an adult body moisturizer with a 24K Egyptian Musk scent for skin that gets tight and flaky, and I want it to come out as a light whipped foam instead of a standard cream.",
        "implicit_rubric_ids": ["r1", "r4"],
        "note": "tight and flaky implies Dry (skin type); light whipped foam implies Mousse (item form). Both require domain knowledge, not synonym substitution.",
        "fewshot_role": "implicit_core",
        "intent_ids": [],
    },
    {
        "id": "electronics_multi_hop",
        "title": "Electronics: multi-hop scenario reasoning",
        "query": "I need headphones for my daily subway commute that completely seal off outside noise by cupping around my entire ear, and they should use the same wireless standard my AirPods use.",
        "implicit_rubric_ids": ["r2", "r5"],
        "note": "cupping around entire ear implies Over Ear (form factor); same standard as AirPods implies Bluetooth (connectivity). Each requires one inference hop from the scenario.",
        "fewshot_role": "implicit_core",
        "intent_ids": [],
    },
    {
        "id": "cellphones_problem_solution",
        "title": "CellPhones: problem-implies-solution",
        "query": "I keep dropping my phone face-down on concrete and the corners always crack first — I need a case that survives those drops. It should be made from that same rubbery polymer they use in car dashboards.",
        "implicit_rubric_ids": ["r1", "r3"],
        "note": "corners crack first on face-down drops implies Bumper (form factor via the problem it solves); rubbery polymer in car dashboards implies TPU (material via world knowledge). Neither clue directly names the attribute value.",
        "fewshot_role": "implicit_core",
        "intent_ids": [],
    },
    {
        "id": "negative_constraint",
        "title": "Negative constraint: excluding specific values in a dimension",
        "query": "I need a power adapter for my Sony Blu-Ray player with the specific round plug that slides over a center pin. It must be Black with at least 500 ratings, and I do NOT want any bulky desk mount or floor mount designs — just the standard wall unit.",
        "implicit_rubric_ids": ["r1"],
        "note": "Negative rubric: 'NOT desk mount or floor mount' excludes two specific Mounting Type values while the target (Wall Mount) still satisfies. Implicit: 'round plug that slides over a center pin' implies Barrel Connector (connector type via physical description). Entity plan: query_surface='power adapter', canonical_entity='power adapter', title_span='Power Cord'.",
        "fewshot_role": "intent_specific",
        "intent_ids": ["negative_constraint"],
    },
    {
        "id": "bad_transparent",
        "title": "BAD EXAMPLE — too transparent (do NOT imitate)",
        "query": "I need a case that wraps around the edges for my phone, and it should plug directly into my device.",
        "implicit_rubric_ids": ["r1", "r2"],
        "note": "BAD: 'wraps around the edges' is just a paraphrase of Bumper; 'plugs directly into my device' is just a paraphrase of Wired. A good implicit clue requires an inference step, not synonym substitution.",
        "fewshot_role": "implicit_bad",
        "intent_ids": [],
    },
    {
        "id": "review_driven",
        "title": "Review-driven: review opinions plus hidden attribute clues",
        "query": "I'm looking for a hair spray with reviews that mention a lovely scent that isn't overpowering and say it adds great texture. I need it for hair that gets frizzy in humidity, and it should come out as a fine mist instead of a heavy stream.",
        "implicit_rubric_ids": ["r3", "r4"],
        "note": "Review-opinion rubrics (r1: lovely scent, r2: great texture) are framed as review evidence the user wants to find, not as reviews they already read. Implicit attribute rubrics: frizzy in humidity implies Wavy/Curly (hair type); fine mist instead of a heavy stream implies Spray (item form). Each implicit clue requires one reasoning hop.",
        "fewshot_role": "intent_specific",
        "intent_ids": ["review_driven"],
    },
]

QUERY_SYSTEM_PROMPT = """You are an expert at creating realistic e-commerce shopping queries.
Given target product evidence and a LOCKED rubric plan, generate a natural user query and structured output.

Core rules:
- Include ALL locked constraints in the query, but add nothing that is not in the locked plan.
- Naturalize wording so the query sounds conversational, but never change the underlying meaning, weaken a requirement, or broaden it in a way that could exclude the target product.
- Every constraint is mandatory. Do not use softeners such as "prefer", "ideally", "if possible", or "would be nice".
- Keep weak metadata phrases close to the locked wording instead of paraphrasing them into broader subjective meaning.
- Follow the preferred opening style.

Implicit-rubric rules:
- If a rubric is selected as implicit, hide the literal expected value and express it through a clue that requires at least one reasoning hop.
- The clue must map uniquely to the hidden value, not to a broader class or multiple possible values.
- The clue must go beyond a synonym or near-paraphrase of the attribute name.
- Reliable clue styles include world-knowledge facts, problem-implies-solution, scenario implication, and distinctive physical-property descriptions.
- Reject weak clues such as "good quality material" for Scratch Resistant or "lightweight material" for Plastic.

Numeric rules:
- Keep numeric constraints explicit and hard.
- Never use vague wording such as "around", "about", "roughly", or "approximately".
- Use round star thresholds, explicit rating-count thresholds, exact price bounds, and explicit after-discount budget wording when vouchers apply.
"""

INTENT_SPECIFIC_RULES = {
    "product_search": "Write a direct product request weaving all attributes into natural prose. Avoid listing attributes as bullet points or comma-separated checklist.",
    "knowledge_reasoning": "Keep the query somewhat indirect and natural, but do NOT invent a new semantic target that is not in the locked plan. Make implicit constraints feel like coherent shopper clues, not a checklist.",
    "use_case_scenario": "Lead with a real-life situation or problem. Attributes should be IMPLIED by the scenario. Make implicit constraints part of one coherent scenario, not disconnected clues.",
    "feature_combination": "Weave 4+ constraints into a coherent narrative with a clear use-case thread. Avoid enumerating features one by one; interleave explicit and implicit constraints naturally.",
    "rating_quality": "Naturally embed rating/review-count thresholds into the request context (e.g., 'well-reviewed' then the specific threshold). Do not lead with the numeric requirement.",
    "negative_constraint": "Express at least one feature as a negative constraint on a DIFFERENT value in the same dimension. The target product's actual value should still satisfy the query. State the exclusion clearly with 'NOT', 'no', or 'avoid'.",
    "coupon_budget": (
        "Naturally state the voucher details (type, threshold, discount amount) and the final budget cap. "
        "For tiered coupons, state BOTH tier thresholds and their discounts, then note which tier the order qualifies for. "
        "For stacked coupons, mention both coupons and specify the sequential application order "
        "(coupon 1 applies to the full price, coupon 2 applies to the price after coupon 1). "
        "Present them as part of the shopping context, not as a math problem. "
        "Never recompute or modify the discount figures — use the exact numbers given."
    ),
    "review_driven": "The query should ask for specific review-based evidence the user wants the product to have, e.g., 'I want something with reviews that mention it is great for short hair'. Do NOT write as if the user already read those reviews, and do NOT convert review-opinion rubrics into product-specification language.",
}

OPENING_STYLES = {
    "review_driven": [
        ("review-targeted", "Start by stating the kind of positive review evidence you want the product's reviews to contain, not by saying you already read those reviews."),
        ("direct-request", "Use a direct request such as 'I want something with reviews that mention...' or 'Find me a product people say...'. Do not narrate that the user has already read the reviews."),
    ],
    "use_case_scenario": [
        ("problem-first", "Start with a real-life problem or symptom before mentioning the product."),
        ("context-first", "Start with a concrete usage situation or daily routine, then ask for the product."),
    ],
    "knowledge_reasoning": [
        ("problem-first", "Start from the user's need/problem and describe clues indirectly before naming the product."),
        ("context-first", "Start with a use case or shopping context, not with a recommendation formula."),
        ("direct-request", "Use a direct request such as 'Find me' or 'I need', but do NOT start with 'Can you recommend' or 'Can you help me'."),
    ],
    "_default": [
        ("problem-first", "Start with a real-life problem or symptom."),
        ("context-first", "Start with a realistic shopping or usage context."),
        ("direct-request", "Use a direct request such as 'Find me', 'Show me', or 'I need'."),
        ("question-first", "Use a question form such as 'What's a good...' or 'Any suggestions for...'."),
    ],
}

USER_PROMPT_SECTIONS = {
    "fewshot_header": "Reference examples (learn the style, do NOT copy content):",
    "fewshot_item": (
        "Example {index} ({title}):\n"
        "- query: {query}\n"
        "- implicit_rubric_ids: {implicit_ids}\n"
        "- note: {note}"
    ),
    "fewshot_anti_reuse": (
        "Few-shot anti-reuse: learn the level of indirection from the examples, but do NOT reuse or lightly rewrite "
        "their original implicit clues. Create a fresh clue instead of recycling example phrases like "
        "'inline skate wheels' or 'face-down'."
    ),
    "plan_header": "Locked rubric plan ({count} rubrics total — your rubric_surfaces must have exactly {count} entries, every item MUST appear in query):",
    "plan_item": (
        "- [{rubric_id}] ({type}, {field}, expected={expected_value}, "
        "implicit_eligible={implicit_eligible}, must_hide_value={must_hide_value}): "
        "{query_hint}"
    ),
    "review_driven_framing": (
        "Review-driven framing: treat every review_opinion rubric as a review trait the user wants to see in the product's reviews. "
        "The user is searching for a product with reviews that mention those traits."
    ),
    "intent_profile_header": "Intent profile:",
    "intent_profile_item": "- {label}: {value}",
    "voucher_header": "Voucher information to include:",
    "implicit_instructions": (
        "Eligible implicit rubric ids: {eligible_ids}\n"
        "You MUST select at least {min_count} from this pool as implicit_rubric_ids.\n"
        "For every selected implicit rubric, do NOT use its must_hide_value literally in the query."
    ),
}

QUERY_OUTPUT_CONTRACT = """Respond in JSON:
{
  "user_query": "...",
  "implicit_rubric_ids": ["r2"],
  "rubric_surfaces": [{"rubric_id": "r1", "query_surface": "..."}],
  "negative_constraints": [{"rubric_id": "r4", "query_surface": "not blue or green", "excluded_values": ["blue", "green"]}],
  "entity_plans": [{"group_index": 0, "query_surface": "phone case", "canonical_entity": "phone case", "title_span": "Phone Case"}]
}

Output rules:
- user_query must be 1-3 English sentences with no product IDs or ASINs.
- rubric_surfaces must contain exactly one entry for every scaffold rubric_id, and each query_surface must be copied VERBATIM from user_query.
- entity_plans must contain exactly one entry.
- entity_plans.query_surface must be copied VERBATIM from user_query.
- entity_plans.canonical_entity may use ONLY words from query_surface.
- entity_plans.title_span must be an EXACT contiguous substring of the target product title.
- For negative_attribute rubrics, negative_constraints.excluded_values must match the negative phrase in the query.
"""

PERSONA_NOISE_SYSTEM_PROMPT = """You are generating realistic background information for a simulated e-commerce user profile.

You will receive:
1. A product category that the user is shopping in.
2. Active product requirements that are already filled (do NOT modify these).

Your task: Generate ONLY the noise/background fields for the persona. These fields must be
completely unrelated to product selection. They exist only to make the profile look realistic.

You can ONLY fill these fields when it's completely unrelated to the product:
- demographics: gender, age_range, location, occupation, education_level
- lifestyle: hobbies (list of 2-3), daily_routine (one sentence), fitness_level, dietary_preference, commute_method, pets
- shopping_habits: payment_method, shopping_frequency, preferred_device, preferred_platform, return_frequency

Rules:
- Make the demographic and lifestyle plausible for someone shopping in the given category.
- NEVER include any product-related preferences (brands, prices, materials, colors, styles, features).
- NEVER include anything that could be interpreted as a product filtering criterion.
- Keep values simple and realistic.

Respond in JSON with exactly these fields:
{
  "demographics": {"gender": "...", "age_range": "...", "location": "...", "occupation": "...", "education_level": "..."},
  "lifestyle": {"hobbies": [...], "daily_routine": "...", "fitness_level": "...", "dietary_preference": "...", "commute_method": "...", "pets": "..."},
  "shopping_habits": {"payment_method": "...", "shopping_frequency": "...", "preferred_device": "...", "preferred_platform": "...", "return_frequency": "..."}
}"""


CLARIFICATION_SYSTEM_PROMPT = """You are generating a clarification script for a simulated e-commerce shopper.

For each hidden product requirement, you need to create:
1. trigger_keywords: A broad set of topic-related keywords. These keywords are matched via
   exact text overlap (not semantic similarity), so use single words or short phrases (1-3 words)
   rather than long descriptive phrases. Include synonyms, related concepts, and category-level
   terms — about 7 keywords covering the topic broadly. IMPORTANT: always include the attribute
   dimension/category name itself (e.g., "color" if the hidden value is "Red", "material" if the
   hidden value is "TPU", "connectivity" if the hidden value is "Bluetooth"). This ensures the
   slot triggers whenever the agent asks about the general attribute category, not just the
   specific value.
2. user_response: A natural, conversational response revealing the requirement. Should sound like
   a real shopper answering a question. Include the key information but keep it natural.
3. hidden_info: A brief description of what information is hidden.

Rules:
- trigger_keywords should be BROAD. If the hidden info is about SPF 50, include keywords like:
  "SPF", "sun protection", "sun", "UV", "protection", "sunscreen", "sunburn", "outdoor"
  Note: "sun protection" is the attribute dimension — it MUST be included.
- user_response should be 1-2 sentences, natural and conversational.
- The response must contain the exact requirement value needed for the rubric.

Respond in JSON:
{
  "slots": [
    {
      "hidden_info": "...",
      "trigger_keywords": ["kw1", "kw2", ...],
      "user_response": "..."
    }
  ]
}"""

__all__ = [
    "FEWSHOT_EXAMPLES",
    "INTENT_SPECIFIC_RULES",
    "OPENING_STYLES",
    "QUERY_OUTPUT_CONTRACT",
    "QUERY_SYSTEM_PROMPT",
    "USER_PROMPT_SECTIONS",
    "PERSONA_NOISE_SYSTEM_PROMPT",
    "CLARIFICATION_SYSTEM_PROMPT",
]
