"""Generation 配置中心。

含属性配置与 persona/clarification 分区配置。
特征采样概率保持不变（store=0, brand=0.2 等）。
"""

from src.generation.utils import normalize_text

# ── 字段定义 ──

DETAIL_FIELDS = [
    # 跨品类通用
    "Brand", "Color", "Material", "Style", "Size",
    "Special Feature", "Number of Items", "Age Range (Description)",
    # Beauty
    "Item Form", "Hair Type", "Skin Type", "Scent",
    "Finish Type", "Product Benefits", "Material Feature",
    # CellPhones / Electronics
    "Form Factor", "Connectivity Technology", "Connector Type",
    "Compatible Phone Models", "Compatible Devices",
    "Screen Size", "Pattern", "Mounting Type", "Shell Type",
    # Computers
    "Operating System", "RAM", "Hard Drive", "Processor",
    "Graphics Coprocessor", "Wireless Type", "Computer Memory Type",
    "Memory Storage Capacity", "Hardware Platform",
    # Camera
    "Lens Type", "Video Capture Resolution", "Water Resistance Level",
    # Cross-electronics
    "Power Source", "Recommended Uses For Product",
    # Office
    "Ink Color", "Shape", "Sheet Size", "Closure", "Point Type",
]

FIELD_NAME_ALIASES: dict[str, str] = {
    "Special features": "Special Feature",
    "Material Type": "Material",
    "Standing screen display size": "Screen Size",
    "Connectivity technologies": "Connectivity Technology",
    "Number Of Items": "Number of Items",
    "OS": "Operating System",
    "Ram Memory Installed Size": "RAM",
    "Wireless Communication Technology": "Wireless Type",
    "Processor Brand": "Processor",
}

_REVERSE_ALIASES: dict[str, list[str]] = {}
for _alias, _canonical in FIELD_NAME_ALIASES.items():
    _REVERSE_ALIASES.setdefault(_canonical, []).append(_alias)

# ── 采样概率 ──

TOP_LEVEL_FIELD_SAMPLE_PROB = {
    "store": 0.0,
    "average_rating": 0.3,
    "rating_number": 0.2,
    "price": 0.4,
}

DETAIL_FIELD_WEIGHT = {
    "Brand": {
        "knowledge_reasoning": 0.2,
        "use_case_scenario": 0.2,
        "coupon_budget": 0.2,
        "_default": 0.2,
    },
}

STORE_WEIGHT_BY_INTENT = {
    "knowledge_reasoning": 0,
    "use_case_scenario": 0,
    "coupon_budget": 0,
    "feature_combination": 0,
    "_default": 0,
}

# ── 字段分类集合 ──

NEGATIVE_ATTRIBUTE_FIELDS = {
    "details.Brand", "details.Color", "details.Material", "details.Style",
    # Beauty
    "details.Item Form", "details.Hair Type", "details.Skin Type",
    "details.Scent", "details.Finish Type",
    "details.Product Benefits", "details.Material Feature",
    # CellPhones / Electronics
    "details.Form Factor", "details.Connectivity Technology",
    "details.Connector Type", "details.Pattern", "details.Mounting Type",
    "details.Shell Type",
    # Computers
    "details.Operating System", "details.Wireless Type",
    # Cross-electronics
    "details.Power Source",
    # Office
    "details.Ink Color", "details.Shape", "details.Closure", "details.Point Type",
}

MULTI_OPTION_FIELDS = {
    "Material", "Color", "Style",
    # Beauty
    "Item Form", "Hair Type", "Skin Type", "Scent",
    "Finish Type", "Product Benefits", "Material Feature",
    # CellPhones / Electronics
    "Form Factor", "Connectivity Technology", "Connector Type",
    "Compatible Devices", "Compatible Phone Models",
    "Pattern", "Ink Color",
    # Computers / Cross-electronics
    "Recommended Uses For Product", "Power Source",
    "Wireless Type", "Computer Memory Type",
}

WEAK_METADATA_FIELDS = {
    "details.Material Feature", "details.Style",
    "details.Finish Type", "details.Special Feature",
    "details.Product Benefits",
    "details.Pattern", "details.Mounting Type",
    "details.Recommended Uses For Product", "details.Hardware Platform",
}

UNSUPPORTED_ATTRIBUTE_FIELDS = {
    "details.Item Weight", "details.Item Volume",
    "details.Package Dimensions", "details.Product Dimensions",
    "details.Date First Available", "details.Manufacturer",
    "details.Item model number", "details.Best Sellers Rank",
    "details.Manufacturer Part Number",
}

IMPLICIT_ELIGIBLE_FIELDS = {
    # 跨品类
    "details.Material", "details.Special Feature", "details.Style",
    # Beauty
    "details.Skin Type", "details.Hair Type", "details.Item Form",
    "details.Product Benefits", "details.Finish Type",
    "details.Material Feature", "details.Scent",
    # CellPhones / Electronics
    "details.Form Factor", "details.Connectivity Technology",
    "details.Connector Type", "details.Pattern", "details.Mounting Type",
    # Computers
    "details.Operating System", "details.RAM", "details.Hard Drive",
    "details.Processor", "details.Wireless Type",
    # Camera
    "details.Lens Type",
    # Cross-electronics
    "details.Power Source", "details.Recommended Uses For Product",
    # Office
    "details.Ink Color",
}

# ── 数值分桶 ──

RATING_NUMBER_BUCKETS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
PRICE_BUCKETS = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 1000]

UNDERUSED_FEATURE_POOL = [
    "Brand", "Color", "Material", "Style", "Size",
    # Beauty
    "Item Form", "Hair Type", "Skin Type", "Scent", "Finish Type",
    # CellPhones / Electronics
    "Form Factor", "Connectivity Technology", "Connector Type",
    "Compatible Devices", "Pattern", "Mounting Type", "Shell Type",
    # Computers
    "Operating System", "RAM", "Hard Drive", "Processor",
    "Graphics Coprocessor", "Wireless Type", "Computer Memory Type",
    "Memory Storage Capacity", "Hardware Platform",
    # Camera
    "Lens Type", "Video Capture Resolution", "Water Resistance Level",
    # Cross-electronics
    "Power Source", "Recommended Uses For Product",
    # Office
    "Ink Color", "Shape", "Closure", "Point Type",
    # top-level
    "average_rating", "rating_number", "price", "store",
]

# ── Intent 规则 ──

MIN_IMPLICIT_COUNT_BY_INTENT = {
    "knowledge_reasoning": 3,
    "use_case_scenario": 3,
    "product_search": 2,
    "feature_combination": 3,
    "rating_quality": 2,
    "coupon_budget": 2,
    "comparison_selection": 2,
    "negative_constraint": 2,
    "review_driven": 2,
}

INTENT_GENERATION_RULES = {
    "_default": {
        "sampling_mode": "single_product",
        "requires_voucher": False,
        "ensure_store_rubric": False,
        "ensure_budget_rubric": False,
        "requires_content_evidence": False,
        "requires_price": False,
    },
    "knowledge_reasoning": {
        "requires_content_evidence": True,
    },
    "use_case_scenario": {
        "requires_content_evidence": True,
    },
    "coupon_budget": {
        "requires_voucher": True,
        "ensure_budget_rubric": True,
        "requires_price": True,
    },
    "review_driven": {
        "requires_review_opinions": True,
    },
}

# ── Hint 模板 ──

ATTRIBUTE_HINT_TEMPLATES = {
    "details.Brand": "Mention the exact brand name '{value}' as a hard requirement.",
    "details.Material": "Mention that it must be made of or made from '{value}'.",
    "details.Color": "Mention the exact color requirement '{value}'.",
    "details.Hair Type": "Mention the exact hair-type or texture requirement '{value}'.",
    "details.Skin Type": "Mention the exact skin-type requirement '{value}'.",
    "details.Item Form": "Mention the exact item form '{value}'. Do not reinterpret it as a different format.",
    "details.Age Range (Description)": "Mention the exact age-range requirement '{value}'.",
    # CellPhones / Electronics
    "details.Form Factor": "Mention the exact form factor '{value}'.",
    "details.Connectivity Technology": "Mention the connectivity type '{value}'.",
    "details.Connector Type": "Mention the exact connector type '{value}'.",
    "details.Compatible Phone Models": "Mention compatibility with '{value}'.",
    "details.Compatible Devices": "Mention that it must work with '{value}'.",
    "details.Screen Size": "Mention the exact screen size '{value}'.",
    "details.Shell Type": "Mention the case/shell type '{value}'.",
    # Computers
    "details.Operating System": "Mention the operating system requirement '{value}'.",
    "details.RAM": "Mention the RAM requirement '{value}'.",
    "details.Hard Drive": "Mention the storage/hard drive requirement '{value}'.",
    "details.Processor": "Mention the processor requirement '{value}'.",
    "details.Graphics Coprocessor": "Mention the graphics card requirement '{value}'.",
    "details.Wireless Type": "Mention the wireless connectivity type '{value}'.",
    "details.Computer Memory Type": "Mention the memory type '{value}'.",
    "details.Memory Storage Capacity": "Mention the storage capacity '{value}'.",
    # Camera
    "details.Lens Type": "Mention the lens type '{value}'.",
    "details.Video Capture Resolution": "Mention the video resolution '{value}'.",
    "details.Water Resistance Level": "Mention the water resistance level '{value}'.",
    # Cross-electronics
    "details.Power Source": "Mention the power source '{value}'.",
    # Office
    "details.Ink Color": "Mention the ink color '{value}'.",
    "details.Shape": "Mention the shape '{value}'.",
    "details.Sheet Size": "Mention the sheet size '{value}'.",
    "details.Closure": "Mention the closure type '{value}'.",
    "details.Point Type": "Mention the point type '{value}'.",
    "details.Pattern": "Mention the pattern '{value}'.",
    "details.Mounting Type": "Mention the mounting type '{value}'.",
}

WEAK_FIELD_HINT_TEMPLATE = (
    "Use the exact metadata phrase '{value}' in the query. "
    "Do not paraphrase it into a broader subjective meaning for {field_name}."
)

DEFAULT_ATTRIBUTE_HINT_TEMPLATE = (
    "Mention the exact {field_name} requirement '{value}'."
)

SINGLE_ITEM_VALUES = {"1", "1 count", "1 pack of 1", "1 item"}

# ── Access 函数 ──


def get_min_implicit_count(intent_id: str) -> int:
    return MIN_IMPLICIT_COUNT_BY_INTENT.get(intent_id, 1)


def get_intent_generation_rules(intent_id: str) -> dict:
    rules = dict(INTENT_GENERATION_RULES["_default"])
    rules.update(INTENT_GENERATION_RULES.get(intent_id, {}))
    rules["min_implicit_count"] = get_min_implicit_count(intent_id)
    return rules


def requires_voucher(intent_id: str) -> bool:
    return bool(get_intent_generation_rules(intent_id).get("requires_voucher"))


def should_add_budget_rubric(intent_id: str) -> bool:
    return bool(get_intent_generation_rules(intent_id).get("ensure_budget_rubric"))


def requires_content_evidence(intent_id: str) -> bool:
    return bool(get_intent_generation_rules(intent_id).get("requires_content_evidence"))


def requires_price(intent_id: str) -> bool:
    return bool(get_intent_generation_rules(intent_id).get("requires_price"))


def requires_review_opinions(intent_id: str) -> bool:
    return bool(get_intent_generation_rules(intent_id).get("requires_review_opinions"))


def count_implicit_eligible_rubrics(rubrics: list[dict]) -> int:
    return sum(1 for rubric in rubrics if rubric.get("implicit_eligible"))


def get_field_prob(field: str, intent_id: str) -> float:
    if field != "store":
        return TOP_LEVEL_FIELD_SAMPLE_PROB.get(field, 1.0)
    return STORE_WEIGHT_BY_INTENT.get(intent_id, STORE_WEIGHT_BY_INTENT["_default"])


def get_detail_field_prob(field: str, intent_id: str) -> float:
    weights = DETAIL_FIELD_WEIGHT.get(field)
    if not weights:
        return 1.0
    return weights.get(intent_id, weights.get("_default", 1.0))


def render_attribute_query_hint(field: str, expected_value: object) -> str:
    value = str(expected_value).strip()
    field_name = field.split(".", 1)[1] if field.startswith("details.") else field

    template = ATTRIBUTE_HINT_TEMPLATES.get(field)
    if template:
        return template.format(value=value, field_name=field_name)

    if field == "details.Number of Items":
        return render_number_of_items_hint(value)

    if field in WEAK_METADATA_FIELDS:
        return WEAK_FIELD_HINT_TEMPLATE.format(value=value, field_name=field_name)

    return DEFAULT_ATTRIBUTE_HINT_TEMPLATE.format(value=value, field_name=field_name)


def render_number_of_items_hint(value: str) -> str:
    normalized = normalize_text(value)
    if normalized in SINGLE_ITEM_VALUES:
        return "Say that it must be a single-item listing."
    tokens = normalized.split()
    if tokens and tokens[0].isdigit():
        return f"Mention an exact {tokens[0]}-item requirement."
    return f"Mention the exact package-count requirement '{value}'."


def resolve_detail_value(details: dict, canonical_field: str) -> str | None:
    """从 details dict 中获取值，优先用标准名，fallback 到 alias。"""
    val = details.get(canonical_field)
    if val:
        return val
    for alias in _REVERSE_ALIASES.get(canonical_field, ()):
        val = details.get(alias)
        if val:
            return val
    return None


# ══════════════════════════════════════════════════════
# Persona / Clarification 分区配置
# ══════════════════════════════════════════════════════

SAFE_NOISE_FIELDS = {
    "demographics.gender",
    "demographics.age_range",
    "demographics.location",
    "demographics.occupation",
    "demographics.education_level",
    "lifestyle.hobbies",
    "lifestyle.daily_routine",
    "lifestyle.fitness_level",
    "lifestyle.dietary_preference",
    "lifestyle.commute_method",
    "lifestyle.pets",
    "shopping_habits.payment_method",
    "shopping_habits.shopping_frequency",
    "shopping_habits.preferred_device",
    "shopping_habits.preferred_platform",
    "shopping_habits.return_frequency",
}

PERSONA_FIELD_MAPPING = {
    # 跨品类
    "details.Brand": "brand_preference",
    "details.Color": "color_preference",
    "details.Material": "material_need",
    "details.Style": "style_preference",
    "details.Size": "size_preference",
    # Beauty
    "details.Skin Type": "skin_type",
    "details.Hair Type": "hair_type",
    "details.Item Form": "form_preference",
    "details.Scent": "scent_preference",
    "details.Finish Type": "finish_preference",
    # CellPhones / Electronics
    "details.Form Factor": "form_factor_preference",
    "details.Shell Type": "case_type_preference",
    "details.Compatible Devices": "device_compatibility",
    "details.Compatible Phone Models": "phone_model_compatibility",
    # Computers
    "details.Operating System": "os_preference",
    "details.RAM": "ram_requirement",
    "details.Hard Drive": "storage_preference",
    "details.Processor": "processor_preference",
    "details.Wireless Type": "wireless_preference",
    "details.Memory Storage Capacity": "storage_capacity_preference",
    # Camera
    "details.Lens Type": "lens_preference",
    "details.Water Resistance Level": "water_resistance_need",
    # Cross-electronics
    "details.Power Source": "power_source_preference",
    # Price
    "price": "budget",
}

PERSONA_IMPLICIT_MAPPINGS: dict[str, dict[str, str]] = {}
