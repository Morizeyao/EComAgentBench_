"""Prompt 构造：组装 generation 阶段的受控查询生成 prompt。"""

import random

from src.prompts.generation import (
    FEWSHOT_EXAMPLES,
    INTENT_SPECIFIC_RULES,
    OPENING_STYLES,
    QUERY_OUTPUT_CONTRACT,
    USER_PROMPT_SECTIONS,
)

S = USER_PROMPT_SECTIONS


def choose_opening_style(intent_id: str, rng: random.Random) -> tuple[str, str]:
    """根据意图类型随机选择一种 query 开头风格，增加多样性。"""
    choices = OPENING_STYLES.get(intent_id, OPENING_STYLES["_default"])
    return rng.choice(choices)


def select_fewshot_examples(intent_id: str) -> list[dict]:
    selected = []
    for ex in FEWSHOT_EXAMPLES:
        role = ex.get("fewshot_role")
        if role in {"implicit_core", "implicit_bad"}:
            selected.append(ex)
            continue
        if intent_id in ex.get("intent_ids", []):
            selected.append(ex)
    return selected


def build_query_prompt(intent: dict, product: dict,
                       rubrics: list[dict],
                       voucher: dict | None = None,
                       opening_style: tuple[str, str] | None = None,
                       min_implicit_count: int = 0) -> str:
    """组装完整的 user prompt，传给 LLM 生成 query 与 query-surface 映射。"""
    lines = []
    fewshot_examples = select_fewshot_examples(intent["id"])

    # --- few-shot ---
    lines.append(S["fewshot_header"])
    for i, ex in enumerate(fewshot_examples, 1):
        lines.append(S["fewshot_item"].format(
            index=i, title=ex["title"], query=ex["query"],
            implicit_ids=ex["implicit_rubric_ids"], note=ex["note"],
        ))
    lines.append(S["fewshot_anti_reuse"])
    lines.append("")

    # --- intent / style ---
    lines.append(S["intent_profile_header"])
    lines.append(S["intent_profile_item"].format(label="Category", value=intent["name"]))
    lines.append(S["intent_profile_item"].format(label="Description", value=intent["description"]))
    intent_rule = INTENT_SPECIFIC_RULES.get(intent["id"])
    if intent_rule:
        lines.append(S["intent_profile_item"].format(label="Writing rule", value=intent_rule))
    if intent["id"] == "review_driven":
        lines.append(S["review_driven_framing"])
    if opening_style:
        lines.append(S["intent_profile_item"].format(
            label="Preferred opening style",
            value=f"{opening_style[0]} — {opening_style[1]}",
        ))
    lines.append("")

    # --- target product ---
    lines.append("Target product:")
    lines.append(f"- Title: {product.get('title', '')}")
    lines.append("")

    # --- locked rubric plan ---
    lines.append(S["plan_header"].format(count=len(rubrics)))
    eligible_ids = []
    for rubric in rubrics:
        lines.append(S["plan_item"].format(
            rubric_id=rubric["id"], type=rubric["type"],
            field=rubric["field"], expected_value=rubric["expected_value"],
            implicit_eligible=bool(rubric.get("implicit_eligible")),
            must_hide_value=rubric.get("must_hide_value"),
            query_hint=rubric.get("query_hint", ""),
        ))
        if rubric.get("implicit_eligible"):
            eligible_ids.append(rubric["id"])
    lines.append("")

    # --- voucher ---
    if voucher:
        lines.extend(_render_voucher_prompt_section(voucher))
        lines.append("")

    # --- implicit requirement ---
    lines.append(S["implicit_instructions"].format(
        eligible_ids=eligible_ids, min_count=min_implicit_count,
    ))
    lines.append("")

    # --- output contract ---
    lines.append(QUERY_OUTPUT_CONTRACT)

    return "\n".join(lines)


def _render_voucher_prompt_section(voucher: dict) -> list[str]:
    """渲染 voucher 信息块，支持 simple / tiered / stacked 三种类型。"""
    dtype = voucher.get("discount_type")
    lines = [S["voucher_header"]]

    if dtype == "tiered":
        lines.append(f"- Type: {voucher['voucher_type']} (tiered discount)")
        lines.append(f"- Total original price: ${voucher['total_original_price']}")
        for t in voucher.get("tiers", []):
            tier_num = t["tier"]
            thr = t["threshold"]
            if t.get("discount_type") == "fixed":
                disc_str = f"${t['face_value']} off"
            else:
                pct = int(round(t["discount"] * 100))
                cap_str = f" (cap ${t['cap']})" if t.get("cap") else ""
                disc_str = f"{pct}% off{cap_str}"
            lines.append(f"  - Tier {tier_num}: spend over ${thr} → {disc_str}")
        lines.append(
            f"- Active tier: 1 (price ${voucher['total_original_price']} "
            f"> tier 1 threshold ${voucher['tiers'][0]['threshold']})"
        )
        lines.append(f"- Price after voucher: ${voucher['price_after_voucher']}")

    elif dtype == "stacked":
        lines.append("- Type: stacked (two coupons applied sequentially)")
        lines.append(f"- Total original price: ${voucher['total_original_price']}")
        for i, c in enumerate(voucher.get("coupons", []), 1):
            if c.get("discount_type") == "percentage":
                pct = int(round(c["discount"] * 100))
                lines.append(
                    f"  - Coupon {i} ({c['voucher_type']}): {pct}% off on orders over "
                    f"${c['threshold']} (cap ${c['cap']}) → saves ${c['actual_discount']}"
                )
            else:
                lines.append(
                    f"  - Coupon {i} ({c['voucher_type']}): ${c['face_value']} off on orders over "
                    f"${c['threshold']} → saves ${c['actual_discount']}"
                )
        lines.append(f"- Price after both coupons: ${voucher['price_after_voucher']}")

    else:
        lines.append(f"- Type: {voucher['voucher_type']}, Threshold: ${voucher['threshold']}")
        if voucher.get("face_value"):
            lines.append(f"- Fixed discount: ${voucher['face_value']}")
        if voucher.get("discount"):
            pct = f"{voucher['discount'] * 100}%"
            cap = f", cap ${voucher['cap']}" if voucher.get("cap") else ""
            lines.append(f"- Percentage discount: {pct}{cap}")

    lines.append(f"- Budget: ${voucher['budget']}")
    return lines
