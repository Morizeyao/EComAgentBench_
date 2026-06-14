"""Rubric 构建与工具函数。"""

from src.generation.config import (
    IMPLICIT_ELIGIBLE_FIELDS,
    render_attribute_query_hint,
    should_add_budget_rubric,
)
from src.generation.utils import normalize_numeric_expected


def build_rubrics(
    partitioned_features: dict[str, list[dict]],
    intent_id: str,
    product: dict | None = None,
    voucher: dict | None = None,
) -> list[dict]:
    """根据分区后的特征构建 rubric 列表，每条 rubric 带 info_source。

    partitioned_features: {"query": [...], "persona": [...], "clarification": [...]}
    """
    if not product:
        return []
    pid = product["product_id"]

    rubric_counter = 0
    all_rubrics: list[dict] = []

    for source, features in partitioned_features.items():
        pid_features = [f for f in features if f.get("product_id") == pid]

        for f in pid_features:
            rubric_counter += 1

            if f.get("negative"):
                field_name = f["field"].split(".")[-1]
                actual_value = f["value"]
                all_rubrics.append({
                    "id": f"r{rubric_counter}",
                    "description": (
                        f"Exclude concrete alternative values on the {field_name} dimension "
                        f"while keeping the target product value '{actual_value}' valid"
                    ),
                    "type": "negative_attribute",
                    "field": f["field"],
                    "expected_value": [],
                    "info_source": source,
                    "implicit_eligible": False,
                    "must_hide_value": None,
                    "query_hint": (
                        f"Express a negative constraint on the {field_name} dimension by excluding other concrete values. "
                        f"Do not exclude the product's actual value '{actual_value}'."
                    ),
                })
            else:
                expected_value = (
                    normalize_numeric_expected(f["rubric"], f["value"])
                    if f["type"] == "numeric_range" else f["value"]
                )
                is_implicit = (
                    f["type"] == "attribute_match"
                    and str(f["field"]) in IMPLICIT_ELIGIBLE_FIELDS
                    and source == "query"
                )
                all_rubrics.append({
                    "id": f"r{rubric_counter}",
                    "description": f["rubric"],
                    "type": f["type"],
                    "field": f["field"],
                    "expected_value": expected_value,
                    "info_source": source,
                    "implicit_eligible": is_implicit,
                    "must_hide_value": str(expected_value).strip() if is_implicit else None,
                    "query_hint": _build_query_hint(
                        f["type"], f["field"], expected_value, voucher,
                    ),
                })

    if should_add_budget_rubric(intent_id) and voucher:
        rubric_counter += 1
        all_rubrics.append({
            "id": f"r{rubric_counter}",
            "description": f"The total cost after applying the voucher should be within the budget of ${voucher['budget']}",
            "type": "budget_match",
            "field": "budget",
            "expected_value": voucher["budget"],
            "info_source": "query",
            "implicit_eligible": False,
            "must_hide_value": None,
            "query_hint": _render_budget_hint(voucher, voucher["budget"]),
        })

    return all_rubrics



# ── Rubric 分组/合并工具 ──

def filter_rubrics_by_source(
    all_rubrics: list[dict], source: str,
) -> list[dict]:
    """过滤出指定 info_source 的 rubrics。"""
    return [r for r in all_rubrics if r.get("info_source") == source]


def merge_rubrics(
    all_rubrics: list[dict],
    compiled_query_rubrics: list[dict],
) -> list[dict]:
    """将 compiled query rubrics 合并回 all_rubrics（替换 query-source rubrics）。"""
    compiled_index = {r["id"]: r for r in compiled_query_rubrics}

    merged = []
    for r in all_rubrics:
        if r["id"] in compiled_index:
            compiled_r = dict(compiled_index[r["id"]])
            compiled_r["info_source"] = "query"
            merged.append(compiled_r)
        else:
            merged.append(r)

    existing_ids = {r["id"] for r in merged}
    for cr in compiled_query_rubrics:
        if cr["id"] not in existing_ids:
            cr_copy = dict(cr)
            cr_copy["info_source"] = "query"
            merged.append(cr_copy)

    return merged


# ── Hint 渲染 ──

def _build_query_hint(rubric_type: str, field: str, expected_value: object,
                      voucher: dict | None) -> str:
    if rubric_type == "entity_match":
        return f"Mention the exact product entity '{expected_value}' in the query."
    if rubric_type == "store_match" or field == "store":
        return f"State explicitly that the item must come from the '{expected_value}' store or seller."
    if rubric_type == "numeric_range":
        return _render_numeric_hint(field, expected_value)
    if rubric_type == "budget_match":
        return _render_budget_hint(voucher, expected_value)
    if rubric_type == "attribute_match":
        return render_attribute_query_hint(field, expected_value)
    if rubric_type == "review_opinion":
        opinion = expected_value.get("opinion", "") if isinstance(expected_value, dict) else str(expected_value)
        return (
            f"Mention that you want a product with reviews mentioning '{opinion}'. "
            f"Frame it as review evidence you want to find, not as reviews the user has already read."
        )
    return f"Mention the exact requirement '{expected_value}'."


def _fmt_price(value: float) -> str:
    return f"${value:.2f}".rstrip("0").rstrip(".")


def _render_numeric_hint(field: str, expected_value: object) -> str:
    if field == "average_rating":
        formatter = lambda v: f"{v:g} stars"
    elif field == "rating_number":
        formatter = lambda v: f"{int(v)} ratings" if float(v).is_integer() else f"{v:g} ratings"
    elif field == "price":
        formatter = _fmt_price
    else:
        formatter = lambda v: f"{v:g}"

    if isinstance(expected_value, dict):
        if "min" in expected_value and "max" in expected_value:
            return f"Mention an exact range between {formatter(expected_value['min'])} and {formatter(expected_value['max'])}."
        if "min" in expected_value:
            return f"Mention that it must be at least {formatter(expected_value['min'])}."
        if "max" in expected_value:
            return f"Mention that it must be at most {formatter(expected_value['max'])}."
        if "value" in expected_value:
            op = expected_value.get("op", "==")
            phrase = {">": "over", ">=": "at least", "<": "under", "<=": "at most", "==": "exactly"}.get(op, "exactly")
            return f"Mention that it must be {phrase} {formatter(expected_value['value'])}."
    if isinstance(expected_value, (int, float)):
        return f"Mention the exact threshold {formatter(expected_value)}."
    return f"Mention the exact numeric requirement '{expected_value}'."


def _render_budget_hint(voucher: dict | None, expected_value: object) -> str:
    budget = expected_value if isinstance(expected_value, (int, float)) else None
    budget_text = _fmt_price(budget) if budget is not None else str(expected_value)
    if not voucher:
        return f"Mention that the final total must stay under {budget_text}."

    dtype = voucher.get("discount_type")
    if dtype == "tiered":
        return _render_tiered_hint(voucher, budget_text)
    if dtype == "stacked":
        return _render_stacked_hint(voucher, budget_text)

    # 原有简单券逻辑
    threshold = voucher.get("threshold")
    threshold_text = _fmt_price(threshold) if isinstance(threshold, (int, float)) else None
    if isinstance(voucher.get("face_value"), (int, float)):
        face_value = _fmt_price(voucher["face_value"])
        return (
            f"Mention the fixed {voucher.get('voucher_type', 'coupon')} coupon worth {face_value} "
            f"{'on orders over ' + threshold_text if threshold_text else ''} and say the final total must stay under {budget_text}."
        ).strip()
    if isinstance(voucher.get("discount"), (int, float)):
        discount_pct = int(round(voucher["discount"] * 100))
        cap = voucher.get("cap")
        cap_text = f", capped at {_fmt_price(cap)}" if isinstance(cap, (int, float)) else ""
        return (
            f"Mention the {discount_pct}% {voucher.get('voucher_type', 'coupon')} coupon "
            f"{'on orders over ' + threshold_text if threshold_text else ''}{cap_text}, "
            f"and say the final total must stay under {budget_text}."
        )
    return f"Mention that the final total after the coupon must stay under {budget_text}."


def _render_tiered_hint(voucher: dict, budget_text: str) -> str:
    """阶梯折扣券的 hint：写出两档，标明 tier1 生效。"""
    vtype = voucher.get("voucher_type", "coupon")
    tiers = voucher.get("tiers", [])
    if len(tiers) < 2:
        return f"Mention that the final total after the tiered coupon must stay under {budget_text}."

    t1, t2 = tiers[0], tiers[1]

    def _tier_desc(t: dict) -> str:
        if t.get("discount_type") == "fixed":
            return f"{_fmt_price(t['face_value'])} off"
        pct = int(round(t["discount"] * 100))
        cap_text = f", capped at {_fmt_price(t['cap'])}" if isinstance(t.get("cap"), (int, float)) else ""
        return f"{pct}% off{cap_text}"

    t1_desc = _tier_desc(t1)
    t2_desc = _tier_desc(t2)
    total_price_text = _fmt_price(voucher["total_original_price"])

    return (
        f"Mention the tiered {vtype} coupon with two spending thresholds: "
        f"spend over {_fmt_price(t1['threshold'])} to get {t1_desc}, "
        f"or spend over {_fmt_price(t2['threshold'])} to get {t2_desc}. "
        f"The order total is {total_price_text}, so tier 1 ({_fmt_price(t1['threshold'])} threshold) applies. "
        f"State that the final total must stay under {budget_text}."
    )


def _render_stacked_hint(voucher: dict, budget_text: str) -> str:
    """两券叠加的 hint：写出两张券及顺序叠加逻辑。"""
    coupons = voucher.get("coupons", [])
    if len(coupons) < 2:
        return f"Mention that the final total after stacked coupons must stay under {budget_text}."

    c0, c1 = coupons[0], coupons[1]
    c0_pct = int(round(c0["discount"] * 100))
    c0_cap_text = f", capped at {_fmt_price(c0['cap'])}" if isinstance(c0.get("cap"), (int, float)) else ""
    total_price_text = _fmt_price(voucher["total_original_price"])

    return (
        f"Mention two stacked coupons that both apply: "
        f"(1) a {c0_pct}% {c0['voucher_type']} coupon on orders over {_fmt_price(c0['threshold'])}"
        f"{c0_cap_text}, applied first to the full price {total_price_text}; "
        f"(2) a {c1['voucher_type']} coupon of {_fmt_price(c1['face_value'])} off on orders over "
        f"{_fmt_price(c1['threshold'])}, applied to the price after coupon 1. "
        f"State that the final total after both coupons must stay under {budget_text}."
    )
