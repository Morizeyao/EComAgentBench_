"""模拟优惠券生成（coupon_budget意图）。"""

import random


VOUCHER_TYPES = ["platform", "shop", "brand"]
DISCOUNT_TYPES = ["fixed", "percentage"]

# 三种复杂度场景的采样权重（可调整）
SCENARIO_WEIGHTS = [0.50, 0.25, 0.25]  # simple / tiered / stacked


def generate_voucher(product: dict, rng: random.Random | None = None) -> dict:
    """为给定商品生成一张模拟优惠券和预算约束。

    随机选择三种复杂度场景：
    - simple (50%): 单张固定/百分比券，始终触发
    - tiered (25%): 两档门槛，tier1必触发，tier2必不触发
    - stacked (25%): 两张券顺序叠加，均触发

    返回 dict，失败时返回 {}。
    """
    rng = rng or random.Random()

    total_price = product.get("price", 0) or 0
    if total_price <= 0:
        return {}

    scenario = rng.choices(["simple", "tiered", "stacked"], weights=SCENARIO_WEIGHTS, k=1)[0]

    if scenario == "tiered":
        return _generate_tiered_voucher(total_price, rng)
    if scenario == "stacked":
        return _generate_stacked_voucher(total_price, rng)
    return _generate_simple_voucher(total_price, rng)


# ── 场景生成器 ──

def _generate_simple_voucher(total_price: float, rng: random.Random) -> dict:
    """单张券，始终触发（阈值=40-80% of price），预算弹性0-8%。"""
    voucher_type = rng.choice(VOUCHER_TYPES)
    discount_type = rng.choice(DISCOUNT_TYPES)

    # 阈值：总价的40%-80%（始终低于商品价格，确保触发）
    threshold = round(total_price * rng.uniform(0.4, 0.8), 2)

    if discount_type == "fixed":
        face_value = round(total_price * rng.uniform(0.05, 0.25), 2)
        discount = None
        cap = None
        actual_discount = face_value
    else:
        face_value = None
        discount = round(rng.uniform(0.05, 0.20), 2)
        cap = round(total_price * rng.uniform(0.1, 0.3), 2)
        actual_discount = round(min(total_price * discount, cap), 2)

    price_after = round(total_price - actual_discount, 2)
    budget = round(price_after * rng.uniform(1.0, 1.08), 2)

    return {
        "voucher_type": voucher_type,
        "threshold": threshold,
        "discount_type": discount_type,
        "face_value": face_value,
        "discount": discount,
        "cap": cap,
        "budget": budget,
        "total_original_price": total_price,
        "price_after_voucher": price_after,
    }


def _generate_tiered_voucher(total_price: float, rng: random.Random) -> dict:
    """两档阶梯折扣券。
    tier1 门槛=40-70% of price（必触发），tier2 门槛=110-150% of price（必不触发）。
    """
    voucher_type = rng.choice(VOUCHER_TYPES)

    # Tier 1：必触发
    t1_threshold = round(total_price * rng.uniform(0.40, 0.70), 2)
    t1_dtype = rng.choice(DISCOUNT_TYPES)
    if t1_dtype == "fixed":
        t1_fv = round(total_price * rng.uniform(0.04, 0.10), 2)
        t1_disc = None
        t1_cap = None
        t1_actual = t1_fv
    else:
        t1_disc = round(rng.uniform(0.05, 0.12), 2)
        t1_cap = round(total_price * rng.uniform(0.08, 0.18), 2)
        t1_fv = None
        t1_actual = round(min(total_price * t1_disc, t1_cap), 2)

    # Tier 2：必不触发（门槛高于商品价格）
    t2_threshold = round(total_price * rng.uniform(1.10, 1.50), 2)
    t2_dtype = rng.choice(DISCOUNT_TYPES)
    if t2_dtype == "fixed":
        t2_fv = round(total_price * rng.uniform(0.15, 0.25), 2)
        t2_disc = None
        t2_cap = None
    else:
        t2_disc = round(rng.uniform(0.15, 0.25), 2)
        t2_cap = round(total_price * rng.uniform(0.25, 0.40), 2)
        t2_fv = None

    price_after = round(total_price - t1_actual, 2)
    budget = round(price_after * rng.uniform(1.0, 1.08), 2)

    return {
        "voucher_type": voucher_type,
        "discount_type": "tiered",
        # 向后兼容字段（均指向 tier1 或置 None）
        "threshold": t1_threshold,
        "face_value": None,
        "discount": None,
        "cap": None,
        # 新增字段
        "tiers": [
            {
                "tier": 1,
                "threshold": t1_threshold,
                "discount_type": t1_dtype,
                "face_value": t1_fv,
                "discount": t1_disc,
                "cap": t1_cap,
            },
            {
                "tier": 2,
                "threshold": t2_threshold,
                "discount_type": t2_dtype,
                "face_value": t2_fv,
                "discount": t2_disc,
                "cap": t2_cap,
            },
        ],
        "active_tier": 1,
        "budget": budget,
        "total_original_price": total_price,
        "price_after_voucher": price_after,
    }


def _generate_stacked_voucher(total_price: float, rng: random.Random) -> dict:
    """两张券顺序叠加：coupon0(platform/brand, %) → coupon1(shop, fixed)。
    两张券门槛均低于商品价格，均触发。
    """
    # Coupon 0：platform 或 brand，百分比折扣
    c0_vtype = rng.choice(["platform", "brand"])
    c0_threshold = round(total_price * rng.uniform(0.30, 0.60), 2)
    c0_disc = round(rng.uniform(0.05, 0.15), 2)
    c0_cap = round(total_price * rng.uniform(0.08, 0.20), 2)
    c0_actual = round(min(total_price * c0_disc, c0_cap), 2)
    price_after_c0 = round(total_price - c0_actual, 2)

    # Coupon 1：shop，固定折扣
    c1_threshold = round(total_price * rng.uniform(0.20, 0.50), 2)
    c1_fv = round(total_price * rng.uniform(0.03, 0.12), 2)
    # 安全防护：两张券合计折扣不超过原价 40%
    if (c0_actual + c1_fv) > total_price * 0.40:
        c1_fv = round(total_price * 0.04, 2)
    c1_actual = c1_fv

    price_after_both = round(price_after_c0 - c1_actual, 2)
    if price_after_both <= 0:
        return {}

    budget = round(price_after_both * rng.uniform(1.0, 1.08), 2)

    return {
        "voucher_type": "stacked",
        "discount_type": "stacked",
        # 向后兼容字段
        "threshold": c0_threshold,
        "face_value": None,
        "discount": None,
        "cap": None,
        # 新增字段
        "coupons": [
            {
                "coupon_index": 0,
                "voucher_type": c0_vtype,
                "discount_type": "percentage",
                "threshold": c0_threshold,
                "discount": c0_disc,
                "cap": c0_cap,
                "actual_discount": c0_actual,
            },
            {
                "coupon_index": 1,
                "voucher_type": "shop",
                "discount_type": "fixed",
                "threshold": c1_threshold,
                "face_value": c1_fv,
                "actual_discount": c1_actual,
            },
        ],
        "budget": budget,
        "total_original_price": total_price,
        "price_after_voucher": price_after_both,
    }
