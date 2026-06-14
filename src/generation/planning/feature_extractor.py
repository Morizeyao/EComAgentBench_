"""特征抽取：最大化提取商品可用特征。

与 v2 的核心区别：
- 保留 v2 的全部下采样逻辑（store=0, brand=0.2 等不变）
- 提高 max_features 上限，减少 _select_diverse 截断
- 目标是尽量用完商品的可用特征
"""

import math
import random
import re

from src.generation.config import (
    DETAIL_FIELDS,
    MULTI_OPTION_FIELDS,
    NEGATIVE_ATTRIBUTE_FIELDS,
    PRICE_BUCKETS,
    RATING_NUMBER_BUCKETS,
    get_detail_field_prob,
    get_field_prob,
    resolve_detail_value,
)

# 连续空白字符，用于将属性值中的多余空格归一化为单个空格
_WHITESPACE_RE = re.compile(r"\s+")

# Amazon 数据常见 artifact 模式（任一命中即视为脏值）
_ARTIFACT_RE = re.compile(
    r"See more|See less"     # Amazon 页面截断标记，如 "… See more"
    r"|https?://"            # 混入的 URL 链接
    r"|&[a-z]+;"             # 未转义的 HTML 实体，如 &amp; &apos;
    r"|›"                    # Amazon 品类面包屑分隔符
    r"|^B0[A-Z0-9]{8}$",    # ASIN 编码被误填为属性值，如 Color="B07CZ76391"
    re.IGNORECASE,
)
# Amazon 变体编码后缀，如 "Black-d04"、"Multi-a84"、"black-q41"
_VARIANT_SUFFIX_RE = re.compile(r"-[a-zA-Z]\d{2,}$")
# Emoji 字符（装饰性 Unicode 符号和杂项符号区段）
_EMOJI_RE = re.compile(r"[\U0001F300-\U0001F9FF\U00002600-\U000027BF]")
# 纯数字值，如 "1"、"8456"、"99.99"
_PURE_DIGIT_RE = re.compile(r"^\d+\.?\d*$")
# 允许纯数字值的字段（这些字段的数字值本身是有意义的）
_PURE_DIGIT_OK_FIELDS = {"Number of Items", "Screen Size"}


def _is_clean_attribute_value(field: str, value: str) -> bool:
    """过滤不适合作为 rubric 约束的 Amazon 数据 artifact。"""
    if not value or len(value) > 120:
        return False
    if _ARTIFACT_RE.search(value):
        return False
    if _VARIANT_SUFFIX_RE.search(value):
        return False
    if _EMOJI_RE.search(value):
        return False
    if _PURE_DIGIT_RE.fullmatch(value) and field not in _PURE_DIGIT_OK_FIELDS:
        return False
    if len(value) > 50 and value[:30] in value[len(value) // 2:]:
        return False
    return True


def _canonicalize_detail_feature_value(field: str, value: object,
                                       rng: random.Random) -> str:
    """把多值 metadata 压成单一、可解释的候选值。"""
    raw = _WHITESPACE_RE.sub(" ", str(value)).strip()
    if not raw:
        return ""
    if field not in MULTI_OPTION_FIELDS:
        return raw

    options = [
        option.strip(" -")
        for option in re.split(r"\s*(?:,|/|\|)\s*", raw)
        if option.strip(" -")
    ]
    if len(options) < 2:
        return raw

    filtered = [
        option for option in options
        if len(option.split()) <= 4 and len(option) <= 40
    ]
    if not filtered:
        return raw
    return rng.choice(filtered)


def _rating_number_bucket(value: int) -> int:
    """把 ratings count 压到常见阈值桶，方便稳定生成 query。"""
    buckets = RATING_NUMBER_BUCKETS
    eligible = [bucket for bucket in buckets if bucket <= value]
    return eligible[-1] if eligible else buckets[0]


def _price_max_bucket(value: float) -> int:
    """把价格转换成常见上界，避免 query 里出现过细价格。"""
    for bucket in PRICE_BUCKETS:
        if value <= bucket:
            return bucket
    return int(math.ceil(value / 50.0) * 50)


def _is_valid_negative_feature(feature: dict) -> bool:
    """筛掉不适合构造 negative_attribute 的脏值或弱约束。"""
    field = feature["field"]
    if field not in NEGATIVE_ATTRIBUTE_FIELDS:
        return False

    value = str(feature["value"]).strip()
    bare_field = field.split(".")[-1] if "." in field else field
    if not _is_clean_attribute_value(bare_field, value):
        return False

    lower = value.lower()
    if field in {"details.Product Benefits", "details.Material Feature"}:
        if len(value) > 50:
            return False
        if "," not in value and len(value.split()) > 6:
            return False

    if field == "details.Style":
        if any(ch.isdigit() for ch in value):
            return False
        if any(token in lower for token in ("bundle", "pack", "piece set", "piece", "count")):
            return False

    return True


def extract_features(product: dict, intent_id: str,
                        min_features: int = 8, max_features: int = 20,
                        rng: random.Random | None = None,
                        prefer_fields: list[str] | None = None) -> list[dict]:
    """从单个商品中提取可验证的特征列表。"""
    rng = rng or random.Random()
    features = []

    pid = product["product_id"]
    details = product.get("details", {})
    available_fields = [f for f in DETAIL_FIELDS if resolve_detail_value(details, f)]
    if prefer_fields:
        available_fields.sort(key=lambda f: (f not in prefer_fields, rng.random()))
    else:
        rng.shuffle(available_fields)

    for field in available_fields:
        if rng.random() >= get_detail_field_prob(field, intent_id):
            continue
        val = _canonicalize_detail_feature_value(field, resolve_detail_value(details, field), rng)
        if val and _is_clean_attribute_value(field, val):
            features.append({
                "field": f"details.{field}",
                "value": val,
                "type": "attribute_match",
                "product_id": pid,
                "rubric": f"The recommended product's {field} should be/contain '{val}'",
            })

    if product.get("average_rating") and rng.random() < get_field_prob("average_rating", intent_id):
        threshold = max(1.0, math.floor(float(product["average_rating"]) * 2) / 2)
        features.append({
            "field": "average_rating",
            "value": threshold,
            "type": "numeric_range",
            "product_id": pid,
            "rubric": f"The recommended product should have an average rating of at least {threshold:g}",
        })
    if product.get("rating_number") and product["rating_number"] >= 10 and rng.random() < get_field_prob("rating_number", intent_id):
        threshold = _rating_number_bucket(int(product["rating_number"]))
        features.append({
            "field": "rating_number",
            "value": threshold,
            "type": "numeric_range",
            "product_id": pid,
            "rubric": f"The recommended product should have at least {threshold} ratings",
        })

    if product.get("price") and rng.random() < get_field_prob("price", intent_id):
        price_cap = _price_max_bucket(float(product["price"]))
        features.append({
            "field": "price",
            "value": price_cap,
            "type": "numeric_range",
            "product_id": pid,
            "rubric": f"The recommended product should be priced under ${price_cap:g}",
        })

    if product.get("store") and rng.random() < get_field_prob("store", intent_id):
        features.append({
            "field": "store",
            "value": product["store"],
            "type": "attribute_match",
            "product_id": pid,
            "rubric": f"The recommended product should be from store '{product['store']}'",
        })

    # 去重
    seen = set()
    unique = []
    for f in features:
        key = (f["field"], f.get("product_id"))
        if key not in seen:
            seen.add(key)
            unique.append(f)

    # negative_constraint 标记
    if intent_id == "negative_constraint":
        attr_features = [f for f in unique if f["type"] == "attribute_match"
                         and _is_valid_negative_feature(f)]
        n_neg = min(rng.randint(1, 2), len(attr_features))
        for f in rng.sample(attr_features, n_neg):
            f["negative"] = True

    # 宽松的多样性选择，保留更多特征
    rng.shuffle(unique)
    selected = _select_diverse(unique, min_features, max_features, rng)

    if intent_id == "negative_constraint":
        has_neg = any(f.get("negative") for f in selected)
        if not has_neg:
            neg_pool = [f for f in unique if f.get("negative")]
            if neg_pool:
                selected[-1] = neg_pool[0]

    return selected


def _select_diverse(features: list, min_n: int, max_n: int,
                       rng: random.Random) -> list:
    """多样性选择：保留尽量多的特征。"""
    if len(features) <= max_n:
        return features

    by_type: dict[str, list] = {}
    for f in features:
        by_type.setdefault(f["type"], []).append(f)

    selected = []
    for group in by_type.values():
        selected.append(rng.choice(group))

    remaining = [f for f in features if f not in selected]
    rng.shuffle(remaining)

    while len(selected) < max_n and remaining:
        selected.append(remaining.pop())

    return selected[:max_n]
