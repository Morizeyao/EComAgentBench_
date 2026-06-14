"""特征分区算法：将 features 按 50/25/25 分配到 query/persona/clarification。"""

import math
import random

from src.generation.utils import normalize_text


def partition_features(
    features: list[dict],
    intent: dict,
    *,
    product: dict | None = None,
    persona_ratio: float = 0.25,
    clarification_ratio: float = 0.25,
    min_query: int = 3,
    min_persona: int = 1,
    min_clarification: int = 1,
    rng: random.Random | None = None,
) -> dict[str, list[dict]] | None:
    """将 features 分区到三个 info_source。

    Returns:
        {"query": [...], "persona": [...], "clarification": [...]}
        如果特征不足以满足 min 约束则返回 None。
    """
    rng = rng or random.Random()

    persona_eligible_set = set(intent.get("persona_eligible_fields", []))
    clarification_eligible_set = set(intent.get("clarification_eligible_fields", []))

    title_text = ""
    if product:
        title_text = normalize_text(product.get("title", ""))

    # 步骤1：分类 features
    fixed_query = []  # 固定 query 的 features
    persona_eligible = []
    clarification_eligible = []
    query_only = []

    for f in features:
        field = f["field"]
        ftype = f["type"]

        # entity_match / store_match / review_opinion / budget_match 固定在 query
        if ftype in ("store_match", "review_opinion", "budget_match"):
            fixed_query.append(f)
            continue

        # 标题内嵌值强制归 query：如果 feature value 出现在商品标题中，
        # query builder 无法避免提及，分到 persona/clarification 会触发泄漏
        if title_text:
            val_norm = normalize_text(str(f.get("value", "")))
            if val_norm and len(val_norm) > 2 and val_norm in title_text:
                fixed_query.append(f)
                continue

        # negative 标记的 feature 可以放 persona（avoid_preferences）或 query
        is_persona_ok = field in persona_eligible_set
        is_clarification_ok = field in clarification_eligible_set

        if f.get("negative"):
            is_persona_ok = True

        if is_persona_ok:
            persona_eligible.append(f)
        elif is_clarification_ok:
            clarification_eligible.append(f)
        else:
            query_only.append(f)

    # 双重 eligible 的优先分给 persona
    both_eligible = []
    pure_persona = []
    pure_clarification = list(clarification_eligible)

    for f in persona_eligible:
        field = f["field"]
        if field in clarification_eligible_set:
            both_eligible.append(f)
        else:
            pure_persona.append(f)

    # 步骤2：计算分配数量
    total_partitionable = len(pure_persona) + len(pure_clarification) + len(both_eligible) + len(query_only)
    total_all = total_partitionable + len(fixed_query)

    if total_all < min_query + min_persona + min_clarification:
        return None

    n_persona_target = max(min_persona, math.floor(total_partitionable * persona_ratio))
    n_clarification_target = max(min_clarification, math.floor(total_partitionable * clarification_ratio))

    # 步骤3：分配 persona
    n_persona = min(n_persona_target, len(pure_persona) + len(both_eligible))
    persona_selected = []
    rng.shuffle(pure_persona)
    rng.shuffle(both_eligible)

    for f in pure_persona:
        if len(persona_selected) >= n_persona:
            break
        persona_selected.append(f)

    for f in both_eligible:
        if len(persona_selected) >= n_persona:
            break
        persona_selected.append(f)

    persona_ids = {id(f) for f in persona_selected}

    # 步骤4：分配 clarification
    remaining_for_clarification = [
        f for f in pure_clarification if id(f) not in persona_ids
    ] + [
        f for f in both_eligible if id(f) not in persona_ids
    ]
    rng.shuffle(remaining_for_clarification)

    n_clarification = min(n_clarification_target, len(remaining_for_clarification))
    clarification_selected = remaining_for_clarification[:n_clarification]
    clarification_ids = {id(f) for f in clarification_selected}

    # 步骤5：其余全部归 query
    query_selected = list(fixed_query) + list(query_only)
    for f in persona_eligible + clarification_eligible + both_eligible:
        if id(f) not in persona_ids and id(f) not in clarification_ids:
            query_selected.append(f)

    # 步骤6：保底检查
    if len(query_selected) < min_query:
        return None
    if len(persona_selected) < min_persona:
        return None
    if len(clarification_selected) < min_clarification:
        return None

    return {
        "query": query_selected,
        "persona": persona_selected,
        "clarification": clarification_selected,
    }
