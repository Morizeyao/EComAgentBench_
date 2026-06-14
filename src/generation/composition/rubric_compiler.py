"""把统一 rubric plan 和 LLM query surface 编译成最终 rubrics。"""

from src.generation.utils import keyword_tokens, normalize_text


def compile_final_rubrics(
    rubrics: list[dict],
    product: dict,
    rubric_surfaces: object,
    negative_constraints: object = None,
    entity_plans: object = None,
    max_rubric_id: int = 0,
    user_query: str = "",
) -> list[dict]:
    """仅允许 LLM 提供 query surface，不允许改变 rubric 语义。"""
    surface_map = _index_surface_map(rubric_surfaces)
    negative_map = _index_negative_map(negative_constraints)
    entity_map = _index_entity_map(entity_plans, product)

    next_rubric_num = max(max_rubric_id + 1, _next_rubric_number(rubrics)) if entity_map else 0
    compiled: list[dict] = []
    for rubric in rubrics:
        query_surface = surface_map.get(rubric["id"], "")
        if user_query and query_surface:
            query_surface = _repair_surface(query_surface, user_query)
        compiled_rubric = {
            "id": rubric["id"],
            "type": rubric["type"],
            "field": rubric["field"],
            "expected_value": rubric.get("expected_value"),
            "description": query_surface or str(rubric.get("description", "")),
            "query_surface": query_surface,
        }

        if compiled_rubric["type"] == "negative_attribute":
            negative = negative_map.get(compiled_rubric["id"])
            if negative:
                compiled_rubric["expected_value"] = negative["excluded_values"]
                if negative.get("query_surface"):
                    neg_surface = negative["query_surface"]
                    if user_query:
                        neg_surface = _repair_surface(neg_surface, user_query)
                    compiled_rubric["description"] = neg_surface
                    compiled_rubric["query_surface"] = neg_surface

        compiled.append(compiled_rubric)

    entity_plan = entity_map.get(0)
    if entity_plan:
        compiled.append({
            "id": f"r{next_rubric_num}",
            "type": "entity_match",
            "field": "title",
            "expected_value": entity_plan["query_surface"],
            "description": entity_plan["query_surface"],
            "query_surface": entity_plan["query_surface"],
            "canonical_entity": entity_plan["canonical_entity"],
            "title_span": entity_plan["title_span"],
        })

    return compiled


# ── 工具函数 ──

def _repair_surface(surface: str, user_query: str) -> str:
    """尝试把 surface 修复为 user_query 中词重叠最高的精确子串。

    若 surface 已锚定（normalize 后是 user_query 的子串），直接返回原值。
    否则滑动与 surface 等词数的窗口，取 Jaccard 最高且 >= 0.5 的 span。
    """
    surface_norm = normalize_text(surface)
    query_norm = normalize_text(user_query)
    if not surface_norm or surface_norm in query_norm:
        return surface

    surface_tokens = set(keyword_tokens(surface))
    if not surface_tokens:
        return surface

    query_words = user_query.split()
    n = len(surface.split())
    if n < 1 or n > len(query_words):
        return surface

    best_score, best_span = 0.0, surface
    for i in range(len(query_words) - n + 1):
        span = " ".join(query_words[i : i + n])
        span_tokens = set(keyword_tokens(span))
        union = surface_tokens | span_tokens
        score = len(surface_tokens & span_tokens) / len(union) if union else 0.0
        if score > best_score:
            best_score, best_span = score, span

    return best_span if best_score >= 0.5 else surface


# ── 私有索引构建 ──

def _is_valid_short_text(value: object, max_words: int = 10,
                         max_chars: int = 80) -> bool:
    """判断 value 是否为合规的短文本。"""
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or len(text) > max_chars:
        return False
    if any(ch in text for ch in ";!?"):
        return False
    return len(text.split()) <= max_words


def _index_surface_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, list):
        return {}
    indexed: dict[str, str] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        rubric_id = str(item.get("rubric_id", "")).strip()
        query_surface = str(item.get("query_surface", "")).strip()
        if not rubric_id or not _is_valid_short_text(query_surface, max_words=18, max_chars=140):
            continue
        indexed[rubric_id] = query_surface
    return indexed


def _index_negative_map(raw: object) -> dict[str, dict]:
    if not isinstance(raw, list):
        return {}
    indexed: dict[str, dict] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        rubric_id = str(item.get("rubric_id", "")).strip()
        excluded = item.get("excluded_values")
        if not rubric_id or not isinstance(excluded, list) or not excluded:
            continue
        cleaned = [
            str(value).strip()
            for value in excluded
            if _is_valid_short_text(value, max_words=8, max_chars=40)
        ]
        if not cleaned:
            continue
        query_surface = str(item.get("query_surface", "")).strip()
        indexed[rubric_id] = {
            "excluded_values": cleaned,
            "query_surface": query_surface if normalize_text(query_surface) else "",
        }
    return indexed


def _index_entity_map(raw: object, product: dict) -> dict[int, dict]:
    if not isinstance(raw, list):
        return {}
    indexed: dict[int, dict] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            group_index = int(item.get("group_index"))
        except (TypeError, ValueError):
            continue
        if group_index != 0:
            continue

        query_surface = str(item.get("query_surface", "")).strip()
        canonical_entity = str(item.get("canonical_entity", "")).strip()
        title_span = str(item.get("title_span", "")).strip()
        if not (
            _is_valid_short_text(query_surface, max_words=10, max_chars=90)
            and _is_valid_short_text(canonical_entity, max_words=6, max_chars=60)
            and _is_valid_short_text(title_span, max_words=8, max_chars=80)
        ):
            continue
        title_norm = normalize_text(product.get("title", ""))
        title_span_norm = normalize_text(title_span)
        if not title_norm or not title_span_norm or title_span_norm not in title_norm:
            continue
        entity_tokens = set(keyword_tokens(canonical_entity))
        query_tokens = set(keyword_tokens(query_surface))
        if not entity_tokens or not query_tokens or not entity_tokens <= query_tokens:
            continue
        indexed[0] = {
            "query_surface": query_surface,
            "canonical_entity": canonical_entity,
            "title_span": title_span,
        }
    return indexed


def _next_rubric_number(rubrics: list[dict]) -> int:
    max_num = 0
    for rubric in rubrics:
        rubric_id = str(rubric.get("id", "")).strip()
        if rubric_id.startswith("r") and rubric_id[1:].isdigit():
            max_num = max(max_num, int(rubric_id[1:]))
    return max_num + 1
