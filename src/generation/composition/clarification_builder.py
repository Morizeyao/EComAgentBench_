"""Clarification Script 生成：为追问需求构建 slot 结构。"""

import logging

from src.llm import LLMClient
from src.prompts.generation import CLARIFICATION_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


def build_clarification_script(
    clarification_rubrics: list[dict],
    product: dict,
    llm_client: LLMClient,
    max_turns: int = 5,
) -> dict:
    """为 clarification-source rubrics 生成追问脚本。"""
    if not clarification_rubrics:
        return _empty_script(max_turns)

    slots = _generate_slots_with_llm(clarification_rubrics, product, llm_client)

    if not slots or len(slots) < len(clarification_rubrics):
        logger.warning(
            "LLM slot generation incomplete (%d/%d), using template fallback",
            len(slots) if slots else 0,
            len(clarification_rubrics),
        )
        slots = _generate_slots_template(clarification_rubrics)

    clarification_slots = []
    for i, (rubric, slot_data) in enumerate(zip(clarification_rubrics, slots)):
        clarification_slots.append({
            "slot_id": f"cl_{i + 1}",
            "linked_rubric_ids": [rubric["id"]],
            "hidden_info": slot_data.get("hidden_info", f"Hidden requirement for {rubric['field']}"),
            "trigger_keywords": slot_data.get("trigger_keywords", _default_keywords(rubric)),
            "user_response": slot_data.get("user_response", _default_response(rubric)),
            "revealed": False,
        })

    return {
        "clarification_slots": clarification_slots,
        "default_response": (
            "Hmm, I'm not sure about that. Could you ask me about something more "
            "specific? Like what features or specs I'm looking for?"
        ),
        "max_clarification_turns": max_turns,
    }


def _generate_slots_with_llm(
    rubrics: list[dict],
    product: dict,
    llm_client: LLMClient,
) -> list[dict] | None:
    """用 LLM 生成自然的 clarification slots。"""
    rubric_descriptions = []
    for r in rubrics:
        field_name = r["field"].split(".")[-1] if "." in r["field"] else r["field"]
        expected = r["expected_value"]
        rubric_descriptions.append(
            f"- Field: {field_name}, Expected value: {expected}, Type: {r['type']}"
        )

    prompt = (
        f"Product: {product.get('title', 'Unknown product')}\n"
        f"Category: {product.get('main_category', 'General')}\n\n"
        f"Hidden requirements to create clarification slots for:\n"
        + "\n".join(rubric_descriptions)
    )

    try:
        result = llm_client.chat_json(messages=[
            {"role": "system", "content": CLARIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ])
        slots = result.get("slots", [])
        if isinstance(slots, list) and len(slots) >= len(rubrics):
            return slots[:len(rubrics)]
        return None
    except Exception as e:
        logger.warning("LLM clarification generation failed: %s", e)
        return None


def _generate_slots_template(rubrics: list[dict]) -> list[dict]:
    """模板化生成 clarification slots（fallback）。"""
    slots = []
    for r in rubrics:
        field_name = r["field"].split(".")[-1] if "." in r["field"] else r["field"]
        expected = r["expected_value"]
        slots.append({
            "hidden_info": f"User needs {field_name}: {expected}",
            "trigger_keywords": _default_keywords(r),
            "user_response": _default_response(r),
        })
    return slots


def _default_keywords(rubric: dict) -> list[str]:
    """从 rubric 字段生成宽泛的 trigger keywords。"""
    field = rubric["field"]
    field_name = field.split(".")[-1] if "." in field else field
    expected = str(rubric.get("expected_value", ""))

    keywords = [field_name.lower()]

    field_keyword_map = {
        # 跨品类
        "Special Feature": ["feature", "special", "function", "capability", "technology", "spec"],
        "Material Feature": ["material", "fabric", "texture", "feel", "composition"],
        "Product Benefits": ["benefit", "advantage", "help", "good for", "useful", "purpose"],
        "Finish Type": ["finish", "surface", "coating", "texture", "look", "appearance"],
        "Recommended Uses For Product": ["use", "purpose", "scenario", "application", "suitable for", "designed for", "intended"],
        # CellPhones / Electronics
        "Form Factor": ["form", "shape", "type", "design", "format", "style", "form factor"],
        "Connectivity Technology": ["connect", "wireless", "bluetooth", "wifi", "cable", "port", "network"],
        "Connector Type": ["connector", "plug", "port", "cable", "usb", "charging", "interface"],
        "Pattern": ["pattern", "design", "print", "motif"],
        "Mounting Type": ["mount", "install", "attach", "hang", "wall", "setup"],
        "Screen Size": ["screen", "display", "size", "inch", "monitor"],
        "Shell Type": ["case", "shell", "cover", "housing", "enclosure", "body"],
        # Computers
        "Graphics Coprocessor": ["graphics", "gpu", "video card", "display adapter", "graphic card", "rendering"],
        "Computer Memory Type": ["memory type", "ddr", "ram type", "sdram", "memory standard"],
        "Hardware Platform": ["platform", "hardware", "architecture", "system", "chipset"],
        # Camera
        "Video Capture Resolution": ["video", "resolution", "recording", "capture", "fps", "4k", "1080p"],
        # Beauty
        "Scent": ["scent", "smell", "fragrance", "aroma", "odor"],
        "Item Form": ["form", "type", "format", "consistency", "texture"],
        # Office
        "Ink Color": ["ink", "color", "pen color", "writing color"],
        "Shape": ["shape", "form", "geometry", "profile"],
        "Closure": ["closure", "seal", "fastener", "zip", "snap", "clasp"],
        "Point Type": ["point", "tip", "nib", "pen tip", "writing point"],
        "Sheet Size": ["sheet", "paper size", "letter", "a4", "legal", "paper"],
        "Number of Items": ["quantity", "count", "pack", "pieces", "set", "how many"],
        # Numeric
        "average_rating": ["rating", "star", "review", "quality", "score", "rated"],
        "rating_number": ["review", "rating", "popular", "reviews count", "how many reviews"],
        "price": ["price", "cost", "budget", "spend", "expensive", "cheap", "affordable"],
    }

    extra = field_keyword_map.get(field_name, [])
    keywords.extend(extra)

    if expected and len(expected) < 30:
        keywords.append(expected.lower())
        for word in expected.lower().split():
            if len(word) >= 4:
                keywords.append(word)

    return list(dict.fromkeys(keywords))


def _default_response(rubric: dict) -> str:
    field_name = rubric["field"].split(".")[-1] if "." in rubric["field"] else rubric["field"]
    expected = rubric.get("expected_value", "")

    if rubric["type"] == "numeric_range":
        if isinstance(expected, dict):
            if "min" in expected:
                return f"I need it to be at least {expected['min']} for {field_name.lower()}."
            if "max" in expected:
                return f"I'd like {field_name.lower()} to be no more than {expected['max']}."
        return f"I'm looking for {field_name.lower()} of about {expected}."

    return f"I need {field_name.lower()} to be {expected}."


def _empty_script(max_turns: int) -> dict:
    return {
        "clarification_slots": [],
        "default_response": "I think I've covered everything in my initial request.",
        "max_clarification_turns": max_turns,
    }


def validate_clarification_script(
    script: dict,
    clarification_rubrics: list[dict],
) -> list[str]:
    """验证 clarification script 完备性。"""
    issues = []
    slots = script.get("clarification_slots", [])

    rubric_ids_covered = set()
    for slot in slots:
        for rid in slot.get("linked_rubric_ids", []):
            rubric_ids_covered.add(rid)

        if not slot.get("trigger_keywords"):
            issues.append(f"Slot {slot.get('slot_id', '?')} has no trigger_keywords")
        if not slot.get("user_response"):
            issues.append(f"Slot {slot.get('slot_id', '?')} has no user_response")

    for rubric in clarification_rubrics:
        if rubric["id"] not in rubric_ids_covered:
            issues.append(f"Clarification rubric {rubric['id']} has no corresponding slot")

    return issues
