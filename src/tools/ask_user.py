"""向用户提问澄清工具。"""

import logging

from src.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)

_EXHAUSTED_RESPONSE = (
    "You have asked too much. I'm not interested in answering your questions anymore."
)
_TOO_MANY_CHARS = "There is over 100 chars in your question. I can't answer that. Shorten your question."


@register_tool("ask_user")
class AskUserTool(BaseTool):
    description = (
        "Ask the user a clarifying question to gather information. The number of chars in question must be under 100, otherwise the tool will return error. You can only ask user up to 10 turns."
    )
    parameters = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user",
            },
        },
        "required": ["question"],
    }

    def __init__(self, db, clarification_script: dict | None = None):
        super().__init__(db)
        self._script = clarification_script or {}
        self._slots = [dict(s) for s in self._script.get("clarification_slots", [])]
        self._default = self._script.get(
            "default_response",
            "I'm not sure about that. Is there anything else you'd like to ask?",
        )
        self._max_turns = self._script.get("max_clarification_turns", 10)
        self._turn_count = 0

    def call(self, params: dict) -> dict:
        question = params.get("question", "")
        self._turn_count += 1

        if self._turn_count > self._max_turns:
            return {"user_response": _EXHAUSTED_RESPONSE, "turns_remaining": 0}
        if len(question) > 100:
            return {"user_response": _TOO_MANY_CHARS, 
            "turns_remaining": self._max_turns - self._turn_count}

        matched_slot = self._match_slot(question)
        if matched_slot:
            matched_slot["revealed"] = True
            response = matched_slot["user_response"]
            logger.info("Clarification slot '%s' triggered", matched_slot.get("slot_id", "?"))
        else:
            response = self._default

        return {
            "user_response": response,
            "turns_remaining": self._max_turns - self._turn_count,
        }

    def _match_slot(self, question: str) -> dict | None:
        question_lower = question.lower()
        question_words = set(question_lower.split())

        best_slot = None
        best_score = 0
        for slot in self._slots:
            if slot.get("revealed"):
                continue
            keywords = slot.get("trigger_keywords", [])
            score = 0
            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in question_lower:
                    score += 2
                elif any(kw_lower in w or w in kw_lower for w in question_words if len(w) >= 3):
                    score += 1
            if score > best_score:
                best_score = score
                best_slot = slot

        return best_slot if best_score > 0 else None
