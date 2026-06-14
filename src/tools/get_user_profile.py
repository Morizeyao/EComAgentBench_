"""获取用户画像工具。"""

from src.tools.base import BaseTool, register_tool


@register_tool("get_user_profile")
class GetUserProfileTool(BaseTool):
    description = (
        "Retrieve the current user's profile including demographics, lifestyle, "
        "shopping habits, and any product-specific requirements (e.g. skin type, "
        "preferred size, brand preference). Call this to discover implicit needs "
        "that the user hasn't stated in their query."
    )
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def __init__(self, db, persona: dict | None = None):
        super().__init__(db)
        self._persona = persona or {}

    def call(self, params: dict) -> dict:
        if not self._persona:
            return {"profile": {}, "message": "No user profile available"}
        return {"profile": self._persona}
