"""推荐工具：recommend_product（终止工具）。"""

from src.tools.base import BaseTool, register_tool


@register_tool("recommend_product")
class RecommendProductTool(BaseTool):
    description = (
        "Submit your final product recommendation. Call this only when you have enough evidence. "
        "This ends the search process."
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "Product ID to recommend",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this product is recommended",
            },
        },
        "required": ["product_id", "reasoning"],
    }

    def call(self, params: dict) -> dict:
        pid = params["product_id"]
        p = self.db.get_product(pid)
        recommended_product = {"product_id": pid, "title": p.get("title", "")} if p else None
        return {
            "recommended_product": recommended_product,
            "reasoning": params.get("reasoning", ""),
        }
