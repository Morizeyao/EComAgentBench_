"""Review内容搜索工具：get_review_content。"""

from src.tools.base import BaseTool, register_tool


@register_tool("get_review_content")
class GetReviewContentTool(BaseTool):
    description = (
        "Search product reviews by text query for a specific product. "
        "Returns the top 5 most relevant reviews with full content "
        "including rating, title, text."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for review content",
            },
            "product_id": {
                "type": "string",
                "description": "Product ID to search reviews for",
            },
        },
        "required": ["query", "product_id"],
    }

    def call(self, params: dict) -> dict:
        query = params["query"]
        product_id = params["product_id"]
        reviews = self.db.search_reviews(query, product_id=product_id, limit=5)
        if not reviews:
            return {"reviews": [], "message": "No matching reviews found"}
        return {
            "reviews": [
                {
                    "product_id": r["product_id"],
                    "rating": r["rating"],
                    "title": r["title"],
                    "text": r["text"],
                }
                for r in reviews
            ]
        }

