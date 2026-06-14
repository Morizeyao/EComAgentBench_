"""Review搜索工具：check_review_stats。"""

from src.tools.base import BaseTool, register_tool




@register_tool("get_product_review_stats")
class GetProductReviewStatsTool(BaseTool):
    description = (
        "Get rating statistics for one or more products (max 10). "
        "Returns only product_id, average_rating, and rating_number. "
        "Does NOT return review text content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Product IDs to retrieve rating info for (max 10)",
                "maxItems": 10,
            },
        },
        "required": ["product_ids"],
    }

    def call(self, params: dict) -> dict:
        pids = params["product_ids"][:10]
        results = []
        for pid in pids:
            p = self.db.get_product(pid)
            if p:
                results.append({
                    "product_id": pid,
                    "average_rating": p.get("average_rating"),
                    "rating_number": p.get("rating_number"),
                })
        return {"products": results}
