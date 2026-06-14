"""详情类工具：get_product_details。"""

from src.tools.base import BaseTool, register_tool


def _clean_product(p: dict) -> dict:
    return {
        "product_id": p["product_id"],
        "title": p.get("title", ""),
        "main_category": p.get("main_category", ""),
        "store": p.get("store", ""),
        "price": p.get("price"),
        "features": p.get("features", []),
        "description": p.get("description", []),
        "details": p.get("details", {}),
        "bought_together": p.get("bought_together"),
    }


@register_tool("get_product_details")
class GetProductDetailsTool(BaseTool):
    description = (
        "Get full structured details of a single product, including title, store, price, "
        "structured detail attributes, and feature descriptions. Does NOT include rating or review data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "string",
                "description": "Product ID to retrieve",
            },
        },
        "required": ["product_id"],
    }

    def call(self, params: dict) -> dict:
        pid = params["product_id"]
        p = self.db.get_product(pid)
        if not p:
            return {"error": f"Product {pid} not found"}
        return {"product": _clean_product(p)}
        
