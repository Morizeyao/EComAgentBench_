"""店铺工具：get_store_products。"""

from src.tools.base import BaseTool, register_tool

_PAGE_SIZE = 20


@register_tool("get_store_products")
class GetStoreProductsTool(BaseTool):
    description = (
        "Get products from a specific store. "
        "Returns up to 20 results per page with product_id and title."
    )
    parameters = {
        "type": "object",
        "properties": {
            "store_name": {"type": "string", "description": "Exact store name"},
            "page": {
                "type": "integer",
                "description": "Page number (0-indexed)",
                "default": 0,
            },
        },
        "required": ["store_name"],
    }

    def call(self, params: dict) -> dict:
        store = params["store_name"]
        page = max(0, params.get("page", 0))
        all_pids = self.db.get_store_products(store)
        start = page * _PAGE_SIZE
        page_pids = all_pids[start:start + _PAGE_SIZE]
        products = []
        for pid in page_pids:
            p = self.db.get_product(pid)
            if p:
                products.append({
                    "product_id": pid,
                    "title": p.get("title", ""),
                })
        return {
            "store": store,
            "products": products,
            "page": page,
            "total_count": len(all_pids),
        }
