"""搜索类工具：search_products。"""

from src.tools.base import BaseTool, register_tool

_PAGE_SIZE = 20

@register_tool("search_products")
class SearchProductsTool(BaseTool):
    description = (
        "Search the product catalog by text query. Only searches product TITLES (not descriptions, "
        "features, or attributes). Uses BM25 ranking. Returns up to 20 results per page with "
        "product_id and title."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query text"},
            "page": {
                "type": "integer",
                "description": "Page number (0-indexed). Page 0 returns the first 20 results, page 1 returns results 21-40, etc.",
                "default": 0,
            },
        },
        "required": ["query"],
    }

    def call(self, params: dict) -> dict:
        query = params["query"]
        page = max(0, params.get("page", 0))
        start = page * _PAGE_SIZE
        pids = self.db.search(query, limit=start + _PAGE_SIZE)
        page_pids = pids[start:start + _PAGE_SIZE]
        products = []
        for pid in page_pids:
            p = self.db.get_product(pid)
            if p:
                products.append({
                    "product_id": pid,
                    "title": p.get("title", ""),
                })
        return {"products": products, "page": page, "total_found": len(pids)}
