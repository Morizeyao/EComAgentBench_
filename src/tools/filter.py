"""过滤类工具：filter_by_attribute。"""

from src.tools.base import BaseTool, register_tool

@register_tool("filter_by_attribute")
class FilterByAttributeTool(BaseTool):
    description = (
        "Filter candidate products by a structured detail attribute using case-insensitive substring "
        "matching. Use the canonical detail key name exactly as stored in the catalog (e.g. 'Brand', "
        "'Material', 'Compatible Phone Models', 'Connector Type'). Only filters within the provided "
        "candidate set (max 10 products). Does NOT search the full catalog."
    )
    parameters = {
        "type": "object",
        "properties": {
            "product_ids": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Product IDs to filter (max 10)",
                "maxItems": 10,
            },
            "attribute_name": {
                "type": "string",
                "description": "Canonical attribute key to filter on, such as 'Brand', 'Item Form', 'Hair Type', or 'Skin Type'",
            },
            "attribute_value": {
                "type": "string",
                "description": "The attribute value to match (case-insensitive)",
            },
        },
        "required": ["product_ids", "attribute_name", "attribute_value"],
    }

    def call(self, params: dict) -> dict:
        attr_name = params["attribute_name"].lower()
        value_lower = params["attribute_value"].lower()
        result = []
        for pid in params["product_ids"][:10]:
            p = self.db.get_product(pid)
            if not p:
                continue
            details = p.get("details", {}) or {}
            normalized_details = {
                str(key).lower(): value
                for key, value in details.items()
            }
            attr_val = str(normalized_details.get(attr_name, "")).lower()
            if attr_val and value_lower in attr_val:
                result.append(pid)
        return {"product_ids": result}
