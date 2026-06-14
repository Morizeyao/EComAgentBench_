"""工具模块：含 get_user_profile 和 ask_user。"""

from src.tools.base import TOOL_REGISTRY, BaseTool
from src.tools.search_products import SearchProductsTool
from src.tools.filter import FilterByAttributeTool
from src.tools.detail import GetProductDetailsTool
from src.tools.store import GetStoreProductsTool
from src.tools.answer import RecommendProductTool
from src.tools.python_execute import PythonExecuteTool
from src.tools.review_content import GetReviewContentTool
from src.tools.review_stats import GetProductReviewStatsTool

from src.tools.get_user_profile import GetUserProfileTool
from src.tools.ask_user import AskUserTool

TOOL_ORDER = [
    "search_products",
    "filter_by_attribute",
    "get_product_details",
    "get_review_content",
    "get_product_review_stats",
    "get_store_products",
    "get_user_profile",
    "ask_user",
    "python_execute",
    "recommend_product",
]


def create_tools(db, sample_data: dict | None = None) -> dict[str, BaseTool]:
    """创建全部工具实例。

    Args:
        db: ProductDB 实例
        sample_data: 包含 user_persona 和 clarification_script 的 benchmark sample
    """
    tools = {}
    ordered_names = [name for name in TOOL_ORDER if name in TOOL_REGISTRY]
    remaining = [name for name in TOOL_REGISTRY if name not in ordered_names]
    for name in [*ordered_names, *remaining]:
        cls = TOOL_REGISTRY[name]
        if name == "get_user_profile":
            tools[name] = cls(db, persona=sample_data.get("user_persona") if sample_data else None)
        elif name == "ask_user":
            tools[name] = cls(db, clarification_script=sample_data.get("clarification_script") if sample_data else None)
        else:
            tools[name] = cls(db)
    return tools


def get_all_schemas(tools: dict[str, BaseTool]) -> list[dict]:
    return [t.openai_schema for t in tools.values()]
