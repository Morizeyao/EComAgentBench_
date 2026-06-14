"""工具基类与注册机制。"""

TOOL_REGISTRY: dict[str, type] = {}


def register_tool(name: str):
    def decorator(cls):
        cls.name = name
        TOOL_REGISTRY[name] = cls
        return cls
    return decorator


class BaseTool:
    name: str = ""
    description: str = ""
    parameters: dict = {}

    def __init__(self, db):
        self.db = db

    def call(self, params: dict) -> dict:
        raise NotImplementedError

    @property
    def openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
