"""外部工具：python_execute。"""

import io
import logging
from contextlib import redirect_stdout

from src.tools.base import BaseTool, register_tool

logger = logging.getLogger(__name__)

@register_tool("python_execute")
class PythonExecuteTool(BaseTool):
    description = (
        "Run short Python code for calculations or lightweight text/data transformation when other "
        "tools are not enough. Returns printed stdout only. Limited to a small set of safe built-in modules."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    }

    SAFE_MODULES = {"math", "statistics", "decimal", "fractions", "json", "re"}

    def call(self, params: dict) -> dict:
        code = params["code"]
        stdout = io.StringIO()
        local_ns = {"__builtins__": {
            "print": print, "range": range, "len": len, "int": int, "float": float,
            "str": str, "list": list, "dict": dict, "tuple": tuple, "set": set,
            "sum": sum, "min": min, "max": max, "abs": abs, "round": round,
            "sorted": sorted, "enumerate": enumerate, "zip": zip, "map": map,
            "filter": filter, "isinstance": isinstance, "type": type, "bool": bool,
            "__import__": lambda name: __import__(name) if name in self.SAFE_MODULES else None,
        }}
        with redirect_stdout(stdout):
            exec(code, local_ns)
        return {"output": stdout.getvalue(), "error": None}
