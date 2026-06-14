"""Agent：集成 get_user_profile 和 ask_user 工具。"""

import json
import logging

from src.llm import LLMClient
from src.prompts.agent import SYSTEM_PROMPT
from src.tools import create_tools, get_all_schemas
from src.database.product_db import ProductDB

logger = logging.getLogger(__name__)

_TOOL_REMINDER = (
    "Do not answer directly. Continue by calling a tool if more evidence is needed, "
    "or call recommend_product if you are ready to finish."
)
_MULTI_TOOL_CALL_REMINDER = (
    "Error: only one tool call is allowed per assistant turn. "
    "You called multiple tools in one turn. Continue with exactly one tool call."
)


def _assistant_message_for_trajectory(message: dict, reasoning_entries: list[dict] | None = None) -> dict:
    cleaned = {
        "role": "assistant",
        "content": message.get("content", "") or "",
    }
    if message.get("tool_calls"):
        cleaned["tool_calls"] = [
            {
                "id": tc.get("id", ""),
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                },
            }
            for tc in message.get("tool_calls", [])
        ]
    if reasoning_entries:
        cleaned["reasoning_trace"] = reasoning_entries
    return cleaned


class ShoppingAgent:
    """电商导购 Agent：支持用户画像和多轮澄清。"""

    def __init__(self, db: ProductDB, llm_client: LLMClient,
                 max_steps: int = 60):
        self.db = db
        self.llm = llm_client
        self.max_steps = max_steps

    def run(self, sample: dict, system_prompt: str | None = None) -> dict:
        """执行 Agent 推理循环。

        Args:
            sample: benchmark sample（含 user_query, user_persona, clarification_script）

        Returns:
            {
                "predicted_product_id": "...",
                "reasoning": "...",
                "trajectory": [...],
                "reasoning_trace": [...],
                "total_tool_calls": int,
                "ask_user_calls": int,
                "get_profile_calls": int,
                "finished": bool,
            }
        """
        tools = create_tools(
            self.db,
            sample_data=sample,
        )
        tool_schemas = get_all_schemas(tools)

        user_query = sample["user_query"]
        messages = [
            {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        llm_messages = list(messages)
        trajectory_messages = list(messages)

        total_tool_calls = 0
        ask_user_calls = 0
        get_profile_calls = 0
        predicted_id = None
        reasoning = ""
        reasoning_trace = []
        finished = False
        no_tool_corrections = 0

        for step in range(self.max_steps):
            logger.info("Step %d/%d", step + 1, self.max_steps)

            assistant_msg = self.llm.chat(llm_messages, tools=tool_schemas)
            step_reasoning = []
            for entry in assistant_msg.get("reasoning_trace", []) or []:
                enriched_entry = dict(entry)
                enriched_entry.setdefault("step", step + 1)
                step_reasoning.append(enriched_entry)
                reasoning_trace.append(enriched_entry)
            trajectory_messages.append(
                _assistant_message_for_trajectory(assistant_msg, step_reasoning or None)
            )

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                llm_messages.append(assistant_msg)
                if no_tool_corrections < 1:
                    logger.info("No tool calls; prompting for tool use")
                    no_tool_corrections += 1
                    reminder_msg = {"role": "user", "content": _TOOL_REMINDER}
                    llm_messages.append(reminder_msg)
                    trajectory_messages.append(reminder_msg)
                    continue
                logger.info("No tool calls after reminder, agent finished")
                reasoning = assistant_msg.get("content", "") or reasoning
                finished = True
                break

            no_tool_corrections = 0
            multi_tool_call_violation = False
            if len(tool_calls) > 1:
                logger.warning("Multiple tool calls; only first will run")
                tool_calls = [tool_calls[0]]
                multi_tool_call_violation = True
            llm_assistant_msg = dict(assistant_msg)
            llm_assistant_msg["tool_calls"] = tool_calls
            llm_messages.append(llm_assistant_msg)

            for tc in tool_calls:
                total_tool_calls += 1
                func_name = tc["function"]["name"]
                try:
                    func_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError as exc:
                    result = {"error": f"Invalid tool arguments JSON: {exc}"}
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.get("call_id", tc["id"]),
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                    llm_messages.append(tool_msg)
                    trajectory_messages.append(tool_msg)
                    continue

                logger.info("Tool call: %s(%s)", func_name, json.dumps(func_args, ensure_ascii=False)[:200])

                if func_name == "ask_user":
                    ask_user_calls += 1
                elif func_name == "get_user_profile":
                    get_profile_calls += 1

                tool = tools.get(func_name)
                if not tool:
                    result = {"error": f"Unknown tool: {func_name}"}
                else:
                    try:
                        result = tool.call(func_args)
                    except Exception as exc:
                        logger.warning("Tool %s raised: %s", func_name, exc)
                        result = {"error": f"Tool call failed: {exc}"}

                if func_name == "recommend_product":
                    pid = func_args.get("product_id", "")
                    if pid and self.db.get_product(pid):
                        predicted_id = pid
                    reasoning = func_args.get("reasoning", "")
                    finished = True

                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc.get("call_id", tc["id"]),
                    "content": json.dumps(result, ensure_ascii=False),
                }
                llm_messages.append(tool_msg)
                trajectory_messages.append(tool_msg)

            if multi_tool_call_violation:
                reminder_msg = {"role": "user", "content": _MULTI_TOOL_CALL_REMINDER}
                llm_messages.append(reminder_msg)
                trajectory_messages.append(reminder_msg)

            if finished:
                break

        return {
            "predicted_product_id": predicted_id,
            "reasoning": reasoning,
            "trajectory": trajectory_messages,
            "reasoning_trace": reasoning_trace,
            "total_tool_calls": total_tool_calls,
            "ask_user_calls": ask_user_calls,
            "get_profile_calls": get_profile_calls,
            "finished": finished,
        }
