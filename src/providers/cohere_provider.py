"""
Cohere provider — uses cohere's own SDK and chat message format.
Tool use is supported via Cohere's native function-calling API.
"""
import json
from typing import Iterator

import cohere

from .base import BaseProvider, Message, ToolCall, ToolResult


class CohereProvider(BaseProvider):
    name = "Cohere"

    DEFAULT_MODELS = {
        "command-a-03-2025": "command-a-03-2025",
        "command-r-plus": "command-r-plus",
        "command-r": "command-r",
        "command-r7b-12-2024": "command-r7b-12-2024",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "command-a-03-2025")
        self.client = cohere.Client(api_key=api_key)

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        chat_history, last_user_msg = _format_messages(messages, system)
        cohere_tools = _convert_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "message": last_user_msg,
            "chat_history": chat_history,
            "stream": True,
        }
        if cohere_tools:
            kwargs["tools"] = cohere_tools

        stream = self.client.chat_stream(**kwargs)

        tool_calls_seen: list[ToolCall] = []

        for event in stream:
            if event.event_type == "text-generation":
                yield event.text
            elif event.event_type == "tool-calls-generation":
                if hasattr(event, "tool_calls") and event.tool_calls:
                    for tc in event.tool_calls:
                        try:
                            args = tc.parameters if isinstance(tc.parameters, dict) else json.loads(str(tc.parameters))
                        except Exception:
                            args = {}
                        tool_calls_seen.append(
                            ToolCall(
                                id=f"cohere_{tc.name}_{id(tc)}",
                                name=tc.name,
                                arguments=args,
                            )
                        )
            elif event.event_type == "stream-end":
                # Emit any tool calls collected
                for tc in tool_calls_seen:
                    yield tc
                tool_calls_seen.clear()


def _format_messages(messages: list[Message], system: str) -> tuple[list[dict], str]:
    """
    Cohere's API takes:
      - chat_history: list of past messages
      - message: the latest user message (string)
    Returns (chat_history, last_user_message).
    """
    history = []
    last_user_msg = ""

    # Add system prompt as the first CHATBOT message (Cohere convention)
    if system:
        history.append({"role": "SYSTEM", "message": system})

    for i, msg in enumerate(messages):
        if msg.role == "user":
            # Check if this is the last user message
            remaining = messages[i + 1:]
            has_assistant_after = any(m.role in ("assistant", "tool") for m in remaining)
            if not has_assistant_after and i == len(messages) - 1:
                last_user_msg = str(msg.content)
            else:
                history.append({"role": "USER", "message": str(msg.content)})

        elif msg.role == "assistant":
            text = str(msg.content) if msg.content else ""
            if msg.tool_calls:
                tool_results = []
                for tc in msg.tool_calls:
                    tool_results.append({
                        "call": {"name": tc.name, "parameters": tc.arguments},
                        "outputs": [{"result": "pending"}],
                    })
                history.append({
                    "role": "CHATBOT",
                    "message": text,
                    "tool_calls": [{"name": tc.name, "parameters": tc.arguments} for tc in msg.tool_calls],
                })
            else:
                history.append({"role": "CHATBOT", "message": text})

        elif msg.role == "tool":
            # Tool results become a TOOL role message
            for tr in msg.tool_results:
                history.append({
                    "role": "TOOL",
                    "tool_results": [
                        {
                            "call": {"name": tr.tool_call_id, "parameters": {}},
                            "outputs": [{"result": tr.content}],
                        }
                    ],
                })

    if not last_user_msg:
        # Fallback: use last user message from history
        for m in reversed(history):
            if m.get("role") == "USER":
                last_user_msg = m["message"]
                history.remove(m)
                break

    return history, last_user_msg or "Hello"


def _convert_tools(tools: list[dict]) -> list[dict]:
    cohere_tools = []
    for t in tools:
        params = t.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        param_defs = {}
        for pname, pdef in props.items():
            param_defs[pname] = {
                "description": pdef.get("description", ""),
                "type": _map_type(pdef.get("type", "str")),
                "required": pname in required,
            }

        cohere_tools.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "parameter_definitions": param_defs,
        })
    return cohere_tools


def _map_type(json_type: str) -> str:
    return {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "array": "list",
        "object": "dict",
    }.get(json_type, "str")
