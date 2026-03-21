import json
from typing import Iterator

from openai import OpenAI

from .base import BaseProvider, Message, ToolCall


class OpenAIProvider(BaseProvider):
    name = "OpenAI (ChatGPT)"

    DEFAULT_MODELS = {
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "gpt-4-turbo": "gpt-4-turbo",
        "o1": "o1",
        "o3-mini": "o3-mini",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "gpt-4o")
        self.client = OpenAI(api_key=api_key)

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        formatted = _format_messages(messages, system)
        openai_tools = _convert_tools(tools)

        kwargs = {
            "model": self.model,
            "messages": formatted,
            "stream": True,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(**kwargs)

        # Track streaming tool calls
        tool_calls_map: dict[int, dict] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                yield delta.content

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_map:
                        tool_calls_map[idx] = {
                            "id": tc_delta.id or "",
                            "name": tc_delta.function.name if tc_delta.function else "",
                            "args": "",
                        }
                    if tc_delta.id:
                        tool_calls_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_map[idx]["args"] += tc_delta.function.arguments

            finish = chunk.choices[0].finish_reason if chunk.choices else None
            if finish in ("tool_calls", "stop") and tool_calls_map:
                for tc in tool_calls_map.values():
                    try:
                        args = json.loads(tc["args"]) if tc["args"] else {}
                    except json.JSONDecodeError:
                        args = {"raw": tc["args"]}
                    yield ToolCall(id=tc["id"], name=tc["name"], arguments=args)
                tool_calls_map.clear()


def _format_messages(messages: list[Message], system: str) -> list[dict]:
    result = [{"role": "system", "content": system}]
    for msg in messages:
        if msg.role == "tool":
            for tr in msg.tool_results:
                result.append({
                    "role": "tool",
                    "tool_call_id": tr.tool_call_id,
                    "content": tr.content,
                })
        elif msg.tool_calls:
            openai_tcs = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
            result.append({
                "role": "assistant",
                "content": str(msg.content) if msg.content else None,
                "tool_calls": openai_tcs,
            })
        else:
            result.append({"role": msg.role, "content": str(msg.content)})
    return result


def _convert_tools(tools: list[dict]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]
