import json
import re
from typing import Iterator

import anthropic

from .base import BaseProvider, Message, ToolCall


class AnthropicProvider(BaseProvider):
    name = "Anthropic (Claude)"
    SUPPORTS_IMAGES = True

    DEFAULT_MODELS = {
        "claude-sonnet-4-6": "claude-sonnet-4-6",
        "claude-opus-4-6": "claude-opus-4-6",
        "claude-haiku-4-5-20251001": "claude-haiku-4-5-20251001",
        "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022": "claude-3-5-haiku-20241022",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "claude-sonnet-4-6")
        self.client = anthropic.Anthropic(api_key=api_key)

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        formatted = _format_messages(messages)
        anthropic_tools = _convert_tools(tools)

        with self.client.messages.stream(
            model=self.model,
            max_tokens=8096,
            system=system,
            messages=formatted,
            tools=anthropic_tools if anthropic_tools else anthropic.NOT_GIVEN,
        ) as stream:
            current_tool_id = None
            current_tool_name = None
            current_tool_args = ""

            for event in stream:
                etype = event.type

                if etype == "content_block_start":
                    if hasattr(event, "content_block"):
                        cb = event.content_block
                        if cb.type == "tool_use":
                            current_tool_id = cb.id
                            current_tool_name = cb.name
                            current_tool_args = ""

                elif etype == "content_block_delta":
                    if hasattr(event, "delta"):
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield delta.text
                        elif delta.type == "input_json_delta":
                            current_tool_args += delta.partial_json

                elif etype == "content_block_stop":
                    if current_tool_name:
                        try:
                            args = json.loads(current_tool_args) if current_tool_args else {}
                        except json.JSONDecodeError:
                            args = {"raw": current_tool_args}
                        yield ToolCall(
                            id=current_tool_id,
                            name=current_tool_name,
                            arguments=args,
                        )
                        current_tool_id = None
                        current_tool_name = None
                        current_tool_args = ""


def _format_messages(messages: list[Message]) -> list[dict]:
    result = []
    for msg in messages:
        if msg.role == "tool":
            # Tool results go back as user messages in Anthropic
            result.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tr.tool_call_id,
                        "content": tr.content,
                        "is_error": tr.is_error,
                    }
                    for tr in msg.tool_results
                ],
            })
        elif msg.tool_calls:
            # Assistant message with tool calls
            content = []
            if msg.content:
                content.append({"type": "text", "text": str(msg.content)})
            for tc in msg.tool_calls:
                content.append({
                    "type": "tool_use",
                    "id": tc.id,
                    "name": tc.name,
                    "input": tc.arguments,
                })
            result.append({"role": "assistant", "content": content})
        else:
            result.append({"role": msg.role, "content": _format_anthropic_content(msg.content)})
    return result


def _format_anthropic_content(content):
    if not isinstance(content, list):
        return str(content)
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif item.get("type") == "image_url":
            media_type, data = _parse_data_url(item.get("url", ""))
            if media_type and data:
                parts.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                })
    return parts or ""


def _parse_data_url(url: str) -> tuple[str | None, str | None]:
    match = re.match(r"data:([^;]+);base64,(.*)", url or "", flags=re.DOTALL)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _convert_tools(tools: list[dict]) -> list[dict]:
    result = []
    for t in tools:
        result.append({
            "name": t["name"],
            "description": t.get("description", ""),
            "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
        })
    return result
