import json
from typing import Iterator

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool

from .base import BaseProvider, Message, ToolCall


class GeminiProvider(BaseProvider):
    name = "Google (Gemini)"

    DEFAULT_MODELS = {
        "gemini-3.1-pro": "gemini-3.1-pro-preview",
        "gemini-3.1-flash-lite": "gemini-3.1-flash-lite-preview",
        "gemini-3-flash": "gemini-3-flash",
        "gemini-2.5-pro": "gemini-2.5-pro",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini-2.5-flash-lite": "gemini-2.5-flash-lite",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "gemini-3-flash")
        # Check for OAuth credentials first
        if api_key == "__oauth__":
            try:
                from ..core.auth import get_google_credentials
                creds = get_google_credentials()
                if creds:
                    genai.configure(credentials=creds)
                    return
            except Exception:
                pass
        genai.configure(api_key=api_key)

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        gemini_tools = _convert_tools(tools)
        formatted = _format_messages(messages)

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
            tools=gemini_tools if gemini_tools else None,
        )

        response = model.generate_content(
            formatted,
            stream=True,
            generation_config=genai.types.GenerationConfig(max_output_tokens=8096),
        )

        for chunk in response:
            if not chunk.candidates:
                continue
            for part in chunk.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    yield part.text
                elif hasattr(part, "function_call") and part.function_call:
                    fc = part.function_call
                    args = dict(fc.args) if fc.args else {}
                    yield ToolCall(
                        id=f"gemini_{fc.name}_{id(fc)}",
                        name=fc.name,
                        arguments=args,
                    )


def _format_messages(messages: list[Message]) -> list[dict]:
    result = []
    for msg in messages:
        role = "model" if msg.role == "assistant" else "user"
        if msg.role == "tool":
            parts = []
            for tr in msg.tool_results:
                parts.append({
                    "function_response": {
                        "name": tr.tool_call_id,
                        "response": {"result": tr.content},
                    }
                })
            result.append({"role": "user", "parts": parts})
        elif msg.tool_calls:
            parts = []
            if msg.content:
                parts.append({"text": str(msg.content)})
            for tc in msg.tool_calls:
                parts.append({
                    "function_call": {"name": tc.name, "args": tc.arguments}
                })
            result.append({"role": "model", "parts": parts})
        else:
            result.append({"role": role, "parts": [{"text": str(msg.content)}]})
    return result


def _convert_tools(tools: list[dict]) -> list[GeminiTool]:
    if not tools:
        return []
    decls = []
    for t in tools:
        params = t.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])
        gemini_props = {}
        for pname, pdef in props.items():
            gemini_props[pname] = _map_type(pdef)

        decls.append(
            FunctionDeclaration(
                name=t["name"],
                description=t.get("description", ""),
                parameters={
                    "type": "OBJECT",
                    "properties": gemini_props,
                    "required": required,
                },
            )
        )
    return [GeminiTool(function_declarations=decls)]


def _map_type(pdef: dict) -> dict:
    type_map = {
        "string": "STRING",
        "integer": "INTEGER",
        "number": "NUMBER",
        "boolean": "BOOLEAN",
        "array": "ARRAY",
        "object": "OBJECT",
    }
    t = pdef.get("type", "string")
    result = {"type": type_map.get(t, "STRING")}
    if "description" in pdef:
        result["description"] = pdef["description"]
    if t == "array" and "items" in pdef:
        result["items"] = _map_type(pdef["items"])
    return result
