import json
from typing import Iterator

import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool as GeminiTool
from google.generativeai.types import content_types

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
        # Store raw response parts for thought_signature preservation
        self._last_raw_parts = []
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

    @classmethod
    def fetch_available_models(cls, api_key: str) -> list[str]:
        """Fetch available Gemini models via the API."""
        try:
            if api_key == "__oauth__":
                return list(cls.DEFAULT_MODELS.values())
            genai.configure(api_key=api_key)
            models = []
            for m in genai.list_models():
                name = m.name.replace("models/", "")
                if "generateContent" in [ms for ms in (m.supported_generation_methods or [])]:
                    models.append(name)
            return sorted(models) if models else list(cls.DEFAULT_MODELS.values())
        except Exception:
            return list(cls.DEFAULT_MODELS.values())

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        gemini_tools = _convert_tools(tools)

        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
            tools=gemini_tools if gemini_tools else None,
        )

        # Use ChatSession to auto-handle thought_signature for Gemini 3.x
        history = _format_history(messages[:-1])
        chat = model.start_chat(history=history)

        # Get the last user message
        last_content = _get_last_content(messages)

        response = chat.send_message(
            last_content,
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


def _get_last_content(messages: list[Message]) -> str | list[dict]:
    """Extract the last message content for send_message()."""
    if not messages:
        return "Hello"
    last = messages[-1]
    if last.role == "tool":
        # Return function responses for tool results
        parts = []
        for tr in last.tool_results:
            parts.append(
                genai.protos.Part(
                    function_response=genai.protos.FunctionResponse(
                        name=tr.tool_call_id,
                        response={"result": tr.content},
                    )
                )
            )
        return parts
    return str(last.content)


def _format_history(messages: list[Message]) -> list[genai.protos.Content]:
    """Format message history for ChatSession, preserving structure."""
    result = []
    for msg in messages:
        if msg.role == "tool":
            parts = []
            for tr in msg.tool_results:
                parts.append(
                    genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=tr.tool_call_id,
                            response={"result": tr.content},
                        )
                    )
                )
            result.append(genai.protos.Content(role="user", parts=parts))
        elif msg.role == "assistant":
            parts = []
            if msg.content:
                parts.append(genai.protos.Part(text=str(msg.content)))
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        genai.protos.Part(
                            function_call=genai.protos.FunctionCall(
                                name=tc.name,
                                args=tc.arguments,
                            )
                        )
                    )
            if parts:
                result.append(genai.protos.Content(role="model", parts=parts))
        elif msg.role == "user":
            result.append(
                genai.protos.Content(
                    role="user",
                    parts=[genai.protos.Part(text=str(msg.content))],
                )
            )
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
