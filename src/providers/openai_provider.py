"""
OpenAI-compatible provider base class.
Groq, Mistral, DeepSeek, Ollama all use the same OpenAI API format
with a different base_url and api_key.
"""
import json
from typing import Iterator

import requests
from openai import OpenAI

from .base import BaseProvider, Message, ToolCall


class OpenAICompatibleProvider(BaseProvider):
    """Shared implementation for any OpenAI-compatible API endpoint."""

    BASE_URL: str | None = None  # Override in subclass

    # Model ID substrings to exclude when fetching live models
    _EXCLUDE_PATTERNS: list[str] = ["embed", "tts", "whisper", "dall-e", "moderation", "audio", "realtime", "transcri"]

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or self.BASE_URL,
        )

    @classmethod
    def fetch_available_models(cls, api_key: str) -> list[str]:
        """Fetch models from the OpenAI-compatible /v1/models endpoint."""
        try:
            base = cls.BASE_URL or "https://api.openai.com/v1"
            resp = requests.get(
                f"{base}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            ids = sorted(
                m["id"] for m in data
                if not any(ex in m["id"].lower() for ex in cls._EXCLUDE_PATTERNS)
            )
            return ids if ids else list(cls.DEFAULT_MODELS.values())
        except Exception:
            return list(cls.DEFAULT_MODELS.values())

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        formatted = _format_messages(messages, system)
        openai_tools = _convert_tools(tools)

        kwargs: dict = {
            "model": self.model,
            "messages": formatted,
            "stream": True,
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        stream = self.client.chat.completions.create(**kwargs)
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


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIProvider(OpenAICompatibleProvider):
    name = "OpenAI (ChatGPT)"
    BASE_URL = None  # uses default openai.com endpoint

    DEFAULT_MODELS = {
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-nano": "gpt-4.1-nano",
        "gpt-4o": "gpt-4o",
        "gpt-4o-mini": "gpt-4o-mini",
        "o3": "o3",
        "o4-mini": "o4-mini",
        "o3-mini": "o3-mini",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "gpt-4.1-mini")


# ── Groq ──────────────────────────────────────────────────────────────────────

class GroqProvider(OpenAICompatibleProvider):
    name = "Groq"
    BASE_URL = "https://api.groq.com/openai/v1"

    DEFAULT_MODELS = {
        "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant": "llama-3.1-8b-instant",
        "gemma2-9b-it": "gemma2-9b-it",
        "deepseek-r1-distill-llama-70b": "deepseek-r1-distill-llama-70b",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "llama-3.3-70b-versatile")


# ── Mistral ───────────────────────────────────────────────────────────────────

class MistralProvider(OpenAICompatibleProvider):
    name = "Mistral AI"
    BASE_URL = "https://api.mistral.ai/v1"

    DEFAULT_MODELS = {
        "mistral-large-latest": "mistral-large-latest",
        "mistral-medium-latest": "mistral-medium-latest",
        "mistral-small-latest": "mistral-small-latest",
        "codestral-latest": "codestral-latest",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "mistral-large-latest")


# ── DeepSeek ──────────────────────────────────────────────────────────────────

class DeepSeekProvider(OpenAICompatibleProvider):
    name = "DeepSeek"
    BASE_URL = "https://api.deepseek.com/v1"

    DEFAULT_MODELS = {
        "deepseek-chat": "deepseek-chat",
        "deepseek-reasoner": "deepseek-reasoner",
    }

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(api_key, model or "deepseek-chat")


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(OpenAICompatibleProvider):
    name = "Ollama (local)"
    BASE_URL = "http://localhost:11434/v1"

    DEFAULT_MODELS = {
        "llama3.2": "llama3.2",
        "qwen2.5-coder": "qwen2.5-coder",
        "phi4": "phi4",
        "deepseek-r1": "deepseek-r1",
        "mistral": "mistral",
        "codellama": "codellama",
        "gemma2": "gemma2",
    }

    def __init__(self, api_key: str = "ollama", model: str | None = None, base_url: str | None = None):
        # Ollama doesn't need a real API key
        super().__init__(api_key or "ollama", model or "llama3.2", base_url or self.BASE_URL)

    @classmethod
    def fetch_available_models(cls, api_key: str = "ollama") -> list[str]:
        """Fetch locally available Ollama models via /api/tags."""
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return sorted(m["name"].split(":")[0] for m in models) or list(cls.DEFAULT_MODELS.values())
        except Exception:
            return list(cls.DEFAULT_MODELS.values())


# ── Shared helpers (same as before) ───────────────────────────────────────────

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
