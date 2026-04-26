"""
OpenAI-compatible provider base class.
Groq, Mistral, DeepSeek, Ollama all use the same OpenAI API format
with a different base_url and api_key.
"""
import json
import re
from typing import Iterator

import requests
import httpx
from openai import APIStatusError, NotFoundError, OpenAI, RateLimitError

from .base import BaseProvider, Message, ToolCall


class OpenAICompatibleProvider(BaseProvider):
    """Shared implementation for any OpenAI-compatible API endpoint."""

    BASE_URL: str | None = None  # Override in subclass

    # Model ID substrings to exclude when fetching live models
    _EXCLUDE_PATTERNS: list[str] = ["embed", "tts", "whisper", "dall-e", "moderation", "audio", "realtime", "transcri"]
    _EXCLUDE_MODEL_IDS: set[str] = set()

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model)
        self._suppress_content = False
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or self.BASE_URL,
            timeout=180.0,  # Increased to 3 minutes for slower NIM models
        )

    @classmethod
    def fetch_available_models(cls, api_key: str) -> list[str]:
        """Fetch models from the OpenAI-compatible /v1/models endpoint."""
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
            if m["id"] not in cls._EXCLUDE_MODEL_IDS
            and not any(ex in m["id"].lower() for ex in cls._EXCLUDE_PATTERNS)
        )
        return ids if ids else list(cls.DEFAULT_MODELS.values())

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        formatted = self._format_messages(messages, system)
        openai_tools = self._convert_tools(tools) if getattr(self, "SUPPORTS_TOOLS", True) else []
        self._suppress_content = False

        kwargs: dict = {
            "model": self.model,
            "messages": formatted,
            "stream": True,
            "temperature": 0.0,  # Use 0 for more stable/predictable code generation
            "max_tokens": getattr(self, "MAX_TOKENS", 4096),
        }
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        try:
            stream = self.client.chat.completions.create(**kwargs)
        except RateLimitError as exc:
            raise RuntimeError(self._format_rate_limit_error(exc)) from exc
        except APIStatusError as exc:
            if exc.status_code == 429:
                raise RuntimeError(self._format_rate_limit_error(exc)) from exc
            if exc.status_code == 404:
                raise RuntimeError(self._format_not_found_error(exc)) from exc
            if exc.status_code == 410:
                raise RuntimeError(self._format_gone_error(exc)) from exc
            raise
        tool_calls_map: dict[int, dict] = {}

        try:
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                if delta.content:
                    yield self._clean_content_delta(delta.content)

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
                if finish == "length":
                    raise RuntimeError(
                        "Model output was truncated before completion. "
                        "No reliable tool call can be executed from this partial response. "
                        "Try a shorter request, a larger-output model, or build mode."
                    )
                if finish == "content_filter":
                    raise RuntimeError("Model output was blocked by the provider content filter.")
                if finish in ("tool_calls", "stop") and tool_calls_map:
                    for tc in tool_calls_map.values():
                        try:
                            args = json.loads(tc["args"]) if tc["args"] else {}
                        except json.JSONDecodeError:
                            args = {"raw": tc["args"]}
                        yield ToolCall(id=tc["id"], name=tc["name"], arguments=args)
                    tool_calls_map.clear()
        except RateLimitError as exc:
            raise RuntimeError(self._format_rate_limit_error(exc)) from exc
        except APIStatusError as exc:
            if exc.status_code == 429:
                raise RuntimeError(self._format_rate_limit_error(exc)) from exc
            if exc.status_code == 404:
                raise RuntimeError(self._format_not_found_error(exc)) from exc
            if exc.status_code == 410:
                raise RuntimeError(self._format_gone_error(exc)) from exc
            raise
        except httpx.RemoteProtocolError as exc:
            raise RuntimeError(self._format_stream_interrupted_error(exc)) from exc
        except NotFoundError as exc:
            raise RuntimeError(self._format_not_found_error(exc)) from exc

    def _format_stream_interrupted_error(self, exc: Exception) -> str:
        if isinstance(self, NvidiaProvider):
            return (
                f"NVIDIA NIM stream for model '{self.model}' closed before the response finished. "
                "This is usually a provider/network streaming interruption, not a local file error. "
                "Retry the request, switch to a smaller/faster model, or shorten the prompt."
            )
        return (
            f"Provider stream for model '{self.model}' closed before the response finished. "
            "Retry the request or shorten the prompt."
        )

    def _format_rate_limit_error(self, exc: Exception) -> str:
        retry_after = None
        response = getattr(exc, "response", None)
        if response is not None:
            retry_after = response.headers.get("retry-after")
        suffix = f" Retry after about {retry_after} seconds." if retry_after else " Wait a bit and retry."
        if isinstance(self, NvidiaProvider):
            return (
                f"NVIDIA NIM rate limit hit for model '{self.model}' (HTTP 429 Too Many Requests)."
                f"{suffix} For free NIM keys, try a smaller/faster model or wait before sending another request."
            )
        return f"Provider rate limit hit for model '{self.model}' (HTTP 429 Too Many Requests).{suffix}"

    def _format_not_found_error(self, exc: Exception) -> str:
        if isinstance(self, NvidiaProvider):
            return (
                f"NVIDIA NIM model/function for '{self.model}' was not found for this account (HTTP 404). "
                "This usually means the model ID is unavailable, deprecated, or not enabled for your free key. "
                "Use Update Models to refresh the live catalog, or switch to qwen/qwen3-coder-480b-a35b-instruct, "
                "moonshotai/kimi-k2-instruct-0905, nvidia/nemotron-3-super-120b-a12b, or nvidia/nemotron-3-nano-30b-a3b if available."
            )
        return f"Provider model/function for '{self.model}' was not found (HTTP 404)."

    def _format_gone_error(self, exc: Exception) -> str:
        detail = ""
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                payload = response.json()
                detail = str(payload.get("detail") or payload.get("title") or "")
            except Exception:
                detail = getattr(response, "text", "") or ""
        if isinstance(self, NvidiaProvider):
            return (
                f"NVIDIA NIM model '{self.model}' is no longer available (HTTP 410 Gone). "
                f"{detail} "
                "Use Update Models to refresh the live catalog, or switch to qwen/qwen3-coder-480b-a35b-instruct, "
                "moonshotai/kimi-k2-instruct-0905, nvidia/nemotron-3-super-120b-a12b, or nvidia/nemotron-3-nano-30b-a3b."
            ).strip()
        return f"Provider model '{self.model}' is no longer available (HTTP 410 Gone). {detail}".strip()

    def _clean_content_delta(self, content: str) -> str:
        return content

    def _format_messages(self, messages: list[Message], system: str) -> list[dict]:
        """Convert Internal Message objects to OpenAI dict format."""
        return _format_messages(messages, system)

    def _convert_tools(self, tools: list[dict]) -> list[dict]:
        """Convert internal tool schemas to OpenAI tool format."""
        return _convert_tools(tools)


# ── OpenAI ────────────────────────────────────────────────────────────────────

class OpenAIProvider(OpenAICompatibleProvider):
    name = "OpenAI (ChatGPT)"
    BASE_URL = None  # uses default openai.com endpoint
    SUPPORTS_IMAGES = True
    SUPPORTS_TOOLS = True

    DEFAULT_MODELS = {
        "gpt-5.5": "gpt-5.5",
        "gpt-5.4": "gpt-5.4",
        "gpt-5.4-mini": "gpt-5.4-mini",
        "gpt-5.4-nano": "gpt-5.4-nano",
        "gpt-5": "gpt-5",
        "gpt-5-mini": "gpt-5-mini",
        "gpt-5-nano": "gpt-5-nano",
        "gpt-4.1": "gpt-4.1",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1-nano": "gpt-4.1-nano",
        "o3": "o3",
        "o4-mini": "o4-mini",
    }

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model or "gpt-5.4-mini", base_url)


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

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model or "llama-3.3-70b-versatile", base_url)


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

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model or "mistral-large-latest", base_url)


# ── DeepSeek ──────────────────────────────────────────────────────────────────

class DeepSeekProvider(OpenAICompatibleProvider):
    name = "DeepSeek"
    BASE_URL = "https://api.deepseek.com/v1"

    DEFAULT_MODELS = {
        "deepseek-chat": "deepseek-chat",
        "deepseek-reasoner": "deepseek-reasoner",
    }

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model or "deepseek-chat", base_url)


# ── NVIDIA NIM ────────────────────────────────────────────────────────────────

class NvidiaProvider(OpenAICompatibleProvider):
    name = "NVIDIA NIM"
    BASE_URL = "https://integrate.api.nvidia.com/v1"
    SUPPORTS_TOOLS = True
    SUPPORTS_IMAGES = False
    VISION_MODELS = {
        "microsoft/phi-4-multimodal-instruct",
        "microsoft/phi-3-vision-128k-instruct",
        "nvidia/neva-22b",
    }
    MAX_TOKENS = 16384
    _EXCLUDE_MODEL_IDS = {
        "deepseek-ai/deepseek-v3.1",
        "deepseek-ai/deepseek-r1",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.3-70b-instruct",
    }

    DEFAULT_MODELS = {
        "microsoft/phi-4-multimodal-instruct": "microsoft/phi-4-multimodal-instruct",
        "microsoft/phi-3-vision-128k-instruct": "microsoft/phi-3-vision-128k-instruct",
        "nvidia/neva-22b": "nvidia/neva-22b",
        "qwen/qwen3-coder-480b-a35b-instruct": "qwen/qwen3-coder-480b-a35b-instruct",
        "moonshotai/kimi-k2-instruct-0905": "moonshotai/kimi-k2-instruct-0905",
        "nvidia/nemotron-3-super-120b-a12b": "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3-nano-30b-a3b": "nvidia/nemotron-3-nano-30b-a3b",
        "moonshotai/kimi-k2.5": "moonshotai/kimi-k2.5",
    }

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model or "qwen/qwen3-coder-480b-a35b-instruct", base_url)

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        # For NIM, reinforce tool usage and handle potential tag-based calling
        if tools:
            system += "\n\nIMPORTANT: You have tools available. If you need to inspect files or the environment, use them immediately. Do not explain why you can't."
        
        # If we have tools and we're in a mode that demands them, try to force it
        # Note: We use super().stream_response which handles the actual call
        return super().stream_response(messages, tools, system)

    def _format_messages(self, messages: list[Message], system: str) -> list[dict]:
        # NVIDIA NIM models often handle 'system' role poorly.
        # We merge it into the first user message or latest user message.
        formatted = _format_messages(messages, "") # Get messages without system prompt
        
        # Filter out the empty system message that helper always adds at index 0
        formatted = [m for m in formatted if not (m["role"] == "system" and not m["content"])]
        
        if not system:
            return formatted

        # Merge system prompt into the FIRST user message for maximum context influence
        for msg in formatted:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    msg["content"] = f"SYSTEM INSTRUCTIONS:\n{system}\n\nUSER MESSAGE:\n{msg['content']}"
                elif isinstance(msg["content"], list):
                    msg["content"].insert(0, {"type": "text", "text": f"SYSTEM INSTRUCTIONS:\n{system}"})
                break
        else:
            # If no user message found, add one with just the instructions
            formatted.insert(0, {"role": "user", "content": f"SYSTEM INSTRUCTIONS:\n{system}"})
            
        return formatted

    def _clean_content_delta(self, content: str) -> str:
        # Keep provider-emitted tool JSON visible to Agent._parse_manual_tool_calls.
        # Some NIM/Nemotron responses put callable JSON behind text tags instead of
        # returning OpenAI-compatible `tool_calls`; suppressing it makes execution impossible.
        return content


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
            result.append({"role": msg.role, "content": _format_openai_content(msg.content)})
    return result


def _format_openai_content(content):
    if not isinstance(content, list):
        return str(content)
    parts = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            parts.append({"type": "text", "text": item.get("text", "")})
        elif item.get("type") == "image_url":
            parts.append({"type": "image_url", "image_url": {"url": item.get("url", "")}})
    return parts or ""


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
