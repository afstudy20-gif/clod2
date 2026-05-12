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
    API_NAME = "openai-completions"
    CONTEXT_WINDOW = 128_000
    MAX_OUTPUT_TOKENS = 4096

    # Model ID substrings to exclude when fetching live models
    _EXCLUDE_PATTERNS: list[str] = ["embed", "tts", "whisper", "dall-e", "moderation", "audio", "realtime", "transcri"]

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model)
        self._suppress_content = False
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url or self.BASE_URL,
            timeout=180.0,  # Increased to 3 minutes for slower NIM models
        )

    _model_cache: dict[str, tuple[float, list[str]]] = {}
    _MODEL_CACHE_TTL = 86_400.0  # 24h

    @classmethod
    def fetch_available_models(cls, api_key: str) -> list[str]:
        """Fetch models from the OpenAI-compatible /v1/models endpoint with 24h cache."""
        import time as _time
        base = cls.BASE_URL or "https://api.openai.com/v1"
        cache_key = f"{cls.__name__}:{base}:{api_key[:8]}"
        cached = cls._model_cache.get(cache_key)
        if cached and (_time.time() - cached[0]) < cls._MODEL_CACHE_TTL:
            return cached[1]
        try:
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
            result = ids if ids else list(cls.DEFAULT_MODELS.values())
        except Exception:
            if cached:
                return cached[1]
            result = list(cls.DEFAULT_MODELS.values())
        cls._model_cache[cache_key] = (_time.time(), result)
        return result

    # Per-model capability hooks. Subclasses override these to mark certain
    # models as not supporting tools or as having a lower max_tokens cap.
    NO_TOOL_PATTERNS: tuple[str, ...] = ()
    MAX_TOKENS_BY_PATTERN: dict[str, int] = {}

    def _model_supports_tools(self, model: str) -> bool:
        if not getattr(self, "SUPPORTS_TOOLS", True):
            return False
        m = (model or "").lower()
        return not any(p in m for p in self.NO_TOOL_PATTERNS)

    def _model_max_tokens(self, model: str) -> int:
        m = (model or "").lower()
        for pat, mx in self.MAX_TOKENS_BY_PATTERN.items():
            if pat in m:
                return mx
        return getattr(self, "MAX_TOKENS", 4096)

    def stream_response(
        self,
        messages: list[Message],
        tools: list[dict],
        system: str,
    ) -> Iterator[str | ToolCall]:
        formatted = self._format_messages(messages, system)
        model_supports_tools = self._model_supports_tools(self.model)
        openai_tools = self._convert_tools(tools) if model_supports_tools else []
        self._suppress_content = False

        model_lower = self.model.lower()
        reasoning_markers = getattr(self, "REASONING_MODEL_MARKERS", ())
        is_openai_reasoning = any(m in model_lower for m in ("o1", "o3", "o4", "gpt-5"))
        is_provider_reasoning = bool(reasoning_markers) and any(m in model_lower for m in reasoning_markers)

        if is_openai_reasoning:
            temperature = 1.0
        elif is_provider_reasoning:
            temperature = 0.6
        else:
            temperature = 0.0

        kwargs: dict = {
            "model": self.model,
            "messages": formatted,
            "stream": True,
            "temperature": temperature,
        }

        max_tokens = self._model_max_tokens(self.model)
        if is_openai_reasoning:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
        if openai_tools:
            kwargs["tools"] = openai_tools
            kwargs["tool_choice"] = "auto"

        stream = self._open_stream_with_retry(kwargs)
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
                    yield (
                        "\n\n[warning: output truncated at max_tokens limit. "
                        "Partial response above. Retry with shorter prompt or larger-output model.]"
                    )
                if finish == "content_filter":
                    raise RuntimeError("Model output was blocked by the provider content filter.")
        except RateLimitError as exc:
            raise RuntimeError(self._format_rate_limit_error(exc)) from exc
        except APIStatusError as exc:
            if exc.status_code == 429:
                raise RuntimeError(self._format_rate_limit_error(exc)) from exc
            if exc.status_code == 404:
                raise RuntimeError(self._format_not_found_error(exc)) from exc
            raise
        except httpx.RemoteProtocolError as exc:
            raise RuntimeError(self._format_stream_interrupted_error(exc)) from exc
        except NotFoundError as exc:
            raise RuntimeError(self._format_not_found_error(exc)) from exc

        for tc in tool_calls_map.values():
            try:
                args = json.loads(tc["args"]) if tc["args"] else {}
            except json.JSONDecodeError:
                args = {"raw": tc["args"]}
            yield ToolCall(id=tc["id"], name=tc["name"], arguments=args)

    def _open_stream_with_retry(self, kwargs: dict):
        """Open the streaming chat completion, auto-recovering from 400/422
        errors that complain about tool_choice/tools/max_tokens.

        Many OpenAI-compatible endpoints (NVIDIA NIM gemma/qwen, etc.) reject
        `tools=`/`tool_choice="auto"` when the served model wasn't started with
        tool-calling enabled, or reject `max_tokens > 4096`. Rather than
        bubbling a confusing error, we strip the offending fields and retry
        once. The user still gets a useful answer; tool calling is silently
        disabled for that one model.
        """
        attempts = 0
        last: Exception | None = None
        while attempts < 3:
            try:
                return self.client.chat.completions.create(**kwargs)
            except RateLimitError as exc:
                raise RuntimeError(self._format_rate_limit_error(exc)) from exc
            except APIStatusError as exc:
                if exc.status_code == 429:
                    raise RuntimeError(self._format_rate_limit_error(exc)) from exc
                if exc.status_code == 404:
                    raise RuntimeError(self._format_not_found_error(exc)) from exc
                if exc.status_code in (400, 422):
                    msg = ""
                    try:
                        msg = str(getattr(exc, "message", "") or exc)
                    except Exception:
                        msg = str(exc)
                    msg_l = msg.lower()
                    mutated = False
                    if any(s in msg_l for s in (
                        "tool_choice", "tools", "tool use", "auto tool choice",
                        "tool-call-parser", "enable-auto-tool-choice",
                    )):
                        if kwargs.pop("tools", None) is not None:
                            mutated = True
                        if kwargs.pop("tool_choice", None) is not None:
                            mutated = True
                    if "max_tokens" in msg_l or "max_completion_tokens" in msg_l:
                        # NVIDIA gateway caps some models at <= 4096 even
                        # when the model itself accepts more.
                        for key in ("max_tokens", "max_completion_tokens"):
                            cur = kwargs.get(key)
                            if isinstance(cur, int) and cur > 2048:
                                kwargs[key] = 2048
                                mutated = True
                    if mutated:
                        attempts += 1
                        last = exc
                        continue
                raise
            attempts += 1
        if last:
            raise last
        return self.client.chat.completions.create(**kwargs)

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
            retry_after = response.headers.get("retry-after") or response.headers.get("Retry-After")
        if retry_after:
            try:
                wait = float(retry_after)
                import time as _time
                _time.sleep(min(wait, 30.0))
            except (TypeError, ValueError):
                pass
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
                "Use Update Models to refresh the live catalog, or switch to meta/llama-3.3-70b-instruct, "
                "nvidia/llama-3.3-nemotron-super-49b-v1, qwen/qwen2.5-coder-32b-instruct, or deepseek-ai/deepseek-r1."
            )
        return f"Provider model/function for '{self.model}' was not found (HTTP 404)."

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
    API_NAME = "openai-completions"
    REASONING_MODEL_MARKERS = ("o3", "o4", "gpt-5")

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
    CONTEXT_WINDOW = 131_072

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
    CONTEXT_WINDOW = 128_000

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
    CONTEXT_WINDOW = 64_000
    REASONING_MODEL_MARKERS = ("reasoner", "r1")

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
        "meta/llama-3.2-90b-vision-instruct",
        "meta/llama-3.2-11b-vision-instruct",
        "microsoft/phi-4-multimodal-instruct",
    }
    MAX_TOKENS = 16384
    CONTEXT_WINDOW = 128_000
    MAX_OUTPUT_TOKENS = 16_384
    REASONING_MODEL_MARKERS = ("reasoning", "r1")

    # Models served on NVIDIA NIM without tool-calling enabled. Sending
    # `tools=`/`tool_choice="auto"` to these returns 400 / 422. Match by
    # substring so future variants (e.g. gemma-3-27b-it) are covered.
    NO_TOOL_PATTERNS = (
        "gemma",                  # google/gemma-3-*-it, codegemma, recurrentgemma
        "qwen2.5-coder",          # explicitly errors "Tool use has not been enabled"
        "phi-4-multimodal",       # vision-only
    )

    # NVIDIA gateway caps several gemma deployments at <= 4096. Stay under it.
    MAX_TOKENS_BY_PATTERN = {
        "gemma": 2048,
        "codegemma": 2048,
        "phi-4": 4096,
    }

    DEFAULT_MODELS = {
        "meta/llama-3.3-70b-instruct": "meta/llama-3.3-70b-instruct",
        "meta/llama-3.1-405b-instruct": "meta/llama-3.1-405b-instruct",
        "meta/llama-3.2-90b-vision-instruct": "meta/llama-3.2-90b-vision-instruct",
        "nvidia/llama-3.1-nemotron-70b-instruct": "nvidia/llama-3.1-nemotron-70b-instruct",
        "nvidia/llama-3.3-nemotron-super-49b-v1": "nvidia/llama-3.3-nemotron-super-49b-v1",
        "nvidia/llama-3.1-nemotron-ultra-253b-v1": "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "qwen/qwen2.5-coder-32b-instruct": "qwen/qwen2.5-coder-32b-instruct",
        "qwen/qwen3-coder-480b-a35b-instruct": "qwen/qwen3-coder-480b-a35b-instruct",
        "deepseek-ai/deepseek-r1": "deepseek-ai/deepseek-r1",
        "mistralai/mixtral-8x22b-instruct-v0.1": "mistralai/mixtral-8x22b-instruct-v0.1",
        "microsoft/phi-4-multimodal-instruct": "microsoft/phi-4-multimodal-instruct",
    }

    def __init__(self, api_key: str, model: str | None = None, base_url: str | None = None):
        super().__init__(api_key, model or "meta/llama-3.3-70b-instruct", base_url)

    def _clean_content_delta(self, content: str) -> str:
        # Strip <think>...</think> reasoning blocks emitted by Nemotron/DeepSeek-R1.
        # Keep tool JSON visible for Agent._parse_manual_tool_calls fallback.
        if not content:
            return content
        if self._suppress_content:
            if "</think>" in content:
                self._suppress_content = False
                content = content.split("</think>", 1)[1]
            else:
                return ""
        if "<think>" in content:
            before, _, rest = content.partition("<think>")
            if "</think>" in rest:
                _, _, after = rest.partition("</think>")
                content = before + after
            else:
                self._suppress_content = True
                content = before
        return content


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(OpenAICompatibleProvider):
    name = "Ollama (local)"
    BASE_URL = "http://localhost:11434/v1"
    CONTEXT_WINDOW = 128_000

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
