"""OpenRouter provider — aggregates many AI models via OpenAI-compatible API."""
import requests

from .openai_provider import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "OpenRouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    DEFAULT_MODELS = {
        # Anthropic
        "anthropic/claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
        "anthropic/claude-opus-4-6": "anthropic/claude-opus-4-6",
        "anthropic/claude-haiku-4-5": "anthropic/claude-haiku-4-5",
        # OpenAI
        "openai/gpt-5.4": "openai/gpt-5.4",
        "openai/gpt-5.4-mini": "openai/gpt-5.4-mini",
        "openai/o3": "openai/o3",
        "openai/o4-mini": "openai/o4-mini",
        # Google
        "google/gemini-3.1-pro": "google/gemini-3.1-pro-preview",
        "google/gemini-3-flash": "google/gemini-3-flash",
        "google/gemini-2.5-pro": "google/gemini-2.5-pro",
        # Meta
        "meta-llama/llama-4-scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        # DeepSeek
        "deepseek/deepseek-chat-v3-0324": "deepseek/deepseek-chat-v3-0324",
        # Mistral
        "mistralai/mistral-large": "mistralai/mistral-large",
        # Free models
        "meta-llama/llama-3.3-70b-instruct:free": "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.5-flash:free": "google/gemini-2.5-flash:free",
    }

    @classmethod
    def fetch_available_models(cls, api_key: str) -> list[str]:
        """Fetch available models from OpenRouter (no auth needed)."""
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            # Return top models sorted by popularity, limit to manageable list
            models = [m["id"] for m in data if m.get("id")]
            # Prioritize major providers
            priority = ["anthropic/", "openai/", "google/", "meta-llama/", "deepseek/", "mistralai/"]
            top = [m for m in models if any(m.startswith(p) for p in priority)]
            return sorted(top)[:50] if top else list(cls.DEFAULT_MODELS.values())
        except Exception:
            return list(cls.DEFAULT_MODELS.values())

    def __init__(self, api_key: str, model: str | None = None):
        super().__init__(
            api_key,
            model or "anthropic/claude-sonnet-4-6",
            base_url=self.BASE_URL,
        )
        # Set OpenRouter-specific headers
        self.client.default_headers = {
            "HTTP-Referer": "https://github.com/afstudy20-gif/Cclaude",
            "X-Title": "CClaude",
        }
