"""OpenRouter provider — aggregates many AI models via OpenAI-compatible API."""
from .openai_provider import OpenAICompatibleProvider


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "OpenRouter"
    BASE_URL = "https://openrouter.ai/api/v1"

    DEFAULT_MODELS = {
        # Anthropic
        "anthropic/claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
        "anthropic/claude-haiku-4-5": "anthropic/claude-haiku-4-5",
        # OpenAI
        "openai/gpt-4.1": "openai/gpt-4.1",
        "openai/gpt-4.1-mini": "openai/gpt-4.1-mini",
        "openai/o3": "openai/o3",
        "openai/o4-mini": "openai/o4-mini",
        # Google
        "google/gemini-2.5-pro": "google/gemini-2.5-pro-preview-05-06",
        "google/gemini-2.5-flash": "google/gemini-2.5-flash-preview-04-17",
        # Meta
        "meta-llama/llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct",
        # DeepSeek
        "deepseek/deepseek-chat-v3-0324": "deepseek/deepseek-chat-v3-0324",
        # Mistral
        "mistralai/mistral-large": "mistralai/mistral-large",
        # Free models
        "meta-llama/llama-3.3-70b-instruct:free": "meta-llama/llama-3.3-70b-instruct:free",
        "google/gemini-2.5-flash-preview-04-17:free": "google/gemini-2.5-flash-preview-04-17:free",
    }

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
