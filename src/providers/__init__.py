from .base import BaseProvider, Message, ToolCall, ToolResult
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider

PROVIDERS = {
    "claude": AnthropicProvider,
    "anthropic": AnthropicProvider,
    "openai": OpenAIProvider,
    "chatgpt": OpenAIProvider,
    "gpt": OpenAIProvider,
    "gemini": GeminiProvider,
    "google": GeminiProvider,
}

def get_provider(name: str, **kwargs) -> BaseProvider:
    key = name.lower()
    if key not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[key](**kwargs)
