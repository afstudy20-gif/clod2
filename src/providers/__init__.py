from .base import BaseProvider, Message, ToolCall, ToolEvent, ToolResult
from .anthropic_provider import AnthropicProvider
from .openai_provider import OpenAIProvider, GroqProvider, MistralProvider, DeepSeekProvider, NvidiaProvider, OllamaProvider
from .gemini_provider import GeminiProvider
from .cohere_provider import CohereProvider
from .openrouter_provider import OpenRouterProvider

PROVIDERS = {
    # Anthropic Claude
    "claude": AnthropicProvider,
    "anthropic": AnthropicProvider,

    # OpenAI
    "openai": OpenAIProvider,
    "chatgpt": OpenAIProvider,
    "gpt": OpenAIProvider,

    # Google Gemini
    "gemini": GeminiProvider,
    "google": GeminiProvider,

    # Groq (fast inference, free tier)
    "groq": GroqProvider,

    # Mistral AI
    "mistral": MistralProvider,

    # DeepSeek (very cheap)
    "deepseek": DeepSeekProvider,

    # NVIDIA NIM / build.nvidia.com
    "nvidia": NvidiaProvider,
    "nim": NvidiaProvider,

    # Ollama (local, free)
    "ollama": OllamaProvider,
    "local": OllamaProvider,

    # Cohere
    "cohere": CohereProvider,

    # OpenRouter (aggregator — many providers via one API)
    "openrouter": OpenRouterProvider,
}


def get_provider(name: str, **kwargs) -> BaseProvider:
    key = name.lower()
    if key not in PROVIDERS:
        available = ", ".join(sorted(set(PROVIDERS.keys())))
        raise ValueError(f"Unknown provider: '{name}'. Available: {available}")
    return PROVIDERS[key](**kwargs)
