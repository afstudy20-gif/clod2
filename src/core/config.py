"""Configuration loading from env, .env file, and config.json."""
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load .env if present

CONFIG_FILE = Path.home() / ".cclaude" / "config.json"


def load_config() -> dict:
    """Load config from file, falling back to empty dict."""
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text())
        except Exception:
            pass
    return {}


def save_config(config: dict):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps(config, indent=2))


def get_api_key(provider: str, config: dict | None = None) -> str | None:
    """Get API key for a provider from env vars or config file."""
    env_map = {
        "claude": "ANTHROPIC_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "chatgpt": "OPENAI_API_KEY",
        "gpt": "OPENAI_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "google": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        # Ollama is local — no key needed
        "ollama": None,
        "local": None,
    }
    key_lower = provider.lower()
    env_var = env_map.get(key_lower)

    # Ollama is local — return a placeholder key so the provider can init
    if key_lower in ("ollama", "local"):
        return "ollama"

    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val

    if config:
        keys = config.get("api_keys", {})
        return keys.get(key_lower)
    return None


def set_api_key(provider: str, key: str):
    """Persist an API key to config file."""
    config = load_config()
    config.setdefault("api_keys", {})[provider.lower()] = key
    save_config(config)
    print(f"Saved API key for {provider} to {CONFIG_FILE}")


def get_last_model(provider: str) -> str | None:
    """Get the last-used model for a provider."""
    config = load_config()
    return config.get("last_models", {}).get(provider.lower())


def set_last_model(provider: str, model: str):
    """Persist the last-used model for a provider."""
    config = load_config()
    config.setdefault("last_models", {})[provider.lower()] = model
    save_config(config)
