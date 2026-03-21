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
    }
    env_var = env_map.get(provider.lower())
    if env_var:
        val = os.environ.get(env_var)
        if val:
            return val

    if config:
        keys = config.get("api_keys", {})
        return keys.get(provider.lower())
    return None


def set_api_key(provider: str, key: str):
    """Persist an API key to config file."""
    config = load_config()
    config.setdefault("api_keys", {})[provider.lower()] = key
    save_config(config)
    print(f"Saved API key for {provider} to {CONFIG_FILE}")
