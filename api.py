"""FastAPI web interface for Cclaude — wraps the multi-provider agent."""
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.core.config import get_api_key, load_config
from src.core.session import list_sessions, load_session, save_session, delete_session
from src.providers import get_provider, PROVIDERS

app = FastAPI(title="Cclaude API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (the web UI)
_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


# ── Request / response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    provider: str = "openai"
    model: str | None = None
    system: str = "You are a helpful assistant."
    github_repo: str | None = None   # e.g. "owner/repo" — included in system prompt if set
    github_branch: str = "main"
    workspace: str = "/workspace"    # local scratch directory inside the container
    session_id: str | None = None    # optional session ID for persistence


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index = Path(__file__).parent / "static" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"name": "Cclaude API", "version": "0.1.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    """Stream a chat response as Server-Sent Events."""
    config = load_config()
    api_key = get_api_key(req.provider, config)

    if not api_key and req.provider.lower() not in ("ollama", "local"):
        raise HTTPException(
            status_code=400,
            detail=f"No API key found for provider '{req.provider}'. "
                   f"Set the env var or use /config to save it.",
        )

    try:
        provider = get_provider(req.provider, api_key=api_key, model=req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from src.core.agent import Agent
    from src.providers.base import Message as InternalMessage, ToolEvent
    from src.tools.registry import get_default_registry

    # Load session history if provided
    session_messages: list[InternalMessage] = []
    if req.session_id:
        try:
            session_messages, _ = load_session(req.session_id)
        except FileNotFoundError:
            pass

    internal_messages = session_messages + [
        InternalMessage(role=m.role, content=m.content)
        for m in req.messages
    ]

    registry = get_default_registry()
    agent = Agent(provider, registry)
    agent.history = internal_messages[:-1]  # All but last (chat() appends it)

    last_msg = req.messages[-1].content if req.messages else ""

    def generate():
        try:
            for chunk in agent.chat(last_msg):
                if isinstance(chunk, ToolEvent):
                    yield f"data: {json.dumps({'tool_event': {'type': chunk.type, 'name': chunk.tool_name, 'result': (chunk.result or '')[:2000]}})}\n\n"
                else:
                    yield f"data: {json.dumps({'text': str(chunk)})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Auto-save session if session_id provided
        if req.session_id:
            save_session(req.session_id, agent.history, req.provider, req.model or "")

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


class KeyRequest(BaseModel):
    provider: str
    key: str


@app.post("/config/key")
def set_key(req: KeyRequest):
    """Save an API key to the config file."""
    from src.core.config import set_api_key
    set_api_key(req.provider, req.key)
    return {"saved": True, "provider": req.provider}


@app.post("/config/key/test")
def test_key(req: KeyRequest):
    """Test an API key by listing models (fast, no generation needed)."""
    import concurrent.futures

    cls = PROVIDERS.get(req.provider.lower())
    if not cls:
        return {"valid": False, "provider": req.provider, "error": "Unknown provider"}

    def _test():
        return cls.fetch_available_models(req.key)

    try:
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(_test)
            models = future.result(timeout=15)  # Hard 15s timeout
        if models:
            return {"valid": True, "provider": req.provider, "models_found": len(models)}
        return {"valid": False, "provider": req.provider, "error": "No models returned"}
    except concurrent.futures.TimeoutError:
        return {"valid": False, "provider": req.provider, "error": "Timeout — key may be valid but API is slow"}
    except Exception as e:
        return {"valid": False, "provider": req.provider, "error": str(e)[:200]}


@app.get("/config/keys")
def get_saved_keys():
    """Return which providers have saved API keys (without revealing the keys)."""
    config = load_config()
    keys = config.get("api_keys", {})
    import os
    env_map = {
        "openai": "OPENAI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    result = {}
    for p, env_var in env_map.items():
        has_env = bool(os.environ.get(env_var))
        has_saved = bool(keys.get(p))
        result[p] = {
            "configured": has_env or has_saved,
            "source": "env" if has_env else ("saved" if has_saved else None),
        }
    result["ollama"] = {"configured": True, "source": "local"}
    return result


# ── Model refresh ─────────────────────────────────────────────────────────────

# In-memory cache of live model lists
_live_models: dict[str, list[str]] = {}


@app.post("/models/refresh")
def refresh_models():
    """Fetch live model lists from each configured provider API."""
    global _live_models
    config = load_config()
    result = {}
    seen_classes = set()

    for alias, cls in PROVIDERS.items():
        if cls in seen_classes:
            continue
        seen_classes.add(cls)

        api_key = get_api_key(alias, config)
        if not api_key and alias not in ("ollama", "local"):
            # No key → return static defaults
            result[alias] = list(cls.DEFAULT_MODELS.values())
            continue

        try:
            live = cls.fetch_available_models(api_key or "ollama")
            result[alias] = live
        except Exception:
            result[alias] = list(cls.DEFAULT_MODELS.values())

    _live_models = result
    return result


@app.get("/providers")
def list_providers():
    """Return available providers and their models (live if refreshed, else defaults)."""
    seen = set()
    result = {}
    for alias, cls in PROVIDERS.items():
        if cls in seen:
            continue
        seen.add(cls)
        models = _live_models.get(alias, list(cls.DEFAULT_MODELS.values()))
        result[alias] = {
            "name": cls.name if isinstance(cls.name, str) else alias,
            "models": models,
        }
    return result


# ── Session endpoints ────────────────────────────────────────────────────────

@app.get("/sessions")
def get_sessions():
    """List all saved sessions."""
    return list_sessions()


@app.delete("/sessions/{session_id}")
def remove_session(session_id: str):
    """Delete a saved session."""
    if delete_session(session_id):
        return {"deleted": True}
    raise HTTPException(status_code=404, detail="Session not found")
