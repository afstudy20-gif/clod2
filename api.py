"""FastAPI web interface for Cclaude — wraps the multi-provider agent."""
import json
import os
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.core.config import get_api_key, load_config
from src.providers import get_provider, PROVIDERS

app = FastAPI(title="Cclaude API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / response models ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    provider: str = "openai"
    model: str | None = None
    system: str = "You are a helpful assistant."


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/providers")
def list_providers():
    """Return available providers and their default models."""
    seen = set()
    result = {}
    for alias, cls in PROVIDERS.items():
        if cls in seen:
            continue
        seen.add(cls)
        result[alias] = {
            "name": cls.name if isinstance(cls.name, str) else alias,
            "models": list(cls.DEFAULT_MODELS.keys()),
        }
    return result


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

    # Convert to internal Message format
    from src.providers.base import Message as InternalMessage

    internal_messages = [
        InternalMessage(role=m.role, content=m.content)
        for m in req.messages
    ]

    def generate():
        try:
            for chunk in provider.stream_response(internal_messages, [], req.system):
                text = str(chunk)
                yield f"data: {json.dumps({'text': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/config/key")
def set_key(provider: str, key: str):
    """Save an API key to the config file."""
    from src.core.config import set_api_key
    set_api_key(provider, key)
    return {"saved": True, "provider": provider}
