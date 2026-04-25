"""FastAPI web interface for Cclaude — wraps the multi-provider agent."""
import json
import os
import subprocess
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
    content: str | list[dict]


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    provider: str = "openai"
    model: str | None = None
    system: str = "You are a helpful assistant."
    github_repo: str | None = None   # e.g. "owner/repo" — included in system prompt if set
    github_branch: str = "main"
    workspace: str | None = None     # local project directory for file tools
    session_id: str | None = None    # optional session ID for persistence


class WorkspaceRequest(BaseModel):
    workspace: str | None = None


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
    from src.tools.implementations import set_project_root

    workspace = Path(req.workspace).expanduser().resolve() if req.workspace else Path.cwd()
    try:
        workspace.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not access workspace '{req.workspace}': {e}")
    if not workspace.is_dir():
        raise HTTPException(status_code=400, detail=f"Workspace is not a directory: {workspace}")
    set_project_root(str(workspace))

    # Load session history if provided
    session_messages: list[InternalMessage] = []
    if req.session_id:
        try:
            session_messages, _ = load_session(req.session_id)
        except FileNotFoundError:
            pass

    supports_images = getattr(provider, "SUPPORTS_IMAGES", False)
    internal_messages = session_messages + [
        InternalMessage(role=m.role, content=_prepare_message_content(m.content, supports_images))
        for m in req.messages
    ]

    registry = get_default_registry()
    agent = Agent(provider, registry, project_root=str(workspace))
    agent.history = internal_messages[:-1]  # All but last (chat() appends it)

    last_msg = _prepare_message_content(req.messages[-1].content, supports_images) if req.messages else ""

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


def _message_text(content: str | list[dict]) -> str:
    if isinstance(content, str):
        return content
    texts = [str(item.get("text", "")) for item in content if isinstance(item, dict) and item.get("type") == "text"]
    return "\n".join(t for t in texts if t).strip()


def _prepare_message_content(content: str | list[dict], supports_images: bool):
    if isinstance(content, str):
        return content
    if supports_images:
        return content
    text = _message_text(content)
    images = [item for item in content if isinstance(item, dict) and item.get("type") == "image_url"]
    if images:
        text += f"\n\n[Attached {len(images)} image(s), but the selected provider cannot inspect images.]"
    return text.strip()


class PushRequest(BaseModel):
    workspace: str | None = None
    remote: str = "origin"
    branch: str | None = None
    force: bool = False


class TerminalRunRequest(BaseModel):
    command: str
    workspace: str | None = None
    timeout: int = 120


@app.post("/git/push")
def push_to_github(req: PushRequest):
    """Push the selected local workspace to a GitHub remote."""
    from src.tools.git_tools import git_push
    from src.tools.implementations import set_project_root

    workspace = Path(req.workspace).expanduser().resolve() if req.workspace else Path.cwd()
    if not workspace.is_dir():
        raise HTTPException(status_code=400, detail=f"Workspace is not a directory: {workspace}")
    if not (workspace / ".git").exists():
        raise HTTPException(status_code=400, detail=f"Workspace is not a git repository: {workspace}")

    set_project_root(str(workspace))
    output = git_push(remote=req.remote, branch=req.branch or "", force=req.force)
    return {"ok": not output.lower().startswith("error:"), "output": output}


@app.post("/terminal/run")
def run_terminal_command(req: TerminalRunRequest):
    """Run a shell command in the selected local workspace."""
    command = req.command.strip()
    if not command:
        raise HTTPException(status_code=400, detail="Command is required")

    workspace = Path(req.workspace).expanduser().resolve() if req.workspace else Path.cwd()
    if not workspace.is_dir():
        raise HTTPException(status_code=400, detail=f"Workspace is not a directory: {workspace}")

    timeout = max(1, min(req.timeout, 120))
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=workspace,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as e:
        output = ((e.stdout or "") + (e.stderr or "")).strip()
        return {
            "ok": False,
            "exit_code": None,
            "output": output or f"Command timed out after {timeout}s",
            "timed_out": True,
        }

    output = (result.stdout or "") + (result.stderr or "")
    return {
        "ok": result.returncode == 0,
        "exit_code": result.returncode,
        "output": output.strip() or "(no output)",
        "timed_out": False,
    }


class KeyRequest(BaseModel):
    provider: str
    key: str


class ModelUpdateRequest(BaseModel):
    provider: str | None = None


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


@app.post("/config/browse")
def browse_folder():
    """Open a native folder picker on the host machine (Mac-only helper)."""
    import subprocess
    import os
    
    # Use AppleScript to pick a folder on Mac
    script = 'POSIX path of (choose folder with prompt "Select Workspace")'
    try:
        proc = subprocess.run(['osascript', '-e', script], capture_output=True, text=True, timeout=60)
        if proc.returncode == 0:
            path = proc.stdout.strip()
            return {"path": path}
        return {"path": None, "error": proc.stderr.strip() or "Cancelled"}
    except Exception as e:
        return {"path": None, "error": str(e)}


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
        "nvidia": "NVIDIA_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "tavily": "TAVILY_API_KEY",
    }
    return {p: (keys.get(p) or os.environ.get(env_map.get(p, ""))) is not None for p in PROVIDERS}


@app.post("/git/push")
def git_push(req: WorkspaceRequest):
    """Stage, commit, and push changes to GitHub."""
    import subprocess
    import os
    from pathlib import Path

    ws_path = Path(req.workspace).expanduser().resolve() if req.workspace else Path.cwd()
    if not ws_path.exists() or not ws_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid workspace path")

    def run_cmd(args):
        return subprocess.run(args, cwd=str(ws_path), capture_output=True, text=True, timeout=60)

    try:
        # Check if it's a git repo
        if not (ws_path / ".git").exists():
            # Initialize if not
            run_cmd(["git", "init"])
            run_cmd(["git", "remote", "add", "origin", "https://github.com/afstudy20-gif/Cclaude.git"])
        
        # Add all
        run_cmd(["git", "add", "."])
        
        # Commit (only if there are changes)
        check = run_cmd(["git", "status", "--porcelain"])
        if not check.stdout.strip():
            return {"ok": True, "output": "No changes to commit"}
            
        commit_msg = "Update from CClaude assistant"
        run_cmd(["git", "commit", "-m", commit_msg])
        
        # Push
        # We try to push to current branch, usually main or master
        res = run_cmd(["git", "branch", "--show-current"])
        branch = res.stdout.strip() or "main"
        
        proc = run_cmd(["git", "push", "origin", branch])
        if proc.returncode == 0:
            return {"ok": True, "output": proc.stdout or f"Successfully pushed to GitHub ({branch})"}
        else:
            # Maybe need to set upstream
            run_cmd(["git", "push", "--set-upstream", "origin", branch])
            return {"ok": False, "output": proc.stderr or "Push failed"}

    except Exception as e:
        return {"ok": False, "output": str(e)}


@app.get("/config/keys/status")
def get_keys_status():
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
        "nvidia": "NVIDIA_API_KEY",
        "cohere": "COHERE_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "tavily": "TAVILY_API_KEY",
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


def _update_model_cache(provider_name: str | None = None) -> dict:
    """Fetch live model lists and update the in-memory cache."""
    global _live_models
    config = load_config()
    result = {}
    seen_classes = set()
    target = provider_name.lower() if provider_name else None

    for alias, cls in PROVIDERS.items():
        if target and alias != target:
            continue
        if cls in seen_classes:
            continue
        seen_classes.add(cls)

        api_key = get_api_key(alias, config)
        if not api_key and alias not in ("ollama", "local"):
            models = list(cls.DEFAULT_MODELS.values())
            result[alias] = {
                "models": models,
                "count": len(models),
                "source": "defaults",
                "message": "No API key configured; using built-in defaults.",
            }
            continue

        try:
            live = cls.fetch_available_models(api_key or "ollama")
            result[alias] = {
                "models": live,
                "count": len(live),
                "source": "live",
                "message": "Fetched from provider API.",
            }
        except Exception as e:
            models = list(cls.DEFAULT_MODELS.values())
            result[alias] = {
                "models": models,
                "count": len(models),
                "source": "defaults",
                "message": f"Provider fetch failed; using built-in defaults: {str(e)[:160]}",
            }

    _live_models.update({provider: info["models"] for provider, info in result.items()})
    return result


@app.post("/models/refresh")
def refresh_models():
    """Fetch live model lists from each configured provider API."""
    return {
        provider: info["models"]
        for provider, info in _update_model_cache().items()
    }


@app.post("/models/update")
def update_models(req: ModelUpdateRequest):
    """Update cached model lists for one provider or all providers."""
    if req.provider:
        key = req.provider.lower()
        if key not in PROVIDERS:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {req.provider}")
        updated = _update_model_cache(key)
    else:
        updated = _update_model_cache()

    return {
        "updated": True,
        "providers": updated,
    }


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
            "supports_tools": getattr(cls, "SUPPORTS_TOOLS", True),
            "supports_images": getattr(cls, "SUPPORTS_IMAGES", False),
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
