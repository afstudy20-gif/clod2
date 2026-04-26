"""FastAPI web interface for Clod — wraps the multi-provider agent."""
import json
import os
import re
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

app = FastAPI(title="Clod API", version="0.1.0")
SERVER_CHAT_PERSISTENCE = False

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
    toolEvents: list[dict] | None = None


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    provider: str = "openai"
    model: str | None = None
    system: str = "You are a helpful assistant."
    mode: str = "chat"               # "chat" | "build" | "debug" | "explore_plan"
    github_repo: str | None = None   # e.g. "owner/repo" — included in system prompt if set
    github_branch: str = "main"
    workspace: str | None = None     # local project directory for file tools
    session_id: str | None = None    # optional session ID for persistence


class WorkspaceRequest(BaseModel):
    workspace: str | None = None


class ToolRunRequest(BaseModel):
    tool: str
    arguments: dict
    workspace: str | None = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    index = Path(__file__).parent / "static" / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"name": "Clod API", "version": "0.1.0", "docs": "/docs"}


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

    model = _select_available_model(req.provider, req.model, config)

    try:
        provider = get_provider(req.provider, api_key=api_key, model=model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    from src.core.agent import Agent
    from src.providers.base import Message as InternalMessage, ToolEvent
    from src.tools.registry import get_default_registry
    from src.tools.implementations import set_project_root

    try:
        workspace = Path(req.workspace).expanduser().resolve() if req.workspace else Path.cwd()
        workspace.mkdir(parents=True, exist_ok=True)
        set_project_root(str(workspace))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Workspace error: {e}")

    # Web chat persistence is intentionally disabled. The browser UI stores chats
    # in localStorage and /chat must not write conversation history on the server.
    session_messages: list[InternalMessage] = []
    if SERVER_CHAT_PERSISTENCE and req.session_id:
        try:
            session_messages, _ = load_session(req.session_id)
        except Exception as e:
            print(f"Warning: could not load session {req.session_id}: {e}")

    try:
        supports_images = _model_supports_images(provider, model)
        internal_messages = session_messages + [
            InternalMessage(role=m.role, content=_prepare_message_content(m.content, supports_images))
            for m in req.messages
        ]

        registry = get_default_registry()
        agent = Agent(provider, registry, project_root=str(workspace))
        agent.history = internal_messages[:-1]  # All but last (chat() appends it)

        last_msg = _prepare_message_content(req.messages[-1].content, supports_images) if req.messages else ""
        if _is_resume_command(last_msg):
            last_msg = _build_resume_message(internal_messages[:-1], req.messages[:-1])
        last_msg = _apply_chat_mode(last_msg, req.mode)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Setup error: {e}")

    def generate():
        try:
            for chunk in agent.chat(last_msg):
                if isinstance(chunk, ToolEvent):
                    yield f"data: {json.dumps({'tool_event': {'type': chunk.type, 'name': chunk.tool_name, 'arguments': chunk.arguments, 'result': (chunk.result or '')[:2000]}})}\n\n"
                else:
                    yield f"data: {json.dumps({'text': str(chunk)})}\n\n"
        except Exception as e:
            import traceback
            traceback.print_exc()  # Log to server console
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

        # Server-side web chat persistence is disabled by default.
        if SERVER_CHAT_PERSISTENCE and req.session_id:
            save_session(req.session_id, agent.history, req.provider, model or "")

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _apply_chat_mode(message, mode: str):
    """Convert the web UI mode selector into the same slash commands as the CLI."""
    normalized = (mode or "chat").strip().lower()
    if normalized == "build_debug":
        normalized = _infer_build_or_debug_mode(message)
    if normalized in {"explore", "plan"}:
        normalized = "explore_plan"
    if normalized not in {"build", "debug", "explore_plan"}:
        return message
    if not isinstance(message, str):
        return message
    stripped = message.lstrip()
    if stripped.startswith(("/build", "/debug", "/explore", "/plan")):
        return message
    if normalized == "explore_plan":
        return f"/explore {message}"
    return f"/{normalized} {message}"


def _is_resume_command(message) -> bool:
    """Detect short continuation commands after an interrupted/unfinished task."""
    if not isinstance(message, str):
        return False
    normalized = re.sub(r"[\s.!?]+", " ", message.strip().lower()).strip()
    return normalized in {
        "resume",
        "go",
        "go on",
        "continue",
        "continue please",
        "finish",
        "go on and finish",
        "ok",
        "okay",
        "tamam",
        "devam",
        "devam et",
        "bitir",
    }


def _build_resume_message(history: list, chat_messages: list[ChatMessage] | None = None) -> str:
    previous_user = ""
    recent_context: list[str] = []

    for msg in reversed(history):
        if msg.role == "user":
            text = _message_text(msg.content).strip()
            if text and not _is_resume_command(text):
                previous_user = text
                break

    for msg in history[-12:]:
        if msg.role == "assistant" and msg.tool_calls:
            for call in msg.tool_calls[-3:]:
                recent_context.append(f"tool_start {call.name}: {json.dumps(call.arguments, ensure_ascii=False)[:500]}")
        elif msg.role == "tool" and msg.tool_results:
            for result in msg.tool_results[-3:]:
                status = "error" if result.is_error else "ok"
                recent_context.append(f"tool_result {status}: {result.content[:700]}")

    if chat_messages:
        for msg in chat_messages[-8:]:
            if msg.role != "assistant" or not msg.toolEvents:
                continue
            for event in msg.toolEvents[-10:]:
                name = event.get("name", "tool")
                event_type = event.get("type", "")
                args = event.get("arguments")
                result = str(event.get("result", ""))
                if event_type == "start":
                    recent_context.append(f"ui_tool_start {name}: {json.dumps(args, ensure_ascii=False)[:500]}")
                elif event_type == "result":
                    recent_context.append(f"ui_tool_result {name}: {result[:1000]}")

    context_text = "\n".join(recent_context[-16:]) or "(No recent tool details were saved.)"
    previous_text = previous_user or "(Could not identify the previous user task from history.)"
    return (
        "Continue the previous unfinished task. Do not restart from scratch unless the existing files are missing. "
        "First inspect the recent tool results and current files, then perform the next needed action. "
        "If the previous task requested a final artifact or exact output, verify it and provide that final result. "
        "If the previous task failed, fix the failure and continue. Do not stop after only saying you resumed.\n\n"
        f"Previous user task:\n{previous_text}\n\n"
        f"Recent tool context:\n{context_text}"
    )


def _infer_build_or_debug_mode(message) -> str:
    """Use the stricter /debug path for bug-fix requests, otherwise /build."""
    if not isinstance(message, str):
        return "build"
    lowered = message.lower()
    debug_markers = (
        "/debug",
        "debug",
        "bug",
        "error",
        "exception",
        "traceback",
        "stack trace",
        "failed",
        "failing",
        "failure",
        "fix",
        "crash",
        "broken",
        "not working",
        "doesn't work",
        "division by zero",
        "zerodivisionerror",
    )
    return "debug" if any(marker in lowered for marker in debug_markers) else "build"


def _select_available_model(provider: str, model: str | None, config: dict | None = None) -> str | None:
    """Choose a currently available model, falling back automatically if needed."""
    normalized = _normalize_requested_model(provider, model)
    key = provider.lower()
    cls = PROVIDERS.get(key)
    if not cls or not normalized:
        return normalized

    cached = _live_models.get(key)
    if cached:
        return normalized if normalized in cached else _first_model(cached, cls)

    if model and normalized != model:
        return _first_model(list(cls.DEFAULT_MODELS.values()), cls)

    api_key = get_api_key(key, config or load_config())
    if not api_key and key not in ("ollama", "local"):
        return normalized

    try:
        live = cls.fetch_available_models(api_key or "ollama")
    except Exception:
        return normalized

    if live:
        _live_models[key] = live
        return normalized if normalized in live else _first_model(live, cls)

    return normalized


def _first_model(models: list[str], cls) -> str | None:
    if models:
        return models[0]
    defaults = list(cls.DEFAULT_MODELS.values())
    return defaults[0] if defaults else None


def _normalize_requested_model(provider: str, model: str | None) -> str | None:
    """Avoid sending known-stale model ids to providers."""
    if provider.lower() not in {"nvidia", "nim"} or not model:
        return model

    stale_nvidia_models = {
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.3-70b-instruct",
        "deepseek-ai/deepseek-v3.1",
        "deepseek-ai/deepseek-r1",
    }
    if model in stale_nvidia_models:
        return None
    return model


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


def _model_supports_images(provider, model: str | None) -> bool:
    vision_models = getattr(provider, "VISION_MODELS", None)
    if vision_models is not None:
        return bool(model and model in vision_models)
    return getattr(provider, "SUPPORTS_IMAGES", False)


class PushRequest(BaseModel):
    workspace: str | None = None
    remote: str = "origin"
    branch: str | None = None
    force: bool = False
    commit_message: str = "Update from Clod assistant"


class TerminalRunRequest(BaseModel):
    command: str
    workspace: str | None = None
    timeout: int = 120


@app.post("/git/push")
def push_to_github(req: PushRequest):
    """Stage, commit, and push changes to GitHub."""
    import subprocess
    from pathlib import Path
    
    ws_path = Path(req.workspace).expanduser().resolve() if req.workspace else Path.cwd()
    if not ws_path.exists() or not ws_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid workspace path")

    def run_cmd(args):
        return subprocess.run(args, cwd=str(ws_path), capture_output=True, text=True, timeout=60)

    try:
        # Check if it's a git repo
        if not (ws_path / ".git").exists():
            run_cmd(["git", "init"])
            # If no remote is provided, we can't push, but we can at least init
            if not req.remote.startswith("http"):
                return {"ok": False, "output": "Not a git repository and no remote URL provided."}
            run_cmd(["git", "remote", "add", "origin", req.remote])
        
        # Add all
        run_cmd(["git", "add", "."])
        
        # Commit (only if there are changes)
        check = run_cmd(["git", "status", "--porcelain"])
        if not check.stdout.strip():
            return {"ok": True, "output": "No changes to commit"}
            
        run_cmd(["git", "commit", "-m", req.commit_message])
        
        # Push
        res = run_cmd(["git", "branch", "--show-current"])
        branch = req.branch or res.stdout.strip() or "main"
        
        proc = run_cmd(["git", "push", req.remote, branch, "--force" if req.force else ""])
        if proc.returncode == 0:
            return {"ok": True, "output": proc.stdout or f"Successfully pushed to GitHub ({branch})"}
        else:
            return {"ok": False, "output": proc.stderr or "Push failed"}

    except Exception as e:
        return {"ok": False, "output": str(e)}


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
    """Open a native folder picker when available on the host machine."""
    import subprocess
    import platform
    import shutil

    system = platform.system().lower()
    if system != "darwin":
        return {
            "path": None,
            "unsupported": True,
            "cwd": str(Path.cwd()),
            "home": str(Path.home()),
            "error": (
                "Native folder browsing is only available when Clod runs locally on macOS. "
                "Enter the workspace path manually on this hosted/Linux deployment."
            ),
        }
    if not shutil.which("osascript"):
        return {
            "path": None,
            "unsupported": True,
            "cwd": str(Path.cwd()),
            "home": str(Path.home()),
            "error": "macOS folder picker is unavailable because osascript was not found.",
        }
    
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


@app.post("/tools/run")
def run_tool_manually(req: ToolRunRequest):
    """Execute a tool manually from the UI."""
    from src.tools.registry import get_default_registry
    from src.tools.implementations import set_project_root
    
    workspace = Path(req.workspace).expanduser().resolve() if req.workspace else Path.cwd()
    set_project_root(str(workspace))
    
    registry = get_default_registry()
    try:
        result = registry.execute(req.tool, req.arguments)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
            "vision_models": sorted(getattr(cls, "VISION_MODELS", [])),
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
