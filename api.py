"""FastAPI web interface for Clod — wraps the multi-provider agent."""
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

app = FastAPI(title="Clod API", version="0.1.0")
REMOTE_CACHE_FILE = Path.home() / ".clod" / "github_remotes.json"

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
    mode: str = "chat"               # "chat" | "build" | "debug" | "explore" | "plan"
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
        return FileResponse(
            str(index), 
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
        )
    return {"name": "Clod API", "creator": "Yusuf Hosoglu", "version": "0.1.0", "docs": "/docs"}


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

    model = _normalize_requested_model(req.provider, req.model)

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

    # Load session history if provided
    session_messages: list[InternalMessage] = []
    if req.session_id:
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

        # Auto-save session if session_id provided
        if req.session_id:
            save_session(req.session_id, agent.history, req.provider, model or "")

        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


def _apply_chat_mode(message, mode: str):
    """Convert the web UI mode selector into the same slash commands as the CLI."""
    normalized = (mode or "chat").strip().lower()
    if normalized == "build_debug":
        normalized = _infer_build_or_debug_mode(message)
    if normalized not in {"build", "debug", "explore", "plan"}:
        return message
    if not isinstance(message, str):
        return message
    stripped = message.lstrip()
    if stripped.startswith(("/build", "/debug", "/explore", "/plan")):
        return message
    return f"/{normalized} {message}"


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


def _normalize_requested_model(provider: str, model: str | None) -> str | None:
    """Avoid sending known-stale model ids to providers."""
    if provider.lower() not in {"nvidia", "nim"} or not model:
        return model

    stale_nvidia_models = {
        "nvidia/llama-3.1-nemotron-ultra-253b-v1",
        "nvidia/llama-3.3-nemotron-super-49b-v1",
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.3-70b-instruct",
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
    if content is None:
        return ""
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
    
    ws_path = Path(req.workspace).expanduser().resolve() if req.workspace else Path(__file__).parent.resolve()
    if not ws_path.exists() or not ws_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid workspace path")

    repo_path = _resolve_git_worktree(ws_path)

    def run_cmd(args, cwd: Path | None = None):
        return subprocess.run(args, cwd=str(cwd or repo_path), capture_output=True, text=True, timeout=60)

    def combined_output(proc):
        return ((proc.stdout or "") + (proc.stderr or "")).strip()

    try:
        remote_input = (req.remote or "origin").strip() or "origin"
        cached_remote = _load_cached_remote(repo_path)
        if remote_input == "origin" and cached_remote:
            remote_input = cached_remote
        remote_name = "origin"
        remote_is_url = _looks_like_git_remote_url(remote_input)

        # Check if it's a git repo
        if not (repo_path / ".git").exists():
            init_proc = run_cmd(["git", "init"], cwd=repo_path)
            if init_proc.returncode != 0:
                return {"ok": False, "output": combined_output(init_proc)}
            # If no remote is provided, we can't push, but we can at least init.
            if not remote_is_url:
                return {
                    "ok": False,
                    "output": (
                        "Not a git repository and no remote URL provided.\n"
                        f"Workspace: {ws_path}\n"
                        f"Repo root: {repo_path}"
                    ),
                }
            remote_proc = run_cmd(["git", "remote", "add", remote_name, remote_input])
            if remote_proc.returncode != 0:
                return {"ok": False, "output": combined_output(remote_proc)}
        elif remote_is_url:
            remotes = run_cmd(["git", "remote"])
            if remote_name in remotes.stdout.split():
                remote_proc = run_cmd(["git", "remote", "set-url", remote_name, remote_input])
            else:
                remote_proc = run_cmd(["git", "remote", "add", remote_name, remote_input])
            if remote_proc.returncode != 0:
                return {"ok": False, "output": combined_output(remote_proc)}
        else:
            remote_name = remote_input

        branch_probe = run_cmd(["git", "branch", "--show-current"])
        current_branch = branch_probe.stdout.strip() or "main"
        before_status = run_cmd(["git", "status", "--short"])
        diagnostics = [
            f"Workspace: {ws_path}",
            f"Repo root: {repo_path}",
            f"Branch: {(req.branch or '').strip() or current_branch}",
            f"Remote: {remote_input if remote_is_url else remote_name}",
            "Status before staging:",
            before_status.stdout.strip() or "(clean)",
        ]
        
        # Add all
        add_proc = run_cmd(["git", "add", "."])
        if add_proc.returncode != 0:
            return {"ok": False, "output": combined_output(add_proc)}
        
        # Commit (only if there are changes)
        check = run_cmd(["git", "status", "--porcelain"])
        notes = diagnostics + [""]
        if not check.stdout.strip():
            notes.append(
                "No changes to commit in the repo root above. "
                "If you edited files externally, make sure the Workspace field points at that repository."
            )
        else:
            commit_proc = run_cmd(["git", "commit", "-m", req.commit_message])
            if commit_proc.returncode != 0:
                return {"ok": False, "output": combined_output(commit_proc)}
            notes.append(combined_output(commit_proc))
            
        # Push
        branch = (req.branch or "").strip() or current_branch

        push_args = ["git", "push", "-u", remote_name, branch]
        if req.force:
            push_args.append("--force-with-lease")
        proc = run_cmd(push_args)
        if proc.returncode == 0:
            if remote_is_url:
                _save_cached_remote(repo_path, remote_input)
            output = "\n".join([note for note in notes if note] + [combined_output(proc) or f"Successfully pushed to GitHub ({branch})"])
            return {"ok": True, "output": output}
        else:
            return {"ok": False, "output": combined_output(proc) or "Push failed"}

    except Exception as e:
        return {"ok": False, "output": str(e)}


def _looks_like_git_remote_url(value: str) -> bool:
    lowered = (value or "").strip().lower()
    return (
        lowered.startswith("http://")
        or lowered.startswith("https://")
        or lowered.startswith("ssh://")
        or lowered.startswith("git@")
    )


def _resolve_git_worktree(workspace: Path) -> Path:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return Path(proc.stdout.strip()).resolve()
    except Exception:
        pass
    return workspace


def _load_cached_remote(workspace: Path) -> str:
    try:
        data = json.loads(REMOTE_CACHE_FILE.read_text(encoding="utf-8"))
        value = str(data.get(str(workspace)) or "")
        return value if _looks_like_git_remote_url(value) else ""
    except Exception:
        return ""


def _save_cached_remote(workspace: Path, remote_url: str) -> None:
    if not _looks_like_git_remote_url(remote_url):
        return
    try:
        REMOTE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(REMOTE_CACHE_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        data[str(workspace)] = remote_url
        REMOTE_CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


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


class OAuthLoginRequest(BaseModel):
    provider: str


class CodexLoginRequest(BaseModel):
    method: str = "device"


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


@app.get("/auth/oauth/status")
def oauth_status():
    """Return OAuth auth state for providers with browser/device login support."""
    import shutil
    import subprocess

    from src.core.auth import google_oauth_status

    google = google_oauth_status()
    codex_bin = shutil.which("codex")
    codex = {
        "configured": False,
        "source": None,
        "available": bool(codex_bin),
        "binary": codex_bin,
        "message": "Codex CLI not found.",
    }
    if codex_bin:
        try:
            proc = subprocess.run(
                [codex_bin, "login", "status"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            output = ((proc.stdout or "") + (proc.stderr or "")).strip()
            is_logged_in = proc.returncode == 0 and "logged in" in output.lower()
            codex.update({
                "configured": is_logged_in,
                "source": "oauth" if is_logged_in else None,
                "message": output or ("Logged in." if is_logged_in else "Not logged in."),
            })
        except Exception as e:
            codex["message"] = str(e)

    return {
        "gemini": google,
        "google": google,
        "codex": codex,
    }


@app.post("/auth/oauth/login")
def oauth_login(req: OAuthLoginRequest):
    """Start a provider OAuth login flow."""
    provider = req.provider.lower()
    if provider not in ("gemini", "google"):
        raise HTTPException(status_code=400, detail=f"Unsupported OAuth provider: {req.provider}")

    from src.core.auth import google_oauth_status, login_google

    before = google_oauth_status()
    if not before["client_secret_exists"]:
        raise HTTPException(
            status_code=400,
            detail=f"Missing Google OAuth client secret at {before['client_secret_path']}",
        )

    token = login_google()
    if not token:
        raise HTTPException(status_code=400, detail="Google OAuth login was cancelled or failed.")

    return {"ok": True, "provider": "gemini", "status": google_oauth_status()}


@app.post("/auth/codex/login")
def codex_login(req: CodexLoginRequest):
    """Open the official Codex CLI login flow."""
    import platform
    import shlex
    import shutil
    import subprocess

    codex_bin = shutil.which("codex")
    if not codex_bin:
        raise HTTPException(status_code=400, detail="Codex CLI was not found on PATH.")

    method = req.method.lower()
    args = [codex_bin, "login"]
    if method in ("device", "device-code", "device_code"):
        args.append("--device-auth")
    elif method not in ("browser", "chatgpt"):
        raise HTTPException(status_code=400, detail=f"Unsupported Codex login method: {req.method}")

    if platform.system() == "Darwin":
        command = " ".join(shlex.quote(part) for part in args)
        script = (
            'tell application "Terminal"\n'
            f'  do script "{command}; echo; echo Clod Codex login finished. You can close this window."\n'
            "  activate\n"
            "end tell"
        )
        subprocess.Popen(["osascript", "-e", script])
        return {"ok": True, "provider": "codex", "method": method, "message": "Opened Codex login in Terminal."}

    subprocess.Popen(args)
    return {"ok": True, "provider": "codex", "method": method, "message": "Started Codex login."}


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
        "anthropic": "ANTHROPIC_API_KEY",
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
        has_oauth = p == "gemini" and bool(config.get("oauth", {}).get("google"))
        result[p] = {
            "configured": has_env or has_saved or has_oauth,
            "source": "env" if has_env else ("oauth" if has_oauth else ("saved" if has_saved else None)),
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
        model_infos = [info.to_dict() for info in cls.list_model_infos(models)]
        result[alias] = {
            "name": cls.name if isinstance(cls.name, str) else alias,
            "models": models,
            "model_info": model_infos,
            "supports_tools": getattr(cls, "SUPPORTS_TOOLS", True),
            "supports_images": getattr(cls, "SUPPORTS_IMAGES", False),
            "vision_models": sorted(getattr(cls, "VISION_MODELS", [])),
            "api": getattr(cls, "API_NAME", "unknown"),
            "context_window": getattr(cls, "CONTEXT_WINDOW", 0),
            "max_output_tokens": getattr(cls, "MAX_OUTPUT_TOKENS", 0),
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
