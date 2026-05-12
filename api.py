"""FastAPI web interface for Clod — wraps the multi-provider agent."""
import json
import os
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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
    toolEvents: list[dict] = Field(default_factory=list)


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
    soul: str | None = None          # persona preset: default | karpathy | concise | proactive | debugger
    autonomy: str | None = None      # auto-safe | confirm | full-auto


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
    from src.providers.base import Message as InternalMessage, ToolCall, ToolEvent
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
        agent = Agent(
            provider,
            registry,
            project_root=str(workspace),
            soul=req.soul,
            autonomy=req.autonomy,
        )
        agent.prior_tool_calls = _extract_prior_tool_calls(req.messages, ToolCall)
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

    stale_nvidia_models: set[str] = set()
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


def _extract_prior_tool_calls(messages: list[ChatMessage], tool_call_cls) -> list:
    """Recover browser-stored toolEvents so the agent can avoid repeating visible actions."""
    prior = []
    for msg in messages[:-1]:
        for index, event in enumerate(msg.toolEvents or []):
            if not isinstance(event, dict) or event.get("type") != "start":
                continue
            name = event.get("name")
            arguments = event.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                continue
            prior.append(
                tool_call_cls(
                    id=f"prior-{len(prior)}-{index}",
                    name=name,
                    arguments=arguments,
                )
            )
    return prior


class PushRequest(BaseModel):
    workspace: str | None = None
    remote: str = "origin"
    branch: str | None = None
    force: bool = False
    commit_message: str = "Update from Clod assistant"
    commit_body: str | None = None
    allow_secrets: bool = False
    skip_hooks: bool = False
    github_token: str | None = None
    user_name: str | None = None
    user_email: str | None = None


class TerminalRunRequest(BaseModel):
    command: str
    workspace: str | None = None
    timeout: int = 120


_SECRET_PATTERNS = (
    ".env", ".env.local", ".env.production",
    "id_rsa", "id_ed25519", "id_dsa", "id_ecdsa",
    ".pem", ".key", ".pfx", ".p12",
    "credentials.json", "service-account.json",
    ".aws/credentials", ".npmrc", ".pypirc",
)


def _scan_staged_for_secrets(run_cmd) -> list[str]:
    proc = run_cmd(["git", "diff", "--cached", "--name-only"])
    if proc.returncode != 0:
        return []
    flagged: list[str] = []
    for line in proc.stdout.splitlines():
        path = line.strip()
        if not path:
            continue
        lowered = path.lower()
        if any(lowered.endswith(p) or lowered == p or f"/{p}" in f"/{lowered}" for p in _SECRET_PATTERNS):
            flagged.append(path)
    return flagged


def _inject_token_into_url(url: str, token: str) -> str:
    """Insert GitHub token into HTTPS URL for one-shot push (env-only, never persisted)."""
    if not url.lower().startswith(("http://", "https://")):
        return url
    scheme, _, rest = url.partition("://")
    if "@" in rest.split("/", 1)[0]:
        return url  # already has creds
    return f"{scheme}://x-access-token:{token}@{rest}"


@app.post("/git/push")
def push_to_github(req: PushRequest):
    """Stage, commit, and push changes to GitHub."""
    import subprocess
    import os
    from pathlib import Path

    ws_path = Path(req.workspace).expanduser().resolve() if req.workspace else Path(__file__).parent.resolve()
    if not ws_path.exists() or not ws_path.is_dir():
        raise HTTPException(status_code=400, detail="Invalid workspace path")

    repo_path = _resolve_git_worktree(ws_path)

    base_env = os.environ.copy()
    # Prevent password prompts from blocking
    base_env.setdefault("GIT_TERMINAL_PROMPT", "0")
    base_env.setdefault("GIT_ASKPASS", "echo")

    def run_cmd(args, cwd: Path | None = None, timeout: int = 60, env: dict | None = None):
        return subprocess.run(
            args,
            cwd=str(cwd or repo_path),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env or base_env,
        )

    def combined_output(proc):
        return ((proc.stdout or "") + (proc.stderr or "")).strip()

    try:
        remote_input = (req.remote or "origin").strip() or "origin"
        cached_remote = _load_cached_remote(repo_path)
        if remote_input == "origin" and cached_remote:
            remote_input = cached_remote
        remote_name = "origin"
        remote_is_url = _looks_like_git_remote_url(remote_input)

        # Detect repo via rev-parse (handles worktrees, submodules, .git files)
        is_repo_proc = run_cmd(["git", "rev-parse", "--is-inside-work-tree"])
        is_repo = is_repo_proc.returncode == 0 and is_repo_proc.stdout.strip() == "true"

        if not is_repo:
            init_proc = run_cmd(["git", "init"], cwd=repo_path)
            if init_proc.returncode != 0:
                return {"ok": False, "output": combined_output(init_proc)}
            if not remote_is_url:
                return {
                    "ok": False,
                    "output": (
                        "Not a git repository and no remote URL provided.\n"
                        f"Workspace: {ws_path}\nRepo root: {repo_path}"
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
            verify = run_cmd(["git", "remote", "get-url", remote_name])
            if verify.returncode != 0:
                return {
                    "ok": False,
                    "output": f"Remote '{remote_name}' not configured. Provide a URL or run: git remote add {remote_name} <url>",
                }

        # Detect detached HEAD
        branch_probe = run_cmd(["git", "branch", "--show-current"])
        current_branch = branch_probe.stdout.strip()
        if not current_branch:
            head_proc = run_cmd(["git", "rev-parse", "--verify", "HEAD"])
            if head_proc.returncode == 0:
                # Detached HEAD with commits: require explicit branch
                if not (req.branch or "").strip():
                    return {
                        "ok": False,
                        "output": "Detached HEAD detected. Specify a branch name to push.",
                    }
                current_branch = req.branch.strip()
            else:
                # Empty repo: use requested branch or 'main'
                current_branch = (req.branch or "").strip() or "main"
                run_cmd(["git", "checkout", "-B", current_branch])

        # Ensure user.name/email configured
        name_proc = run_cmd(["git", "config", "user.name"])
        email_proc = run_cmd(["git", "config", "user.email"])
        if name_proc.returncode != 0 or not name_proc.stdout.strip():
            name = req.user_name or os.environ.get("GIT_AUTHOR_NAME") or "Clod Assistant"
            run_cmd(["git", "config", "user.name", name])
        if email_proc.returncode != 0 or not email_proc.stdout.strip():
            email = req.user_email or os.environ.get("GIT_AUTHOR_EMAIL") or "clod@local"
            run_cmd(["git", "config", "user.email", email])

        before_status = run_cmd(["git", "status", "--short"])
        diagnostics = [
            f"Workspace: {ws_path}",
            f"Repo root: {repo_path}",
            f"Branch: {(req.branch or '').strip() or current_branch}",
            f"Remote: {remote_input if remote_is_url else remote_name}",
            "Status before staging:",
            before_status.stdout.strip() or "(clean)",
        ]

        add_proc = run_cmd(["git", "add", "."])
        if add_proc.returncode != 0:
            return {"ok": False, "output": combined_output(add_proc)}

        # Secret scan on staged files
        if not req.allow_secrets:
            flagged = _scan_staged_for_secrets(run_cmd)
            if flagged:
                run_cmd(["git", "reset", "HEAD", "--"] + flagged)
                diagnostics.append(
                    "Unstaged suspicious files (set allow_secrets=true to override): "
                    + ", ".join(flagged)
                )

        check = run_cmd(["git", "status", "--porcelain"])
        notes = diagnostics + [""]
        commit_sha = None
        if not check.stdout.strip():
            notes.append(
                "No changes to commit. "
                "If you edited files externally, make sure the Workspace field points at that repository."
            )
        else:
            msg = req.commit_message
            if req.commit_body:
                msg = f"{msg}\n\n{req.commit_body}"
            commit_args = ["git", "commit", "-m", msg]
            if req.skip_hooks:
                commit_args.append("--no-verify")
            commit_proc = run_cmd(commit_args)
            if commit_proc.returncode != 0:
                return {"ok": False, "output": combined_output(commit_proc)}
            notes.append(combined_output(commit_proc))
            sha_proc = run_cmd(["git", "rev-parse", "HEAD"])
            if sha_proc.returncode == 0:
                commit_sha = sha_proc.stdout.strip()

        branch = (req.branch or "").strip() or current_branch

        # Pre-push fetch to detect non-fast-forward
        fetch_env = dict(base_env)
        if req.github_token:
            url_proc = run_cmd(["git", "remote", "get-url", remote_name])
            if url_proc.returncode == 0:
                authed = _inject_token_into_url(url_proc.stdout.strip(), req.github_token)
                fetch_env["GIT_CONFIG_COUNT"] = "1"
                fetch_env["GIT_CONFIG_KEY_0"] = f"url.{authed}.insteadOf"
                fetch_env["GIT_CONFIG_VALUE_0"] = url_proc.stdout.strip()

        run_cmd(["git", "fetch", remote_name, branch], timeout=120, env=fetch_env)

        push_args = ["git", "push", "-u", remote_name, branch]
        if req.force:
            push_args.append("--force-with-lease")
        if req.skip_hooks:
            push_args.append("--no-verify")

        proc = run_cmd(push_args, timeout=300, env=fetch_env)
        out = combined_output(proc)
        if proc.returncode == 0:
            if remote_is_url:
                _save_cached_remote(repo_path, remote_input)
            output = "\n".join([n for n in notes if n] + [out or f"Successfully pushed to {branch}"])
            return {"ok": True, "output": output, "commit_sha": commit_sha, "branch": branch}

        # Diagnose common failures
        lower = out.lower()
        if "non-fast-forward" in lower or "rejected" in lower and "fetch first" in lower:
            hint = "\n\nHint: Remote has commits you don't. Run a pull/rebase first, or set force=true (uses --force-with-lease)."
        elif "authentication failed" in lower or "could not read" in lower or "permission denied" in lower:
            hint = "\n\nHint: Auth failed. Pass github_token, configure a credential helper, or use SSH."
        elif "src refspec" in lower and "does not match" in lower:
            hint = "\n\nHint: Branch has no commits yet. Make at least one commit before pushing."
        elif "terminal prompts disabled" in lower:
            hint = "\n\nHint: Git asked for credentials but prompting is disabled. Pass github_token or configure a credential helper."
        else:
            hint = ""

        return {"ok": False, "output": (out or "Push failed") + hint}

    except subprocess.TimeoutExpired as e:
        return {"ok": False, "output": f"Git operation timed out: {e.cmd}"}
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


@app.get("/souls")
def get_souls():
    """Return available soul presets (persona + autonomy defaults)."""
    from src.core.souls import list_souls
    return {"souls": list_souls()}


# ── Clarify endpoints ────────────────────────────────────────────────────────


class ClarifyResolveRequest(BaseModel):
    session_id: str
    response: str


@app.get("/clarify/{session_id}")
def clarify_status(session_id: str):
    from src.core import clarify as clarify_mod
    return {"pending": clarify_mod.peek(session_id)}


@app.post("/clarify/respond")
def clarify_respond(req: ClarifyResolveRequest):
    from src.core import clarify as clarify_mod
    n = clarify_mod.resolve(req.session_id, req.response)
    return {"resolved": n}


# ── Checkpoint endpoints ─────────────────────────────────────────────────────


class CheckpointCreateRequest(BaseModel):
    workspace: str | None = None
    label: str = ""


class CheckpointRestoreRequest(BaseModel):
    workspace: str | None = None
    checkpoint_id: str


@app.get("/checkpoints")
def checkpoints_list(workspace: str | None = None):
    from src.core import checkpoint as ckpt
    return {"checkpoints": ckpt.list_checkpoints(workspace)}


@app.post("/checkpoints")
def checkpoints_create(req: CheckpointCreateRequest):
    from src.core import checkpoint as ckpt
    return ckpt.create_checkpoint(req.workspace, label=req.label)


@app.post("/checkpoints/restore")
def checkpoints_restore(req: CheckpointRestoreRequest):
    from src.core import checkpoint as ckpt
    return ckpt.restore_checkpoint(req.workspace, req.checkpoint_id)


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


# ── Coolify deployment analyzer ───────────────────────────────────────────────

class CoolifyAnalyzeRequest(BaseModel):
    workspace: str
    write_files: bool = False
    overwrite: bool = False


def _read_text_safe(path: Path, limit: int = 200_000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")[:limit]
    except Exception:
        return ""


def _read_json_safe(path: Path) -> dict:
    try:
        return json.loads(_read_text_safe(path))
    except Exception:
        return {}


def _detect_stack(root: Path) -> dict:
    """Detect framework, language, build/start commands, port, build pack."""
    indicators: list[str] = []
    info: dict = {
        "framework": None,
        "language": None,
        "package_manager": None,
        "build_pack": None,        # "dockerfile" | "nixpacks" | "static" | "compose"
        "build_command": None,
        "install_command": None,
        "start_command": None,
        "port": None,
        "publish_dir": None,
        "is_static": False,
        "is_ssr": False,
        "needs_database": False,
        "env_required": [],
        "persistent_paths": [],
        "healthcheck_path": "/",
        "indicators": indicators,
        "warnings": [],
    }

    pkg = root / "package.json"
    pyproject = root / "pyproject.toml"
    reqs = root / "requirements.txt"
    gomod = root / "go.mod"
    cargo = root / "Cargo.toml"
    gemfile = root / "Gemfile"
    composer = root / "composer.json"
    dockerfile = root / "Dockerfile"
    compose = root / "docker-compose.yml"
    compose_yaml = root / "docker-compose.yaml"
    index_html = root / "index.html"

    if dockerfile.exists():
        info["build_pack"] = "dockerfile"
        indicators.append("Dockerfile")
        df_text = _read_text_safe(dockerfile)
        # Try to read EXPOSE
        for line in df_text.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("EXPOSE "):
                try:
                    info["port"] = int(stripped.split()[1].split("/")[0])
                except Exception:
                    pass
                break
    if compose.exists() or compose_yaml.exists():
        indicators.append("docker-compose")
        if not info["build_pack"]:
            info["build_pack"] = "compose"

    # Node.js / JS frameworks
    if pkg.exists():
        data = _read_json_safe(pkg)
        scripts = data.get("scripts", {}) or {}
        deps = {**(data.get("dependencies") or {}), **(data.get("devDependencies") or {})}
        info["language"] = "javascript"
        indicators.append("package.json")

        if (root / "pnpm-lock.yaml").exists():
            info["package_manager"] = "pnpm"
        elif (root / "yarn.lock").exists():
            info["package_manager"] = "yarn"
        elif (root / "bun.lockb").exists() or (root / "bun.lock").exists():
            info["package_manager"] = "bun"
        else:
            info["package_manager"] = "npm"
        pm = info["package_manager"]
        info["install_command"] = {"npm": "npm ci", "pnpm": "pnpm install --frozen-lockfile",
                                    "yarn": "yarn install --frozen-lockfile", "bun": "bun install"}[pm]
        info["build_command"] = f"{pm} run build" if "build" in scripts else None
        info["start_command"] = f"{pm} start" if "start" in scripts else None

        if "next" in deps:
            info["framework"] = "Next.js"
            info["port"] = info["port"] or 3000
            next_cfg = next((p for p in [root / "next.config.js", root / "next.config.mjs",
                                          root / "next.config.ts"] if p.exists()), None)
            cfg_text = _read_text_safe(next_cfg) if next_cfg else ""
            if "output:" in cfg_text and "'export'" in cfg_text or '"export"' in cfg_text:
                info["is_static"] = True
                info["publish_dir"] = "out"
                info["build_pack"] = info["build_pack"] or "static"
            else:
                info["is_ssr"] = True
                info["build_pack"] = info["build_pack"] or "nixpacks"
        elif "nuxt" in deps or "nuxt3" in deps:
            info["framework"] = "Nuxt"
            info["port"] = info["port"] or 3000
            info["is_ssr"] = True
            info["start_command"] = "node .output/server/index.mjs"
            info["build_pack"] = info["build_pack"] or "nixpacks"
        elif "@remix-run/serve" in deps or "@remix-run/node" in deps:
            info["framework"] = "Remix"
            info["port"] = info["port"] or 3000
            info["is_ssr"] = True
            info["build_pack"] = info["build_pack"] or "nixpacks"
        elif "@sveltejs/kit" in deps:
            info["framework"] = "SvelteKit"
            info["port"] = info["port"] or 3000
            if "@sveltejs/adapter-static" in deps:
                info["is_static"] = True
                info["publish_dir"] = "build"
                info["build_pack"] = info["build_pack"] or "static"
            else:
                info["is_ssr"] = True
                info["build_pack"] = info["build_pack"] or "nixpacks"
                info["start_command"] = "node build"
        elif "astro" in deps:
            info["framework"] = "Astro"
            info["port"] = info["port"] or 4321
            adapters = [d for d in deps if d.startswith("@astrojs/") and d != "@astrojs/check"]
            if any(a in deps for a in ("@astrojs/node", "@astrojs/vercel", "@astrojs/cloudflare")):
                info["is_ssr"] = True
                info["build_pack"] = info["build_pack"] or "nixpacks"
            else:
                info["is_static"] = True
                info["publish_dir"] = "dist"
                info["build_pack"] = info["build_pack"] or "static"
        elif "gatsby" in deps:
            info["framework"] = "Gatsby"
            info["is_static"] = True
            info["publish_dir"] = "public"
            info["build_pack"] = info["build_pack"] or "static"
        elif "@angular/core" in deps:
            info["framework"] = "Angular"
            info["is_static"] = True
            info["publish_dir"] = "dist"
            info["build_pack"] = info["build_pack"] or "static"
        elif "vite" in deps and ("react" in deps or "vue" in deps or "svelte" in deps or "solid-js" in deps):
            info["framework"] = "Vite SPA"
            info["is_static"] = True
            info["publish_dir"] = "dist"
            info["build_pack"] = info["build_pack"] or "static"
        elif "express" in deps or "fastify" in deps or "hono" in deps or "koa" in deps:
            info["framework"] = "Node API"
            info["port"] = info["port"] or 3000
            info["is_ssr"] = True
            info["build_pack"] = info["build_pack"] or "nixpacks"
        elif "react-scripts" in deps:
            info["framework"] = "Create React App"
            info["is_static"] = True
            info["publish_dir"] = "build"
            info["build_pack"] = info["build_pack"] or "static"

        # Refine start command from scripts
        for key in ("start", "serve", "preview"):
            if key in scripts and not info["is_static"]:
                info["start_command"] = f"{pm} run {key}"
                break

    # Python
    elif pyproject.exists() or reqs.exists() or (root / "manage.py").exists():
        info["language"] = "python"
        text = ""
        if pyproject.exists():
            indicators.append("pyproject.toml")
            text += _read_text_safe(pyproject).lower()
        if reqs.exists():
            indicators.append("requirements.txt")
            text += _read_text_safe(reqs).lower()
        if (root / "manage.py").exists():
            indicators.append("manage.py")
            info["framework"] = "Django"
            info["port"] = 8000
            info["start_command"] = "gunicorn --bind 0.0.0.0:8000 $(ls */wsgi.py | head -1 | sed 's|/wsgi.py||').wsgi"
            info["install_command"] = "pip install -r requirements.txt gunicorn"
            info["healthcheck_path"] = "/"
            info["env_required"].extend(["SECRET_KEY", "DEBUG", "ALLOWED_HOSTS", "DATABASE_URL"])
        elif "fastapi" in text:
            info["framework"] = "FastAPI"
            info["port"] = 8000
            info["start_command"] = "uvicorn main:app --host 0.0.0.0 --port 8000"
            info["install_command"] = "pip install -r requirements.txt"
            info["healthcheck_path"] = "/docs"
        elif "flask" in text:
            info["framework"] = "Flask"
            info["port"] = 5000
            info["start_command"] = "gunicorn --bind 0.0.0.0:5000 app:app"
            info["install_command"] = "pip install -r requirements.txt gunicorn"
        elif "streamlit" in text:
            info["framework"] = "Streamlit"
            info["port"] = 8501
            info["start_command"] = "streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
        else:
            info["framework"] = "Python"
            info["port"] = 8000
            info["install_command"] = "pip install -r requirements.txt" if reqs.exists() else "pip install ."
        info["build_pack"] = info["build_pack"] or "nixpacks"

    # Go
    elif gomod.exists():
        info["language"] = "go"
        info["framework"] = "Go"
        info["port"] = 8080
        info["build_command"] = "go build -o app ."
        info["start_command"] = "./app"
        info["build_pack"] = info["build_pack"] or "nixpacks"
        indicators.append("go.mod")

    # Rust
    elif cargo.exists():
        info["language"] = "rust"
        info["framework"] = "Rust"
        info["port"] = 8080
        info["build_command"] = "cargo build --release"
        info["start_command"] = "./target/release/$(grep '^name' Cargo.toml | head -1 | cut -d'\"' -f2)"
        info["build_pack"] = info["build_pack"] or "nixpacks"
        indicators.append("Cargo.toml")

    # Ruby
    elif gemfile.exists():
        info["language"] = "ruby"
        text = _read_text_safe(gemfile).lower()
        if "rails" in text:
            info["framework"] = "Rails"
            info["port"] = 3000
            info["start_command"] = "bundle exec rails server -b 0.0.0.0 -p 3000"
            info["env_required"].extend(["RAILS_ENV", "SECRET_KEY_BASE", "DATABASE_URL"])
        else:
            info["framework"] = "Ruby"
            info["port"] = 3000
        info["install_command"] = "bundle install"
        info["build_pack"] = info["build_pack"] or "nixpacks"
        indicators.append("Gemfile")

    # PHP
    elif composer.exists():
        info["language"] = "php"
        data = _read_json_safe(composer)
        deps = {**(data.get("require") or {}), **(data.get("require-dev") or {})}
        if "laravel/framework" in deps:
            info["framework"] = "Laravel"
            info["port"] = 8000
            info["start_command"] = "php artisan serve --host=0.0.0.0 --port=8000"
            info["env_required"].extend(["APP_KEY", "APP_ENV", "DATABASE_URL"])
            info["persistent_paths"].append("storage")
        else:
            info["framework"] = "PHP"
            info["port"] = 8000
        info["build_pack"] = info["build_pack"] or "nixpacks"
        indicators.append("composer.json")

    # Pure static site
    elif index_html.exists():
        info["language"] = "html"
        info["framework"] = "Static HTML"
        info["is_static"] = True
        info["publish_dir"] = "."
        info["build_pack"] = "static"
        indicators.append("index.html")

    # Database / cache hints
    blob = ""
    for f in (pkg, pyproject, reqs, gemfile, composer, gomod, cargo):
        if f.exists():
            blob += _read_text_safe(f).lower()
    if "postgres" in blob or "psycopg" in blob or "pg" in blob.split():
        info["needs_database"] = True
        info["env_required"].append("DATABASE_URL")
    if "redis" in blob:
        info["env_required"].append("REDIS_URL")
    if "mysql" in blob or "mariadb" in blob:
        info["needs_database"] = True
        info["env_required"].append("DATABASE_URL")

    # Persistent paths heuristic
    for candidate in ("uploads", "storage", "data", "media", "public/uploads", "var"):
        if (root / candidate).exists() and (root / candidate).is_dir():
            if candidate not in info["persistent_paths"]:
                info["persistent_paths"].append(candidate)
    for sqlite in ("db.sqlite3", "database.sqlite", "sqlite.db"):
        if (root / sqlite).exists():
            info["persistent_paths"].append(sqlite)
            info["needs_database"] = True

    # .env.example → required env keys
    env_example = root / ".env.example"
    if env_example.exists():
        for line in _read_text_safe(env_example).splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key = line.split("=", 1)[0].strip()
            if key and key not in info["env_required"]:
                info["env_required"].append(key)

    # Healthcheck refinement
    if info["framework"] in ("FastAPI",):
        info["healthcheck_path"] = "/docs"
    elif info["is_static"]:
        info["healthcheck_path"] = "/"

    # Warnings
    if info["framework"] is None:
        info["warnings"].append("Could not detect a framework. Falling back to generic config.")
        info["build_pack"] = info["build_pack"] or "nixpacks"
    if info["is_ssr"] and not info["start_command"]:
        info["warnings"].append("SSR detected but no start command found. Add a 'start' script.")
    if info["build_pack"] == "dockerfile" and not info["port"]:
        info["warnings"].append("Dockerfile present but no EXPOSE directive. Set port manually in Coolify.")

    return info


def _generate_dockerfile(info: dict) -> str:
    fw = info.get("framework") or ""
    lang = info.get("language") or ""
    port = info.get("port") or 3000
    install = info.get("install_command") or ""
    build = info.get("build_command") or ""
    start = info.get("start_command") or ""
    pm = info.get("package_manager") or "npm"

    if lang == "javascript":
        node_image = "node:20-alpine"
        return f"""# Auto-generated by Clod for Coolify
FROM {node_image} AS deps
WORKDIR /app
COPY package*.json pnpm-lock.yaml* yarn.lock* bun.lock* ./
RUN {install or f'{pm} install'}

FROM {node_image} AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
{f'RUN {build}' if build else '# no build step'}

FROM {node_image} AS runner
WORKDIR /app
ENV NODE_ENV=production
ENV PORT={port}
COPY --from=builder /app ./
EXPOSE {port}
CMD {json.dumps((start or f'{pm} start').split())}
"""
    if lang == "python":
        return f"""# Auto-generated by Clod for Coolify
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt* pyproject.toml* ./
RUN {install or 'pip install -r requirements.txt'}
COPY . .
EXPOSE {port}
CMD {json.dumps((start or f'uvicorn main:app --host 0.0.0.0 --port {port}').split())}
"""
    if lang == "go":
        return f"""# Auto-generated by Clod for Coolify
FROM golang:1.22-alpine AS builder
WORKDIR /src
COPY go.* ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o /app .

FROM alpine:3.20
WORKDIR /app
COPY --from=builder /app /app/app
EXPOSE {port}
CMD ["/app/app"]
"""
    if lang == "rust":
        return f"""# Auto-generated by Clod for Coolify
FROM rust:1.78 AS builder
WORKDIR /src
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=builder /src/target/release/* /app/
EXPOSE {port}
CMD ["/app/app"]
"""
    if lang == "php":
        return f"""# Auto-generated by Clod for Coolify
FROM php:8.3-cli
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y unzip git \\
 && curl -sS https://getcomposer.org/installer | php -- --install-dir=/usr/local/bin --filename=composer \\
 && composer install --no-dev --optimize-autoloader
EXPOSE {port}
CMD {json.dumps((start or f'php -S 0.0.0.0:{port} -t public').split())}
"""
    if info.get("is_static"):
        publish = info.get("publish_dir") or "."
        return f"""# Auto-generated by Clod for Coolify (static)
FROM nginx:alpine
COPY {publish} /usr/share/nginx/html
EXPOSE 80
"""
    return f"""# Auto-generated by Clod for Coolify (generic)
FROM debian:bookworm-slim
WORKDIR /app
COPY . .
EXPOSE {port}
CMD ["sh", "-c", "echo 'Configure start command'"]
"""


def _generate_dockerignore(lang: str) -> str:
    common = [".git", ".gitignore", ".env", ".env.*", "!.env.example",
              "node_modules", "*.log", ".DS_Store", "coverage", ".vscode", ".idea"]
    by_lang = {
        "javascript": ["dist", "build", ".next", ".nuxt", ".turbo", ".cache"],
        "python": ["__pycache__", "*.pyc", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"],
        "go": ["bin", "*.test"],
        "rust": ["target"],
        "php": ["vendor"],
        "ruby": ["vendor", "tmp"],
    }
    return "\n".join(common + by_lang.get(lang, [])) + "\n"


def _generate_coolify_json(info: dict) -> dict:
    return {
        "name": "auto-detected",
        "type": "application",
        "build_pack": info.get("build_pack"),
        "ports_exposes": str(info.get("port") or 3000),
        "install_command": info.get("install_command"),
        "build_command": info.get("build_command"),
        "start_command": info.get("start_command"),
        "publish_directory": info.get("publish_dir"),
        "health_check_path": info.get("healthcheck_path") or "/",
        "health_check_enabled": True,
        "static_image": info.get("is_static"),
        "environment_variables": [
            {"key": k, "value": "", "is_required": True} for k in info.get("env_required") or []
        ],
        "persistent_storages": [
            {"mount_path": f"/app/{p}", "host_path": None, "name": p.replace("/", "_")}
            for p in info.get("persistent_paths") or []
        ],
    }


def _build_recommendations(info: dict) -> list[str]:
    recs: list[str] = []
    if info.get("build_pack") == "dockerfile":
        recs.append("Use 'Dockerfile' build pack — repo already contains one.")
    elif info.get("build_pack") == "static":
        recs.append(f"Use 'Static' build pack with publish dir '{info.get('publish_dir')}'.")
    elif info.get("build_pack") == "compose":
        recs.append("Use 'Docker Compose' build pack — docker-compose detected.")
    else:
        recs.append("Use 'Nixpacks' build pack — auto-detects runtime.")
    if info.get("port"):
        recs.append(f"Set Port: {info['port']}")
    if info.get("healthcheck_path"):
        recs.append(f"Healthcheck path: {info['healthcheck_path']}")
    if info.get("env_required"):
        recs.append("Required env vars: " + ", ".join(info["env_required"]))
    if info.get("needs_database"):
        recs.append("Provision a database service (Postgres/MySQL) in Coolify and set DATABASE_URL.")
    if info.get("persistent_paths"):
        recs.append("Add persistent volumes for: " + ", ".join(info["persistent_paths"]))
    if info.get("is_ssr"):
        recs.append("Enable HTTP/2 and websockets in Coolify if your framework streams.")
    if info.get("warnings"):
        recs.extend(f"⚠ {w}" for w in info["warnings"])
    return recs


@app.post("/deploy/coolify/analyze")
def coolify_analyze(req: CoolifyAnalyzeRequest):
    """Detect stack and produce Coolify deployment configuration."""
    ws = Path(req.workspace).expanduser().resolve()
    if not ws.exists() or not ws.is_dir():
        raise HTTPException(status_code=400, detail="Invalid workspace path")

    info = _detect_stack(ws)
    coolify_cfg = _generate_coolify_json(info)
    recommendations = _build_recommendations(info)

    files = {
        "Dockerfile": _generate_dockerfile(info),
        ".dockerignore": _generate_dockerignore(info.get("language") or ""),
        "coolify.json": json.dumps(coolify_cfg, indent=2),
    }

    written: list[str] = []
    skipped: list[str] = []
    if req.write_files:
        for name, content in files.items():
            target = ws / name
            if target.exists() and not req.overwrite:
                skipped.append(name)
                continue
            try:
                target.write_text(content, encoding="utf-8")
                written.append(name)
            except Exception as e:
                skipped.append(f"{name} ({e})")

    return {
        "ok": True,
        "workspace": str(ws),
        "detection": info,
        "coolify_config": coolify_cfg,
        "recommendations": recommendations,
        "files": files,
        "written": written,
        "skipped": skipped,
    }
