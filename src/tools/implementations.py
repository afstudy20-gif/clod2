"""Tool implementations - filesystem, shell, and search operations."""
import fnmatch
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Project root for resolving relative paths
_project_root: str | None = None


def set_project_root(root: str | None):
    """Set the project root for resolving relative paths."""
    global _project_root
    _project_root = root


def _resolve_path(path: str) -> Path:
    """Resolve a path, making relative paths relative to project root."""
    p = Path(path).expanduser()
    if not p.is_absolute() and _project_root:
        p = Path(_project_root) / p
    return p


def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file and return its contents with line numbers."""
    p = _resolve_path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    if not p.is_file():
        return f"Error: Not a file: {path}"
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        start = max(0, offset)
        end = start + limit
        selected = lines[start:end]
        numbered = [f"{start + i + 1:4d}\t{line}" for i, line in enumerate(selected)]
        result = "\n".join(numbered)
        if end < len(lines):
            result += f"\n... ({len(lines) - end} more lines)"
        return result
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent dirs as needed."""
    p = _resolve_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def scaffold_macos_app(
    app_name: str = "Clod App",
    app_dir: str = "desktop-app",
    backend_command: str = "",
    start_url: str = "",
    health_path: str = "",
    port: int = 8765,
    overwrite: bool = False,
) -> str:
    """Create an Electron macOS desktop wrapper for the selected project."""
    root = Path(_project_root or os.getcwd()).resolve()
    target = Path(app_dir).expanduser()
    if not target.is_absolute():
        target = root / target
    target = target.resolve()
    if target.exists() and any(target.iterdir()) and not overwrite:
        return (
            f"Error: {app_dir} already exists and is not empty. "
            "Call again with overwrite=true if you want to replace the scaffold files."
        )

    command = backend_command.strip() or _infer_macos_backend_command(root)
    resolved_port = max(1, min(int(port or 8765), 65535))
    url = start_url.strip() or f"http://127.0.0.1:{{port}}"
    health_url = ""
    if health_path.strip():
        health_url = _join_local_url(url, health_path.strip())
    elif (root / "api.py").exists():
        health_url = _join_local_url(url, "/health")

    target.mkdir(parents=True, exist_ok=True)
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    package = _macos_package_json(app_name, app_dir)
    (target / "package.json").write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    (target / "main.js").write_text(
        _macos_main_js(app_name, command, url, health_url, resolved_port),
        encoding="utf-8",
    )
    (target / "README.md").write_text(
        _macos_readme(app_name, app_dir, command, url),
        encoding="utf-8",
    )

    setup_script = scripts_dir / "setup-macos-app.sh"
    setup_script.write_text(_macos_setup_script(app_dir), encoding="utf-8")
    setup_script.chmod(0o755)
    _ensure_gitignore_entries(root, [f"{app_dir}/node_modules/", f"{app_dir}/dist/"])

    changed = [
        str((target / "package.json").relative_to(root)),
        str((target / "main.js").relative_to(root)),
        str((target / "README.md").relative_to(root)),
        str(setup_script.relative_to(root)),
    ]
    return (
        "Created macOS Electron app scaffold.\n"
        f"App name: {app_name}\n"
        f"Backend command: {command or '(none; loads start_url directly)'}\n"
        f"Start URL: {url}\n"
        f"Health URL: {health_url or '(disabled)'}\n"
        "Changed files:\n- "
        + "\n- ".join(changed)
        + "\n\nNext steps:\n"
        f"1. bash scripts/setup-macos-app.sh\n"
        f"2. open {app_dir}/dist/mac-arm64/{_safe_product_name(app_name)}.app"
    )


def execute_sandbox_python(code: str) -> str:
    """Execute Python code in Docker when available, otherwise in a local temp process."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = os.path.join(temp_dir, "agent_script.py")
        
        with open(temp_file_path, 'w', encoding='utf-8') as f:
            f.write(code)

        if not shutil.which("docker"):
            return _execute_local_temp_python(temp_file_path, temp_dir)

        try:
            docker_command = [
                "docker", "run", 
                "--rm", 
                "--network", "none", 
                "--memory", "256m", 
                "--cpus", "0.5", 
                "-v", f"{temp_dir}:/sandbox:ro", 
                "-w", "/sandbox", 
                "python:3.10-slim", 
                "python", "agent_script.py"
            ]
            
            result = subprocess.run(
                docker_command,
                capture_output=True,
                text=True,
                timeout=15 
            )
            
            output = result.stdout.strip()
            errors = result.stderr.strip()
            
            if result.returncode == 0:
                return f"Execution successful (Sandboxed).\\nOutput:\\n{output}"
            else:
                return f"Execution failed.\\nError:\\n{errors}"
                
        except subprocess.TimeoutExpired:
            return "Error: Code execution took too long and was terminated by the sandbox."
        except Exception as e:
            return f"Sandbox system error: {str(e)}"


def _execute_local_temp_python(script_path: str, temp_dir: str) -> str:
    """Fallback for hosts without Docker. Isolated by temp cwd, but not a security sandbox."""
    try:
        result = subprocess.run(
            [sys.executable or "python3", script_path],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=temp_dir,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return "Error: Code execution took too long and was terminated by the local fallback runner."
    except Exception as e:
        return f"Local Python execution error: {e}"

    output = result.stdout.strip()
    errors = result.stderr.strip()
    notice = "Docker not found; ran with local Python in a temporary directory instead."
    if result.returncode == 0:
        return f"Execution successful (Local fallback). {notice}\nOutput:\n{output}"
    return f"Execution failed (Local fallback). {notice}\nError:\n{errors}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace the first occurrence of old_string with new_string in a file."""
    p = _resolve_path(path)
    if not p.exists():
        return f"Error: File not found: {path}"
    try:
        content = p.read_text(encoding="utf-8", errors="replace")
        if old_string not in content:
            return f"Error: Text not found in {path}:\n{old_string[:200]}"
        count = content.count(old_string)
        if count > 1:
            return f"Error: Found {count} occurrences of the text. Provide more context to make it unique."
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {e}"


def bash(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return stdout + stderr."""
    if timeout > 120:
        timeout = 120
    command = _normalize_shell_command(command)
    if _should_detach_background_command(command):
        return _run_detached_background_command(command)
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_project_root,
            stdin=subprocess.DEVNULL,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            recovered = _recover_git_checkout_existing_branch(command, result.stderr, timeout)
            if recovered is not None:
                return recovered
        if result.returncode != 0:
            recovered = _recover_typescript_ts5112(command, output, timeout)
            if recovered is not None:
                return recovered
        if result.returncode != 0 and _is_nonfatal_process_probe(command, output):
            output = output.strip()
            return output or "No matching process or listening port found."
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        output = output.strip() or "(no output)"
        if result.returncode != 0:
            return f"Error: {output}"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def _should_detach_background_command(command: str) -> bool:
    lowered = command.lower()
    if "&" not in command:
        return False
    background_markers = (
        "nohup ",
        "python",
        "uvicorn",
        "flask",
        "npm run",
        "vite",
        "server.py",
    )
    return any(marker in lowered for marker in background_markers)


def _run_detached_background_command(command: str) -> str:
    try:
        subprocess.Popen(
            command,
            shell=True,
            cwd=_project_root,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return (
            "Started background command without waiting for it to exit. "
            "Use a separate verification command such as `sleep 1 && curl ...`, "
            "`ps`, or `cat server.log` to confirm it is running."
        )
    except Exception as e:
        return f"Error: failed to start background command: {e}"


def _normalize_shell_command(command: str) -> str:
    """Normalize common copied Unicode punctuation that breaks shell/git commands."""
    normalized = (
        command
        .replace("\u2014-", "--")
        .replace("\u2013-", "--")
        .replace("\u2014", "--")
        .replace("\u2013", "--")
    )
    normalized = _normalize_process_command_for_platform(normalized)
    return normalized


def _normalize_process_command_for_platform(command: str) -> str:
    if _is_windows_host():
        return _normalize_windows_process_command(command)
    return _normalize_posix_process_command(command)


def _normalize_posix_process_command(command: str) -> str:
    normalized = command
    normalized = re.sub(r"\blsof\s+-ti:(\d+)\b", r"lsof -ti tcp:\1", normalized)
    normalized = re.sub(
        r"\bfuser\s+-k\s+(\d+)/tcp\b\s*(?:2>/dev/null)?\s*(?:\|\|\s*true)?",
        r"lsof -ti tcp:\1 | xargs kill -9 2>/dev/null || true",
        normalized,
    )
    normalized = re.sub(
        r"\bss\s+-tlnp\s*\|\s*grep\s+:?(\d+)\b",
        r"lsof -nP -iTCP:\1 -sTCP:LISTEN",
        normalized,
    )
    normalized = re.sub(
        r"\bnetstat\s+-tlnp(?:\s+2>/dev/null)?\s*\|\s*grep\s+:?(\d+)\b",
        r"lsof -nP -iTCP:\1 -sTCP:LISTEN",
        normalized,
    )
    return normalized


def _normalize_windows_process_command(command: str) -> str:
    port = _extract_port_from_process_command(command)
    if port is None:
        return command

    lowered = command.lower()
    wants_kill = (
        "fuser" in lowered
        or "xargs kill" in lowered
        or lowered.startswith("pkill ")
        or " pkill " in lowered
        or "kill -9" in lowered
    )
    if wants_kill:
        return (
            "powershell -NoProfile -ExecutionPolicy Bypass -Command "
            f"\"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
            "ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }\""
        )
    if any(marker in lowered for marker in ("lsof", "fuser", "ss ", "netstat")):
        return (
            "powershell -NoProfile -ExecutionPolicy Bypass -Command "
            f"\"Get-NetTCPConnection -LocalPort {port} -ErrorAction SilentlyContinue | "
            "Select-Object LocalAddress,LocalPort,State,OwningProcess\""
        )
    return command


def _extract_port_from_process_command(command: str) -> str | None:
    patterns = (
        r"tcp:(\d+)",
        r":(\d+)",
        r"\b(\d+)/tcp\b",
        r"localport\s+(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, command, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _is_windows_host() -> bool:
    return platform.system().lower().startswith("win")


def _is_nonfatal_process_probe(command: str, output: str) -> bool:
    lowered = command.lower()
    if "|| true" in lowered:
        return True
    if lowered.startswith("pkill ") or " pkill " in lowered:
        return True
    if "get-nettcpconnection" in lowered or "stop-process" in lowered:
        return True
    if ("lsof " in lowered or "grep" in lowered) and not output.strip():
        return True
    return False


def _recover_git_checkout_existing_branch(command: str, stderr: str, timeout: int) -> str | None:
    """Turn `git checkout -b existing` into `git checkout existing` for repeatable tests."""
    if "a branch named" not in stderr or "already exists" not in stderr:
        return None
    match = re.search(r"\bgit\s+checkout\s+-b\s+([^\s;&|]+)", command)
    if not match:
        return None
    branch = match.group(1).strip("'\"")
    repaired = re.sub(
        r"\bgit\s+checkout\s+-b\s+([^\s;&|]+)",
        f"git checkout {branch}",
        command,
        count=1,
    )
    try:
        retry = subprocess.run(
            repaired,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_project_root,
        )
    except Exception as e:
        return f"Error: {stderr.strip()}\nRecovery failed: {e}"

    retry_output = ((retry.stdout or "") + (retry.stderr or "")).strip()
    note = (
        f"Recovered: branch '{branch}' already existed, so ran `{repaired}` instead."
    )
    if retry.returncode != 0:
        return f"Error: {stderr.strip()}\n{note}\n{retry_output}\n[Exit code: {retry.returncode}]"
    return "\n".join(part for part in [note, retry_output] if part)


def _recover_typescript_ts5112(command: str, output: str, timeout: int) -> str | None:
    """Recover from TS5112 by running ts-node with an explicit project config."""
    if "TS5112" not in output and "tsconfig.json is present" not in output:
        return None
    match = re.search(r"\bnpx\s+ts-node\s+(?!.*--project)([^\s;&|]+\.tsx?)\b", command)
    if not match:
        return None
    file_arg = match.group(1)
    repaired = command.replace(f"npx ts-node {file_arg}", f"npx ts-node --project tsconfig.json {file_arg}", 1)
    try:
        retry = subprocess.run(
            repaired,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_project_root,
            stdin=subprocess.DEVNULL,
        )
    except Exception as e:
        return f"Error: {output.strip()}\nRecovery failed: {e}"

    retry_output = ((retry.stdout or "") + (retry.stderr or "")).strip()
    note = (
        "Recovered from TS5112 by running ts-node with `--project tsconfig.json`."
    )
    if retry.returncode != 0:
        return f"Error: {output.strip()}\n{note}\n{retry_output}\n[Exit code: {retry.returncode}]"
    return "\n".join(part for part in [note, retry_output] if part)


def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern."""
    base = _resolve_path(path)
    try:
        matches = sorted(base.rglob(pattern.lstrip("**/").lstrip("/")))
        if not matches:
            # Try the pattern as-is
            matches = sorted(base.glob(pattern))
        if not matches:
            return f"No files found matching: {pattern}"
        lines = [str(m.relative_to(base)) for m in matches if m.is_file()]
        if not lines:
            return f"No files found matching: {pattern}"
        return "\n".join(lines[:500])
    except Exception as e:
        return f"Error: {e}"


def grep_search(
    pattern: str,
    path: str = ".",
    file_glob: str = "*",
    case_insensitive: bool = False,
    context: int = 0,
) -> str:
    """Search file contents using regex."""
    flags = re.IGNORECASE if case_insensitive else 0
    base = _resolve_path(path)
    results = []
    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        return f"Invalid regex: {e}"

    try:
        all_files = [f for f in base.rglob("*") if f.is_file()]
        matched_files = [f for f in all_files if fnmatch.fnmatch(f.name, file_glob)]
    except Exception as e:
        return f"Error listing files: {e}"

    for filepath in sorted(matched_files)[:1000]:
        try:
            lines = filepath.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for i, line in enumerate(lines):
            if compiled.search(line):
                rel = filepath.relative_to(base)
                if context > 0:
                    start = max(0, i - context)
                    end = min(len(lines), i + context + 1)
                    block = [f"{rel}:{j+1}: {lines[j]}" for j in range(start, end)]
                    results.append("\n".join(block))
                else:
                    results.append(f"{rel}:{i+1}: {line}")

    if not results:
        return f"No matches for: {pattern}"
    if len(results) > 200:
        results = results[:200]
        results.append(f"... (truncated, showing 200 of many matches)")
    return "\n".join(results)


def github_sync(commit_message: str, branch_name: str = "main", remote_url: str | None = None) -> str:
    """Advanced Git sync: pulls before pushing, handles new repos, and manages conflicts."""
    def run_cmd(cmd: str) -> tuple[int, str, str]:
        cwd = _project_root or os.getcwd()
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, shell=True)
        return result.returncode, result.stdout.strip(), result.stderr.strip()

    try:
        # Check if it is a git repository
        code, out, err = run_cmd("git rev-parse --is-inside-work-tree")
        is_existing_repo = (code == 0)

        if not is_existing_repo:
            if not remote_url:
                return "Error: This is not a git repository. Provide a remote_url to initialize and sync."
            run_cmd("git init")
            run_cmd(f"git branch -M {branch_name}")
            run_cmd(f"git remote add origin {remote_url}")
        else:
            # For existing repos, pull first to avoid conflicts with human developers
            # Use --rebase to keep a clean history
            pull_code, pull_out, pull_err = run_cmd(f"git pull --rebase origin {branch_name}")
            if pull_code != 0:
                lowered_err = pull_err.lower()
                if "could not resolve host" in lowered_err or "access denied" in lowered_err:
                    return f"Error: Connectivity or access issue during pull. {pull_err}"
                # If there is a merge conflict, stop and tell the AI
                if "conflict" in pull_out.lower() or "conflict" in pull_err.lower():
                    run_cmd("git rebase --abort")
                    return f"Error: Git pull resulted in a merge conflict. Cannot sync automatically. {pull_err}"

        # Stage all changes
        run_cmd("git add .")
        
        # Check if there is anything to commit
        code, out, err = run_cmd("git status --porcelain")
        if not out:
            return "Everything is up to date. Nothing to commit."

        # Commit
        safe_message = commit_message.replace('"', '\\"')
        code, out, err = run_cmd(f'git commit -m "{safe_message}"')
        if code != 0:
             return f"Commit failed: {err}"
        
        # Push
        code, out, err = run_cmd(f"git push -u origin {branch_name}")
        if code != 0: 
            return f"Push failed. Check authentication or branch names. Error: {err}"

        return f"Success! Code synced to GitHub on branch '{branch_name}'."

    except Exception as e:
        return f"Unexpected Git error: {str(e)}"


def list_dir(path: str = ".") -> str:
    """List directory contents."""
    p = _resolve_path(path)
    if not p.exists():
        return f"Error: Path not found: {path}"
    if not p.is_dir():
        return f"Error: Not a directory: {path}"
    try:
        entries = sorted(p.iterdir(), key=lambda e: (not e.is_dir(), e.name))
        lines = []
        for e in entries[:500]:
            prefix = "d " if e.is_dir() else "f "
            size = ""
            if e.is_file():
                try:
                    size = f" ({e.stat().st_size:,} bytes)"
                except Exception:
                    pass
            lines.append(f"{prefix}{e.name}{size}")
        return "\n".join(lines) or "(empty directory)"
    except Exception as e:
        return f"Error: {e}"


def _infer_macos_backend_command(root: Path) -> str:
    if (root / "api.py").exists():
        return "python3 -m uvicorn api:app --host 127.0.0.1 --port {port}"

    package_json = root / "package.json"
    if package_json.exists():
        try:
            data = json.loads(package_json.read_text(encoding="utf-8"))
            scripts = data.get("scripts", {})
        except Exception:
            scripts = {}
        if "dev" in scripts:
            return "PORT={port} npm run dev"
        if "start" in scripts:
            return "PORT={port} npm start"

    return ""


def _join_local_url(base: str, path_part: str) -> str:
    if path_part.startswith("http://") or path_part.startswith("https://"):
        return path_part
    return base.rstrip("/") + "/" + path_part.lstrip("/")


def _safe_product_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 ._-]+", "", name).strip()
    return cleaned or "Clod App"


def _safe_app_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "clod-app"
    return f"com.clod.{slug}"


def _macos_package_json(app_name: str, app_dir: str) -> dict:
    product_name = _safe_product_name(app_name)
    return {
        "name": re.sub(r"[^a-z0-9-]+", "-", product_name.lower()).strip("-") or "clod-app",
        "version": "0.1.0",
        "description": f"macOS desktop shell for {product_name}",
        "main": "main.js",
        "private": True,
        "scripts": {
            "start": "electron .",
            "package:mac": "electron-builder --mac --dir",
            "dist:mac": "electron-builder --mac dmg zip",
        },
        "build": {
            "appId": _safe_app_id(product_name),
            "productName": product_name,
            "mac": {
                "category": "public.app-category.developer-tools",
                "target": ["dir", "dmg", "zip"],
            },
            "files": ["main.js", "package.json"],
            "extraResources": [
                {
                    "from": "../",
                    "to": "app",
                    "filter": [
                        "**/*",
                        f"!{app_dir}/**",
                        "!node_modules/**",
                        "!.git/**",
                        "!venv/**",
                        "!.venv/**",
                        "!__pycache__/**",
                        "!*.pyc",
                        "!.DS_Store",
                    ],
                }
            ],
        },
        "devDependencies": {
            "electron": "^30.5.1",
            "electron-builder": "^24.13.3",
        },
    }


def _macos_main_js(app_name: str, backend_command: str, start_url: str, health_url: str, port: int) -> str:
    product = _safe_product_name(app_name)
    return f'''const {{ app, BrowserWindow, Menu, dialog, shell }} = require("electron");
const {{ spawn }} = require("child_process");
const fs = require("fs");
const net = require("net");
const path = require("path");

const PRODUCT_NAME = {json.dumps(product)};
const DEFAULT_PORT = Number(process.env.MACOS_APP_PORT || {port});
const BACKEND_COMMAND = {json.dumps(backend_command)};
const START_URL_TEMPLATE = {json.dumps(start_url)};
const HEALTH_URL_TEMPLATE = {json.dumps(health_url)};

let backendProcess = null;
let mainWindow = null;

function appRoot() {{
  if (app.isPackaged) return path.join(process.resourcesPath, "app");
  return path.resolve(__dirname, "..");
}}

function renderTemplate(value, port, root = appRoot()) {{
  const python = process.env.CLOD_PYTHON || "python3";
  return String(value || "")
    .replace(/\\{{port\\}}/g, String(port))
    .replace(/\\{{project_dir\\}}/g, root)
    .replace(/\\{{project_root\\}}/g, root)
    .replace(/\\{{python\\}}/g, python);
}}

function isPortOpen(port) {{
  return new Promise((resolve) => {{
    const socket = net.createConnection({{ host: "127.0.0.1", port }});
    socket.once("connect", () => {{
      socket.destroy();
      resolve(true);
    }});
    socket.once("error", () => resolve(false));
    socket.setTimeout(500, () => {{
      socket.destroy();
      resolve(false);
    }});
  }});
}}

async function findPort(startPort) {{
  for (let port = startPort; port < startPort + 50; port += 1) {{
    if (!(await isPortOpen(port))) return port;
  }}
  throw new Error(`No free localhost port found near ${{startPort}}`);
}}

async function sleep(ms) {{
  return new Promise((resolve) => setTimeout(resolve, ms));
}}

async function waitForHealth(url, timeoutMs = 30000) {{
  if (!url) {{
    await sleep(1800);
    return;
  }}
  const startedAt = Date.now();
  let lastError = "";
  while (Date.now() - startedAt < timeoutMs) {{
    try {{
      const response = await fetch(url);
      if (response.ok) return;
      lastError = `HTTP ${{response.status}}`;
    }} catch (error) {{
      lastError = error.message;
    }}
    await sleep(500);
  }}
  throw new Error(`Backend did not become healthy: ${{lastError}}`);
}}

async function startBackend() {{
  if (!BACKEND_COMMAND) return renderTemplate(START_URL_TEMPLATE, DEFAULT_PORT);

  const port = await findPort(DEFAULT_PORT);
  const root = appRoot();
  const command = renderTemplate(BACKEND_COMMAND, port, root);
  const startUrl = renderTemplate(START_URL_TEMPLATE, port, root);
  const healthUrl = renderTemplate(HEALTH_URL_TEMPLATE, port, root);
  const env = {{ ...process.env, PORT: String(port), PYTHONUNBUFFERED: "1" }};

  backendProcess = spawn(command, {{
    cwd: root,
    env,
    shell: true,
    stdio: ["ignore", "pipe", "pipe"]
  }});

  backendProcess.stdout.on("data", (data) => process.stdout.write(`[app-backend] ${{data}}`));
  backendProcess.stderr.on("data", (data) => process.stderr.write(`[app-backend] ${{data}}`));
  await waitForHealth(healthUrl);
  return startUrl;
}}

function createMenu() {{
  const template = [
    {{
      label: PRODUCT_NAME,
      submenu: [
        {{ role: "about" }},
        {{ type: "separator" }},
        {{ role: "quit" }}
      ]
    }},
    {{
      label: "Edit",
      submenu: [
        {{ role: "undo" }},
        {{ role: "redo" }},
        {{ type: "separator" }},
        {{ role: "cut" }},
        {{ role: "copy" }},
        {{ role: "paste" }},
        {{ role: "pasteAndMatchStyle" }},
        {{ role: "delete" }},
        {{ type: "separator" }},
        {{ role: "selectAll" }}
      ]
    }},
    {{
      label: "View",
      submenu: [
        {{ role: "reload" }},
        {{ role: "forceReload" }},
        {{ type: "separator" }},
        {{ role: "toggleDevTools" }},
        {{ type: "separator" }},
        {{ role: "resetZoom" }},
        {{ role: "zoomIn" }},
        {{ role: "zoomOut" }}
      ]
    }},
    {{
      label: "Help",
      submenu: [
        {{ label: "Open Project Folder", click: () => shell.openPath(appRoot()) }}
      ]
    }}
  ];
  Menu.setApplicationMenu(Menu.buildFromTemplate(template));
}}

function installContextMenu(win) {{
  win.webContents.on("context-menu", (_event, params) => {{
    const template = [];
    if (params.isEditable) {{
      template.push(
        {{ role: "undo", enabled: params.editFlags.canUndo }},
        {{ role: "redo", enabled: params.editFlags.canRedo }},
        {{ type: "separator" }},
        {{ role: "cut", enabled: params.editFlags.canCut }},
        {{ role: "copy", enabled: params.editFlags.canCopy }},
        {{ role: "paste", enabled: params.editFlags.canPaste }},
        {{ role: "pasteAndMatchStyle", enabled: params.editFlags.canPaste }},
        {{ role: "delete", enabled: params.editFlags.canDelete }},
        {{ type: "separator" }},
        {{ role: "selectAll", enabled: params.editFlags.canSelectAll }}
      );
    }} else {{
      template.push(
        {{ role: "copy", enabled: params.selectionText.length > 0 }},
        {{ role: "selectAll" }}
      );
    }}
    Menu.buildFromTemplate(template).popup({{ window: win }});
  }});
}}

async function createWindow() {{
  mainWindow = new BrowserWindow({{
    width: 1280,
    height: 860,
    minWidth: 900,
    minHeight: 640,
    title: PRODUCT_NAME,
    backgroundColor: "#101114",
    webPreferences: {{
      contextIsolation: true,
      nodeIntegration: false
    }}
  }});
  installContextMenu(mainWindow);

  try {{
    const url = await startBackend();
    await mainWindow.loadURL(url);
  }} catch (error) {{
    await dialog.showMessageBox({{
      type: "error",
      title: `${{PRODUCT_NAME}} could not start`,
      message: "The local app backend could not be started.",
      detail: error.stack || error.message
    }});
    app.quit();
  }}
}}

function stopBackend() {{
  if (backendProcess && !backendProcess.killed) backendProcess.kill("SIGTERM");
}}

app.whenReady().then(() => {{
  createMenu();
  createWindow();
  app.on("activate", () => {{
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  }});
}});

app.on("before-quit", stopBackend);
app.on("window-all-closed", () => {{
  if (process.platform !== "darwin") app.quit();
}});
'''


def _macos_readme(app_name: str, app_dir: str, backend_command: str, start_url: str) -> str:
    product = _safe_product_name(app_name)
    return f"""# {product} Desktop

Electron macOS shell generated by Clod.

## Development

```bash
cd {app_dir}
npm install
npm start
```

## Build

```bash
bash scripts/setup-macos-app.sh
```

The generated app is expected at:

```text
{app_dir}/dist/mac-arm64/{product}.app
```

Backend command:

```text
{backend_command or '(none; loads URL directly)'}
```

Start URL:

```text
{start_url}
```
"""


def _macos_setup_script(app_dir: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

# Syntax check
bash -n "${{BASH_SOURCE[0]}}" || exit 1

ROOT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")/.." && pwd)"
DESKTOP_DIR="$ROOT_DIR/{app_dir}"

cd "$DESKTOP_DIR"

if [ ! -d "node_modules" ]; then
  npm install
else
  echo "node_modules exists. Skipping npm install."
fi

echo "Cleaning previous build..."
rm -rf dist

echo "Packaging macOS app..."
npm run package:mac

echo "macOS app ready under: $DESKTOP_DIR/dist"
"""


def _ensure_gitignore_entries(root: Path, entries: list[str]) -> None:
    path = root / ".gitignore"
    existing = path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []
    normalized = {line.strip() for line in existing}
    missing = [entry for entry in entries if entry not in normalized]
    if not missing:
        return
    content = "\n".join(existing).rstrip()
    if content:
        content += "\n"
    content += "\n".join(missing) + "\n"
    path.write_text(content, encoding="utf-8")
