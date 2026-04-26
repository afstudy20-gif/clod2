"""Tool implementations - filesystem, shell, and search operations."""
import fnmatch
import os
import platform
import re
import subprocess
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
        guard_error = _guard_env_secret_overwrite(p, content)
        if guard_error:
            return guard_error
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


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
        guard_error = _guard_env_secret_overwrite(p, new_content)
        if guard_error:
            return guard_error
        p.write_text(new_content, encoding="utf-8")
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing file: {e}"


def _guard_env_secret_overwrite(path: Path, new_content: str) -> str | None:
    """Prevent agents from replacing real .env secrets with placeholders."""
    if path.name != ".env" or not path.exists():
        return None
    try:
        old_content = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    old_values = _parse_env_assignments(old_content)
    new_values = _parse_env_assignments(new_content)
    for key, old_value in old_values.items():
        if key not in new_values:
            continue
        new_value = new_values[key]
        if _looks_like_real_secret(old_value) and _looks_like_placeholder_secret(new_value):
            return (
                f"Error: Refusing to replace existing .env secret `{key}` with a placeholder. "
                "Keep the user's real key unchanged, or ask the user before modifying secrets."
            )
    return None


def _parse_env_assignments(content: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key:
            values[key] = value
    return values


def _looks_like_placeholder_secret(value: str) -> bool:
    lowered = value.strip().lower()
    if not lowered:
        return True
    return any(
        marker in lowered
        for marker in (
            "placeholder",
            "insert_your",
            "your_api_key",
            "your-key",
            "your_key",
            "replace_me",
            "changeme",
            "todo",
            "example",
        )
    )


def _looks_like_real_secret(value: str) -> bool:
    stripped = value.strip()
    if _looks_like_placeholder_secret(stripped):
        return False
    if len(stripped) < 12:
        return False
    return bool(re.search(r"[A-Za-z]", stripped) and re.search(r"\d|[_\-]", stripped))


def bash(command: str, timeout: int = 30) -> str:
    """Execute a shell command and return stdout + stderr."""
    if timeout > 120:
        timeout = 120
    command = _normalize_shell_command(command)
    if _is_persistent_cd_misuse(command):
        return (
            "Note: `cd` by itself only affects this one shell command and will not persist "
            "to the next tool call. Use `cd target_dir && <command>` in a single bash call, "
            "or use file tools with paths relative to the selected workspace."
        )
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
    if not re.search(r"(?<!&)&(?!&)", command):
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


def _is_persistent_cd_misuse(command: str) -> bool:
    """Detect a bare cd command that the agent may wrongly expect to persist."""
    stripped = command.strip()
    return bool(re.fullmatch(r"cd(?:\s+[^;&|]+)?", stripped))


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
