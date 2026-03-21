"""Tool implementations - filesystem, shell, and search operations."""
import fnmatch
import os
import re
import subprocess
from pathlib import Path


def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file and return its contents with line numbers."""
    p = Path(path).expanduser()
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
    p = Path(path).expanduser()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Replace the first occurrence of old_string with new_string in a file."""
    p = Path(path).expanduser()
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
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += result.stderr
        if result.returncode != 0:
            output += f"\n[Exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: Command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def glob_files(pattern: str, path: str = ".") -> str:
    """Find files matching a glob pattern."""
    base = Path(path).expanduser()
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
    base = Path(path).expanduser()
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
    p = Path(path).expanduser()
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
