"""Load project markdown instructions and skill files."""
from pathlib import Path

MAX_INSTRUCTION_CHARS = 40_000


def load_project_instructions(project_root: str | None) -> str:
    """Load AGENTS.md, CLAUDE.md, and skills/*/SKILL.md from a project."""
    if not project_root:
        return ""

    root = Path(project_root).expanduser()
    if not root.is_dir():
        return ""

    parts: list[str] = []
    for path in _instruction_paths(root):
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            continue
        if not text:
            continue
        rel = path.relative_to(root)
        parts.append(f"## {rel}\n{_strip_front_matter(text)}")

    body = "\n\n".join(parts).strip()
    if len(body) > MAX_INSTRUCTION_CHARS:
        body = body[:MAX_INSTRUCTION_CHARS] + "\n\n[Project instructions truncated]"
    return body


def list_project_instruction_files(project_root: str | None) -> list[str]:
    """Return visible project instruction and skill markdown files."""
    if not project_root:
        return []
    root = Path(project_root).expanduser()
    if not root.is_dir():
        return []
    return [str(path.relative_to(root)) for path in _instruction_paths(root) if path.exists()]


def _instruction_paths(root: Path) -> list[Path]:
    paths = [root / "AGENTS.md", root / "CLAUDE.md"]
    skills_dir = root / "skills"
    if skills_dir.is_dir():
        paths.extend(sorted(skills_dir.glob("*/SKILL.md")))
    return paths


def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    lines = text.splitlines()
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            return "\n".join(lines[i + 1:]).strip()
    return text
