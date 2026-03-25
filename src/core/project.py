"""Project root detection and project-based working directories."""
import os
from pathlib import Path

PROJECT_MARKERS = (
    ".git",
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "pom.xml",
    ".cclaude-project",
)


def detect_project_root(start: str | None = None) -> str | None:
    """Walk up from start (or cwd) looking for a project marker.
    Returns the project root path or None if not found.
    """
    current = Path(start or os.getcwd()).resolve()
    home = Path.home()

    while current != current.parent:
        for marker in PROJECT_MARKERS:
            if (current / marker).exists():
                return str(current)
        # Don't go above home directory
        if current == home:
            break
        current = current.parent

    return None


def project_name(root: str) -> str:
    """Extract a short project name from the root path."""
    return Path(root).name
