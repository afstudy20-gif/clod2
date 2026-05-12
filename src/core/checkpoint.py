"""Git-based workspace checkpoints.

Inspired by hermes-webui's rollback module. Before destructive edits the
agent snapshots the workspace to ~/.clod/checkpoints/<workspace-hash>/<id>/
as a bare git repo containing the workspace tree at that moment. Restore
copies files back without touching the user's primary git repo.

This is intentionally simpler than hermes' implementation: we use a single
copy + git init per checkpoint rather than a shared bare repo, which avoids
edge cases around tracked/untracked files at the cost of disk space. Cleanup
is responsibility of a future GC task; for now we keep the most recent N.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Skip these directories when snapshotting — they bloat checkpoints fast and
# rarely contain content the user would want to roll back.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".pytest_cache",
    "dist", "build", ".next", ".cache", ".turbo", "target",
}
MAX_CHECKPOINT_FILES = 5000   # safety cap so a checkpoint never spirals
MAX_CHECKPOINTS_PER_WS = 20   # rolling retention


def _clod_home() -> Path:
    return Path.home() / ".clod"


def _checkpoint_root() -> Path:
    root = _clod_home() / "checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _workspace_hash(workspace: str) -> str:
    return hashlib.sha1(str(Path(workspace).resolve()).encode()).hexdigest()[:12]


def _resolve_workspace(workspace: str | None) -> Path:
    if not workspace:
        return Path.cwd()
    return Path(workspace).expanduser().resolve()


def _find_git() -> str | None:
    return shutil.which("git")


def _ws_dir(workspace: str | None) -> Path:
    ws_hash = _workspace_hash(str(_resolve_workspace(workspace)))
    d = _checkpoint_root() / ws_hash
    d.mkdir(parents=True, exist_ok=True)
    return d


def _enforce_retention(ws_dir: Path) -> None:
    """Drop oldest checkpoints beyond the retention cap."""
    entries = sorted(
        [p for p in ws_dir.iterdir() if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in entries[MAX_CHECKPOINTS_PER_WS:]:
        shutil.rmtree(old, ignore_errors=True)


def create_checkpoint(workspace: str | None, label: str = "") -> dict[str, Any]:
    """Snapshot the workspace into a new checkpoint dir; commit to inner git."""
    ws = _resolve_workspace(workspace)
    if not ws.is_dir():
        return {"ok": False, "error": f"Workspace not found: {ws}"}

    git = _find_git()
    if not git:
        return {"ok": False, "error": "git binary not found"}

    ts = time.strftime("%Y%m%dT%H%M%S")
    ckpt_id = f"{ts}_{_workspace_hash(str(ws))[:6]}"
    ckpt_dir = _ws_dir(str(ws)) / ckpt_id
    ckpt_dir.mkdir(parents=True, exist_ok=False)

    copied = 0
    for src in ws.rglob("*"):
        rel = src.relative_to(ws)
        # Skip if any path component matches SKIP_DIRS.
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        if src.is_dir():
            continue
        if copied >= MAX_CHECKPOINT_FILES:
            break
        dst = ckpt_dir / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
        except OSError:
            continue

    # Init inner git so we can diff and inspect later.
    try:
        subprocess.run([git, "-C", str(ckpt_dir), "init", "-q"],
                       check=False, timeout=10)
        subprocess.run([git, "-C", str(ckpt_dir), "add", "-A"],
                       check=False, timeout=30)
        msg = label or f"checkpoint {ts}"
        subprocess.run(
            [git, "-C", str(ckpt_dir), "-c", "user.email=clod@local",
             "-c", "user.name=clod", "commit", "-q", "-m", msg],
            check=False, timeout=30,
        )
    except (subprocess.TimeoutExpired, OSError) as e:
        return {"ok": False, "error": f"git init failed: {e}"}

    _enforce_retention(_ws_dir(str(ws)))

    return {
        "ok": True,
        "id": ckpt_id,
        "path": str(ckpt_dir),
        "label": label,
        "files": copied,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def list_checkpoints(workspace: str | None) -> list[dict[str, Any]]:
    ws_dir = _ws_dir(workspace)
    out = []
    git = _find_git()
    for p in sorted(ws_dir.iterdir(), key=lambda x: x.stat().st_mtime if x.is_dir() else 0, reverse=True):
        if not p.is_dir():
            continue
        info = {"id": p.name, "path": str(p), "files": 0, "label": ""}
        try:
            info["files"] = sum(1 for _ in p.rglob("*") if _.is_file() and ".git/" not in str(_))
        except OSError:
            pass
        if git and (p / ".git").is_dir():
            try:
                r = subprocess.run(
                    [git, "-C", str(p), "log", "-1", "--format=%s%n%aI"],
                    capture_output=True, text=True, timeout=5,
                )
                if r.returncode == 0:
                    parts = r.stdout.strip().split("\n", 1)
                    info["label"] = parts[0] if parts else ""
                    info["date"] = parts[1] if len(parts) > 1 else ""
            except (subprocess.TimeoutExpired, OSError):
                pass
        out.append(info)
    return out


def restore_checkpoint(workspace: str | None, checkpoint_id: str) -> dict[str, Any]:
    ws = _resolve_workspace(workspace)
    if not ws.is_dir():
        return {"ok": False, "error": f"Workspace not found: {ws}"}
    ckpt_dir = _ws_dir(str(ws)) / checkpoint_id
    if not ckpt_dir.is_dir():
        return {"ok": False, "error": f"Checkpoint not found: {checkpoint_id}"}

    restored = 0
    for src in ckpt_dir.rglob("*"):
        rel = src.relative_to(ckpt_dir)
        if any(part in {".git"} for part in rel.parts):
            continue
        if src.is_dir():
            continue
        dst = ws / rel
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            restored += 1
        except OSError:
            continue
    return {"ok": True, "restored": restored, "id": checkpoint_id}
