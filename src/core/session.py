"""Session persistence: save and load conversation history."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..providers.base import Message, ToolCall, ToolResult

SESSIONS_DIR = Path.home() / ".cclaude" / "sessions"


def _serialize_message(msg: Message) -> dict:
    """Convert a Message to a JSON-serializable dict."""
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ],
        "tool_results": [
            {"tool_call_id": tr.tool_call_id, "content": tr.content, "is_error": tr.is_error}
            for tr in msg.tool_results
        ],
    }


def _deserialize_message(data: dict) -> Message:
    """Reconstruct a Message from a dict."""
    return Message(
        role=data["role"],
        content=data["content"],
        tool_calls=[
            ToolCall(id=tc["id"], name=tc["name"], arguments=tc["arguments"])
            for tc in data.get("tool_calls", [])
        ],
        tool_results=[
            ToolResult(
                tool_call_id=tr["tool_call_id"],
                content=tr["content"],
                is_error=tr.get("is_error", False),
            )
            for tr in data.get("tool_results", [])
        ],
    )


def save_session(
    session_id: str,
    messages: list[Message],
    provider: str = "",
    model: str = "",
    project: str = "",
) -> str:
    """Save conversation history to disk. Returns the file path."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    filepath = SESSIONS_DIR / f"{session_id}.json"

    # Preserve existing metadata if updating
    existing: dict[str, Any] = {}
    if filepath.exists():
        try:
            existing = json.loads(filepath.read_text())
        except Exception:
            pass

    data = {
        "id": session_id,
        "created": existing.get("created", datetime.now(timezone.utc).isoformat()),
        "updated": datetime.now(timezone.utc).isoformat(),
        "provider": provider or existing.get("provider", ""),
        "model": model or existing.get("model", ""),
        "project": project or existing.get("project", ""),
        "messages": [_serialize_message(m) for m in messages],
    }

    filepath.write_text(json.dumps(data, indent=2))
    return str(filepath)


def load_session(session_id: str) -> tuple[list[Message], dict]:
    """Load a session from disk. Returns (messages, metadata)."""
    filepath = SESSIONS_DIR / f"{session_id}.json"
    if not filepath.exists():
        raise FileNotFoundError(f"Session not found: {session_id}")

    data = json.loads(filepath.read_text())
    messages = [_deserialize_message(m) for m in data.get("messages", [])]
    metadata = {k: v for k, v in data.items() if k != "messages"}
    return messages, metadata


def list_sessions() -> list[dict]:
    """List all saved sessions with metadata (no messages)."""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(f.read_text())
            msg_count = len(data.get("messages", []))
            sessions.append({
                "id": data.get("id", f.stem),
                "created": data.get("created", ""),
                "updated": data.get("updated", ""),
                "provider": data.get("provider", ""),
                "model": data.get("model", ""),
                "project": data.get("project", ""),
                "messages": msg_count,
            })
        except Exception:
            continue
    return sessions


def delete_session(session_id: str) -> bool:
    """Delete a saved session. Returns True if deleted."""
    filepath = SESSIONS_DIR / f"{session_id}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def get_last_session_id(project: str = "") -> str | None:
    """Get the most recently updated session ID, optionally filtered by project."""
    sessions = list_sessions()
    if project:
        sessions = [s for s in sessions if s.get("project") == project]
    return sessions[0]["id"] if sessions else None
