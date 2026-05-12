"""Mid-run clarification prompts.

Inspired by hermes-webui's clarify flow. The agent emits a clarification
request, the loop blocks (with timeout) until the user posts a response via
the API, then the loop resumes with the response folded into history as a
synthetic user message.

A clarify entry de-duplicates on (question, choices) so the same question
isn't queued twice while still pending.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

DEFAULT_TIMEOUT_SECONDS = 120

_lock = threading.Lock()
_queues: dict[str, list["ClarifyEntry"]] = {}


@dataclass
class ClarifyEntry:
    session_key: str
    question: str
    choices: list[str] = field(default_factory=list)
    requested_at: float = field(default_factory=time.time)
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[str] = None

    @property
    def expires_at(self) -> float:
        return self.requested_at + self.timeout_seconds

    def to_dict(self) -> dict:
        return {
            "session_key": self.session_key,
            "question": self.question,
            "choices": list(self.choices),
            "requested_at": self.requested_at,
            "timeout_seconds": self.timeout_seconds,
            "expires_at": self.expires_at,
            "resolved": self.event.is_set(),
            "response": self.response,
        }


def submit(session_key: str, question: str, choices: list[str] | None = None,
           timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> ClarifyEntry:
    """Queue a clarify request. Returns the entry; caller waits on entry.event."""
    choices = list(choices or [])
    with _lock:
        q = _queues.setdefault(session_key, [])
        # De-dup against the most recent unresolved entry.
        if q and not q[-1].event.is_set():
            last = q[-1]
            if last.question == question and last.choices == choices:
                return last
        entry = ClarifyEntry(
            session_key=session_key,
            question=question,
            choices=choices,
            timeout_seconds=timeout_seconds,
        )
        q.append(entry)
        return entry


def wait_for(entry: ClarifyEntry) -> Optional[str]:
    """Block until the entry is resolved or times out."""
    remaining = max(1.0, entry.expires_at - time.time())
    entry.event.wait(timeout=remaining)
    return entry.response


def resolve(session_key: str, response: str) -> int:
    """Resolve the oldest pending entry; return number resolved (0 or 1)."""
    with _lock:
        q = _queues.get(session_key) or []
        for entry in q:
            if not entry.event.is_set():
                entry.response = response
                entry.event.set()
                return 1
    return 0


def peek(session_key: str) -> Optional[dict]:
    with _lock:
        q = _queues.get(session_key) or []
        for entry in q:
            if not entry.event.is_set():
                return entry.to_dict()
    return None


def clear(session_key: str) -> int:
    """Cancel all pending entries for the session."""
    with _lock:
        q = _queues.pop(session_key, [])
    n = 0
    for entry in q:
        if not entry.event.is_set():
            entry.event.set()
            n += 1
    return n
