"""
Context management: keep conversation history within token budget.

Strategy (v2):
1. Estimate tokens with a 4-char heuristic.
2. If under budget, return history as-is.
3. Else: keep the *first user message* (the original task anchor) and the
   last `keep_last_n` messages (recent context). Drop or summarize the middle.
4. Dropped tool results are replaced by a one-line summary so the model still
   knows what was probed without the full payload.

This is a simplification of full semantic compaction but recovers ~30-50%
context budget on long sessions where stale tool output dominates.
"""
from __future__ import annotations

from ..providers.base import Message, ToolResult

CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 80_000

# Cap a single tool result body when summarizing (head + tail keeps signal).
TOOL_RESULT_HEAD_CHARS = 400
TOOL_RESULT_TAIL_CHARS = 200


def estimate_tokens(text: str) -> int:
    return max(1, len(str(text)) // CHARS_PER_TOKEN)


def message_tokens(msg: Message) -> int:
    total = estimate_tokens(str(msg.content))
    for tc in msg.tool_calls:
        total += estimate_tokens(str(tc.arguments))
    for tr in msg.tool_results:
        total += estimate_tokens(tr.content)
    return total


def _summarize_tool_result(tr: ToolResult) -> ToolResult:
    """Trim a tool result to head+tail so the gist survives compaction."""
    body = str(tr.content or "")
    if len(body) <= TOOL_RESULT_HEAD_CHARS + TOOL_RESULT_TAIL_CHARS + 80:
        return tr
    head = body[:TOOL_RESULT_HEAD_CHARS]
    tail = body[-TOOL_RESULT_TAIL_CHARS:]
    omitted = len(body) - len(head) - len(tail)
    summary = f"{head}\n\n[... {omitted} chars elided by context compaction ...]\n\n{tail}"
    return ToolResult(
        tool_call_id=tr.tool_call_id,
        content=summary,
        is_error=tr.is_error,
    )


def _compact_message(msg: Message) -> Message:
    """Return a compacted copy of `msg` with summarized tool results."""
    if not msg.tool_results:
        return msg
    return Message(
        role=msg.role,
        content=msg.content,
        tool_calls=msg.tool_calls,
        tool_results=[_summarize_tool_result(tr) for tr in msg.tool_results],
        provider=getattr(msg, "provider", ""),
        model=getattr(msg, "model", ""),
    )


def trim_history(
    history: list[Message],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    keep_last_n: int = 4,
) -> list[Message]:
    """
    Trim conversation history to fit within token budget.

    Always preserves:
    - The first user message (the original task anchor).
    - The last `keep_last_n` messages (recent context).

    Middle messages are first compacted (tool results summarized), then
    dropped from oldest to newest until the budget fits.
    """
    if not history:
        return history

    total = sum(message_tokens(m) for m in history)
    if total <= max_tokens:
        return history

    # Identify the anchor: first user message in history.
    anchor_idx = next(
        (i for i, m in enumerate(history) if m.role == "user"),
        None,
    )

    if len(history) <= keep_last_n + (1 if anchor_idx is not None else 0):
        return history

    tail = list(history[-keep_last_n:])
    if anchor_idx is not None and anchor_idx < len(history) - keep_last_n:
        anchor = [history[anchor_idx]]
        middle = list(history[anchor_idx + 1 : -keep_last_n])
    else:
        anchor = []
        middle = list(history[: -keep_last_n])

    # Pass 1: compact tool-result heavy messages in the middle.
    middle = [_compact_message(m) for m in middle]

    def total_tokens() -> int:
        return sum(message_tokens(m) for m in anchor + middle + tail)

    # Pass 2: drop oldest middle messages until under budget.
    while middle and total_tokens() > max_tokens:
        middle.pop(0)

    return anchor + middle + tail
