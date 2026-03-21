"""
Context management: keep conversation history within token budget.
Uses a simple heuristic (4 chars ≈ 1 token) to estimate usage.
Trims oldest non-system messages when budget is exceeded.
"""
from ..providers.base import Message

# Rough chars-per-token estimate
CHARS_PER_TOKEN = 4

# Default max tokens to send (leave headroom for response)
DEFAULT_MAX_TOKENS = 80_000


def estimate_tokens(text: str) -> int:
    return max(1, len(str(text)) // CHARS_PER_TOKEN)


def message_tokens(msg: Message) -> int:
    total = estimate_tokens(str(msg.content))
    for tc in msg.tool_calls:
        total += estimate_tokens(str(tc.arguments))
    for tr in msg.tool_results:
        total += estimate_tokens(tr.content)
    return total


def trim_history(
    history: list[Message],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    keep_last_n: int = 4,
) -> list[Message]:
    """
    Trim conversation history to fit within token budget.
    Always keeps the last `keep_last_n` message pairs.
    """
    if not history:
        return history

    total = sum(message_tokens(m) for m in history)
    if total <= max_tokens:
        return history

    # Keep last N messages unconditionally
    protected = history[-keep_last_n:]
    trimmable = list(history[:-keep_last_n])

    while trimmable and total > max_tokens:
        removed = trimmable.pop(0)
        total -= message_tokens(removed)

    return trimmable + protected
