"""Pre-API history sanitization.

Inspired by hermes-webui's `_sanitize_messages_for_api` and
`_strip_thinking_markup` / `_strip_xml_tool_calls`. Some providers re-emit
`<thinking>` blocks or XML-style tool calls in assistant content. Re-sending
that to the next round bloats context and confuses some endpoints.

This module strips:
- `<thinking>...</thinking>` blocks (and `<thought>`, `<reasoning>`).
- Inline XML-style tool calls like `<tool_call>...</tool_call>`,
  `<function_call>...</function_call>`, `<tool>...</tool>`.
- Code fences whose tag is `tool_call` / `tool_use`.

It is intentionally non-destructive on prose and structured `tool_calls` /
`tool_results` fields — those still drive control flow.
"""
from __future__ import annotations

import re
from copy import copy

from ..providers.base import Message

_THINKING_RE = re.compile(
    r"<(thinking|thought|reasoning)>.*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
_XML_TOOL_RE = re.compile(
    r"<(tool_call|function_call|tool_use|tool)[\s>][^<]*?</\1>",
    flags=re.IGNORECASE | re.DOTALL,
)
_FENCE_TOOL_RE = re.compile(
    r"```(?:tool_call|tool_use)\s*\n.*?\n```",
    flags=re.IGNORECASE | re.DOTALL,
)


def strip_thinking(text: str) -> str:
    if not text or "<" not in text:
        return text
    text = _THINKING_RE.sub("", text)
    return text


def strip_inline_tool_xml(text: str) -> str:
    if not text:
        return text
    text = _XML_TOOL_RE.sub("", text)
    text = _FENCE_TOOL_RE.sub("", text)
    return text


def sanitize_text(text: str) -> str:
    """Strip thinking + xml-style tool noise; collapse runaway whitespace."""
    if not text:
        return text
    text = strip_thinking(text)
    text = strip_inline_tool_xml(text)
    # Collapse 3+ blank lines (common after stripping).
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def sanitize_history(history: list[Message]) -> list[Message]:
    """Return a shallow-copied history with assistant/user text sanitized."""
    if not history:
        return history
    cleaned: list[Message] = []
    for msg in history:
        if not isinstance(msg.content, str):
            cleaned.append(msg)
            continue
        if msg.role not in ("assistant", "user"):
            cleaned.append(msg)
            continue
        new_text = sanitize_text(msg.content)
        if new_text == msg.content:
            cleaned.append(msg)
            continue
        m = copy(msg)
        m.content = new_text
        cleaned.append(m)
    return cleaned
