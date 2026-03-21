"""Agentic loop: sends messages, handles tool calls, loops until done."""
from typing import Iterator

from ..providers.base import BaseProvider, Message, ToolCall, ToolResult
from ..tools.registry import ToolRegistry
from .context import trim_history

SYSTEM_PROMPT = """You are an AI coding assistant (similar to Claude Code) running in the terminal.
You help users with software engineering tasks: writing code, debugging, refactoring, explaining code, running commands, and navigating codebases.

You have access to tools that let you:
- Read, write, and edit files
- Run shell commands
- Search files by name (glob) or content (grep)
- List directories

Guidelines:
- Be concise and direct in your responses
- Prefer reading files before modifying them
- Use bash for tasks like running tests, git operations, installing packages
- When editing files, use edit_file for small changes and write_file to create new files
- Break complex tasks into steps
- Always show the user what you're doing when using tools
"""


class Agent:
    def __init__(
        self,
        provider: BaseProvider,
        registry: ToolRegistry,
        max_tool_rounds: int = 20,
    ):
        self.provider = provider
        self.registry = registry
        self.max_tool_rounds = max_tool_rounds
        self.history: list[Message] = []

    def chat(self, user_message: str) -> Iterator[str]:
        """Send a user message and yield response text chunks."""
        self.history.append(Message(role="user", content=user_message))

        for _ in range(self.max_tool_rounds):
            tool_calls_seen: list[ToolCall] = []
            text_buffer = []

            # Trim history to stay within token budget
            active_history = trim_history(self.history)

            for item in self.provider.stream_response(
                active_history,
                self.registry.get_schemas(),
                SYSTEM_PROMPT,
            ):
                if isinstance(item, str):
                    yield item
                    text_buffer.append(item)
                elif isinstance(item, ToolCall):
                    tool_calls_seen.append(item)

            # Save assistant message
            assistant_text = "".join(text_buffer)
            self.history.append(
                Message(
                    role="assistant",
                    content=assistant_text,
                    tool_calls=tool_calls_seen,
                )
            )

            if not tool_calls_seen:
                break  # No tools called, we're done

            # Execute tools and collect results
            tool_results = []
            for tc in tool_calls_seen:
                yield f"\n[Tool: {tc.name}({_summarize_args(tc.arguments)})]\n"
                result = self.registry.execute(tc.name, tc.arguments)
                yield f"```\n{result[:3000]}\n```\n"
                tool_results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=result,
                        is_error=result.startswith("Error:"),
                    )
                )

            # Add tool results to history
            self.history.append(
                Message(role="tool", content="", tool_results=tool_results)
            )

    def reset(self):
        self.history.clear()


def _summarize_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 60:
            s = s[:57] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)
