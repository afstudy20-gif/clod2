"""Agentic loop: sends messages, handles tool calls, loops until done."""
from typing import Iterator

from ..providers.base import BaseProvider, Message, ToolCall, ToolEvent, ToolResult
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

EXPLORE_SYSTEM_PROMPT = """You are in EXPLORE MODE. Help the user understand the codebase.

You can ONLY use read-only tools: read_file, glob, grep, list_dir, github_read_file, github_list_dir, github_search_code.
DO NOT modify any files. Focus on explaining code, architecture, patterns, and relationships.
When the user asks about code, read the relevant files and explain clearly.
Provide thorough analysis with file paths and line references.
"""

PLAN_SYSTEM_PROMPT = """You are in PLAN MODE. Your task is to explore the codebase using read-only tools and produce a structured implementation plan.

DO NOT modify any files. You may only use read-only tools: read_file, glob, grep, list_dir.

After exploring, produce a plan in this format:

## Plan
1. [Step with file path and description of change]
2. ...

## Files to Modify
- path/to/file.py - description of changes

## Files to Create
- path/to/new_file.py - purpose

## Risks
- potential issues or edge cases
"""


class Agent:
    def __init__(
        self,
        provider: BaseProvider,
        registry: ToolRegistry,
        max_tool_rounds: int = 20,
        project_root: str | None = None,
    ):
        self.provider = provider
        self.registry = registry
        self.max_tool_rounds = max_tool_rounds
        self.history: list[Message] = []
        self.mode: str = "normal"  # "normal", "explore", "plan"
        self.project_root = project_root
        self.session_id: str | None = None

    def _get_system_prompt(self) -> str:
        if self.mode == "explore":
            base = EXPLORE_SYSTEM_PROMPT
        elif self.mode == "plan":
            base = PLAN_SYSTEM_PROMPT
        else:
            base = SYSTEM_PROMPT

        if self.project_root:
            base += f"\n\nProject root: {self.project_root}"
            base += "\nAll relative paths should be resolved from this directory."

        return base

    def _get_tool_schemas(self) -> list[dict]:
        if self.mode in ("explore", "plan"):
            return self.registry.get_schemas(readonly_only=True)
        return self.registry.get_schemas()

    def chat(self, user_message: str) -> Iterator[str | ToolEvent]:
        """Send a user message and yield response text chunks and ToolEvents."""
        self.history.append(Message(role="user", content=user_message))

        system_prompt = self._get_system_prompt()
        tool_schemas = self._get_tool_schemas()

        for _ in range(self.max_tool_rounds):
            tool_calls_seen: list[ToolCall] = []
            text_buffer = []

            # Trim history to stay within token budget
            active_history = trim_history(self.history)

            for item in self.provider.stream_response(
                active_history,
                tool_schemas,
                system_prompt,
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
                yield ToolEvent(
                    type="start",
                    tool_name=tc.name,
                    arguments=tc.arguments,
                )
                result = self.registry.execute(tc.name, tc.arguments)
                is_error = result.startswith("Error:")
                yield ToolEvent(
                    type="result",
                    tool_name=tc.name,
                    arguments=tc.arguments,
                    result=result[:3000],
                    is_error=is_error,
                )
                tool_results.append(
                    ToolResult(
                        tool_call_id=tc.id,
                        content=result,
                        is_error=is_error,
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
