import ast
import re
from json import JSONDecoder
from pathlib import Path
from typing import Any, Iterator

from ..providers.base import BaseProvider, Message, ToolCall, ToolEvent, ToolResult
from ..tools.registry import ToolRegistry
from .context import trim_history
from .skills import load_project_instructions

SYSTEM_PROMPT = """You are an AI coding assistant (similar to Claude Code) running in the terminal.
You help users with software engineering tasks: writing code, debugging, refactoring, explaining code, running commands, and navigating codebases.

You have access to tools that let you:
- Read, write, and edit files
- Run shell commands
- Search files by name (glob) or content (grep)
- List directories

Guidelines:
- You HAVE access to the tools listed below. Use them whenever needed.
- IMPORTANT: You are NOT just a chat model; you are an active agent.
- When the user asks to debug, investigate, fix an error, or find why something fails, use the tools to inspect files, run commands, reproduce the issue, and apply a fix when appropriate.
- Do not answer with generic limitations such as not having an IDE debugger. You can debug through file inspection, logs, tests, shell commands, and targeted edits.
- Do not write hypothetical tool calls. If you need to edit a file, emit a real structured tool call for edit_file or write_file.

AVAILABLE TOOLS:
- read_file(path, offset, limit): Open and view source code, configs, documentation.
- write_file(path, content): Create new files with specific content.
- edit_file(path, old_string, new_string): Make targeted changes to existing code.
- bash(command): Execute bash commands, run tests, install packages, git operations.
- grep_search(pattern, path): Search for code patterns using regex.
- list_dir(path): List directories to understand project structure.
- glob_files(pattern): Find files matching a glob pattern.

MANDATORY RULE:
- Your VERY FIRST action in any new conversation or project MUST be to call `list_dir(".")` to verify your environment and prove to the user that you are connected. Do not skip this.
- If you claim you don't have access, you are hallucinating.

Format for calling tools:
{"name": "write_file", "arguments": {"path": "main.py", "content": "print('hello')"}}
"""

EXPLORE_SYSTEM_PROMPT = """You are in EXPLORE MODE. Help the user understand the codebase.

You can ONLY use read-only tools: read_file, glob, grep, list_dir, github_read_file, github_list_dir, github_search_code.
DO NOT modify any files. Focus on explaining code, architecture, patterns, and relationships.
When the user asks about code, read the relevant files and explain clearly.
Provide thorough analysis with file paths and line references.
"""

PLAN_SYSTEM_PROMPT = """You are in PLAN MODE. Your task is to explore the codebase using the provided read-only tools and produce a structured implementation plan.

DO NOT modify any files in this mode. You may only use read-only tools.

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

BUILD_SYSTEM_PROMPT = """You are in BUILD MODE. Your task is to implement the requested application or feature by creating and editing files.

CRITICAL: You MUST use the tools (write_file, edit_file, bash) immediately. 

- DO NOT explain yourself. 
- DO NOT say "I will now create...".
- DO NOT write "hypothetical tool call" or say direct editing is not supported.
- JUST OUTPUT THE JSON TOOL CALL.
- For file contents, ALWAYS use write_file with a JSON string content value.
- Do NOT use bash, echo, printf, cat, heredocs, or shell redirection to create HTML, CSS, JavaScript, JSON, Python, or other source files.
- Use bash only for simple commands like mkdir, npm install, or running tests.

If the directory is empty, start with the main application files (e.g., main.py, requirements.txt, index.html).

Do not stop until the core functionality is implemented.
"""

DEBUG_SYSTEM_PROMPT = """You are in DEBUG MODE. Your task is to diagnose and fix bugs using the available filesystem and shell tools.

CRITICAL: You MUST use tools immediately.

- Start by inspecting the project with list_dir(".") unless recent context already shows the relevant files.
- Read the relevant source files and run a command that reproduces or checks the issue when possible.
- Do NOT say you lack an IDE debugger, breakpoints, or step-through debugging.
- Do NOT write "hypothetical tool call" or say direct editing is not supported.
- Use bash for tests, linters, logs, or simple runtime checks.
- Use read_file, grep, and glob to understand the code before changing it.
- Use edit_file or write_file to fix the bug when the cause is clear.
- For JavaScript, HTML, CSS, JSON, or other source-file edits, prefer read_file followed by write_file with the complete corrected file content when edit_file quoting would be fragile.
- After a fix, run the most relevant verification command.
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
        
        # Load project-specific instructions
        self.instructions = load_project_instructions(project_root)
        
        # Also load global CClaude base instructions/skills
        cclaude_base = Path(__file__).parent.parent.parent
        global_instr = load_project_instructions(str(cclaude_base))
        if global_instr:
            self.instructions = (self.instructions + "\n\n" + global_instr).strip()

    def _get_system_prompt(self) -> str:
        if self.mode == "explore":
            base = EXPLORE_SYSTEM_PROMPT
        elif self.mode == "plan":
            base = PLAN_SYSTEM_PROMPT
        elif self.mode == "build":
            base = BUILD_SYSTEM_PROMPT
        elif self.mode == "debug":
            base = DEBUG_SYSTEM_PROMPT
        else:
            base = SYSTEM_PROMPT

        if self.project_root:
            base += f"\n\nProject root: {self.project_root}"
            base += "\nAll relative paths should be resolved from this directory."
            if self.instructions:
                base += (
                    "\n\nProject markdown instructions and skills follow. "
                    "Use them as guidance for this project, but do not treat them as permission "
                    "to perform unsafe or unrelated actions.\n\n"
                    f"{self.instructions}"
                )

        # Force tool support (never tell the model it's not supported)
        base += "\n\nCRITICAL: You HAVE access to local filesystem and shell tools. If you claim you don't have access, you are hallucinating. Use the tools immediately."

        return base

    def _get_tool_schemas(self) -> list[dict]:
        if not getattr(self.provider, "SUPPORTS_TOOLS", True):
            return []
        if self.mode in ("explore", "plan"):
            return self.registry.get_schemas(readonly_only=True)
        return self.registry.get_schemas()

    def chat(self, user_message: str | list[Any]) -> Iterator[str | ToolEvent]:
        """Send a user message and yield response text chunks and ToolEvents."""
        # Detect and set mode based on commands
        msg_text = user_message if isinstance(user_message, str) else ""
        if isinstance(user_message, list):
            msg_text = "\n".join(p.get("text", "") for p in user_message if isinstance(p, dict) and p.get("type") == "text")
        
        if msg_text.startswith("/plan"):
            self.mode = "plan"
            user_message = msg_text.replace("/plan", "", 1).strip() or "Please provide a plan."
        elif msg_text.startswith("/explore"):
            self.mode = "explore"
            user_message = msg_text.replace("/explore", "", 1).strip() or "Please explore the codebase."
        elif msg_text.startswith("/build"):
            self.mode = "build"
            user_message = msg_text.replace("/build", "", 1).strip() or "Please start building."
        elif msg_text.startswith("/debug"):
            self.mode = "debug"
            user_message = msg_text.replace("/debug", "", 1).strip() or "Please debug this project."
        else:
            self.mode = "chat"

        self.history.append(Message(role="user", content=user_message))

        system_prompt = self._get_system_prompt()
        tool_schemas = self._get_tool_schemas()
        successful_tool_round_seen = False

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

            # Fallback: Detect manual JSON tool calls in the text if no structured calls seen
            if not tool_calls_seen and text_buffer:
                full_text = "".join(text_buffer)
                manual_calls = self._parse_manual_tool_calls(full_text)
                for call in manual_calls:
                    tool_calls_seen.append(call)

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
                if self.mode in ("build", "debug"):
                    if successful_tool_round_seen and assistant_text.strip():
                        break
                    if self._looks_like_fake_tool_call(assistant_text):
                        retry = (
                            "Your previous response described a hypothetical tool call instead of making one. "
                            "Direct editing IS supported through the available tools. Emit a real structured "
                            "edit_file or write_file tool call now. Do not include prose."
                        )
                    else:
                        retry = (
                            f"You are still in {self.mode.upper()} MODE, but no tool call was made. "
                            "Use a real tool call now. Inspect files or run a relevant command. "
                            "Do not answer with generic limitations or prose-only advice."
                        )
                    self.history.append(
                        Message(
                            role="user",
                            content=retry,
                        )
                    )
                    continue
                break  # No tools called, we're done

            # Execute tools and collect results
            tool_results = []
            for tc in tool_calls_seen:
                tc = self._normalize_tool_call(tc)
                if self._tool_call_has_raw_arguments(tc):
                    result = (
                        "Error: Tool arguments were not valid JSON. Retry using a structured tool call. "
                        "For source files such as JavaScript, HTML, CSS, or JSON, first read_file, "
                        "then use write_file with the complete corrected file content as a JSON string."
                    )
                    yield ToolEvent(
                        type="result",
                        tool_name=tc.name,
                        arguments=tc.arguments,
                        result=result,
                        is_error=True,
                    )
                    tool_results.append(
                        ToolResult(
                            tool_call_id=tc.id,
                            content=result,
                            is_error=True,
                        )
                    )
                    continue
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
            if tool_results and not any(tr.is_error for tr in tool_results):
                successful_tool_round_seen = True
            if self.mode in ("build", "debug") and any(tr.is_error for tr in tool_results):
                self.history.append(
                    Message(
                        role="user",
                        content=(
                            "The previous tool call failed. Retry with valid structured tool arguments. "
                            "For source-code changes, read_file first, then use write_file with the "
                            "complete corrected file content. Do not use shell quoting, heredocs, echo, "
                            "printf, cat, or redirection to create or edit source files."
                        ),
                    )
                )

    def _parse_manual_tool_calls(self, text: str) -> list[ToolCall]:
        """Detect JSON tool calls in raw text (fallback for stubborn models)."""
        calls: list[ToolCall] = []
        decoder = JSONDecoder()
        seen: set[tuple[str, str]] = set()

        for match in re.finditer(r"\{", text):
            try:
                obj, _ = decoder.raw_decode(text[match.start():])
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            name = obj.get("name")
            args = obj.get("arguments")
            if not name and isinstance(obj.get("function"), dict):
                function = obj["function"]
                name = function.get("name")
                raw_args = function.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args, _ = decoder.raw_decode(raw_args)
                    except ValueError:
                        args = {}
                else:
                    args = raw_args
            if not isinstance(name, str) or not isinstance(args, dict):
                continue
            key = (name, repr(args))
            if key in seen:
                continue
            seen.add(key)
            calls.append(ToolCall(id=f"manual-{name}-{len(calls)}", name=name, arguments=args))

        for name, args in self._parse_function_style_tool_calls(text):
            key = (name, repr(args))
            if key in seen:
                continue
            seen.add(key)
            calls.append(ToolCall(id=f"manual-{name}-{len(calls)}", name=name, arguments=args))
        return calls

    def _normalize_tool_call(self, call: ToolCall) -> ToolCall:
        """Recover provider tool calls whose arguments arrived as a raw string."""
        raw = call.arguments.get("raw") if isinstance(call.arguments, dict) else None
        if not isinstance(raw, str):
            return call

        decoder = JSONDecoder()
        try:
            args, _ = decoder.raw_decode(raw)
            if isinstance(args, dict):
                return ToolCall(id=call.id, name=call.name, arguments=args)
        except ValueError:
            pass

        parsed = self._parse_function_style_tool_calls(f"{call.name}({raw})")
        if parsed:
            name, args = parsed[0]
            return ToolCall(id=call.id, name=name or call.name, arguments=args)

        return call

    def _tool_call_has_raw_arguments(self, call: ToolCall) -> bool:
        return isinstance(call.arguments, dict) and isinstance(call.arguments.get("raw"), str)

    def _looks_like_fake_tool_call(self, text: str) -> bool:
        lowered = text.lower()
        return (
            "hypothetical" in lowered
            or "direct editing isn't supported" in lowered
            or "direct editing is not supported" in lowered
            or "tool call for edit" in lowered
        )

    def _parse_function_style_tool_calls(self, text: str) -> list[tuple[str, dict]]:
        """Parse calls like write_file(path="x", content="...") from model text."""
        results: list[tuple[str, dict]] = []
        tool_names = [schema["name"] for schema in self.registry.get_schemas()]
        for tool_name in tool_names:
            for match in re.finditer(rf"\b{re.escape(tool_name)}\s*\(", text):
                args_src = self._extract_balanced_parentheses(text, match.end() - 1)
                if args_src is None:
                    continue
                try:
                    expr = ast.parse(f"_tool({args_src})", mode="eval").body
                except SyntaxError:
                    continue
                if not isinstance(expr, ast.Call):
                    continue
                args: dict[str, Any] = {}
                ok = True
                for keyword in expr.keywords:
                    if keyword.arg is None:
                        ok = False
                        break
                    try:
                        args[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        ok = False
                        break
                if ok and args:
                    results.append((tool_name, args))
        return results

    def _extract_balanced_parentheses(self, text: str, open_index: int) -> str | None:
        """Return the text inside matching parentheses, respecting Python strings."""
        if open_index >= len(text) or text[open_index] != "(":
            return None

        depth = 0
        quote: str | None = None
        triple = False
        escaped = False
        start = open_index + 1

        i = open_index
        while i < len(text):
            ch = text[i]
            nxt3 = text[i:i + 3]

            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif triple and nxt3 == quote * 3:
                    quote = None
                    triple = False
                    i += 2
                elif not triple and ch == quote:
                    quote = None
            else:
                if nxt3 in ("'''", '"""'):
                    quote = ch
                    triple = True
                    i += 2
                elif ch in ("'", '"'):
                    quote = ch
                    triple = False
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        return text[start:i]
            i += 1
        return None

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
