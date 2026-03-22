"""Tool registry - maps tool schemas to implementation functions."""
from typing import Any, Callable

from .implementations import bash, edit_file, glob_files, grep_search, list_dir, read_file, write_file
from .github_tools import (
    github_delete_file,
    github_list_dir,
    github_read_file,
    github_search_code,
    github_write_file,
)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, dict] = {}
        self._handlers: dict[str, Callable] = {}

    def register(self, schema: dict, handler: Callable):
        name = schema["name"]
        self._tools[name] = schema
        self._handlers[name] = handler

    def get_schemas(self) -> list[dict]:
        return list(self._tools.values())

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name not in self._handlers:
            return f"Unknown tool: {name}"
        try:
            return self._handlers[name](**arguments)
        except TypeError as e:
            return f"Tool call error for '{name}': {e}"
        except Exception as e:
            return f"Tool execution error: {e}"


def get_default_registry() -> ToolRegistry:
    reg = ToolRegistry()

    reg.register(
        {
            "name": "read_file",
            "description": "Read the contents of a file. Returns content with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Absolute or relative path to the file"},
                    "offset": {"type": "integer", "description": "Line number to start reading from (0-indexed)"},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read (default 2000)"},
                },
                "required": ["path"],
            },
        },
        read_file,
    )

    reg.register(
        {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed. OVERWRITES existing content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to write to"},
                    "content": {"type": "string", "description": "Content to write"},
                },
                "required": ["path", "content"],
            },
        },
        write_file,
    )

    reg.register(
        {
            "name": "edit_file",
            "description": "Replace a unique string in a file with new text. The old_string must appear exactly once.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File to edit"},
                    "old_string": {"type": "string", "description": "Exact text to find and replace (must be unique)"},
                    "new_string": {"type": "string", "description": "Replacement text"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
        edit_file,
    )

    reg.register(
        {
            "name": "bash",
            "description": "Execute a shell command. Avoid interactive commands. Max timeout 120s.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds (max 120, default 30)"},
                },
                "required": ["command"],
            },
        },
        bash,
    )

    reg.register(
        {
            "name": "glob",
            "description": "Find files matching a glob pattern like '**/*.py' or '*.json'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern to match"},
                    "path": {"type": "string", "description": "Directory to search in (default: current dir)"},
                },
                "required": ["pattern"],
            },
        },
        glob_files,
    )

    reg.register(
        {
            "name": "grep",
            "description": "Search file contents with a regex pattern. Returns matching lines with file paths and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern to search for"},
                    "path": {"type": "string", "description": "Directory to search in (default: current dir)"},
                    "file_glob": {"type": "string", "description": "Filter files by name pattern (e.g. '*.py')"},
                    "case_insensitive": {"type": "boolean", "description": "Case-insensitive search"},
                    "context": {"type": "integer", "description": "Lines of context around matches"},
                },
                "required": ["pattern"],
            },
        },
        grep_search,
    )

    reg.register(
        {
            "name": "list_dir",
            "description": "List directory contents showing files and subdirectories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path to list (default: current dir)"},
                },
                "required": [],
            },
        },
        list_dir,
    )

    # ── GitHub tools ─────────────────────────────────────────────────────────

    reg.register(
        {
            "name": "github_read_file",
            "description": "Read a file from a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo e.g. 'afstudy20-gif/Cclaude'"},
                    "path": {"type": "string", "description": "File path inside the repo e.g. 'src/api.py'"},
                    "ref": {"type": "string", "description": "Branch, tag or commit SHA (default: HEAD)"},
                    "token": {"type": "string", "description": "GitHub token (optional, uses GITHUB_TOKEN env var if omitted)"},
                },
                "required": ["repo", "path"],
            },
        },
        github_read_file,
    )

    reg.register(
        {
            "name": "github_write_file",
            "description": "Create or update a file in a GitHub repository (commits directly).",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string", "description": "File path inside the repo"},
                    "content": {"type": "string", "description": "New file content (plain text)"},
                    "message": {"type": "string", "description": "Commit message"},
                    "branch": {"type": "string", "description": "Branch to commit to (default: main)"},
                    "token": {"type": "string", "description": "GitHub token (optional)"},
                },
                "required": ["repo", "path", "content", "message"],
            },
        },
        github_write_file,
    )

    reg.register(
        {
            "name": "github_list_dir",
            "description": "List files and folders in a GitHub repository directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string", "description": "Directory path (empty = repo root)"},
                    "ref": {"type": "string", "description": "Branch, tag or commit SHA (default: HEAD)"},
                    "token": {"type": "string", "description": "GitHub token (optional)"},
                },
                "required": ["repo"],
            },
        },
        github_list_dir,
    )

    reg.register(
        {
            "name": "github_delete_file",
            "description": "Delete a file from a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "path": {"type": "string", "description": "File path to delete"},
                    "message": {"type": "string", "description": "Commit message"},
                    "branch": {"type": "string", "description": "Branch to commit to (default: main)"},
                    "token": {"type": "string", "description": "GitHub token (optional)"},
                },
                "required": ["repo", "path", "message"],
            },
        },
        github_delete_file,
    )

    reg.register(
        {
            "name": "github_search_code",
            "description": "Search for code inside a GitHub repository.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo": {"type": "string", "description": "owner/repo"},
                    "query": {"type": "string", "description": "Search query"},
                    "token": {"type": "string", "description": "GitHub token (optional)"},
                },
                "required": ["repo", "query"],
            },
        },
        github_search_code,
    )

    return reg
