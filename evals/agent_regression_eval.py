"""Minimal regression checks for provider metadata and tool validation.

Run with:
    python3 evals/agent_regression_eval.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import os

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.agent import Agent
from src.core.session import export_context, import_context
from src.providers import PROVIDERS
from src.providers.base import Message, ToolCall
from src.tools.registry import get_default_registry
from api import _load_cached_remote, _looks_like_git_remote_url, _resolve_git_worktree, _save_cached_remote


class FakeProvider:
    name = "Fake"
    model = "fake-model"
    SUPPORTS_TOOLS = True


def main():
    registry = get_default_registry()

    bad = registry.execute("write_file", {"path": "x", "content": 123})
    if "must be string" not in bad:
        raise AssertionError(f"tool schema validation failed: {bad}")

    compat = registry.execute("read_file", {"path": str(ROOT / "README.md"), "view_range": [1, 2]})
    if "Tool call error" in compat or "unknown argument" in compat:
        raise AssertionError(f"read_file view_range compatibility failed: {compat[:200]}")

    openai_info = PROVIDERS["openai"].model_info("gpt-5-mini")
    if openai_info.api != "openai-completions" or not openai_info.supports_tools:
        raise AssertionError(f"bad OpenAI model metadata: {openai_info}")
    if not openai_info.reasoning:
        raise AssertionError("gpt-5-mini should be marked as reasoning-capable")

    gemini_info = PROVIDERS["gemini"].model_info("gemini-2.5-flash")
    if "image" not in gemini_info.input:
        raise AssertionError(f"Gemini metadata should advertise image input: {gemini_info}")

    messages = [Message(role="assistant", content="done", provider="OpenAI", model="gpt-5-mini")]
    payload = export_context(messages, provider="openai", model="gpt-5-mini", system_prompt="test")
    restored, metadata = import_context(payload)
    if metadata.get("provider") != "openai" or restored[0].model != "gpt-5-mini":
        raise AssertionError(f"context export/import lost metadata: {payload}")

    if not _looks_like_git_remote_url("https://github.com/afstudy20-gif/clod2.git"):
        raise AssertionError("https GitHub remote URL was not detected")
    if not _looks_like_git_remote_url("git@github.com:afstudy20-gif/clod2.git"):
        raise AssertionError("ssh GitHub remote URL was not detected")
    if _looks_like_git_remote_url("origin"):
        raise AssertionError("remote name should not be treated as a URL")
    _save_cached_remote(ROOT, "https://github.com/afstudy20-gif/clod2.git")
    if _load_cached_remote(ROOT) != "https://github.com/afstudy20-gif/clod2.git":
        raise AssertionError("cached GitHub remote was not saved/loaded")
    if _resolve_git_worktree(ROOT / "src" / "core") != ROOT:
        raise AssertionError("git worktree root was not resolved from a subdirectory")

    agent = Agent(FakeProvider(), registry)
    if not agent._is_safe_repeated_tool_call(ToolCall(id="1", name="git_status", arguments={})):
        raise AssertionError("git_status repeat should be safe")
    if not agent._is_safe_repeated_tool_call(ToolCall(id="2", name="git_diff", arguments={"staged": True})):
        raise AssertionError("git_diff repeat should be safe")
    if agent._is_safe_repeated_tool_call(ToolCall(id="3", name="git_push", arguments={})):
        raise AssertionError("git_push repeat should not be safe")

    # Temporary test project generation & macOS app packaging eval
    print("Testing scaffold_macos_app in a temporary project...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        # Setup dummy package
        (temp_path / "package.json").write_text('{"name": "test-app", "scripts": {"start": "echo hi"}}')
        
        # Override project root for registry tools
        from src.tools.implementations import set_project_root
        set_project_root(temp_dir)
        
        result = registry.execute("scaffold_macos_app", {
            "app_name": "Test App",
            "app_dir": "desktop-test",
            "backend_command": "npm start",
            "start_url": "http://127.0.0.1:8765",
            "port": 8765
        })
        
        if "Error:" in result:
            raise AssertionError(f"scaffold_macos_app failed: {result}")
            
        if not (temp_path / "desktop-test" / "package.json").exists():
            raise AssertionError("macOS scaffold did not create desktop-test/package.json")
            
        if not (temp_path / "scripts" / "setup-macos-app.sh").exists():
            raise AssertionError("macOS scaffold did not create scripts/setup-macos-app.sh")

        # Revert project root
        set_project_root(str(ROOT))

    print("agent regression evals passed")


if __name__ == "__main__":
    main()
