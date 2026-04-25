#!/usr/bin/env python3
"""
CClaude - A multi-provider AI coding assistant (Claude Code alternative)

Supports: Anthropic Claude, OpenAI ChatGPT, Google Gemini, Groq, Mistral, DeepSeek, NVIDIA NIM, Tavily, Ollama, Cohere
"""
import os
import sys
import threading
import uuid

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.core.agent import Agent
from src.core.auth import OAUTH_PROVIDERS, login_openrouter, login_google, get_google_credentials
from src.core.config import get_api_key, get_last_model, load_config, set_api_key, set_last_model
from src.core.loop import LoopRunner
from src.core.project import detect_project_root, project_name
from src.core.skills import list_project_instruction_files
from src.core.session import (
    delete_session,
    get_last_session_id,
    list_sessions,
    load_session,
    save_session,
)
from src.providers import PROVIDERS, get_provider
from src.providers.base import ToolEvent
from src.tools import get_default_registry
from src.tools.git_tools import (
    git_add,
    git_branch,
    git_checkout,
    git_commit,
    git_create_pr,
    git_diff,
    git_log,
    git_pull,
    git_push,
    git_status,
)
from src.tools.implementations import glob_files, grep_search, list_dir, read_file, set_project_root

console = Console()
output_lock = threading.Lock()

HISTORY_FILE = os.path.expanduser("~/.cclaude/history")
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

BANNER = """
  ██████╗ ██████╗██╗      █████╗ ██╗   ██╗██████╗ ███████╗
 ██╔════╝██╔════╝██║     ██╔══██╗██║   ██║██╔══██╗██╔════╝
 ██║     ██║     ██║     ███████║██║   ██║██║  ██║█████╗
 ██║     ██║     ██║     ██╔══██║██║   ██║██║  ██║██╔══╝
 ╚██████╗╚██████╗███████╗██║  ██║╚██████╔╝██████╔╝███████╗
  ╚═════╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝
"""

PROVIDER_COLORS = {
    "claude": "green",
    "anthropic": "green",
    "openai": "cyan",
    "chatgpt": "cyan",
    "gpt": "cyan",
    "gemini": "blue",
    "google": "blue",
    "groq": "magenta",
    "mistral": "yellow",
    "deepseek": "bright_blue",
    "nvidia": "bright_green",
    "nim": "bright_green",
    "ollama": "bright_green",
    "local": "bright_green",
    "cohere": "bright_magenta",
    "openrouter": "bright_yellow",
    "tavily": "bright_cyan",
    "search": "bright_cyan",
}


def print_banner(provider_name: str, model: str, project_root: str | None = None):
    color = PROVIDER_COLORS.get(provider_name.lower(), "white")
    console.print(BANNER, style=f"bold {color}")
    info = (
        f"Provider: [bold]{provider_name}[/bold]  |  Model: [bold]{model}[/bold]\n"
    )
    if project_root:
        info += f"Project: [bold]{project_name(project_root)}[/bold] ({project_root})\n"
    info += "Commands: /help  /status  /clear  /model  /provider  /plan  /diff  /push  /exit"
    console.print(Panel(info, title="CClaude - Multi-API Coding Assistant", border_style=color))


def get_prompt_style(provider_name: str) -> Style:
    color_map = {
        "claude": "#00aa00", "anthropic": "#00aa00",
        "openai": "#00aaaa", "chatgpt": "#00aaaa", "gpt": "#00aaaa",
        "gemini": "#0088ff", "google": "#0088ff",
        "groq": "#cc00cc",
        "mistral": "#ccaa00",
        "deepseek": "#0055ff",
        "nvidia": "#76b900", "nim": "#76b900",
        "ollama": "#00cc44", "local": "#00cc44",
        "cohere": "#cc0088",
        "openrouter": "#ccaa00",
        "tavily": "#00c7b7", "search": "#00c7b7",
    }
    color = color_map.get(provider_name.lower(), "#ffffff")
    return Style.from_dict({"prompt": color})


def build_prompt_text(provider_name: str, model: str, mode: str, proj_name: str | None) -> str:
    """Build the REPL prompt string."""
    parts = [provider_name]
    if proj_name:
        parts.append(proj_name)
    short_model = model.split("/")[-1]  # strip provider prefix if any
    if len(short_model) > 25:
        short_model = short_model[:22] + "..."
    parts.append(short_model)
    if mode != "normal":
        parts.append(mode)
    return f"\n[{'|'.join(parts)}]> "


def print_help():
    console.print(
        Panel(
            """[bold]Commands:[/bold]
  /help                    Show this help
  /reset, /clear, /new      Clear conversation history
  /status                  Show provider, model, project, mode, and git status
  /model [name]            Switch model or pick interactively
  /provider <name>         Switch provider
  /models                  List available models
  /key <key>               Set API key for current provider
  /compact                 Trim older conversation history
  /doctor                  Check local setup

[bold]Modes:[/bold]
  /explore                 Toggle explore mode (read-only, explains code)
  /plan <request>          Plan mode: explore, plan, then execute on approval
  /normal                  Return to normal mode
  /permissions             Show tool access by mode

[bold]Sessions:[/bold]
  /session save [name]     Save current session
  /session load <name>     Load a saved session
  /session list            List saved sessions
  /session delete <name>   Delete a saved session
  /save [name]             Alias for /session save
  /load <name>             Alias for /session load
  /history                 Alias for /session list

[bold]Project:[/bold]
  /project                 Show current project root
  /project set <path>      Set project root
  /pwd                     Show current project root
  /cd <path>               Set project root
  /ls [path]               List files
  /read <path>             Read a file
  /find <glob>             Find files by glob
  /search <regex>          Search file contents
  /init                    Create AGENTS.md project instructions
  /memory [add <text>]     Show or append project instructions
  /skills                  List loaded markdown skills

[bold]Git:[/bold]
  /diff [path]             Show git diff
  /add [paths]             Stage files
  /commit <message>        Commit staged changes
  /pull [remote] [branch]  Pull from remote
  /push [remote] [branch]  Push current project to GitHub remote
  /branch [name]           List or create branches
  /checkout <ref>          Switch branch/ref
  /log [count]             Show recent commits
  /pr <title>              Create GitHub PR with gh CLI

[bold]Loop:[/bold]
  /loop <secs> <prompt>    Run a prompt every N seconds
  /stop                    Stop the active loop

[bold]Auth:[/bold]
  /login [provider]        Sign in via OAuth (openrouter, gemini)
  /key <key>               Set API key manually

[bold]Other:[/bold]
  /exit or /quit           Exit

[bold]Providers:[/bold]
  claude / anthropic  →  Anthropic Claude models
  openai / chatgpt    →  OpenAI GPT models
  gemini / google     →  Google Gemini models
  openrouter          →  Many models via single sign-in (OAuth)
  groq                →  Groq (fast, free tier)
  mistral             →  Mistral AI
  deepseek            →  DeepSeek (very cheap)
  nvidia / nim        →  NVIDIA NIM models from build.nvidia.com
  tavily / search     →  Tavily AI search modes
  ollama / local      →  Ollama (local, free)
  cohere              →  Cohere Command-R
""",
            title="Help",
            border_style="yellow",
        )
    )


def stream_response_to_console(agent: Agent, user_input: str):
    """Stream agent response to console with Rich formatting for tools."""
    console.print()
    for chunk in agent.chat(user_input):
        if isinstance(chunk, ToolEvent):
            if chunk.type == "start":
                args_summary = _summarize_args(chunk.arguments)
                console.print(f"  [dim]⚙ {chunk.tool_name}({args_summary})[/dim]")
            elif chunk.type == "result":
                if chunk.is_error:
                    console.print(f"  [red]✗ Error: {(chunk.result or '')[:200]}[/red]")
                else:
                    result_preview = (chunk.result or "")[:300]
                    if len(result_preview) > 100:
                        result_preview = result_preview[:100] + "..."
                    console.print(f"  [green]✓[/green] [dim]{result_preview}[/dim]")
        else:
            console.print(chunk, end="", markup=False)
    console.print()


def _summarize_args(args: dict) -> str:
    parts = []
    for k, v in args.items():
        s = str(v)
        if len(s) > 50:
            s = s[:47] + "..."
        parts.append(f"{k}={s!r}")
    return ", ".join(parts)


def _print_command_output(output: str):
    style = "red" if output.lower().startswith("error:") else "green"
    console.print(output, style=style)


def _set_project_root(path: str, agent: Agent) -> tuple[str | None, str | None]:
    root = os.path.abspath(os.path.expanduser(path))
    if not os.path.isdir(root):
        console.print(f"[red]Not a directory: {path}[/red]")
        return None, None
    set_project_root(root)
    agent.project_root = root
    return root, project_name(root)


@click.command()
@click.option("--provider", "-p", default="claude", help="AI provider: claude, openai, gemini")
@click.option("--model", "-m", default=None, help="Model name (provider-specific)")
@click.option("--key", "-k", default=None, help="API key (overrides env var)")
@click.option("--set-key", is_flag=True, help="Save API key to config")
@click.option("--list-providers", is_flag=True, help="List available providers and exit")
@click.option("--project", "-d", default=None, help="Project directory (auto-detected if omitted)")
@click.option("--resume", "-r", is_flag=True, help="Resume the last session")
def main(
    provider: str,
    model: str | None,
    key: str | None,
    set_key: bool,
    list_providers: bool,
    project: str | None,
    resume: bool,
):
    """CClaude - Multi-provider AI coding assistant."""

    if list_providers:
        console.print("\n[bold]Available providers:[/bold]")
        for alias in sorted(PROVIDERS.keys()):
            console.print(f"  {alias}")
        return

    config = load_config()
    api_key = key or get_api_key(provider, config)

    if not api_key:
        console.print(
            f"[red]No API key found for provider '{provider}'.[/red]\n"
            f"Set the environment variable or use --key YOUR_API_KEY\n"
            f"  Claude:  export ANTHROPIC_API_KEY=...\n"
            f"  OpenAI:  export OPENAI_API_KEY=...\n"
            f"  Gemini:  export GOOGLE_API_KEY=...\n"
            f"  NVIDIA: export NVIDIA_API_KEY=...\n"
            f"  Tavily: export TAVILY_API_KEY=..."
        )
        sys.exit(1)

    if set_key and key:
        set_api_key(provider, key)

    # Resolve model: explicit > last-used > provider default
    if not model:
        model = get_last_model(provider)

    try:
        ai_provider = get_provider(provider, api_key=api_key, model=model)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    # Project root detection
    proj_root = project or detect_project_root()
    if proj_root:
        set_project_root(proj_root)
    proj = project_name(proj_root) if proj_root else None

    registry = get_default_registry()
    agent = Agent(ai_provider, registry, project_root=proj_root)
    session_id: str | None = None

    # Resume last session if requested
    if resume:
        last_id = get_last_session_id(project=proj_root or "")
        if last_id:
            try:
                messages, meta = load_session(last_id)
                agent.history = messages
                session_id = last_id
                console.print(f"[green]Resumed session: {last_id} ({len(messages)} messages)[/green]")
            except Exception as e:
                console.print(f"[yellow]Could not resume session: {e}[/yellow]")

    print_banner(provider, ai_provider.model, proj_root)

    prompt_session = PromptSession(
        history=FileHistory(HISTORY_FILE),
        style=get_prompt_style(provider),
    )

    current_provider_name = provider
    loop_runner = LoopRunner()

    while True:
        try:
            prompt_text = build_prompt_text(
                current_provider_name, ai_provider.model, agent.mode, proj
            )
            user_input = prompt_session.prompt(prompt_text).strip()
        except (KeyboardInterrupt, EOFError):
            # Auto-save on exit if session is active
            if session_id and agent.history:
                save_session(session_id, agent.history, current_provider_name, ai_provider.model, proj_root or "")
                console.print(f"\n[dim]Session auto-saved: {session_id}[/dim]")
            console.print("[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit", "/q"):
                if session_id and agent.history:
                    save_session(session_id, agent.history, current_provider_name, ai_provider.model, proj_root or "")
                    console.print(f"[dim]Session auto-saved: {session_id}[/dim]")
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "/help":
                print_help()

            elif cmd in ("/reset", "/clear", "/new", "/restart"):
                agent.reset()
                console.print("[yellow]Conversation history cleared.[/yellow]")

            elif cmd == "/status":
                console.print(Panel(
                    f"Provider: [bold]{current_provider_name}[/bold]\n"
                    f"Model: [bold]{ai_provider.model}[/bold]\n"
                    f"Mode: [bold]{agent.mode}[/bold]\n"
                    f"Project: [bold]{proj_root or '(none)'}[/bold]\n"
                    f"Messages: [bold]{len(agent.history)}[/bold]\n\n"
                    f"[bold]Git[/bold]\n{git_status()}",
                    title="Status",
                    border_style="cyan",
                ))

            elif cmd == "/compact":
                from src.core.context import trim_history

                before = len(agent.history)
                agent.history = trim_history(agent.history)
                after = len(agent.history)
                console.print(f"[green]Compacted history: {before} -> {after} messages[/green]")

            elif cmd == "/doctor":
                checks = [
                    ("Project root", proj_root or "(none)"),
                    ("Git status", git_status().splitlines()[0] if proj_root else "No project root"),
                    ("Provider", current_provider_name),
                    ("Model", ai_provider.model),
                    ("API key", "configured" if get_api_key(current_provider_name, load_config()) else "missing"),
                    ("Python", sys.executable),
                ]
                console.print(Panel(
                    "\n".join(f"{name}: [bold]{value}[/bold]" for name, value in checks),
                    title="Doctor",
                    border_style="cyan",
                ))

            elif cmd == "/models":
                models = ai_provider.list_models()
                console.print("[bold]Available models:[/bold]")
                for i, m in enumerate(models, 1):
                    marker = " [green]<- current[/green]" if m == ai_provider.model else ""
                    console.print(f"  {i}. {m}{marker}")

            elif cmd == "/model":
                if not arg:
                    # Interactive model picker
                    models = ai_provider.list_models()
                    console.print("[bold]Select model:[/bold]")
                    for i, m in enumerate(models, 1):
                        marker = " [green]<- current[/green]" if m == ai_provider.model else ""
                        console.print(f"  {i}. {m}{marker}")
                    try:
                        choice = prompt_session.prompt("Enter number or model name: ").strip()
                        if choice.isdigit():
                            idx = int(choice) - 1
                            if 0 <= idx < len(models):
                                arg = models[idx]
                            else:
                                console.print("[red]Invalid selection.[/red]")
                                continue
                        else:
                            arg = choice
                    except (KeyboardInterrupt, EOFError):
                        continue

                ai_provider.model = arg
                set_last_model(current_provider_name, arg)
                agent.reset()
                console.print(f"[green]Switched to model: {arg} (history cleared)[/green]")

            elif cmd == "/provider":
                if not arg:
                    console.print(f"[yellow]Current provider: {current_provider_name}[/yellow]")
                else:
                    new_key = get_api_key(arg, config)
                    if not new_key:
                        console.print(f"[red]No API key found for '{arg}'. Set the env var first.[/red]")
                    else:
                        try:
                            new_model = get_last_model(arg)
                            ai_provider = get_provider(arg, api_key=new_key, model=new_model)
                            registry = get_default_registry()
                            agent = Agent(ai_provider, registry, project_root=proj_root)
                            current_provider_name = arg
                            console.print(
                                f"[green]Switched to {ai_provider.name} ({ai_provider.model})[/green]"
                            )
                        except ValueError as e:
                            console.print(f"[red]{e}[/red]")

            elif cmd == "/key":
                if not arg:
                    console.print("[yellow]Usage: /key YOUR_API_KEY[/yellow]")
                else:
                    set_api_key(current_provider_name, arg)
                    ai_provider = get_provider(current_provider_name, api_key=arg, model=ai_provider.model)
                    agent = Agent(ai_provider, registry, project_root=proj_root)
                    console.print(f"[green]API key updated for {current_provider_name}[/green]")

            # ── OAuth login ───────────────────────────────────────────────

            elif cmd == "/login":
                if not arg:
                    # Show available OAuth providers
                    console.print("[bold]Available OAuth sign-in:[/bold]")
                    for key, info in OAUTH_PROVIDERS.items():
                        if key in ("gemini",):
                            continue  # Skip alias
                        note = f"  [dim]{info.get('setup_note', '')}[/dim]" if info.get("setup_note") else ""
                        console.print(f"  /login {key:12s}  {info['description']}{note}")
                else:
                    target = arg.strip().lower()
                    if target not in OAUTH_PROVIDERS:
                        console.print(f"[red]No OAuth flow for '{target}'. Try: openrouter, gemini[/red]")
                    else:
                        info = OAUTH_PROVIDERS[target]
                        console.print(f"[cyan]Opening browser for {info['name']} sign-in...[/cyan]")
                        result = info["login"]()
                        if result:
                            console.print(f"[green]Signed in to {info['name']} successfully![/green]")
                            # Auto-switch to the provider
                            if target == "openrouter":
                                try:
                                    ai_provider = get_provider("openrouter", api_key=result)
                                    registry = get_default_registry()
                                    agent = Agent(ai_provider, registry, project_root=proj_root)
                                    current_provider_name = "openrouter"
                                    console.print(f"[green]Switched to OpenRouter ({ai_provider.model})[/green]")
                                except ValueError as e:
                                    console.print(f"[red]{e}[/red]")
                            elif target in ("google", "gemini"):
                                try:
                                    ai_provider = get_provider("gemini", api_key="__oauth__")
                                    registry = get_default_registry()
                                    agent = Agent(ai_provider, registry, project_root=proj_root)
                                    current_provider_name = "gemini"
                                    console.print(f"[green]Switched to Gemini via OAuth ({ai_provider.model})[/green]")
                                except ValueError as e:
                                    console.print(f"[red]{e}[/red]")
                        else:
                            msg = "[red]Sign-in failed or was cancelled.[/red]"
                            if info.get("setup_note"):
                                msg += f"\n[yellow]{info['setup_note']}[/yellow]"
                            console.print(msg)

            # ── Mode commands ────────────────────────────────────────────────

            elif cmd == "/explore":
                if agent.mode == "explore":
                    agent.mode = "normal"
                    console.print("[green]Exited explore mode.[/green]")
                else:
                    agent.mode = "explore"
                    console.print(
                        "[cyan]Entered explore mode.[/cyan] Read-only tools only. "
                        "Use /normal or /explore again to exit."
                    )

            elif cmd == "/normal":
                agent.mode = "normal"
                console.print("[green]Switched to normal mode.[/green]")

            elif cmd == "/permissions":
                readonly = ", ".join(t["name"] for t in registry.get_schemas(readonly_only=True))
                all_tools = ", ".join(t["name"] for t in registry.get_schemas())
                console.print(Panel(
                    "[bold]Normal mode[/bold]\n"
                    f"{all_tools}\n\n"
                    "[bold]Explore/plan modes[/bold]\n"
                    f"{readonly}",
                    title="Tool Permissions",
                    border_style="cyan",
                ))

            elif cmd == "/plan":
                if not arg:
                    console.print("[yellow]Usage: /plan <describe what you want to build>[/yellow]")
                else:
                    # Phase 1: Plan (read-only)
                    agent.mode = "plan"
                    console.print("[cyan]Planning...[/cyan]")
                    plan_parts = []
                    try:
                        for chunk in agent.chat(arg):
                            if isinstance(chunk, ToolEvent):
                                if chunk.type == "start":
                                    console.print(f"  [dim]⚙ {chunk.tool_name}[/dim]")
                            else:
                                plan_parts.append(chunk)
                                console.print(chunk, end="", markup=False)
                        console.print()
                    except KeyboardInterrupt:
                        console.print("\n[yellow](planning interrupted)[/yellow]")
                        agent.mode = "normal"
                        continue

                    plan_text = "".join(plan_parts)
                    console.print()
                    console.print(Panel(
                        Markdown(plan_text),
                        title="Implementation Plan",
                        border_style="cyan",
                    ))

                    # Phase 2: Ask for approval
                    try:
                        approval = prompt_session.prompt("\nExecute this plan? [y/n]: ").strip().lower()
                    except (KeyboardInterrupt, EOFError):
                        approval = "n"

                    if approval in ("y", "yes"):
                        agent.mode = "normal"
                        console.print("[green]Executing plan...[/green]")
                        try:
                            stream_response_to_console(agent, "Execute the plan you just created above. Implement all the changes step by step.")
                        except KeyboardInterrupt:
                            console.print("\n[yellow](execution interrupted)[/yellow]")
                    else:
                        agent.mode = "normal"
                        console.print("[yellow]Plan discarded. Back to normal mode.[/yellow]")

            # ── Session commands ──────────────────────────────────────────────

            elif cmd == "/save":
                name = arg or session_id or str(uuid.uuid4())[:8]
                save_session(name, agent.history, current_provider_name, ai_provider.model, proj_root or "")
                session_id = name
                console.print(f"[green]Session saved: {name}[/green]")

            elif cmd == "/load":
                if not arg:
                    console.print("[yellow]Usage: /load <name>[/yellow]")
                else:
                    try:
                        messages, meta = load_session(arg)
                        agent.history = messages
                        session_id = arg
                        console.print(
                            f"[green]Loaded session: {arg} "
                            f"({len(messages)} messages, provider: {meta.get('provider', '?')})[/green]"
                        )
                    except FileNotFoundError:
                        console.print(f"[red]Session not found: {arg}[/red]")

            elif cmd == "/history":
                sessions = list_sessions()
                if not sessions:
                    console.print("[dim]No saved sessions.[/dim]")
                else:
                    console.print("[bold]Saved sessions:[/bold]")
                    for s in sessions[:20]:
                        active = " [green]<- active[/green]" if s["id"] == session_id else ""
                        console.print(
                            f"  {s['id']}  [{s.get('provider','')}:{s.get('model','')}]  "
                            f"{s['messages']} msgs  {s.get('updated', '')[:16]}{active}"
                        )

            elif cmd == "/session":
                sub_parts = arg.split(maxsplit=1)
                sub_cmd = sub_parts[0].lower() if sub_parts else ""
                sub_arg = sub_parts[1] if len(sub_parts) > 1 else ""

                if sub_cmd == "save":
                    name = sub_arg or session_id or str(uuid.uuid4())[:8]
                    save_session(name, agent.history, current_provider_name, ai_provider.model, proj_root or "")
                    session_id = name
                    console.print(f"[green]Session saved: {name}[/green]")

                elif sub_cmd == "load":
                    if not sub_arg:
                        console.print("[yellow]Usage: /session load <name>[/yellow]")
                    else:
                        try:
                            messages, meta = load_session(sub_arg)
                            agent.history = messages
                            session_id = sub_arg
                            console.print(
                                f"[green]Loaded session: {sub_arg} "
                                f"({len(messages)} messages, provider: {meta.get('provider', '?')})[/green]"
                            )
                        except FileNotFoundError:
                            console.print(f"[red]Session not found: {sub_arg}[/red]")

                elif sub_cmd == "list":
                    sessions = list_sessions()
                    if not sessions:
                        console.print("[dim]No saved sessions.[/dim]")
                    else:
                        console.print("[bold]Saved sessions:[/bold]")
                        for s in sessions[:20]:
                            active = " [green]<- active[/green]" if s["id"] == session_id else ""
                            console.print(
                                f"  {s['id']}  [{s.get('provider','')}:{s.get('model','')}]  "
                                f"{s['messages']} msgs  {s.get('updated', '')[:16]}{active}"
                            )

                elif sub_cmd == "delete":
                    if not sub_arg:
                        console.print("[yellow]Usage: /session delete <name>[/yellow]")
                    elif delete_session(sub_arg):
                        if session_id == sub_arg:
                            session_id = None
                        console.print(f"[green]Deleted session: {sub_arg}[/green]")
                    else:
                        console.print(f"[red]Session not found: {sub_arg}[/red]")

                else:
                    console.print(
                        "[yellow]Usage: /session save|load|list|delete [name][/yellow]"
                    )

            # ── Project commands ──────────────────────────────────────────────

            elif cmd == "/project":
                if not arg:
                    if proj_root:
                        console.print(f"[bold]Project:[/bold] {proj} ({proj_root})")
                    else:
                        console.print("[dim]No project detected. Use /project set <path>[/dim]")
                elif arg.startswith("set "):
                    new_root, new_proj = _set_project_root(arg[4:].strip(), agent)
                    if new_root:
                        proj_root, proj = new_root, new_proj
                        console.print(f"[green]Project root set to: {proj_root}[/green]")
                else:
                    console.print("[yellow]Usage: /project or /project set <path>[/yellow]")

            elif cmd == "/pwd":
                console.print(proj_root or os.getcwd())

            elif cmd == "/cd":
                if not arg:
                    console.print("[yellow]Usage: /cd <path>[/yellow]")
                else:
                    new_root, new_proj = _set_project_root(arg, agent)
                    if new_root:
                        proj_root, proj = new_root, new_proj
                        console.print(f"[green]Project root set to: {proj_root}[/green]")

            elif cmd == "/ls":
                console.print(list_dir(arg or "."))

            elif cmd == "/read":
                if not arg:
                    console.print("[yellow]Usage: /read <path>[/yellow]")
                else:
                    console.print(read_file(arg))

            elif cmd in ("/find", "/glob"):
                if not arg:
                    console.print("[yellow]Usage: /find <glob>[/yellow]")
                else:
                    console.print(glob_files(arg))

            elif cmd in ("/search", "/grep"):
                if not arg:
                    console.print("[yellow]Usage: /search <regex>[/yellow]")
                else:
                    console.print(grep_search(arg))

            elif cmd == "/init":
                root = proj_root or os.getcwd()
                path = os.path.join(root, "AGENTS.md")
                if os.path.exists(path):
                    console.print(f"[yellow]AGENTS.md already exists: {path}[/yellow]")
                else:
                    content = (
                        "# Project Instructions\n\n"
                        "- Run tests before finishing changes when practical.\n"
                        "- Prefer the existing code style and local helper APIs.\n"
                        "- Keep edits focused on the requested task.\n"
                    )
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    console.print(f"[green]Created {path}[/green]")

            elif cmd == "/memory":
                root = proj_root or os.getcwd()
                agents_path = os.path.join(root, "AGENTS.md")
                claude_path = os.path.join(root, "CLAUDE.md")
                if arg.startswith("add "):
                    note = arg[4:].strip()
                    if not note:
                        console.print("[yellow]Usage: /memory add <text>[/yellow]")
                    else:
                        with open(agents_path, "a", encoding="utf-8") as f:
                            f.write(f"\n- {note}\n")
                        console.print(f"[green]Added memory to {agents_path}[/green]")
                else:
                    found = False
                    for rel in list_project_instruction_files(root):
                        path = os.path.join(root, rel)
                        if os.path.isfile(path):
                            found = True
                            console.print(Panel(read_file(path), title=rel, border_style="cyan"))
                    if not found:
                        console.print("[dim]No AGENTS.md, CLAUDE.md, or skills/*/SKILL.md found. Use /init to create AGENTS.md.[/dim]")

            elif cmd == "/skills":
                root = proj_root or os.getcwd()
                skill_files = [p for p in list_project_instruction_files(root) if p.startswith("skills/")]
                if not skill_files:
                    console.print("[dim]No markdown skills found under skills/*/SKILL.md[/dim]")
                else:
                    console.print("[bold]Loaded markdown skills:[/bold]")
                    for path in skill_files:
                        console.print(f"  {path}")

            # ── Git commands ─────────────────────────────────────────────────

            elif cmd == "/diff":
                staged = arg == "--staged" or arg.startswith("--staged ")
                path = arg.replace("--staged", "", 1).strip() if staged else arg
                console.print(git_diff(staged=staged, path=path))

            elif cmd == "/add":
                _print_command_output(git_add(arg or "."))

            elif cmd == "/commit":
                if not arg:
                    console.print("[yellow]Usage: /commit <message>[/yellow]")
                else:
                    _print_command_output(git_commit(arg))

            elif cmd == "/pull":
                pull_parts = arg.split()
                remote = pull_parts[0] if len(pull_parts) >= 1 else "origin"
                branch = pull_parts[1] if len(pull_parts) >= 2 else ""
                _print_command_output(git_pull(remote=remote, branch=branch))

            elif cmd == "/push":
                push_parts = arg.split()
                remote = push_parts[0] if len(push_parts) >= 1 else "origin"
                branch = push_parts[1] if len(push_parts) >= 2 else ""
                _print_command_output(git_push(remote=remote, branch=branch))

            elif cmd == "/branch":
                if not arg:
                    console.print(git_branch(list_all=True))
                else:
                    _print_command_output(git_branch(name=arg))

            elif cmd == "/checkout":
                if not arg:
                    console.print("[yellow]Usage: /checkout <ref>[/yellow]")
                else:
                    _print_command_output(git_checkout(arg))

            elif cmd == "/log":
                try:
                    count = int(arg) if arg else 10
                except ValueError:
                    count = 10
                console.print(git_log(count=count))

            elif cmd == "/pr":
                if not arg:
                    console.print("[yellow]Usage: /pr <title>[/yellow]")
                else:
                    _print_command_output(git_create_pr(title=arg))

            # ── Loop commands ─────────────────────────────────────────────────

            elif cmd == "/loop":
                loop_parts = arg.split(maxsplit=1)
                if len(loop_parts) < 2:
                    console.print("[yellow]Usage: /loop <seconds> <prompt>[/yellow]")
                else:
                    try:
                        interval = float(loop_parts[0])
                    except ValueError:
                        console.print("[red]Invalid interval. Use a number of seconds.[/red]")
                        continue
                    loop_prompt = loop_parts[1]

                    # Create a separate agent for the loop to avoid history conflicts
                    loop_agent = Agent(ai_provider, registry, project_root=proj_root)

                    def run_loop_iteration(prompt: str):
                        with output_lock:
                            console.print(f"\n[dim]--- loop: {prompt[:50]}... ---[/dim]")
                            try:
                                for chunk in loop_agent.chat(prompt):
                                    if isinstance(chunk, ToolEvent):
                                        if chunk.type == "start":
                                            console.print(f"  [dim]⚙ {chunk.tool_name}[/dim]")
                                    else:
                                        console.print(chunk, end="", markup=False)
                                console.print()
                            except Exception as e:
                                console.print(f"[red]Loop error: {e}[/red]")
                            loop_agent.reset()

                    loop_runner.start(loop_prompt, interval, run_loop_iteration)
                    console.print(
                        f"[green]Loop started: every {interval}s[/green] — use /stop to halt"
                    )

            elif cmd == "/stop":
                if loop_runner.is_running:
                    loop_runner.stop()
                    console.print("[green]Loop stopped.[/green]")
                else:
                    console.print("[dim]No active loop.[/dim]")

            else:
                console.print(f"[red]Unknown command: {cmd}[/red]. Type /help for help.")
            continue

        # Regular message - stream response
        try:
            stream_response_to_console(agent, user_input)
        except KeyboardInterrupt:
            console.print("\n[yellow](interrupted)[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
