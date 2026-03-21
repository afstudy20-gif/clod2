#!/usr/bin/env python3
"""
CClaude - A multi-provider AI coding assistant (Claude Code alternative)

Supports: Anthropic Claude, OpenAI ChatGPT, Google Gemini
"""
import os
import sys

import click
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from src.core.agent import Agent
from src.core.config import get_api_key, load_config, set_api_key
from src.providers import PROVIDERS, get_provider
from src.tools import get_default_registry

console = Console()

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
}


def print_banner(provider_name: str, model: str):
    color = PROVIDER_COLORS.get(provider_name.lower(), "white")
    console.print(BANNER, style=f"bold {color}")
    console.print(
        Panel(
            f"Provider: [bold]{provider_name}[/bold]  |  Model: [bold]{model}[/bold]\n"
            "Commands: /help  /reset  /model <name>  /provider <name>  /exit",
            title="CClaude - Multi-API Coding Assistant",
            border_style=color,
        )
    )


def get_prompt_style(provider_name: str) -> Style:
    color_map = {"claude": "#00aa00", "openai": "#00aaaa", "gemini": "#0088ff"}
    color = color_map.get(provider_name.lower(), "#ffffff")
    return Style.from_dict({"prompt": color})


def print_help():
    console.print(
        Panel(
            """[bold]Commands:[/bold]
  /help              Show this help
  /reset             Clear conversation history
  /model <name>      Switch model (e.g. /model gpt-4o)
  /provider <name>   Switch provider (claude/openai/gemini)
  /models            List available models for current provider
  /key <key>         Set API key for current provider
  /exit or /quit     Exit

[bold]Providers:[/bold]
  claude / anthropic  →  Anthropic Claude models
  openai / chatgpt    →  OpenAI GPT models
  gemini / google     →  Google Gemini models

[bold]API Key Setup:[/bold]
  Set env vars: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY
  Or use:  /key YOUR_API_KEY
  Or create a .env file in the working directory
""",
            title="Help",
            border_style="yellow",
        )
    )


@click.command()
@click.option("--provider", "-p", default="claude", help="AI provider: claude, openai, gemini")
@click.option("--model", "-m", default=None, help="Model name (provider-specific)")
@click.option("--key", "-k", default=None, help="API key (overrides env var)")
@click.option("--set-key", is_flag=True, help="Save API key to config")
@click.option("--list-providers", is_flag=True, help="List available providers and exit")
def main(provider: str, model: str | None, key: str | None, set_key: bool, list_providers: bool):
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
            f"  Gemini:  export GOOGLE_API_KEY=..."
        )
        sys.exit(1)

    if set_key and key:
        set_api_key(provider, key)

    try:
        ai_provider = get_provider(provider, api_key=api_key, model=model)
    except ValueError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    registry = get_default_registry()
    agent = Agent(ai_provider, registry)

    print_banner(provider, ai_provider.model)

    session = PromptSession(
        history=FileHistory(HISTORY_FILE),
        style=get_prompt_style(provider),
    )

    current_provider_name = provider

    while True:
        try:
            user_input = session.prompt(f"\n[{current_provider_name}]> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # Handle slash commands
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd in ("/exit", "/quit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "/help":
                print_help()

            elif cmd == "/reset":
                agent.reset()
                console.print("[yellow]Conversation history cleared.[/yellow]")

            elif cmd == "/models":
                models = ai_provider.list_models()
                console.print("[bold]Available models:[/bold]")
                for m in models:
                    marker = " [green]<- current[/green]" if m == ai_provider.model else ""
                    console.print(f"  {m}{marker}")

            elif cmd == "/model":
                if not arg:
                    console.print(f"[yellow]Current model: {ai_provider.model}[/yellow]")
                else:
                    ai_provider.model = arg
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
                            ai_provider = get_provider(arg, api_key=new_key)
                            registry = get_default_registry()
                            agent = Agent(ai_provider, registry)
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
                    # Reinitialize with new key
                    ai_provider = get_provider(current_provider_name, api_key=arg, model=ai_provider.model)
                    agent = Agent(ai_provider, registry)
                    console.print(f"[green]API key updated for {current_provider_name}[/green]")

            else:
                console.print(f"[red]Unknown command: {cmd}[/red]. Type /help for help.")
            continue

        # Regular message - stream response
        try:
            response_parts = []
            console.print()
            for chunk in agent.chat(user_input):
                # Print tool blocks as-is (already formatted)
                if chunk.startswith("\n[Tool:") or chunk.startswith("```"):
                    console.print(chunk, end="")
                else:
                    console.print(chunk, end="", markup=False)
                response_parts.append(chunk)
            console.print()  # Final newline
        except KeyboardInterrupt:
            console.print("\n[yellow](interrupted)[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


if __name__ == "__main__":
    main()
