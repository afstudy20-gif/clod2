# Clod — Multi-Provider AI Coding Assistant

Clod is a multi-provider AI coding assistant created by **Yusuf Hosoglu**. It works with **Anthropic**, **OpenAI ChatGPT**, **Google Gemini**, **Groq**, **Mistral**, **DeepSeek**, **NVIDIA NIM**, **Tavily Search**, **Cohere**, and **Ollama** — using your own API keys.

## Features

- **Multi-provider**: Switch between 10 providers in one session
- **Agentic tool use**: reads/writes/edits files, runs bash commands, searches code
- **Session resume**: Save and load conversation history across restarts
- **Project-aware**: Auto-detects project root, resolves paths relative to it
- **Modes**: Normal, Explore (read-only), and Plan (plan then execute)
- **Loop mode**: Run prompts on a recurring interval
- **Interactive model picker**: Browse and switch models easily
- **Streaming**: Responses stream in real time with tool execution indicators
- **Context management**: Automatically trims history to stay within token limits
- **Model metadata**: Provider listings include API family, tool/image support, reasoning hints, context window, and max output tokens
- **Portable context**: Saved sessions preserve provider/model metadata for cross-provider handoffs
- **Validated tools**: Tool arguments are normalized and schema-checked before execution so models can retry from clear tool errors

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
# or install as a command:
pip install -e .
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env and add your key(s)
```

Or use environment variables directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AIza...
export NVIDIA_API_KEY=nvapi-...
export TAVILY_API_KEY=tvly-...
```

### OAuth Options

Clod supports browser/device login for providers where OAuth is appropriate:

- **Gemini OAuth**: put your Google OAuth desktop client secret at `~/.clod/google_client_secret.json`, then open **API Keys -> Connect Google**. Clod stores refresh tokens in `~/.clod/config.json` and uses them for Gemini when no `GOOGLE_API_KEY` env var is set.
- **Codex OAuth**: install/login with the official `codex` CLI, then open **API Keys -> Connect Codex**. Clod starts the official `codex login` flow and shows the current Codex login status. OpenAI API calls still use `OPENAI_API_KEY`; Codex OAuth is handled through the local Codex CLI/app-server auth model.

### 3. Run

```bash
# Default: Anthropic
python main.py

# Use OpenAI
python main.py --provider openai

# Specify model
python main.py --provider openai --model gpt-4o-mini

# Resume last session
python main.py --resume

# Set project directory explicitly
python main.py --project /path/to/project
```

## macOS VS Code Usage

Clod includes a local VS Code extension wrapper under `vscode-extension/`.

```bash
cd /Users/yh/Desktop/clod2/clod-agent
./scripts/setup-vscode-macos.sh
code /Users/yh/Desktop/clod2/clod-agent/vscode-extension
```

Then press `F5` in VS Code, open the project you want Clod to edit in the Extension Development Host, and run:

- `Clod: Start Backend`
- `Clod: Open Chat`

Set these extension settings when needed:

```json
{
  "clod.backendPath": "/Users/yh/Desktop/clod2/clod-agent",
  "clod.pythonPath": "/Users/yh/Desktop/clod2/venv/bin/python"
}
```

The extension sends the active VS Code workspace path to the backend, so edits, terminal checks, dev servers, and GitHub pushes run in the repo opened in VS Code.

## macOS Desktop App

Clod also includes an Electron desktop shell under `desktop-app/`. It starts the FastAPI backend on localhost, waits for `/health`, then opens the Clod web UI in a native macOS window.

```bash
cd /Users/yh/Desktop/clod2/clod-agent
bash scripts/setup-desktop-macos.sh
open /Users/yh/Desktop/clod2/clod-agent/desktop-app/dist/mac-arm64/Clod.app
```

For development:

```bash
cd /Users/yh/Desktop/clod2/clod-agent/desktop-app
npm start
```

If the app cannot find the correct Python environment, set `CLOD_PYTHON` to the virtualenv Python that has `uvicorn` installed:

```bash
export CLOD_PYTHON=/Users/yh/Desktop/clod2/clod-agent/venv/bin/python
```

## Building macOS Apps With Clod

Clod can scaffold Electron-based macOS desktop packages for other projects. In the UI, select a workspace and click **Build macOS App**, or ask:

```text
Turn this project into a macOS desktop app package.
```

The agent will inspect the project, choose the backend command and localhost URL, call `scaffold_macos_app`, then package with Electron when practical. The scaffold includes native macOS copy/paste support, a right-click edit menu, build scripts, and an Electron `package:mac` target.

## Commands

### Core
| Command | Description |
|---------|-------------|
| `/help` | Show all commands |
| `/reset` | Clear conversation history |
| `/clear`, `/new`, `/restart` | Common aliases for `/reset` |
| `/status` | Show provider, model, project, mode, and git status |
| `/model [name]` | Switch model (interactive picker if no name given) |
| `/models` | List available models for current provider |
| `/provider <name>` | Switch provider |
| `/key <api_key>` | Update API key |
| `/compact` | Trim older conversation history |
| `/doctor` | Check local setup |
| `/exit` | Quit (auto-saves active session) |

### Modes
| Command | Description |
|---------|-------------|
| `/explore` | Toggle explore mode — read-only tools, explains code |
| `/plan <request>` | Plan mode — explores, produces plan, asks approval, then executes |
| `/normal` | Return to normal mode |
| `/permissions` | Show available tools by mode |

### Sessions
| Command | Description |
|---------|-------------|
| `/session save [name]` | Save current conversation |
| `/session load <name>` | Load a saved session |
| `/session list` | List all saved sessions |
| `/session delete <name>` | Delete a session |
| `/save [name]` | Alias for `/session save` |
| `/load <name>` | Alias for `/session load` |
| `/history` | Alias for `/session list` |

### Project
| Command | Description |
|---------|-------------|
| `/project` | Show current project root |
| `/project set <path>` | Set project root directory |
| `/pwd` | Show current project root |
| `/cd <path>` | Set project root directory |
| `/ls [path]` | List files |
| `/read <path>` | Read a file |
| `/find <glob>` | Find files by glob |
| `/search <regex>` | Search file contents |
| `/init` | Create an `AGENTS.md` project instructions file |
| `/memory [add <text>]` | Show or append project instructions |
| `/skills` | List loaded markdown skills from `skills/*/SKILL.md` |

### Git
| Command | Description |
|---------|-------------|
| `/diff [path]` | Show git diff |
| `/add [paths]` | Stage files |
| `/commit <message>` | Commit staged changes |
| `/pull [remote] [branch]` | Pull from a remote |
| `/push [remote] [branch]` | Push the current project to a GitHub remote |
| `/branch [name]` | List or create branches |
| `/checkout <ref>` | Switch branch/ref |
| `/log [count]` | Show recent commits |
| `/pr <title>` | Create a GitHub pull request with `gh` |

### Loop
| Command | Description |
|---------|-------------|
| `/loop <secs> <prompt>` | Run a prompt every N seconds |
| `/stop` | Stop the active loop |

## Available Tools

| Tool | Description | Mode |
|------|-------------|------|
| `read_file` | Read file contents with line numbers | all |
| `write_file` | Create or overwrite a file | normal only |
| `edit_file` | Replace a unique string in a file | normal only |
| `bash` | Run shell commands | normal only |
| `glob` | Find files by pattern (e.g. `**/*.py`) | all |
| `grep` | Search file contents with regex | all |
| `list_dir` | List directory contents | all |

In **explore** and **plan** modes, only read-only tools (read_file, glob, grep, list_dir) are available.

Tool execution follows a typed validation loop inspired by `@mariozechner/pi-ai`: arguments are normalized for provider quirks, checked against the tool schema, and validation failures are returned as tool errors rather than Python exceptions. For example, provider-style `read_file` ranges such as `view_range: [10, 20]` are converted to Clod's `offset`/`limit` form.

## Model Metadata & Handoffs

`/providers` returns both the raw model IDs and structured metadata for each model:

- `api`: provider API family such as `anthropic-messages`, `google-generative-ai`, or `openai-completions`
- `input`: supported input types such as `text` and `image`
- `supports_tools`: whether the provider can call tools
- `reasoning`: heuristic flag for thinking/reasoning-capable models
- `context_window` and `max_output_tokens`: routing hints for long tasks

Saved sessions also serialize assistant message provider/model metadata. This keeps conversation history portable when a session is resumed with a different model after provider outages, rate limits, or manual model switching.

## Markdown Skills

Clod loads project markdown instructions into the agent system prompt from:

- `AGENTS.md`
- `CLOD.md`
- `skills/*/SKILL.md`

This repository includes `skills/karpathy-guidelines/SKILL.md`, adapted from `afstudy20-gif/karpathy-skills`, to encourage simpler, more surgical, and more verifiable coding changes.

## Providers & Models

| Provider | Alias(es) | Key env var | Cost |
|----------|-----------|-------------|------|
| **Anthropic** | `anthropic` | `ANTHROPIC_API_KEY` | $$ |
| **OpenAI ChatGPT** | `openai`, `chatgpt`, `gpt` | `OPENAI_API_KEY` | $$ |
| **Google Gemini** | `gemini`, `google` | `GOOGLE_API_KEY` | $ |
| **Groq** | `groq` | `GROQ_API_KEY` | free tier |
| **Mistral AI** | `mistral` | `MISTRAL_API_KEY` | $ |
| **DeepSeek** | `deepseek` | `DEEPSEEK_API_KEY` | very cheap |
| **NVIDIA NIM** | `nvidia`, `nim` | `NVIDIA_API_KEY` | build.nvidia.com |
| **Tavily Search** | `tavily`, `search` | `TAVILY_API_KEY` | search API |
| **Cohere** | `cohere` | `COHERE_API_KEY` | $ |
| **Ollama** | `ollama`, `local` | *(none — local)* | free |

## Architecture

```
main.py                  <- CLI entry point (Click + Rich + prompt_toolkit)
api.py                   <- FastAPI web server with SSE streaming
static/index.html        <- Web UI
src/
  providers/
    base.py              <- BaseProvider interface + data models
    anthropic_provider.py
    openai_provider.py
    gemini_provider.py
    cohere_provider.py
  tools/
    implementations.py   <- Tool functions (read/write/edit/bash/glob/grep)
    registry.py          <- Tool schemas + handler registry (with readonly tags)
    github_tools.py      <- GitHub REST API tools
  core/
    agent.py             <- Agentic loop with mode support (normal/explore/plan)
    config.py            <- API key + model preference management
    context.py           <- Token budget & history trimming
    project.py           <- Project root detection
    session.py           <- Session save/load/list/delete
    loop.py              <- Loop mode runner (background thread)
```
