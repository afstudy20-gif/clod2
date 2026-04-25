# CClaude — Multi-Provider AI Coding Assistant

A terminal-based AI coding assistant (Claude Code alternative) that works with **Anthropic Claude**, **OpenAI ChatGPT**, **Google Gemini**, **Groq**, **Mistral**, **DeepSeek**, **NVIDIA NIM**, **Cohere**, and **Ollama** — using your own API keys.

## Features

- **Multi-provider**: Switch between 9 providers in one session
- **Agentic tool use**: reads/writes/edits files, runs bash commands, searches code
- **Session resume**: Save and load conversation history across restarts
- **Project-aware**: Auto-detects project root, resolves paths relative to it
- **Modes**: Normal, Explore (read-only), and Plan (plan then execute)
- **Loop mode**: Run prompts on a recurring interval
- **Interactive model picker**: Browse and switch models easily
- **Streaming**: Responses stream in real time with tool execution indicators
- **Context management**: Automatically trims history to stay within token limits

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
```

### 3. Run

```bash
# Default: Anthropic Claude
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

## Providers & Models

| Provider | Alias(es) | Key env var | Cost |
|----------|-----------|-------------|------|
| **Anthropic Claude** | `claude`, `anthropic` | `ANTHROPIC_API_KEY` | $$ |
| **OpenAI ChatGPT** | `openai`, `chatgpt`, `gpt` | `OPENAI_API_KEY` | $$ |
| **Google Gemini** | `gemini`, `google` | `GOOGLE_API_KEY` | $ |
| **Groq** | `groq` | `GROQ_API_KEY` | free tier |
| **Mistral AI** | `mistral` | `MISTRAL_API_KEY` | $ |
| **DeepSeek** | `deepseek` | `DEEPSEEK_API_KEY` | very cheap |
| **NVIDIA NIM** | `nvidia`, `nim` | `NVIDIA_API_KEY` | build.nvidia.com |
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
