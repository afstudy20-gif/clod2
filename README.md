# CClaude — Multi-Provider AI Coding Assistant

A terminal-based AI coding assistant (Claude Code alternative) that works with **Anthropic Claude**, **OpenAI ChatGPT**, and **Google Gemini** — using your own API keys.

## Features

- **Multi-provider**: Switch between Claude, GPT-4o, and Gemini in one session
- **Agentic tool use**: reads/writes/edits files, runs bash commands, searches code
- **Context management**: automatically trims history to stay within token limits
- **Streaming**: responses stream in real time
- **Persistent history**: command history saved across sessions

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
# or install as a command:
pip install -e .
```

### 2. Set API Keys

Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
# Edit .env and add your key(s)
```

Or use environment variables directly:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GOOGLE_API_KEY=AIza...
```

### 3. Run

```bash
# Default: Anthropic Claude
python main.py

# Use OpenAI
python main.py --provider openai

# Use Gemini
python main.py --provider gemini

# Specify a model
python main.py --provider openai --model gpt-4o-mini

# Pass API key directly (one-time)
python main.py --provider claude --key sk-ant-...
```

## In-Session Commands

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/reset` | Clear conversation history |
| `/model <name>` | Switch model |
| `/provider <name>` | Switch provider |
| `/models` | List models for current provider |
| `/key <api_key>` | Update API key |
| `/exit` | Quit |

## Available Tools

The assistant has access to these tools (function calling):

| Tool | Description |
|------|-------------|
| `read_file` | Read file contents with line numbers |
| `write_file` | Create or overwrite a file |
| `edit_file` | Replace a unique string in a file |
| `bash` | Run shell commands |
| `glob` | Find files by pattern (e.g. `**/*.py`) |
| `grep` | Search file contents with regex |
| `list_dir` | List directory contents |

## Cost Tips

API costs depend on token usage. To keep costs low:

- The assistant uses **tools to search** for relevant files rather than reading everything
- **Context trimming** automatically drops old messages when nearing token limits
- Use cheaper models for simple tasks: `gpt-4o-mini`, `claude-haiku-4-5`, `gemini-1.5-flash`

## Providers & Models

| Provider | Alias(es) | Key env var | Cost |
|----------|-----------|-------------|------|
| **Anthropic Claude** | `claude`, `anthropic` | `ANTHROPIC_API_KEY` | $$ |
| **OpenAI ChatGPT** | `openai`, `chatgpt`, `gpt` | `OPENAI_API_KEY` | $$ |
| **Google Gemini** | `gemini`, `google` | `GOOGLE_API_KEY` | $ |
| **Groq** | `groq` | `GROQ_API_KEY` | free tier |
| **Mistral AI** | `mistral` | `MISTRAL_API_KEY` | $ |
| **DeepSeek** | `deepseek` | `DEEPSEEK_API_KEY` | very cheap |
| **Cohere** | `cohere` | `COHERE_API_KEY` | $ |
| **Ollama** | `ollama`, `local` | *(none — local)* | free |

### Anthropic (Claude)
- `claude-sonnet-4-6` ← default, best balance
- `claude-opus-4-6` ← most capable
- `claude-haiku-4-5-20251001` ← fastest/cheapest

### OpenAI (ChatGPT)
- `gpt-4o` ← default
- `gpt-4o-mini` ← cheaper
- `o1`, `o3-mini` ← reasoning models

### Google (Gemini)
- `gemini-2.0-flash` ← default, fast
- `gemini-1.5-pro` ← most capable
- `gemini-1.5-flash` ← cheapest

### Groq (free tier, very fast)
- `llama-3.3-70b-versatile` ← default
- `llama-3.1-8b-instant` ← fastest
- `mixtral-8x7b-32768`

### Mistral AI
- `mistral-large-latest` ← default
- `codestral-latest` ← optimized for code
- `open-mistral-nemo` ← small/cheap

### DeepSeek (cheapest paid option)
- `deepseek-chat` ← default
- `deepseek-reasoner` ← reasoning model

### Cohere
- `command-r-plus` ← default, best
- `command-r` ← cheaper
- `command-light` ← fastest

### Ollama (100% local, free)
```bash
# Install Ollama first: https://ollama.com
ollama pull llama3.2         # download a model
python main.py --provider ollama --model llama3.2
```
- `llama3.2`, `llama3.1`, `codellama`, `mistral`, `qwen2.5-coder`, `phi4`, `deepseek-r1`

## Architecture

```
main.py                  ← CLI entry point (Click + Rich + prompt_toolkit)
src/
  providers/
    base.py              ← Abstract BaseProvider interface
    anthropic_provider.py
    openai_provider.py
    gemini_provider.py
  tools/
    implementations.py   ← Tool functions (read/write/edit/bash/glob/grep)
    registry.py          ← Tool schemas + handler registry
  core/
    agent.py             ← Agentic loop (streams response, handles tool calls)
    config.py            ← API key management (.env / config file)
    context.py           ← Token budget & history trimming
```

## Alternatives to Consider

Before building your own, these open-source projects do the same thing with your own API key:

- **[Aider](https://aider.chat/)** — CLI pair programmer, supports all major providers
- **[Cline](https://github.com/cline/cline)** — VS Code extension, autonomous agent
- **[Continue](https://continue.dev/)** — VS Code/JetBrains extension
- **[Open Interpreter](https://github.com/OpenInterpreter/open-interpreter)** — General-purpose agent

This project (`CClaude`) is a **from-scratch implementation** for learning and customization.
