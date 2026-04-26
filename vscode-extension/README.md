# Clod Agent VSCode Extension

Native VSCode extension wrapper for the local Clod backend.

## Run In Development

1. Open this folder in VSCode:

   `/Users/yh/claude/cclaude/vscode-extension`

2. Press `F5` to launch an Extension Development Host.

3. Run `Clod: Start Backend` if the backend is not already running.

4. Run `Clod: Open Chat`.

The extension sends the active VSCode workspace path to the backend, so file tools and git commands run inside the project you opened in VSCode.

## Commands

- `Clod: Open Chat`
- `Clod: Send Selection`
- `Clod: Start Backend`

## Settings

- `clod.serverUrl`
- `clod.backendPath`
- `clod.pythonPath`
- `clod.defaultProvider`
- `clod.defaultModel`
