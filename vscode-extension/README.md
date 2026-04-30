# Clod Agent VS Code Extension on macOS

Native VS Code panel for the local Clod backend.

## 1. Prepare macOS Dependencies

From the Clod repo root:

```bash
cd /Users/yh/Desktop/clod2/clod-agent
./scripts/setup-vscode-macos.sh
```

The script creates `.venv`, installs Python requirements, and runs `npm install` inside `vscode-extension` when npm is available.

## 2. Open The Extension In VS Code

```bash
code /Users/yh/Desktop/clod2/clod-agent/vscode-extension
```

If the `code` command is missing, open VS Code, press `Cmd+Shift+P`, run `Shell Command: Install 'code' command in PATH`, then retry.

## 3. Configure Extension Settings

In VS Code settings JSON for the Extension Development Host, use:

```json
{
  "clod.backendPath": "/Users/yh/Desktop/clod2/clod-agent",
  "clod.pythonPath": "/Users/yh/Desktop/clod2/venv/bin/python",
  "clod.serverUrl": "http://127.0.0.1:8000",
  "clod.defaultProvider": "nvidia",
  "clod.defaultModel": "nvidia/nemotron-3-super-120b-a12b"
}
```

You can switch provider/model later from settings. The backend still needs the matching API key in your shell environment or saved Clod config.

## 4. Run In Development

1. Open `/Users/yh/Desktop/clod2/clod-agent/vscode-extension` in VS Code.
2. Press `F5` to launch the Extension Development Host.
3. In the new VS Code window, open the project you want Clod to edit.
4. Press `Cmd+Shift+P`.
5. Run `Clod: Start Backend`.
6. Run `Clod: Open Chat`.

The extension sends the active VS Code workspace folder to the backend, so file tools and git commands run inside the project opened in the Extension Development Host.

## Commands

- `Clod: Open Chat`
- `Clod: Send Selection`
- `Clod: Start Backend`

## Useful Workflow

- Open your app/project folder in the Extension Development Host.
- Select code and run `Clod: Send Selection`.
- Ask Clod to edit, debug, run tests, start a local dev server, or push to GitHub.
- For GitHub pushes, make sure the opened workspace is the repo you want to push.

## Troubleshooting

- Backend path error: set `clod.backendPath` to `/Users/yh/Desktop/clod2/clod-agent`.
- Python import errors: use `/Users/yh/Desktop/clod2/venv/bin/python` or run `./scripts/setup-vscode-macos.sh` to create `/Users/yh/Desktop/clod2/clod-agent/.venv`.
- API key missing: export the key before launching VS Code from terminal, or save it in the Clod web UI.
- Push says no changes: check that the Extension Development Host opened the repo you edited, not the extension folder.
