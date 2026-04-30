#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

echo "Clod macOS VS Code setup"
echo "Repo: $ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required. Install it with Homebrew or python.org first." >&2
  exit 1
fi

python3 -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install -r "$ROOT_DIR/requirements.txt"

if command -v npm >/dev/null 2>&1; then
  (cd "$ROOT_DIR/vscode-extension" && npm install)
else
  echo "npm was not found. The extension can still run in development if VS Code can load it, but npm install/check is unavailable."
fi

echo
echo "Done."
echo "VS Code settings:"
echo "  clod.backendPath = $ROOT_DIR"
echo "  clod.pythonPath  = $VENV_DIR/bin/python"
echo
echo "Open the extension folder in VS Code:"
echo "  code $ROOT_DIR/vscode-extension"
echo
echo "Then press F5, run 'Clod: Start Backend', and run 'Clod: Open Chat'."
