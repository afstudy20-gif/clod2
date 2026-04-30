#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DESKTOP_DIR="$ROOT_DIR/desktop-app"

cd "$ROOT_DIR"

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
elif [ -x "$ROOT_DIR/venv/bin/python" ]; then
  PYTHON="$ROOT_DIR/venv/bin/python"
elif [ -x "/Users/yh/Desktop/clod2/venv/bin/python" ]; then
  PYTHON="/Users/yh/Desktop/clod2/venv/bin/python"
else
  python3 -m venv "$ROOT_DIR/.venv"
  PYTHON="$ROOT_DIR/.venv/bin/python"
fi

"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"

# Setup script syntax check
bash -n "$0" || exit 1

cd "$DESKTOP_DIR"

if [ ! -d "node_modules" ]; then
  npm install
else
  echo "node_modules exists. Skipping npm install. (Run 'npm install' manually if you changed package.json)"
fi

echo "Cleaning previous build..."
rm -rf dist

echo "Packaging macOS app..."
npm run package:mac

echo "Clod.app ready at: $DESKTOP_DIR/dist/mac-arm64/Clod.app"
