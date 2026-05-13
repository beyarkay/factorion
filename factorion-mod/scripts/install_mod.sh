#!/usr/bin/env bash
# Symlink the mod/ directory into Factorio's mods/ folder so changes to
# control.lua and friends are picked up on the next /mods reload (no zip
# step needed during development).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_SRC="$(cd "$SCRIPT_DIR/../mod" && pwd)"

# Read the internal name + version from info.json
NAME=$(grep -E '^\s*"name"' "$MOD_SRC/info.json" | head -1 | sed -E 's/.*"name"\s*:\s*"([^"]+)".*/\1/')
VERSION=$(grep -E '^\s*"version"' "$MOD_SRC/info.json" | head -1 | sed -E 's/.*"version"\s*:\s*"([^"]+)".*/\1/')

case "$(uname -s)" in
  Darwin*) FACTORIO_DIR="$HOME/Library/Application Support/factorio" ;;
  Linux*)  FACTORIO_DIR="$HOME/.factorio" ;;
  MINGW*|MSYS*|CYGWIN*) FACTORIO_DIR="${APPDATA:-$HOME/AppData/Roaming}/Factorio" ;;
  *)       echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

MODS_DIR="$FACTORIO_DIR/mods"
if [[ ! -d "$MODS_DIR" ]]; then
  echo "Factorio mods dir not found at $MODS_DIR — launch Factorio once to create it." >&2
  exit 1
fi

TARGET="$MODS_DIR/${NAME}_${VERSION}"

if [[ -L "$TARGET" || -e "$TARGET" ]]; then
  echo "Removing existing $TARGET"
  rm -rf "$TARGET"
fi

ln -s "$MOD_SRC" "$TARGET"
echo "Linked $MOD_SRC -> $TARGET"

# Make sure mod-list.json enables the mod (idempotent).
MOD_LIST="$MODS_DIR/mod-list.json"
if [[ -f "$MOD_LIST" ]]; then
  if ! grep -q "\"$NAME\"" "$MOD_LIST"; then
    echo "Note: '$NAME' is not in mod-list.json yet. Enable it from the in-game Mods menu."
  fi
else
  echo "Note: mod-list.json missing — Factorio will create one on next launch."
fi

echo "Done. Launch Factorio and enable the mod from the Mods menu if it isn't already."
