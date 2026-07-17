#!/usr/bin/env bash
# Symlink the mod/ directory into Factorio's mods/ folder so changes to
# control.lua and friends are picked up on the next /mods reload (no zip
# step needed during development).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_SRC="$(cd "$SCRIPT_DIR/../mod" && pwd)"

# Read the internal name + version from info.json. Use POSIX bracket
# classes ([[:space:]]) not the GNU-only \s escape — BSD/macOS sed treats
# \s as a literal 's', which silently yields the whole line as the value.
NAME=$(grep -E '"name"' "$MOD_SRC/info.json" | head -1 | sed -E 's/.*"name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')
VERSION=$(grep -E '"version"' "$MOD_SRC/info.json" | head -1 | sed -E 's/.*"version"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/')

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

# Remove older development symlinks for this same source tree when the mod
# version changes. Published zip installs are left alone.
for EXISTING in "$MODS_DIR/${NAME}_"*; do
  if [[ -L "$EXISTING" && "$(readlink "$EXISTING")" == "$MOD_SRC" ]]; then
    echo "Removing old development link $EXISTING"
    rm "$EXISTING"
  fi
done

if [[ -L "$TARGET" || -e "$TARGET" ]]; then
  echo "Removing existing $TARGET"
  rm -rf "$TARGET"
fi

ln -s "$MOD_SRC" "$TARGET"
echo "Linked $MOD_SRC -> $TARGET"

# Make sure mod-list.json enables the mod (idempotent). Factorio's JSON is
# structured, so use the system Python rather than brittle sed replacement.
MOD_LIST="$MODS_DIR/mod-list.json"
if [[ -f "$MOD_LIST" ]]; then
  python3 - "$MOD_LIST" "$NAME" <<'PY'
import json
import os
import sys
import tempfile

path, name = sys.argv[1:]
with open(path) as f:
    data = json.load(f)
mods = data.setdefault("mods", [])
entry = next((mod for mod in mods if mod.get("name") == name), None)
if entry is None:
    mods.append({"name": name, "enabled": True})
else:
    entry["enabled"] = True
fd, tmp = tempfile.mkstemp(prefix="mod-list.", suffix=".json", dir=os.path.dirname(path))
try:
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)
print(f"Enabled '{name}' in {path}")
PY
else
  echo "Note: mod-list.json missing — Factorio will create one on next launch."
fi

echo "Done. Restart Factorio so it loads the linked mod."
