#!/usr/bin/env bash
# Download the free headless Factorio server (Linux only, no account needed)
# and link the factorion mod into it. Idempotent; prints the binary path.
#
# Usage:
#   factorion-mod/scripts/install_headless.sh
#
# Env overrides:
#   FACTORIO_HEADLESS_DIR  install dir (default: ~/.cache/factorio-headless)
#   FACTORIO_VERSION       release to download (default: stable)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_SRC="$(cd "$SCRIPT_DIR/../mod" && pwd)"
DIR="${FACTORIO_HEADLESS_DIR:-$HOME/.cache/factorio-headless}"
BIN="$DIR/factorio/bin/x64/factorio"

if [[ ! -x "$BIN" ]]; then
  mkdir -p "$DIR"
  curl -fsSL "https://factorio.com/get-download/${FACTORIO_VERSION:-stable}/headless/linux64" | tar xJ -C "$DIR"
fi

# The headless tarball keeps its write data (mods, saves) inside the install dir.
MODS_DIR="$DIR/factorio/mods"
mkdir -p "$MODS_DIR"
rm -f "$MODS_DIR"/factorion_*
VERSION=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["version"])' "$MOD_SRC/info.json")
ln -s "$MOD_SRC" "$MODS_DIR/factorion_$VERSION"

"$BIN" --version | head -1
echo "$BIN"
