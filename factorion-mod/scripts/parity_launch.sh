#!/usr/bin/env bash
# Launch a headless Factorio with RCON + the factorion mod and run the
# engine-parity harness against it (issue #261). Everything is
# fire-and-forget: Factorio is torn down when the harness exits.
#
# Usage:
#   factorion-mod/scripts/parity_launch.sh [parity.py flags...]
# e.g.
#   factorion-mod/scripts/parity_launch.sh --lessons MOVE_ONE_ITEM --seeds 3
#
# Env overrides:
#   FACTORIO_BIN   path to the factorio binary (auto-detected per OS)
#   RCON_PORT      port to bind (default: random free port)
#   RCON_PASSWORD  RCON password (default: random hex)
#
# Prereq: the mod must be installed (factorion-mod/scripts/install_mod.sh).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ----- locate factorio binary (mirrors launch.sh) ----------------------------
case "$(uname -s)" in
  Darwin*)
    DEFAULT_BINS=(
      "/Applications/factorio.app/Contents/MacOS/factorio"
      "$HOME/Library/Application Support/Steam/steamapps/common/Factorio/factorio.app/Contents/MacOS/factorio"
    )
    ;;
  Linux*)
    DEFAULT_BINS=(
      "$HOME/.factorio/bin/x64/factorio"
      "$HOME/.local/share/Steam/steamapps/common/Factorio/bin/x64/factorio"
      "/usr/share/factorio/bin/x64/factorio"
    )
    ;;
  MINGW*|MSYS*|CYGWIN*)
    DEFAULT_BINS=(
      "/c/Program Files/Factorio/bin/x64/factorio.exe"
      "/c/Program Files (x86)/Steam/steamapps/common/Factorio/bin/x64/factorio.exe"
    )
    ;;
  *) echo "Unsupported OS: $(uname -s)" >&2; exit 1 ;;
esac

if [[ -z "${FACTORIO_BIN:-}" ]]; then
  for candidate in "${DEFAULT_BINS[@]}"; do
    if [[ -x "$candidate" ]]; then FACTORIO_BIN="$candidate"; break; fi
  done
fi

if [[ -z "${FACTORIO_BIN:-}" || ! -x "$FACTORIO_BIN" ]]; then
  echo "Could not find Factorio binary. Set FACTORIO_BIN, e.g.:" >&2
  echo "  FACTORIO_BIN=/path/to/factorio $0 $*" >&2
  exit 1
fi

# ----- pick port + password --------------------------------------------------
pick_free_port() {
  python3 - <<'PY'
import socket
s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
PY
}

: "${RCON_PORT:=$(pick_free_port)}"
: "${RCON_PASSWORD:=$(python3 -c 'import secrets; print(secrets.token_hex(16))')}"

echo "[parity] Factorio binary : $FACTORIO_BIN"
echo "[parity] RCON port       : $RCON_PORT"

# ----- spawn a headless Factorio ---------------------------------------------
# The freeplay scenario gives us a normal game world without needing a
# save file; RCON binds at launch in --start-server modes.
"$FACTORIO_BIN" \
  --start-server-load-scenario base/freeplay \
  --rcon-bind "127.0.0.1:$RCON_PORT" \
  --rcon-password "$RCON_PASSWORD" \
  &
FACTORIO_PID=$!
echo "[parity] Factorio pid    : $FACTORIO_PID"

cleanup() {
  echo "[parity] Stopping Factorio…"
  if kill -0 "$FACTORIO_PID" 2>/dev/null; then
    kill "$FACTORIO_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[parity] Waiting for Factorio RCON on 127.0.0.1:$RCON_PORT …"
for _ in $(seq 1 60); do
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.25); sys.exit(0 if s.connect_ex(('127.0.0.1', $RCON_PORT))==0 else 1)" 2>/dev/null; then
    echo "[parity] RCON listening."
    break
  fi
  sleep 1
done

# ----- run the harness --------------------------------------------------------
cd "$REPO_ROOT"
uv run python factorion-mod/server/parity.py \
  --rcon-host 127.0.0.1 \
  --rcon-port "$RCON_PORT" \
  --rcon-password "$RCON_PASSWORD" \
  "$@"
