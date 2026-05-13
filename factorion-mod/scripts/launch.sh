#!/usr/bin/env bash
# Launch Factorio with RCON enabled (auto-generated port + password) and
# start the Factorion model server pointed at the same RCON endpoint.
# Wraps the flags the user would otherwise have to remember — they only
# pass --checkpoint.
#
# Usage:
#   factorion-mod/scripts/launch.sh path/to/agent.pt [extra server flags...]
#
# Env overrides:
#   FACTORIO_BIN   path to the factorio binary (auto-detected per OS)
#   RCON_PORT      port to bind (default: random free port in 27000..27999)
#   RCON_PASSWORD  RCON password (default: random hex)
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <checkpoint.pt> [extra server flags...]" >&2
  exit 1
fi
CHECKPOINT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# ----- locate factorio binary ------------------------------------------------
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
  echo "  FACTORIO_BIN=/path/to/factorio $0 $CHECKPOINT" >&2
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

echo "[launch] Factorio binary : $FACTORIO_BIN"
echo "[launch] RCON port       : $RCON_PORT"
echo "[launch] RCON password   : (hidden)"

# ----- spawn Factorio --------------------------------------------------------
"$FACTORIO_BIN" \
  --rcon-bind "127.0.0.1:$RCON_PORT" \
  --rcon-password "$RCON_PASSWORD" \
  &
FACTORIO_PID=$!
echo "[launch] Factorio pid    : $FACTORIO_PID"

cleanup() {
  echo "[launch] Stopping…"
  if kill -0 "$FACTORIO_PID" 2>/dev/null; then
    kill "$FACTORIO_PID" 2>/dev/null || true
  fi
  if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# Wait for the RCON port to accept connections. Factorio opens it only
# once a save (or the title screen, depending on version) is fully loaded,
# so this can take a few seconds. The server itself has reconnect logic
# but giving it a head start avoids a spurious "auth failed" on launch.
echo "[launch] Waiting for Factorio RCON on 127.0.0.1:$RCON_PORT …"
for _ in $(seq 1 60); do
  if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(0.25); sys.exit(0 if s.connect_ex(('127.0.0.1', $RCON_PORT))==0 else 1)" 2>/dev/null; then
    echo "[launch] RCON listening."
    break
  fi
  sleep 1
done

# ----- start the server ------------------------------------------------------
cd "$REPO_ROOT"
echo "[launch] Starting Factorion server (uv run python factorion-mod/server/server.py)…"
uv run python factorion-mod/server/server.py \
  --checkpoint "$CHECKPOINT" \
  --rcon-host 127.0.0.1 \
  --rcon-port "$RCON_PORT" \
  --rcon-password "$RCON_PASSWORD" \
  "$@" &
SERVER_PID=$!

wait "$FACTORIO_PID"
