#!/usr/bin/env bash
# Install/enable the mod, read GUI-host RCON settings, and serve a local or
# W&B-hosted checkpoint. The Python server waits until Factorio hosts a game.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <checkpoint.pt|wandb-run-id|wandb-run-url> [server flags...]" >&2
  exit 1
fi
CHECKPOINT="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

case "$(uname -s)" in
  Darwin*) CONFIG="$HOME/Library/Application Support/factorio/config/config.ini" ;;
  Linux*)  CONFIG="$HOME/.factorio/config/config.ini" ;;
  *)
    echo "Automatic GUI RCON discovery supports macOS and Linux." >&2
    echo "Run server/server.py manually on this platform." >&2
    exit 1
    ;;
esac

if [[ ! -f "$CONFIG" ]]; then
  echo "Factorio config not found at $CONFIG; launch Factorio once first." >&2
  exit 1
fi

"$SCRIPT_DIR/install_mod.sh"

SOCKET=$(sed -n -E 's/^[[:space:]]*local-rcon-socket[[:space:]]*=[[:space:]]*([^;[:space:]]+).*/\1/p' "$CONFIG" | tail -1)
PASSWORD=$(sed -n -E 's/^[[:space:]]*local-rcon-password[[:space:]]*=[[:space:]]*(.*[^[:space:]])[[:space:]]*$/\1/p' "$CONFIG" | tail -1)

if [[ -z "$SOCKET" || -z "$PASSWORD" ]]; then
  echo "GUI RCON is not configured in $CONFIG." >&2
  echo "Add these under [other], then restart Factorio:" >&2
  echo "  local-rcon-socket=127.0.0.1:64502" >&2
  echo "  local-rcon-password=<choose-a-password>" >&2
  exit 1
fi

RCON_HOST="${SOCKET%:*}"
RCON_PORT="${SOCKET##*:}"

echo "[serve] Checkpoint: $CHECKPOINT"
echo "[serve] RCON:      $RCON_HOST:$RCON_PORT"
echo "[serve] Start or restart Factorio, then choose Play > Multiplayer > Host new game."
echo "[serve] Waiting is normal until the hosted game has loaded."

cd "$REPO_ROOT"
exec uv run python factorion-mod/server/server.py \
  --checkpoint "$CHECKPOINT" \
  --rcon-host "$RCON_HOST" \
  --rcon-port "$RCON_PORT" \
  --rcon-password "$PASSWORD" \
  "$@"
