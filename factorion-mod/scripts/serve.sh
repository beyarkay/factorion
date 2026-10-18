#!/usr/bin/env bash
# Install/enable the mod and serve whichever checkpoint the mod's "Factorion
# checkpoint" setting names. The server reads (or enables) RCON in Factorio's
# config.ini and waits until Factorio hosts a game.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

"$SCRIPT_DIR/install_mod.sh"

cd "$REPO_ROOT"
exec uv run python -m factorion_server "$@"
