#!/usr/bin/env bash
# Parity regression: after an engine change, check it didn't break Factorio
# parity. This is just the right args to the existing tools — it rebuilds the
# engine, dumps its textual fixtures, and replays every lesson (N seeds each)
# PLUS every fixture through a running Factorio, reporting any sink whose rate
# diverges from the real game. No bespoke comparison logic; parity.py already
# does the sweep + report + exit code.
#
# Needs Factorio hosting the mod with RCON reachable (see factorion-mod/CLAUDE.md).
# Exits non-zero if any factory mismatches beyond tolerance.
#
# Usage:
#   factorion-mod/scripts/parity_regression.sh --rcon-port 64502 \
#       --rcon-password <pw> [--seeds 40] [any other parity.py flags]
#
# Env: SEEDS (default 30), FIXTURES (dump path, default /tmp/factorion_fixtures.json).
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"

FIXTURES="${FIXTURES:-/tmp/factorion_fixtures.json}"
SEEDS="${SEEDS:-30}"

echo "[regression] building the engine…"
uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml >/dev/null

echo "[regression] dumping textual fixtures → $FIXTURES"
FIXTURE_DUMP_PATH="$FIXTURES" cargo test \
  --manifest-path factorion_rs/Cargo.toml --no-default-features \
  dump_fixtures_for_parity -- --ignored --nocapture >/dev/null

echo "[regression] replaying all lessons (${SEEDS} seeds) + fixtures through Factorio…"
# --lessons/--seeds/--game-speed are defaults; anything in "$@" (RCON args,
# a different --seeds, tolerances, --json-out) overrides via argparse.
exec uv run python factorion-mod/server/parity.py \
  --lessons all --seeds "$SEEDS" --game-speed 100 \
  --fixtures "$FIXTURES" "$@"
