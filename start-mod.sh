#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${FACTORION_CHECKPOINT:-hcozpmwt}"
RUST_MANIFEST="$REPO_ROOT/factorion_rs/Cargo.toml"

if [[ $# -gt 0 && "$1" != -* ]]; then
    CHECKPOINT="$1"
    shift
fi

cd "$REPO_ROOT"
if ! uv run python -c \
    'import factorion_rs; assert hasattr(factorion_rs, "simulate_throughput")' \
    >/dev/null 2>&1; then
    echo "[start-mod] Building the Factorion Rust extension (first run or source update)…"
    uv run maturin develop --release --manifest-path "$RUST_MANIFEST"
fi

exec "$REPO_ROOT/factorion-mod/scripts/serve.sh" "$CHECKPOINT" "$@"
