#!/bin/bash
# SessionStart hook: prepare the factorion dev environment so tests, linters,
# and training scripts work in a fresh container (e.g. Claude Code on the web).
#
# Steps mirror the "Setup" section of CLAUDE.md:
#   1. Check out the factorio-data submodule (factorion.py needs it).
#   2. uv sync       -> create .venv and install runtime + dev deps from uv.lock.
#   3. maturin build -> compile the factorion_rs Rust extension into the venv.
#
# The hook is idempotent: re-running it is safe and cheap once things are built.
set -euo pipefail

# Only run automatically in remote/web sessions. Local sessions already have a
# set-up checkout. (Run manually with CLAUDE_CODE_REMOTE=true to test.)
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

cd "${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel)}"

echo "[session-start] Initializing git submodules (factorio-data)..."
git submodule update --init --recursive

echo "[session-start] Syncing Python environment with uv..."
uv sync

echo "[session-start] Building the factorion_rs Rust extension..."
uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml

echo "[session-start] Environment ready."
