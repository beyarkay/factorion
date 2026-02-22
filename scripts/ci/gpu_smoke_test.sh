#!/bin/bash
# gpu_smoke_test.sh — Runs on the RunPod GPU pod.
#
# Installs dependencies, runs a short RL training, and logs to W&B.
# The code should already be transferred to /workspace/factorion/ before
# this script runs.
#
# Required env vars:
#   WANDB_API_KEY       - W&B API key for logging
#   WANDB_PROJECT       - W&B project name (e.g. factorion-ci)
#   WANDB_RUN_NAME      - Unique run name for this CI run
#   TOTAL_TIMESTEPS     - Number of timesteps (default: 10000)
#   PR_NUMBER           - PR number for tagging
#   COMMIT_SHA          - Git commit SHA for tagging

set -euo pipefail

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-10000}"
WANDB_PROJECT="${WANDB_PROJECT:-factorion-ci}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-ci-smoke-test}"
PR_NUMBER="${PR_NUMBER:-unknown}"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

echo "============================================"
echo "  Factorion GPU Smoke Test"
echo "============================================"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CUDA:            $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "  Timesteps:       ${TOTAL_TIMESTEPS}"
echo "  W&B project:     ${WANDB_PROJECT}"
echo "  W&B run name:    ${WANDB_RUN_NAME}"
echo "  PR:              ${PR_NUMBER}"
echo "  Commit:          ${COMMIT_SHA}"
echo "============================================"

cd /workspace/factorion

# ── Install dependencies ──────────────────────────────────────────
echo ""
echo ">>> Installing Python dependencies..."
pip install -q -r requirements.txt

# Install factorion_rs if a Cargo.toml is present (compiled Rust extension)
if [ -f factorion_rs/Cargo.toml ]; then
    echo ">>> Building factorion_rs from source..."
    pip install -q maturin
    cd factorion_rs && maturin develop --release && cd ..
elif pip show factorion-rs > /dev/null 2>&1; then
    echo ">>> factorion_rs already installed"
else
    echo ">>> WARNING: factorion_rs not available."
    echo "    The smoke test may fail if ppo.py requires it."
    echo "    Install it or add Cargo.toml to build from source."
fi

# ── Log in to W&B ────────────────────────────────────────────────
echo ""
echo ">>> Configuring W&B..."
# Use WANDB_API_KEY env var directly (wandb respects it) to avoid
# exposing the key in process listings.
export WANDB_API_KEY

# ── Run the smoke test ───────────────────────────────────────────
echo ""
echo ">>> Starting RL smoke test (${TOTAL_TIMESTEPS} timesteps)..."

# Use a small number of envs and steps to keep it fast but meaningful.
# Tags allow the metrics checker to find this run.
python ppo.py \
    --seed 42 \
    --env-id factorion/FactorioEnv-v0 \
    --track \
    --wandb-project-name "$WANDB_PROJECT" \
    --exp-name "$WANDB_RUN_NAME" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --num-envs 4 \
    --num-steps 128 \
    --tags '["ci", "smoke-test", "pr:'"$PR_NUMBER"'", "sha:'"$COMMIT_SHA"'"]'

echo ""
echo "============================================"
echo "  Smoke test completed successfully"
echo "============================================"
