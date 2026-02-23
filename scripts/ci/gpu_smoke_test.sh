#!/bin/bash
# gpu_smoke_test.sh — Runs on the RunPod GPU pod.
#
# Builds the Rust extension, runs a short RL training, and logs to W&B.
# Python deps and Rust toolchain are pre-installed in the Docker image
# (beyarkay/factorion-ci-gpu:latest). The code should already be
# transferred to /workspace/factorion/ before this script runs.
#
# Required env vars:
#   WANDB_API_KEY       - W&B API key for logging
#   WANDB_PROJECT       - W&B project name (e.g. factorion)
#   TOTAL_TIMESTEPS     - Number of timesteps (default: 20000)
#   PR_NUMBER           - PR number for tagging
#   COMMIT_SHA          - Git commit SHA for tagging
#
# Optional env vars (for self-terminate watchdog):
#   RUNPOD_POD_ID       - RunPod pod ID (set automatically by RunPod)
#   RUNPOD_API_KEY      - RunPod API key (passed via pod env)

set -euo pipefail

TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-20000}"
WANDB_PROJECT="${WANDB_PROJECT:-factorion}"
PR_NUMBER="${PR_NUMBER:-unknown}"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

echo "============================================"
echo "  Factorion GPU Smoke Test"
echo "============================================"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CUDA:            $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "  Timesteps:       ${TOTAL_TIMESTEPS}"
echo "  W&B project:     ${WANDB_PROJECT}"
echo "  PR:              ${PR_NUMBER}"
echo "  Commit:          ${COMMIT_SHA}"
echo "============================================"

# ── Safety net: self-terminate after 2 hours if cleanup fails ─────
if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo ">>> Starting self-terminate watchdog (2h timeout)..."
    nohup bash -c "
      sleep 7200
      curl -s 'https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}' \
        -H 'Content-Type: application/json' \
        -d '{\"query\": \"mutation { podTerminate(input: {podId: \\\"${RUNPOD_POD_ID}\\\"}) }\"}'
    " &>/dev/null &
else
    echo ">>> Watchdog skipped (RUNPOD_POD_ID or RUNPOD_API_KEY not set)"
fi

cd /workspace/factorion

# ── Ensure Rust is on PATH (installed in Docker image at /root/.cargo/bin) ─
export PATH="/root/.cargo/bin:${PATH}"

# ── Build Rust extension (deps cached in image, ~30s) ────────────
if [ -f factorion_rs/Cargo.toml ]; then
    echo ""
    echo ">>> Building factorion_rs from source..."
    (cd factorion_rs && maturin build --release --out dist && pip install dist/*.whl)
else
    echo ">>> WARNING: factorion_rs not available."
    echo "    The smoke test may fail if ppo.py requires it."
fi

# ── Log in to W&B ────────────────────────────────────────────────
echo ""
echo ">>> Configuring W&B..."
export WANDB_API_KEY

# ── Run the smoke test ───────────────────────────────────────────
echo ""
echo ">>> Starting RL smoke test (${TOTAL_TIMESTEPS} timesteps)..."

python ppo.py \
    --seed 1 \
    --env-id factorion/FactorioEnv-v0 \
    --track \
    --wandb-project-name "$WANDB_PROJECT" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --tags '["ci", "smoke-test", "pr:'"$PR_NUMBER"'", "sha:'"$COMMIT_SHA"'"]'

echo ""
echo "============================================"
echo "  Smoke test completed successfully"
echo "============================================"
