#!/bin/bash
# gpu_sweep.sh — Runs a W&B sweep agent on a RunPod GPU pod.
#
# Builds the Rust extension, then launches a W&B sweep agent that
# executes sweep iterations (each running ppo.py with different
# hyperparameters selected by the W&B Bayesian sweep controller).
#
# Pre-installed deps and Rust toolchain are expected in the Docker image
# (beyarkay/factorion-ci-gpu:latest). Code should already be at
# /workspace/factorion/.
#
# Required env vars:
#   WANDB_API_KEY       - W&B API key for logging
#   SWEEP_ID            - Full W&B sweep path (entity/project/sweep_id)
#
# Optional env vars:
#   SWEEP_COUNT         - Number of iterations for this agent (default: 10)
#   WANDB_PROJECT       - W&B project name (default: factorion)
#   PR_NUMBER           - PR number for tagging
#   COMMIT_SHA          - Git commit SHA for tagging
#   AGENT_ID            - Agent identifier for multi-agent runs (default: 0)
#   RUNPOD_POD_ID       - RunPod pod ID (for self-terminate watchdog)
#   RUNPOD_API_KEY      - RunPod API key (for watchdog)

set -euo pipefail

SWEEP_ID="${SWEEP_ID:?Must set SWEEP_ID (entity/project/sweep_id)}"
SWEEP_COUNT="${SWEEP_COUNT:-10}"
WANDB_PROJECT="${WANDB_PROJECT:-factorion}"
PR_NUMBER="${PR_NUMBER:-unknown}"
COMMIT_SHA="${COMMIT_SHA:-unknown}"
AGENT_ID="${AGENT_ID:-0}"

echo "============================================"
echo "  Factorion W&B Sweep Agent"
echo "============================================"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CUDA:            $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "  Sweep ID:        ${SWEEP_ID}"
echo "  Iterations:      ${SWEEP_COUNT}"
echo "  Agent ID:        ${AGENT_ID}"
echo "  W&B project:     ${WANDB_PROJECT}"
echo "  PR:              ${PR_NUMBER}"
echo "  Commit:          ${COMMIT_SHA}"
echo "============================================"

# ── Safety net: self-terminate after 4 hours if cleanup fails ─────
if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo ">>> Starting self-terminate watchdog (4h timeout)..."
    nohup bash -c "
      sleep 14400
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

# ── CuBLAS deterministic mode (required by torch.use_deterministic_algorithms) ─
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ── Build Rust extension (deps cached in image, ~30s) ────────────
if [ -f factorion_rs/Cargo.toml ]; then
    echo ""
    echo ">>> Building factorion_rs from source..."
    (cd factorion_rs && maturin build --release --out dist && pip install dist/*.whl)
else
    echo ">>> WARNING: factorion_rs not available."
    echo "    The sweep may fail if ppo.py requires it."
fi

# ── Override run_sweep.sh for pod environment (no venv needed) ────
# The Docker image has all deps installed globally, so skip venv activation.
cat > run_sweep.sh << 'EOF'
#!/bin/bash
python ppo.py "$@"
EOF
chmod +x run_sweep.sh

# ── Configure W&B ────────────────────────────────────────────────
export WANDB_API_KEY

echo ""
echo ">>> Starting W&B sweep agent (${SWEEP_COUNT} iterations)..."
echo ">>> Sweep: ${SWEEP_ID}"
echo ""

# ── Run the sweep agent ──────────────────────────────────────────
# wandb agent pulls hyperparameters from the W&B sweep controller,
# executes the command defined in sweep.yaml for each iteration,
# and reports metrics back. --count limits the number of runs.
wandb agent --count "$SWEEP_COUNT" "$SWEEP_ID"

echo ""
echo "============================================"
echo "  Sweep agent ${AGENT_ID} completed"
echo "  Ran ${SWEEP_COUNT} iterations"
echo "============================================"
