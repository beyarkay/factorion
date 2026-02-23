#!/bin/bash
# gpu_sweep.sh — Runs W&B sweep agents on a RunPod GPU pod.
#
# Builds the Rust extension, then launches multiple W&B sweep agents in
# parallel (to maximise GPU utilisation via CUDA time-slicing). Each agent
# pulls hyperparameters from the W&B Bayesian sweep controller, runs
# ppo.py, and reports metrics back.
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
#   SWEEP_COUNT         - Number of iterations per agent process (default: 10)
#   AGENTS_PER_POD      - Number of parallel wandb agents on this pod (default: 1)
#   WANDB_PROJECT       - W&B project name (default: factorion)
#   PR_NUMBER           - PR number for tagging
#   COMMIT_SHA          - Git commit SHA for tagging
#   AGENT_ID            - Pod identifier for multi-pod runs (default: 0)
#   RUNPOD_POD_ID       - RunPod pod ID (for self-terminate watchdog)
#   RUNPOD_API_KEY      - RunPod API key (for watchdog)

set -euo pipefail

SWEEP_ID="${SWEEP_ID:?Must set SWEEP_ID (entity/project/sweep_id)}"
SWEEP_COUNT="${SWEEP_COUNT:-10}"
AGENTS_PER_POD="${AGENTS_PER_POD:-1}"
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
echo "  Iters/agent:     ${SWEEP_COUNT}"
echo "  Agents on pod:   ${AGENTS_PER_POD}"
echo "  Pod ID:          ${AGENT_ID}"
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
echo ">>> Launching ${AGENTS_PER_POD} parallel W&B sweep agents (${SWEEP_COUNT} iterations each)..."
echo ">>> Sweep: ${SWEEP_ID}"
echo ""

# ── Launch sweep agents in parallel ──────────────────────────────
# Each agent process independently pulls work from the W&B sweep controller.
# CUDA time-slices between processes, keeping GPU utilisation high.
PIDS=()
mkdir -p /workspace/agent_logs

for i in $(seq 0 $((AGENTS_PER_POD - 1))); do
    LOG="/workspace/agent_logs/agent_${AGENT_ID}_${i}.log"
    echo ">>> Starting agent ${AGENT_ID}.${i} (log: ${LOG})"
    wandb agent --count "$SWEEP_COUNT" "$SWEEP_ID" > "$LOG" 2>&1 &
    PIDS+=($!)
done

echo ">>> All ${AGENTS_PER_POD} agents launched (PIDs: ${PIDS[*]})"
echo ""

# ── Wait for all agents and track failures ───────────────────────
FAILED=0
for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    if wait "$PID"; then
        echo ">>> Agent ${AGENT_ID}.${i} (PID ${PID}) finished successfully"
    else
        EXIT_CODE=$?
        echo ">>> Agent ${AGENT_ID}.${i} (PID ${PID}) failed with exit code ${EXIT_CODE}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "============================================"
echo "  Sweep pod ${AGENT_ID} completed"
echo "  Agents: ${AGENTS_PER_POD}, Iters/agent: ${SWEEP_COUNT}"
echo "  Failed agents: ${FAILED}/${AGENTS_PER_POD}"
echo "============================================"

# Dump agent logs for debugging
for i in $(seq 0 $((AGENTS_PER_POD - 1))); do
    LOG="/workspace/agent_logs/agent_${AGENT_ID}_${i}.log"
    if [ -f "$LOG" ]; then
        echo ""
        echo "--- Agent ${AGENT_ID}.${i} log (last 20 lines) ---"
        tail -20 "$LOG"
    fi
done

# Fail the job if any agent failed
if [ "$FAILED" -gt 0 ]; then
    echo "ERROR: ${FAILED} agent(s) failed"
    exit 1
fi
