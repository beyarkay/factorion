#!/bin/bash
# sft_smoke_test.sh — Runs SFT pre-training smoke test on RunPod GPU.
#
# Builds the Rust extension, runs a short SFT training, and logs to W&B.
# Pre-installed deps and Rust toolchain are expected in the Docker image
# (beyarkay/factorion-ci-gpu:latest). Code should already be at
# /workspace/factorion/.
#
# Required env vars:
#   WANDB_API_KEY       - W&B API key for logging
#   WANDB_PROJECT       - W&B project name (e.g. factorion)
#
# Optional env vars:
#   NUM_SAMPLES         - Number of SFT samples (default: 5000)
#   EPOCHS              - Number of training epochs (default: 10)
#   PR_NUMBER           - PR number for tagging
#   COMMIT_SHA          - Git commit SHA for tagging
#   RUNPOD_POD_ID       - RunPod pod ID (for self-terminate watchdog)
#   RUNPOD_API_KEY      - RunPod API key (for watchdog)

set -euo pipefail

NUM_SAMPLES="${NUM_SAMPLES:-50000}"
EPOCHS="${EPOCHS:-30}"
WANDB_PROJECT="${WANDB_PROJECT:-factorion}"
PR_NUMBER="${PR_NUMBER:-unknown}"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

echo "============================================"
echo "  Factorion SFT Smoke Test"
echo "============================================"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CUDA:            $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "  Samples:         ${NUM_SAMPLES}"
echo "  Epochs:          ${EPOCHS}"
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

# ── Ensure Rust is on PATH ────────────────────────────────────────
export PATH="/root/.cargo/bin:${PATH}"

# ── CuBLAS deterministic mode ────────────────────────────────────
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ── Build Rust extension ─────────────────────────────────────────
if [ -f factorion_rs/Cargo.toml ]; then
    echo ""
    echo ">>> Building factorion_rs from source..."
    (cd factorion_rs && maturin build --release --out dist && pip install dist/*.whl)
else
    echo ">>> WARNING: factorion_rs not available."
fi

# ── Configure W&B ────────────────────────────────────────────────
export WANDB_API_KEY

# ── Run SFT smoke test ───────────────────────────────────────────
echo ""
echo ">>> Starting SFT smoke test (${NUM_SAMPLES} samples, ${EPOCHS} epochs)..."

python sft.py \
    --seed 1 \
    --size 8 \
    --num-samples "$NUM_SAMPLES" \
    --epochs "$EPOCHS" \
    --track \
    --wandb-project-name "$WANDB_PROJECT" \
    --tags ci sft-smoke-test "pr:${PR_NUMBER}" "sha:${COMMIT_SHA}" \
    --summary-path /workspace/factorion/sft_summary.json

echo ""
echo "============================================"
echo "  SFT Smoke test completed successfully"
echo "============================================"

# ── Generate PR summary markdown ─────────────────────────────────
SUMMARY_JSON="/workspace/factorion/sft_summary.json"
SUMMARY_MD="/workspace/summary.md"

if [ -f "$SUMMARY_JSON" ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')
    python3 -c "
import json
s = json.load(open('$SUMMARY_JSON'))
gpu = '''$GPU_NAME'''
pr = '''$PR_NUMBER'''
sha = '''$COMMIT_SHA'''
wandb_url = s.get('wandb_url') or 'N/A'
wandb_link = f'[View on W&B]({wandb_url})' if wandb_url != 'N/A' else 'N/A'
print(f'''## SFT Smoke Test Results

| Metric | Value |
|--------|-------|
| **Best val accuracy** | {s['best_val_acc']:.4f} |
| **Val tile accuracy** | {s['val_tile_acc']:.4f} |
| **Val entity accuracy** | {s['val_ent_acc']:.4f} |
| **Val direction accuracy** | {s['val_dir_acc']:.4f} |
| **Val loss** | {s['val_loss']:.4f} |
| **Samples** | {s['num_samples']:,} |
| **Epochs** | {s['epochs']} |
| **Runtime** | {s['runtime_seconds']:.1f}s |
| **GPU** | {gpu} |
| **Grid size** | {s['size']}x{s['size']} |

{wandb_link}

<sub>Commit {sha[:8]} \u00b7 PR #{pr}</sub>''')
" > "$SUMMARY_MD"
    echo ">>> Summary written to $SUMMARY_MD"
else
    echo ">>> WARNING: sft_summary.json not found, skipping PR summary"
fi
