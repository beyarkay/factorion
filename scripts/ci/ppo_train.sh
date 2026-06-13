#!/bin/bash
# ppo_train.sh — Runs a PPO (RL) training run on RunPod GPU, continuing from an
# SFT checkpoint. Builds the Rust extension, loads the actor (+ critic) via
# --start-from (a W&B run id OR a local .pt path), runs PPO with a critic
# warm-up, and logs to W&B. Pre-installed deps + Rust toolchain are expected in
# the Docker image (beyarkay/factorion-ci-gpu:latest); code should already be at
# /workspace/factorion/.
#
# Required env vars:
#   WANDB_API_KEY          - W&B API key (logging + artifact download for --start-from)
#   WANDB_PROJECT          - W&B project name (e.g. factorion)
#   START_FROM             - SFT checkpoint: a W&B run id OR a local .pt path
#
# Optional env vars:
#   TOTAL_TIMESTEPS        - PPO total timesteps (default: 500000)
#   SIZE                   - grid size; MUST match the SFT checkpoint (default: 11)
#   CRITIC_WARMUP          - critic-only warm-up iterations (default: 10)
#   START_CURRICULUM_LEVEL - num_missing_entities cap; ~2*size blanks the whole
#                            MOVE_ONE_ITEM factory each episode (default: 22)
#   SEED                   - random seed (default: 1)
#   WATCHDOG_SECONDS       - self-terminate watchdog timeout (default: 7200 = 2h)
#   PR_NUMBER, COMMIT_SHA  - tagging
#   RUNPOD_POD_ID, RUNPOD_API_KEY - for the self-terminate watchdog

set -euo pipefail

: "${START_FROM:?START_FROM (a W&B run id or a .pt path) is required}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-500000}"
SIZE="${SIZE:-11}"
CRITIC_WARMUP="${CRITIC_WARMUP:-10}"
START_CURRICULUM_LEVEL="${START_CURRICULUM_LEVEL:-22}"
ENT_COEF="${ENT_COEF:-0}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
TARGET_KL="${TARGET_KL:-0.02}"
SEED="${SEED:-1}"
WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-7200}"
WANDB_PROJECT="${WANDB_PROJECT:-factorion}"
PR_NUMBER="${PR_NUMBER:-unknown}"
COMMIT_SHA="${COMMIT_SHA:-unknown}"

echo "============================================"
echo "  Factorion PPO Train (RL from SFT)"
echo "============================================"
echo "  GPU:                 $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CUDA:                $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "  start_from:          ${START_FROM}"
echo "  total_timesteps:     ${TOTAL_TIMESTEPS}"
echo "  size:                ${SIZE}"
echo "  critic_warmup:       ${CRITIC_WARMUP}"
echo "  start_curric_level:  ${START_CURRICULUM_LEVEL}"
echo "  ent_coef:            ${ENT_COEF}"
echo "  learning_rate:       ${LEARNING_RATE}"
echo "  target_kl:           ${TARGET_KL}"
echo "  seed:                ${SEED}"
echo "  watchdog:            ${WATCHDOG_SECONDS}s"
echo "  W&B project:         ${WANDB_PROJECT}"
echo "  PR:                  ${PR_NUMBER}"
echo "  Commit:              ${COMMIT_SHA}"
echo "============================================"

# ── Safety net: self-terminate after WATCHDOG_SECONDS if cleanup fails ─
if [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    echo ">>> Starting self-terminate watchdog (${WATCHDOG_SECONDS}s timeout)..."
    nohup bash -c "
      sleep ${WATCHDOG_SECONDS}
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

# ── Configure W&B (also used to download the --start-from artifact) ──
export WANDB_API_KEY

# ── Run PPO ──────────────────────────────────────────────────────
echo ""
echo ">>> Starting PPO (${TOTAL_TIMESTEPS} timesteps, from ${START_FROM})..."

python ppo.py \
    --seed "$SEED" \
    --env-id factorion/FactorioEnv-v0 \
    --size "$SIZE" \
    --start-from "$START_FROM" \
    --critic-warmup "$CRITIC_WARMUP" \
    --start-curriculum-level "$START_CURRICULUM_LEVEL" \
    --ent-coef-start "$ENT_COEF" \
    --ent-coef-end "$ENT_COEF" \
    --learning-rate "$LEARNING_RATE" \
    --target-kl "$TARGET_KL" \
    --total-timesteps "$TOTAL_TIMESTEPS" \
    --track \
    --wandb-project-name "$WANDB_PROJECT" \
    --tags ci ppo-train "pr:${PR_NUMBER}" "sha:${COMMIT_SHA}" "from:${START_FROM}" \
    --summary-path /workspace/factorion/ppo_summary.json

echo ""
echo "============================================"
echo "  PPO training completed successfully"
echo "============================================"

# ── Generate summary markdown (fetched by runpod-teardown) ───────
SUMMARY_JSON="/workspace/factorion/ppo_summary.json"
SUMMARY_MD="/workspace/summary.md"

if [ -f "$SUMMARY_JSON" ]; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')
    python3 -c "
import json
s = json.load(open('$SUMMARY_JSON'))
gpu = '''$GPU_NAME'''
pr = '''$PR_NUMBER'''
sha = '''$COMMIT_SHA'''
start_from = '''$START_FROM'''
wandb_url = s.get('wandb_url') or 'N/A'
wandb_link = f'[View on W&B]({wandb_url})' if wandb_url != 'N/A' else 'N/A'
print(f'''## PPO Train Results (RL from SFT)

| Metric | Value |
|--------|-------|
| **Moving-avg throughput** | {s['moving_avg_throughput']:.4f} |
| **Curriculum score** | {s['curriculum_score']:.4f} |
| **Max missing entities** | {s['max_missing_entities']} |
| **Total timesteps** | {s['total_timesteps']:,} |
| **Global step** | {s['global_step']:,} |
| **Runtime** | {s['runtime_human']} |
| **Steps/sec** | {s['sps']:,} |
| **Grid size** | {s['grid_size']}x{s['grid_size']} |
| **Started from** | {start_from} |
| **GPU** | {gpu} |

{wandb_link}

<sub>Commit {sha[:8]} · PR #{pr}</sub>''')
" > "$SUMMARY_MD"
    echo ">>> Summary written to $SUMMARY_MD"
else
    echo ">>> WARNING: ppo_summary.json not found, skipping summary"
fi
