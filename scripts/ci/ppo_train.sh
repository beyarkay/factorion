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
# Optional env vars (each is an OVERRIDE — leave unset/empty to use ppo.py's
# own default from training_config.py, so there is one source of truth):
#   TOTAL_TIMESTEPS        - PPO total timesteps
#   SIZE                   - grid size; MUST match the SFT checkpoint
#   CRITIC_WARMUP          - critic-only warm-up iterations
#   ENT_COEF               - entropy bonus (sets both --ent-coef-start & -end)
#   LEARNING_RATE          - peak LR
#   TARGET_KL              - KL early-stop threshold
#   SEED                   - random seed
#   WATCHDOG_SECONDS       - self-terminate watchdog timeout (default: 7200 = 2h)
#   PR_NUMBER, COMMIT_SHA  - tagging
#   RUNPOD_POD_ID, RUNPOD_API_KEY - for the self-terminate watchdog

set -euo pipefail

: "${START_FROM:?START_FROM (a W&B run id or a .pt path) is required}"
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
echo "  total_timesteps:     ${TOTAL_TIMESTEPS:-(ppo.py default)}"
echo "  size:                ${SIZE:-(ppo.py default)}"
echo "  critic_warmup:       ${CRITIC_WARMUP:-(ppo.py default)}"
echo "  ent_coef:            ${ENT_COEF:-(ppo.py default)}"
echo "  learning_rate:       ${LEARNING_RATE:-(ppo.py default)}"
echo "  target_kl:           ${TARGET_KL:-(ppo.py default)}"
echo "  seed:                ${SEED:-(ppo.py default)}"
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
# Only pass a flag when the caller set the override; otherwise ppo.py's
# training_config default applies (single source of truth).
echo ""
echo ">>> Starting PPO (from ${START_FROM})..."

PPO_ARGS=(--start-from "$START_FROM")
[ -n "${SEED:-}" ]            && PPO_ARGS+=(--seed "$SEED")
[ -n "${SIZE:-}" ]           && PPO_ARGS+=(--size "$SIZE")
[ -n "${CRITIC_WARMUP:-}" ]  && PPO_ARGS+=(--critic-warmup "$CRITIC_WARMUP")
[ -n "${ENT_COEF:-}" ]       && PPO_ARGS+=(--ent-coef-start "$ENT_COEF" --ent-coef-end "$ENT_COEF")
[ -n "${LEARNING_RATE:-}" ]  && PPO_ARGS+=(--learning-rate "$LEARNING_RATE")
[ -n "${TARGET_KL:-}" ]      && PPO_ARGS+=(--target-kl "$TARGET_KL")
[ -n "${TOTAL_TIMESTEPS:-}" ] && PPO_ARGS+=(--total-timesteps "$TOTAL_TIMESTEPS")

python ppo.py \
    "${PPO_ARGS[@]}" \
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
