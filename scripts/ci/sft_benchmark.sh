#!/bin/bash
# sft_benchmark.sh — Multi-seed SFT benchmark for statistical comparison.
#
# Runs sft.py across multiple seeds, collects per-seed results, optionally
# fetches (or runs) a baseline from the main branch, then invokes
# compare_runs.py to produce a statistical comparison report.
#
# Pre-installed deps and Rust toolchain are expected in the Docker image
# (beyarkay/factorion-ci-gpu:latest). Code should already be at
# /workspace/factorion/.
#
# Required env vars:
#   WANDB_API_KEY       - W&B API key for logging
#   WANDB_PROJECT       - W&B project name (e.g. factorion)
#
# Optional env vars:
#   NUM_SEEDS           - Number of seeds to run (default: 5)
#   MAX_PARALLEL        - Max seeds to run concurrently (default: 5)
#   NUM_SAMPLES         - SFT samples per seed (default: 50000)
#   EPOCHS              - SFT epochs per seed (default: 30)
#   PR_NUMBER           - PR number for tagging
#   COMMIT_SHA          - Git commit SHA for tagging
#   BRANCH_LABEL        - Label for this branch (default: "pr")
#   RUNPOD_POD_ID       - RunPod pod ID (for self-terminate watchdog)
#   RUNPOD_API_KEY      - RunPod API key (for watchdog)

set -euo pipefail

NUM_SEEDS="${NUM_SEEDS:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-50000}"
EPOCHS="${EPOCHS:-30}"
WANDB_PROJECT="${WANDB_PROJECT:-factorion}"
PR_NUMBER="${PR_NUMBER:-unknown}"
COMMIT_SHA="${COMMIT_SHA:-unknown}"
BRANCH_LABEL="${BRANCH_LABEL:-pr}"
SHORT_SHA="${COMMIT_SHA:0:7}"

WORK_DIR="/workspace/factorion"
RESULTS_DIR="/workspace/sft_benchmark_results"
PR_RESULTS_DIR="${RESULTS_DIR}/${BRANCH_LABEL}"
# BASELINE_DIR may be set externally to point to pre-fetched baseline results.
# If unset, we'll try to run baseline seeds from main (requires .git in transfer).
BASELINE_RESULTS_DIR="${BASELINE_DIR:-${RESULTS_DIR}/baseline}"
LOG_DIR="/workspace/sft_benchmark_logs"

echo "============================================"
echo "  Factorion SFT Benchmark"
echo "============================================"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CUDA:            $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "  Seeds:           ${NUM_SEEDS}"
echo "  Max parallel:    ${MAX_PARALLEL}"
echo "  Samples/seed:    ${NUM_SAMPLES}"
echo "  Epochs/seed:     ${EPOCHS}"
echo "  W&B project:     ${WANDB_PROJECT}"
echo "  Branch label:    ${BRANCH_LABEL}"
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

cd "$WORK_DIR"

# ── Ensure Rust is on PATH ────────────────────────────────────────
export PATH="/root/.cargo/bin:${PATH}"

# ── CuBLAS deterministic mode ────────────────────────────────────
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ── Build Rust extension ─────────────────────────────────────────
if [ -f factorion_rs/Cargo.toml ]; then
    echo ""
    echo ">>> Building factorion_rs from source..."
    (cd factorion_rs && maturin build --release --out dist && pip install --force-reinstall dist/*.whl)
else
    echo ">>> WARNING: factorion_rs not available."
fi

# ── Log in to W&B ─────────────────────────────────────────────────
export WANDB_API_KEY

# ── Run PR branch seeds (parallel) ───────────────────────────────
mkdir -p "$PR_RESULTS_DIR" "$LOG_DIR"
PR_LOG_DIR="${LOG_DIR}/pr"
mkdir -p "$PR_LOG_DIR"

PR_GROUP="sft-bench-${BRANCH_LABEL}-${SHORT_SHA}"
echo ""
echo ">>> Running ${NUM_SEEDS} SFT seeds for branch '${BRANCH_LABEL}' (up to ${MAX_PARALLEL} in parallel)..."
echo ">>> W&B group: ${PR_GROUP}"

PIDS=()
SEED_FOR_PID=()
for seed in $(seq 1 "$NUM_SEEDS"); do
    while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
        if wait "${PIDS[0]}" 2>/dev/null; then
            echo "[seed ${SEED_FOR_PID[0]}] finished (exit 0)"
        else
            echo "[seed ${SEED_FOR_PID[0]}] WARNING: exited with non-zero status"
        fi
        PIDS=("${PIDS[@]:1}")
        SEED_FOR_PID=("${SEED_FOR_PID[@]:1}")
    done

    echo ">>> Launching SFT seed ${seed}/${NUM_SEEDS}..."
    python sft.py \
        --seed "$seed" \
        --size 8 \
        --num-samples "$NUM_SAMPLES" \
        --epochs "$EPOCHS" \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --wandb-group "$PR_GROUP" \
        --summary-path "${PR_RESULTS_DIR}/seed_${seed}.json" \
        --tags ci sft-benchmark "${BRANCH_LABEL}" "pr:${PR_NUMBER}" "sha:${COMMIT_SHA}" "seed:${seed}" \
        > "${PR_LOG_DIR}/seed_${seed}.log" 2>&1 &
    PIDS+=($!)
    SEED_FOR_PID+=("$seed")
done

# Wait for all remaining background jobs
FAILED=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}" 2>/dev/null; then
        echo "[seed ${SEED_FOR_PID[$i]}] finished (exit 0)"
    else
        echo "[seed ${SEED_FOR_PID[$i]}] WARNING: exited with non-zero status"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo ">>> All ${NUM_SEEDS} PR seeds completed (${FAILED} failed)."

# ── Combine per-seed results ─────────────────────────────────────
python3 -c "
import json, glob
results = []
for path in sorted(glob.glob('${PR_RESULTS_DIR}/seed_*.json')):
    with open(path) as f:
        data = json.load(f)
    seed_num = path.split('seed_')[-1].replace('.json', '')
    data['seed_file'] = seed_num
    results.append(data)
with open('${PR_RESULTS_DIR}/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Combined {len(results)} seed results into all_results.json')
"

# ── Run baseline if needed ────────────────────────────────────────
if [ -n "${BASELINE_DIR:-}" ] && [ -f "${BASELINE_DIR}/all_results.json" ]; then
    echo ""
    echo ">>> Using pre-fetched baseline from ${BASELINE_DIR}"
elif [ -f scripts/ci/sft_baseline_results.json ]; then
    echo ""
    echo ">>> Using committed baseline from scripts/ci/sft_baseline_results.json"
    mkdir -p "$BASELINE_RESULTS_DIR"
    cp scripts/ci/sft_baseline_results.json "${BASELINE_RESULTS_DIR}/all_results.json"
else
    echo ""
    echo ">>> No SFT baseline found. Running ${NUM_SEEDS} baseline seeds on main..."

    mkdir -p "$BASELINE_RESULTS_DIR"

    MAIN_WORK_DIR="/workspace/factorion_main"
    if [ -d "$MAIN_WORK_DIR" ]; then
        rm -rf "$MAIN_WORK_DIR"
    fi
    cp -r "$WORK_DIR" "$MAIN_WORK_DIR"
    cd "$MAIN_WORK_DIR"

    echo ">>> Trying to checkout main branch files for baseline..."
    if git checkout main -- . 2>/dev/null; then
        echo ">>> Checked out from local 'main'"
    elif git checkout origin/main -- . 2>/dev/null; then
        echo ">>> Checked out from 'origin/main'"
    else
        echo ">>> WARNING: Could not checkout main for baseline. Skipping baseline comparison."
        cd "$WORK_DIR"
        # Generate summary without baseline comparison
        python3 -c "
import json
pr_results = json.load(open('${PR_RESULTS_DIR}/all_results.json'))
accs = [r['best_val_acc'] for r in pr_results]
mean_acc = sum(accs) / len(accs)
md = f'''## SFT Benchmark Results

**Mean val accuracy:** {mean_acc:.4f} across {len(accs)} seeds

No baseline available for comparison (main branch does not have sft.py).

<sub>Commit ${COMMIT_SHA[:8]} \u00b7 PR #${PR_NUMBER}</sub>
'''
with open('/workspace/summary.md', 'w') as f:
    f.write(md)
"
        echo "============================================"
        echo "  SFT Benchmark completed (no baseline)"
        echo "============================================"
        exit 0
    fi

    # Rebuild Rust extension for baseline
    if [ -f factorion_rs/Cargo.toml ]; then
        echo ">>> Rebuilding factorion_rs for baseline..."
        (cd factorion_rs && maturin build --release --out dist && pip install --force-reinstall dist/*.whl)
    fi

    BASELINE_LOG_DIR="${LOG_DIR}/baseline"
    mkdir -p "$BASELINE_LOG_DIR"

    # Check if main has sft.py
    if [ ! -f sft.py ]; then
        echo ">>> main branch does not have sft.py, skipping baseline."
        cd "$WORK_DIR"
        if [ -f factorion_rs/Cargo.toml ]; then
            (cd factorion_rs && maturin build --release --out dist && pip install --force-reinstall dist/*.whl)
        fi

        python3 -c "
import json
pr_results = json.load(open('${PR_RESULTS_DIR}/all_results.json'))
accs = [r['best_val_acc'] for r in pr_results]
mean_acc = sum(accs) / len(accs)
md = f'''## SFT Benchmark Results

**Mean val accuracy:** {mean_acc:.4f} across {len(accs)} seeds

No baseline available (main branch does not have sft.py).

<sub>Commit ${COMMIT_SHA[:8]} \u00b7 PR #${PR_NUMBER}</sub>
'''
with open('/workspace/summary.md', 'w') as f:
    f.write(md)
"
        echo "============================================"
        echo "  SFT Benchmark completed (no baseline)"
        echo "============================================"
        exit 0
    fi

    BASELINE_GROUP="sft-bench-baseline-${SHORT_SHA}"
    echo ">>> Running ${NUM_SEEDS} baseline SFT seeds in parallel..."

    PIDS=()
    SEED_FOR_PID=()
    for seed in $(seq 1 "$NUM_SEEDS"); do
        while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
            if wait "${PIDS[0]}" 2>/dev/null; then
                echo "[baseline ${SEED_FOR_PID[0]}] finished (exit 0)"
            else
                echo "[baseline ${SEED_FOR_PID[0]}] WARNING: exited with non-zero status"
            fi
            PIDS=("${PIDS[@]:1}")
            SEED_FOR_PID=("${SEED_FOR_PID[@]:1}")
        done

        echo ">>> Launching baseline SFT seed ${seed}/${NUM_SEEDS}..."
        python sft.py \
            --seed "$seed" \
            --size 8 \
            --num-samples "$NUM_SAMPLES" \
            --epochs "$EPOCHS" \
            --track \
            --wandb-project-name "$WANDB_PROJECT" \
            --wandb-group "$BASELINE_GROUP" \
            --summary-path "${BASELINE_RESULTS_DIR}/seed_${seed}.json" \
            --tags ci sft-benchmark baseline "branch:main" "seed:${seed}" \
            > "${BASELINE_LOG_DIR}/seed_${seed}.log" 2>&1 &
        PIDS+=($!)
        SEED_FOR_PID+=("$seed")
    done

    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}" 2>/dev/null; then
            echo "[baseline ${SEED_FOR_PID[$i]}] finished (exit 0)"
        else
            echo "[baseline ${SEED_FOR_PID[$i]}] WARNING: exited with non-zero status"
        fi
    done

    # Combine baseline results
    python3 -c "
import json, glob
results = []
for path in sorted(glob.glob('${BASELINE_RESULTS_DIR}/seed_*.json')):
    with open(path) as f:
        data = json.load(f)
    seed_num = path.split('seed_')[-1].replace('.json', '')
    data['seed_file'] = seed_num
    results.append(data)
with open('${BASELINE_RESULTS_DIR}/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'Combined {len(results)} baseline results')
"

    # Return to PR code dir and rebuild
    cd "$WORK_DIR"
    if [ -f factorion_rs/Cargo.toml ]; then
        (cd factorion_rs && maturin build --release --out dist && pip install --force-reinstall dist/*.whl)
    fi
fi

# ── Statistical comparison ────────────────────────────────────────
echo ""
echo ">>> Running statistical comparison..."

# Use compare_runs.py with the SFT metric key
if ! python3 scripts/ci/compare_runs.py \
    --pr-results "${PR_RESULTS_DIR}/all_results.json" \
    --baseline-results "${BASELINE_RESULTS_DIR}/all_results.json" \
    --pr-label "${BRANCH_LABEL}" \
    --pr-number "${PR_NUMBER}" \
    --commit-sha "${COMMIT_SHA}" \
    --metric-key "best_val_acc" \
    --output /workspace/summary.md; then
        # If compare_runs.py doesn't support --metric-key, fall back to manual comparison
        echo ">>> compare_runs.py failed, generating manual comparison..."
        python3 -c "
import json, statistics

pr = json.load(open('${PR_RESULTS_DIR}/all_results.json'))
bl = json.load(open('${BASELINE_RESULTS_DIR}/all_results.json'))

pr_accs = [r['best_val_acc'] for r in pr]
bl_accs = [r['best_val_acc'] for r in bl]

pr_mean = statistics.mean(pr_accs)
bl_mean = statistics.mean(bl_accs)
pr_std = statistics.stdev(pr_accs) if len(pr_accs) > 1 else 0
bl_std = statistics.stdev(bl_accs) if len(bl_accs) > 1 else 0

diff = pr_mean - bl_mean
better = 'better' if diff > 0 else 'worse' if diff < 0 else 'same'

md = f'''## SFT Benchmark Comparison

| | PR | Baseline | Delta |
|---|---|---|---|
| **Val accuracy** | {pr_mean:.4f} \u00b1 {pr_std:.4f} | {bl_mean:.4f} \u00b1 {bl_std:.4f} | {diff:+.4f} ({better}) |
| **Seeds** | {len(pr_accs)} | {len(bl_accs)} | |

<sub>Commit ${COMMIT_SHA[:8]} \u00b7 PR #${PR_NUMBER}</sub>
'''
with open('/workspace/summary.md', 'w') as f:
    f.write(md)
print(md)
"
fi

echo ""
echo "============================================"
echo "  SFT Benchmark completed successfully"
echo "============================================"
echo "  Report: /workspace/summary.md"
echo "============================================"
