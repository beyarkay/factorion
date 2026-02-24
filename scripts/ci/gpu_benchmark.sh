#!/bin/bash
# gpu_benchmark.sh — Multi-seed GPU benchmark for statistical comparison.
#
# Runs ppo.py across multiple seeds, collects per-seed results, optionally
# fetches (or runs) a baseline from the main branch, then invokes
# compare_runs.py to produce a statistical comparison report.
#
# Seeds run in parallel (up to MAX_PARALLEL at a time) on a single GPU to
# reduce wall-clock time.  Each seed writes its summary to a unique file
# via --summary-path to avoid conflicts.
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
#   TOTAL_TIMESTEPS     - Timesteps per seed (default: 100000)
#   PR_NUMBER           - PR number for tagging
#   COMMIT_SHA          - Git commit SHA for tagging
#   BRANCH_LABEL        - Label for this branch (default: "pr")
#   BASELINE_DIR        - Path to pre-fetched baseline results (skip baseline run)
#   RUNPOD_POD_ID       - RunPod pod ID (for self-terminate watchdog)
#   RUNPOD_API_KEY      - RunPod API key (for watchdog)

set -euo pipefail

NUM_SEEDS="${NUM_SEEDS:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
WANDB_PROJECT="${WANDB_PROJECT:-factorion}"
PR_NUMBER="${PR_NUMBER:-unknown}"
COMMIT_SHA="${COMMIT_SHA:-unknown}"
BRANCH_LABEL="${BRANCH_LABEL:-pr}"
SHORT_SHA="${COMMIT_SHA:0:7}"

WORK_DIR="/workspace/factorion"
RESULTS_DIR="/workspace/benchmark_results"
PR_RESULTS_DIR="${RESULTS_DIR}/${BRANCH_LABEL}"
BASELINE_RESULTS_DIR="${BASELINE_DIR:-${RESULTS_DIR}/baseline}"
LOG_DIR="/workspace/benchmark_logs"

echo "============================================"
echo "  Factorion GPU Benchmark"
echo "============================================"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CUDA:            $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "  Seeds:           ${NUM_SEEDS}"
echo "  Max parallel:    ${MAX_PARALLEL}"
echo "  Timesteps/seed:  ${TOTAL_TIMESTEPS}"
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

# ── CuBLAS deterministic mode ─────────────────────────────────────
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ── Build Rust extension ──────────────────────────────────────────
if [ -f factorion_rs/Cargo.toml ]; then
    echo ""
    echo ">>> Building factorion_rs from source..."
    (cd factorion_rs && maturin build --release --out dist && pip install dist/*.whl)
else
    echo ">>> WARNING: factorion_rs not available."
fi

# ── Log in to W&B ─────────────────────────────────────────────────
export WANDB_API_KEY

# ── Run PR branch seeds (parallel, up to MAX_PARALLEL at a time) ──
mkdir -p "$PR_RESULTS_DIR" "$LOG_DIR"

PR_GROUP="bench-${BRANCH_LABEL}-${SHORT_SHA}"
echo ""
echo ">>> Running ${NUM_SEEDS} seeds for branch '${BRANCH_LABEL}' (up to ${MAX_PARALLEL} in parallel)..."
echo ">>> W&B group: ${PR_GROUP}"
echo ""

# Launch seeds in parallel, throttled to MAX_PARALLEL concurrent jobs.
# Each seed's output is prefixed with [seed N] and tee'd to a log file
# so progress is visible in the CI console.
PIDS=()
SEED_FOR_PID=()
for seed in $(seq 1 "$NUM_SEEDS"); do
    # If we already have MAX_PARALLEL running, wait for any one to finish
    while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
        # Wait for the earliest PID
        if wait "${PIDS[0]}" 2>/dev/null; then
            echo "[seed ${SEED_FOR_PID[0]}] finished (exit 0)"
        else
            echo "[seed ${SEED_FOR_PID[0]}] WARNING: exited with non-zero status"
        fi
        PIDS=("${PIDS[@]:1}")
        SEED_FOR_PID=("${SEED_FOR_PID[@]:1}")
    done

    echo ">>> Launching seed ${seed}/${NUM_SEEDS}..."
    (
        python ppo.py \
            --seed "$seed" \
            --env-id factorion/FactorioEnv-v0 \
            --track \
            --wandb-project-name "$WANDB_PROJECT" \
            --wandb-group "$PR_GROUP" \
            --total-timesteps "$TOTAL_TIMESTEPS" \
            --summary-path "${PR_RESULTS_DIR}/seed_${seed}.json" \
            --tags ci benchmark "${BRANCH_LABEL}" "pr:${PR_NUMBER}" "sha:${COMMIT_SHA}" "seed:${seed}" \
            2>&1 | sed "s/^/[seed ${seed}] /" | tee "${LOG_DIR}/pr_seed_${seed}.log"
    ) &
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

# Print logs for any seed that didn't produce a result
for seed in $(seq 1 "$NUM_SEEDS"); do
    if [ ! -f "${PR_RESULTS_DIR}/seed_${seed}.json" ]; then
        echo "[seed ${seed}] WARNING: summary not found. Log tail:"
        tail -20 "${LOG_DIR}/pr_seed_${seed}.log" 2>/dev/null || true
    fi
done

echo ""
echo ">>> All ${NUM_SEEDS} PR seeds completed (${FAILED} failed)."

# ── Combine per-seed results into one JSON array ──────────────────
python3 -c "
import json, glob, sys

results = []
for path in sorted(glob.glob('${PR_RESULTS_DIR}/seed_*.json')):
    with open(path) as f:
        data = json.load(f)
    # Extract seed number from filename
    seed_num = path.split('seed_')[-1].replace('.json', '')
    data['seed_file'] = seed_num
    results.append(data)

with open('${PR_RESULTS_DIR}/all_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f'Combined {len(results)} seed results into all_results.json')
"

# ── Run baseline seeds if no pre-fetched baseline ─────────────────
if [ -n "${BASELINE_DIR:-}" ] && [ -f "${BASELINE_DIR}/all_results.json" ]; then
    echo ""
    echo ">>> Using pre-fetched baseline from ${BASELINE_DIR}"
else
    # Check if we have a baseline committed in the repo
    if [ -f scripts/ci/baseline_results.json ]; then
        echo ""
        echo ">>> Using committed baseline from scripts/ci/baseline_results.json"
        mkdir -p "$BASELINE_RESULTS_DIR"
        cp scripts/ci/baseline_results.json "${BASELINE_RESULTS_DIR}/all_results.json"
    else
        echo ""
        echo ">>> No baseline found. Running ${NUM_SEEDS} baseline seeds on current code..."
        echo ">>> (Tip: commit a scripts/ci/baseline_results.json to skip this step)"

        mkdir -p "$BASELINE_RESULTS_DIR"

        # Check out main, build, and run baseline seeds
        # We stash current code and use main for baseline
        MAIN_WORK_DIR="/workspace/factorion_main"
        if [ -d "$MAIN_WORK_DIR" ]; then
            rm -rf "$MAIN_WORK_DIR"
        fi
        cp -r "$WORK_DIR" "$MAIN_WORK_DIR"
        cd "$MAIN_WORK_DIR"

        if ! git checkout main -- . 2>/dev/null && ! git checkout master -- . 2>/dev/null; then
            echo ">>> ERROR: Could not checkout main/master for baseline comparison."
            echo ">>> Ensure .git is included in the tarball transfer (fetch-depth: 0 required)."
            echo ">>> Refusing to compare PR against itself."
            exit 1
        fi
        echo ">>> Successfully checked out main branch for baseline"

        # Rebuild Rust extension for baseline
        if [ -f factorion_rs/Cargo.toml ]; then
            echo ">>> Rebuilding factorion_rs for baseline..."
            (cd factorion_rs && maturin build --release --out dist && pip install dist/*.whl)
        fi

        BASELINE_LOG_DIR="${LOG_DIR}/baseline"
        mkdir -p "$BASELINE_LOG_DIR"
        BASELINE_GROUP="bench-baseline-${SHORT_SHA}"
        echo ">>> W&B group: ${BASELINE_GROUP}"

        PIDS=()
        SEED_FOR_PID=()
        for seed in $(seq 1 "$NUM_SEEDS"); do
            # Throttle to MAX_PARALLEL concurrent jobs
            while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
                if wait "${PIDS[0]}" 2>/dev/null; then
                    echo "[baseline ${SEED_FOR_PID[0]}] finished (exit 0)"
                else
                    echo "[baseline ${SEED_FOR_PID[0]}] WARNING: exited with non-zero status"
                fi
                PIDS=("${PIDS[@]:1}")
                SEED_FOR_PID=("${SEED_FOR_PID[@]:1}")
            done

            echo ">>> Launching baseline seed ${seed}/${NUM_SEEDS}..."
            (
                python ppo.py \
                    --seed "$seed" \
                    --env-id factorion/FactorioEnv-v0 \
                    --track \
                    --wandb-project-name "$WANDB_PROJECT" \
                    --wandb-group "$BASELINE_GROUP" \
                    --total-timesteps "$TOTAL_TIMESTEPS" \
                    --summary-path "${BASELINE_RESULTS_DIR}/seed_${seed}.json" \
                    --tags ci benchmark baseline "branch:main" "seed:${seed}" \
                    2>&1 | sed "s/^/[baseline ${seed}] /" | tee "${BASELINE_LOG_DIR}/seed_${seed}.log"
            ) &
            PIDS+=($!)
            SEED_FOR_PID+=("$seed")
        done

        # Wait for all remaining baseline jobs
        for i in "${!PIDS[@]}"; do
            if wait "${PIDS[$i]}" 2>/dev/null; then
                echo "[baseline ${SEED_FOR_PID[$i]}] finished (exit 0)"
            else
                echo "[baseline ${SEED_FOR_PID[$i]}] WARNING: exited with non-zero status"
            fi
        done

        for seed in $(seq 1 "$NUM_SEEDS"); do
            if [ ! -f "${BASELINE_RESULTS_DIR}/seed_${seed}.json" ]; then
                echo "[baseline ${seed}] WARNING: summary not found. Log tail:"
                tail -20 "${BASELINE_LOG_DIR}/seed_${seed}.log" 2>/dev/null || true
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
            (cd factorion_rs && maturin build --release --out dist && pip install dist/*.whl)
        fi
    fi
fi

# ── Statistical comparison ─────────────────────────────────────────
echo ""
echo ">>> Running statistical comparison..."

python3 scripts/ci/compare_runs.py \
    --pr-results "${PR_RESULTS_DIR}/all_results.json" \
    --baseline-results "${BASELINE_RESULTS_DIR}/all_results.json" \
    --pr-label "${BRANCH_LABEL}" \
    --pr-number "${PR_NUMBER}" \
    --commit-sha "${COMMIT_SHA}" \
    --output /workspace/summary.md

echo ""
echo "============================================"
echo "  Benchmark completed successfully"
echo "============================================"
echo "  Report: /workspace/summary.md"
echo "============================================"
