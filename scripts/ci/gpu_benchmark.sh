#!/bin/bash
# gpu_benchmark.sh — Multi-seed GPU benchmark for statistical comparison.
#
# Runs ppo.py across multiple seeds, collects per-seed results, optionally
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
#   TOTAL_TIMESTEPS     - Timesteps per seed (default: 100000)
#   PR_NUMBER           - PR number for tagging
#   COMMIT_SHA          - Git commit SHA for tagging
#   BRANCH_LABEL        - Label for this branch (default: "pr")
#   BASELINE_DIR        - Path to pre-fetched baseline results (skip baseline run)
#   RUNPOD_POD_ID       - RunPod pod ID (for self-terminate watchdog)
#   RUNPOD_API_KEY      - RunPod API key (for watchdog)

set -euo pipefail

NUM_SEEDS="${NUM_SEEDS:-5}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-100000}"
WANDB_PROJECT="${WANDB_PROJECT:-factorion}"
PR_NUMBER="${PR_NUMBER:-unknown}"
COMMIT_SHA="${COMMIT_SHA:-unknown}"
BRANCH_LABEL="${BRANCH_LABEL:-pr}"

WORK_DIR="/workspace/factorion"
RESULTS_DIR="/workspace/benchmark_results"
PR_RESULTS_DIR="${RESULTS_DIR}/${BRANCH_LABEL}"
BASELINE_RESULTS_DIR="${BASELINE_DIR:-${RESULTS_DIR}/baseline}"

echo "============================================"
echo "  Factorion GPU Benchmark"
echo "============================================"
echo "  GPU:             $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  CUDA:            $(nvcc --version 2>/dev/null | tail -1 || echo 'unknown')"
echo "  Seeds:           ${NUM_SEEDS}"
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

# ── Run PR branch seeds ───────────────────────────────────────────
mkdir -p "$PR_RESULTS_DIR"

echo ""
echo ">>> Running ${NUM_SEEDS} seeds for branch '${BRANCH_LABEL}'..."
echo ""

for seed in $(seq 1 "$NUM_SEEDS"); do
    echo "────────────────────────────────────────"
    echo "  Seed ${seed}/${NUM_SEEDS}"
    echo "────────────────────────────────────────"

    python ppo.py \
        --seed "$seed" \
        --env-id factorion/FactorioEnv-v0 \
        --track \
        --wandb-project-name "$WANDB_PROJECT" \
        --total-timesteps "$TOTAL_TIMESTEPS" \
        --tags ci benchmark "${BRANCH_LABEL}" "pr:${PR_NUMBER}" "sha:${COMMIT_SHA}" "seed:${seed}"

    # Collect the summary
    if [ -f summary.json ]; then
        cp summary.json "${PR_RESULTS_DIR}/seed_${seed}.json"
        echo "  Saved ${PR_RESULTS_DIR}/seed_${seed}.json"
    else
        echo "  WARNING: summary.json not found for seed ${seed}"
    fi
done

echo ""
echo ">>> All ${NUM_SEEDS} PR seeds completed."

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
        git checkout main -- . 2>/dev/null || git checkout master -- . 2>/dev/null || {
            echo ">>> WARNING: Could not checkout main/master. Using current code as baseline."
            cd "$WORK_DIR"
        }

        # Rebuild Rust extension for baseline
        if [ -f factorion_rs/Cargo.toml ]; then
            echo ">>> Rebuilding factorion_rs for baseline..."
            (cd factorion_rs && maturin build --release --out dist && pip install dist/*.whl)
        fi

        for seed in $(seq 1 "$NUM_SEEDS"); do
            echo "────────────────────────────────────────"
            echo "  Baseline Seed ${seed}/${NUM_SEEDS}"
            echo "────────────────────────────────────────"

            python ppo.py \
                --seed "$seed" \
                --env-id factorion/FactorioEnv-v0 \
                --track \
                --wandb-project-name "$WANDB_PROJECT" \
                --total-timesteps "$TOTAL_TIMESTEPS" \
                --tags ci benchmark baseline "branch:main" "seed:${seed}"

            if [ -f summary.json ]; then
                cp summary.json "${BASELINE_RESULTS_DIR}/seed_${seed}.json"
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
