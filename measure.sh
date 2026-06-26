#!/usr/bin/env bash
#
# measure.sh — benchmark ./run.sh with hyperfine and log the result.
#
# Runs the fixed PPO benchmark (./run.sh) under hyperfine, then appends one row
# to ../results.csv (branch, mean, min, max, stddev, median, rollout/update
# breakdown, ...). This is the command you run on each experiment branch.
#
# Usage:
#   ./measure.sh                         # 5 runs + 1 warmup, log to ../results.csv
#   ./measure.sh "moved X to GPU"        # attach a free-text note to the row
#   RUNS=10 WARMUP=2 ./measure.sh        # override hyperfine sampling
#
# The all-in mean (and its stddev) is the headline number. ../results.csv is the
# machine-readable log; ../EXPERIMENTS.md is the human narrative.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

RUNS="${RUNS:-5}"
WARMUP="${WARMUP:-1}"
NOTE="${1:-${NOTE:-}}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then DIRTY="dirty"; else DIRTY="clean"; fi

# ../results.csv lives outside the repo so it survives branch switches.
RESULTS_CSV="$(cd .. && pwd)/results.csv"
HF_JSON="$(mktemp -t hyperfine.XXXXXX.json)"
SUMMARY_JSON="$(mktemp -t ppo_summary.XXXXXX.json)"
trap 'rm -f "$HF_JSON" "$SUMMARY_JSON"' EXIT

echo "Benchmarking branch '$BRANCH' ($COMMIT, $DIRTY): $WARMUP warmup + $RUNS runs"
[ -n "$NOTE" ] && echo "note: $NOTE"

# run.sh writes its rollout/update breakdown to $SUMMARY_PATH; export it so the
# child process picks it up (the last run's summary is the one we log).
export SUMMARY_PATH="$SUMMARY_JSON"

hyperfine \
  --warmup "$WARMUP" \
  --runs "$RUNS" \
  --command-name "$BRANCH" \
  --export-json "$HF_JSON" \
  ./run.sh

HF_JSON="$HF_JSON" SUMMARY_JSON="$SUMMARY_JSON" RESULTS_CSV="$RESULTS_CSV" \
  BRANCH="$BRANCH" COMMIT="$COMMIT" DIRTY="$DIRTY" NOTE="$NOTE" \
  python3 scripts/_log_result.py
