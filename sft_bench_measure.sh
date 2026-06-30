#!/usr/bin/env bash
#
# sft_bench_measure.sh — benchmark ./sft_bench_run.sh with hyperfine + log a row.
#
# Times the fixed SFT training benchmark under hyperfine and appends one row to
# ../sft_bench_results.csv (branch, commit, wall mean/std/min/max, val_loss).
# This is a PURE-SPEED benchmark: it also asserts val_loss is unchanged from the
# baseline (the invariance signature) — a change that moves val_loss altered the
# computation (or skipped a step) and is NOT a valid speed-only win.
#
# Usage:
#   ./sft_bench_measure.sh "killed per-batch .item() syncs"   # note on the row
#   RUNS=5 WARMUP=1 ./sft_bench_measure.sh
#   BASELINE_VAL_LOSS=1.6888 ./sft_bench_measure.sh           # invariance target
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RUNS="${RUNS:-3}"
WARMUP="${WARMUP:-1}"
NOTE="${1:-${NOTE:-}}"
BASELINE_VAL_LOSS="${BASELINE_VAL_LOSS:-1.6888}"

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then DIRTY="dirty"; else DIRTY="clean"; fi

RESULTS_CSV="$(cd .. && pwd)/sft_bench_results.csv"
HF_JSON="$(mktemp -t sft_hf.XXXXXX.json)"
SUMMARY_JSON="$(mktemp -t sft_sum.XXXXXX.json)"
trap 'rm -f "$HF_JSON" "$SUMMARY_JSON"' EXIT

echo "Benchmarking SFT training on '$BRANCH' ($COMMIT, $DIRTY): $WARMUP warmup + $RUNS runs"
[ -n "$NOTE" ] && echo "note: $NOTE"

export SUMMARY_PATH="$SUMMARY_JSON"
hyperfine \
  --warmup "$WARMUP" \
  --runs "$RUNS" \
  --command-name "$BRANCH" \
  --export-json "$HF_JSON" \
  ./sft_bench_run.sh

HF_JSON="$HF_JSON" SUMMARY_JSON="$SUMMARY_JSON" RESULTS_CSV="$RESULTS_CSV" \
  BRANCH="$BRANCH" COMMIT="$COMMIT" DIRTY="$DIRTY" NOTE="$NOTE" \
  BASELINE_VAL_LOSS="$BASELINE_VAL_LOSS" python3 - <<'PY'
import json, os, csv, datetime, pathlib
hf = json.load(open(os.environ["HF_JSON"]))["results"][0]
summ = json.load(open(os.environ["SUMMARY_JSON"]))
val_loss = summ["val_loss"]
baseline = float(os.environ["BASELINE_VAL_LOSS"])
ok = abs(val_loss - baseline) < 1e-4
row = {
    "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "branch": os.environ["BRANCH"], "commit": os.environ["COMMIT"], "dirty": os.environ["DIRTY"],
    "wall_mean_s": round(hf["mean"], 2), "wall_std_s": round(hf["stddev"], 2),
    "wall_min_s": round(hf["min"], 2), "wall_max_s": round(hf["max"], 2),
    "val_loss": val_loss, "invariant": "OK" if ok else f"VIOLATED(base {baseline})",
    "note": os.environ.get("NOTE", ""),
}
p = pathlib.Path(os.environ["RESULTS_CSV"]); new = not p.exists()
with open(p, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new: w.writeheader()
    w.writerow(row)
print(f"\nwall: {row['wall_mean_s']}s ± {row['wall_std_s']}  (min {row['wall_min_s']}, max {row['wall_max_s']})")
print(f"val_loss: {val_loss}  [{row['invariant']}]")
if not ok:
    print("!! INVARIANCE VIOLATED: val_loss changed — this is NOT a pure-speed change.")
print(f"-> {p}")
PY
