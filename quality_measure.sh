#!/usr/bin/env bash
#
# quality_measure.sh — benchmark quality_run.sh (TIME-TO-QUALITY) with hyperfine.
#
# Runs the time-to-quality benchmark (./quality_run.sh) under hyperfine and
# reports the wall-clock time to reach the quality threshold, plus the
# deterministic in-process time-to-quality / crossing-iteration from the run
# summary. Use this for changes that alter the computation (env/batch/LR/AMP):
# "faster" = reaches the same reward quality in less wall-clock time.
#
# Usage:
#   ./quality_measure.sh "bf16 AMP"             # note attached to the row
#   RUNS=3 ./quality_measure.sh                 # fewer runs (default 5, no warmup)
#   LR=3e-4 NUM_ENVS=32 ./quality_measure.sh "lr3e-4 envs32"   # sweep a knob
#
# No --warmup: each run trains from the SFT checkpoint to the quality threshold,
# so a warmup run would just burn ~2 min. The run is deterministic (fixed seed),
# so the crossing iteration is identical across runs and only wall-time jitters
# (~0.1 s in practice) — 5 runs is plenty.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

RUNS="${RUNS:-5}"
NOTE="${1:-${NOTE:-}}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then DIRTY="dirty"; else DIRTY="clean"; fi

RESULTS_CSV="$(cd .. && pwd)/quality_results.csv"
HF_JSON="$(mktemp -t hyperfine.XXXXXX.json)"
SUMMARY_JSON="$(mktemp -t quality_summary.XXXXXX.json)"
trap 'rm -f "$HF_JSON" "$SUMMARY_JSON"' EXIT
export SUMMARY_PATH="$SUMMARY_JSON"

echo "Benchmarking time-to-quality on '$BRANCH' ($COMMIT, $DIRTY): $RUNS runs"
[ -n "$NOTE" ] && echo "note: $NOTE"

hyperfine \
  --warmup 0 \
  --runs "$RUNS" \
  --command-name "$BRANCH" \
  --export-json "$HF_JSON" \
  ./quality_run.sh

# The last run's summary carries the deterministic in-process metrics.
HF_JSON="$HF_JSON" SUMMARY_JSON="$SUMMARY_JSON" RESULTS_CSV="$RESULTS_CSV" \
  BRANCH="$BRANCH" COMMIT="$COMMIT" DIRTY="$DIRTY" NOTE="$NOTE" \
  python3 - <<'PY'
import json, os, csv, datetime, pathlib
hf = json.load(open(os.environ["HF_JSON"]))["results"][0]
s = json.load(open(os.environ["SUMMARY_JSON"]))
row = {
    "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "branch": os.environ["BRANCH"],
    "commit": os.environ["COMMIT"],
    "dirty": os.environ["DIRTY"],
    "wall_mean_s": round(hf["mean"], 3),
    "wall_stddev_s": round(hf.get("stddev") or 0.0, 3),
    "wall_min_s": round(hf["min"], 3),
    "wall_max_s": round(hf["max"], 3),
    "runs": len(hf["times"]),
    "reached_quality": s.get("reached_quality"),
    "time_to_quality_s": s.get("time_to_quality_seconds"),
    "crossing_iter": s.get("num_iterations"),
    "quality_ema_final": s.get("quality_ema_final"),
    "target_metric": s.get("target_metric"),
    "target_value": s.get("target_value"),
    "num_envs": s.get("num_envs"),
    "note": os.environ.get("NOTE", ""),
}
csv_path = pathlib.Path(os.environ["RESULTS_CSV"])
new = not csv_path.exists()
with open(csv_path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new:
        w.writeheader()
    w.writerow(row)
status = "REACHED" if row["reached_quality"] else "DID NOT REACH (capped)"
print(f"\nwall {row['wall_mean_s']}s ± {row['wall_stddev_s']}  |  "
      f"in-process time-to-quality {row['time_to_quality_s']}s @ iter {row['crossing_iter']}  |  {status}")
print(f"-> {csv_path}")
PY
