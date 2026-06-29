#!/usr/bin/env bash
#
# quality_measure.sh — benchmark TIME-TO-QUALITY across seeds with hyperfine.
#
# Sweeps the seed with `hyperfine --parameter-list` (NOT plain --runs: with a
# fixed seed the trajectory is deterministic, so repeating the same command just
# remeasures wall-jitter. Different configs resample the trajectory via FP
# non-determinism, so the honest repeat is over SEEDS — that captures the
# trajectory variance and lets us compare config *means*.)
#
# Usage:
#   ./quality_measure.sh                              # seeds 1..5, baseline recipe
#   NOTE="lr 3e-4" ./quality_measure.sh --learning-rate 3e-4
#   NOTE="envs 32" ./quality_measure.sh --num-envs 32 --num-minibatches 64
#   SEEDS=1,2,3 ./quality_measure.sh                  # fewer seeds (faster, noisier)
#
# Any flags are forwarded to quality_run.sh -> ppo.py and override the recipe
# defaults. NOTE (env) annotates the results row.
#
# Reports, across the seeds: wall-clock mean±std (= ~constant startup +
# time-to-quality) AND the deterministic in-process time_to_quality (cleaner,
# startup-free) read from each seed's summary. ~115 s/seed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SEEDS="${SEEDS:-1,2,3,4,5}"
NOTE="${NOTE:-}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then DIRTY="dirty"; else DIRTY="clean"; fi

RESULTS_CSV="$(cd .. && pwd)/quality_results.csv"
HF_JSON="$(mktemp -t hyperfine.XXXXXX.json)"
SUMDIR="$(mktemp -d -t quality_sums.XXXXXX)"
trap 'rm -f "$HF_JSON"; rm -rf "$SUMDIR"' EXIT

EXTRA="$*"   # forwarded knob overrides (simple flags only)
echo "Benchmarking time-to-quality on '$BRANCH' ($COMMIT, $DIRTY): seeds=$SEEDS"
[ -n "$NOTE" ] && echo "note: $NOTE"
[ -n "$EXTRA" ] && echo "overrides: $EXTRA"

# One run per seed. Each writes its own summary so we can read the deterministic
# in-process time_to_quality per seed.
hyperfine \
  --warmup 0 \
  --runs 1 \
  --parameter-list seed "$SEEDS" \
  --command-name "seed{seed}" \
  --export-json "$HF_JSON" \
  "SUMMARY_PATH=$SUMDIR/s{seed}.json ./quality_run.sh --seed {seed} $EXTRA"

HF_JSON="$HF_JSON" SUMDIR="$SUMDIR" RESULTS_CSV="$RESULTS_CSV" \
  BRANCH="$BRANCH" COMMIT="$COMMIT" DIRTY="$DIRTY" NOTE="$NOTE" EXTRA="$EXTRA" \
  python3 - <<'PY'
import json, os, csv, glob, datetime, pathlib, statistics
# Per-seed in-process time-to-quality (deterministic per seed) + reached.
sums = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(os.environ["SUMDIR"], "s*.json")))]
ttq = [s["time_to_quality_seconds"] for s in sums if s.get("time_to_quality_seconds") is not None]
iters = [s["num_iterations"] for s in sums]
reached = sum(1 for s in sums if s.get("reached_quality"))
n = len(sums)
# Hyperfine wall per seed.
hf = json.load(open(os.environ["HF_JSON"]))["results"]
walls = [r["mean"] for r in hf]   # --runs 1 => mean == the single time
def ms(xs): return (round(statistics.mean(xs),2), round(statistics.pstdev(xs),2)) if xs else (None,None)
ttq_m, ttq_s = ms(ttq)
wall_m, wall_s = ms(walls)
it_m, it_s = ms([float(i) for i in iters])
row = {
    "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "branch": os.environ["BRANCH"], "commit": os.environ["COMMIT"], "dirty": os.environ["DIRTY"],
    "seeds": n, "reached": f"{reached}/{n}",
    "ttq_mean_s": ttq_m, "ttq_std_s": ttq_s, "ttq_min_s": round(min(ttq),2) if ttq else None, "ttq_max_s": round(max(ttq),2) if ttq else None,
    "crossing_iter_mean": it_m, "crossing_iter_std": it_s,
    "wall_mean_s": wall_m, "wall_std_s": wall_s,
    "overrides": os.environ.get("EXTRA",""), "note": os.environ.get("NOTE",""),
}
p = pathlib.Path(os.environ["RESULTS_CSV"]); new = not p.exists()
with open(p, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row));  w.writeheader() if new else None;  w.writerow(row)
print(f"\ntime-to-quality: {ttq_m}s ± {ttq_s}  (min {row['ttq_min_s']}, max {row['ttq_max_s']}, {reached}/{n} reached)")
print(f"crossing iter:   {it_m} ± {it_s}   |   wall {wall_m}s ± {wall_s}")
print(f"-> {p}")
PY
