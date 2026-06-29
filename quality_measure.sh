#!/usr/bin/env bash
#
# quality_measure.sh — benchmark TIME-TO-QUALITY across seeds.
#
# Sweeps the seed (NOT identical reruns: a fixed seed is deterministic, so the
# crossing iteration repeats exactly and reruns only remeasure wall-jitter.
# Changing batch size / precision resamples the trajectory via FP
# non-determinism, so the honest repeat is over SEEDS — that captures the
# trajectory variance and lets us compare config means). One run per seed
# suffices because each seed is deterministic; we report the across-seed
# mean±std of the in-process time-to-quality. (hyperfine isn't used: it needs
# >=2 runs/command, which for a deterministic per-seed run is pure waste.)
#
# Usage:
#   ./quality_measure.sh                              # seeds 1..5, baseline recipe
#   NOTE="lr 3e-4" ./quality_measure.sh --learning-rate 3e-4
#   NOTE="envs 32" ./quality_measure.sh --num-envs 32 --num-minibatches 64
#   SEEDS="1 2 3" ./quality_measure.sh                # fewer seeds (faster, noisier)
#
# Any flags are forwarded to quality_run.sh -> ppo.py and override the recipe
# defaults. NOTE (env) annotates the results row. ~115 s/seed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

SEEDS="${SEEDS:-1 2 3 4 5}"
NOTE="${NOTE:-}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then DIRTY="dirty"; else DIRTY="clean"; fi

RESULTS_CSV="$(cd .. && pwd)/quality_results.csv"
SUMDIR="$(mktemp -d -t quality_sums.XXXXXX)"
trap 'rm -rf "$SUMDIR"' EXIT

echo "Benchmarking time-to-quality on '$BRANCH' ($COMMIT, $DIRTY): seeds=[$SEEDS]"
[ -n "$NOTE" ] && echo "note: $NOTE"
[ -n "$*" ] && echo "overrides: $*"

for s in $SEEDS; do
  t0=$(date +%s.%N)
  SUMMARY_PATH="$SUMDIR/s$s.json" ./quality_run.sh --seed "$s" "$@" > "$SUMDIR/s$s.log" 2>&1 || true
  t1=$(date +%s.%N)
  ttq=$(python3 -c "import json;print(json.load(open('$SUMDIR/s$s.json')).get('time_to_quality_seconds'))" 2>/dev/null || echo None)
  it=$(python3 -c "import json;print(json.load(open('$SUMDIR/s$s.json')).get('num_iterations'))" 2>/dev/null || echo "?")
  echo "  seed $s: time_to_quality=${ttq}s  iter=${it}  (wall $(python3 -c "print(f'{$t1-$t0:.1f}')")s)"
done

SUMDIR="$SUMDIR" RESULTS_CSV="$RESULTS_CSV" BRANCH="$BRANCH" COMMIT="$COMMIT" \
  DIRTY="$DIRTY" NOTE="$NOTE" EXTRA="$*" python3 - <<'PY'
import json, os, csv, glob, datetime, pathlib, statistics
sums = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(os.environ["SUMDIR"], "s*.json")))]
ttq  = [s["time_to_quality_seconds"] for s in sums if s.get("time_to_quality_seconds") is not None]
iters= [s["num_iterations"] for s in sums]
reached = sum(1 for s in sums if s.get("reached_quality"))
n = len(sums)
def ms(xs): return (round(statistics.mean(xs),2), round(statistics.pstdev(xs),2)) if xs else (None,None)
ttq_m, ttq_s = ms(ttq); it_m, it_s = ms([float(i) for i in iters])
row = {
    "timestamp_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    "branch": os.environ["BRANCH"], "commit": os.environ["COMMIT"], "dirty": os.environ["DIRTY"],
    "seeds": n, "reached": f"{reached}/{n}",
    "ttq_mean_s": ttq_m, "ttq_std_s": ttq_s,
    "ttq_min_s": round(min(ttq),2) if ttq else None, "ttq_max_s": round(max(ttq),2) if ttq else None,
    "crossing_iter_mean": it_m, "crossing_iter_std": it_s,
    "overrides": os.environ.get("EXTRA",""), "note": os.environ.get("NOTE",""),
}
p = pathlib.Path(os.environ["RESULTS_CSV"]); new = not p.exists()
with open(p, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(row))
    if new: w.writeheader()
    w.writerow(row)
print(f"\ntime-to-quality: {ttq_m}s ± {ttq_s}  (min {row['ttq_min_s']}, max {row['ttq_max_s']}, {reached}/{n} reached)")
print(f"crossing iter:   {it_m} ± {it_s}")
print(f"-> {p}")
PY
