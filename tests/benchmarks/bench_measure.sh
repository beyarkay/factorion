#!/usr/bin/env bash
#
# bench_measure.sh <kind> [note] [flags...] — measure a benchmark, append a CSV
# row in tests/benchmarks/, and enforce that benchmark's correctness gate.
#
#   ppo-speed    GATED pure-speed: refuse a dirty tree, run pytest (+cargo if
#                factorion_rs changed), check the iter-1 loss/kl/grad-norm
#                signature is unchanged vs the baseline (a *pure-speed* change
#                must reproduce it bit-for-bit), then hyperfine -> results.csv.
#                Escapes: ALLOW_SIGNATURE_CHANGE=1 (intentional numeric change),
#                REFRESH_BASELINE=1 (rewrite the baseline), RUNS=/WARMUP=.
#   ppo-quality  Sweep seeds 1..5 (each deterministic) -> quality_results.csv.
#                SEEDS="1 2 3" to subset.
#   sft          hyperfine + assert val_loss unchanged -> sft_bench_results.csv.
#
# See tests/benchmarks/CLAUDE.md for the playbook, EXPERIMENT_LOG.md for results.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repo root
B="tests/benchmarks"
RUN="$B/bench_run.sh"

KIND="${1:?usage: bench_measure.sh <ppo-speed|ppo-quality|sft> [note] [flags]}"; shift || true
NOTE="${1:-${NOTE:-}}"; if [ $# -gt 0 ]; then shift; fi
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
COMMIT="$(git rev-parse --short HEAD)"
if [ -n "$(git status --porcelain)" ]; then DIRTY="dirty"; else DIRTY="clean"; fi

case "$KIND" in
  ppo-speed)
    RUNS="${RUNS:-5}"; WARMUP="${WARMUP:-1}"
    BASELINE_SIG="$B/baseline_signature.json"
    # Gate 1: clean tree (the official number must come from a commit).
    if [ "$DIRTY" = "dirty" ]; then
      echo "ERROR: working tree dirty; commit before measuring ppo-speed." >&2
      git status --short >&2; exit 1
    fi
    echo "[1/4] clean tree on '$BRANCH' OK"
    # Gate 2: tests (+ cargo/maturin only if factorion_rs changed vs main).
    echo "[2/4] pytest ..."
    WANDB_MODE=disabled WANDB_DISABLED=true uv run python -m pytest tests/ -q
    MB="$(git merge-base HEAD main 2>/dev/null || echo '')"
    if [ -n "$MB" ] && ! git diff --quiet "$MB" HEAD -- factorion_rs; then
      echo "      factorion_rs changed vs main — cargo test + maturin rebuild ..."
      (cd factorion_rs && cargo test)
      uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
    fi
    # Gate 3: iter-1 invariance signature (cheap 1-iter deterministic run).
    echo "[3/4] invariance check (iter-1 loss/kl/grad-norm) ..."
    SIG_JSON="$(mktemp -t bench_sig.XXXXXX.json)"; trap 'rm -f "$SIG_JSON"' EXIT
    TOTAL_TIMESTEPS=4096 SUMMARY_PATH="$SIG_JSON" "$RUN" ppo-speed >/dev/null 2>&1
    if [ "${REFRESH_BASELINE:-}" = "1" ] || [ ! -f "$BASELINE_SIG" ]; then
      python3 -c "import json;json.dump(json.load(open('$SIG_JSON'))['iter1_signature'],open('$BASELINE_SIG','w'),indent=2)"
      echo "      wrote baseline signature -> $BASELINE_SIG"
    else
      SIG_JSON="$SIG_JSON" BASELINE_SIG="$BASELINE_SIG" ALLOW="${ALLOW_SIGNATURE_CHANGE:-}" python3 - <<'PY'
import json,os,sys
cur=json.load(open(os.environ["SIG_JSON"]))["iter1_signature"]; base=json.load(open(os.environ["BASELINE_SIG"]))
if cur==base: print("      signature matches baseline OK (pure-speed change)")
else:
    print("ERROR: iter-1 signature DIFFERS — this change altered the computation:",file=sys.stderr)
    for k in sorted(set(base)|set(cur)):
        if base.get(k)!=cur.get(k): print(f"         {k}: {base.get(k)} -> {cur.get(k)}",file=sys.stderr)
    if os.environ.get("ALLOW")=="1": print("       ALLOW_SIGNATURE_CHANGE=1 — proceeding; REFRESH_BASELINE=1 once merged.",file=sys.stderr)
    else: print("       If intentional (TF32/AMP/reduction order) re-run with ALLOW_SIGNATURE_CHANGE=1.",file=sys.stderr); sys.exit(1)
PY
    fi
    # Gate 4: hyperfine + log row.
    echo "[4/4] hyperfine ($WARMUP warmup + $RUNS runs) ..."
    HF_JSON="$(mktemp -t hf.XXXXXX.json)"; SUMMARY_JSON="$(mktemp -t ppo_sum.XXXXXX.json)"
    trap 'rm -f "$SIG_JSON" "$HF_JSON" "$SUMMARY_JSON"' EXIT
    export SUMMARY_PATH="$SUMMARY_JSON"
    hyperfine --warmup "$WARMUP" --runs "$RUNS" --command-name "$BRANCH" \
      --export-json "$HF_JSON" "$RUN ppo-speed"
    HF_JSON="$HF_JSON" SUMMARY_JSON="$SUMMARY_JSON" RESULTS_CSV="$B/results.csv" \
      BRANCH="$BRANCH" COMMIT="$COMMIT" DIRTY="$DIRTY" NOTE="$NOTE" \
      python3 "$B/_log_result.py"
    ;;

  ppo-quality)
    SEEDS="${SEEDS:-1 2 3 4 5}"
    SUMDIR="$(mktemp -d -t quality_sums.XXXXXX)"; trap 'rm -rf "$SUMDIR"' EXIT
    echo "ppo-quality on '$BRANCH' ($COMMIT, $DIRTY): seeds=[$SEEDS]"
    [ -n "$NOTE" ] && echo "note: $NOTE"
    for s in $SEEDS; do
      t0=$(date +%s.%N)
      SUMMARY_PATH="$SUMDIR/s$s.json" "$RUN" ppo-quality --seed "$s" "$@" > "$SUMDIR/s$s.log" 2>&1 || true
      t1=$(date +%s.%N)
      ttq=$(python3 -c "import json;print(json.load(open('$SUMDIR/s$s.json')).get('time_to_quality_seconds'))" 2>/dev/null || echo None)
      it=$(python3 -c "import json;print(json.load(open('$SUMDIR/s$s.json')).get('num_iterations'))" 2>/dev/null || echo '?')
      echo "  seed $s: time_to_quality=${ttq}s iter=${it} (wall $(python3 -c "print(f'{$t1-$t0:.1f}')")s)"
    done
    SUMDIR="$SUMDIR" RESULTS_CSV="$B/quality_results.csv" BRANCH="$BRANCH" COMMIT="$COMMIT" \
      DIRTY="$DIRTY" NOTE="$NOTE" EXTRA="$*" python3 - <<'PY'
import json,os,csv,glob,datetime,pathlib,statistics
sums=[json.load(open(p)) for p in sorted(glob.glob(os.path.join(os.environ["SUMDIR"],"s*.json")))]
ttq=[s["time_to_quality_seconds"] for s in sums if s.get("time_to_quality_seconds") is not None]
iters=[s["num_iterations"] for s in sums]; reached=sum(1 for s in sums if s.get("reached_quality")); n=len(sums)
ms=lambda xs:(round(statistics.mean(xs),2),round(statistics.pstdev(xs),2)) if xs else (None,None)
tm,ts=ms(ttq); im,is_=ms([float(i) for i in iters])
row={"timestamp_utc":datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),"branch":os.environ["BRANCH"],
 "commit":os.environ["COMMIT"],"dirty":os.environ["DIRTY"],"seeds":n,"reached":f"{reached}/{n}",
 "ttq_mean_s":tm,"ttq_std_s":ts,"ttq_min_s":round(min(ttq),2) if ttq else None,"ttq_max_s":round(max(ttq),2) if ttq else None,
 "crossing_iter_mean":im,"crossing_iter_std":is_,"overrides":os.environ.get("EXTRA",""),"note":os.environ.get("NOTE","")}
p=pathlib.Path(os.environ["RESULTS_CSV"]);new=not p.exists()
with open(p,"a",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(row));
    if new: w.writeheader()
    w.writerow(row)
print(f"\ntime-to-quality: {tm}s +/- {ts} ({reached}/{n} reached) crossing {im}+/-{is_} -> {p}")
PY
    ;;

  sft)
    RUNS="${RUNS:-3}"; WARMUP="${WARMUP:-1}"; BASELINE_VAL_LOSS="${BASELINE_VAL_LOSS:-1.6888}"
    HF_JSON="$(mktemp -t sft_hf.XXXXXX.json)"; SUMMARY_JSON="$(mktemp -t sft_sum.XXXXXX.json)"
    trap 'rm -f "$HF_JSON" "$SUMMARY_JSON"' EXIT
    echo "sft on '$BRANCH' ($COMMIT, $DIRTY): $WARMUP warmup + $RUNS runs"
    [ -n "$NOTE" ] && echo "note: $NOTE"
    export SUMMARY_PATH="$SUMMARY_JSON"
    hyperfine --warmup "$WARMUP" --runs "$RUNS" --command-name "$BRANCH-sft" \
      --export-json "$HF_JSON" "$RUN sft"
    HF_JSON="$HF_JSON" SUMMARY_JSON="$SUMMARY_JSON" RESULTS_CSV="$B/sft_bench_results.csv" \
      BRANCH="$BRANCH" COMMIT="$COMMIT" DIRTY="$DIRTY" NOTE="$NOTE" \
      BASELINE_VAL_LOSS="$BASELINE_VAL_LOSS" python3 - <<'PY'
import json,os,csv,datetime,pathlib
hf=json.load(open(os.environ["HF_JSON"]))["results"][0]; summ=json.load(open(os.environ["SUMMARY_JSON"]))
vl=summ["val_loss"]; base=float(os.environ["BASELINE_VAL_LOSS"]); ok=abs(vl-base)<1e-4
row={"timestamp_utc":datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),"branch":os.environ["BRANCH"],
 "commit":os.environ["COMMIT"],"dirty":os.environ["DIRTY"],"wall_mean_s":round(hf["mean"],2),"wall_std_s":round(hf["stddev"],2),
 "wall_min_s":round(hf["min"],2),"wall_max_s":round(hf["max"],2),"val_loss":vl,
 "invariant":"OK" if ok else f"VIOLATED(base {base})","note":os.environ.get("NOTE","")}
p=pathlib.Path(os.environ["RESULTS_CSV"]);new=not p.exists()
with open(p,"a",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(row))
    if new: w.writeheader()
    w.writerow(row)
print(f"\nwall: {row['wall_mean_s']}s +/- {row['wall_std_s']}  val_loss: {vl} [{row['invariant']}] -> {p}")
if not ok: print("!! val_loss changed — NOT a pure-speed change.")
PY
    ;;
  *)
    echo "unknown kind: $KIND (expected ppo-speed|ppo-quality|sft)" >&2; exit 1 ;;
esac
