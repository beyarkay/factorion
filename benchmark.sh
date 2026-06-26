#!/usr/bin/env bash
#
# benchmark.sh — the gated, official way to benchmark an experiment branch.
#
# measure.sh is the bare "hyperfine + log a CSV row" core (fine for scratch
# runs). benchmark.sh wraps it with the guards that keep the results.csv log
# honest, and runs them fail-fast (cheapest gate first) so a broken or
# math-changing edit is caught in seconds, not after ~3.5 min of benchmarking:
#
#   1. Clean tree         — refuse to measure uncommitted work. The official
#                           number must be reproducible from a commit hash.
#   2. pytest (+cargo)     — the change must not break the env / throughput sim.
#                           cargo test + a maturin rebuild only when factorion_rs
#                           changed vs main (a stale .so would measure old code).
#   3. Invariance guard    — the run is deterministic (seeded + use_deterministic_
#                           algorithms), so a *pure speed* change must reproduce
#                           main's iter-1 loss/kl/grad-norm bit-for-bit. If they
#                           move, the edit changed the computation — abort unless
#                           ALLOW_SIGNATURE_CHANGE=1 (an intentional numeric
#                           change a human has signed off on; also REFRESH the
#                           baseline then).
#   4. measure.sh          — hyperfine + append the row to ../results.csv.
#
# Usage:
#   ./benchmark.sh "moved env stepping off the hot path"   # note -> CSV + stdout
#   RUNS=10 ./benchmark.sh "..."                            # passed to measure.sh
#   ALLOW_SIGNATURE_CHANGE=1 ./benchmark.sh "switch to TF32"  # intentional math change
#   REFRESH_BASELINE=1 ./benchmark.sh "..."                # (re)write the baseline sig
#
# The baseline signature lives at ../baseline_signature.json (outside the repo,
# next to results.csv/EXPERIMENTS.md, so it survives branch switches). It is
# (re)written automatically on the first run that has no baseline, and on any run
# with REFRESH_BASELINE=1.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NOTE="${1:-${NOTE:-}}"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"
PARENT="$(cd .. && pwd)"
BASELINE_SIG="$PARENT/baseline_signature.json"

# ── Gate 1: clean tree ───────────────────────────────────────────────────────
if [ -n "$(git status --porcelain)" ]; then
  echo "ERROR: working tree is dirty. Commit your change before benchmarking" >&2
  echo "       (the official number must be reproducible from a commit)." >&2
  git status --short >&2
  exit 1
fi
echo "[1/4] clean tree on '$BRANCH' ✓"

# ── Gate 2: tests (fast; cheapest correctness signal) ────────────────────────
echo "[2/4] pytest tests/ ..."
WANDB_MODE=disabled WANDB_DISABLED=true uv run python -m pytest tests/ -q

# Rebuild + cargo-test the Rust extension only if it changed vs main — otherwise
# we'd benchmark a stale .so and/or waste a ~minute compile on every Python edit.
MB="$(git merge-base HEAD main 2>/dev/null || echo '')"
if [ -n "$MB" ] && ! git diff --quiet "$MB" HEAD -- factorion_rs; then
  echo "      factorion_rs changed vs main — cargo test + maturin rebuild ..."
  (cd factorion_rs && cargo test)
  uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml
else
  echo "      factorion_rs unchanged vs main — skipping cargo/maturin."
fi

# ── Gate 3: invariance signature (cheap 1-iteration deterministic run) ────────
echo "[3/4] invariance check (iter-1 loss/kl/grad-norm) ..."
SIG_JSON="$(mktemp -t bench_sig.XXXXXX.json)"
trap 'rm -f "$SIG_JSON"' EXIT
TOTAL_TIMESTEPS=4096 SUMMARY_PATH="$SIG_JSON" ./run.sh >/dev/null 2>&1

if [ "${REFRESH_BASELINE:-}" = "1" ] || [ ! -f "$BASELINE_SIG" ]; then
  python3 -c "import json,sys; json.dump(json.load(open('$SIG_JSON'))['iter1_signature'], open('$BASELINE_SIG','w'), indent=2)"
  echo "      wrote baseline signature -> $BASELINE_SIG"
else
  SIG_JSON="$SIG_JSON" BASELINE_SIG="$BASELINE_SIG" \
  ALLOW="${ALLOW_SIGNATURE_CHANGE:-}" python3 - <<'PY'
import json, os, sys
cur = json.load(open(os.environ["SIG_JSON"]))["iter1_signature"]
base = json.load(open(os.environ["BASELINE_SIG"]))
if cur == base:
    print("      signature matches baseline ✓ (pure-speed change)")
else:
    print("ERROR: iter-1 signature DIFFERS from baseline — this change altered the", file=sys.stderr)
    print("       computation, not just its speed. Per-key diff (baseline -> current):", file=sys.stderr)
    for k in sorted(set(base) | set(cur)):
        b, c = base.get(k), cur.get(k)
        if b != c:
            print(f"         {k}: {b} -> {c}", file=sys.stderr)
    if os.environ.get("ALLOW") == "1":
        print("       ALLOW_SIGNATURE_CHANGE=1 set — proceeding (intentional numeric change).", file=sys.stderr)
        print("       Re-run with REFRESH_BASELINE=1 once merged to update the baseline.", file=sys.stderr)
    else:
        print("       If intentional (TF32/AMP/reduction order), re-run with ALLOW_SIGNATURE_CHANGE=1.", file=sys.stderr)
        sys.exit(1)
PY
fi

# ── Gate 4: the actual measurement ───────────────────────────────────────────
echo "[4/4] hyperfine measurement ..."
NOTE="$NOTE" ./measure.sh "$NOTE"
