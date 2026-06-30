# Benchmarks — how to run them

This directory is the single home for the speed benchmarks: two scripts
(`bench_run.sh`, `bench_measure.sh`), their result CSVs, and the experiment
write-up (`EXPERIMENT_LOG.md`). Run everything from anywhere — the scripts `cd`
to the repo root themselves.

## The two scripts

- **`bench_run.sh <kind> [flags]`** — one benchmark execution (forwards trailing
  flags to the underlying `ppo.py`/`sft.py`, which override the recipe).
- **`bench_measure.sh <kind> [note] [flags]`** — repeat/aggregate that benchmark,
  append a row to the matching CSV, and enforce its correctness gate.

`<kind>` ∈ `ppo-speed`, `ppo-quality`, `sft`.

| kind | what it times | result CSV | gate |
|------|---------------|-----------|------|
| `ppo-speed`   | fixed-iteration PPO loop (throughput ~0; pure compute) | `results.csv` | **pure-speed**: clean tree + pytest + iter-1 loss/kl/grad-norm signature unchanged vs `baseline_signature.json` |
| `ppo-quality` | wall to reach EMA(`rollout/reward`) ≥ −0.15 finetuning the cached SFT ckpt | `quality_results.csv` | **numerics-allowed**: sweep 5 seeds, compare means |
| `sft`         | SFT training loop to a fixed `val_loss` (cached dataset, eval off) | `sft_bench_results.csv` | **pure-speed**: `val_loss` must stay bit-identical |

(There is also a Rust throughput micro-bench, `tests/bench_throughput.py` — run
`WANDB_MODE=disabled uv run python tests/bench_throughput.py`.)

## Typical use

```bash
# PPO pure-speed (the gated, official way to log a row):
tests/benchmarks/bench_measure.sh ppo-speed "moved env stepping off the hot path"
RUNS=10 tests/benchmarks/bench_measure.sh ppo-speed "..."
ALLOW_SIGNATURE_CHANGE=1 tests/benchmarks/bench_measure.sh ppo-speed "switch to TF32"  # intentional numeric change
REFRESH_BASELINE=1 tests/benchmarks/bench_measure.sh ppo-speed "..."                    # (re)write baseline sig

# PPO time-to-quality (5-seed sweep):
tests/benchmarks/bench_measure.sh ppo-quality "lr 8e-4" --learning-rate 8e-4
SEEDS="1 2 3" tests/benchmarks/bench_measure.sh ppo-quality

# SFT training speed:
tests/benchmarks/bench_measure.sh sft "GPU-resident data"

# Just one run (timing / a quick check), no logging:
tests/benchmarks/bench_run.sh ppo-speed
TOTAL_TIMESTEPS=4096 tests/benchmarks/bench_run.sh ppo-speed   # 1 iteration
```

## Rules that keep the numbers honest

- **Pure-speed (`ppo-speed`, `sft`) must not change the computation.** The gate
  proves it: `ppo-speed` reproduces the iter-1 signature bit-for-bit; `sft` keeps
  `val_loss` bit-identical. If a change moves them, it altered the math or skipped
  a step — that's a *numerics* change (use `ALLOW_SIGNATURE_CHANGE=1` for
  `ppo-speed`, with a human sign-off) not a speed win, and don't "win" by removing
  real work.
- **`ppo-quality` is numerics-allowed but noisy.** crossing-iter has ±15-23%
  seed variance; a single seed is one draw. Sweep seeds and compare the mean
  (≈7 s ≈ 2σ is the real-change threshold). The deterministic per-iter wall (in
  the run's summary JSON) is the low-noise signal — use it to attribute wins.
- **Measure on a clean tree** (`ppo-speed` enforces this) so a row maps to a
  commit. `ppo-speed` runs 8 iterations on purpose: real training is 100s of
  iters, so steady-state per-iter cost — not startup — is what to optimise.
- **Why a given line is fast lives in `EXPERIMENT_LOG.md`, not the source.** Keep
  code terse; record the rationale (and the dead ends) in the log so nobody
  re-treads them.
