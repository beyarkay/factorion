# PPO speed benchmarking playbook

Goal: make PPO training (`ppo.py`) **faster per wall-clock second** without
changing what it computes. This file is the contract for that work. If you are a
fresh agent picking this up, read it top to bottom before touching anything.

## The measuring stick

```bash
./run.sh        # one fixed-cost PPO run (8 iterations, 32768 timesteps)
./benchmark.sh  # GATED official run: clean-tree + pytest + invariance check, then measure
./measure.sh    # bare core: run.sh under hyperfine (5 runs + 1 warmup) -> ../results.csv
```

We run **8 iterations** deliberately: real training is 6+ hours over 100s of
iterations, so the **steady-state per-iteration cost** is the thing to optimise.
Startup (Python import + one-time `torch.compile` warmup, a few seconds) is
negligible in production, so we run enough iterations that per-iter compute —
not startup — dominates the measured wall time. `mean_s` in `results.csv` is
still the headline, but `rollout_steady_s`/`update_steady_s` (which drop
iteration 1) give the startup-free per-iter view. **Only compare `mean_s`
between rows with the same `total_timesteps`** (older 8192-timestep rows are not
comparable to current 32768 ones).

Use **`./benchmark.sh "<note>"`** for any result you log — it refuses a dirty
tree, runs the tests, and verifies the change didn't alter the computation
(deterministic iter-1 loss/kl/grad-norm vs `../baseline_signature.json`) before
spending ~3.5 min on hyperfine. `measure.sh` is the unguarded core it calls; use
it directly only for throwaway scratch timing.

`run.sh` exercises the full loop — RL rollouts **and** optimiser steps — on a
fixed, reduced config. It is **not** a learning run; reported throughput is ~0.
We only care about how fast it goes. `ppo.py` prints a rollout-vs-optimiser
breakdown at the end and writes it to the summary JSON; `measure.sh` logs it.

Current split (baseline): **rollout dominates the optimiser (~2.3×)** — the
env/rollout path is the first place to look.

## The loop

Run this loop, one idea at a time:

1. **Pick one idea** to speed up rollouts or optimisation. Write it in
   `../EXPERIMENTS.md` *before* coding (hypothesis + expected win).
2. **Branch off `main`** (or the fastest branch so far): `git switch -c
   speedup/<short-name>`. One idea per branch — do **not** stack ideas, or you
   can't attribute the win.
3. **Implement** the change, then **commit it** (the benchmark refuses a dirty
   tree, so the number is always tied to a commit).
4. **Measure**: `./benchmark.sh "<one-line description>"`. Runs the gates
   (clean-tree, pytest, invariance), then appends a row to `../results.csv` and
   prints the mean ± stddev.
5. **Log** the outcome in `../EXPERIMENTS.md`: branch name, result vs baseline,
   what you learned (even — especially — if it was slower or flat).
6. **Repeat.** Keep ideas that win (merge to `main` so the baseline moves),
   drop ideas that don't.

## Rules (these are what keep the numbers honest)

- **`run.sh` is immutable.** Never change its hyperparameters, `--num-envs`,
  `--num-steps`, `--total-timesteps`, grid `--size`, network dims, or remove
  `torch.compile`. The benchmark must measure the *same work* finishing faster.
  Lowering the work is not a speedup — it is hacking the metric. If you think
  the config itself is wrong, raise it with the human; don't silently edit it.
- **One idea per branch, branched off `main`.** Keeps attribution clean and
  matches the repo's small-commit convention.
- **Measure with `./measure.sh`, not by hand.** Same hyperfine flags every
  time so rows are comparable.
- **Merge wins to `main`.** The baseline is "current `main`". When a change
  lands, the bar moves up for the next idea.

## Noise floor

Run-to-run stddev is ≈ ±1s (≈3% of mean). Treat anything under ~5% as noise,
not a win. If a result looks borderline, bump sampling: `RUNS=10 ./measure.sh`.
Keep the box otherwise idle (no other GPU jobs) while measuring.

## Correctness

By decision, runs are **not** gated on a learning metric: RL loss is noisy and
there is no metric that reliably moves in a handful of iterations, and a
time-to-X-loss
gate would need a long rollout. So we run **ungated speed tests** — speed is the
only headline number.

Cheap guards that *don't* depend on learning, in order of value:

1. **`run.sh` immutability** (above) — removes the main reason you'd need a gate
   at all. Most ways to "cheat" the benchmark are just doing less work; freezing
   the work config blocks them.
2. **The existing fast test suite** — `WANDB_MODE=disabled WANDB_DISABLED=true
   uv run python -m pytest tests/ -v`, plus the PPO/SFT smoke tests in
   `CLAUDE.md`, and `cargo test` if you touched Rust. These verify the env and
   throughput simulation still compute correctly, in seconds, with no RL
   learning involved. Run them before merging a change to `main`.
3. **Invariance signature** (optional, for changes that claim to be *pure
   performance*) — the run is deterministic (fixed seed, `torch_deterministic`),
   so a pure-perf change should reproduce the same per-iteration loss / KL /
   grad-norm trajectory, just faster. If those numbers move, the change altered
   the computation — either a bug, or an intentional numeric change (TF32/AMP,
   different reduction order) that a human should sign off on. Don't rely on
   bit-identity blindly: GPU/compile nondeterminism can perturb low-order bits.

## Where things live

- `../results.csv` — append-only machine log (branch, mean, min, max, stddev, …).
  `measure.sh` writes here (one dir above the repo) so it survives branch
  switches. A **committed snapshot** lives at `results.csv` in the repo root.
- `../EXPERIMENTS.md` — human narrative: ideas, hypotheses, outcomes, learnings.
  Live copy is outside the repo; a **committed snapshot** lives at
  `EXPERIMENTS.md` in the repo root (so the log travels with `git push`).
- `run.sh`, `measure.sh`, `benchmark.sh`, `scripts/_log_result.py` — the harness,
  in the repo so every branch inherits the identical measuring stick.

## Stop criterion

There isn't a hard target yet — drive the mean down and stop when ideas stop
paying off (e.g. 3 consecutive ideas each under ~5%). Set a concrete target with
the human if you want one (e.g. "mean ≤ 20s").
