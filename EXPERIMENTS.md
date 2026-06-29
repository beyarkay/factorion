# PPO speedup experiments

Human narrative for the PPO speed-benchmarking effort. The playbook (rules, how
to measure) lives in `factorion/BENCHMARK.md`. The machine-readable results are
in `results.csv` (next to this file). This file is the *why* and *what I
learned*; the CSV is the *numbers*.

Workflow per idea: pick idea → log it here → branch off `main` (or fastest
branch so far) (`speedup/<name>`) → implement → commit → `./benchmark.sh
"<note>"` → record outcome here.

## ⚠️ Handoff note (read first — these numbers are from a CPU box)

This whole session ran on a box whose **GPU driver was too old for the torch
build**, so everything executed on **CPU** (48 cores). That matters for which
wins transfer to a GPU machine:

- **`Cap CPU intra-op threads to 6` (the 2.83× win) is CPU-only** — it's guarded
  by `device.type == "cpu"`, so on GPU it's a no-op. **Do not expect that 2.83×
  on GPU.** It is, however, a real and large win for CPU/CI runs.
- **`numpy per-step diagnostics` and `trim per-step env overhead` transfer** —
  they cut Python/`step()` overhead in the rollout, which is device-independent.
  On GPU the rollout's relative share differs (NN moves to the GPU), so the % may
  change, but the work removed is real everywhere.
- **Re-baseline on the GPU box first** (`./benchmark.sh` on `main`) before
  trusting any comparison — absolute numbers here are not comparable to GPU.
- The biggest *untried* lever flagged for GPU: **NN lower precision (bf16/AMP)**
  for the fwd+bwd — needs a human sign-off (changes numerics) per the invariance
  rule, but that's where the time is once the NN runs on-device.

The committed `results.csv` / `EXPERIMENTS.md` here are a **snapshot** of the CPU
session. During active benchmarking the harness writes the live copies to
`../results.csv` / `../EXPERIMENTS.md` (one dir above the repo) so they survive
branch switches; reconcile as you like on the GPU box.

## Baseline

- **Config**: `run.sh` — **8 iterations (32768 timesteps)** as of `897809f`
  (bumped from 2 to amortize startup; real runs do 100s of iters, so steady-state
  per-iter cost is the goal). 16 envs × 256 steps, net 93/69/96, grid size 11,
  seed 1, `torch.compile` on. CPU execution (no usable GPU on this box).
- **Hardware**: NVIDIA RTX 2000 Ada present but driver too old for this torch
  build → runs on **CPU**, 48 cores.
- **Number** (8-iter config, the live bar to beat): **37.091 s ± 0.627 s** on
  `main` @ `897809f`. Steady per-iter: rollout 2.64s + update 1.45s ≈ 4.1s;
  startup ≈ 4.4s (~12% of wall). *(Historical 2-iter rows: 36.393 s → 12.846 s
  after the thread cap; not comparable to 8-iter rows — different
  `total_timesteps`.)*
- **Known split**: rollout ≈ 2.25× the optimiser (steady-state 11.0 s vs 4.9 s
  per iter) — rollout/env path is the prime suspect.
- **Invariance signature** (`../baseline_signature.json`): policy_loss
  0.22097248, value_loss 1.87765145, grad_norm 44.52099609 — a pure-speed change
  must reproduce these exactly.

## Idea backlog

Candidate directions (not yet tried — move to the log below as you take them):

- Reduce per-step Python<->Rust overhead in the rollout (batch env steps?).
- Vectorise / parallelise env stepping across the 16 envs (CPU-bound rollout).
- Cut host<->device transfers per rollout step; keep tensors on-GPU.
- Reduce `torch.compile` recompiles / warmup cost.
- Larger minibatches / fewer Python-side optimiser iterations.
- Profile the rollout step to find the actual hot line before guessing.
- **Per-step shaping diagnostics** (`_compute_solution_match`, `tile_match_*` in
  `FactorioEnv.step`) are logged in `info` but NOT used in reward, and the
  rollout loop never reads them. The solution mask they use is constant within
  an episode yet recomputed every step (4096×/iter). Cache the per-episode
  constants (mask/n/orig tensors) at `reset`, or vectorise in numpy — same
  values, faster, applies to production too (signature-preserving). ~2.4s of the
  threads=24 profile; relatively bigger now that the NN is 4× cheaper.

### speedup/numpy-diagnostics — per-step metrics on a zero-copy numpy view
- **Hypothesis**: the probe below showed per-step logged-only metrics cost
  ~0.765 s/iter (31% of rollout). They're tiny-grid torch reductions dominated
  by per-op dispatch overhead. Computing them on a zero-copy `.numpy()` view of
  the world (shares the torch buffer → ~free, stays in sync) should recover most
  of that tax while keeping every metric (identical integer-count values).
- **Change**: single `wnp = self._world_CWH.numpy()` view for the diagnostic
  block (tile-match, material_cost, final_dir_reward, num_entities); solution
  arrays cached as numpy at reset; `_compute_solution_match` numpy-ified; Channel
  indices hoisted to module constants.
- **Result**: **30.363 s ± 0.405 s** vs main 35.688 s → **−14.9%** (ranges
  fully separated). rollout_steady **2.401 → 1.777 s/iter (−26%)** — within
  0.41 s of the metrics-stripped LEAN floor (29.95 s), i.e. ~87% of the
  monitorability tax recovered with full monitorability kept. Signature MATCH
  (verified pure-speed); reward-shaping tests pass. Commit `9625ba1`.
- **Verdict**: **KEEP — merged to `main`.** Biggest win since the thread cap.

## Probes (measurements, not merged)

### probe/metrics-overhead — the "monitorability tax"
- **Question** (from human): how much wall time do we pay for the per-step
  logging/metrics we can't remove IRL? Establishes the *ceiling* for optimising
  them.
- **Method**: `BENCH_SKIP_DIAGNOSTICS=1` env gate (default off) skips every
  logged-only per-step metric (tile-match, solution-match shaping,
  material_cost, final_dir_reward, num_placed_entities); info keys kept with
  placeholders. Signature **MATCHES** baseline → confirmed these are pure
  overhead (zero effect on reward/obs/learning).
- **Result**: rollout_steady **2.466 → 1.701 s/iter** = **−0.765 s/iter
  (−31% of rollout)**, update unchanged. ≈6.1 s over 8 iters, ≈17% of total wall.
  Wall-clock: **FULL 35.58 s → LEAN 29.95 s = −5.63 s (−15.8% of wall)**.
- **Takeaway**: the per-step diagnostics are ~a third of the rollout. Strong
  mandate to *optimise* (not remove) them — numpy-on-a-zero-copy-view recovers
  most of this while keeping full monitorability (see speedup/numpy-match-…).
  Not merged (knob is off-by-default infra; can't drop metrics IRL).

## Log

Newest first. One entry per branch.

<!--
### speedup/<name> — <one line>
- **Hypothesis**: why this should be faster.
- **Change**: what you did.
- **Result**: mean Xs ± Y (baseline Zs) → -N% / +N% / flat.
- **Verdict**: keep (merged) / drop. Learnings.
-->

### speedup/env-step-overhead — trim per-step Python in FactorioEnv.step
- **Hypothesis**: with the NN now 4× cheaper (threads=6), `FactorioEnv.step` is
  the largest Python cost in the profile (1.83s self / 4.98s cum over 3 iters).
  Three signature-preserving wastes, each hit ~4096×/iter: (a) `str2ent`/
  `str2item` do a **linear scan** over the entity/item tables on every call and
  are invoked ~10×/step with constant literals (118k+ scans/iter); (b)
  `_compute_solution_match` recomputes episode-constant tensors (mask, n, orig
  ent/dir) every step; (c) `entity_to_be_replaced` is a dead per-step tensor
  index. None feed reward/obs → signature must stay identical.
- **Change**: hoist the constant entity/item ids to module-level constants;
  cache the solution-match mask + masked originals at `reset`; delete the dead
  index.
- **Result**: **35.688 s ± 0.362 s** vs baseline 37.091 s → **−3.8% overall**
  (ranges non-overlapping, so real not noise). The startup-free signal moved
  more: **rollout_steady 2.636 → 2.401 s/iter (−8.9%)**; update_steady flat
  (1.451→1.432) as expected (rollout-side change only). Invariance check PASSED
  (signature identical — confirmed pure-speed). Commit `b6f8442`.
- **Verdict**: **KEEP — merged to `main`.** Small headline % because the
  unaffected startup (~4.4s) and optimiser path dilute it, but the rollout
  per-iter win is real and compounds over 100s of production iterations. Next:
  the simulate_throughput per-step conversion, and the optimiser/NN path.

### speedup/cpu-thread-count — cap CPU intra-op threads
- **Hypothesis**: the profile shows the NN forward/backward dominates (conv2d
  ~11.5s, backward ~6.3s) and runs on **CPU** (no usable GPU on this box). torch
  defaults to 24 intra-op threads, but the tensors are tiny (batch 16, 11×11
  grid), so thread-launch/sync overhead per `conv2d` should swamp the compute.
  Capping `torch.set_num_threads` low should cut both rollout and update time.
- **Change**: `torch.set_num_threads(6)` when running on CPU (hardcoded; GPU
  unaffected). run.sh inherits it automatically (no arg / no run.sh edit).
- **Caveat (signed off)**: changes float reduction order → iter-1 signature
  differs from the threads=24 baseline. Same computation, different rounding.
  Human-approved; `../baseline_signature.json` refreshed at the new config.
- **Thread sweep** (steady per-iter, 2-iter run): 24→15.4s, 16→9.3s, 8→4.9s,
  **6→4.0s, 5→4.0s**, 4→4.2s, 3→4.6s, 2→5.7s. Optimum ≈ 5–6.
- **Result**: **12.846 s ± 0.059 s** vs baseline 36.393 s → **−64.7% (2.83×)**,
  and stddev dropped 0.745→0.059. rollout_steady 11.0→2.58s, update_steady
  4.9→1.42s. Commit `14419d7`.
- **Verdict**: **KEEP — merged to `main`.** Baseline signature refreshed to the
  threads=6 config (grad_norm 49.61). Biggest single lever; the rest of the
  ~12.8s is now dominated by the fixed torch.compile warmup + Python import.
