# Benchmark experiment log

The *why* and *what we learned* across all of the speed benchmarks; the
machine-readable numbers live in the `*.csv` files next to this one, and the
how-to-run playbook is `tests/benchmarks/CLAUDE.md`. Three benchmarks:

1. **PPO pure-speed** (`bench_* ppo-speed` → `results.csv`) — fixed-iteration
   loop, gated by an iter-1 invariance signature. (The bulk of this file;
   historical entries below were run with the old `run.sh`/`benchmark.sh`, now
   `bench_run.sh ppo-speed` / `bench_measure.sh ppo-speed`.)
2. **PPO time-to-quality** (`bench_* ppo-quality` → `quality_results.csv`) — wall
   time to reach a fixed reward finetuning the SFT checkpoint (numerics-allowed).
3. **SFT training speed** (`bench_* sft` → `sft_bench_results.csv`) — the training
   loop to a fixed `val_loss` (pure-speed, bit-identical gate).

This file also holds the **rationale that used to live as code comments** (why a
given line is fast) so the source stays terse — grep here when a change looks
surprising.

Workflow per PPO pure-speed idea: log it here → branch → implement → commit →
`tests/benchmarks/bench_measure.sh ppo-speed "<note>"` → record the outcome.

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
- **Overlap factory-building with GPU compute** (signature-preserving version of
  "pre-build factories"): each reset rebuilds a factory from scratch on the CPU
  (`build_factory` 4.2 s + blank 2.8 s of the 2-iter profile) inline in the
  rollout, while the GPU sits idle during that Python. Prebuild each env's *next*
  factory for its exact upcoming seed (so identical factories, identical order →
  signature preserved) on a CPU thread pool, overlapped with the NN forward/
  backward. Bigger refactor (SyncVectorEnv autoresets inline), but pure-speed.
  NB: a *fixed pool* / cache of factories would change which factories each env
  sees (seeds never repeat within a run) → changes the iter-1 signature → that's
  a numerics change, not pure speed.
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

## GPU re-baseline (this box, after driver fix)

GPU is now usable: driver 580.159.04 / CUDA 13.0, torch 2.12.1+cu130,
`torch.cuda.is_available()` True on an RTX 2000 Ada. (The old session's blocker —
driver too old for the torch build — is gone.) Bootstrapped from scratch: no `uv`
and no `.venv` on the box, so `uv` install + `uv sync` + `maturin develop`.

- **GPU baseline** (`./measure.sh`, `main` @ `e2694fb`): **30.070 s ± 0.239 s**
  (vs the CPU session's 37.091 s). Steady per-iter: **rollout 2.10 s + update
  0.61 s**. The NN/optimiser path dropped to GPU (update 1.45 → 0.61 s/iter);
  **rollout is now ~3.4× the update and is the bottleneck** — it's CPU-bound env
  stepping (peak GPU util ~26%).
- The CPU thread-cap win is a **no-op on GPU** (guarded by `device=="cpu"`).
- `baseline_signature.json` is absent, so the first `./benchmark.sh` run writes a
  fresh **GPU** invariance signature (correct for re-baselining).

### Hardware utilization (measured on this box — for picking a machine)
Measured during a real `run.sh` (lightweight `nvidia-smi dmon`, negligible
perturbation; absolute timing matched the 29 s benchmark):
- **GPU: ~9–12% util (max ~28%), 220 MiB / 16 GB VRAM (1.3%), ~20 W / 70 W.**
  The NN (conv 93/69/96, batch ≤4096, 11×11 grid) is tiny → the RTX 2000 Ada is
  **massively over-provisioned**. A bigger / higher-VRAM GPU will NOT help at this
  model size. Only worth more GPU if the network is scaled up substantially.
- **CPU: single-core-bound** — only ~1–2 of 48 cores active. The rollout is serial
  Python (`SyncVectorEnv` steps 16 envs in a Python loop × 256 steps). **More
  cores won't help as-is**; the two levers are (a) **faster single-core clock**
  (direct win, no code change) or (b) **parallelize env stepping**
  (AsyncVectorEnv/multiprocessing) so many cores finally pay off — but IPC/pickle
  overhead at this small per-step cost may eat the gains; must be tested.
- **Loop split**: rollout 2.05 s/iter (77%) vs update 0.62 s/iter (23%). Rollout
  is the place to optimize. **Measurement caveat learned the hard way:** any
  manual run MUST use the exact `run.sh` args (esp. `--target-kl 0.02`, which
  early-stops update epochs) or update time balloons ~8×; and a CPU sampler steals
  the rollout's single core. Use `run.sh` + its summary JSON
  (`rollout_seconds_steady_mean`/`update_seconds_steady_mean`) for interior
  metrics — those resolve per-call wins the noisy `mean_s` can't (lower
  statistical significance, so repeat for borderline changes).

### GPU rollout profile (cProfile, 2 iters)
`reset()` dominates the rollout: **7.2 s cumulative** — `build_factory` 4.2 s +
`blank_entities`/`_remove_entities` 2.8 s, with **2.32 s pure self-time in
`_remove_entities`** (the single biggest self-time chunk in our code). The env
`step()` is only ~1.6 s. So the GPU rollout bottleneck is per-reset Python, not
the NN. Attacks below target that.

## Log

Newest first. One entry per branch.

### speedup/validity-numpy — numpy-view reads in step()'s validity checks
- **Hypothesis**: after gate-diagnostics + info-skip, `step()`'s own validity
  checks are a top rollout self-time. They read FOOTPRINT/ENTITIES per footprint
  tile via torch scalar indexing (`self._world_CWH[ch, tx, ty]`) inside `any()`
  generators (genexprs at 0.15 s self in the profile). The validity result feeds
  the world mutation → obs → signature, so it must run every step, but the
  *reads* can use a zero-copy numpy view of the CPU world (same trick as attack
  #1). Identical values (world unmutated at that point) → signature safe.
- **Change**: `world_np = self._world_CWH.numpy()`; read
  `world_np[_CH_FOOTPRINT/_CH_ENT, tx, ty]`; add `_CH_FOOTPRINT` constant.
- **Result**: **23.889 s ± 0.228 vs 24.427 → −2.2%** (ranges separated).
  **rollout_steady 1.432 → 1.362 s/iter (−4.9%)**. Signature **MATCHED ✓**; tests
  pass. Commit `f2be109`.
- **Verdict**: **KEEP — merged.** Same numpy-view lesson as attack #1, now on the
  validity hot path that surfaced once the diagnostics were gone.

### speedup/skip-info-nonterminal — skip the per-step info dict in training
- **Hypothesis**: follow-on to gate-diagnostics. `step()` still built the full
  ~25-key `info` dict every step, and `SyncVectorEnv` deep-aggregates it across
  16 envs every step. But the rollout reads `info` only for *finished* envs, and
  tests/eval keep `_full_diagnostics=True`. So on a non-terminal training step,
  emit only the cheap base `_get_info()` and skip the 25-key build + its vector
  aggregation. Subsumes the "lazy invalid_reason" idea (invalid_reason is only
  referenced inside the skipped block).
- **Change**: wrap the `info.update({...})` (and `steps_taken`) in
  `if terminated or truncated or self._full_diagnostics`.
- **Result**: **24.427 s ± 0.263 vs 25.325 → −3.5%** (ranges separated).
  **rollout_steady 1.535 → 1.432 s/iter (−6.7%)**. Signature **MATCHED ✓**;
  reward-shaping tests pass. Commit `8f6efde`.
- **Verdict**: **KEEP — merged.** The per-step info dict build + `SyncVectorEnv`
  deep-aggregation across 16 envs was a real cost; skipping it on non-terminal
  training steps compounds the gate-diagnostics win. Same monitorability tradeoff
  (reversible via `_full_diagnostics`).

### speedup/gate-diagnostics — skip dead per-step diagnostics in training
- **Hypothesis (the big lever)**: `step()` runs `simulate_throughput` + a block of
  numpy diagnostics (tile_match, shaping deltas, material_cost, final_dir_reward,
  num_entities, frac_reachable) on **every** step. But mid-episode none of it
  feeds the reward (just `-step_penalty` until termination) and the PPO rollout
  reads thput/frac_reachable/num_entities only for **finished** envs; the
  shaping/tile-match keys are read by **tests only**, never logged in training.
  The old probe measured this block at ~31% of rollout. Gate it: compute the full
  block only when `terminated or truncated or self._full_diagnostics`. Training
  rollout envs set `_full_diagnostics=False`; tests, eval (`run_rollout_eval`
  makes its own envs), and any per-step inspector keep the default `True`.
- **Why signature-safe**: the terminal reward (the only place throughput enters
  the reward) and every episode-end-consumed value are still computed on the
  terminal step. Verified: iter-1 signature **identical** to baseline; reward-
  shaping tests pass (they run with full diagnostics).
- **Contract note**: this lowers what the training path (and thus `run.sh`)
  computes — but only *dead* instrumentation, so real training gets the same
  speedup. It trades per-step monitorability in training for speed; reversible by
  flipping `_full_diagnostics` back to `True`. Flagged for the human.
- **Result**: **25.325 s ± 0.484 vs 28.785 → −12.0%** (ranges fully separated).
  **rollout_steady 2.002 → 1.535 s/iter (−23%)**; update_steady 0.617 → 0.52.
  Signature **MATCHED ✓**; full suite **3548 passed**. Commit `89d5535`.
- **Verdict**: **KEEP — merged.** Biggest win since the GPU unlock, and the only
  one that confirms the probe's "monitorability tax" (~23% of rollout here). The
  diagnostics block was genuinely dead in training; gating it is a real training
  speedup, not metric-hacking (signature identical, eval/tests unaffected). If
  per-step training monitoring is ever wanted, flip `_full_diagnostics=True`.

### speedup/rollout-update-microopts — dead-code + sync-defer bundle
- **Hypothesis**: a bundle of signature-safe rollout/update micro-opts (verified
  iter-1 signature identical to baseline):
  (a) **remove dead `old_approx_kl`** — `(-logratio_B).mean()` computed every
  minibatch, never read; (b) **remove dead `time_for_get_value` /
  `time_for_get_action_and_value`** timing (set, never read) → drops `time.time()`
  syscalls every forward; (c) **defer `clipfracs.item()`** — was a CUDA sync per
  minibatch, now accumulated on-GPU and converted once (mirror of attack #4);
  (d) **cache `torch.arange(B)`** for the per-tile gather (B constant per
  call-site); (e) **drop redundant `np.array()`** around already-numpy
  reward/obs/done before `as_tensor`.
- **Result**: **28.785 s ± 0.175 vs 28.870 → flat** (−0.3%, noise; rollout_steady
  2.002, update_steady 0.617 — both noise). Signature **MATCHED ✓**. Commit
  `89bf745`.
- **Verdict**: **KEEP — merged (benchmark-flat, cleaner + production-positive).**
  Removes genuinely dead compute (`old_approx_kl`, `time_for_*`) and a per-minibatch
  CUDA sync (clipfrac). Flat on this tiny-net GPU benchmark (same story as the
  other sync-reduction attacks) but strictly less work; worth keeping.

### speedup/blank-world-clone — clone cached blank in rejection loop — DROPPED
- **Hypothesis**: the MOVE_ONE_ITEM rejection loop rebuilds the empty grid via
  `torch.tensor(new_world()).permute()` on each failed/degenerate trial. Cache a
  blank template once and `.clone()` it instead. `new_world()` consumes no RNG →
  build-hash identical.
- **Result**: build-hash identical; MOVE_ONE_ITEM microbench **12.255 vs
  12.281 s = flat**. Resets only fire on rejection *failures* (rare — most random
  source/sink placements connect), and `new_world()` is cheap, so there's nothing
  to save.
- **Verdict**: **DROP (branch discarded).** Together with #2/#8, this conclusively
  shows **build-path micro-opts can't move the benchmark** — the whole build is
  ~11% of rollout and these touch a sub-slice of it. Stop optimizing the builder
  for speed; future effort goes to the per-step rollout, the update loop, and the
  monitorability diagnostics.

### speedup/build-hoist-constants — hoist build_factory loop-invariants — DROPPED
- **Hypothesis**: the build microbench (under cProfile) showed `str2ent` 0.162 s +
  `str2item` genexpr ~0.16 s self-time, and the rejection loops rebuild
  `[d for d in Direction…]` / `[v.value … items…]` every iteration. Hoist all to
  module constants (`DIRECTION_CHOICES`, `_ITEM_VALS_NONEMPTY`, `_TRANSPORT_BELT_VAL`,
  `_ELECTRONIC_CIRCUIT_VAL`, reuse `_SOURCE/_SINK_ENT_VAL`) → fewer linear scans /
  allocations. List order preserved → identical `random.choice` draws.
- **Change** (branch only): add constants; replace ~30 call sites across all
  lesson branches. Build-hash **identical**, lint/type clean.
- **Result**: build microbench **5.699 s vs 5.705 s = flat** (no measurable gain,
  not even on the build-only microbench). The cProfile self-times were inflated
  by per-call instrumentation — `str2ent` over a ~12-entry dict is a handful of
  comparisons and a few calls/build; hoisting saves nothing real.
- **Verdict**: **DROP (branch discarded).** ~30 changed call sites + 8 constants
  for zero measurable benefit = pure churn. Confirms (again) cProfile self-times
  mislead for tiny Python helpers. Not worth a full benchmark run.

### speedup/bfs-inline — inline in_bounds + cache dist in _bfs_shortest
- **Hypothesis**: `_bfs_shortest` (1.54 s self) + its nested `in_bounds` (0.56 s
  over 1.47M calls) are the #2/#3 build-microbench hotspots. Inline the bounds
  check (kill the per-neighbour function call), cache `dist[(r,c)]` once per
  expansion (was looked up 4×), and build each neighbour tuple once. Visit order
  and `parents` append order unchanged → identical paths (and identical
  `random.shuffle` consumption) → byte-identical factories.
- **Change**: rewrite the BFS expansion loop in `_bfs_shortest` (factorion.py).
- **Result**: build-hash **identical** to main across all lessons/seeds. Build
  microbench (2500 builds): **5.74 s vs 6.14 s = −6.5%** — real interior win.
  Full benchmark: **28.828 s ± 0.192 vs 28.870 → flat** (as expected). Signature
  **MATCHED ✓**. Commit `ab74d0b`.
- **Verdict**: **KEEP — merged (benchmark-flat, production-positive).** −6.5% on
  the build microbench is real and compounds over every reset in real training.

### speedup/async-vector-env — AsyncVectorEnv (multiprocess rollout) — DROPPED
- **Hypothesis**: the rollout is single-core-bound (1–2 of 48 cores). Run each of
  the 16 envs in its own worker process (`AsyncVectorEnv`) so env stepping fans
  out across cores. Gated behind `FACTORION_ASYNC_ENVS=1` (default off → CI /
  benchmark untouched). Open risk: per-step IPC overhead, and subprocess
  RNG/autoreset divergence breaking the signature.
- **Change** (branch only, not merged): env-var switch Sync↔Async; set the
  seed-march via `set_attr`; `AgentCNN` reads grid size from
  `single_observation_space` (works for both backends).
- **Result**: **BOTH worse.** (1) **Speed: 33.76 s ± 0.12 vs 28.59 s Sync →
  +18% SLOWER** (3-run hyperfine). User-time ballooned 27.7 s → 54.3 s: the
  workers do *more* total CPU work pickling 16 obs arrays + actions across the
  process boundary every one of 256 steps, and that IPC dwarfs the tiny per-env
  step (~0.5 ms). (2) **Signature DIFFERS** (policy_loss 0.332 vs 0.160 baseline)
  — Async autoreset/seeding diverges from Sync, so it isn't even a drop-in.
- **Verdict**: **DROP (branch discarded).** Confirms the hardware analysis:
  parallelizing across cores can't help when per-step work is far smaller than
  the IPC cost. Multiprocess env stepping would only pay off with much heavier
  per-step env compute (e.g. a real Factorio sim), not this fast Rust throughput
  call. Filed so nobody re-tries it without remembering the IPC wall.

### speedup/step-cpu-microopts — incremental num_placed_entities counter
- **Hypothesis**: a CPU-side (not GPU-sync) rollout win, since the rollout is
  CPU-bound. `FactorioEnv.step` recomputed `num_placed_entities` via
  `len([a for a in self.actions if ...])` — an O(steps) scan of the whole action
  history *every step* → O(steps²) per episode, and it grows as episodes lengthen
  during real training. Track it as an incremental counter (bump on each valid
  non-empty placement) → O(1)/step. Identical value (verified 0 mismatches over
  400 steps vs the old scan); signature unaffected (logged-only field).
- **Change**: `self._num_placed_entities` init at reset, `+= 1` at the valid
  placement site when the entity isn't 'empty'; `info` reads the counter.
- **Result**: **28.870 s ± 0.145 s** vs 28.964 → **flat** (−0.3%, noise; the
  benchmark's random policy gives short episodes so O(steps²) barely bites).
  Signature **MATCHED ✓**. Commit `4352fc0`.
- **Verdict**: **KEEP — merged (benchmark-flat, production-positive).** The
  asymptotic win shows up only when episodes are long, i.e. in real training as
  the policy improves — invisible to the short-episode benchmark.

### speedup/defer-entropy-syncs — accumulate policy/* entropy on-GPU
- **Hypothesis**: the rollout accumulated the per-head entropy + eot-prob for the
  `policy/*` logs via `float(e)` every step — 7 device→host CUDA syncs/step ×
  256 steps, purely for logging (these never feed the loss). Sum the GPU scalars
  on-device and convert to float once at log time → 8 syncs/iter, not 1792.
  Logging-only, so signature must MATCH; the logged value is identical up to
  float32-vs-float64 accumulation order (irrelevant to the gate).
- **Change**: `_head_ent_sum[h] = _head_ent_sum[h] + e` (no per-step `float`);
  `float(...)` only where `policy/entropy*` are emitted.
- **Result**: **28.964 s ± 0.235 s** vs 29.107 → **−0.5% headline** (borderline),
  but **interior `rollout_steady` 2.025 → 1.991 s/iter (−1.7%)** — a real per-call
  rollout win, the best interior signal of the sync-reduction attacks. Signature
  **MATCHED ✓**. Commit `c2919ae`.
- **Verdict**: **KEEP — merged.** First sync-reduction attack to nudge even the
  headline; the interior rollout metric confirms removing 1792 syncs/iter helped.
  Consistent with "rollout is CPU-bound" — this is the rare GPU-sync win that's
  partly visible because it removes 1792 CPU-side `float()` stalls/iter.

### speedup/batch-action-transfer — one device→host copy for the action
- **Hypothesis**: on GPU the rollout's `get_action_and_value` (1.7 s cum) and the
  per-step transfer of the sampled action are the per-step costs. The action was
  moved host-side via `{k: v.cpu().numpy() for k, v in action_ED.items()}` — six
  separate `.cpu()` calls, each forcing its own CUDA sync (~6 syncs/step × 256
  steps). The rollout already builds the stacked `action_EA` (B, 7) on GPU; copy
  *that* once and slice the columns on the host → 1 sync/step instead of 6.
  Action values handed to the env are identical (eot goes float→int64 but the env
  reads `int(action["eot"])`), so the signature must MATCH.
- **Change**: replace the six-way dict-comprehension transfer with a single
  `action_EA.cpu().numpy()` + column slicing into the same 6-key dict.
- **Result**: **29.107 s ± 0.079 s** vs 29.09 → **flat**. Signature **MATCHED ✓**.
  rollout_steady 2.035 → 2.025 (noise). Commit `a9df8ae`.
- **Verdict**: **KEEP — merged (benchmark-flat, production-positive).** Fewer
  syncs + cheaper code; matters more on a bigger net / more-utilized GPU. **Key
  learning: the rollout bottleneck is CPU-side Python (env stepping), NOT GPU
  syncs/transfers.** With the GPU at ~9% util, the CPU env-step work dominates and
  runs concurrently with GPU idle, so cutting CUDA syncs is invisible to the
  benchmark. Benchmark-visible wins must cut **CPU Python in the rollout** (the
  `SyncVectorEnv` 16-env loop / `FactorioEnv.step`) or parallelize it.

### speedup/path-to-belts-revmap — O(1) reverse-map in _path_to_belts
- **Hypothesis**: a focused (non-cProfile-distorted by relative ranking)
  microbench of `build_factory` shows the cost is NOT the torch ops in
  `find_belt_paths_*` (0.018 s self) but the pure-Python path enumeration:
  **`_path_to_belts` is the #1 self-time (2.17 s over 93k calls)**, then
  `_bfs_shortest` (1.54 s) and `in_bounds` (0.56 s, 1.47M calls). `_path_to_belts`
  scans `DIR_TO_DELTA.items()` (O(4) + tuple compare) per path step to find the
  matching direction. A precomputed `DELTA_TO_DIR` reverse map makes it an O(1)
  dict lookup. Every delta is unique so the result is identical; path *order* is
  preserved (critical — `find_belt_paths` output is fed to `random.shuffle`, so a
  reordering would change the chosen path → the factory → the signature).
- **Change**: add module-level `DELTA_TO_DIR = {delta: d}`; rewrite the
  `_path_to_belts` inner loop to `DELTA_TO_DIR.get((dr, dc))`.
- **Result**: **29.090 s ± 0.260 s** vs 29.096 baseline → **flat** (−0.02%, deep
  inside noise). Signature **MATCHED ✓** (first real invariance check against the
  GPU baseline — confirms pure-speed). rollout_steady 2.028 → 2.035 (noise).
  Commit `e60e6de`.
- **Verdict**: **KEEP — merged to `main` (benchmark-flat, production-positive).**
  Per the human's steer: a correct, signature-identical precalc that's strictly
  faster per call is worth keeping even when invisible to the 5-min benchmark,
  because the rollout runs millions of times in real training. **Key learning:
  the build/reset path is NOT a big share of the honest benchmark's rollout —
  cProfile badly overstated it** (per-call instrumentation tax inflates
  Python-heavy fns like `_path_to_belts`/`_bfs_shortest`). So further builder
  micro-opts won't move the *benchmark* much (they still help production).
  **For benchmark-visible wins, pivot to the per-step rollout loop (256 steps ×
  16 envs: NN forward + SyncVectorEnv stepping + host↔device transfers).**

### speedup/remove-entities-numpy — vectorize per-cell reads in _remove_entities
- **Hypothesis**: `_remove_entities` (2.32 s self-time, biggest self-time in our
  code) does two full W×H Python loops reading `world_CWH[ch, x, y].item()` per
  cell — ~250+ torch scalar reads/reset × 965 resets/2-iter run. torch `.item()`
  carries heavy per-op dispatch overhead vs a numpy scalar read. Reading the ENT
  and DIR channels as plain int numpy arrays once per call (the world tensor is
  CPU) and indexing those should erase most of the self-time. Iteration order
  (x outer, y inner) and the single `random.sample(entity_groups, k)` draw are
  kept byte-identical, so the sampled groups — and the built factory — are
  unchanged → invariance signature must MATCH (pure speed).
- **Change**: hoist empty/source/sink entity-ids + empty-item-id + NONE dir/misc
  to module constants (kill the per-cell `str2ent`/`str2item` linear scans); read
  ENT/DIR as int numpy arrays at the top of `_remove_entities`; index numpy in
  both passes; pass Python ints to `py_entity_tiles` (its stub wants `int`).
- **Result**: **29.096 s ± 0.144 s** vs GPU baseline 30.070 s ± 0.239 → **−3.2%**
  (ranges fully non-overlapping: 28.90–29.28 vs 29.84–30.34, so real not noise,
  though under the 5% heuristic). rollout_steady 2.095 → 2.028 s/iter (−3.2%);
  update_steady flat (0.611 → 0.613, rollout-only change as expected). Invariance
  signature established here (file was absent); blank-output hash verified
  byte-identical vs main across all lessons/seeds → pure-speed. Commit `fd9f8b0`.
- **Verdict**: **KEEP — merged to `main`.** Modest but clean; cProfile overstated
  the self-time (instrumentation tax on a Python-heavy fn), so the real win is
  smaller than the 2.32 s profile figure suggested. Compounds over 100s of iters.

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

### speedup/skip-baseinfo — also skip base _get_info() on non-terminal training
- **Hypothesis**: follow-on to skip-info-nonterminal — the base `_get_info()`
  (10-key dict) was still built + vector-aggregated every step. Nothing reads it
  on non-terminal training steps, so return `{}` there too.
- **Change**: move `info = self._get_info()` inside the
  `terminated or truncated or _full_diagnostics` branch; `else: info = {}`.
- **Result**: **23.436 s ± 0.202 vs 23.889 → −1.9%**. rollout_steady 1.362 →
  1.321 (−3.0%). Signature **MATCHED ✓**; tests pass. Commit `c4c3234`.
- **Verdict**: **KEEP — merged.** Last drop of the per-step info-dict tax.

---

**Running tally (GPU box):** CPU-era 37.09 s → GPU baseline 30.07 s (driver fix)
→ **23.44 s** after the pure-speed sweep below = **−22% on GPU / −36.8% vs CPU
era**, all signature-identical (no numerics touched). Biggest levers:
gate-diagnostics (−12%), info-skip (−3.5%), validity-numpy (−2.2%),
_remove_entities-numpy (−3.2%), skip-baseinfo (−1.9%), defer-entropy-syncs
(−0.5%). Dropped: AsyncVectorEnv (+18%, IPC-bound), build-path micro-opts
(flat — build is ~11% of rollout). Remaining lever is the NN forward (numerics /
AMP) — but GPU is ~9% utilized, so it won't help; not pursued.

---

# PPO time-to-quality campaign (113 → 36 s, `bench_* ppo-quality` → `quality_results.csv`)

A second PPO benchmark: wall-clock to finetune the cached SFT checkpoint
(`checkpoints/sft_j0s5y2mc.pt`, offline) up to EMA(`rollout/reward`) ≥ −0.15.
**Numerics-allowed** (LR/batch/precision may change the trajectory), so compare
across the 5 seeds, not a single run. `time_to_quality = crossing_iter ×
wall_per_iter`; the deterministic per-iter wall is the low-noise signal, and
crossing-iter is an unbiased ~25±4 draw, so `E[time_to_quality] ∝ per_iter`.

## WON (cumulative; all confirmed 5-seed, all on `main`)

- **Recipe: `--critic-warmup 5 --learning-rate 7e-4` → 62.9 s (−44%)**, now the
  `PPOArgs` defaults (lr, critic_warmup, target_kl).
  - *lr 7e-4 rationale (moved from the `PPOArgs.learning_rate` docstring):* the
    confirmed SFT→PPO finetune optimum — bigger policy steps converge in fewer
    iters AND hit the `--target-kl 0.02` ceiling sooner, so the update
    early-stops its epochs (cheaper per-iter too). lr 1e-3 overshoots.
  - *critic_warmup 5 (moved from `PPOArgs.critic_warmup`):* sweet spot — 0 is worse
    (a random critic's advantages wreck the SFT policy), 10 wastes dead iters;
    set 0 to disable for from-scratch runs.
  - *target_kl 0.02 (moved from `PPOArgs.target_kl`):* early-stops the update's
    epochs once the policy has moved enough — this is what makes the higher LR
    cheap per-iter as well as fast-converging. None = always run all epochs.
  - Deliberately did NOT bake gamma/gae_lambda/ent_coef into defaults — those are
    inherited j0s5y2mc values, never independently validated.
- **Fused categorical/bernoulli sampling** (drop `torch.distributions`): −21%
  per-iter, and the *enabler* for compile (the Distribution objects graph-broke
  `torch.compile`).
- **`torch.compile(reduce-overhead)` CUDA graphs on the rollout** (62.9→45.4 s):
  the policy forward is kernel-launch bound (B=16≈B=128 cost), and CUDA graphs
  replay the captured sequence with ~0 CPU launch overhead. The old
  `torch.compile(agent)` was a silent no-op (AgentCNN has no `forward()`).
- **CUDA graphs on the PPO update** fwd+bwd (45.4→41.4 s).
- **numpy world-writes in `FactorioEnv.step`** (41.4→36.0 s, bit-identical):
  *rationale (moved from the code comment):* write entity/direction/item/misc
  through the existing `world_np` view (it aliases the same CPU buffer), not via
  `self._world_CWH[ch,x,y]=v` — torch scalar indexed assignment carries ~7 µs of
  per-op dispatch each, so those 4 writes were ~half the step body; numpy scalar
  writes are ~40× cheaper and identical.
- **`functools.cache` on `str2ent`** (factorion): *rationale (moved from the code
  comment):* `entities` is built once at import and never mutated, so it's a pure
  function of `s`; it was a linear scan over `entities.values()` (with a
  `str.replace` per iteration) and one of the hottest calls in `build_factory`
  (~28k calls per rollout iter of resets). Honest note: cProfile overstated it
  (per-call tax); real wall-clock impact was ~0 — kept as clean hygiene.

## LOST / dead ends (measured, don't re-test)

- **More envs / bigger batch** (sync+async 32/64/128): GPU util rises to 87% and
  iters-to-converge drops, but rollout is 256 *sequential* env-steps and per-step
  cost grows with envs → net slower.
- **AsyncVectorEnv re-swept on the compiled code** (4→128 envs, single-seed):
  async loses to sync at *every* env count; best is 16-env sync (the default).
  Opposite of the pre-compile finding (async32 beat sync32) — once the forward is
  CUDA-graphed to ~0, AsyncVectorEnv's per-step IPC barrier is *exposed*.
- **AMP/bf16, `num_steps`<256, ent-coef 0, constant LR, `--critic-lr-mult`**: no
  win (see prior rows + the per-iter campaign).
- **Factory pre-build pool**: break-even (the parallel build won't scale — the
  main process un-pickles factory tensors serially).
- **Async-prefetch factory generation** (mp.Pool background workers, 96% hit):
  a wash (36.2 vs 36.0 s) — the `Pool` result-handler thread contends for the
  GIL with the single-core rollout, cancelling the reset saving.
- **Rust port of the BFS belt-routing**: BFS is only ~25% of `build_factory`
  (cProfile overstated it) — not worth it.

Box is single-core-rollout-bound: the wins were "make the GPU work disappear"
(CUDA graphs) + "kill per-op dispatch on the CPU path" (numpy writes), not
parallelism. The remaining wall is the Python `build_factory` reset path.

---

# SFT training speed (88.4 → 22.6 s, `bench_* sft` → `sft_bench_results.csv`)

Times the `sft.py` training loop to a fixed deterministic `val_loss = 1.6888`
(size 11, real 93-69-96 encoder, 60k samples × 10 epochs, rollout eval off).
**Pure-speed**, gated bit-identically on `val_loss` — every win below keeps it
exactly. `sft.py --dataset-cache` caches the generated `(state,action)` tensors
so the timed run skips the ~26 s `build_factory` data generation (a separate
axis) and measures training; it re-seeds after load/generate so a cached run is
bit-identical to a fresh one.

## WON (each bit-identical, cumulative)

- **GPU-resident training data (train AND val loops)** → the big win. The loops
  were data-path bound (8-9 host→device transfers per batch per epoch), NOT
  compute or sync bound (B=512 keeps the GPU busy). The dataset is tiny (~200 MB),
  so move it on-device ONCE and iterate a `DataLoader` over **indices**
  (`torch.arange`), not the data. *(Why indices, not a hand-rolled `randperm`:
  the index DataLoader reuses the real `RandomSampler` — same length + config →
  the same shuffle order AND the same per-epoch global-RNG draws — so the batch
  order, and thus val_loss, is identical. A manual `randperm` does NOT reproduce
  the DataLoader's 3 global-RNG draws/epoch and shifts val_loss; even the
  shuffle=False val loader still draws 2/epoch that the train shuffle depends on.)*
  train 7.4→1.5, val 0.8→0.2 s/epoch.
- **Sync-free epoch stats**: the loop did ~10 `.item()` GPU→CPU syncs per batch
  (7 losses + grad-norm + 2 accuracy counts); accumulate them on-GPU and convert
  ONCE per epoch. Neutral on its own (B=512 isn't sync-bound — the syncs overlap)
  but clean, and it matters once the data path no longer dominates.
- **int64 → GPU → float transfer**: obs is stored int64 (276 MB); `.to(device)`
  then `.float()` copies the smaller int tensor and casts on the GPU (5× faster,
  no 480 MB CPU float intermediate). Bit-identical (int→float of small ids exact).
- **Lazy-import matplotlib + networkx** in `factorion.py` (visualization-only) —
  ~0.5 s off every `import factorion`, helping all entry points' startup.

Net: 88.4 → 22.6 s (−74 %, 3.9×); the per-epoch cost (what matters for real
multi-hour runs) is 4.8× faster.

## LOST / dead ends (the conv fwd/backward is FLOORED)

The remaining ~15 s is the conv forward+backward (profiled: ~65% backward, ~23%
encoder forward). It can't be sped bit-identically, and tensor cores barely help:
- **`.item()`-sync removal alone**: neutral (B=512 isn't sync-bound). Kept anyway.
- **torch.compile** (42 s), **cudnn.benchmark** (32 s), **AMP/bf16** (net ~0),
  **TF32** (8% gross), **fused AdamW** (~0): every one either breaks the
  bit-identical invariant (different FP) or doesn't help end-to-end. At B=512 the
  11×11 conv is GPU-underutilized and partly **memory-bandwidth-bound** (bf16 > TF32
  at equal FLOPs proves it), so there's no FLOP headroom for tensor cores to take.
  An isolated encoder micro-bench shows "bf16 2.6×" but that's a
  per-step-`cuda.synchronize()` artifact — trust end-to-end, which says ~0.
  This is the OPPOSITE of PPO, where the launch-bound B=16 forward loved CUDA
  graphs. Don't re-try these here.
