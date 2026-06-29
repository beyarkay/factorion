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
- **Result**: iter-1 signature **identical** to baseline. Benchmark: TBD.
- **Verdict**: TBD.

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
