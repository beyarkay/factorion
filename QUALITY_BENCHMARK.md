# Time-to-quality benchmark (numerics-allowed)

`run.sh`/`measure.sh` time a **fixed iteration count** and forbid changing the
computation — good for pure-speed work, useless for changes that trade exactness
for speed (precision, batch size, LR, env count). This benchmark instead times
**how long PPO takes to reach a fixed policy quality**, starting from the
canonical SFT checkpoint. Anything that gets there faster wins — including
numeric changes.

## The measuring stick

```bash
./quality_run.sh                              # one finetune run; stops at the threshold
./quality_run.sh --learning-rate 3e-4 --num-envs 32   # any flag overrides the recipe
NOTE="lr 3e-4" ./quality_measure.sh --learning-rate 3e-4   # sweep seeds, log a row
```

`quality_run.sh` forwards every flag to `ppo.py` (tyro takes the last value, so
the script's defaults are overridden) — no env-var plumbing. `quality_measure.sh`
forwards its flags to `quality_run.sh` and annotates the row via `NOTE=`.

- **Quality metric:** EMA of `rollout/reward` (`--target-metric`,
  `--quality-ema-alpha`). Reward is logged every iteration (no eval cost) and is
  **comparable across the allowed knobs** — num_envs / num_steps / num_minibatches
  / update_epochs / learning_rate / numeric precision don't change the reward
  scale. (Only `step_penalty` / `throughput_reward_scale` would; hold those
  fixed.) `eval/thput_eot` is the "true" quality metric but starts at the SFT
  base (~0.14), barely moves for ~45 min, is noisier, and needs an extra forward
  pass — a bad benchmark target. You *can* point `--target-metric` at it for a
  slower, truer run.
- **Threshold:** `--target-value -0.15` (EMA). On the current box this is reached
  at **iter 33 / ~115 s** from the SFT checkpoint — deep in the learning phase
  (good discrimination) and a monotonic crossing (robust). Override with
  `TARGET_VALUE=...`.
- **Early stop:** `ppo.py` stops the first time the EMA crosses the threshold and
  records `time_to_quality_seconds` in the summary JSON. `--max-seconds` caps a
  stuck/regressing run so the benchmark can't hang (it then reports
  `reached_quality=false`).
- **Repeatability vs the RL noise — two layers:**
  1. *Same config, repeated:* fixed seed + `torch_deterministic` ⇒ the trajectory
     (hence the crossing iteration) is **identical** run-to-run; only wall-jitter
     varies (iter 33 / EMA −0.1319 both repeats; wall 114.55 vs 114.46 s, 0.08%).
     So repeating an *identical* command measures nothing useful.
  2. *Across configs:* changing batch size / precision resamples the trajectory
     (FP reduction order, gradients), so a single seed is one draw from a noisy
     distribution. Therefore **sweep the seed**, not identical reruns.
     `quality_measure.sh` loops seeds 1..5 (one deterministic run each — hyperfine
     is avoided as it requires ≥2 runs/command, wasteful when each seed is
     deterministic). Measured baseline across 5 seeds: crossing iter `[33,32,32,32,34]`
     (32.6 ± 0.8), time-to-quality **113.4 s ± 3.5 s (3.1%)**. The threshold sits
     on the steep part of the reward climb, so trajectory noise maps to
     sub-iteration time noise — tight. Rule of thumb: an improvement bigger than
     ~7 s (≈2σ) is real; smaller is in the seed noise (one seed alone can be ~8%
     off — seed 4 read 106 s vs ~115 s for the rest).

## Offline

Fully offline: `WANDB_MODE=disabled`, `eval-every 0`, and `--start-from` points
at the on-disk SFT checkpoint `checkpoints/sft_j0s5y2mc.pt` (cached from W&B run
`j0s5y2mc` via `_resolve_wandb_checkpoint`; no network at run time). No wandb
logging or network in the timed path.

## Sweeping a knob

Pass `ppo.py` flags straight through (they override the recipe defaults):

```bash
NOTE="lr 3e-4"        ./quality_measure.sh --learning-rate 3e-4
NOTE="envs 32"        ./quality_measure.sh --num-envs 32 --num-minibatches 64
NOTE="bf16"           ./quality_measure.sh --amp        # once an AMP flag exists
SEEDS=1,2,3           ./quality_measure.sh              # fewer seeds = faster, noisier
```

## Results log

`../quality_results.csv` (one dir up, survives branch switches): branch, commit,
hyperfine wall mean/std/min/max, the deterministic in-process
`time_to_quality_s`, `crossing_iter`, `quality_ema_final`, target, num_envs,
note. The headline is **`time_to_quality_s`** (startup-free, deterministic);
`wall_mean_s` = startup + that.

## Caveats

- `--max-seconds` must exceed the slowest config you expect, or it'll cap a
  legitimately-slower-but-still-working config as "did not reach".
- If you change a knob that *does* shift the reward scale (step_penalty,
  throughput_reward_scale, reward shaping), `rollout/reward` is no longer
  comparable — switch the target to `eval/thput_eot` (scale-free) for that sweep.
- The ~38 s critic-warmup (10 iters, actor frozen) is fixed overhead inside every
  run; it's part of the real SFT→PPO recipe, so it stays.

## Findings: driving down time-to-quality (baseline 113 s / iter 32.6)

Sweeps on the SFT→PPO recipe (each confirmed across 5 seeds unless noted).
`time_to_quality = crossing_iter × wall_per_iter`, so the levers are "fewer
iters to converge" and "faster iters".

**WON — `--critic-warmup 5 --learning-rate 7e-4` → 63.1 s ± 9.8 (−44%).**
- **LR is the dominant lever.** lr 1.6e-4→5e-4→7e-4 cut time 113→81→72 s (seed 1);
  it helps twice: bigger policy steps converge in fewer iters AND hit the
  `--target-kl 0.02` ceiling sooner, so the update early-stops its epochs (cheaper
  per-iter). lr 1e-3 overshoots (needs more iters) unless paired with warmup5.
- **critic-warmup 10→5** = −10% (warmup is dead time — actor frozen, reward flat).
  warmup 0 is *worse* (117 s): the random critic's bad advantages hurt the SFT
  policy. Sweet spot ≈ 5.
- Tradeoff: aggressive LR raises seed-to-seed variance (±9.8 s vs ±3 s baseline),
  but all seeds still reach quality and the mean is far below baseline.

**LOST / dead ends (measured, not guessed):**
- **More envs / bigger batch (sync & async, 32/64/128).** Drives GPU util 9%→87%
  and cuts iters-to-converge (33→13), but PPO's rollout is 256 *sequential*
  env-steps and per-step round-trip cost (CPU/IPC) grows with envs faster than
  the iteration savings → net slower (124 s+; 64/128 don't finish in 300 s).
  AsyncVectorEnv helps vs Sync at 32 envs (124 vs 147 s) but still loses to the
  16-env baseline. The box is rollout-sequential-bound, not GPU-bound.
- **AMP / bf16** on the best recipe: 83 s vs 63 s. bf16's precision loss shifts the
  trajectory to need more iters, and the GPU isn't the bottleneck so the per-iter
  speedup never lands. Net-negative here.
- **Lower `num_steps`** (128/64): no gain, and ≤64 *breaks the benchmark* — it
  shifts the reward distribution so −0.15 is met at iter 1 (reward-scale
  comparability, as warned). Keep num_steps = 256.
- **ent-coef 0 / constant LR**: no improvement over the recipe defaults.
- **Higher critic LR (`--critic-lr-mult`) to shorten warmup**: a wash — warmup2 +
  mult5 matches warmup5 (53.5 vs 53.8 s) but doesn't beat it. The bottleneck is
  the actor reaching the reward threshold, not how fast the critic warms.
- **step() pure-speed simplify** (lazy invalid_reason, hoisted lookups): correct +
  signature-identical, but a sub-1% slice of time-to-quality (kept anyway).

Takeaway: on this CPU-rollout-bound box the win is **train smarter (higher LR +
shorter warmup), not bigger** — the GPU/batch levers and numerics don't pay off.

### Full sweep log (so nobody re-runs these)

Single-seed (seed 1) screens unless marked "5-seed". Baseline seed-1 = 115.6 s /
iter 33; baseline 5-seed = 113.2 s ± 3.1. Lower is better. "DNF" = did not reach
−0.15 within `--max-seconds 300`.

LR (warmup 10): 1.6e-4→115.6 · 3e-4→92.7 · 5e-4→81.4 · 6e-4→71.1 · **7e-4→72.0** ·
8e-4→73.9 · 1e-3→93.5(iter38, overshoots).
Warmup (lr 1.6e-4): 10→115.6 · 5→104.5 · 3→106.3 · 0→117.6(worse, bad critic).
Combos: w5+lr5e-4→82.4 · **w5+lr7e-4→53.8 (seed1) / 62.7±9.8 (5-seed) ← BEST** ·
w7+lr7e-4→72.7 · w5+lr1e-3→74.5 · w3+lr1e-3→78.1.
On w5+lr7e-4: ent0→57.8 · no-anneal→63.4 · ent0+no-anneal→69.9 (none beat it).
critic-lr-mult (lr7e-4): w2,mult5→53.5 · w3,mult5→59.9 · w3,mult3→63.1 ·
w5,mult3→93.0 (wash — none beat w5+lr7e-4).
num-envs: sync32→147(iter25) · async32→124(iter25) · sync64→DNF · async64→DNF ·
async128→DNF(GPU 87%). num_steps: 128→63.9(iter48, no gain) · 64→INVALID(iter1,
reward-scale break) · 32envs×64steps→INVALID. AMP bf16 (5-seed): 83.0±14.

Confirmed (5-seed, in `quality_results.csv`): baseline 113.2±3.1 · w5+lr7e-4
62.7±9.8 · w5+lr7e-4+AMP 83.0±14.

## Findings: per-iter compute campaign (62.9 → 36.0 s, the w5+lr7e-4 recipe now baked into Args)

Once the recipe was fixed, the lever became **faster iters** (per-iter wall), not
fewer. Per-iter wall is measured *deterministically* (same seed, fixed 15-30 iter
slice → `Loop time breakdown` steady means): it is low-noise, unlike
time-to-quality whose ±15-23% crossing-iter variance swamps a single per-iter
win. crossing-iter is unbiased by a pure-speed change, so
`E[time_to_quality] ∝ per_iter_wall` — drive per-iter, validate with 5 seeds.

Profiling the (compiled) rollout: NN forward, env-step body, and the
reset/`build_factory` path each matter; the policy forward was **kernel-launch
bound** (a B=16 and a B=128 forward cost the same ~2 ms — pure CPU dispatch).

**WON (each confirmed 5-seed, cumulative; all on `main`):**
- **Fused categorical/bernoulli** (drop `torch.distributions`; `log_softmax`+
  `multinomial`+`gather`/`BCE` directly): −21 % per-iter. Also the *enabler* for
  compile — the Distribution objects graph-broke `torch.compile`.
- **CUDA-graph compile of the rollout** (`torch.compile(reduce-overhead)` on the
  no_grad `get_action_and_value`/`get_value`): 62.9 → **45.4 s**. The old
  `torch.compile(agent)` was a silent no-op (AgentCNN has no `forward()`).
  Needs `cudagraph_mark_step_begin()` per call + no retained outputs; dropped
  `_arange_cache` (a cached arange aliases graph memory).
- **CUDA-graph compile of the PPO update** (fwd+bwd, separate handle): 45.4 →
  **41.4 s** (update −47 % per-iter). reduce-overhead spans the autograd backward;
  the warm-up requires_grad flip recompiles once.
- **numpy world-writes in `step()`** (write the world through the existing
  `world_np` view, not torch scalar indexing — ~40× cheaper, bit-identical):
  41.4 → **36.0 s**, trajectory-identical.

Net: **113 → 62.9 → 36.0 s** (−68 % overall, −43 % vs the recipe baseline).

**LOST / dead ends (measured this campaign):**
- **Pre-build a factory pool** (parallel `build_factory` at startup, draw on
  reset): pool-draw resets are 4× cheaper, but the parallel build doesn't scale
  (main process serially un-pickles the factory tensors; even build-and-discard
  peaks at ~2×, COW/memory-bound), so ~3.5 s build ≈ ~3.7 s saved over ~26 iters
  → break-even for the benchmark. (A real win for *long* training.)
- **Async-prefetch factory generation** (background `mp.Pool` workers pre-build
  full reset bundles, env draws by seed; 96 % cache-hit, deterministic & correct):
  a **wash** — 36.2 s vs 36.0 s. The ~316 ms/iter reset saving is cancelled by
  ~400 ms/iter of multiprocessing overhead, because the box is
  single-core-rollout-bound and the Pool's result-handler thread contends for the
  GIL with the rollout loop. Confirms the env-scaling/async dead-end above.
- **Rust port of the BFS belt-routing** in `build_factory`: honest wall-clock
  shows `_bfs_shortest`+`_path_to_belts` are only ~25 % of build (cProfile's
  per-call tax overstated them), so ~25 ms/iter — not worth it. The other ~75 %
  is distributed generation work with no fat target.
- **str2ent memoization** (`functools.cache`): negligible (~0.04 ms/reset; cProfile
  overstated it). Kept — correct and removes an O(n) scan.

Takeaway: the win was **make the GPU work disappear** (CUDA graphs on a
launch-bound forward) + **kill per-op dispatch in the hot CPU path** (numpy
writes). The remaining wall is the reset/`build_factory` Python generation, which
the box's single-core nature makes hard to parallelize away.

### Untried / next ideas (for a future session)
The clean per-iter levers are spent. Remaining angles, all harder:
a **full** `build_factory` Rust port (the whole generation core, not just BFS —
big, but the only clean attack on the ~316 ms/iter reset wall); a lower-overhead
async prefetch (raw `Pipe`/`SimpleQueue`, no `Pool` helper threads — might beat
the wash, low confidence given the box); a stronger SFT base (raises the start
point); or chasing crossing-iter by re-tuning lr/warmup/minibatches on the
now-compiled code (noisy to validate). The metric could also be made
sustained-crossing (K consecutive evals above threshold) to harden it.
