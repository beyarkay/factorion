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
     `quality_measure.sh` uses `hyperfine --parameter-list seed 1,2,3,4,5`.
     Measured baseline across 5 seeds: crossing iter `[33,32,32,32,34]`
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
