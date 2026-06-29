# Time-to-quality benchmark (numerics-allowed)

`run.sh`/`measure.sh` time a **fixed iteration count** and forbid changing the
computation — good for pure-speed work, useless for changes that trade exactness
for speed (precision, batch size, LR, env count). This benchmark instead times
**how long PPO takes to reach a fixed policy quality**, starting from the
canonical SFT checkpoint. Anything that gets there faster wins — including
numeric changes.

## The measuring stick

```bash
./quality_run.sh                 # one finetune run; stops at the quality threshold
./quality_measure.sh "<note>"    # hyperfine --runs 5 (no warmup) -> ../quality_results.csv
```

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
- **Why it's repeatable despite noisy RL:** fixed `seed=1` + `torch_deterministic`
  make the training trajectory deterministic, so the **crossing iteration is
  identical** run-to-run; only per-iteration wall-time jitters. Measured
  back-to-back: iter 33 / EMA −0.1319 both times, wall 114.55 s vs 114.46 s
  (0.08%). That is why `hyperfine --runs 5 --warmup 0` gives a tight number.

## Offline

Fully offline: `WANDB_MODE=disabled`, `eval-every 0`, and `--start-from` points
at the on-disk SFT checkpoint `checkpoints/sft_j0s5y2mc.pt` (cached from W&B run
`j0s5y2mc` via `_resolve_wandb_checkpoint`; no network at run time). No wandb
logging or network in the timed path.

## Sweeping a knob

`quality_run.sh` reads overrides from the environment, so experiments don't edit
the script:

```bash
LR=3e-4 ./quality_measure.sh "lr 3e-4"
NUM_ENVS=32 NUM_MINIBATCHES=64 ./quality_measure.sh "envs 32"
EXTRA_ARGS=... # add a flag (e.g. --amp) once it exists, then thread it through quality_run.sh
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
