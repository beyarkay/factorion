# Factorion CI

Everything CI lives in this directory. GPU training jobs are launched with one
CLI, run on **fire-and-forget RunPod pods**, and log to W&B — no GitHub
Actions runner stays alive during training, no labels, no 6-hour cap.

```bash
export RUNPOD_API_KEY=...   # https://www.runpod.io/console/user/settings
export WANDB_API_KEY=...    # https://wandb.ai/authorize

# SFT from scratch at a commitish (branch / tag / SHA — must be pushed)
uv run python -m ci sft --ref my-branch
uv run python -m ci sft --ref my-branch --num-samples 5000000

# PPO from an SFT checkpoint (W&B run id) — identical flow to sft
uv run python -m ci ppo --ref main --start-from j0s5y2mc

# Hyperparameter sweeps (config: ci/sweep_{sft,ppo}.yaml at that commit)
uv run python -m ci sweep-sft --ref my-branch --pods 2 --agents-per-pod 5
uv run python -m ci sweep-ppo --ref main

# Compare a branch's SFT vs origin/main over N seeds, then diff EVERY logged
# metric (paired t-tests, sorted by significance)
uv run python -m ci compare --ref my-branch --seeds 3
uv run python -m ci compare-report --ref my-branch          # rerun anytime

# Pod management
uv run python -m ci pods            # list CI pods + cost + deadline
uv run python -m ci kill <pod_id>   # or --all
uv run python -m ci watchdog        # what the 6-hourly GH cron runs

# Reports / history
uv run python -m ci sweep-report --sweep entity/factorion/<sweep_id>
uv run python -m ci history        # regenerate ci/history.csv from W&B
```

Every launch command supports `--dry-run` (print what would happen) and
`--gpu-type` (defaults to the RTX 2000 Ada, with automatic fallbacks).

## How a job runs

1. `ci/launch.py` resolves your `--ref` to a **pushed** SHA and creates a pod
   whose docker command is a tiny bootstrap (shipped base64-encoded in an env
   var — no scp, no ssh session).
2. The bootstrap clones the repo at that SHA and runs `ci/runner.sh`, which
   builds the Rust extension and hands off to `python -m ci.jobs`.
3. `ci/jobs.py` decodes the job spec (`FCI_JOB_B64`) and runs the training
   command. This is the **only** place CI training commands are built: they
   contain `--track`, tags, and the few whitelisted overrides — every other
   hyperparameter comes from `training_config.py`. If a knob isn't in
   `ci/config.py`'s job specs, CI cannot change it.
4. The pod **terminates itself** when the job ends, however it ends.

Progress lives in W&B (runs are tagged `ci`, `fci:<kind>`, `sha:<sha7>`);
raw logs are in the RunPod console, or `ssh` to the pod and read
`/workspace/job.log`.

## Pods can't leak (three layers)

1. **EXIT trap** in the bootstrap terminates the pod when the job finishes,
   fails, or crashes.
2. **In-pod deadline timer** (`ci/runner.sh`) hard-kills the pod at the
   deadline computed from the job size, even if the job wedges.
3. **`pod-watchdog.yml`** runs `python -m ci.watchdog` every 6 hours: every CI
   pod's name encodes its creation time and deadline
   (`fci-<kind>-c<epoch>-d<epoch>-<sha7>`), so the watchdog can reap pods that
   never even booted. There's also a 48h absolute cap. Pods not named `fci-*`
   are never touched.

## Files

- `cli.py` / `__main__.py` — the CLI (`uv run python -m ci --help`).
- `config.py` — infrastructure knobs (image, GPU lineup, budgets, pod naming)
  and the job specs (= the complete CI override surface).
- `launch.py` — ref resolution, bootstrap, pod creation, sweep creation.
- `jobs.py` — pod-side dispatcher; builds and runs the training commands.
- `runner.sh` — pod-side setup (Rust build, deadline timer, hand-off).
- `watchdog.py` — leaked-pod reaper (needs only `pip install runpod`).
- `report.py` — every-metric compare report, sweep report, history CSV.
- `stats.py` — dependency-free paired/Welch t-tests.
- `sweep_ppo.yaml` / `sweep_sft.yaml` — W&B sweep configs.

## GitHub Actions

Workflows are thin pointers into this directory:

- `ci.yml` — lint + Python tests + Rust tests on PRs. No GPU jobs.
- `pod-watchdog.yml` — the 6-hourly reaper.
- `launch.yml` — optional `workflow_dispatch` bridge that just calls this CLI
  (for kicking off jobs from the GitHub UI / phone); it exits as soon as the
  pod is created.
