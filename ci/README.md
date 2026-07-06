# Factorion CI

Everything CI lives in this directory. GPU training jobs run on
**fire-and-forget RunPod pods** and log to W&B; **PR comments are the
backbone of reporting**.

## Triggering jobs: comment `/ci ...` on a PR

`/ci help` posts this grammar with examples. Square brackets mark optional
flags (don't type the brackets).

```
/ci sft [--num-samples N]                  # SFT from scratch at the PR head
/ci ppo --start-from j0s5y2mc              # PPO from an SFT checkpoint
/ci compare sft [--seeds 3] [--num-samples N]   # PR head vs main, seed-paired
assert pr:val/thput > main:val/thput       # optional pass/fail conditions
assert pr:val/acc >= 0.5                   #   → commit status check
assert pr:sps == main:sps +- 100           # ~equal within a tolerance
/ci compare ppo --start-from j0s5y2mc      # PPO compare, same flow
/ci sweep sft [--pods 2] [--agents-per-pod 5]   # W&B sweep (ci/sweep_sft.yaml)
/ci sweep ppo                              # ... from ci/sweep_ppo.yaml
/ci pods                                   # list CI pods + cost + deadline
/ci kill --all                             # terminate CI pods (or: /ci kill <pod_id>)
/ci watchdog --dry-run                     # preview the leaked-pod reaper
/ci help                                   # usage + examples
```

What comes back as PR comments:

- reactions on your comment: **&#x1F440; instantly** (no eyes within ~30s
  means GitHub dropped the event — repost) and **&#x1F44D; when the command
  has run to completion** (for `compare`, that includes the report posting
  and its assertions passing);
- a **launch comment** immediately (pod ids, W&B links);
- for `sft`/`ppo`: a **result comment** with headline metrics when the run
  finishes (posted by the reporter cron, so runs longer than GitHub's 6h job
  limit still report);
- for `compare`: an **every-metric comparison** — headline metrics up front,
  the full table (100s of metrics, seed-paired t-tests, sorted by p-value)
  inside a `<details>` block — plus a `factorion-ci/compare` **commit
  status** that passes/fails your `assert` lines. Headline metrics are the
  `HEADLINE_PATTERNS` regexes in `ci/report.py` (thput_eot overall + per
  lesson, overall + per-head accuracies for SFT; eval/rollout/critic
  headliners for PPO) — edit freely;
- for sweeps: the ranked **sweep report** when the sweep drains its run_cap.

For jobs not tied to a PR (e.g. a production SFT run from main), use the
**Launch (manual)** `workflow_dispatch` in the Actions tab — same commands,
results in the job summary + W&B. The CLI also works locally
(`uv run python -m ci --help`; e.g. `uv run python -m ci compare sft --ref
my-branch`) if you export `RUNPOD_API_KEY` + `WANDB_API_KEY`, but GitHub is
the intended path.

## How a job runs

1. `ci/launch.py` resolves the commitish to a **pushed** SHA and creates a
   pod whose docker command is a tiny bootstrap, shipped base64-encoded in an
   env var (base64 sidesteps the quoting hazards of docker-args).
2. The bootstrap clones the repo at that SHA and runs `ci/runner.sh`, which
   builds the Rust extension and hands off to `python -m ci.jobs`.
3. `ci/jobs.py` decodes the job spec (`FCI_JOB_B64`) and runs the training
   command. This is the **only** place CI training commands are built: they
   contain `--track`, tags, and the few whitelisted overrides — every other
   hyperparameter comes from `training_config.py`. If a knob isn't in
   `ci/config.py`'s job specs, CI cannot change it.
4. The pod **terminates itself** when the job ends, however it ends.

A `compare` fans out into 2 × seeds pods — **one training run per pod**, so
seeds never compete for CPU — grouped in W&B by role (`cmp-<sha7>-test` /
`cmp-<sha7>-base`); the report is assembled from W&B afterwards.

W&B runs are tagged `ci`, `kind:<sft|ppo>`, `sha:<sha7>`, `pr:<num>` (and
`cmp:<sha7>` + `cmp-role:<test|base>` for compare runs). Raw logs are in the
RunPod console (the pod's container logs; the job also tees to
`/workspace/job.log`).

## Pods can't leak (three layers)

1. **EXIT trap** in the bootstrap terminates the pod when the job finishes,
   fails, or crashes.
2. **In-pod deadline timer** (`ci/runner.sh`) hard-kills the pod at the
   deadline computed from the job size, even if the job wedges.
3. **`pod-watchdog.yml`** runs `python -m ci.watchdog` every 6 hours: every
   CI pod's name encodes its creation time and deadline
   (`factorion-ci-<kind>-c<epoch>-d<epoch>-<sha7>`), so the watchdog can reap
   pods that never even booted. There's also a 48h absolute cap. Pods not
   named `factorion-ci-*` are never touched.

## Files

- `gh_command.py` — the `/ci` PR-comment dispatcher (parse → launch → report).
- `cli.py` / `__main__.py` — the CLI (`uv run python -m ci --help`).
- `config.py` — infrastructure knobs (image, GPU lineup, budgets, pod naming)
  and the job specs (= the complete CI override surface).
- `launch.py` — ref resolution, bootstrap, pod creation, compare fan-out,
  sweep creation.
- `jobs.py` — pod-side dispatcher; builds and runs the training commands.
- `runner.sh` — pod-side setup (Rust build, deadline timer, hand-off).
- `watchdog.py` — leaked-pod reaper (needs only `pip install runpod`).
- `report.py` — every-metric compare report + assertions, per-run PR
  summaries, sweep report, history CSV (`python -m ci history`).
- `github_api.py` — PR comments + commit statuses.
- `stats.py` — dependency-free paired/Welch t-tests.
- `sweep_ppo.yaml` / `sweep_sft.yaml` — W&B sweep configs.

## GitHub Actions (thin pointers into this directory)

- `ci.yml` — lint + Python tests + Rust tests on PRs. No GPU jobs.
- `ci-command.yml` — the `/ci` comment dispatcher (collaborators only).
- `ci-reporter.yml` — 30-min cron posting result comments for finished runs.
- `pod-watchdog.yml` — the 6-hourly leaked-pod reaper.
- `launch.yml` — manual `workflow_dispatch` bridge to the same CLI.

Secrets used: `RUNPOD_API_KEY`, `WANDB_API_KEY` (plus the automatic
`GITHUB_TOKEN`).
