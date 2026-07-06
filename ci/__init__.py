"""Factorion CI: launch and manage GPU training jobs on RunPod.

Everything CI-related lives in this directory. The entry point is the CLI:

    uv run python -m ci --help

Jobs are fire-and-forget: `ci/launch.py` creates a pod whose docker command
clones the repo at the requested commit and runs `ci/runner.sh`, which hands
off to the job dispatcher (`ci/jobs.py`). The pod terminates itself when the
job finishes; results live in W&B. No GitHub Actions runner has to stay alive.

Keep this module import-light (stdlib only) — `python -m ci.watchdog` runs in
a GitHub cron job with only `runpod` installed.
"""
