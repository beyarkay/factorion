"""Launch fire-and-forget CI jobs on RunPod pods.

The pod's docker command decodes a base64-embedded bootstrap script that
clones the repo at the requested commit, runs ci/runner.sh, and terminates the
pod on exit (success, failure, or crash — the EXIT trap always fires). Nothing
on the launching machine needs to stay alive; results land in W&B.

Required env vars: RUNPOD_API_KEY, WANDB_API_KEY.
"""

from __future__ import annotations

import base64
import json
import os
import secrets
import string
import subprocess
import time
from typing import Optional

from ci.config import (
    WANDB_PROJECT,
    Job,
    compare_fanout,
    job_to_dict,
    pod_name,
)
from ci.config import REPO_URL as DEFAULT_REPO_URL

# Phase-1 bootstrap, shipped to the pod as FCI_BOOT_B64 (base64 dodges the
# quoting hazards of RunPod's docker-args plumbing). Static on purpose: all
# per-job values arrive via env vars, never via string templating.
#
# Container logs vanish when the pod terminates itself, so a failing
# bootstrap uploads its log to W&B (run tagged `boot-failure`) first —
# otherwise pod-side failures would be invisible.
BOOTSTRAP = """\
set -euo pipefail
mkdir -p /workspace
exec > >(tee -a /workspace/boot.log) 2>&1
on_exit() {
    rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "[fci] job failed (exit ${rc}); uploading boot log to W&B"
        FCI_RC="$rc" python - << 'PY' || true
import os

import wandb

run = wandb.init(
    project=os.environ.get("FCI_WANDB_PROJECT", "factorion"),
    name=f"boot-failure-{os.environ.get('RUNPOD_POD_ID', 'unknown')}",
    tags=["ci", "boot-failure", f"sha:{os.environ.get('FCI_SHA', 'unknown')[:7]}"],
)
run.summary["exit_code"] = int(os.environ.get("FCI_RC", "1"))
if os.path.exists("/workspace/boot.log"):
    wandb.save("/workspace/boot.log", base_path="/workspace")
run.finish(exit_code=1)
PY
    fi
    echo "[fci] terminating pod ${RUNPOD_POD_ID:-unknown}"
    python -c 'import os, runpod; runpod.api_key = os.environ["RUNPOD_API_KEY"]; runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])' || true
}
trap on_exit EXIT
echo "[fci] cloning ${FCI_REPO_URL} @ ${FCI_SHA}"
cd /workspace
git clone --quiet "${FCI_REPO_URL}" factorion
cd factorion
git checkout --quiet "${FCI_SHA}"
bash ci/runner.sh
"""

# The pod's docker command. Single quotes only — the RunPod SDK interpolates
# this string into a GraphQL query, so it must not contain double quotes.
DOCKER_ARGS = "bash -c 'echo $FCI_BOOT_B64 | base64 -d | bash'"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def resolve_ref(ref: str) -> str:
    """Resolve a commitish to a full SHA that exists on origin.

    Branch/tag names resolve against origin directly (`git ls-remote`), so you
    always get what's pushed, not a stale local ref. Raw SHAs are verified to
    be reachable from some origin branch — a pod can only clone pushed code.
    """
    out = subprocess.run(
        ["git", "ls-remote", "origin", ref, f"refs/heads/{ref}", f"refs/tags/{ref}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines():
        sha, _, refname = line.partition("\t")
        if refname in (f"refs/heads/{ref}", f"refs/tags/{ref}", ref):
            return sha
    if out.splitlines():
        return out.splitlines()[0].split("\t")[0]

    # Not a remote ref name — treat as a local commitish and verify it's pushed.
    try:
        sha = _git("rev-parse", "--verify", f"{ref}^{{commit}}")
    except subprocess.CalledProcessError:
        raise SystemExit(f"error: '{ref}' is neither a ref on origin nor a local commit")
    if not _git("branch", "-r", "--contains", sha):
        raise SystemExit(
            f"error: commit {sha[:12]} exists locally but not on origin — push it first"
        )
    return sha


def launch(
    job: Job,
    gpu_type: str,
    dry_run: bool = False,
    wait: bool = True,
    repo_url: str = DEFAULT_REPO_URL,
) -> dict:
    """Create the pod for `job` and (optionally) wait for it to boot.

    Returns {"pod_id", "pod_name", "deadline", "job"} (pod_id None on dry-run).
    """
    now = int(time.time())
    deadline = now + job.budget_seconds()
    name = pod_name(job.KIND, now, deadline, job.sha)
    spec = job_to_dict(job)
    # Mint the W&B run id here so the run URL is known (and linkable from the
    # PR) before the pod boots. Travels as env, not in the job spec, so pods
    # running older commits simply ignore it. Sweep agents mint their own ids.
    wandb_run_id = (
        "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        if job.KIND in ("sft", "ppo")
        else None
    )
    info = {
        "pod_id": None,
        "pod_name": name,
        "deadline": deadline,
        "job": spec,
        "wandb_run_id": wandb_run_id,
    }

    env = {
        "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "FCI_BOOT_B64": base64.b64encode(BOOTSTRAP.encode()).decode(),
        "FCI_JOB_B64": base64.b64encode(json.dumps(spec).encode()).decode(),
        "FCI_REPO_URL": repo_url,
        "FCI_SHA": job.sha,
        "FCI_DEADLINE": str(deadline),
        "FCI_WANDB_PROJECT": WANDB_PROJECT,
    }
    if wandb_run_id:
        env["FCI_WANDB_RUN_ID"] = wandb_run_id

    print(f"Job:      {spec}")
    print(f"Pod name: {name}")
    if wandb_run_id:
        print(f"W&B run:  {wandb_run_id} (URL live once the pod starts logging)")
    print(f"GPU:      {gpu_type} (with fallbacks)")
    print(f"Deadline: {time.strftime('%Y-%m-%d %H:%M:%S %z', time.localtime(deadline))} "
          f"({job.budget_seconds() // 60} min budget; watchdogs kill the pod after this)")

    if dry_run:
        print("--dry-run: not creating a pod.\n")
        return info

    for var in ("RUNPOD_API_KEY", "WANDB_API_KEY"):
        if not env[var]:
            raise SystemExit(f"error: {var} must be set in the environment")

    from ci import runpod_api

    pod = runpod_api.create_pod(name=name, gpu_type=gpu_type, docker_args=DOCKER_ARGS, env=env)
    pod_id = pod["id"]
    info["pod_id"] = pod_id

    print(f"\nPod {pod_id} created. The job runs unattended and the pod")
    print("terminates itself when done. Track progress:")
    print(f"  W&B:    https://wandb.ai/ (project {WANDB_PROJECT}, tag sha:{job.sha[:7]})")
    print(f"  RunPod: https://www.runpod.io/console/pods (container logs for {name})")
    print(f"  CLI:    uv run python -m ci pods   |   uv run python -m ci kill {pod_id}")

    if not wait:
        return info
    try:
        status = runpod_api.wait_until_running(pod_id)
        gpu = status.get("machine", {}).get("gpuDisplayName", "unknown")
        print(f"Pod is running on {gpu}.")
    except KeyboardInterrupt:
        print("\nStopped waiting — the pod keeps running (check `python -m ci pods`).")
    except TimeoutError:
        print("Pod failed to boot in time; terminating it.")
        runpod_api.terminate_with_retry(pod_id)
        raise SystemExit(1)
    return info


def launch_compare(
    algo: str,
    sha: str,
    base_sha: str,
    seeds: int,
    num_samples: int,
    start_from: Optional[str],
    total_timesteps: Optional[int],
    gpu_type: str,
    dry_run: bool = False,
    extra_tags: Optional[list[str]] = None,
) -> list[dict]:
    """Fan a compare out into 2 x seeds single-run pods (one run per pod, so
    seeds never compete for CPU). Returns the launch info of every pod."""
    jobs = compare_fanout(
        algo=algo,
        sha=sha,
        base_sha=base_sha,
        seeds=seeds,
        num_samples=num_samples,
        start_from=start_from,
        total_timesteps=total_timesteps,
        extra_tags=extra_tags,
    )
    infos = []
    for job in jobs:
        # Never block on boot: 2 x seeds pods launch back-to-back.
        infos.append(launch(job, gpu_type, dry_run=dry_run, wait=False))
    return infos


def read_sweep_config(algo: str, sha: str) -> dict:
    """Load ci/sweep_{algo}.yaml as it exists AT THE COMMIT (`git show`, not
    the working tree), so the sweep is true to the commit being swept."""
    import yaml

    try:
        raw = _git("show", f"{sha}:ci/sweep_{algo}.yaml")
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"error: ci/sweep_{algo}.yaml does not exist at {sha[:12]} — "
            "sweeps need a commit that contains the ci/ directory"
        )
    return yaml.safe_load(raw)


def sweep_summary_line(config: dict) -> str:
    """One line describing a sweep config: metric, run cap, swept params."""
    metric = config.get("metric", {})
    params = config.get("parameters", {})
    return (
        f"{metric.get('goal', '?')} `{metric.get('name', '?')}` over "
        f"{len(params)} parameter(s), run_cap {config.get('run_cap', 'unset')}: "
        f"{', '.join(f'`{p}`' for p in sorted(params))}"
    )


def create_sweep(algo: str, sha: str) -> str:
    """Create a W&B sweep from the commitish's own ci/sweep_{algo}.yaml."""
    import wandb

    sweep_config = read_sweep_config(algo, sha)
    sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT)
    entity = wandb.Api().default_entity
    sweep_path = f"{entity}/{WANDB_PROJECT}/{sweep_id}"
    print(f"Sweep created: https://wandb.ai/{entity}/{WANDB_PROJECT}/sweeps/{sweep_id}")
    print(f"Sweep config: {sweep_summary_line(sweep_config)}")
    return sweep_path
