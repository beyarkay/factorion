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
import subprocess
import time

from ci.config import (
    WANDB_PROJECT,
    Job,
    job_to_dict,
    pod_name,
)
from ci.config import REPO_URL as DEFAULT_REPO_URL

# Phase-1 bootstrap, shipped to the pod as FCI_BOOT_B64 (base64 dodges the
# quoting hazards of RunPod's docker-args plumbing). Static on purpose: all
# per-job values arrive via env vars, never via string templating.
BOOTSTRAP = """\
set -euo pipefail
terminate_pod() {
    echo "[fci] terminating pod ${RUNPOD_POD_ID:-unknown}"
    python -c 'import os, runpod; runpod.api_key = os.environ["RUNPOD_API_KEY"]; runpod.terminate_pod(os.environ["RUNPOD_POD_ID"])' || true
}
trap terminate_pod EXIT
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
) -> None:
    """Create the pod for `job` and (optionally) wait for it to boot."""
    now = int(time.time())
    deadline = now + job.budget_seconds()
    name = pod_name(job.KIND, now, deadline, job.sha)
    spec = job_to_dict(job)

    env = {
        "RUNPOD_API_KEY": os.environ.get("RUNPOD_API_KEY", ""),
        "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
        "FCI_BOOT_B64": base64.b64encode(BOOTSTRAP.encode()).decode(),
        "FCI_JOB_B64": base64.b64encode(json.dumps(spec).encode()).decode(),
        "FCI_REPO_URL": repo_url,
        "FCI_SHA": job.sha,
        "FCI_DEADLINE": str(deadline),
    }

    print(f"Job:      {spec}")
    print(f"Pod name: {name}")
    print(f"GPU:      {gpu_type} (with fallbacks)")
    print(f"Deadline: {time.strftime('%Y-%m-%d %H:%M:%S %z', time.localtime(deadline))} "
          f"({job.budget_seconds() // 60} min budget; watchdogs kill the pod after this)")

    if dry_run:
        print("\n--dry-run: not creating a pod. Bootstrap script:\n")
        print(BOOTSTRAP)
        return

    for var in ("RUNPOD_API_KEY", "WANDB_API_KEY"):
        if not env[var]:
            raise SystemExit(f"error: {var} must be set in the environment")

    from ci import runpod_api

    pod = runpod_api.create_pod(name=name, gpu_type=gpu_type, docker_args=DOCKER_ARGS, env=env)
    pod_id = pod["id"]

    print(f"\nPod {pod_id} created. The job runs unattended and the pod")
    print("terminates itself when done. Track progress:")
    print(f"  W&B:    https://wandb.ai/ (project {WANDB_PROJECT}, tag sha:{job.sha[:7]})")
    print(f"  RunPod: https://www.runpod.io/console/pods (container logs for {name})")
    print(f"  CLI:    uv run python -m ci pods   |   uv run python -m ci kill {pod_id}")

    if not wait:
        return
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


def create_sweep(algo: str, sha: str) -> str:
    """Create a W&B sweep from the commitish's own ci/sweep_{algo}.yaml.

    Reading the config via `git show <sha>:...` (not the working tree) keeps
    the sweep true to the commit being swept.
    """
    import wandb
    import yaml

    try:
        raw = _git("show", f"{sha}:ci/sweep_{algo}.yaml")
    except subprocess.CalledProcessError:
        raise SystemExit(
            f"error: ci/sweep_{algo}.yaml does not exist at {sha[:12]} — "
            "sweeps need a commit that contains the ci/ directory"
        )
    sweep_config = yaml.safe_load(raw)
    sweep_id = wandb.sweep(sweep=sweep_config, project=WANDB_PROJECT)
    entity = wandb.Api().default_entity
    sweep_path = f"{entity}/{WANDB_PROJECT}/{sweep_id}"
    print(f"Sweep created: https://wandb.ai/{entity}/{WANDB_PROJECT}/sweeps/{sweep_id}")
    return sweep_path
