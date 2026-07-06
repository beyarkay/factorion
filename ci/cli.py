"""CLI command functions. `python -m ci` maps each onto a subcommand.

Deliberately tiny override surface: a commitish plus the few job-specific
knobs (SFT num_samples, PPO start_from/total_timesteps, compare seeds, sweep
pod counts). Every other hyperparameter comes from training_config.py.
"""

from __future__ import annotations

import time
from typing import Optional

from ci.config import (
    COMPARE_NUM_SAMPLES_DEFAULT,
    COMPARE_SEEDS_DEFAULT,
    GPU_FALLBACKS,
    PpoJob,
    SftJob,
    SweepJob,
    compare_group,
    parse_pod_name,
)
from ci.launch import create_sweep, launch, launch_compare, resolve_ref

DEFAULT_GPU = GPU_FALLBACKS[0]


def sft(
    ref: str,
    num_samples: Optional[int] = None,
    gpu_type: str = DEFAULT_GPU,
    dry_run: bool = False,
    no_wait: bool = False,
) -> None:
    """Launch a from-scratch SFT training run at a commitish.

    Args:
        ref: Commitish to train (branch / tag / SHA); must be pushed to origin.
        num_samples: Samples per epoch; default = SftArgs().num_samples.
        gpu_type: RunPod GPU type (falls back through the standard lineup).
        dry_run: Print what would launch without creating a pod.
        no_wait: Return right after pod creation instead of waiting for boot.
    """
    job = SftJob(sha=resolve_ref(ref), num_samples=num_samples)
    launch(job, gpu_type, dry_run=dry_run, wait=not no_wait)


def ppo(
    ref: str,
    start_from: str,
    total_timesteps: Optional[int] = None,
    gpu_type: str = DEFAULT_GPU,
    dry_run: bool = False,
    no_wait: bool = False,
) -> None:
    """Launch a PPO run at a commitish, starting from an SFT checkpoint.

    Args:
        ref: Commitish to train (branch / tag / SHA); must be pushed to origin.
        start_from: W&B run id of the SFT checkpoint (e.g. j0s5y2mc).
        total_timesteps: Override; default = PpoArgs().total_timesteps.
        gpu_type: RunPod GPU type (falls back through the standard lineup).
        dry_run: Print what would launch without creating a pod.
        no_wait: Return right after pod creation instead of waiting for boot.
    """
    job = PpoJob(sha=resolve_ref(ref), start_from=start_from, total_timesteps=total_timesteps)
    launch(job, gpu_type, dry_run=dry_run, wait=not no_wait)


def sweep(
    algo: str,
    /,
    ref: str = "main",
    pods: int = 1,
    agents_per_pod: int = 5,
    gpu_type: str = DEFAULT_GPU,
    dry_run: bool = False,
) -> None:
    """Run a W&B hyperparameter sweep, e.g. `sweep sft` or `sweep ppo`.

    The sweep config is ci/sweep_<algo>.yaml as it exists AT THE REF, so the
    sweep is true to the commit being swept.

    Args:
        algo: "sft" or "ppo".
        ref: Commitish to sweep; must be pushed to origin.
        pods: Number of RunPod pods to launch.
        agents_per_pod: Parallel `wandb agent` processes per pod (GPU
            time-slicing). Total runs are capped by run_cap in the sweep yaml.
        gpu_type: RunPod GPU type (falls back through the standard lineup).
        dry_run: Print what would launch without creating pods or the sweep.
    """
    if algo not in ("sft", "ppo"):
        raise SystemExit(f"error: algo must be 'sft' or 'ppo', got {algo!r}")
    sha = resolve_ref(ref)
    if dry_run:
        sweep_path = f"<entity>/<project>/<sweep-id-for-ci/sweep_{algo}.yaml@{sha[:7]}>"
    else:
        sweep_path = create_sweep(algo, sha)
    for _ in range(pods):
        job = SweepJob(sha=sha, algo=algo, sweep_path=sweep_path, agents_per_pod=agents_per_pod)
        # Never block on boot: with several pods, waiting serially is useless.
        launch(job, gpu_type, dry_run=dry_run, wait=False)
    print(f"\nWhen done: uv run python -m ci sweep-report --sweep {sweep_path}")


def compare(
    algo: str = "sft",
    /,
    *,
    ref: str,
    base_ref: str = "main",
    seeds: int = COMPARE_SEEDS_DEFAULT,
    num_samples: int = COMPARE_NUM_SAMPLES_DEFAULT,
    start_from: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    gpu_type: str = DEFAULT_GPU,
    dry_run: bool = False,
) -> None:
    """Compare a commitish against a base (default origin/main), multi-seed.

    Invoked as `compare sft --ref X` or `compare ppo --ref X --start-from ID`.
    Fans out into 2 x seeds pods — one training run per pod, so seeds never
    compete for CPU. Both sides run their own commit's code. On a PR, the
    /ci compare comment command launches this AND posts the every-metric
    seed-paired report (plus the assert commit status) when the runs finish.

    Args:
        algo: "sft" (from scratch) or "ppo" (finetune from --start-from on
            both commits).
        ref: Commitish under test; must be pushed to origin.
        base_ref: Baseline commitish (default: main).
        seeds: Seeds per side; runs are seed-paired for the t-test.
        num_samples: SFT samples per run (smaller than a production run so a
            compare finishes in hours, not days). Ignored for ppo.
        start_from: W&B SFT run id; required for ppo.
        total_timesteps: PPO override; default = PpoArgs().total_timesteps.
        gpu_type: RunPod GPU type (falls back through the standard lineup).
        dry_run: Print what would launch without creating pods.
    """
    sha = resolve_ref(ref)
    launch_compare(
        algo=algo,
        sha=sha,
        base_sha=resolve_ref(base_ref),
        seeds=seeds,
        num_samples=num_samples,
        start_from=start_from,
        total_timesteps=total_timesteps,
        gpu_type=gpu_type,
        dry_run=dry_run,
    )
    print(
        f"\nRuns land in W&B groups {compare_group(sha, 'test')} / "
        f"{compare_group(sha, 'base')}; on a PR, /ci compare posts the report."
    )


def pods() -> None:
    """List CI pods (name, status, GPU, uptime, cost, deadline)."""
    from ci import runpod_api

    ci_pods = runpod_api.list_ci_pods()
    if not ci_pods:
        print("No CI pods running.")
        return
    now = time.time()
    for pod in ci_pods:
        meta = parse_pod_name(pod.get("name") or "")
        uptime = (pod.get("runtime") or {}).get("uptimeInSeconds") or 0
        cost_hr = pod.get("costPerHr", "?")
        gpu = pod.get("machine", {}).get("gpuDisplayName", "?")
        deadline = (
            f"deadline in {runpod_api.format_uptime(meta.deadline - now)}"
            if meta and meta.deadline > now
            else "PAST DEADLINE"
            if meta
            else "no deadline (unparseable name)"
        )
        print(
            f"{pod['id']}  {pod.get('name')}  [{pod.get('desiredStatus')}]  "
            f"{gpu}  up {runpod_api.format_uptime(uptime)}  ${cost_hr}/hr  {deadline}"
        )


def kill(pod_id: str = "", all: bool = False) -> None:
    """Terminate a CI pod by id, or all CI pods with --all.

    Args:
        pod_id: RunPod pod id to terminate.
        all: Terminate every fci-* pod.
    """
    from ci import runpod_api

    if all:
        for pod in runpod_api.list_ci_pods():
            runpod_api.terminate_with_retry(pod["id"])
        return
    if not pod_id:
        raise SystemExit("error: pass a pod id or --all")
    runpod_api.terminate_with_retry(pod_id)


def watchdog(dry_run: bool = False) -> None:
    """Reap CI pods that are past their deadline (same sweep the GH cron runs).

    Args:
        dry_run: Report what would be terminated without terminating.
    """
    from ci import watchdog as watchdog_mod

    watchdog_mod.run(dry_run=dry_run)


def post_pending_reports(dry_run: bool = False, window_days: int = 3) -> None:
    """Post PR summary comments for finished runs that lack one (reporter cron).

    Args:
        dry_run: Report what would be posted without posting.
        window_days: How far back to scan W&B for finished runs.
    """
    from ci import report

    report.post_pending_reports(window_days=window_days, dry_run=dry_run)


def sweep_report(sweep: str, top_n: int = 5, out: str = "") -> None:
    """Print (and optionally save) a summary of a W&B sweep.

    Args:
        sweep: Full sweep path (entity/project/sweep_id).
        top_n: Number of top runs to include.
        out: Optional path to also write the markdown report to.
    """
    from ci import report

    md = report.sweep_report(sweep, top_n=top_n)
    print(md)
    if out:
        with open(out, "w") as f:
            f.write(md)


def history(out: str = "ci/history.csv", limit: int = 500) -> None:
    """Regenerate the metrics-over-time CSV from W&B runs tagged `ci`.

    Args:
        out: CSV path to write (commit it for plotting over time).
        limit: Maximum number of runs to include.
    """
    from ci import report

    n = report.history_csv(out, limit=limit)
    print(f"Wrote {n} run(s) to {out}")
