"""Pod-side job dispatcher: `python -m ci.jobs`, invoked by ci/runner.sh.

Decodes the job spec from the FCI_JOB_B64 env var and runs it. This is the
ONLY place CI training commands are built: a command contains `--track`, tags,
and the few overrides the job spec is allowed to carry — nothing else, so
every other hyperparameter comes from training_config.py.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys

from ci.config import (
    CompareJob,
    Job,
    PpoJob,
    SftJob,
    SweepJob,
    job_from_dict,
)

WORK_DIR = "/workspace/factorion"


def _tags(job: SftJob | PpoJob, *extra: str) -> list[str]:
    return [
        "--tags",
        "ci",
        f"kind:{job.KIND}",
        f"sha:{job.sha[:7]}",
        *extra,
        *job.extra_tags,
    ]


def _run_overrides(job: SftJob | PpoJob) -> list[str]:
    out = []
    if job.seed is not None:
        out += ["--seed", str(job.seed)]
    if job.group is not None:
        out += ["--wandb-group", job.group]
    run_id = os.environ.get("FCI_WANDB_RUN_ID")
    if run_id:
        out += ["--wandb-run-id", run_id]
    return out


def sft_command(job: SftJob) -> list[str]:
    cmd = [sys.executable, "sft.py", "--track"]
    if job.num_samples is not None:
        cmd += ["--num-samples", str(job.num_samples)]
    return cmd + _run_overrides(job) + _tags(job)


def ppo_command(job: PpoJob) -> list[str]:
    cmd = [sys.executable, "ppo.py", "--track", "--start-from", job.start_from]
    if job.total_timesteps is not None:
        cmd += ["--total-timesteps", str(job.total_timesteps)]
    return cmd + _run_overrides(job) + _tags(job, f"from:{job.start_from}")


def sweep_agent_command(job: SweepJob) -> list[str]:
    return ["wandb", "agent", job.sweep_path]


def _compare_subjob(job: CompareJob, seed: int) -> SftJob | PpoJob:
    """The single-run job for one seed of a compare side — carries the same
    group and tags so every seed lands in one W&B group."""
    if job.algo == "sft":
        return SftJob(
            sha=job.sha,
            num_samples=job.num_samples,
            seed=seed,
            group=job.group,
            extra_tags=job.extra_tags,
        )
    assert job.start_from is not None  # enforced by compare_fanout
    return PpoJob(
        sha=job.sha,
        start_from=job.start_from,
        total_timesteps=job.total_timesteps,
        seed=seed,
        group=job.group,
        extra_tags=job.extra_tags,
    )


# ── Execution ──────────────────────────────────────────────────────


def run_compare(job: CompareJob) -> None:
    """Run each seed sequentially on this one pod. Every seed is attempted even
    if an earlier one fails, so one bad seed doesn't cost the whole side; the
    pod fails (and reports) only if every seed failed."""
    failed = []
    for seed in job.seeds:
        sub = _compare_subjob(job, seed)
        cmd = sft_command(sub) if isinstance(sub, SftJob) else ppo_command(sub)
        print(f">>> Compare seed {seed}/{job.seeds[-1]}: {cmd}", flush=True)
        if subprocess.run(cmd, cwd=WORK_DIR).returncode != 0:
            failed.append(seed)
            print(f">>> Compare seed {seed} FAILED", flush=True)
    if len(failed) == len(job.seeds):
        raise RuntimeError(f"All {len(failed)} compare seeds failed")
    if failed:
        print(f">>> Compare finished with failed seed(s): {failed}", flush=True)


def run_sweep(job: SweepJob) -> None:
    """Launch N parallel `wandb agent` processes; they drain the sweep until
    its run_cap (set in the sweep yaml) is reached."""
    procs = []
    for i in range(job.agents_per_pod):
        log_path = f"/workspace/agent_{i}.log"
        print(f">>> Starting sweep agent {i} (log: {log_path})", flush=True)
        log = open(log_path, "w")
        procs.append(
            (subprocess.Popen(sweep_agent_command(job), cwd=WORK_DIR, stdout=log, stderr=log), log, log_path)
        )

    failed = 0
    for proc, log, log_path in procs:
        code = proc.wait()
        log.close()
        if code != 0:
            failed += 1
            print(f">>> Agent FAILED (exit {code}): {log_path} — last lines:", flush=True)
            with open(log_path) as f:
                print("".join(f.readlines()[-30:]), flush=True)
    if failed == job.agents_per_pod:
        raise RuntimeError(f"All {failed} sweep agents failed")


def run(job: Job) -> None:
    if isinstance(job, SftJob):
        subprocess.run(sft_command(job), cwd=WORK_DIR, check=True)
    elif isinstance(job, PpoJob):
        subprocess.run(ppo_command(job), cwd=WORK_DIR, check=True)
    elif isinstance(job, SweepJob):
        run_sweep(job)
    elif isinstance(job, CompareJob):
        run_compare(job)
    else:
        raise ValueError(f"Unknown job: {job!r}")


def main() -> None:
    spec = json.loads(base64.b64decode(os.environ["FCI_JOB_B64"]))
    job = job_from_dict(spec)
    print(f">>> Job: {job}", flush=True)
    run(job)
    print(">>> Job completed successfully", flush=True)


if __name__ == "__main__":
    main()
