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


# ── Execution ──────────────────────────────────────────────────────


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
