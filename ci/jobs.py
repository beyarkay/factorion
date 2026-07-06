"""Pod-side job dispatcher: `python -m ci.jobs`, invoked by ci/runner.sh.

Decodes the job spec from the FCI_JOB_B64 env var and runs it. This is the
ONLY place CI training commands are built: a command contains `--track`, tags,
and the few overrides the job spec is allowed to carry — nothing else, so
every other hyperparameter comes from training_config.py.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import subprocess
import sys
import time

from ci.config import (
    CompareJob,
    Job,
    PpoJob,
    SftJob,
    SweepJob,
    compare_group,
    job_from_dict,
)

WORK_DIR = "/workspace/factorion"
BASE_WORK_DIR = "/workspace/factorion-base"


def _tags(job_kind: str, sha: str, *extra: str) -> list[str]:
    return ["--tags", "ci", f"fci:{job_kind}", f"sha:{sha[:7]}", *extra]


def sft_command(job: SftJob) -> list[str]:
    cmd = [sys.executable, "sft.py", "--track"]
    if job.num_samples is not None:
        cmd += ["--num-samples", str(job.num_samples)]
    return cmd + _tags(job.KIND, job.sha)


def ppo_command(job: PpoJob) -> list[str]:
    cmd = [sys.executable, "ppo.py", "--track", "--start-from", job.start_from]
    if job.total_timesteps is not None:
        cmd += ["--total-timesteps", str(job.total_timesteps)]
    return cmd + _tags(job.KIND, job.sha, f"from:{job.start_from}")


def compare_seed_command(job: CompareJob, role: str, role_sha: str, seed: int) -> list[str]:
    """One SFT seed of a compare job (role: 'test' | 'base')."""
    return [
        sys.executable,
        "sft.py",
        "--track",
        "--seed",
        str(seed),
        "--num-samples",
        str(job.num_samples),
        "--wandb-group",
        compare_group(job.sha, role),
        "--checkpoint-path",
        f"/workspace/ckpt_{role}_seed{seed}.pt",
    ] + _tags(job.KIND, role_sha, f"cmp:{job.sha[:7]}", f"cmp-role:{role}")


def sweep_agent_command(job: SweepJob) -> list[str]:
    return ["wandb", "agent", job.sweep_path]


# ── Execution ──────────────────────────────────────────────────────


def _run_parallel(cmds: list[list[str]], cwd: str, log_paths: list[str]) -> int:
    """Run commands concurrently, streaming each to its log. Returns #failed."""
    procs = []
    for cmd, log_path in zip(cmds, log_paths):
        print(f">>> Launching: {' '.join(cmd)} (log: {log_path})", flush=True)
        log = open(log_path, "w")
        procs.append((subprocess.Popen(cmd, cwd=cwd, stdout=log, stderr=log), log, log_path))

    failed = 0
    for proc, log, log_path in procs:
        code = proc.wait()
        log.close()
        if code == 0:
            print(f">>> Finished OK: {log_path}", flush=True)
        else:
            failed += 1
            print(f">>> FAILED (exit {code}): {log_path} — last lines:", flush=True)
            with open(log_path) as f:
                print("".join(f.readlines()[-30:]), flush=True)
    return failed


def _build_rust_extension(repo_dir: str) -> None:
    rs_dir = os.path.join(repo_dir, "factorion_rs")
    subprocess.run(
        ["maturin", "build", "--release", "--out", "dist"], cwd=rs_dir, check=True
    )
    wheels = sorted(glob.glob(os.path.join(rs_dir, "dist", "*.whl")))
    subprocess.run(
        ["pip", "install", "--force-reinstall", wheels[-1]], cwd=rs_dir, check=True
    )


def run_sft(job: SftJob) -> None:
    subprocess.run(sft_command(job), cwd=WORK_DIR, check=True)


def run_ppo(job: PpoJob) -> None:
    subprocess.run(ppo_command(job), cwd=WORK_DIR, check=True)


def run_sweep(job: SweepJob) -> None:
    """Launch N parallel `wandb agent` processes; they drain the sweep until
    its run_cap (set in the sweep yaml) is reached."""
    cmds = [sweep_agent_command(job) for _ in range(job.agents_per_pod)]
    logs = [f"/workspace/agent_{i}.log" for i in range(job.agents_per_pod)]
    failed = _run_parallel(cmds, cwd=WORK_DIR, log_paths=logs)
    if failed == job.agents_per_pod:
        raise RuntimeError(f"All {failed} sweep agents failed")


def run_compare(job: CompareJob) -> None:
    """Run seeds on the test commit, then on the base commit, then report.

    The base commit runs from a git worktree with its own rust-extension build,
    so each side executes exactly the code of its commit.
    """
    seeds = list(range(1, job.seeds + 1))

    # Test side (repo already checked out at job.sha, extension already built).
    failed = _run_parallel(
        [compare_seed_command(job, "test", job.sha, s) for s in seeds],
        cwd=WORK_DIR,
        log_paths=[f"/workspace/cmp_test_seed{s}.log" for s in seeds],
    )
    if failed == len(seeds):
        raise RuntimeError("All test-side seeds failed")

    # Base side: worktree at base_sha + its own extension build.
    subprocess.run(
        ["git", "worktree", "add", "--detach", BASE_WORK_DIR, job.base_sha],
        cwd=WORK_DIR,
        check=True,
    )
    _build_rust_extension(BASE_WORK_DIR)
    failed = _run_parallel(
        [compare_seed_command(job, "base", job.base_sha, s) for s in seeds],
        cwd=BASE_WORK_DIR,
        log_paths=[f"/workspace/cmp_base_seed{s}.log" for s in seeds],
    )
    if failed == len(seeds):
        raise RuntimeError("All base-side seeds failed")

    # Report: compare every logged metric across the two W&B groups. Give the
    # W&B backend a moment to finalize the just-finished runs first.
    time.sleep(60)
    from ci.report import compare_report

    md = compare_report(
        base_group=compare_group(job.sha, "base"),
        test_group=compare_group(job.sha, "test"),
    )
    with open("/workspace/compare_report.md", "w") as f:
        f.write(md)
    print(md, flush=True)


def run(job: Job) -> None:
    if isinstance(job, SftJob):
        run_sft(job)
    elif isinstance(job, PpoJob):
        run_ppo(job)
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
