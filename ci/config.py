"""CI infrastructure settings and job specs.

This module is the single source of truth for *infrastructure* knobs (docker
image, GPU lineup, pod naming, time budgets). Training hyperparameters are NOT
here — they live in `training_config.py` and flow into jobs untouched: a job
spec only carries the handful of fields the CLI is allowed to override
(commitish, SFT num_samples, PPO start_from/total_timesteps, compare seeds).
Anything a spec leaves as None is simply not passed on the training command
line, so the `training_config.py` default applies.

Keep this module a leaf: stdlib + `training_config` only, so the watchdog can
import it in a bare GitHub cron environment.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import ClassVar, Optional

from training_config import PpoArgs, SftArgs

DOCKER_IMAGE = "beyarkay/factorion-ci-gpu:latest"
CONTAINER_DISK_GB = 40
REPO_URL = "https://github.com/beyarkay/factorion"
WANDB_PROJECT = SftArgs().wandb_project_name

# Preferred first, then availability fallbacks (each pricier/bigger). The
# default is the RTX 2000 Ada — the GPU the speed benchmarks were tuned on;
# these models are tiny and the workload is single-core-CPU-bound, so a bigger
# GPU is overkill (see tests/benchmarks/EXPERIMENT_LOG.md).
GPU_FALLBACKS = [
    "NVIDIA RTX 2000 Ada Generation",
    "NVIDIA RTX 4000 Ada Generation",
    "NVIDIA RTX 6000 Ada Generation",
    "NVIDIA GeForce RTX 4090",
    "NVIDIA RTX A6000",
    "NVIDIA A100 80GB PCIe",
    "NVIDIA A100-SXM4-80GB",
]

# Only schedule on hosts whose driver supports a CUDA version new enough for
# our torch build. uv.lock pins torch 2.12.x+cu130, which needs a CUDA 13.0
# runtime (host driver >= 580). Keep in sync with the torch cuXXX build.
ALLOWED_CUDA_VERSIONS = ["13.0"]

# ── Pod naming ─────────────────────────────────────────────────────
# Every CI pod encodes its own creation time and kill-by deadline in its name,
# so the watchdog can enforce cleanup statelessly from `runpod.get_pods()`
# alone: factorion-ci-<kind>-c<created_epoch>-d<deadline_epoch>-<sha7>
POD_PREFIX = "factorion-ci-"
_POD_NAME_RE = re.compile(
    r"^factorion-ci-(?P<kind>[a-z-]+)-c(?P<created>\d+)-d(?P<deadline>\d+)-(?P<sha7>[0-9a-f]+)$"
)

# Absolute backstop: no CI pod may outlive this, deadline or not. Generous
# because the default 45M-sample SFT run legitimately takes ~20h.
MAX_POD_AGE_SECONDS = 48 * 3600


def pod_name(kind: str, created_epoch: int, deadline_epoch: int, sha: str) -> str:
    return f"{POD_PREFIX}{kind}-c{created_epoch}-d{deadline_epoch}-{sha[:7]}"


@dataclass
class PodMeta:
    kind: str
    created: int
    deadline: int
    sha7: str


def parse_pod_name(name: str) -> Optional[PodMeta]:
    """Parse a CI pod name; None for foreign pods or unrecognized names."""
    m = _POD_NAME_RE.match(name)
    if m is None:
        return None
    return PodMeta(
        kind=m.group("kind"),
        created=int(m.group("created")),
        deadline=int(m.group("deadline")),
        sha7=m.group("sha7"),
    )


# ── Time budgets ───────────────────────────────────────────────────
# Calibration carried over from the old GH workflows: SFT streams ~1000
# sample-epochs/sec, PPO ~200 env-steps/sec, both CPU-bound so the rates hold
# across the GPU lineup. 1.5-2x safety margin so the watchdog never kills a
# legitimate run, plus slack for pod boot + clone + rust build.
SETUP_SLACK_SECONDS = 1800
SWEEP_BUDGET_SECONDS = 8 * 3600  # sweeps run until run_cap (from the yaml)


def sft_budget_seconds(num_samples: int, epochs: int) -> int:
    return int(num_samples * epochs / 1000 * 1.5) + SETUP_SLACK_SECONDS


def ppo_budget_seconds(total_timesteps: int) -> int:
    return int(total_timesteps / 200 * 2) + SETUP_SLACK_SECONDS


# ── Job specs ──────────────────────────────────────────────────────
# The full override surface of CI training jobs. Serialized to JSON, handed to
# the pod via env, and decoded by ci/jobs.py. If a knob isn't here, it can't
# be changed from the CI side — training_config.py decides it.
#
# seed / group / extra_tags are infrastructure, not hyperparameters: the
# compare fan-out uses them to pair runs across commits, and PR-triggered
# jobs use extra_tags to link W&B runs back to their PR (tag "pr:<num>").


@dataclass
class SftJob:
    """From-scratch SFT training run at a commit."""

    sha: str
    num_samples: Optional[int] = None  # None = SftArgs default
    seed: Optional[int] = None  # None = SftArgs default
    group: Optional[str] = None  # W&B group
    extra_tags: list[str] = field(default_factory=list)

    KIND: ClassVar[str] = "sft"

    def budget_seconds(self) -> int:
        defaults = SftArgs()
        n = self.num_samples if self.num_samples is not None else defaults.num_samples
        return sft_budget_seconds(n, defaults.epochs)


@dataclass
class PpoJob:
    """PPO run at a commit, starting from an SFT checkpoint (W&B run id)."""

    sha: str
    start_from: str
    total_timesteps: Optional[int] = None  # None = PpoArgs default
    seed: Optional[int] = None  # None = PpoArgs default
    group: Optional[str] = None  # W&B group
    extra_tags: list[str] = field(default_factory=list)

    KIND: ClassVar[str] = "ppo"

    def budget_seconds(self) -> int:
        t = (
            self.total_timesteps
            if self.total_timesteps is not None
            else PpoArgs().total_timesteps
        )
        return ppo_budget_seconds(t)


@dataclass
class SweepJob:
    """One pod's worth of W&B sweep agents (sweep created at launch time)."""

    sha: str
    algo: str  # "sft" | "ppo"
    sweep_path: str  # entity/project/sweep_id
    agents_per_pod: int = 5

    KIND: ClassVar[str] = "sweep"

    def budget_seconds(self) -> int:
        return SWEEP_BUDGET_SECONDS


Job = SftJob | PpoJob | SweepJob


def job_to_dict(job: Job) -> dict:
    return {"kind": job.KIND, **asdict(job)}


def job_from_dict(d: dict) -> Job:
    kinds = {cls.KIND: cls for cls in (SftJob, PpoJob, SweepJob)}
    d = dict(d)
    cls = kinds[d.pop("kind")]
    return cls(**d)


# ── Compare fan-out ────────────────────────────────────────────────
# A compare is not a pod-side job kind: it fans out into 2 x seeds ordinary
# single-run pods (one run per pod so seeds never compete for CPU), grouped in
# W&B by role. The report is assembled from W&B afterwards.

COMPARE_SEEDS_DEFAULT = 3
COMPARE_NUM_SAMPLES_DEFAULT = 5_000_000  # hours, not days, per compare


def compare_group(sha: str, role: str) -> str:
    """W&B group name for one side of a compare (role: 'test' | 'base')."""
    return f"cmp-{sha[:7]}-{role}"


def compare_fanout(
    algo: str,
    sha: str,
    base_sha: str,
    seeds: int = COMPARE_SEEDS_DEFAULT,
    num_samples: int = COMPARE_NUM_SAMPLES_DEFAULT,
    start_from: Optional[str] = None,
    total_timesteps: Optional[int] = None,
    extra_tags: Optional[list[str]] = None,
) -> list[SftJob | PpoJob]:
    """Build the 2 x seeds single-run job specs for a compare.

    algo "sft" compares from-scratch SFT; algo "ppo" compares PPO finetuning
    from the same start_from checkpoint on both commits.
    """
    if algo not in ("sft", "ppo"):
        raise ValueError(f"algo must be 'sft' or 'ppo', got {algo!r}")
    if algo == "ppo" and not start_from:
        raise ValueError("PPO compare needs --start-from (a W&B SFT run id)")

    jobs: list[SftJob | PpoJob] = []
    for role, role_sha in (("test", sha), ("base", base_sha)):
        for seed in range(1, seeds + 1):
            tags = [f"cmp:{sha[:7]}", f"cmp-role:{role}", *(extra_tags or [])]
            if algo == "sft":
                jobs.append(
                    SftJob(
                        sha=role_sha,
                        num_samples=num_samples,
                        seed=seed,
                        group=compare_group(sha, role),
                        extra_tags=tags,
                    )
                )
            else:
                assert start_from is not None
                jobs.append(
                    PpoJob(
                        sha=role_sha,
                        start_from=start_from,
                        total_timesteps=total_timesteps,
                        seed=seed,
                        group=compare_group(sha, role),
                        extra_tags=tags,
                    )
                )
    return jobs
