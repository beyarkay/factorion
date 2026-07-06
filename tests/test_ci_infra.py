"""Tests for the ci/ package: job specs, command building, pod-name deadline
encoding, watchdog decisions, and the every-metric compare report.

Everything here runs offline — the pure decision/formatting cores are tested
directly; nothing talks to RunPod or W&B.
"""

import math
import time

from ci.config import (
    MAX_POD_AGE_SECONDS,
    CompareJob,
    PpoJob,
    SftJob,
    SweepJob,
    compare_group,
    job_from_dict,
    job_to_dict,
    parse_pod_name,
    pod_name,
    ppo_budget_seconds,
    sft_budget_seconds,
)
from ci.jobs import (
    compare_seed_command,
    ppo_command,
    sft_command,
    sweep_agent_command,
)
from ci.report import (
    MetricRow,
    compare_metric_rows,
    flatten_summary,
    metric_direction,
    render_compare_markdown,
)
from ci.watchdog import decide_terminations

SHA = "0123456789abcdef0123456789abcdef01234567"
BASE_SHA = "fedcba9876543210fedcba9876543210fedcba98"


# ── Single source of truth: commands carry ONLY the allowed overrides ──


class TestSftCommand:
    def test_defaults_pass_no_overrides(self):
        cmd = sft_command(SftJob(sha=SHA))
        assert "sft.py" in cmd[1]
        assert "--track" in cmd
        # No hyperparameter leaks: everything comes from training_config.py.
        forbidden = {"--num-samples", "--epochs", "--size", "--lr", "--seed", "--batch-size"}
        assert forbidden.isdisjoint(cmd)

    def test_num_samples_is_the_only_knob(self):
        cmd = sft_command(SftJob(sha=SHA, num_samples=123))
        assert cmd[cmd.index("--num-samples") + 1] == "123"

    def test_tags_identify_the_run(self):
        cmd = sft_command(SftJob(sha=SHA))
        assert "ci" in cmd and "fci:sft" in cmd and f"sha:{SHA[:7]}" in cmd


class TestPpoCommand:
    def test_start_from_required_total_timesteps_optional(self):
        cmd = ppo_command(PpoJob(sha=SHA, start_from="j0s5y2mc"))
        assert cmd[cmd.index("--start-from") + 1] == "j0s5y2mc"
        assert "--total-timesteps" not in cmd
        forbidden = {"--learning-rate", "--seed", "--size", "--critic-warmup", "--ent-coef-start"}
        assert forbidden.isdisjoint(cmd)

    def test_total_timesteps_override(self):
        cmd = ppo_command(PpoJob(sha=SHA, start_from="j0s5y2mc", total_timesteps=99))
        assert cmd[cmd.index("--total-timesteps") + 1] == "99"


class TestCompareCommand:
    def test_seed_group_and_role(self):
        job = CompareJob(sha=SHA, base_sha=BASE_SHA, seeds=3, num_samples=1000)
        cmd = compare_seed_command(job, "base", BASE_SHA, seed=2)
        assert cmd[cmd.index("--seed") + 1] == "2"
        assert cmd[cmd.index("--wandb-group") + 1] == compare_group(SHA, "base")
        # The base side is tagged with the BASE sha (what actually ran).
        assert f"sha:{BASE_SHA[:7]}" in cmd
        assert f"cmp:{SHA[:7]}" in cmd and "cmp-role:base" in cmd

    def test_checkpoints_do_not_collide_across_seeds_or_roles(self):
        job = CompareJob(sha=SHA, base_sha=BASE_SHA, seeds=2, num_samples=1000)
        paths = set()
        for role, role_sha in (("test", SHA), ("base", BASE_SHA)):
            for seed in (1, 2):
                cmd = compare_seed_command(job, role, role_sha, seed)
                paths.add(cmd[cmd.index("--checkpoint-path") + 1])
        assert len(paths) == 4


class TestSweepCommand:
    def test_agent_targets_the_sweep(self):
        job = SweepJob(sha=SHA, algo="sft", sweep_path="me/factorion/ab12cd34")
        assert sweep_agent_command(job) == ["wandb", "agent", "me/factorion/ab12cd34"]


# ── Job spec serialization (launcher → pod round trip) ─────────────


class TestJobSerialization:
    def test_round_trip_every_kind(self):
        jobs = [
            SftJob(sha=SHA, num_samples=5),
            PpoJob(sha=SHA, start_from="j0s5y2mc", total_timesteps=7),
            SweepJob(sha=SHA, algo="ppo", sweep_path="e/p/s", agents_per_pod=2),
            CompareJob(sha=SHA, base_sha=BASE_SHA, seeds=4, num_samples=9),
        ]
        for job in jobs:
            assert job_from_dict(job_to_dict(job)) == job


# ── Pod naming + watchdog ──────────────────────────────────────────


class TestPodName:
    def test_round_trip(self):
        name = pod_name("sft", 1000, 5000, SHA)
        meta = parse_pod_name(name)
        assert meta is not None
        assert (meta.kind, meta.created, meta.deadline, meta.sha7) == (
            "sft",
            1000,
            5000,
            SHA[:7],
        )

    def test_foreign_names_rejected(self):
        for name in ("my-dev-pod", "fci-sft-nonsense", "", "ci-smoke-12345"):
            assert parse_pod_name(name) is None


class TestWatchdog:
    def _pod(self, name, uptime=0):
        return {"id": f"id-{name}", "name": name, "runtime": {"uptimeInSeconds": uptime}}

    def test_kills_past_deadline_keeps_live_ones(self):
        now = time.time()
        expired = self._pod(pod_name("sft", int(now - 7200), int(now - 60), SHA))
        alive = self._pod(pod_name("ppo", int(now - 60), int(now + 7200), SHA))
        doomed = decide_terminations([expired, alive], now=now)
        assert [p["name"] for p, _ in doomed] == [expired["name"]]

    def test_never_touches_non_fci_pods(self):
        now = time.time()
        personal = self._pod("my-experiment-pod", uptime=10 * MAX_POD_AGE_SECONDS)
        assert decide_terminations([personal], now=now) == []

    def test_unparseable_fci_pod_reaped_after_max_age(self):
        now = time.time()
        young = self._pod("fci-sft-renamed", uptime=60)
        old = self._pod("fci-sft-renamed-old", uptime=MAX_POD_AGE_SECONDS + 1)
        doomed = decide_terminations([young, old], now=now)
        assert [p["name"] for p, _ in doomed] == ["fci-sft-renamed-old"]

    def test_absolute_age_cap_beats_a_far_future_deadline(self):
        now = time.time()
        created = int(now - MAX_POD_AGE_SECONDS - 1)
        runaway = self._pod(pod_name("sweep", created, int(now + 7200), SHA))
        doomed = decide_terminations([runaway], now=now)
        assert len(doomed) == 1


# ── Budgets ────────────────────────────────────────────────────────


class TestBudgets:
    def test_budgets_scale_with_work(self):
        assert sft_budget_seconds(10_000_000, 1) > sft_budget_seconds(1_000_000, 1)
        assert ppo_budget_seconds(1_000_000) > ppo_budget_seconds(100_000)

    def test_compare_budget_covers_both_roles(self):
        job = CompareJob(sha=SHA, base_sha=BASE_SHA, seeds=3, num_samples=1_000_000)
        single = sft_budget_seconds(1_000_000, 1)
        assert job.budget_seconds() > 2 * single


# ── Compare report ─────────────────────────────────────────────────


class TestFlattenSummary:
    def test_nested_namespaces_and_stat_dicts(self):
        summary = {
            "val": {"thput": 0.3, "thput_eot": {"max": 0.2}},
            "loss": 1.5,
            "_runtime": 99,  # W&B internal: skipped
            "note": "text",  # non-numeric: skipped
            "flag": True,  # bool: skipped
            "bad": float("nan"),  # non-finite: skipped
        }
        assert flatten_summary(summary) == {
            "val/thput": 0.3,
            "val/thput_eot": 0.2,
            "loss": 1.5,
        }


class TestMetricDirection:
    def test_heuristics(self):
        assert metric_direction("val/thput_eot") == "higher"
        assert metric_direction("val/acc") == "higher"
        assert metric_direction("train/loss") == "lower"
        assert metric_direction("some/unknown_metric") is None


class TestCompareRows:
    def _sides(self):
        base = {s: {"val/thput": 0.10 + 0.01 * s, "train/loss": 2.0} for s in (1, 2, 3)}
        test = {s: {"val/thput": 0.30 + 0.01 * s, "train/loss": 1.0} for s in (1, 2, 3)}
        return base, test

    def test_paired_and_significant_improvement(self):
        base, test = self._sides()
        rows = {r.name: r for r in compare_metric_rows(base, test)}
        thput = rows["val/thput"]
        assert thput.paired
        assert math.isclose(thput.delta, 0.2, abs_tol=1e-9)
        assert thput.p_value is not None and thput.p_value < 0.05
        assert thput.verdict() == "better"

    def test_lower_is_better_direction(self):
        base = {s: {"train/loss": 2.0 + 0.001 * s} for s in (1, 2, 3)}
        test = {s: {"train/loss": 1.0 + 0.001 * s} for s in (1, 2, 3)}
        (row,) = compare_metric_rows(base, test)
        assert row.verdict() == "better"

    def test_one_sided_metric_is_skipped(self):
        base, test = self._sides()
        base[1]["only/base"] = 1.0
        base[2]["only/base"] = 1.0
        names = {r.name for r in compare_metric_rows(base, test)}
        assert "only/base" not in names

    def test_markdown_renders_all_rows(self):
        base, test = self._sides()
        rows = compare_metric_rows(base, test)
        md = render_compare_markdown(rows, "grp-base", "grp-test")
        assert "val/thput" in md and "train/loss" in md and "grp-base" in md

    def test_unpaired_falls_back_to_welch(self):
        base = {s: {"m": 1.0 + 0.1 * s} for s in (1, 2, 3)}
        test = {s: {"m": 5.0 + 0.1 * s} for s in (7, 8, 9)}
        (row,) = compare_metric_rows(base, test)
        assert not row.paired
        assert row.p_value is not None

    def test_verdict_requires_significance(self):
        row = MetricRow(
            name="val/thput",
            base_mean=0.1,
            base_std=0.1,
            test_mean=0.11,
            test_std=0.1,
            n_base=3,
            n_test=3,
            p_value=0.9,
            paired=True,
        )
        assert row.verdict() == ""
