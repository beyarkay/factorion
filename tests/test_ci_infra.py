"""Tests for the ci/ package: job specs, command building, compare fan-out,
pod-name deadline encoding, watchdog decisions, assertions, and reports.

Everything here runs offline — the pure decision/formatting cores are tested
directly; nothing talks to RunPod, W&B, or GitHub.
"""

import math
import time

from ci.config import (
    MAX_POD_AGE_SECONDS,
    PpoJob,
    SftJob,
    SweepJob,
    compare_fanout,
    compare_group,
    job_from_dict,
    job_to_dict,
    parse_pod_name,
    pod_name,
    ppo_budget_seconds,
    sft_budget_seconds,
)
from ci.gh_command import parse_comment
from ci.jobs import ppo_command, sft_command, sweep_agent_command
from ci.report import (
    MetricRow,
    compare_metric_rows,
    evaluate_assertion,
    flatten_summary,
    metric_direction,
    render_compare_markdown,
    run_summary_markdown,
    select_headline,
    select_unreported,
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

    def test_num_samples_is_the_only_hyperparam_knob(self):
        cmd = sft_command(SftJob(sha=SHA, num_samples=123))
        assert cmd[cmd.index("--num-samples") + 1] == "123"

    def test_tags_identify_the_run(self):
        cmd = sft_command(SftJob(sha=SHA, extra_tags=["pr:42"]))
        assert "ci" in cmd and "kind:sft" in cmd and f"sha:{SHA[:7]}" in cmd
        assert "pr:42" in cmd


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


class TestSweepCommand:
    def test_agent_targets_the_sweep(self):
        job = SweepJob(sha=SHA, algo="sft", sweep_path="me/factorion/ab12cd34")
        assert sweep_agent_command(job) == ["wandb", "agent", "me/factorion/ab12cd34"]


# ── Compare fan-out: one pod per (side, seed) ──────────────────────


class TestCompareFanout:
    def test_sft_fanout_shape(self):
        jobs = compare_fanout("sft", SHA, BASE_SHA, nonce="ab12", seeds=3, num_samples=1000)
        assert len(jobs) == 6
        pr_side = [j for j in jobs if j.sha == SHA]
        main_side = [j for j in jobs if j.sha == BASE_SHA]
        assert len(pr_side) == len(main_side) == 3
        # Seeds pair up across sides; each side gets its own W&B group.
        assert sorted(j.seed for j in pr_side) == [1, 2, 3]
        assert {j.group for j in pr_side} == {compare_group(SHA, "sft", "ab12", "pr")}
        assert {j.group for j in main_side} == {compare_group(SHA, "sft", "ab12", "main")}
        # Groups are keyed on the PR sha for both sides (one compare = one key).
        assert all(SHA[:7] in (j.group or "") for j in jobs)

    def test_groups_unique_per_launch(self):
        # Two compares at the same commit (rerun, or sft + ppo back to back)
        # must NOT share W&B groups — a shared group let one compare's runs
        # pollute the other's report (seen live on PR #243).
        a = compare_fanout("sft", SHA, BASE_SHA, nonce="aaaa", seeds=1, num_samples=10)
        b = compare_fanout("sft", SHA, BASE_SHA, nonce="bbbb", seeds=1, num_samples=10)
        p = compare_fanout("ppo", SHA, BASE_SHA, nonce="aaaa", seeds=1, start_from="x")
        assert {j.group for j in a}.isdisjoint({j.group for j in b})
        assert {j.group for j in a}.isdisjoint({j.group for j in p})

    def test_fanout_tags_mark_compare_runs(self):
        jobs = compare_fanout(
            "sft", SHA, BASE_SHA, nonce="ab12", seeds=1, num_samples=10, extra_tags=["pr:7"]
        )
        for job in jobs:
            assert f"cmp:{SHA[:7]}" in job.extra_tags
            assert "pr:7" in job.extra_tags
        sides = {t for j in jobs for t in j.extra_tags if t.startswith("cmp-side:")}
        assert sides == {"cmp-side:pr", "cmp-side:main"}

    def test_ppo_fanout_uses_same_checkpoint_both_sides(self):
        jobs = compare_fanout(
            "ppo", SHA, BASE_SHA, nonce="ab12", seeds=2, start_from="j0s5y2mc", total_timesteps=100
        )
        ppo_jobs = [j for j in jobs if isinstance(j, PpoJob)]
        assert len(ppo_jobs) == len(jobs) == 4
        assert {j.start_from for j in ppo_jobs} == {"j0s5y2mc"}

    def test_ppo_fanout_requires_start_from(self):
        try:
            compare_fanout("ppo", SHA, BASE_SHA, nonce="ab12")
            assert False, "expected ValueError"
        except ValueError:
            pass


# ── Job spec serialization (launcher → pod round trip) ─────────────


class TestJobSerialization:
    def test_round_trip_every_kind(self):
        jobs = [
            SftJob(sha=SHA, num_samples=5, seed=2, group="g", extra_tags=["pr:1"]),
            PpoJob(sha=SHA, start_from="j0s5y2mc", total_timesteps=7),
            SweepJob(sha=SHA, algo="ppo", sweep_path="e/p/s", agents_per_pod=2),
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

    def test_timestamps_are_iso8601(self):
        # 2026-07-06 15:16:10 UTC — readable in the RunPod console at a glance.
        name = pod_name("ppo", 1783350970, 1783356170, SHA)
        assert name == f"factorion-ci-ppo-c20260706T151610Z-d20260706T164250Z-{SHA[:7]}"

    def test_legacy_epoch_names_still_parse(self):
        # Pods launched before the ISO switch keep epoch-seconds names; the
        # watchdog must be able to reap them until they've all cycled out.
        meta = parse_pod_name(f"factorion-ci-ppo-c1783350970-d1783356170-{SHA[:7]}")
        assert meta is not None
        assert (meta.created, meta.deadline) == (1783350970, 1783356170)

    def test_foreign_names_rejected(self):
        for name in ("my-dev-pod", "factorion-ci-sft-nonsense", "", "ci-smoke-12345"):
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

    def test_never_touches_non_ci_pods(self):
        now = time.time()
        personal = self._pod("my-experiment-pod", uptime=10 * MAX_POD_AGE_SECONDS)
        assert decide_terminations([personal], now=now) == []

    def test_unparseable_ci_pod_reaped_after_max_age(self):
        now = time.time()
        young = self._pod("factorion-ci-sft-renamed", uptime=60)
        old = self._pod("factorion-ci-sft-old", uptime=MAX_POD_AGE_SECONDS + 1)
        doomed = decide_terminations([young, old], now=now)
        assert [p["name"] for p, _ in doomed] == ["factorion-ci-sft-old"]

    def test_reaps_a_pod_stuck_pulling_the_image(self):
        # A pod wedged in image pull has desiredStatus RUNNING but no runtime;
        # the deadline in its NAME must be enough to reap it (observed live:
        # two pods spent 50+ min pulling and sailed past their deadlines).
        now = time.time()
        stuck = {
            "id": "id-stuck",
            "name": pod_name("sft", int(now - 7200), int(now - 60), SHA),
            "desiredStatus": "RUNNING",
            "runtime": None,
        }
        doomed = decide_terminations([stuck], now=now)
        assert [p["id"] for p, _ in doomed] == ["id-stuck"]

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

    def test_compare_pods_budget_like_single_runs(self):
        (job, *_) = compare_fanout(
            "sft", SHA, BASE_SHA, nonce="ab12", seeds=1, num_samples=1_000_000
        )
        assert job.budget_seconds() == sft_budget_seconds(1_000_000, 1)


# ── /ci comment parsing ────────────────────────────────────────────


class TestParseComment:
    def test_command_and_assert_lines(self):
        body = (
            "/ci compare --seeds 3\n"
            "assert pr:val/thput > main:val/thput\n"
            "assert pr:val/acc >= 0.5\n"
        )
        tokens, assertions = parse_comment(body)
        assert tokens == ["compare", "--seeds", "3"]
        assert assertions == [
            "pr:val/thput > main:val/thput",
            "pr:val/acc >= 0.5",
        ]

    def test_surrounding_prose_is_ignored(self):
        body = "some context first\n\n/ci sft --num-samples 200000\nthanks!"
        tokens, assertions = parse_comment(body)
        assert tokens == ["sft", "--num-samples", "200000"]
        assert assertions == []

    def test_no_command(self):
        assert parse_comment("just a normal comment") == ([], [])


# ── Assertions ─────────────────────────────────────────────────────


def _rows_for_assertions():
    return [
        MetricRow("val/thput", 0.10, 0.01, 0.30, 0.01, 3, 3, 0.001, True),
        MetricRow("val/acc", 0.80, 0.01, 0.70, 0.01, 3, 3, 0.001, True),
    ]


class TestAssertions:
    def test_pr_beats_main_passes(self):
        r = evaluate_assertion("pr:val/thput > main:val/thput", _rows_for_assertions())
        assert r.passed

    def test_regression_fails(self):
        r = evaluate_assertion("pr:val/acc >= main:val/acc", _rows_for_assertions())
        assert not r.passed

    def test_numeric_threshold_and_aliases(self):
        rows = _rows_for_assertions()
        assert evaluate_assertion("test:val/thput >= 0.3", rows).passed
        assert evaluate_assertion("base:val/thput < 0.2", rows).passed

    def test_unknown_metric_fails_gracefully(self):
        r = evaluate_assertion("pr:val/nope > main:val/nope", _rows_for_assertions())
        assert not r.passed
        assert "not found" in r.detail

    def test_unparseable_fails_gracefully(self):
        r = evaluate_assertion("gibberish", _rows_for_assertions())
        assert not r.passed

    def test_approx_equal_within_default_tolerance(self):
        rows = [MetricRow("m", 0.5000, 0.0, 0.5004, 0.0, 3, 3, 0.9, True)]
        assert evaluate_assertion("pr:m == main:m", rows).passed
        assert evaluate_assertion("pr:m ~= 0.5", rows).passed

    def test_approx_equal_fails_outside_default_tolerance(self):
        r = evaluate_assertion("pr:val/acc == main:val/acc", _rows_for_assertions())
        assert not r.passed  # 0.70 vs 0.80 is nowhere near 1e-3
        assert "tolerance" in r.detail

    def test_approx_equal_custom_tolerance(self):
        rows = _rows_for_assertions()
        assert evaluate_assertion("pr:val/acc == main:val/acc +- 0.2", rows).passed
        assert evaluate_assertion("pr:val/acc == main:val/acc +/- 0.2", rows).passed
        assert not evaluate_assertion("pr:val/acc == main:val/acc +- 0.05", rows).passed

    def test_tolerance_rejected_on_ordering_comparators(self):
        r = evaluate_assertion("pr:val/acc > main:val/acc +- 0.2", _rows_for_assertions())
        assert not r.passed
        assert "only applies" in r.detail


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

    def test_unpaired_falls_back_to_welch(self):
        base = {s: {"m": 1.0 + 0.1 * s} for s in (1, 2, 3)}
        test = {s: {"m": 5.0 + 0.1 * s} for s in (7, 8, 9)}
        (row,) = compare_metric_rows(base, test)
        assert not row.paired
        assert row.p_value is not None

    def test_verdict_requires_significance(self):
        row = MetricRow("val/thput", 0.1, 0.1, 0.11, 0.1, 3, 3, 0.9, True)
        assert row.verdict() == ""

    def test_markdown_headline_outside_details(self):
        base, test = self._sides()
        base[1]["obscure/metric"] = 1.0
        test[1]["obscure/metric"] = 1.1
        rows = compare_metric_rows(base, test)
        md = render_compare_markdown(rows, "grp-base", "grp-test", headline=["val/thput"])
        before_details, details = md.split("<details>", 1)
        assert "val/thput" in before_details  # headline shown up front
        assert "obscure/metric" not in before_details  # long tail hidden…
        assert "obscure/metric" in details  # …but present in the full table


# ── Reporter (PR summary comments) ─────────────────────────────────


class TestReporter:
    def test_select_unreported_skips_marked_runs(self):
        candidates = [{"run_id": "aaa"}, {"run_id": "bbb"}]
        bodies = ["intro", "report\n<!-- factorion-ci-run:aaa -->"]
        assert select_unreported(candidates, bodies) == [{"run_id": "bbb"}]

    def test_run_summary_contains_marker_and_headline(self):
        md = run_summary_markdown(
            run_id="abc123",
            name="sft-s11-x",
            state="finished",
            url="https://wandb.ai/x/y/runs/abc123",
            kind="sft",
            sha7=SHA[:7],
            summary_flat={"val/thput_eot": 0.31, "val/acc": 0.9, "obscure/x": 1.0},
        )
        assert "<!-- factorion-ci-run:abc123 -->" in md
        before_details, details = md.split("<details>", 1)
        assert "val/thput_eot" in before_details
        assert "obscure/x" not in before_details
        assert "obscure/x" in details


class TestSelectHeadline:
    def test_sft_patterns_cover_every_lesson_and_head(self):
        names = [
            "val/thput_eot",
            "val/thput",  # not headline: only the EOT-respecting number leads
            "val/MOVE_ONE_ITEM/thput_eot",
            "val/SPLITTER_SPLIT/thput_eot",
            "val/SOME_FUTURE_LESSON_9/thput_eot",  # lessons matched, not hardcoded
            "val/MOVE_ONE_ITEM/acc",  # per-lesson acc stays in the long tail
            "val/acc",
            "val/tile_acc",
            "val/eot_acc",
            "val/eot_pos_recall",  # not an accuracy
            "perf/train_seconds",
            "train/loss",
        ]
        got = select_headline(names)
        assert got == [
            "val/thput_eot",
            "val/MOVE_ONE_ITEM/thput_eot",
            "val/SOME_FUTURE_LESSON_9/thput_eot",
            "val/SPLITTER_SPLIT/thput_eot",
            "val/acc",
            "val/eot_acc",
            "val/tile_acc",
            "perf/train_seconds",
        ]

    def test_ppo_patterns(self):
        names = [
            "eval/thput",  # eval/ stays in the long tail
            "eval/thput_eot",
            "rollout/thput",
            "rollout/reward",
            "rollout/length",
            "rollout/invalid_frac",
            "rollout/eot_rate",  # not headline
            "rollout/MOVE_ONE_ITEM/thput",
            "rollout/SOME_FUTURE_LESSON_9/thput",  # lessons matched, not hardcoded
            "rollout/MOVE_ONE_ITEM/reward",  # per-lesson non-thput stays in the tail
            "critic/explained_variance",
            "policy/entropy",
            "perf/update_seconds",
            "perf/rollout_seconds",
            "perf/eval_seconds",
            "perf/sps",
            "losses/value",
            "optim/lr",
        ]
        got = select_headline(names)
        assert got == [
            "rollout/thput",
            "rollout/reward",
            "rollout/length",
            "rollout/invalid_frac",
            "rollout/MOVE_ONE_ITEM/thput",
            "rollout/SOME_FUTURE_LESSON_9/thput",
            "perf/update_seconds",
            "perf/rollout_seconds",
            "perf/eval_seconds",
            "perf/sps",
        ]
