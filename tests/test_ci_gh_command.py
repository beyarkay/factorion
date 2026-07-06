"""End-to-end tests for the `/ci` PR-comment dispatcher (ci/gh_command.py).

Drives the real dispatch path — comment parsing, job-spec construction, pod
env assembly, PR feedback, commit statuses — with the RunPod and GitHub
layers mocked out, asserting exactly which pods and comments would be
created. The only untested seam left is the live RunPod/GitHub APIs, which
the first real `/ci` comment exercises after merge.
"""

import base64
import json

import pytest

from ci import gh_command

SHA = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def gh_ctx(monkeypatch):
    """Env + mocks for a /ci comment on PR #42; returns the capture dict."""
    captured = {"pods": [], "comments": [], "statuses": [], "edits": []}

    monkeypatch.setenv("PR_NUMBER", "42")
    monkeypatch.setenv("HEAD_SHA", SHA)
    monkeypatch.setenv(
        "COMMENT_URL", "https://github.com/beyarkay/factorion/pull/42#issuecomment-999"
    )
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-test")
    monkeypatch.setenv("WANDB_API_KEY", "wb-test")
    monkeypatch.setenv("GITHUB_TOKEN", "gh-test")
    monkeypatch.setenv("GITHUB_REPOSITORY", "beyarkay/factorion")

    def fake_create_pod(name, gpu_type, docker_args, env):
        captured["pods"].append(
            {"name": name, "gpu_type": gpu_type, "docker_args": docker_args, "env": env}
        )
        return {"id": f"pod-{len(captured['pods'])}"}

    monkeypatch.setattr("ci.runpod_api.create_pod", fake_create_pod)
    monkeypatch.setattr(gh_command, "_wandb_entity", lambda: "testent")
    monkeypatch.setattr(gh_command, "_missing_run_warnings", lambda infos: "")
    # Live pod/run statuses hit RunPod + W&B; give every pod a "running" pair.
    monkeypatch.setattr(
        gh_command,
        "_launch_statuses",
        lambda infos: {i["pod_id"]: ("🚀", "🚀") for i in infos},
    )

    def fake_post_comment(pr, body):
        captured["comments"].append((pr, body))
        return 9000 + len(captured["comments"])  # a comment id, like the real API

    monkeypatch.setattr(gh_command.github_api, "post_pr_comment", fake_post_comment)
    monkeypatch.setattr(
        gh_command.github_api,
        "update_pr_comment",
        lambda comment_id, body: captured["edits"].append((comment_id, body)),
    )
    monkeypatch.setattr(
        gh_command.github_api,
        "set_commit_status",
        lambda sha, state, context, description: captured["statuses"].append(
            (sha, state, context, description)
        ),
    )
    return captured


def _job_spec(pod):
    return json.loads(base64.b64decode(pod["env"]["FCI_JOB_B64"]))


class TestSftDispatch:
    def test_launches_one_pod_and_comments(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci sft --num-samples 200000")
        gh_command.main()

        (pod,) = gh_ctx["pods"]
        spec = _job_spec(pod)
        assert spec["kind"] == "sft"
        assert spec["sha"] == SHA
        assert spec["num_samples"] == 200000
        assert "pr:42" in spec["extra_tags"]
        # The pod gets the secrets + the pinned commit to clone.
        assert pod["env"]["WANDB_API_KEY"] == "wb-test"
        assert pod["env"]["FCI_SHA"] == SHA
        assert pod["name"].startswith("factorion-ci-sft-")

        ((pr, body),) = gh_ctx["comments"]
        assert pr == 42
        assert "pod-1" in body
        # The W&B run id is minted at launch and linked in the comment.
        run_id = pod["env"]["FCI_WANDB_RUN_ID"]
        assert len(run_id) == 8
        assert f"https://wandb.ai/testent/factorion/runs/{run_id}" in body
        assert "https://console.runpod.io/pods?id=pod-1" in body
        # Every CI comment links back to the /ci comment that triggered it.
        assert "Originally triggered by" in body
        assert "#issuecomment-999" in body


class TestPpoDispatch:
    def test_ppo_carries_checkpoint(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci ppo --start-from j0s5y2mc")
        gh_command.main()

        (pod,) = gh_ctx["pods"]
        spec = _job_spec(pod)
        assert spec["kind"] == "ppo"
        assert spec["start_from"] == "j0s5y2mc"
        assert spec["total_timesteps"] is None  # PpoArgs default applies


class TestCompareDispatch:
    def test_fans_out_pods_waits_and_gates(self, gh_ctx, monkeypatch):
        monkeypatch.setenv(
            "COMMENT_BODY",
            "/ci compare --seeds 2 --num-samples 1000\n"
            "assert pr:val/thput > main:val/thput\n",
        )
        # Base ref resolution + the W&B-touching steps are stubbed out.
        monkeypatch.setattr(gh_command, "resolve_ref", lambda ref: "b" * 40)
        waits = []
        monkeypatch.setattr(
            "ci.report.wait_for_groups", lambda **kw: waits.append(kw)
        )
        reports = []

        def fake_compare_report(main_group, pr_group, assertions=None):
            reports.append((main_group, pr_group, assertions))
            return "REPORT-MD", True

        monkeypatch.setattr("ci.report.compare_report", fake_compare_report)

        gh_command.main()

        # 2 seeds x 2 sides = 4 pods; sides split across the two shas.
        assert len(gh_ctx["pods"]) == 4
        shas = [_job_spec(p)["sha"] for p in gh_ctx["pods"]]
        assert shas.count(SHA) == 2 and shas.count("b" * 40) == 2

        # Waited for both groups, then reported with the comment's assertion.
        assert waits[0]["expect_each"] == 2
        assert reports[0][2] == ["pr:val/thput > main:val/thput"]

        # The launch comment starts all-pending and is edited in place with
        # live pod/run statuses (final refresh after the wait ends). The +1
        # on each count is the status legend line.
        launch_body = gh_ctx["comments"][0][1]
        assert launch_body.count("⏳") == 2 * len(gh_ctx["pods"]) + 1
        ((edit_id, edit_body),) = gh_ctx["edits"]
        assert edit_id == 9001  # edits target the launch comment, not a new one
        assert edit_body.count("🚀") == 2 * len(gh_ctx["pods"]) + 1

        # Status: pending at launch, success after the passing report.
        states = [s[1] for s in gh_ctx["statuses"]]
        assert states == ["pending", "success"]
        assert any("REPORT-MD" in body for _, body in gh_ctx["comments"])

    def test_failed_assertion_fails_the_check(self, gh_ctx, monkeypatch):
        monkeypatch.setenv(
            "COMMENT_BODY", "/ci compare --seeds 1\nassert pr:val/acc > 1000"
        )
        monkeypatch.setattr(gh_command, "resolve_ref", lambda ref: "b" * 40)
        monkeypatch.setattr("ci.report.wait_for_groups", lambda **kw: None)
        monkeypatch.setattr(
            "ci.report.compare_report",
            lambda main_group, pr_group, assertions=None: ("BAD-MD", False),
        )
        with pytest.raises(SystemExit) as exc:
            gh_command.main()
        assert exc.value.code == 1
        assert [s[1] for s in gh_ctx["statuses"]] == ["pending", "failure"]

    def test_launch_comment_echoes_job_spec(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci sft --num-samples 200000")
        gh_command.main()
        ((_, body),) = gh_ctx["comments"]
        assert '"num_samples": 200000' in body  # params visible up front
        assert "training_config.py" in body


class TestComparePpoDispatch:
    def test_positional_algo_ppo(self, gh_ctx, monkeypatch):
        monkeypatch.setenv(
            "COMMENT_BODY", "/ci compare ppo --start-from j0s5y2mc --seeds 1"
        )
        monkeypatch.setattr(gh_command, "resolve_ref", lambda ref: "b" * 40)
        monkeypatch.setattr("ci.report.wait_for_groups", lambda **kw: None)
        monkeypatch.setattr(
            "ci.report.compare_report",
            lambda main_group, pr_group, assertions=None: ("MD", True),
        )
        gh_command.main()

        assert len(gh_ctx["pods"]) == 2  # 1 seed x 2 sides
        specs = [_job_spec(p) for p in gh_ctx["pods"]]
        assert {s["kind"] for s in specs} == {"ppo"}
        assert {s["start_from"] for s in specs} == {"j0s5y2mc"}


class TestSweepDispatch:
    def test_positional_algo_sft(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci sweep sft --pods 2")
        monkeypatch.setattr(
            gh_command, "create_sweep", lambda algo, sha: f"me/factorion/swp-{algo}"
        )
        monkeypatch.setattr(
            gh_command,
            "read_sweep_config",
            lambda algo, sha: {
                "metric": {"name": "val/acc", "goal": "maximize"},
                "run_cap": 30,
                "parameters": {"lr": {}, "batch_size": {}},
            },
        )
        monkeypatch.setattr(gh_command, "_wait_for_sweep", lambda path: None)
        monkeypatch.setattr("ci.report.sweep_report", lambda path: "SWEEP-MD")
        gh_command.main()

        assert len(gh_ctx["pods"]) == 2
        specs = [_job_spec(p) for p in gh_ctx["pods"]]
        assert {s["kind"] for s in specs} == {"sweep"}
        assert {s["sweep_path"] for s in specs} == {"me/factorion/swp-sft"}
        # The launch comment shows what's being swept, not just pod ids.
        launch_body = gh_ctx["comments"][0][1]
        assert "val/acc" in launch_body and "run_cap 30" in launch_body
        assert any("SWEEP-MD" in body for _, body in gh_ctx["comments"])

    def test_sweep_requires_algo(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci sweep")
        with pytest.raises(SystemExit):
            gh_command.main()
        assert gh_ctx["pods"] == []


class TestBadInput:
    def test_unparseable_command_posts_usage(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci frobnicate --hard")
        with pytest.raises(SystemExit):
            gh_command.main()
        ((_, body),) = gh_ctx["comments"]
        assert "could not parse" in body
        assert gh_ctx["pods"] == []

    def test_help_covers_every_command_with_examples(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci help")
        gh_command.main()
        ((_, body),) = gh_ctx["comments"]
        for snippet in (
            "/ci sft --num-samples 200000",  # concrete example, not just grammar
            "/ci ppo --start-from",
            "/ci compare sft",
            "/ci compare ppo --start-from",
            "assert pr:val/thput > main:val/thput",
            "/ci sweep sft",
            "/ci pods",
            "/ci kill",
            "/ci watchdog",
        ):
            assert snippet in body, f"help is missing {snippet!r}"
        assert gh_ctx["pods"] == []
