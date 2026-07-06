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
    captured = {"pods": [], "comments": [], "statuses": []}

    monkeypatch.setenv("PR_NUMBER", "42")
    monkeypatch.setenv("HEAD_SHA", SHA)
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
    monkeypatch.setattr(
        gh_command.github_api,
        "post_pr_comment",
        lambda pr, body: captured["comments"].append((pr, body)),
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

        def fake_compare_report(base_group, test_group, assertions=None):
            reports.append((base_group, test_group, assertions))
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

        # Status: pending at launch, success after the passing report.
        states = [s[1] for s in gh_ctx["statuses"]]
        assert states == ["pending", "success"]
        assert any("REPORT-MD" in body for _, body in gh_ctx["comments"])

    def test_failed_assertion_fails_the_check(self, gh_ctx, monkeypatch):
        monkeypatch.setenv(
            "COMMENT_BODY", "/ci compare-report\nassert pr:val/acc > 1000"
        )
        monkeypatch.setattr(
            "ci.report.compare_report",
            lambda base_group, test_group, assertions=None: ("BAD-MD", False),
        )
        with pytest.raises(SystemExit) as exc:
            gh_command.main()
        assert exc.value.code == 1
        assert gh_ctx["statuses"][-1][1] == "failure"
        assert gh_ctx["pods"] == []  # compare-report launches nothing


class TestBadInput:
    def test_unparseable_command_posts_usage(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci frobnicate --hard")
        with pytest.raises(SystemExit):
            gh_command.main()
        ((_, body),) = gh_ctx["comments"]
        assert "could not parse" in body
        assert gh_ctx["pods"] == []

    def test_help(self, gh_ctx, monkeypatch):
        monkeypatch.setenv("COMMENT_BODY", "/ci help")
        gh_command.main()
        ((_, body),) = gh_ctx["comments"]
        assert "/ci compare" in body
        assert gh_ctx["pods"] == []
