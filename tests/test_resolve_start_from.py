"""Tests for the `--start-from` checkpoint resolver in ppo.py.

`--start-from` accepts either a local .pt path or a W&B run id (e.g. an SFT run
like 'j0s5y2mc'). `_resolve_start_from` picks the branch; `_resolve_wandb_checkpoint`
(shared with scripts/factory_builder.py) does the artifact download. These cover
both, so the "download the SFT checkpoint from a run id" contract documented on
the flag (and relied on by ci/jobs.py) stays wired up.
"""

import os
import sys

import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ppo  # noqa: E402
from ppo import _resolve_start_from, _resolve_wandb_checkpoint  # noqa: E402


# ── _resolve_start_from: local-path vs run-id branch ────────────────────────


def test_existing_local_path_returned_unchanged(tmp_path):
    ckpt = tmp_path / "sft_checkpoint.pt"
    torch.save({"x": 1}, ckpt)
    assert _resolve_start_from(str(ckpt), "factorion") == str(ckpt)


def test_missing_pt_path_raises_filenotfound(tmp_path):
    missing = tmp_path / "nope.pt"
    with pytest.raises(FileNotFoundError):
        _resolve_start_from(str(missing), "factorion")


def test_run_id_delegates_to_wandb_resolver(monkeypatch, tmp_path):
    """A value that is neither an existing path nor a *.pt is treated as a W&B
    run id and routed through the shared _resolve_wandb_checkpoint."""
    pt = tmp_path / "sft_checkpoint.pt"
    torch.save({"x": 1}, pt)
    captured = {}

    def fake_resolve(run_spec, project, entity):
        captured["args"] = (run_spec, project, entity)
        return str(pt), {"kind": "wandb", "run_id": run_spec}

    monkeypatch.setattr(ppo, "_resolve_wandb_checkpoint", fake_resolve)
    assert _resolve_start_from("j0s5y2mc", "factorion", entity="t") == str(pt)
    assert captured["args"] == ("j0s5y2mc", "factorion", "t")


# ── _resolve_wandb_checkpoint: artifact download ────────────────────────────


class _FakeArtifact:
    def __init__(self, atype, files_dir, name="sft-art:v0", created_at="2026-01-01"):
        self.type = atype
        self.name = name
        self.created_at = created_at
        self._files_dir = files_dir

    def download(self, root=None):  # real wandb downloads into `root`
        return self._files_dir


class _FakeRun:
    def __init__(self, artifacts, run_id="j0s5y2mc"):
        self._artifacts = artifacts
        self.id = run_id
        self.url = f"https://wandb.ai/x/factorion/runs/{run_id}"
        self.name = "sft-run"

    def logged_artifacts(self):
        return self._artifacts


def _patch_api(monkeypatch, run, default_entity="myteam"):
    import wandb

    captured = {}

    class _FakeApi:
        def __init__(self):
            self.default_entity = default_entity

        def run(self, run_path):
            captured["run_path"] = run_path
            return run

    monkeypatch.setattr(wandb, "Api", _FakeApi)
    return captured


def test_run_id_downloads_model_artifact_pt(tmp_path, monkeypatch):
    pt = tmp_path / "sft_checkpoint.pt"
    torch.save({"x": 1}, pt)
    (tmp_path / "sft_summary.json").write_text("{}")  # non-.pt sibling, ignored
    run = _FakeRun([_FakeArtifact("model", str(tmp_path))])
    captured = _patch_api(monkeypatch, run)

    path, source = _resolve_wandb_checkpoint("j0s5y2mc", "factorion", None)
    assert path == str(pt)
    # bare id -> "<default_entity>/<project>/<run_id>"
    assert captured["run_path"] == "myteam/factorion/j0s5y2mc"
    assert source["kind"] == "wandb"
    assert source["run_id"] == "j0s5y2mc"
    assert source["artifact"] == "sft-art:v0"


def test_full_path_bypasses_default_entity(tmp_path, monkeypatch):
    torch.save({"x": 1}, tmp_path / "ckpt.pt")
    run = _FakeRun([_FakeArtifact("model", str(tmp_path))])
    captured = _patch_api(monkeypatch, run)

    _resolve_wandb_checkpoint("user/factorion/abc123", "factorion", None)
    assert captured["run_path"] == "user/factorion/abc123"


def test_explicit_entity_used(tmp_path, monkeypatch):
    torch.save({"x": 1}, tmp_path / "ckpt.pt")
    run = _FakeRun([_FakeArtifact("model", str(tmp_path))])
    captured = _patch_api(monkeypatch, run)

    _resolve_wandb_checkpoint("abc123", "factorion", "myteam")
    assert captured["run_path"] == "myteam/factorion/abc123"


def test_run_with_no_model_artifact_raises(tmp_path, monkeypatch):
    run = _FakeRun([_FakeArtifact("dataset", str(tmp_path))])
    _patch_api(monkeypatch, run)
    with pytest.raises(RuntimeError, match="no artifacts of type=model"):
        _resolve_wandb_checkpoint("j0s5y2mc", "factorion", None)


def test_model_artifact_without_pt_raises(tmp_path, monkeypatch):
    (tmp_path / "summary.json").write_text("{}")  # model artifact, but no .pt
    run = _FakeRun([_FakeArtifact("model", str(tmp_path))])
    _patch_api(monkeypatch, run)
    with pytest.raises(RuntimeError, match="no .pt file"):
        _resolve_wandb_checkpoint("j0s5y2mc", "factorion", None)
