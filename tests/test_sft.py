"""Tests for SFT pre-training pipeline."""

import itertools
import json
import os
import re
import sys
from typing import Any

import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import TINY, rs_throughput
from denoise import load_denoiser
from factorion import Channel, LessonKind, build_factory
from sft import (
    SftArgs,
    _artifact_name,
    _iter_factories,
    boxed_factory,
    _humanize_count,
    _humanize_lr,
    build_lr_schedule,
    noise_levels,
    train_sft,
)

# Generation eval shrunk to a couple of short chains per factory.
FAST_EVAL: dict[str, Any] = dict(
    eval_seeds_per_kind=1, box_eval_seeds_per_kind=1, eval_best_of=2, gen_steps=2, gen_refine=2
)


class TestNoiseLevels:
    def test_point_masses_and_range(self):
        args = SftArgs(noise_zero_frac=0.2, noise_one_frac=0.3)
        t = noise_levels(20_000, args, torch.device("cpu"))
        assert ((t >= 0) & (t <= 1)).all()
        assert (t == 0).float().mean() == pytest.approx(0.2, abs=0.02)
        assert (t == 1).float().mean() == pytest.approx(0.3, abs=0.02)


class TestBoxedFactory:
    def test_lesson_keeps_its_throughput_inside_its_box(self):
        for kind in (LessonKind.OPPOSITE_SIDES_1IN, LessonKind.ROUTE_BELTS):
            boxed = [(seed, f) for seed in range(6) if (f := boxed_factory(kind, seed, 32)) is not None]
            assert boxed, kind
            for seed, f in boxed:
                buildable = f.world_CWH[Channel.FOOTPRINT.value].bool()
                xs, ys = torch.nonzero(buildable, as_tuple=True)
                w, h = int(xs.max() - xs.min()) + 1, int(ys.max() - ys.min()) + 1
                assert int(buildable.sum()) == w * h and 5 <= min(w, h) and max(w, h) <= 32
                assert not f.world_CWH[Channel.ENTITIES.value][~buildable].any()
                square = build_factory(size=min(w, h), kind=kind, seed=seed)
                assert square is not None
                thput = [rs_throughput(x.world_CWH.permute(1, 2, 0))[0] for x in (f, square)]
                assert thput[0] == thput[1] > 0

    def test_stream_yields_boxed_factories(self):
        states = itertools.islice(_iter_factories(32, 1, 0, 1, box_frac=1.0), 4)
        assert any((s[Channel.FOOTPRINT.value] == 0).any() for s in states)


class TestTrainSFTEndToEnd:
    def test_trains_evaluates_and_checkpoints(self, tmp_path):
        ckpt, summary = tmp_path / "ckpt.pt", tmp_path / "summary.json"
        args = SftArgs(
            size=15, num_samples=64, batch_size=16, **FAST_EVAL,
            checkpoint_path=str(ckpt), summary_path=str(summary), **TINY,
        )
        model = train_sft(args)
        out = json.loads(summary.read_text())
        assert out["samples_seen"] == 64 and 0.0 <= out["best_val_throughput"] <= 1.0
        loaded = load_denoiser(ckpt)
        assert loaded.arch == model.arch
        for (k, a), b in zip(loaded.state_dict().items(), model.state_dict().values()):
            assert torch.equal(a, b.cpu()), k

        # --start-from resumes those weights (and the noise prior with them).
        args.start_from, args.num_samples = str(ckpt), 16
        args.checkpoint_path = str(tmp_path / "ckpt2.pt")
        assert torch.equal(train_sft(args).noise_prior.cpu(), loaded.noise_prior)


@pytest.fixture(scope="module")
def tracked_run(tmp_path_factory):
    """One tiny tracked run, capturing what it sends to W&B."""
    import wandb
    from unittest.mock import MagicMock

    captured: dict = {"logged": {}}
    fake_run = MagicMock()
    fake_run.url = "http://test/run"
    fake_run.summary = {}
    fake_run.log.side_effect = lambda d, *a, **k: captured["logged"].update(d)

    def fake_init(*a, **k):
        captured["name"] = k.get("name")
        return fake_run

    def fake_artifact(*a, **k):
        captured["metadata"] = k.get("metadata", {})
        return MagicMock()

    tmp_path = tmp_path_factory.mktemp("tracked")
    args = SftArgs(
        seed=1,
        size=5,
        num_samples=200,
        batch_size=32,
        **TINY,
        **FAST_EVAL,
        track=True,
        min_upload_minutes=0,
        checkpoint_path=str(tmp_path / "k.pt"),
        summary_path=str(tmp_path / "k.json"),
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(wandb, "init", fake_init)
        mp.setattr(wandb, "Artifact", fake_artifact)
        train_sft(args)  # must not raise at the artifact step
    return args, fake_run, captured


class TestTrackedArtifact:
    """The track=True checkpoint-upload path. No test exercised it, so #146's
    chan1/2/3 -> layers rename left a stale args.chan1 in the artifact metadata
    that crashed every tracked run with AttributeError *after* training — only
    at upload. This pins the path end-to-end."""

    def test_tracked_run_finishes_with_arch_metadata(self, tracked_run):
        _, _, captured = tracked_run
        meta = captured["metadata"]
        assert meta["width"] == 16
        assert meta["attn_layers"] == 1

    def test_run_named_by_artifact_name(self, tracked_run):
        """The W&B run name is the full hyperparameter signature (== the model
        artifact name) so runs are distinguishable in the table — not all
        "sft-{size}x{size}"."""
        args, _, captured = tracked_run
        assert captured["name"] == _artifact_name(args)


    def test_each_new_best_replaces_the_runs_previous_upload(self, monkeypatch, tmp_path):
        """A run uploads each strictly better val/thput checkpoint as it trains,
        so one killed mid-run still leaves its best in W&B, and deletes its own
        previous upload each time, so it holds one version at the end."""
        import itertools
        import wandb
        from unittest.mock import MagicMock

        import sft

        thputs = itertools.chain([0.1, 0.1, 0.3], itertools.repeat(0.2))
        real_generation = sft.generation_metrics

        def fake_generation(model, factories, args, *a, **k):
            metrics = real_generation(model, factories, args, *a, **k)
            metrics[f"val/thput@{args.eval_best_of}"] = next(thputs)
            return metrics

        artifacts, uploads = [], []

        def fake_artifact(*a, **k):
            artifacts.append(MagicMock())
            return artifacts[-1]

        def fake_log_artifact(artifact, aliases):
            uploads.append(MagicMock())
            return uploads[-1]

        fake_run = MagicMock()
        fake_run.url = "http://test/run"
        fake_run.summary = {}
        fake_run.log_artifact.side_effect = fake_log_artifact
        monkeypatch.setattr(wandb, "init", lambda *a, **k: fake_run)
        monkeypatch.setattr(wandb, "Artifact", fake_artifact)
        monkeypatch.setattr(sft, "generation_metrics", fake_generation)

        args = SftArgs(
            seed=1,
            size=5,
            num_samples=400,
                batch_size=32,
            **TINY,
            **FAST_EVAL,
            track=True,
            min_upload_minutes=0,
            eval_every_n_samples=64,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)

        # 0.1 and 0.3 upload mid-run (the 0.1 tie and the later 0.2s don't),
        # then the final checkpoint + summary.
        assert len(uploads) == 3
        assert all(u.delete.call_count == 1 for u in uploads[:-1])
        uploads[-1].delete.assert_not_called()
        final_files = [c.args[0] for c in artifacts[-1].add_file.call_args_list]
        assert final_files == [args.checkpoint_path, args.summary_path]

    def test_run_younger_than_min_upload_minutes_uploads_nothing(self, monkeypatch, tmp_path):
        import wandb
        from unittest.mock import MagicMock

        fake_run = MagicMock()
        fake_run.url = "http://test/run"
        fake_run.summary = {}
        monkeypatch.setattr(wandb, "init", lambda *a, **k: fake_run)
        args = SftArgs(
            seed=1,
            size=5,
            num_samples=200,
                batch_size=32,
            **TINY,
            **FAST_EVAL,
            track=True,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)
        fake_run.log_artifact.assert_not_called()
        assert os.path.exists(args.checkpoint_path)

    def test_generation_metrics_are_logged_as_best_of_n(self, tracked_run):
        """Every metric from the best-of-N generation eval carries @N, and no
        plain key (which would read as best-of-1) duplicates it."""
        args, fake_run, captured = tracked_run
        logged = captured["logged"]

        n = args.eval_best_of
        assert f"val/thput@{n}" in logged
        assert f"val/box/thput@{n}" in logged
        assert f"val/inserters_no_input@{n}" in logged
        assert f"val/unpowered@{n}" in logged
        assert any(k.endswith(f"/MOVE_ONE_ITEM/thput@{n}") for k in logged)
        assert all(re.search(r"@\d+$", k) for k in logged if "thput" in k or "inserters_no" in k)
        assert f"best_val_throughput@{n}" in fake_run.summary


class TestArtifactNameHelpers:
    """The W&B artifact name encodes hyperparams so runs with identical
    configs collapse into versions of one artifact. These tests pin the
    format so accidentally renaming a hyperparam doesn't silently
    fragment the artifact namespace."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (500, "500"),
            (1_000, "1k"),
            (50_000, "50k"),
            (200_000, "200k"),
            (1_000_000, "1M"),
            (2_500_000, "2.5M"),
        ],
    )
    def test_humanize_count(self, n, expected):
        assert _humanize_count(n) == expected

    @pytest.mark.parametrize(
        "lr,expected",
        [
            (1e-3, "1e-3"),
            (3e-4, "3e-4"),
            (1e-4, "1e-4"),
            (5e-4, "5e-4"),
        ],
    )
    def test_humanize_lr_round(self, lr, expected):
        """Mantissas that are clean integers don't get a decimal point."""
        assert _humanize_lr(lr) == expected

    def test_humanize_lr_zero(self):
        assert _humanize_lr(0) == "0"

    def test_artifact_name_larger_run(self):
        args = SftArgs(
            size=16,
            num_samples=200_000,
            batch_size=1024,
            lr=3e-4,
            width=48,
            levels=3,
        )
        assert _artifact_name(args) == "dn-s16-n200k-bs1024-lr3e-4-w48x3b2a4"

    def test_artifact_name_stable_across_runs(self):
        """Two SftArgs with the same hyperparams must produce the same
        artifact name (otherwise we lose version collapsing)."""
        a = SftArgs(seed=1)
        b = SftArgs(seed=999)  # seed isn't part of the artifact name
        assert _artifact_name(a) == _artifact_name(b)


class TestLRSchedule:
    """The WSD schedule should warm up to `args.lr` over an absolute step
    count, hold `args.lr` flat, and cool down to `args.lr * args.min_lr_ratio`
    over the final `cooldown_frac` of the run."""

    def _step_n(self, scheduler, optimizer, n):
        lrs = []
        for _ in range(n):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        return lrs

    def _lrs(self, args, total_steps, n=None):
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, total_steps, args)
        return self._step_n(scheduler, optimizer, n or total_steps)

    def test_warmup_stable_decay(self):
        args = SftArgs(lr=1e-3, warmup_steps=10, cooldown_frac=0.2, min_lr_ratio=0.01)
        lrs = self._lrs(args, 100)

        assert lrs[0] == pytest.approx(args.lr / 10, rel=1e-6)
        assert lrs[10:81] == pytest.approx([args.lr] * 71, rel=1e-9)
        # (1-sqrt) drops fast: a quarter into the cooldown it is already halfway.
        assert lrs[85] == pytest.approx(args.lr * 0.5, rel=0.02)
        assert lrs[-1] == pytest.approx(args.lr * args.min_lr_ratio, abs=args.lr * 1e-9)

    def test_shared_prefix_across_run_lengths(self):
        """The property the schedule exists for: two runs of different budgets
        are on the same LR at the same step until the shorter one starts
        cooling, so their curves are directly comparable."""
        args = SftArgs(lr=1e-3, warmup_steps=50, cooldown_frac=0.2)
        short = self._lrs(args, 1_000)
        long = self._lrs(args, 100_000, n=800)
        assert short[:800] == pytest.approx(long[:800], rel=1e-9)

    def test_handles_run_shorter_than_warmup(self):
        args = SftArgs(lr=1e-3, warmup_steps=300)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, 1, args)
        optimizer.step()
        scheduler.step()  # must not raise
