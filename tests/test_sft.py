"""Tests for the denoising trainer (sft.py)."""

import json
import os
import sys
from typing import Any

import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import LessonKind, build_factory  # noqa: E402
from denoise import DenoiserUNet, load_denoiser, world_to_state  # noqa: E402
from sft import SftArgs, build_lr_schedule, noise_levels, throughput_pg_loss, train_sft  # noqa: E402

TINY: dict[str, Any] = dict(width=16, levels=2, blocks=1, attn_layers=1, attn_heads=4)


def _batch(n=4, size=15):
    factories = [f for s in range(n) if (f := build_factory(size=size, kind=LessonKind.SAME_SIDE_1IN, seed=s))]
    assert len(factories) == n
    state = torch.stack([torch.from_numpy(world_to_state(f.world_CWH)) for f in factories]).long()
    return state, torch.tensor([f.max_throughput for f in factories])


class TestNoiseLevels:
    def test_point_masses_and_range(self):
        args = SftArgs(noise_zero_frac=0.2, noise_one_frac=0.3)
        t = noise_levels(20_000, args, torch.device("cpu"))
        assert ((t >= 0) & (t <= 1)).all()
        assert (t == 0).float().mean() == pytest.approx(0.2, abs=0.02)
        assert (t == 1).float().mean() == pytest.approx(0.3, abs=0.02)


class TestThroughputLoss:
    def test_reinforces_sampled_factories(self):
        from denoise import free_mask

        state, max_thput = _batch()
        model = DenoiserUNet(**TINY)
        logits = model(state)
        args = SftArgs(rl_inputs=2, rl_samples=3, rl_max_noise=0.5)
        loss, score = throughput_pg_loss(logits, state, free_mask(state), max_thput, torch.zeros(4), args)
        loss.backward()
        assert torch.isfinite(loss) and score >= 0
        assert model.head[-1].weight.grad is not None

    def test_skips_heavily_corrupted_inputs(self):
        from denoise import free_mask

        state, max_thput = _batch(2)
        logits = DenoiserUNet(**TINY)(state)
        loss, score = throughput_pg_loss(logits, state, free_mask(state), max_thput, torch.ones(2), SftArgs())
        assert float(loss) == 0.0 and score == 0.0


class TestTrainSFTEndToEnd:
    def test_trains_evaluates_and_checkpoints(self, tmp_path):
        ckpt, summary = tmp_path / "ckpt.pt", tmp_path / "summary.json"
        args = SftArgs(
            size=15, num_samples=64, batch_size=16, eval_seeds_per_kind=1,
            eval_best_of=2, gen_steps=2, gen_refine=2, rl_inputs=2, rl_samples=2, rl_max_noise=1.0,
            checkpoint_path=str(ckpt), summary_path=str(summary), **TINY,
        )
        model = train_sft(args)
        out = json.loads(summary.read_text())
        assert out["samples_seen"] == 64 and 0.0 <= out["best_val_throughput"] <= 1.0
        loaded = load_denoiser(ckpt)
        assert loaded.arch == model.arch
        for (k, a), b in zip(loaded.state_dict().items(), model.state_dict().values()):
            assert torch.equal(a, b), k

        # --start-from resumes those weights (and the noise prior with them).
        args.start_from, args.num_samples = str(ckpt), 16
        args.checkpoint_path = str(tmp_path / "ckpt2.pt")
        assert torch.equal(train_sft(args).noise_prior, loaded.noise_prior)


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

    def _lrs(self, args, total_steps):
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, total_steps, args)
        return self._step_n(scheduler, optimizer, total_steps)

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
        long = self._lrs(args, 100_000)
        assert short[:800] == pytest.approx(long[:800], rel=1e-9)

    def test_handles_run_shorter_than_warmup(self):
        args = SftArgs(lr=1e-3, warmup_steps=300)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, 1, args)
        optimizer.step()
        scheduler.step()  # must not raise
