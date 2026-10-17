"""Tests for RL fine-tuning of the denoiser (rl.py)."""

import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import rl  # noqa: E402
from denoise import DenoiserUNet, load_denoiser, world_to_state  # noqa: E402
from helpers import TINY  # noqa: E402
from rl import _factories, rollout, update  # noqa: E402
from training_config import RlArgs, SftArgs  # noqa: E402

REPO = os.path.join(os.path.dirname(__file__), "..")


def test_trains_toward_each_factorys_best_build(monkeypatch):
    """A factory's best build is a target; a factory whose builds tie has none."""
    scores, worlds = iter([0.1, 0.5, 0.2, 0.3, 0.3, 0.3]), []

    def fake_score(world, *_):
        worlds.append(world)
        return {"score": next(scores), "thput": 0.0, "entity_cost": 0.0}

    monkeypatch.setattr(rl, "score_world", fake_score)
    model = DenoiserUNet(**TINY)
    args = RlArgs(groups=2, group_size=3)
    stream = _factories(args)
    roll = rollout(model, [next(stream), next(stream)], args, SftArgs(gen_steps=2, gen_refine=1), None)
    assert roll["best"].shape[0] == 1
    assert torch.equal(roll["best"][0].cpu(), torch.as_tensor(world_to_state(worlds[1])))
    stats = update(model, torch.optim.Adam(model.parameters(), lr=1e-3), roll, args, SftArgs())
    assert stats["losses/raft"] > 0


def test_rl_trains_a_denoiser_checkpoint(tmp_path):
    """`rl.py --start-from <denoiser>` runs end to end and saves a checkpoint
    the denoiser loader reads back."""
    model = DenoiserUNet(**TINY)
    start = tmp_path / "dn.pt"
    torch.save({"arch": model.arch, "model": model.state_dict()}, start)
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "rl.py"), "--start-from", str(start), "--groups", "2",
         "--group-size", "2", "--total-timesteps", "64", "--eval-best-of", "1", "--size", "15"],
        cwd=tmp_path,
        env={**os.environ, "CI": "1", "WANDB_MODE": "disabled"},
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "RAFT on cpu: 2 iterations" in proc.stdout
    (saved,) = (tmp_path / "artifacts").glob("raft-*.pt")
    assert load_denoiser(saved).arch == model.arch
