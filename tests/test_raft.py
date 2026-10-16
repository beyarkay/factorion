"""Tests for RAFT finetuning of the denoiser (raft.py)."""

import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import raft  # noqa: E402
from denoise import DenoiserUNet, load_denoiser, world_to_state  # noqa: E402
from raft import _factories, rollout, update  # noqa: E402
from training_config import PpoArgs, SftArgs  # noqa: E402

TINY = dict(width=16, levels=2, blocks=1, attn_layers=1, attn_heads=4)
REPO = os.path.join(os.path.dirname(__file__), "..")


def test_trains_toward_each_factorys_best_build(monkeypatch):
    """A factory's best build is a target; a factory whose builds tie has none."""
    scores, worlds = iter([0.1, 0.5, 0.2, 0.3, 0.3, 0.3]), []

    def fake_score(world, *_):
        worlds.append(world)
        return {"score": next(scores), "thput": 0.0, "entity_cost": 0.0}

    monkeypatch.setattr(raft, "score_world", fake_score)
    model = DenoiserUNet(**TINY)
    args = PpoArgs(raft_groups=2, raft_group_size=3)
    stream = _factories(args)
    roll = rollout(model, [next(stream), next(stream)], args, SftArgs(gen_steps=2, gen_refine=1), None)
    assert roll["best"].shape[0] == 1
    assert torch.equal(roll["best"][0].cpu(), torch.as_tensor(world_to_state(worlds[1])))
    stats = update(model, torch.optim.Adam(model.parameters(), lr=1e-3), roll, args, SftArgs())
    assert stats["losses/raft"] > 0


def test_ppo_trains_a_denoiser_checkpoint_with_raft(tmp_path):
    """`ppo.py --start-from <denoiser>` runs RAFT end to end and saves a
    checkpoint the denoiser loader reads back."""
    model = DenoiserUNet(**TINY)
    start = tmp_path / "dn.pt"
    torch.save({"arch": model.arch, "model": model.state_dict()}, start)
    proc = subprocess.run(
        [sys.executable, os.path.join(REPO, "ppo.py"), "--start-from", str(start), "--raft-groups", "2",
         "--raft-group-size", "2", "--total-timesteps", "64", "--eval-best-of", "1", "--size", "15"],
        cwd=tmp_path,
        env={**os.environ, "CI": "1", "WANDB_MODE": "disabled"},
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "RAFT on cpu: 2 iterations" in proc.stdout
    (saved,) = (tmp_path / "artifacts").glob("raft-*.pt")
    assert load_denoiser(saved).arch == model.arch
