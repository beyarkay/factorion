"""Tests for the GRPO RL finetuning pipeline."""

import json
import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402
from grpo import GRPOArgs, train_grpo  # noqa: E402

ENV_ID = "factorion/FactorioEnv-v0-grpo-test"

# Tiny network so tests stay fast; these must match across the fake SFT
# checkpoint and the GRPOArgs that loads it.
TINY = dict(chan1=16, chan2=16, chan3=16, flat_dim=64)


@pytest.fixture(scope="module")
def registered_env():
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=FactorioEnv)


def _make_sft_checkpoint(path: str, size: int) -> None:
    """Stand in for a real SFT checkpoint: a freshly-initialised AgentCNN
    state_dict saved exactly the way sft.train_sft writes it."""
    envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "grpo-test")])
    agent = AgentCNN(envs, **TINY)
    envs.close()
    torch.save(agent.state_dict(), path)


class TestTrainGRPOEndToEnd:
    def test_requires_start_from(self, registered_env):
        """GRPO finetunes a prior — refuse to run without --start-from."""
        with pytest.raises(ValueError):
            train_grpo(GRPOArgs(seed=1, size=5, start_from="", **TINY))

    def test_produces_checkpoint_and_summary(self, registered_env, tmp_path):
        """End-to-end: train_grpo() runs, saves a checkpoint + summary, and
        the checkpoint reloads into a fresh AgentCNN."""
        size = 5
        sft_ckpt = str(tmp_path / "sft.pt")
        _make_sft_checkpoint(sft_ckpt, size)

        out_ckpt = str(tmp_path / "grpo.pt")
        summary = str(tmp_path / "grpo_summary.json")
        args = GRPOArgs(
            seed=1,
            size=size,
            start_from=sft_ckpt,
            num_iterations=2,
            num_grids=2,
            group_size=2,
            num_missing_max=2,
            checkpoint_path=out_ckpt,
            summary_path=summary,
            **TINY,
        )
        train_grpo(args)

        assert os.path.exists(out_ckpt), "GRPO checkpoint should be saved"
        assert os.path.exists(summary), "Summary JSON should be saved"

        with open(summary) as f:
            s = json.load(f)
        assert s["size"] == size
        assert s["start_from"] == sft_ckpt

        # The saved checkpoint must load back into a fresh AgentCNN (so
        # ppo.py --start_from and a follow-up GRPO run can both consume it).
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "grpo-test")])
        agent = AgentCNN(envs, **TINY)
        agent.load_state_dict(torch.load(out_ckpt))
        envs.close()
