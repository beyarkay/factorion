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
from grpo import GRPOArgs, policy_step, train_grpo  # noqa: E402

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


class TestPolicyStep:
    """policy_step must be a faithful, temperature-aware reimplementation of
    AgentCNN.get_action_and_value's action path."""

    def _agent_and_obs(self, size=5, batch=6, seed=0):
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "grpo-test")])
        agent = AgentCNN(envs, **TINY)
        envs.close()
        torch.manual_seed(seed)
        # plausible integer observation in the catalog id range
        obs = torch.randint(
            0, 3, (batch, len(agent_channels()), size, size), dtype=torch.float32
        )
        return agent, obs

    def test_t1_matches_native_logp_and_entropy(self, registered_env):
        """At T=1.0, scoring an action with policy_step yields the same logp
        and entropy that get_action_and_value reports for that same action."""
        agent, obs = self._agent_and_obs()
        agent.eval()
        with torch.no_grad():
            torch.manual_seed(123)
            action_d, logp_native, ent_native, _v = agent.get_action_and_value(obs)
            # native action dict -> (B, 6) [x, y, ent, dir, item, misc]
            x, y = action_d["xy"].unbind(dim=1)
            action_B6 = torch.stack(
                [x, y, action_d["entity"], action_d["direction"], action_d["item"], action_d["misc"]],
                dim=1,
            )
            _out, logp_mine, ent_mine = policy_step(
                agent, obs, temperature=1.0, action_B6=action_B6
            )
        assert torch.allclose(logp_mine, logp_native, atol=1e-5), (
            logp_mine,
            logp_native,
        )
        assert torch.allclose(ent_mine, ent_native, atol=1e-5)

    def test_scoring_roundtrips_sampled_action(self, registered_env):
        """Sampling then scoring the returned action reproduces its logp
        (the (x,y)->tile_idx reconstruction is exact)."""
        agent, obs = self._agent_and_obs(seed=1)
        agent.eval()
        gen = torch.Generator().manual_seed(7)
        with torch.no_grad():
            action_B6, logp_sampled, _e = policy_step(
                agent, obs, temperature=1.3, generator=gen
            )
            _out, logp_scored, _e2 = policy_step(
                agent, obs, temperature=1.3, action_B6=action_B6
            )
        assert torch.allclose(logp_sampled, logp_scored, atol=1e-6)

    def test_temperature_increases_entropy(self, registered_env):
        """Higher temperature flattens the head distributions -> more entropy."""
        agent, obs = self._agent_and_obs(seed=2)
        agent.eval()
        with torch.no_grad():
            _a, _lp, ent_cold = policy_step(agent, obs, temperature=0.5, action_B6=_dummy_action(obs))
            _a, _lp, ent_hot = policy_step(agent, obs, temperature=2.0, action_B6=_dummy_action(obs))
        assert (ent_hot > ent_cold).all()


def agent_channels():
    from factorion import Channel

    return list(Channel)


def _dummy_action(obs):
    B = obs.shape[0]
    return torch.zeros((B, 6), dtype=torch.long)
