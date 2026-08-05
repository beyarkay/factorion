"""The EOT head's weight in the entropy bonus (PpoArgs.ent_mult_eot, #235).

Bernoulli entropy peaks at p(eot)=0.5 — expected episode length ~2 — so PPO
excludes the EOT head from the entropy bonus by default; every other consumer
keeps the plain six-head sum.
"""

import os
import sys

import gymnasium as gym
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import AgentCNN, make_env  # noqa: E402
from training_config import PpoArgs  # noqa: E402
from factorion import Channel  # noqa: E402

ENV_ID = "factorion/FactorioEnv-v0-eot-entropy-test"


@pytest.fixture(scope="module")
def envs():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")
    return gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, False, 5, "test") for i in range(2)]
    )


def _entropy_and_heads(agent, obs):
    """The summed entropy the PPO loss consumes, next to the unweighted
    per-head means the policy/* metrics log."""
    torch.manual_seed(0)
    out = agent.sample_action(obs)
    return out["entropy"].mean(), dict(agent._last_head_entropy)


def test_ppo_default_excludes_the_eot_head(envs):
    """At the PpoArgs default the loss entropy is the sum of the five
    placement heads — the EOT head contributes nothing to the bonus."""
    assert PpoArgs.ent_mult_eot == 0.0
    agent = AgentCNN(envs, layers=(16, 16, 16))
    agent.ent_mult_eot = PpoArgs.ent_mult_eot
    obs = torch.randn(4, len(Channel), 5, 5)
    total, heads = _entropy_and_heads(agent, obs)
    without_eot = sum(v for k, v in heads.items() if k != "eot")
    assert total.item() == pytest.approx(without_eot.item(), abs=1e-5)


def test_weight_one_recovers_the_plain_sum(envs):
    """--ent-mult-eot 1.0 (and AgentCNN's own default, which SFT and the mod
    server see) is exactly the previous behaviour."""
    agent = AgentCNN(envs, layers=(16, 16, 16))
    assert agent.ent_mult_eot == 1.0
    obs = torch.randn(4, len(Channel), 5, 5)
    total, heads = _entropy_and_heads(agent, obs)
    assert total.item() == pytest.approx(sum(heads.values()).item(), abs=1e-5)


def test_stashed_head_entropies_stay_unweighted(envs):
    """policy/entropy_eot must stay comparable across weightings: zeroing the
    bonus must not zero the logged EOT entropy."""
    agent = AgentCNN(envs, layers=(16, 16, 16))
    agent.ent_mult_eot = 0.0
    obs = torch.randn(4, len(Channel), 5, 5)
    _, heads = _entropy_and_heads(agent, obs)
    assert heads["eot"].item() > 0.0
