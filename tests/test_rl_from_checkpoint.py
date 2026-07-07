"""End-to-end tests for the RL-from-SFT-checkpoint setup.

Covers the four behaviours that make `ppo.py --start_from <sft-ckpt>` a
sensible RL fine-tuning run rather than RL-from-scratch:

1. throughput-dominant reward (solution-matching shaping off by default),
2. full-blank build-from-empty task by default (num_missing_entities=inf),
3. end-of-turn as a trained Bernoulli *action* that ends the episode,
4. the critic warm-up actor/critic param split + freeze.
"""

import os
import sys

import numpy as np
import pytest
import torch
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import AgentCNN, PpoArgs, FactorioEnv, make_env, layer_init  # noqa: E402
from helpers import Channel  # noqa: E402
from factorion import LessonKind  # noqa: E402

NUM_CHANNELS = len(Channel)
ENV_ID = "factorion/FactorioEnv-v0-rlckpt-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


@pytest.fixture()
def envs(registered_env):
    return gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, False, 5, "test") for i in range(2)]
    )


@pytest.fixture()
def agent(envs):
    return AgentCNN(envs, layers=(16, 16, 16))


class TestFullBlankDefault:
    def test_reset_without_option_fully_blanks(self, registered_env):
        """Omitting num_missing_entities blanks the whole factory (inf),
        identical to passing inf explicitly, and removes ≥1 entity."""
        env = FactorioEnv(size=8)
        env.reset(seed=3, options={"num_missing_entities": float("inf")})
        removed_inf = env.min_entities_required
        env.reset(seed=3, options={})  # no num_missing_entities -> default inf
        removed_default = env.min_entities_required
        assert removed_default == removed_inf
        assert removed_default > 0

    def test_default_blank_removes_more_than_partial(self, registered_env):
        """The default (full) blank removes strictly more entity units than a
        1-unit blank. Pin the lesson kind so both resets build the identical
        factory (env.reset reseeds build_factory, but blank_entities draws
        from the global RNG, so counts are the stable invariant to assert)."""
        env = FactorioEnv(size=8)
        env.reset(seed=7, options={"kind": LessonKind.MOVE_ONE_ITEM, "num_missing_entities": 1})
        removed_partial = env.min_entities_required
        env.reset(seed=7, options={"kind": LessonKind.MOVE_ONE_ITEM})  # default full blank
        removed_full = env.min_entities_required
        assert removed_full > removed_partial

    def test_make_env_max_steps_is_size_squared(self, registered_env):
        """A from-empty build needs ~size*size placements, so make_env sizes
        max_steps to size*size (not the old 2*size)."""
        env = make_env(ENV_ID, 0, False, 7, "t")()
        assert env.unwrapped.max_steps == 49


class TestEotAction:
    def test_eot_is_part_of_the_action(self, agent):
        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        action, logp, entropy, value = agent.get_action_and_value(obs)
        assert "eot" in action
        assert action["eot"].shape == (4,)
        assert set(action["eot"].unique().tolist()) <= {0.0, 1.0}
        # eot's log-prob/entropy fold into the joint action distribution.
        assert logp.shape == (4,)
        assert entropy.shape == (4,)

    def test_eot_logprob_roundtrips(self, agent):
        """Recomputing log-prob from a stored 7-dim action (incl. eot at
        index 6) must match the sampled log-prob, so PPO's importance ratio
        is well-defined and the eot head actually gets trained."""
        torch.manual_seed(0)
        obs = torch.randn(3, NUM_CHANNELS, 5, 5)
        action, logp, _, _ = agent.get_action_and_value(obs)
        x_B, y_B = action["xy"].unbind(dim=1)
        action_BA = torch.stack(
            [
                x_B, y_B,
                action["entity"], action["direction"],
                action["item"], action["misc"],
                action["eot"].long(),
            ],
            dim=1,
        )
        assert action_BA.shape == (3, 7)
        _, logp_recomputed, _, _ = agent.get_action_and_value(obs, action_BA)
        torch.testing.assert_close(logp, logp_recomputed)


class TestEotTerminationAndMetrics:
    def _action(self, eot):
        return {
            "xy": np.array([0, 0]),
            "entity": 0, "direction": 0, "item": 0, "misc": 0,
            "eot": int(eot),
        }

    def test_eot_terminates_and_records_episode(self, registered_env):
        """eot=1 must terminate via the env so the RecordEpisodeStatistics
        wrapper (added by make_env) emits "episode" — i.e. eot-ended episodes
        are NOT missing from the metric averages."""
        env = make_env(ENV_ID, 0, False, 5, "t")()
        env.reset(seed=1, options={"num_missing_entities": 99})
        _, _, terminated, truncated, info = env.step(self._action(eot=1))
        assert terminated and not truncated
        assert "episode" in info, "eot-terminated episode must emit RecordEpisodeStatistics info"

    def test_no_eot_keeps_running(self, registered_env):
        """Without eot (and far from solved), the episode keeps going — no
        spurious termination, no episode stats yet."""
        env = make_env(ENV_ID, 0, False, 5, "t")()
        env.reset(seed=1, options={"num_missing_entities": 99})
        _, _, terminated, truncated, info = env.step(self._action(eot=0))
        assert not terminated and not truncated
        assert "episode" not in info

    def test_reward_hparams_default(self):
        a = PpoArgs()
        assert a.throughput_reward_scale == 1.0
        assert a.step_penalty == 0.0


class TestCriticWarmupParamSplit:
    def _split(self, agent):
        critic_params = list(agent.critic_head.parameters())
        critic_ids = {id(p) for p in critic_params}
        actor_params = [p for p in agent.parameters() if id(p) not in critic_ids]
        return actor_params, critic_params

    def test_split_is_a_partition(self, agent):
        actor_params, critic_params = self._split(agent)
        assert len(actor_params) + len(critic_params) == len(list(agent.parameters()))
        assert len(critic_params) > 0 and len(actor_params) > 0

    def test_encoder_and_policy_heads_are_actor(self, agent):
        """The shared encoder and every policy/eot head count as the actor;
        only the value head is the critic. Freezing the encoder too is what
        keeps the policy genuinely fixed during warm-up."""
        actor_params, _ = self._split(agent)
        actor_ids = {id(p) for p in actor_params}
        for module in (agent.encoder, agent.eot_head, agent.tile_logits,
                       agent.ent_head, agent.dir_head):
            assert all(id(p) in actor_ids for p in module.parameters())

    def test_frozen_actor_trains_only_critic(self, agent):
        """With the actor frozen, a value-loss backward populates grads on
        the critic head alone — the warm-up's intended behaviour."""
        actor_params, critic_params = self._split(agent)
        for p in actor_params:
            p.requires_grad_(False)

        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        value = agent.get_value(obs)
        (value ** 2).mean().backward()

        assert all(p.grad is not None for p in critic_params)
        assert all(p.grad is None for p in actor_params)


class TestCriticHeadStd:
    def test_default_is_one(self):
        assert PpoArgs().critic_head_std == 1.0

    def test_construction_scales_value_head_magnitude(self, envs):
        """A smaller critic_head_std yields a smaller-magnitude value head; the
        orthogonal (1, N) init makes the row's L2 norm equal to the std."""
        small = AgentCNN(envs, layers=(16, 16, 16), critic_head_std=0.01)
        big = AgentCNN(envs, layers=(16, 16, 16), critic_head_std=1.0)
        assert small.critic_head[-1].weight.norm().item() < big.critic_head[-1].weight.norm().item()
        assert small.critic_head[-1].weight.norm().item() == pytest.approx(0.01, abs=1e-3)
        assert big.critic_head[-1].weight.norm().item() == pytest.approx(1.0, abs=1e-2)

    def test_post_load_reinit_replaces_weights_in_place_at_new_std(self, agent):
        """The post-load re-init (layer_init on the value head) rewrites it in
        place to the requested std, so a checkpoint's untrained critic doesn't
        clobber --critic-head-std."""
        head_weight = agent.critic_head[-1].weight
        before = head_weight.detach().clone()
        layer_init(agent.critic_head[-1], std=0.02)
        assert head_weight is agent.critic_head[-1].weight  # in place
        assert not torch.equal(before, head_weight)
        assert head_weight.norm().item() == pytest.approx(0.02, abs=1e-3)
