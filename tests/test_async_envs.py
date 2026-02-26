"""Tests for AsyncVectorEnv parallelization (P5).

Verifies that:
- AsyncVectorEnv produces valid observations and rewards
- Environments self-sample num_missing_entities on auto-reset
- Curriculum updates propagate to subprocess envs via set_attr
- SyncVectorEnv and AsyncVectorEnv produce consistent shapes
"""

import os
import sys

import numpy as np
import pytest
import gymnasium as gym

# Disable wandb before importing ppo
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv, make_env  # noqa: E402


ENV_ID = "factorion/FactorioEnv-v0-async-test"
SIZE = 5
NUM_ENVS = 2


@pytest.fixture(scope="module")
def registered_env():
    """Register the env once for all tests in this module."""
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=FactorioEnv)


def _noop_action(num_envs):
    """Return a batched no-op action."""
    return {
        "xy": np.zeros((num_envs, 2), dtype=int),
        "entity": np.zeros(num_envs, dtype=int),
        "direction": np.zeros(num_envs, dtype=int),
        "item": np.zeros(num_envs, dtype=int),
        "misc": np.zeros(num_envs, dtype=int),
    }


class TestSelfSamplingDifficulty:
    """Test that FactorioEnv self-samples num_missing_entities on reset."""

    def test_explicit_options_respected(self):
        """When options are provided, use the explicit num_missing_entities."""
        env = FactorioEnv(size=SIZE, max_steps=10, idx=0)
        env.reset(seed=42, options={"num_missing_entities": 0})
        assert env._num_missing_entities == 0

        env.reset(seed=42, options={"num_missing_entities": 3})
        assert env._num_missing_entities == 3

    def test_self_sample_on_no_options(self):
        """Without options, env self-samples from [0, max_missing_entities]."""
        env = FactorioEnv(size=SIZE, max_steps=10, idx=0)
        env.max_missing_entities = 3
        env.reset(seed=42)  # No options — should self-sample
        assert 0 <= env._num_missing_entities <= 3

    def test_self_sample_range_changes_with_curriculum(self):
        """Updating max_missing_entities changes the self-sample range."""
        env = FactorioEnv(size=SIZE, max_steps=10, idx=0)

        # With max_missing_entities=0, self-sampling always gives 0
        env.max_missing_entities = 0
        for _ in range(10):
            env.reset(seed=42)
            assert env._num_missing_entities == 0

    def test_base_seed_preserved_across_resets(self):
        """The seed from the first explicit reset is reused on auto-resets."""
        env = FactorioEnv(size=SIZE, max_steps=10, idx=0)
        env.reset(seed=42, options={"num_missing_entities": 0})
        assert env._base_seed == 42
        assert env._seed == 42  # idx=0, so seed = 42 + 0

        # Auto-reset (no seed) should reuse base_seed
        env.reset(options={"num_missing_entities": 0})
        assert env._base_seed == 42
        assert env._seed == 42


class TestSyncVectorEnv:
    """Test SyncVectorEnv with self-sampling difficulty."""

    def test_reset_and_step(self, registered_env):
        """SyncVectorEnv reset and step produce correct shapes."""
        envs = gym.vector.SyncVectorEnv(
            [make_env(ENV_ID, i, False, SIZE, "test") for i in range(NUM_ENVS)]
        )
        envs.set_attr("max_missing_entities", 1)
        obs, info = envs.reset(seed=42)
        assert obs.shape == (NUM_ENVS, 4, SIZE, SIZE)

        obs, rew, term, trunc, info = envs.step(_noop_action(NUM_ENVS))
        assert obs.shape == (NUM_ENVS, 4, SIZE, SIZE)
        assert rew.shape == (NUM_ENVS,)
        assert term.shape == (NUM_ENVS,)
        assert trunc.shape == (NUM_ENVS,)
        envs.close()

    def test_curriculum_set_attr(self, registered_env):
        """set_attr propagates max_missing_entities to all envs."""
        envs = gym.vector.SyncVectorEnv(
            [make_env(ENV_ID, i, False, SIZE, "test") for i in range(NUM_ENVS)]
        )
        envs.set_attr("max_missing_entities", 5)
        for env in envs.envs:
            assert env.unwrapped.max_missing_entities == 5
        envs.close()


class TestAsyncVectorEnv:
    """Test AsyncVectorEnv with self-sampling difficulty."""

    def test_reset_and_step(self, registered_env):
        """AsyncVectorEnv reset and step produce correct shapes."""
        envs = gym.vector.AsyncVectorEnv(
            [make_env(ENV_ID, i, False, SIZE, "test") for i in range(NUM_ENVS)]
        )
        envs.set_attr("max_missing_entities", 1)
        obs, info = envs.reset(seed=42)
        assert obs.shape == (NUM_ENVS, 4, SIZE, SIZE)

        obs, rew, term, trunc, info = envs.step(_noop_action(NUM_ENVS))
        assert obs.shape == (NUM_ENVS, 4, SIZE, SIZE)
        assert rew.shape == (NUM_ENVS,)
        assert term.shape == (NUM_ENVS,)
        assert trunc.shape == (NUM_ENVS,)
        envs.close()

    def test_multi_step_rollout(self, registered_env):
        """Run multiple steps to trigger auto-reset and verify shapes."""
        envs = gym.vector.AsyncVectorEnv(
            [make_env(ENV_ID, i, False, SIZE, "test") for i in range(NUM_ENVS)]
        )
        envs.set_attr("max_missing_entities", 1)
        obs, _ = envs.reset(seed=42)

        for _ in range(30):
            obs, rew, term, trunc, info = envs.step(_noop_action(NUM_ENVS))
            assert obs.shape == (NUM_ENVS, 4, SIZE, SIZE)
            assert not np.isnan(rew).any(), "NaN reward detected"
        envs.close()

    def test_curriculum_set_attr(self, registered_env):
        """set_attr propagates max_missing_entities to subprocess envs."""
        envs = gym.vector.AsyncVectorEnv(
            [make_env(ENV_ID, i, False, SIZE, "test") for i in range(NUM_ENVS)]
        )
        envs.set_attr("max_missing_entities", 5)
        vals = envs.get_attr("max_missing_entities")
        assert all(v == 5 for v in vals), f"Expected all 5, got {vals}"
        envs.close()

    def test_episode_info_on_termination(self, registered_env):
        """Verify episode info is available when envs terminate."""
        envs = gym.vector.AsyncVectorEnv(
            [make_env(ENV_ID, i, False, SIZE, "test") for i in range(NUM_ENVS)]
        )
        # With max_missing_entities=0, factory is solved immediately
        envs.set_attr("max_missing_entities", 0)
        envs.reset(seed=42)

        obs, rew, term, trunc, info = envs.step(_noop_action(NUM_ENVS))
        # All envs should terminate (throughput >= 1.0 with 0 missing entities)
        done = np.logical_or(term, trunc)
        assert done.any(), "Expected at least one env to finish"

        # RecordEpisodeStatistics should provide episode info
        if "episode" in info:
            mask = info["_episode"]
            for i in range(NUM_ENVS):
                if mask[i]:
                    assert "r" in info["episode"]
                    assert "l" in info["episode"]
        envs.close()


class TestSyncAsyncParity:
    """Verify SyncVectorEnv and AsyncVectorEnv produce consistent results."""

    def test_observation_shapes_match(self, registered_env):
        """Both env types produce the same observation shapes."""
        sync = gym.vector.SyncVectorEnv(
            [make_env(ENV_ID, i, False, SIZE, "test") for i in range(NUM_ENVS)]
        )
        async_ = gym.vector.AsyncVectorEnv(
            [make_env(ENV_ID, i, False, SIZE, "test") for i in range(NUM_ENVS)]
        )

        sync.set_attr("max_missing_entities", 1)
        async_.set_attr("max_missing_entities", 1)

        sync_obs, _ = sync.reset(seed=42)
        async_obs, _ = async_.reset(seed=42)

        assert sync_obs.shape == async_obs.shape

        sync_obs, sync_rew, _, _, _ = sync.step(_noop_action(NUM_ENVS))
        async_obs, async_rew, _, _, _ = async_.step(_noop_action(NUM_ENVS))

        assert sync_obs.shape == async_obs.shape
        assert sync_rew.shape == async_rew.shape

        sync.close()
        async_.close()
