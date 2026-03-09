"""Tests for AsyncVectorEnv migration: curriculum sampling, set_max_missing,
env registration in subprocesses, and autoreset behavior."""

import os
import sys

import gymnasium as gym
import numpy as np

# Disable wandb before importing ppo
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv, _ensure_env_registered, make_env  # noqa: E402


def _make_env(size=5, max_steps=10, max_missing_entities=1):
    """Create a FactorioEnv for testing."""
    return FactorioEnv(
        size=size, max_steps=max_steps, idx=0,
        max_missing_entities=max_missing_entities,
    )


def _noop_action():
    """Return a no-op action (place empty entity at 0,0)."""
    return {
        "xy": np.array([0, 0]),
        "entity": 0,
        "direction": 0,
        "item": 0,
        "misc": 0,
    }


class TestSetMaxMissing:
    """Test the set_max_missing() method used by envs.call()."""

    def test_updates_max_missing_entities(self):
        env = _make_env(max_missing_entities=1)
        assert env.max_missing_entities == 1
        env.set_max_missing(5)
        assert env.max_missing_entities == 5

    def test_autoreset_uses_updated_max(self):
        """After set_max_missing(3), autoreset should sample from [0, 3]."""
        env = _make_env(size=5, max_steps=10, max_missing_entities=1)
        env.reset(seed=42, options={"num_missing_entities": 0})

        env.set_max_missing(3)

        # Reset without options (simulating autoreset)
        env.reset(seed=99)
        assert 0 <= env._num_missing_entities <= 3

    def test_autoreset_samples_vary(self):
        """Autoreset with max_missing_entities=3 should produce varied values
        across many resets (not always the same)."""
        env = _make_env(size=5, max_steps=10, max_missing_entities=3)
        env.reset(seed=0, options={"num_missing_entities": 0})

        values = set()
        for seed in range(50):
            env.reset(seed=seed)
            values.add(env._num_missing_entities)

        assert len(values) > 1, (
            f"Expected varied num_missing_entities across resets, got only {values}"
        )


class TestAutoresetCurriculumSampling:
    """Test that autoreset (options=None) samples num_missing_entities
    instead of reusing the stale initial value."""

    def test_autoreset_does_not_reuse_initial_options(self):
        """If the first reset passes num_missing_entities=0, subsequent
        autosets (options=None) should NOT always use 0."""
        env = _make_env(size=5, max_steps=10, max_missing_entities=3)

        # First explicit reset with num_missing_entities=0
        env.reset(seed=42, options={"num_missing_entities": 0})
        assert env._num_missing_entities == 0

        # Simulate multiple autoreset calls (no options)
        values = set()
        for seed in range(50):
            env.reset(seed=seed)
            values.add(env._num_missing_entities)

        # Should not always be 0 — the random sampling path should trigger
        assert values != {0}, (
            f"Autoreset always produced 0, suggesting _reset_options stickiness bug"
        )

    def test_explicit_options_still_honored(self):
        """Passing options with num_missing_entities should override sampling."""
        env = _make_env(size=5, max_steps=10, max_missing_entities=5)
        env.reset(seed=42, options={"num_missing_entities": 2})
        assert env._num_missing_entities == 2

    def test_autoreset_respects_max_bound(self):
        """Autoreset should never exceed max_missing_entities."""
        env = _make_env(size=5, max_steps=10, max_missing_entities=2)
        for seed in range(50):
            env.reset(seed=seed)
            assert 0 <= env._num_missing_entities <= 2


class TestEnsureEnvRegistered:
    """Test the _ensure_env_registered helper for subprocess compatibility."""

    def test_registers_when_missing(self):
        # Temporarily remove the registration if present
        was_registered = "factorion/FactorioEnv-v0" in gym.registry
        if was_registered:
            del gym.registry["factorion/FactorioEnv-v0"]

        try:
            assert "factorion/FactorioEnv-v0" not in gym.registry
            _ensure_env_registered()
            assert "factorion/FactorioEnv-v0" in gym.registry
        finally:
            # Clean up: re-register if it was there before
            if was_registered and "factorion/FactorioEnv-v0" not in gym.registry:
                gym.register(id="factorion/FactorioEnv-v0", entry_point=FactorioEnv)

    def test_idempotent(self):
        """Calling _ensure_env_registered twice should not raise."""
        _ensure_env_registered()
        _ensure_env_registered()
        assert "factorion/FactorioEnv-v0" in gym.registry


class TestMaxMissingEntitiesConstructor:
    """Test that max_missing_entities is properly passed through make_env."""

    def test_default_max_missing_entities(self):
        env = _make_env()
        assert env.max_missing_entities == 1

    def test_custom_max_missing_entities(self):
        env = _make_env(max_missing_entities=4)
        assert env.max_missing_entities == 4


class TestAsyncVectorEnvIntegration:
    """Integration tests with actual AsyncVectorEnv."""

    def test_async_env_creates_and_steps(self):
        """AsyncVectorEnv should create envs, reset, and step without error."""
        _ensure_env_registered()
        num_envs = 4
        envs = gym.vector.AsyncVectorEnv(
            [make_env("factorion/FactorioEnv-v0", i, False, 5, "test")
             for i in range(num_envs)],
            autoreset_mode="SameStep",
        )
        try:
            obs, info = envs.reset(
                seed=42, options={"num_missing_entities": 1}
            )
            assert obs.shape[0] == num_envs

            actions = {
                "xy": np.zeros((num_envs, 2), dtype=int),
                "entity": np.zeros(num_envs, dtype=int),
                "direction": np.zeros(num_envs, dtype=int),
                "item": np.zeros(num_envs, dtype=int),
                "misc": np.zeros(num_envs, dtype=int),
            }
            obs, rewards, terminated, truncated, infos = envs.step(actions)
            assert obs.shape[0] == num_envs
            assert rewards.shape == (num_envs,)
        finally:
            envs.close()

    def test_async_env_call_set_max_missing(self):
        """envs.call('set_max_missing', ...) should update all subprocess envs."""
        _ensure_env_registered()
        num_envs = 4
        envs = gym.vector.AsyncVectorEnv(
            [make_env("factorion/FactorioEnv-v0", i, False, 5, "test")
             for i in range(num_envs)],
            autoreset_mode="SameStep",
        )
        try:
            envs.reset(seed=42, options={"num_missing_entities": 0})

            # Update curriculum level
            envs.call("set_max_missing", 3)

            # Verify all envs received the update
            results = envs.get_attr("max_missing_entities")
            assert all(v == 3 for v in results), (
                f"Expected all envs to have max_missing_entities=3, got {results}"
            )
        finally:
            envs.close()
