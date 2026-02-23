"""Tests that FactorioEnv works with various grid sizes.

Verifies that creating environments with size=8 (new default) and size=12
and running random steps doesn't crash.
"""

import os
import sys

import numpy as np

# Disable wandb before importing ppo
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv  # noqa: E402


def make_valid_action(env, rng):
    """Build a random action that satisfies step()'s assertions."""
    n_ent = len(env.entities) - 2  # source/sink excluded
    n_dir = len(env.Direction)
    n_items = len(env.items)
    n_misc = len(env.Misc)
    return {
        "xy": rng.integers(0, env.size, size=2),
        "entity": rng.integers(0, n_ent),
        "direction": rng.integers(0, n_dir),
        "item": rng.integers(0, n_items),
        "misc": rng.integers(0, n_misc),
    }


class TestGridSizes:
    def test_size_8_random_steps(self):
        """Create a FactorioEnv with size=8 and run random steps."""
        env = FactorioEnv(size=8, max_steps=16)
        obs, info = env.reset(seed=42, options={'num_missing_entities': 2})
        assert obs.shape == (len(env.Channel), 8, 8)
        rng = np.random.default_rng(42)
        for _ in range(10):
            action = make_valid_action(env, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (len(env.Channel), 8, 8)
            if terminated or truncated:
                obs, info = env.reset()
        env.close()

    def test_size_12_random_steps(self):
        """Create a FactorioEnv with size=12 and run random steps."""
        env = FactorioEnv(size=12, max_steps=24)
        obs, info = env.reset(seed=42, options={'num_missing_entities': 2})
        assert obs.shape == (len(env.Channel), 12, 12)
        rng = np.random.default_rng(42)
        for _ in range(10):
            action = make_valid_action(env, rng)
            obs, reward, terminated, truncated, info = env.step(action)
            assert obs.shape == (len(env.Channel), 12, 12)
            if terminated or truncated:
                obs, info = env.reset()
        env.close()

    def test_default_size_is_8(self):
        """Verify the default size in Args is 8."""
        from ppo import Args
        args = Args()
        assert args.size == 8
