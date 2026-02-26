"""Tests for pre-allocated numpy buffer sync with _world_CWH.

Verifies that _world_WHC_np stays perfectly in sync with _world_CWH after
reset() and after each step(), including both valid and invalid actions.
"""

import os
import sys

import numpy as np
import torch

# Disable wandb before importing ppo
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv  # noqa: E402


def _make_env(size=5, max_steps=10):
    """Create a FactorioEnv for testing."""
    return FactorioEnv(size=size, max_steps=max_steps, idx=0)


def _assert_buffer_in_sync(env, label=""):
    """Assert that _world_WHC_np matches _world_CWH after permutation."""
    expected = env._world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
    np.testing.assert_array_equal(
        env._world_WHC_np,
        expected,
        err_msg=f"Buffer out of sync {label}",
    )


def _noop_action():
    """Return a no-op action (place empty entity at 0,0)."""
    return {
        "xy": np.array([0, 0]),
        "entity": 0,      # empty
        "direction": 0,    # NONE
        "item": 0,         # empty
        "misc": 0,         # NONE
    }


class TestBufferSyncAfterReset:
    """Buffer should be in sync immediately after reset()."""

    def test_sync_after_reset_zero_missing(self):
        env = _make_env()
        env.reset(seed=42, options={"num_missing_entities": 0})
        _assert_buffer_in_sync(env, "after reset(num_missing=0)")

    def test_sync_after_reset_one_missing(self):
        env = _make_env()
        env.reset(seed=42, options={"num_missing_entities": 1})
        _assert_buffer_in_sync(env, "after reset(num_missing=1)")

    def test_sync_after_multiple_resets(self):
        env = _make_env()
        for seed in range(5):
            env.reset(seed=seed, options={"num_missing_entities": seed % 3})
            _assert_buffer_in_sync(env, f"after reset(seed={seed})")


class TestBufferSyncAfterStep:
    """Buffer should be in sync after each step(), for both valid and invalid actions."""

    def test_sync_after_noop_step(self):
        env = _make_env()
        env.reset(seed=42, options={"num_missing_entities": 0})
        env.step(_noop_action())
        _assert_buffer_in_sync(env, "after noop step")

    def test_sync_after_invalid_action(self):
        """Invalid actions (e.g. placing on source/sink) should not desync the buffer."""
        env = _make_env(size=8)
        env.reset(seed=42, options={"num_missing_entities": 1})

        # Try placing entity at (0,0) with a direction but empty entity — invalid
        action = {
            "xy": np.array([0, 0]),
            "entity": 0,       # empty
            "direction": 1,    # non-NONE direction with empty entity → invalid
            "item": 0,
            "misc": 0,
        }
        env.step(action)
        _assert_buffer_in_sync(env, "after invalid action")

    def test_sync_after_valid_placement(self):
        """Valid entity placement should update both _world_CWH and _world_WHC_np."""
        env = _make_env(size=8)
        env.reset(seed=42, options={"num_missing_entities": 2})

        # Place a transport belt at a tile that isn't source/sink
        # Find a tile that is empty (not source or sink)
        for x in range(env.size):
            for y in range(env.size):
                ent = env._world_CWH[env.Channel.ENTITIES.value, x, y].item()
                if ent not in (len(env.entities) - 1, len(env.entities) - 2):
                    # Place transport belt facing east
                    action = {
                        "xy": np.array([x, y]),
                        "entity": env.str2ent('transport_belt').value,
                        "direction": env.Direction.EAST.value,
                        "item": 0,
                        "misc": 0,
                    }
                    env.step(action)
                    _assert_buffer_in_sync(env, f"after valid placement at ({x},{y})")
                    return
        raise AssertionError("Could not find an empty tile for valid placement")

    def test_sync_through_full_episode(self):
        """Buffer stays in sync through an entire episode of random-ish actions."""
        env = _make_env(size=8, max_steps=20)
        env.reset(seed=123, options={"num_missing_entities": 1})
        _assert_buffer_in_sync(env, "after reset")

        rng = np.random.default_rng(seed=456)
        for step in range(20):
            action = {
                "xy": np.array([rng.integers(0, env.size), rng.integers(0, env.size)]),
                "entity": int(rng.integers(0, len(env.entities) - 2)),
                "direction": int(rng.integers(0, len(env.Direction))),
                "item": 0,
                "misc": 0,
            }
            _, _, terminated, truncated, _ = env.step(action)
            _assert_buffer_in_sync(env, f"after step {step}")
            if terminated or truncated:
                break


class TestBufferSyncAfterResetFollowingSteps:
    """After stepping then resetting, buffer should reflect the new world."""

    def test_reset_after_steps(self):
        env = _make_env(size=8, max_steps=5)

        # First episode
        env.reset(seed=10, options={"num_missing_entities": 1})
        env.step(_noop_action())
        env.step(_noop_action())

        # Second episode — buffer should match fresh world
        env.reset(seed=20, options={"num_missing_entities": 0})
        _assert_buffer_in_sync(env, "after reset following steps")


class TestBufferShape:
    """Buffer should have the correct shape and dtype."""

    def test_shape_and_dtype(self):
        env = _make_env(size=8)
        assert env._world_WHC_np.shape == (8, 8, 4)
        assert env._world_WHC_np.dtype == np.int64

    def test_shape_different_sizes(self):
        for size in [5, 8, 12]:
            env = _make_env(size=size)
            assert env._world_WHC_np.shape == (size, size, 4)
