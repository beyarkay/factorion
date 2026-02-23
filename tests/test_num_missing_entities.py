"""Tests for num_missing_entities behavior.

Verifies that:
- num_missing_entities=0 produces an already-complete factory (throughput=1.0)
- num_missing_entities=1 produces a factory with a gap (throughput<1.0)
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest

# Disable wandb before importing ppo
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ppo  # noqa: E402


@pytest.fixture(autouse=True)
def register_env():
    """Register the FactorioEnv if not already registered."""
    env_id = "factorion/FactorioEnv-v0"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=ppo.FactorioEnv)
    yield


def _make_env(size=5):
    """Create a bare FactorioEnv (no wrappers)."""
    return gym.make(
        "factorion/FactorioEnv-v0",
        size=size,
        max_steps=2 * size,
    )


def _noop_action(env):
    """Return a no-op action that places 'empty' at (0,0)."""
    return {
        "xy": np.array([0, 0]),
        "entity": 0,  # empty
        "direction": 0,
        "item": 0,
        "misc": 0,
    }


class TestNumMissingEntities:
    def test_zero_missing_is_already_solved(self):
        """With num_missing_entities=0, the factory is complete and throughput=1.0."""
        env = _make_env()
        env.reset(seed=42, options={"num_missing_entities": 0})

        # Take a no-op step so throughput is computed
        _obs, _reward, _term, _trunc, info = env.step(_noop_action(env))
        assert info["throughput"] == pytest.approx(1.0, abs=1e-6), (
            f"Expected throughput=1.0 for num_missing_entities=0, got {info['throughput']}"
        )
        env.close()

    def test_one_missing_has_gap(self):
        """With num_missing_entities=1, the factory has a gap and throughput<1.0."""
        env = _make_env()
        env.reset(seed=42, options={"num_missing_entities": 1})

        # Take a no-op step so throughput is computed
        _obs, _reward, _term, _trunc, info = env.step(_noop_action(env))
        assert info["throughput"] < 1.0, (
            f"Expected throughput<1.0 for num_missing_entities=1, got {info['throughput']}"
        )
        env.close()
