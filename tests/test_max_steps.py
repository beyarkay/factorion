"""Tests that max_steps scales dynamically with num_missing_entities."""

import os
import sys

import numpy as np

# Disable wandb before importing ppo
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv  # noqa: E402


def make_env(num_missing_entities, size=5):
    """Create a FactorioEnv and reset it with the given num_missing_entities."""
    env = FactorioEnv(size=size)
    env.reset(seed=42, options={"num_missing_entities": num_missing_entities})
    return env


class TestMaxStepsFormula:
    """max_steps = num_missing_entities * 2 + 1 (or 1 when 0)."""

    def test_missing_0(self):
        env = make_env(0)
        assert env.max_steps == 1

    def test_missing_1(self):
        env = make_env(1)
        assert env.max_steps == 3

    def test_missing_2(self):
        env = make_env(2)
        assert env.max_steps == 5

    def test_missing_3(self):
        env = make_env(3)
        assert env.max_steps == 7

    def test_missing_8(self):
        env = make_env(8)
        assert env.max_steps == 17


class TestMaxStepsChangesPerReset:
    """max_steps updates when reset() is called with different options."""

    def test_changes_between_resets(self):
        env = FactorioEnv(size=5)
        env.reset(seed=42, options={"num_missing_entities": 1})
        assert env.max_steps == 3

        env.reset(seed=42, options={"num_missing_entities": 5})
        assert env.max_steps == 11

        env.reset(seed=42, options={"num_missing_entities": 0})
        assert env.max_steps == 1


class TestZeroMissingEntities:
    """Episodes with num_missing_entities=0 should not crash."""

    def test_zero_missing_does_not_crash(self):
        env = make_env(0)
        action = {
            "xy": np.array([0, 0]),
            "entity": 0,
            "direction": 0,
            "item": 0,
            "misc": 0,
        }
        # The env should not crash when stepping with 0 missing entities.
        # Keep stepping until the episode ends.
        done = False
        for _ in range(env.max_steps + 2):
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                break
        assert done, "Episode with 0 missing entities never terminated"


class TestEnoughStepsToSolve:
    """The agent should have at least num_missing_entities steps available."""

    def test_steps_geq_missing(self):
        for n in range(0, 10):
            env = make_env(n)
            assert env.max_steps >= max(1, n), (
                f"num_missing_entities={n} but max_steps={env.max_steps}"
            )
