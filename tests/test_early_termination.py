"""Tests for early termination when the agent solves the puzzle."""

import os
import sys

import numpy as np

# Disable wandb before importing ppo
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv  # noqa: E402


def _make_env(size=5, max_steps=10):
    """Create a FactorioEnv for testing."""
    return FactorioEnv(size=size, max_steps=max_steps, idx=0)


def _noop_action():
    """Return a no-op action (place empty entity at 0,0)."""
    return {
        "xy": np.array([0, 0]),
        "entity": 0,      # empty
        "direction": 0,    # NONE
        "item": 0,         # empty
        "misc": 0,         # NONE
    }


class TestEarlyTermination:
    """Test that episodes terminate early when throughput >= 1.0."""

    def test_termination_on_full_throughput(self):
        """With num_missing_entities=0 the factory is already solved.

        The first step should produce throughput >= 1.0 and terminated=True.
        """
        env = _make_env(size=5, max_steps=10)
        env.reset(seed=42, options={"num_missing_entities": 0})

        _, _, terminated, truncated, info = env.step(_noop_action())

        assert terminated is True, f"Expected terminated=True, got {terminated}"
        assert truncated is False, f"Expected truncated=False, got {truncated}"
        assert info["throughput"] >= 1.0

    def test_truncation_without_solving(self):
        """With many missing entities the factory can't be solved by no-ops.

        Stepping max_steps+1 times should produce truncated=True on the last step.
        """
        max_steps = 5
        env = _make_env(size=5, max_steps=max_steps)
        env.reset(seed=42, options={"num_missing_entities": 99})

        # Step until truncation
        terminated = False
        truncated = False
        step_count = 0
        while not terminated and not truncated:
            _, _, terminated, truncated, info = env.step(_noop_action())
            step_count += 1

        assert truncated is True, f"Expected truncated=True, got {truncated}"
        assert terminated is False, f"Expected terminated=False, got {terminated}"
        assert info["throughput"] < 1.0

    def test_terminated_and_truncated_are_mutually_exclusive(self):
        """terminated and truncated should never both be True."""
        env = _make_env(size=5, max_steps=10)

        # Case 1: solved factory (terminated=True)
        env.reset(seed=42, options={"num_missing_entities": 0})
        _, _, terminated, truncated, _ = env.step(_noop_action())
        assert not (terminated and truncated), "terminated and truncated are both True"

        # Case 2: unsolved factory (truncated=True)
        env.reset(seed=42, options={"num_missing_entities": 99})
        for _ in range(20):  # more than max_steps
            _, _, terminated, truncated, _ = env.step(_noop_action())
            assert not (terminated and truncated), "terminated and truncated are both True"
            if terminated or truncated:
                break


class TestCompletionBonus:
    """Test that the completion bonus is proportional to remaining steps."""

    def test_bonus_on_immediate_solve(self):
        """Solving on step 0 gives the maximum completion bonus."""
        max_steps = 10
        env = _make_env(size=5, max_steps=max_steps)
        env.reset(seed=42, options={"num_missing_entities": 0})

        _, reward, terminated, _, info = env.step(_noop_action())

        assert terminated is True
        # Reward should be pre_reward + (max_steps - steps)
        # At step 0: bonus = max_steps - 0 = 10
        # pre_reward is in [0, 1] (normalized weighted average), so reward > max_steps - 1
        assert reward > max_steps - 1, f"Expected reward > {max_steps - 1}, got {reward}"
        assert info["completion_bonus"] == max_steps  # max_steps - 0

    def test_no_bonus_on_truncation(self):
        """Truncating gives no completion bonus in the reward."""
        max_steps = 5
        env = _make_env(size=5, max_steps=max_steps)
        env.reset(seed=42, options={"num_missing_entities": 99})

        # Step until truncation
        for _ in range(max_steps + 2):
            _, reward, terminated, truncated, info = env.step(_noop_action())
            if truncated:
                break

        assert truncated is True
        assert terminated is False
        # On truncation, reward is just pre_reward (no bonus added)
        # pre_reward is a normalized value in [0, 1]
        assert reward <= 1.0, f"Expected reward <= 1.0 (no bonus), got {reward}"


class TestStepsTaken:
    """Test that steps_taken is correct in the info dict."""

    def test_steps_taken_on_termination(self):
        """steps_taken should be 0 when solved on the first step."""
        env = _make_env(size=5, max_steps=10)
        env.reset(seed=42, options={"num_missing_entities": 0})

        _, _, terminated, _, info = env.step(_noop_action())

        assert terminated is True
        assert "steps_taken" in info, "steps_taken missing from info on termination"
        assert info["steps_taken"] == 0, f"Expected steps_taken=0, got {info['steps_taken']}"

    def test_steps_taken_on_truncation(self):
        """steps_taken should equal the step count at truncation."""
        max_steps = 3
        env = _make_env(size=5, max_steps=max_steps)
        env.reset(seed=42, options={"num_missing_entities": 99})

        for _ in range(max_steps + 2):
            _, _, terminated, truncated, info = env.step(_noop_action())
            if truncated:
                break

        assert truncated is True
        assert "steps_taken" in info, "steps_taken missing from info on truncation"
        # truncated fires when self.steps > max_steps, so steps_taken == max_steps + 1
        assert info["steps_taken"] == max_steps + 1, (
            f"Expected steps_taken={max_steps + 1}, got {info['steps_taken']}"
        )

    def test_steps_taken_absent_mid_episode(self):
        """steps_taken should NOT be in info during the middle of an episode."""
        env = _make_env(size=5, max_steps=20)
        env.reset(seed=42, options={"num_missing_entities": 99})

        _, _, terminated, truncated, info = env.step(_noop_action())

        # Episode isn't over yet
        assert not terminated and not truncated
        assert "steps_taken" not in info, "steps_taken should not be in info mid-episode"
