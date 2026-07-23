"""Tests for early termination when the agent solves the puzzle."""

import os
import sys

import numpy as np
import pytest

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
    """Episodes terminate only when the agent declares eot, and truncate at
    max_steps. A full-throughput solve does NOT auto-terminate — otherwise the
    eot head would never learn to fire on a finished factory."""

    def test_solve_does_not_auto_terminate(self):
        """With num_missing_entities=0 the factory is already solved, but
        without an eot the episode keeps running — the agent must declare done.
        """
        env = _make_env(size=5, max_steps=10)
        env.reset(seed=42, options={"num_missing_entities": 0})

        _, _, terminated, truncated, info = env.step(_noop_action())

        # A fully-solved factory reaches its per-factory max → normed == 1.0
        # (regardless of absolute belt speed), but with no eot declared the
        # episode must NOT auto-terminate.
        assert info["thput_normed"] >= 1.0
        assert terminated is False, "a solve must NOT auto-terminate"
        assert truncated is False

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
        assert info["thput_normed"] < 1.0

    def test_terminated_and_truncated_are_mutually_exclusive(self):
        """terminated and truncated should never both be True."""
        env = _make_env(size=5, max_steps=10)

        # Case 1: agent declares eot (terminated=True)
        env.reset(seed=42, options={"num_missing_entities": 0})
        action = _noop_action()
        action["eot"] = 1
        _, _, terminated, truncated, _ = env.step(action)
        assert terminated and not truncated

        # Case 2: unsolved factory (truncated=True)
        env.reset(seed=42, options={"num_missing_entities": 99})
        for _ in range(20):  # more than max_steps
            _, _, terminated, truncated, _ = env.step(_noop_action())
            assert not (terminated and truncated), "terminated and truncated are both True"
            if terminated or truncated:
                break


class TestReward:
    """Reward = throughput_reward_scale * throughput - step_penalty * num_steps
    - entity_penalty_scale * num_entities: a per-step penalty plus the terminal
    throughput reward (less the per-entity frugality penalty) when the episode
    ends (solve / eot / max_steps)."""

    def test_solved_factory_with_eot_pays_full_reward(self):
        """Declaring eot on a solved factory pays the full terminal throughput
        reward (throughput >= 1.0) minus one step's penalty."""
        env = _make_env(size=5, max_steps=10)
        env.reset(seed=42, options={"num_missing_entities": 0})

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        assert info["thput_normed"] >= 1.0
        expected = (env.throughput_reward_scale * info["thput_normed"]
                    - env.step_penalty
                    - env.entity_penalty_scale * info["num_entities"])
        assert reward == pytest.approx(expected)

    def test_step_penalty_only_mid_episode(self):
        """A non-terminal step pays just -step_penalty (no throughput reward)."""
        env = _make_env(size=5, max_steps=20)
        env.reset(seed=42, options={"num_missing_entities": 99})

        _, reward, terminated, truncated, _ = env.step(_noop_action())

        assert not terminated and not truncated
        assert reward == pytest.approx(-env.step_penalty)

    def test_terminal_throughput_reward_on_truncation(self):
        """At max_steps the episode still banks the terminal throughput reward."""
        max_steps = 5
        env = _make_env(size=5, max_steps=max_steps)
        env.reset(seed=42, options={"num_missing_entities": 99})

        for _ in range(max_steps + 2):
            _, reward, terminated, truncated, info = env.step(_noop_action())
            if truncated:
                break

        assert truncated is True
        assert terminated is False
        expected = (env.throughput_reward_scale * info["thput_normed"]
                    - env.step_penalty
                    - env.entity_penalty_scale * info["num_entities"])
        assert reward == pytest.approx(expected)

    def test_eot_action_terminates_episode(self):
        """A non-solved factory ends immediately when the agent declares eot=1,
        and pays the terminal throughput reward."""
        env = _make_env(size=5, max_steps=50)
        env.reset(seed=42, options={"num_missing_entities": 99})

        action = _noop_action()
        action["entity"] = env._source_id  # ignored because EOT is not a placement
        action["eot"] = 1
        _, reward, terminated, truncated, info = env.step(action)

        assert terminated is True, "eot=1 should terminate the episode"
        assert truncated is False
        assert info["frac_invalid_actions"] == 0
        assert info["thput_normed"] < 1.0  # ended early, not a full solve
        expected = (env.throughput_reward_scale * info["thput_normed"]
                    - env.step_penalty
                    - env.entity_penalty_scale * info["num_entities"])
        assert reward == pytest.approx(expected)

    def test_entity_penalty_scales_with_entity_count(self):
        """The terminal reward drops by entity_penalty_scale per non-empty
        entity, so a wasteful factory is penalised more than a frugal one."""
        env = _make_env(size=5, max_steps=10)
        env.entity_penalty_scale = 0.01
        env.reset(seed=42, options={"num_missing_entities": 0})

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        expected = (env.throughput_reward_scale * info["thput_normed"]
                    - env.step_penalty
                    - env.entity_penalty_scale * info["num_entities"])
        assert reward == pytest.approx(expected)
        # A non-zero penalty must actually pull the reward below the penalty-free value.
        assert info["num_entities"] > 0
        assert reward < env.throughput_reward_scale * info["thput_normed"] - env.step_penalty


class TestStepsTaken:
    """Test that steps_taken is correct in the info dict."""

    def test_steps_taken_on_termination(self):
        """steps_taken should be 0 when the agent declares eot on the first step."""
        env = _make_env(size=5, max_steps=10)
        env.reset(seed=42, options={"num_missing_entities": 0})

        action = _noop_action()
        action["eot"] = 1
        _, _, terminated, _, info = env.step(action)

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
