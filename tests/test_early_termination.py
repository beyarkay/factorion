"""Tests for early termination when the agent solves the puzzle."""

import os
import sys

import numpy as np
import pytest

# Disable wandb before importing ppo
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorion import Channel, Direction, LessonKind, str2ent  # noqa: E402
from ppo import FactorioEnv  # noqa: E402


def _make_env(size=5, max_steps=10, **kwargs):
    """Create a FactorioEnv for testing."""
    return FactorioEnv(size=size, max_steps=max_steps, idx=0, **kwargs)


def _expected_total(env, info):
    """The episode's improvement over reset in reference-rate units — what the
    dense per-step rewards must sum (or a lone effective step must pay)."""
    score = info["thput_raw"] * info["cost_efficiency"]
    return (
        (score - env._reward_baseline) / env._max_throughput
        if env._max_throughput > 0
        else 0.0
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


class TestEarlyTermination:
    """Episodes terminate only when the agent declares eot, and truncate at
    max_steps. A full-throughput solve does NOT auto-terminate — otherwise the
    eot head would never learn to fire on a finished factory."""

    def test_solve_does_not_auto_terminate(self):
        """With num_missing_entities=0 the factory is already solved, but
        without an eot the episode keeps running — the agent must declare done.
        """
        env = _make_env(size=5, max_steps=10)
        env.reset(seed=42, options={"num_missing_entities": 0, "kind": LessonKind.MOVE_ONE_ITEM})

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
    """Terminal reward = (thput_raw * cost_efficiency - reset baseline) /
    max_throughput: the agent is paid only for improving on the world it was
    handed, in units of the factory's reference rate."""

    def test_eot_without_improvement_pays_zero(self):
        """A factory already solved at reset earns nothing: the baseline
        absorbs its whole score, so eot-without-acting pays exactly zero."""
        env = _make_env(size=5, max_steps=10)
        env.reset(seed=42, options={"num_missing_entities": 0, "kind": LessonKind.MOVE_ONE_ITEM})

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        assert info["thput_normed"] >= 1.0
        assert reward == 0.0

    def test_mid_episode_reward_is_zero(self):
        env = _make_env(size=5, max_steps=20)
        env.reset(seed=42, options={"num_missing_entities": 99})

        _, reward, terminated, truncated, _ = env.step(_noop_action())

        assert not terminated and not truncated
        assert reward == 0

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
        assert reward == pytest.approx(_expected_total(env, info))

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
        assert reward == pytest.approx(_expected_total(env, info))

    def test_junk_placement_on_unimproved_factory_pays_negative(self):
        """Adding an entity that raises cost without raising throughput drops
        the score below the reset baseline, so the reward goes negative."""
        env = _make_env(size=7, max_steps=10)
        env.entity_cost_scale = 0.01
        env.reset(
            seed=42,
            options={"num_missing_entities": 0, "kind": LessonKind.MOVE_ONE_ITEM},
        )

        # A belt whose four neighbours are all empty cannot touch the solved
        # route, so it changes cost but not throughput.
        entity_grid = env._world_CWH[Channel.ENTITIES.value]
        ent = entity_grid.numpy()
        x, y = next(
            (int(x), int(y))
            for x, y in np.argwhere(ent == 0)
            if 0 < x < env.size - 1
            and 0 < y < env.size - 1
            and ent[x - 1, y] == 0
            and ent[x + 1, y] == 0
            and ent[x, y - 1] == 0
            and ent[x, y + 1] == 0
        )
        entity_grid[x, y] = str2ent("transport_belt").value
        env._world_CWH[Channel.DIRECTION.value, x, y] = Direction.NORTH.value

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        assert info["thput_normed"] >= 1.0
        assert reward == pytest.approx(_expected_total(env, info))
        assert reward < 0

    def test_zero_throughput_cost_penalty_is_never_negative(self):
        """Entities that deliver nothing must score exactly zero, not negative:
        else an empty grid beats a nearly-complete factory and the policy is
        rewarded for building nothing. Holds because cost multiplies the
        achieved score — zero throughput on a zero-baseline factory stays
        exactly zero however large the cost."""
        env = FactorioEnv(
            size=5,
            max_steps=10,
            idx=0,
            entity_cost_scale=1_000_000.0,
        )
        env.reset(seed=42, options={"num_missing_entities": 99})

        # Add one belt far from every source/sink so cost is non-zero while
        # throughput remains zero.
        entity_grid = env._world_CWH[Channel.ENTITIES.value]
        markers = np.argwhere(
            (entity_grid.numpy() == env._source_id)
            | (entity_grid.numpy() == env._sink_id)
        )
        x, y = next(
            (int(x), int(y))
            for x, y in np.argwhere(entity_grid.numpy() == 0)
            if all(abs(x - mx) + abs(y - my) > 1 for mx, my in markers)
        )
        entity_grid[x, y] = str2ent("transport_belt").value
        env._world_CWH[Channel.DIRECTION.value, x, y] = Direction.NORTH.value

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        assert info["thput_raw"] == 0
        assert info["entity_cost"] == pytest.approx(2.0)
        assert reward == 0


class TestRewardScaleNormalization:
    """The point of dividing by the factory's reference rate: lessons whose
    achievable items/s differ by orders of magnitude must pay near-identical
    reward for a full solve."""

    def _solved_reward(self, kind, size=11):
        """Terminal reward for building `kind`'s reference solve from blank."""
        env = _make_env(size=size, max_steps=10)
        for seed in range(20):
            try:
                env.reset(seed=seed, options={"kind": kind})
                break
            except RuntimeError:
                continue
        env._world_CWH.copy_(env._solved_world_CWH)
        action = _noop_action()
        action["eot"] = 1
        _, reward, _, _, info = env.step(action)
        return reward, info["thput_raw"]

    def test_belt_and_assembler_solves_pay_the_same(self):
        belt_r, belt_thput = self._solved_reward(LessonKind.MOVE_ONE_ITEM)
        asm_r, asm_thput = self._solved_reward(
            LessonKind.MEMORISE_4_INGREDIENT_RECIPES
        )

        assert belt_thput / asm_thput > 50, (
            "expected a large raw items/s gap between lessons"
        )
        assert 0.8 < belt_r <= 1.0
        assert 0.8 < asm_r <= 1.0


class TestMarginalRewardBaseline:
    """CROSS_UNDER_BELT's protected obstruction line delivers before the agent
    acts. The reward must not pay for that pre-existing flow (instant EOT would
    bank it) and must charge for destroying it."""

    def _reset_cross(self, env):
        for seed in range(20):
            try:
                return env.reset(
                    seed=seed, options={"kind": LessonKind.CROSS_UNDER_BELT}
                )
            except RuntimeError:
                continue
        pytest.fail("no CROSS_UNDER_BELT factory built in 20 seeds")

    def test_pre_existing_flow_pays_zero(self):
        env = _make_env(size=11, max_steps=10)
        self._reset_cross(env)

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        assert info["thput_raw"] > 0, "the protected line should deliver"
        assert reward == 0.0

    def test_destroying_the_protected_line_pays_negative(self):
        env = _make_env(size=11, max_steps=10)
        self._reset_cross(env)

        ent = env._world_CWH[Channel.ENTITIES.value].numpy()
        xs, ys = np.nonzero(ent == str2ent("transport_belt").value)
        action = _noop_action()
        action["xy"] = np.array([int(xs[0]), int(ys[0])])
        _, destroy_reward, terminated, truncated, _ = env.step(action)
        assert not terminated and not truncated
        assert destroy_reward < 0, "destroying flow must pay negative on that step"

        action = _noop_action()
        action["eot"] = 1
        _, eot_reward, terminated, _, info = env.step(action)

        assert terminated is True
        assert info["thput_raw"] < env._reward_baseline
        assert eot_reward == pytest.approx(0.0), "the EOT step changed nothing"
        assert destroy_reward + eot_reward == pytest.approx(_expected_total(env, info))


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
