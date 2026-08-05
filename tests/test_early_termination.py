"""Tests for early termination when the agent solves the puzzle."""

import math
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
    """Create a FactorioEnv for testing, with the connection shaping off unless
    a test opts in — the reward tests below are about the throughput term."""
    kwargs.setdefault("connect_coef", 0.0)
    return FactorioEnv(size=size, max_steps=max_steps, idx=0, **kwargs)


def _expected_reward(env, info):
    """The terminal reward the env's configured scheme should have paid."""
    reward = info["thput_raw"] * info["cost_efficiency"]
    if env.reward_symlog_r0 > 0:
        return math.log1p(reward / env.reward_symlog_r0)
    return reward


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
    """Reward = raw_throughput * cost_efficiency, log-compressed at r0 unless
    the compression is disabled. The cost multiplier is bounded in (0, 1] and
    log1p is non-negative on it, so cost can reduce reward but never make it
    negative under either scheme."""

    def test_solved_factory_with_eot_pays_raw_throughput_reward(self):
        """Declaring eot on a solved factory pays cost-adjusted raw throughput."""
        env = _make_env(size=5, max_steps=10)
        env.reset(seed=42, options={"num_missing_entities": 0})

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        assert info["thput_normed"] >= 1.0
        assert reward == pytest.approx(_expected_reward(env, info))

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
        assert reward == pytest.approx(_expected_reward(env, info))

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
        assert reward == pytest.approx(_expected_reward(env, info))

    def test_entity_cost_reduces_reward(self):
        """A non-zero entity cost lowers the reward below the pure
        throughput term."""
        env = _make_env(size=5, max_steps=10)
        env.entity_cost_scale = 0.01
        env.reset(seed=42, options={"num_missing_entities": 0})

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        assert reward == pytest.approx(_expected_reward(env, info))
        assert info["entity_cost"] > 0
        assert reward < math.log1p(info["thput_raw"] / env.reward_symlog_r0)

    def test_entity_cost_reduces_reward_multiplicatively_without_log(self):
        """With the log transform off, cost is a multiplier bounded in (0, 1]."""
        env = _make_env(size=5, max_steps=10, reward_symlog_r0=0.0)
        env.entity_cost_scale = 0.01
        env.reset(seed=42, options={"num_missing_entities": 0})

        action = _noop_action()
        action["eot"] = 1
        _, reward, terminated, _, info = env.step(action)

        assert terminated is True
        expected = info["thput_raw"] * info["cost_efficiency"]
        assert reward == pytest.approx(expected)
        assert info["entity_cost"] > 0
        assert 0 < info["cost_efficiency"] < 1
        assert reward < info["thput_raw"]

    @pytest.mark.parametrize("reward_symlog_r0", [0.0, 0.01])
    def test_zero_throughput_cost_penalty_is_never_negative(self, reward_symlog_r0):
        """Entities that deliver nothing must score exactly zero, not negative:
        else an empty grid beats a nearly-complete factory and the policy is
        rewarded for building nothing. Holds under either scheme because the
        cost multiplier is bounded in (0, 1] and log1p(0) == 0."""
        env = FactorioEnv(
            size=5,
            max_steps=10,
            idx=0,
            entity_cost_scale=1_000_000.0,
            reward_symlog_r0=reward_symlog_r0,
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


class TestLogRewardScaleCompression:
    """The point of the log transform: lessons whose achievable items/s differ
    by orders of magnitude must not differ by orders of magnitude in reward."""

    def _solved_reward(self, kind, size=11):
        """Terminal reward for eot on an already-solved factory of `kind`."""
        env = _make_env(size=size, max_steps=10)
        env.reset(seed=7, options={"num_missing_entities": 0, "kind": kind})
        action = _noop_action()
        action["eot"] = 1
        _, reward, _, _, info = env.step(action)
        return reward, info["thput_raw"]

    def test_belt_and_assembler_rewards_are_within_one_order_of_magnitude(self):
        belt_r, belt_thput = self._solved_reward(LessonKind.MOVE_ONE_ITEM)
        asm_r, asm_thput = self._solved_reward(
            LessonKind.MEMORISE_4_INGREDIENT_RECIPES
        )

        raw_ratio = belt_thput / asm_thput
        reward_ratio = belt_r / asm_r

        assert raw_ratio > 50, "expected a large raw items/s gap between lessons"
        assert reward_ratio < 10, (
            f"log reward still spreads lessons {reward_ratio:.1f}x "
            f"(raw gap {raw_ratio:.1f}x)"
        )


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




def _solved_belt_actions(env):
    """The solved factory's belt placements, ordered source-outward.

    Valid for single-route lessons (MOVE_ONE_ITEM): #349 routes are shortest
    paths, so Manhattan distance from the source is strictly increasing along
    the route and sorting by it recovers path order.
    """
    ent = env._solved_world_CWH[Channel.ENTITIES.value].numpy()
    dir_ = env._solved_world_CWH[Channel.DIRECTION.value].numpy()
    belt = str2ent("transport_belt").value
    xs, ys = np.nonzero(ent == str2ent("stack_inserter").value)
    sx, sy = int(xs[0]), int(ys[0])
    tiles = [(int(x), int(y)) for x, y in zip(*np.nonzero(ent == belt))]
    tiles.sort(key=lambda t: abs(t[0] - sx) + abs(t[1] - sy))
    actions = []
    for x, y in tiles:
        a = _noop_action()
        a["xy"] = np.array([x, y])
        a["entity"] = belt
        a["direction"] = int(dir_[x, y])
        actions.append(a)
    return actions


class TestPerStepGapShaping:
    """The gap-closing bonus is potential-based and paid per step
    (F_t = gamma*Phi(s') - Phi(s)), so credit lands on the placement that
    closed the gap instead of a terminal lump GAE discounts away (#353
    post-mortem). Asserted against rewards the env actually paid."""

    KIND = LessonKind.MOVE_ONE_ITEM

    def _blank_env(self, **kwargs):
        kwargs.setdefault("connect_coef", 0.25)
        env = _make_env(size=11, max_steps=50, **kwargs)
        env.reset(seed=7, options={"num_missing_entities": float("inf"),
                                   "kind": self.KIND})
        return env

    def test_placement_that_closes_gap_pays_that_step(self):
        """At gamma=1 the stream is exactly Phi(s')-Phi(s): rebuilding the
        solved route pays each gap-closing placement on the spot, and a no-op
        pays exactly 0."""
        env = self._blank_env(gamma=1.0)
        mid_rewards, acrs = [], []
        for a in _solved_belt_actions(env):
            _, reward, _, _, info = env.step(a)
            mid_rewards.append(reward)
            acrs.append(info["almost_connected_reward"])
        assert acrs[-1] == pytest.approx(1.0), "full route must close the gap"
        assert acrs == sorted(acrs), "rebuilding the route must be monotone"
        assert all(r >= 0 for r in mid_rewards), mid_rewards
        assert sum(mid_rewards) > 0, "the stream must pay before the terminal"

        _, noop_r, _, _, _ = env.step(_noop_action())
        assert noop_r == 0.0, "no progress, no pay"

        eot = _noop_action()
        eot["eot"] = 1
        _, terminal, terminated, _, info = env.step(eot)
        assert terminated
        assert terminal == pytest.approx(_expected_reward(env, info)), (
            "at gamma=1 the terminal step carries no shaping (world unchanged)"
        )

    def test_return_ladder_from_blank(self):
        """Building more of the route then stopping must collect strictly more
        total reward — the ladder #353 wanted, in its honest form: earned by
        building, not banked from the reset state."""
        totals = []
        for frac in (0.0, 0.5, 1.0):
            env = self._blank_env(gamma=1.0)
            actions = _solved_belt_actions(env)
            total = 0.0
            for a in actions[: round(len(actions) * frac)]:
                _, r, _, _, _ = env.step(a)
                total += r
            eot = _noop_action()
            eot["eot"] = 1
            _, r, _, _, _ = env.step(eot)
            totals.append(total + r)
        assert totals == sorted(totals) and totals[0] < totals[-1], totals
        assert totals[0] == 0.0, "a bare-markers quit is the zero of the scale"

    def test_no_credit_for_structure_the_agent_did_not_build(self):
        """Quitting instantly on a factory that reset() delivered almost-solved
        pays nothing: Phi(s_0) is baseline, not credit. (Under #353's terminal
        form this banked the full bonus for free.)"""
        env = _make_env(size=11, max_steps=50, connect_coef=0.25)
        env.reset(seed=7, options={"num_missing_entities": 1, "kind": self.KIND})
        eot = _noop_action()
        eot["eot"] = 1
        _, reward, _, _, info = env.step(eot)
        assert info["thput_raw"] == 0.0, "one missing entity breaks the chain"
        assert reward <= 0.0, reward

    def test_holding_potential_is_taxed_at_the_discount(self):
        """With gamma<1, idling on partial progress pays (gamma-1)*Phi < 0 per
        step — the missing pressure to either keep building or declare done
        (#352's diagnosis)."""
        env = self._blank_env()
        assert env.gamma < 1.0
        actions = _solved_belt_actions(env)
        for a in actions[: len(actions) // 2]:
            env.step(a)
        _, noop_r, _, _, info = env.step(_noop_action())
        assert info["almost_connected_reward"] > 0
        assert noop_r < 0.0, noop_r

    def test_whole_stream_worth_less_than_the_solve(self):
        """Summed over a perfect build, the shaping is a fraction of the solve
        itself (connect_coef scales against each lesson's own ceiling), so a
        connected-but-dead factory can never outrank a delivering one."""
        returns = {}
        for coef in (0.25, 0.0):
            env = self._blank_env(gamma=1.0, connect_coef=coef)
            total = 0.0
            for a in _solved_belt_actions(env):
                _, r, _, _, _ = env.step(a)
                total += r
            eot = _noop_action()
            eot["eot"] = 1
            _, r, _, _, info = env.step(eot)
            returns[coef] = total + r
            assert info["thput_raw"] > 0, "replay must actually solve the lesson"
        shaping_total = returns[0.25] - returns[0.0]
        assert 0 < shaping_total < returns[0.0]

    def test_zero_coef_is_a_true_noop(self):
        env = self._blank_env(connect_coef=0.0)
        for a in _solved_belt_actions(env):
            _, r, _, _, info = env.step(a)
            assert r == 0.0
            assert info["almost_connected_reward"] == 0.0
        eot = _noop_action()
        eot["eot"] = 1
        _, terminal, _, _, info = env.step(eot)
        assert terminal == pytest.approx(_expected_reward(env, info))
