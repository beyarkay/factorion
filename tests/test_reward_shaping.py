"""Tests for delta-based reward shaping (PBRS)."""

import os
import sys

import pytest
import torch
import gymnasium as gym
import numpy as np

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv, make_env  # noqa: E402
from helpers import generate_lesson, LessonKind  # noqa: E402


ENV_ID = "factorion/FactorioEnv-v0-reward-shaping-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point=FactorioEnv)


@pytest.fixture()
def env(registered_env):
    env = gym.make(ENV_ID, size=5, max_steps=10)
    yield env
    env.close()


NOOP = {
    "xy": np.array([0, 0]),
    "entity": 0,  # empty
    "direction": 0,  # NONE
    "item": 0,
    "misc": 0,
}


class TestDeltaRewardShaping:
    def test_perfect_factory_match_is_1(self, env):
        """When num_missing_entities=0, all three match values should be 1.0."""
        obs, info = env.reset(seed=42, options={"num_missing_entities": 0})
        obs, reward, term, trunc, info = env.step(NOOP)
        assert info["shaping_location_match"] == 1.0
        assert info["shaping_entity_match"] == 1.0
        assert info["shaping_direction_match"] == 1.0

    def test_missing_entity_match_below_1(self, env):
        """When num_missing_entities=1, match values should be < 1.0."""
        obs, info = env.reset(seed=42, options={"num_missing_entities": 1})
        obs, reward, term, trunc, info = env.step(NOOP)
        # At least one of location/entity match must be below 1.0
        assert info["shaping_location_match"] < 1.0 or info["shaping_entity_match"] < 1.0, (
            f"Expected at least one match < 1.0 with missing entity"
        )

    def test_noop_gives_zero_deltas(self, env):
        """Placing empty entity should give all deltas = 0 (no free reward)."""
        obs, info = env.reset(seed=42, options={"num_missing_entities": 1})
        obs, reward, term, trunc, info = env.step(NOOP)
        assert info["shaping_location_delta"] == 0.0, (
            f"Expected 0 delta for noop, got {info['shaping_location_delta']}"
        )
        assert info["shaping_entity_delta"] == 0.0, (
            f"Expected 0 delta for noop, got {info['shaping_entity_delta']}"
        )
        assert info["shaping_direction_delta"] == 0.0, (
            f"Expected 0 delta for noop, got {info['shaping_direction_delta']}"
        )

    def test_deltas_nonpositive_for_intact_factory(self, env):
        """Can't improve a perfect factory — all deltas should be <= 0."""
        obs, info = env.reset(seed=42, options={"num_missing_entities": 0})
        obs, reward, term, trunc, info = env.step(NOOP)
        assert info["shaping_location_delta"] <= 0.0
        assert info["shaping_entity_delta"] <= 0.0
        assert info["shaping_direction_delta"] <= 0.0

    def test_match_values_between_0_and_1(self, env):
        """All match values should be bounded in [0, 1] across seeds."""
        for seed in range(5):
            for missing in [0, 1]:
                obs, info = env.reset(
                    seed=seed, options={"num_missing_entities": missing}
                )
                obs, reward, term, trunc, info = env.step(NOOP)
                for key in [
                    "shaping_location_match",
                    "shaping_entity_match",
                    "shaping_direction_match",
                ]:
                    assert 0.0 <= info[key] <= 1.0, (
                        f"{key}={info[key]} out of [0,1] (seed={seed}, missing={missing})"
                    )

    def test_shaping_info_keys_present(self, env):
        """Info dict should contain all delta and match keys."""
        obs, info = env.reset(seed=42, options={"num_missing_entities": 1})
        obs, reward, term, trunc, info = env.step(NOOP)
        for key in [
            "shaping_location_match",
            "shaping_entity_match",
            "shaping_direction_match",
            "shaping_location_delta",
            "shaping_entity_delta",
            "shaping_direction_delta",
        ]:
            assert key in info, f"Missing key: {key}"

    def test_diagnostic_tile_match_keys_still_present(self, env):
        """Backward-compat: diagnostic tile_match keys should still be in info."""
        obs, info = env.reset(seed=42, options={"num_missing_entities": 1})
        obs, reward, term, trunc, info = env.step(NOOP)
        assert "tile_match_location" in info
        assert "tile_match_entity" in info
        assert "tile_match_direction" in info


class TestSeedConsistency:
    """Verify that generate_lesson with the same seed produces the same layout.

    This is critical for reward shaping: we call generate_lesson twice
    (once with num_missing_entities=0 for the solved factory, once with
    the actual value) and rely on the same seed producing the same layout.
    """

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
    def test_same_seed_same_layout(self, seed):
        """Two calls with same seed and num_missing_entities=0 produce identical worlds."""
        world_a, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=0, seed=seed,
        )
        world_b, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=0, seed=seed,
        )
        assert torch.equal(world_a, world_b), (
            f"seed={seed}: two generate_lesson calls with same seed differ"
        )

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
    def test_solved_is_superset_of_incomplete(self, seed):
        """The solved factory (num_missing=0) should have entities everywhere
        the incomplete factory (num_missing=1) does, plus the removed ones."""
        solved, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=0, seed=seed,
        )
        incomplete, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=1, seed=seed,
        )
        from helpers import Channel, str2ent  # noqa: E402

        solved_ent = solved[Channel.ENTITIES.value]
        incomplete_ent = incomplete[Channel.ENTITIES.value]

        # Every non-empty tile in the incomplete factory should match the solved factory
        nonempty_mask = (incomplete_ent != str2ent("empty").value)
        assert torch.equal(
            solved_ent[nonempty_mask], incomplete_ent[nonempty_mask]
        ), f"seed={seed}: incomplete factory has entities that differ from solved"

        # The solved factory should have >= as many non-empty entities
        assert (solved_ent != str2ent("empty").value).sum() >= (
            incomplete_ent != str2ent("empty").value
        ).sum(), f"seed={seed}: solved factory has fewer entities than incomplete"

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
    def test_different_seeds_different_layouts(self, seed):
        """Different seeds should (almost always) produce different layouts."""
        world_a, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=0, seed=seed,
        )
        world_b, _ = generate_lesson(
            size=5,
            kind=LessonKind.MOVE_ONE_ITEM,
            num_missing_entities=0,
            seed=seed + 1000,
        )
        assert not torch.equal(world_a, world_b), (
            f"seeds {seed} and {seed + 1000} produced identical worlds"
        )
