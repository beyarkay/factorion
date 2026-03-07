"""Tests for the FOOTPRINT mask channel."""

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
from helpers import Channel, Direction, Misc, make_world, set_entity, str2ent  # noqa: E402

ENV_ID = "factorion/FactorioEnv-v0-mask-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point=FactorioEnv)


@pytest.fixture()
def env(registered_env):
    envs = gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, 0, False, 5, "test")]
    )
    return envs.envs[0].unwrapped


def find_tile(env, entity_name):
    """Find the first tile with the given entity type. Returns (x, y) or None."""
    val = str2ent(entity_name).value
    for x in range(env.size):
        for y in range(env.size):
            if env._world_CWH[Channel.ENTITIES.value, x, y] == val:
                return x, y
    return None


class TestFootprintChannel:
    def test_channel_exists(self):
        """FOOTPRINT channel is defined and has value 4."""
        assert hasattr(Channel, "FOOTPRINT")
        assert Channel.FOOTPRINT.value == 4

    def test_world_has_five_channels(self):
        """new_world creates tensors with 5 channels."""
        world = make_world(5)
        assert world.shape[2] == len(Channel)
        assert world.shape[2] == 5

    def test_default_footprint_is_available(self, env):
        """By default all tiles should be AVAILABLE (editable)."""
        obs, _ = env.reset(options={"num_missing_entities": 1})
        footprint = obs[Channel.FOOTPRINT.value]
        assert (footprint == 1).all(), f"Expected all AVAILABLE, got:\n{footprint}"

    def test_observation_space_shape(self, env):
        """Observation space should have 5 channels."""
        assert env.observation_space.shape[0] == len(Channel)


class TestMaskedPlacement:
    def test_placement_on_masked_tile_is_invalid(self, env):
        """Placing an entity on a FOOTPRINT=0 tile should be rejected."""
        env.reset(options={"num_missing_entities": 1})
        target = find_tile(env, "empty")
        assert target is not None, "No empty tile found"
        tx, ty = target

        env._world_CWH[Channel.FOOTPRINT.value, tx, ty] = 0

        invalid_before = env.invalid_actions
        env.step({
            "xy": np.array([tx, ty]),
            "entity": 1,  # transport_belt
            "direction": Direction.EAST.value,
            "item": 0,
            "misc": Misc.NONE.value,
        })

        assert env.invalid_actions == invalid_before + 1
        assert env._world_CWH[Channel.ENTITIES.value, tx, ty] == str2ent("empty").value

    def test_placement_on_available_tile_succeeds(self, env):
        """Placing an entity on a FOOTPRINT=1 tile should work normally."""
        env.reset(options={"num_missing_entities": 1})
        target = find_tile(env, "empty")
        assert target is not None, "No empty tile found"
        tx, ty = target

        assert env._world_CWH[Channel.FOOTPRINT.value, tx, ty] == 1

        invalid_before = env.invalid_actions
        env.step({
            "xy": np.array([tx, ty]),
            "entity": 1,  # transport_belt
            "direction": Direction.EAST.value,
            "item": 0,
            "misc": Misc.NONE.value,
        })

        assert env.invalid_actions == invalid_before
        assert env._world_CWH[Channel.ENTITIES.value, tx, ty] == str2ent("transport_belt").value

    def test_mask_does_not_change_on_invalid_action(self, env):
        """The footprint channel should not be modified by a rejected action."""
        env.reset(options={"num_missing_entities": 1})
        target = find_tile(env, "empty")
        assert target is not None, "No empty tile found"
        tx, ty = target

        env._world_CWH[Channel.FOOTPRINT.value, tx, ty] = 0
        footprint_before = env._world_CWH[Channel.FOOTPRINT.value].clone()

        env.step({
            "xy": np.array([tx, ty]),
            "entity": 1,
            "direction": Direction.EAST.value,
            "item": 0,
            "misc": Misc.NONE.value,
        })

        torch.testing.assert_close(
            env._world_CWH[Channel.FOOTPRINT.value],
            footprint_before,
        )

    def test_placing_empty_on_masked_tile_is_invalid(self, env):
        """Even placing 'empty' on a masked tile should be rejected."""
        env.reset(options={"num_missing_entities": 1})

        env._world_CWH[Channel.FOOTPRINT.value, 0, 0] = 0

        invalid_before = env.invalid_actions
        env.step({
            "xy": np.array([0, 0]),
            "entity": 0,  # empty
            "direction": Direction.NONE.value,
            "item": 0,
            "misc": Misc.NONE.value,
        })

        assert env.invalid_actions == invalid_before + 1

    def test_mask_checked_before_source_sink(self, env):
        """Mask check should fire before the source/sink replacement check."""
        env.reset(options={"num_missing_entities": 1})
        target = find_tile(env, "source")
        assert target is not None, "No source tile found"
        src_x, src_y = target

        env._world_CWH[Channel.FOOTPRINT.value, src_x, src_y] = 0

        invalid_before = env.invalid_actions
        env.step({
            "xy": np.array([src_x, src_y]),
            "entity": 1,
            "direction": Direction.EAST.value,
            "item": 0,
            "misc": Misc.NONE.value,
        })

        assert env.invalid_actions == invalid_before + 1


class TestMixedMaskScenarios:
    def test_all_tiles_masked_all_actions_invalid(self, env):
        """If entire grid is masked, every placement should be invalid."""
        env.reset(options={"num_missing_entities": 0})
        env._world_CWH[Channel.FOOTPRINT.value, :, :] = 0

        invalid_before = env.invalid_actions
        for x in range(env.size):
            for y in range(env.size):
                env.step({
                    "xy": np.array([x, y]),
                    "entity": 1,
                    "direction": Direction.EAST.value,
                    "item": 0,
                    "misc": Misc.NONE.value,
                })
        assert env.invalid_actions == invalid_before + env.size * env.size

    def test_partial_mask(self, env):
        """Only unmasked tiles should accept placements."""
        env.reset(options={"num_missing_entities": 0})

        env._world_CWH[Channel.FOOTPRINT.value, :, :] = 0

        target = find_tile(env, "empty")
        if target is None:
            pytest.skip("No empty tile found")
        tx, ty = target

        env._world_CWH[Channel.FOOTPRINT.value, tx, ty] = 1

        invalid_before = env.invalid_actions
        env.step({
            "xy": np.array([tx, ty]),
            "entity": 1,
            "direction": Direction.EAST.value,
            "item": 0,
            "misc": Misc.NONE.value,
        })
        assert env.invalid_actions == invalid_before

    def test_mask_persists_across_steps(self, env):
        """Mask values should not change between steps."""
        env.reset(options={"num_missing_entities": 1})

        env._world_CWH[Channel.FOOTPRINT.value, 0, :] = 0
        env._world_CWH[Channel.FOOTPRINT.value, 1, :] = 1
        mask_snapshot = env._world_CWH[Channel.FOOTPRINT.value].clone()

        for _ in range(3):
            env.step({
                "xy": np.array([2, 2]),
                "entity": 1,
                "direction": Direction.EAST.value,
                "item": 0,
                "misc": Misc.NONE.value,
            })

        torch.testing.assert_close(
            env._world_CWH[Channel.FOOTPRINT.value],
            mask_snapshot,
        )


class TestThroughputWithMask:
    def test_mask_does_not_affect_throughput(self):
        """The FOOTPRINT channel should be ignored by the throughput engine."""
        import factorion_rs

        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "transport_belt", Direction.EAST)
        set_entity(world, 4, 0, "bulk_inserter", Direction.EAST, "copper_cable")

        t1, u1 = factorion_rs.simulate_throughput(world.numpy().astype(np.int64))

        world[2, 2, Channel.FOOTPRINT.value] = 0
        world[3, 3, Channel.FOOTPRINT.value] = 0
        t2, u2 = factorion_rs.simulate_throughput(world.numpy().astype(np.int64))

        assert t1 == t2
        assert u1 == u2
