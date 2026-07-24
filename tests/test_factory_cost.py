"""Tests for recursive per-entity cost data and aggregation."""

import os

import numpy as np
import pytest

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

from factorion import Channel, LessonKind, str2ent
from ppo import (
    FactorioEnv,
    _ENTITY_FOOTPRINT_AREAS,
    _ENTITY_UNIT_COSTS,
)


def test_recursive_unit_costs_include_raw_items_and_craft_time():
    belt = str2ent("transport_belt")
    inserter = str2ent("inserter")
    assembler = str2ent("assembling_machine_1")

    # One craft costs 3 iron + 1 cumulative second and yields two belts.
    assert _ENTITY_UNIT_COSTS[belt.value] == pytest.approx(2.0)
    # More deeply nested recipes cost more than the belt recipe.
    assert _ENTITY_UNIT_COSTS[inserter.value] == pytest.approx(7.75)
    assert _ENTITY_UNIT_COSTS[assembler.value] == pytest.approx(33.25)


def test_entity_cost_charges_multitile_entity_once_and_sums_types():
    env = FactorioEnv(size=8, max_steps=10, idx=0)
    env.reset(
        seed=42,
        options={
            "kind": LessonKind.MOVE_ONE_ITEM,
            "num_missing_entities": 99,
        },
    )

    entity_grid = env._world_CWH[Channel.ENTITIES.value]
    belt = str2ent("transport_belt")
    inserter = str2ent("inserter")
    assembler = str2ent("assembling_machine_1")

    anchor = next(
        (x, y)
        for x in range(6)
        for y in range(6)
        if np.all(entity_grid[x:x + 3, y:y + 3].numpy() == 0)
    )
    x, y = anchor
    entity_grid[x:x + 3, y:y + 3] = assembler.value
    remaining = [
        (int(x), int(y))
        for x, y in np.argwhere(entity_grid.numpy() == 0)
    ]
    entity_grid[remaining[0]] = belt.value
    entity_grid[remaining[1]] = inserter.value

    action = {
        "xy": np.array([0, 0]),
        "entity": 0,
        "direction": 0,
        "item": 0,
        "misc": 0,
        "eot": 1,
    }
    _, _, terminated, _, info = env.step(action)

    assert terminated
    assert _ENTITY_FOOTPRINT_AREAS[assembler.value] == 9
    assert info["entity_cost"] == pytest.approx(33.25 + 2.0 + 7.75)


def test_two_underground_belts_cost_more_than_four_transport_belts():
    belt = str2ent("transport_belt")
    underground = str2ent("underground_belt")

    # Two undergrounds are one craft: 17.5 raw iron + 3.5 cumulative seconds.
    assert 2 * _ENTITY_UNIT_COSTS[underground.value] == pytest.approx(21.0)
    # Four belts are two crafts: 6 raw iron + 2 cumulative seconds.
    assert 4 * _ENTITY_UNIT_COSTS[belt.value] == pytest.approx(8.0)
