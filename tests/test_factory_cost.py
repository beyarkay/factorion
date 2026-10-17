"""Tests for recursive per-entity cost data and aggregation."""

import numpy as np
import pytest

from helpers import Channel, str2ent
from denoise import _ENTITY_FOOTPRINT_AREAS, _ENTITY_UNIT_COSTS, entity_cost

BELT = str2ent("transport_belt").value
INSERTER = str2ent("inserter").value
ASM = str2ent("assembling_machine_1").value
UG = str2ent("underground_belt").value


def test_recursive_unit_costs_include_raw_items_and_craft_time():
    # One craft costs 3 iron + 1 cumulative second and yields two belts.
    assert _ENTITY_UNIT_COSTS[BELT] == pytest.approx(2.0)
    # More deeply nested recipes cost more than the belt recipe.
    assert _ENTITY_UNIT_COSTS[INSERTER] == pytest.approx(7.75)
    assert _ENTITY_UNIT_COSTS[ASM] == pytest.approx(33.25)


def test_entity_cost_charges_multitile_entity_once_and_sums_types():
    world = np.zeros((len(Channel), 8, 8), dtype=np.int64)
    ent = world[Channel.ENTITIES.value]
    ent[0:3, 0:3] = ASM
    ent[5, 5], ent[6, 6] = BELT, INSERTER
    assert _ENTITY_FOOTPRINT_AREAS[ASM] == 9
    cost, efficiency = entity_cost(world, 0.001)
    assert cost == pytest.approx(33.25 + 2.0 + 7.75)
    assert efficiency == pytest.approx(1 / (1 + 0.001 * cost))


def test_two_underground_belts_cost_more_than_four_transport_belts():
    # Two undergrounds are one craft: 17.5 raw iron + 3.5 cumulative seconds.
    assert 2 * _ENTITY_UNIT_COSTS[UG] == pytest.approx(21.0)
    # Four belts are two crafts: 6 raw iron + 2 cumulative seconds.
    assert 4 * _ENTITY_UNIT_COSTS[BELT] == pytest.approx(8.0)
