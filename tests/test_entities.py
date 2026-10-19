"""End-to-end behaviour of the entities only human blueprints use: inert
footprints, chests, furnaces and the big electric pole."""

import numpy as np
import pytest
import torch

from helpers import Channel, LessonKind, Misc, build_factory, entities, str2ent
import factorion_rs
from factorion import INERT_ENT_IDS, apply_placement_action, count_unpowered


def _place(world, x, y, name):
    action = {"xy": (x, y), "entity": str2ent(name).value, "direction": 0, "item": 0, "misc": Misc.NONE.value}
    assert apply_placement_action(world, action) is None


def _throughput(world_CWH):
    return factorion_rs.simulate_throughput(np.ascontiguousarray(world_CWH.permute(1, 2, 0).numpy(), dtype=np.int64))


@pytest.mark.parametrize("entity", INERT_ENT_IDS, ids=lambda e: entities[e].name)
def test_inert_entities_leave_a_lessons_throughput_unchanged(entity):
    factory = build_factory(32, LessonKind.OPPOSITE_SIDES_1IN, seed=0)
    assert factory is not None
    world = factory.world_CWH.clone()
    ent = world[Channel.ENTITIES.value].numpy()
    side = entities[entity].width
    placed = 0
    for x in range(0, 32 - side + 1, side):
        for y in range(0, 32 - side + 1, side):
            if not ent[x : x + side, y : y + side].any():
                action = {"xy": (x, y), "entity": entity, "direction": 0, "item": 0, "misc": Misc.NONE.value}
                assert apply_placement_action(world, action) is None
                placed += 1
    assert placed > 0
    assert _throughput(world)[0] == pytest.approx(factory.max_throughput)
    # No node of their own, so nothing new is unreachable.
    assert _throughput(world)[1] == _throughput(factory.world_CWH)[1]
    assert (world[Channel.POWER.value] == factory.world_CWH[Channel.POWER.value]).all()


def test_only_the_electric_furnace_needs_power():
    world = torch.zeros((len(Channel), 20, 20), dtype=torch.int64)
    world[Channel.FOOTPRINT.value] = 1
    _place(world, 10, 10, "electric_furnace")
    _place(world, 2, 10, "stone_furnace")
    _place(world, 5, 10, "steel_furnace")
    assert count_unpowered(world) == 1
    _place(world, 13, 10, "medium_electric_pole")
    _place(world, 16, 10, "medium_electric_pole")
    assert count_unpowered(world) == 0


@pytest.mark.parametrize(
    "poles, powered",
    [
        (("big_electric_pole", "big_electric_pole"), True),
        # 30 tiles is beyond a medium pole's 9-tile wire.
        (("medium_electric_pole", "big_electric_pole"), False),
        # Over 32 tiles from every edge, the far pole is live only through the near one.
        ((None, "big_electric_pole"), False),
    ],
)
def test_a_big_pole_powers_an_assembler_across_a_wide_gap(poles, powered):
    world = torch.zeros((len(Channel), 100, 100), dtype=torch.int64)
    world[Channel.FOOTPRINT.value] = 1
    action = {"xy": (34, 50), "entity": str2ent("assembling_machine_1").value, "direction": 0,
              "item": str2ent("iron_gear_wheel").value, "misc": Misc.NONE.value}
    assert apply_placement_action(world, action) is None
    near, far = poles
    if near:
        _place(world, 2, 50, near)
    _place(world, 32, 50, far)
    assert count_unpowered(world) == (0 if powered else 1)
