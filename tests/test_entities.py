"""End-to-end behaviour of the entities only human blueprints use: inert
footprints, chests, furnaces and the big electric pole."""

import numpy as np
import pytest

from helpers import Channel, LessonKind, Misc, build_factory, entities, str2ent
import factorion_rs
from factorion import INERT_ENT_IDS, apply_placement_action


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
