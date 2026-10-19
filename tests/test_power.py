"""Electric power end to end: generated factories carry their own poles,
the POWER channel follows every pole placement, and an unpowered factory
makes nothing (wiki/electric-network.md)."""

import numpy as np
import pytest
import torch

from helpers import Channel, Direction, LessonKind, Misc, blank_entities, build_factory, str2ent
import factorion_rs
from denoise import TYPES
from factorion import POLE_ENT_IDS, UNDIRECTED_ENT_IDS, Footprint, apply_placement_action, count_unpowered

POWER = Channel.POWER.value
MEDIUM_POLE = str2ent("medium_electric_pole").value


def _place(world, x, y, entity, direction=Direction.NONE):
    return apply_placement_action(
        world, {"xy": (x, y), "entity": entity, "direction": direction.value, "item": 0, "misc": Misc.NONE.value}
    )


def _empty_world(size):
    world = torch.zeros((len(Channel), size, size))
    world[Channel.FOOTPRINT.value] = 1
    return world


@pytest.mark.parametrize("seed", range(3))
def test_assembler_lessons_need_their_poles(seed):
    factory = build_factory(15, LessonKind.OPPOSITE_SIDES_1IN, seed=seed)
    assert factory is not None
    world = factory.world_CWH.clone()
    poles = np.isin(world[Channel.ENTITIES.value].numpy(), POLE_ENT_IDS)
    assert poles.any()
    world_WHC = np.ascontiguousarray(world.permute(1, 2, 0).numpy(), dtype=np.int64)
    assert np.array_equal(world_WHC[..., POWER].astype(bool), factorion_rs.py_power_map(world_WHC[None])[0])
    assert factorion_rs.simulate_throughput(world_WHC)[0] == pytest.approx(factory.max_throughput)

    inserters = np.isin(world_WHC[..., Channel.ENTITIES.value], [str2ent(n).value for n in ("inserter", "long_handed_inserter")]).sum()
    assert count_unpowered(world) == 0
    world_WHC[poles, Channel.ENTITIES.value] = 0
    # Every inserter, plus each 3x3 assembler once.
    assemblers = (world_WHC[..., Channel.ENTITIES.value] == str2ent("assembling_machine_1").value).sum() // 9
    assert count_unpowered(torch.from_numpy(world_WHC).permute(2, 0, 1)) == inserters + assemblers
    assert factorion_rs.simulate_throughput(world_WHC)[0] == 0


def test_blanking_unpowers_the_grid():
    factory = build_factory(15, LessonKind.OPPOSITE_SIDES_1IN, seed=0)
    assert factory is not None and factory.world_CWH[POWER].any()
    blank, _ = blank_entities(factory)
    assert not blank[POWER].any()


def test_placing_and_removing_a_pole_moves_power():
    world = _empty_world(15)
    assert _place(world, 0, 0, MEDIUM_POLE) is None
    assert world[POWER, :4, :4].all() and world[POWER].sum() == 16
    assert _place(world, 0, 0, str2ent("empty").value) is None
    assert not world[POWER].any()


def test_a_pole_too_far_from_the_edge_powers_nothing():
    # The medium pole's wire reaches 9: from (9, 9) the nearest off-map
    # tile is 10 away.
    world = _empty_world(19)
    _place(world, 9, 9, MEDIUM_POLE)
    assert not world[POWER].any()
    # A relay within reach of both the edge and the first pole powers it.
    _place(world, 1, 9, MEDIUM_POLE)
    assert world[POWER, 9, 12]


def test_a_pole_has_no_facing():
    world = _empty_world(5)
    assert _place(world, 2, 2, MEDIUM_POLE, Direction.EAST) == "undirected_with_direction"
    assert not world[Channel.ENTITIES.value].any()


def test_the_denoiser_places_undirected_entities_facing_nowhere():
    assert {d for e, d, _ in TYPES if e in UNDIRECTED_ENT_IDS} == {Direction.NONE.value}


def test_unknown_tiles_on_the_edge_power_a_pole_as_unbuildable_ones_do():
    """What lies past a human blueprint's window is off the map to the factory,
    so a pole can draw power from it just as from unbuildable tiles."""
    for footprint in (Footprint.UNAVAILABLE, Footprint.UNKNOWN):
        world = _empty_world(30)
        world[Channel.FOOTPRINT.value, :20] = footprint.value
        assert _place(world, 22, 5, MEDIUM_POLE) is None
        assert world[POWER, 22, 5] == 1
