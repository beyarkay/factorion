"""`factorion.apply_placement_action`: the one placement path for the factory
builder and the mod server."""

import pytest
import torch

from helpers import Channel, Direction, Footprint, Misc, str2ent
from factorion import apply_placement_action

ENT, DIR, ITEMS, MISC, FOOT = (c.value for c in Channel)
EMPTY = str2ent("empty").value
SOURCE = str2ent("source").value
BELT = str2ent("transport_belt").value
ASM = str2ent("assembling_machine_1").value
SPLITTER = str2ent("splitter").value
UG = str2ent("underground_belt").value
RECIPE = str2ent("electronic_circuit").value
NONE, EAST = Direction.NONE.value, Direction.EAST.value


def _world():
    w = torch.zeros(len(Channel), 6, 6, dtype=torch.int64)
    w[FOOT] = Footprint.AVAILABLE.value
    w[ENT, 0, 0], w[DIR, 0, 0] = SOURCE, EAST
    return w


def _place(w, x, y, entity, direction=EAST, item=0, misc=Misc.NONE.value):
    return apply_placement_action(w, {"xy": (x, y), "entity": entity, "direction": direction, "item": item, "misc": misc})


def _asm(w, x, y):
    return _place(w, x, y, ASM, NONE, RECIPE)


def test_assembler_fills_its_three_by_three_footprint():
    w = _world()
    assert _asm(w, 0, 3) is None
    assert (w[ENT, 0:3, 3:6] == ASM).all() and (w[ITEMS, 0:3, 3:6] == RECIPE).all()
    assert (w[ENT] == ASM).sum() == 9


@pytest.mark.parametrize(
    ("action", "reason"),
    [
        ((2, 2, ASM, NONE), "place_asm_mach_wo_recipe"),
        ((4, 4, ASM, NONE, RECIPE), "too_wide"),
        ((0, 0, BELT), "replaced_source_or_sink"),
        ((1, 0, SOURCE), "placed_source_or_sink"),
        ((1, 1, BELT, NONE), "placement_wo_direction"),
        ((1, 1, EMPTY), "direction_wo_entity"),
        ((1, 1, UG), "ug_belt_wo_up_or_down"),
        ((1, 1, BELT, EAST, 0, Misc.UNDERGROUND_UP.value), "placement_with_unneeded_misc"),
        ((5, 2, SPLITTER, Direction.NORTH.value), "too_wide"),
        ((2, 5, SPLITTER, EAST), "too_wide"),
    ],
)
def test_rejected_placements_leave_the_world_unchanged(action, reason):
    w = _world()
    before = w.clone()
    assert _place(w, *action) == reason
    assert torch.equal(w, before)


def test_walls_cannot_be_built_on_or_deleted():
    w = _world()
    w[FOOT, 2, 2] = Footprint.UNAVAILABLE.value
    w[ENT, 2, 2], w[DIR, 2, 2] = BELT, EAST
    assert _place(w, 2, 2, EMPTY, NONE) == "placed_on_masked_tile"
    assert _asm(w, 1, 1) == "placed_on_masked_tile"
    assert w[ENT, 2, 2] == BELT


@pytest.mark.parametrize("tile", [(x, y) for x in range(3) for y in range(3, 6)])
def test_empty_on_any_assembler_tile_removes_the_whole_assembler(tile):
    w = _world()
    _asm(w, 0, 3)
    x, y = tile
    assert _place(w, x, y, EMPTY, NONE) is None
    assert not (w[ENT] == ASM).any() and not w[ITEMS].any()


def test_an_edit_removes_only_the_touched_one_of_two_adjacent_assemblers():
    w = _world()
    _asm(w, 0, 3)
    _asm(w, 3, 3)
    assert _place(w, 3, 5, BELT) is None
    assert (w[ENT, 0:3, 3:6] == ASM).all()
    assert w[ENT, 3, 5] == BELT and (w[ENT] == ASM).sum() == 9


@pytest.mark.parametrize("direction", [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST])
def test_splitter_fills_both_tiles(direction):
    w = _world()
    assert _place(w, 2, 2, SPLITTER, direction.value) is None
    assert (w[ENT] == SPLITTER).sum() == 2
