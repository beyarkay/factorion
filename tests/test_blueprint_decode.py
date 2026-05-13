"""Decoder tests for `factorion.blueprint2world`.

Blueprint fixtures live as one-string-per-file under `tests/blueprints/`.
Every `*.txt` in that directory is auto-discovered by
`test_blueprint_fixture_decodes` and must decode without error into a
non-empty world tensor. To add a new blueprint to the test suite,
paste the b64 string into a new file under `tests/blueprints/`; pytest
will pick it up on the next run.

Fixtures that warrant stronger assertions (specific entity counts,
recipes, directions) get their own named test functions below, which
load the same fixture file by name.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from factorion import (
    Channel,
    Direction,
    blueprint2world,
    str2ent,
    str2item,
)

# Round-trip tests need the mod's tensor → blueprint encoder.
_SERVER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "factorion-mod", "server"
)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

_BP_DIR = Path(__file__).parent / "blueprints"


def _load_bp(name: str) -> str:
    """Read a blueprint fixture by stem (e.g. "gears_factory")."""
    return (_BP_DIR / f"{name}.txt").read_text().strip()


# --- Generic smoke check over every fixture --------------------------------


@pytest.mark.parametrize(
    "bp_path",
    sorted(_BP_DIR.glob("*.txt")),
    ids=lambda p: p.stem,
)
def test_blueprint_fixture_decodes(bp_path):
    """Every fixture in tests/blueprints/ must decode into a 5-channel,
    non-empty world tensor. New blueprints get this check for free —
    add specific assertions in a named test below if you want more."""
    bp = bp_path.read_text().strip()
    w = blueprint2world(bp)
    assert w.ndim == 3 and w.shape[0] == 5, (
        f"unexpected tensor shape for {bp_path.name}: {w.shape}"
    )
    assert w.shape[1] > 0 and w.shape[2] > 0
    assert int((w[Channel.ENTITIES.value] != 0).sum()) > 0, (
        f"{bp_path.name} decoded to an empty entity layer"
    )


# --- gears_factory: 8× iron-gear-wheel assemblers, the user's fixture ------


def test_decode_gears_factory_shape():
    w = blueprint2world(_load_bp("gears_factory"))
    assert w.shape == (5, 13, 15)
    assert w.dtype.kind == "i"


def test_decode_gears_factory_entity_counts():
    """8 3×3 assemblers fully expanded to 72 tiles; 3
    constant-combinators split into 2 sources + 1 sink."""
    w = blueprint2world(_load_bp("gears_factory"))
    ent = w[Channel.ENTITIES.value]

    counts = {
        "transport_belt": 45,
        "inserter": 16,
        "assembling_machine_1": 72,  # 8 × 9
        "stack_inserter": 2,
        "bulk_inserter": 1,
    }
    for name, expected in counts.items():
        got = int((ent == str2ent(name).value).sum())
        assert got == expected, f"{name}: expected {expected}, got {got}"


def test_decode_gears_factory_recipes_and_marker_items():
    """Every assembler tile carries `iron_gear_wheel`; sources carry
    `iron_plate`, sinks carry `iron_gear_wheel`."""
    w = blueprint2world(_load_bp("gears_factory"))
    ent = w[Channel.ENTITIES.value]
    itm = w[Channel.ITEMS.value]

    asm_id = str2ent("assembling_machine_1").value
    gear_id = str2item("iron_gear_wheel").value
    plate_id = str2item("iron_plate").value
    src_id = str2ent("stack_inserter").value
    snk_id = str2ent("bulk_inserter").value

    asm_mask = ent == asm_id
    assert asm_mask.sum() > 0
    assert np.all(itm[asm_mask] == gear_id)
    assert np.all(itm[ent == src_id] == plate_id)
    assert np.all(itm[ent == snk_id] == gear_id)


def test_decode_gears_factory_inserter_directions_are_pickup_pointing():
    """The decoder flips blueprint inserter direction (drop side) by
    +8 mod 16 to get the model's pickup-pointing direction. All 16
    inserters in this factory end up facing East or West."""
    w = blueprint2world(_load_bp("gears_factory"))
    ent = w[Channel.ENTITIES.value]
    dir_ch = w[Channel.DIRECTION.value]
    ins_id = str2ent("inserter").value
    dirs = dir_ch[ent == ins_id]
    assert set(dirs.tolist()).issubset(
        {Direction.EAST.value, Direction.WEST.value}
    )


# --- Edge cases ------------------------------------------------------------


def test_decode_rejects_blueprint_with_no_recognized_entities():
    """An empty blueprint must raise rather than silently produce a
    zero-size world."""
    from factorion import dict2b64

    empty_bp = dict2b64({"blueprint": {"entities": []}})
    with pytest.raises(ValueError, match="no recognized entities"):
        blueprint2world(empty_bp)


# --- Round-trip via the mod's encoder --------------------------------------


def _round_trip(world_CWH):
    """Encode a (C, W, H) world to a blueprint string, then decode it
    back. The encoder crops to non-empty cells, so the result is
    tightly bounded around the placed entities."""
    from blueprint import world_tensor_to_blueprint_string

    bp = world_tensor_to_blueprint_string(world_CWH)
    return blueprint2world(bp)


def test_round_trip_single_belt_north():
    from factorion import new_world

    w = new_world(width=3, height=3)
    tb = str2ent("transport_belt")
    w[1, 1, Channel.ENTITIES.value] = tb.value
    w[1, 1, Channel.DIRECTION.value] = Direction.NORTH.value
    decoded = _round_trip(np.transpose(w, (2, 0, 1)))

    assert decoded.shape == (5, 1, 1)
    assert int(decoded[Channel.ENTITIES.value, 0, 0]) == tb.value
    assert int(decoded[Channel.DIRECTION.value, 0, 0]) == Direction.NORTH.value


@pytest.mark.parametrize(
    "dir_",
    [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST],
)
def test_round_trip_inserter_direction_preserved(dir_):
    """Pickup-pointing inserter direction must survive the encoder's
    drop-side flip and the decoder's pickup-side flip."""
    from factorion import new_world

    w = new_world(width=3, height=3)
    ins = str2ent("inserter")
    w[1, 1, Channel.ENTITIES.value] = ins.value
    w[1, 1, Channel.DIRECTION.value] = dir_.value
    decoded = _round_trip(np.transpose(w, (2, 0, 1)))

    ys, xs = np.where(decoded[Channel.ENTITIES.value] == ins.value)
    assert len(ys) == 1
    assert int(decoded[Channel.DIRECTION.value, ys[0], xs[0]]) == dir_.value


def test_round_trip_source_marker_carries_item_and_direction():
    from factorion import new_world

    w = new_world(width=3, height=3)
    src = str2ent("stack_inserter")
    item_val = str2item("iron_plate").value
    w[1, 1, Channel.ENTITIES.value] = src.value
    w[1, 1, Channel.DIRECTION.value] = Direction.EAST.value
    w[1, 1, Channel.ITEMS.value] = item_val
    decoded = _round_trip(np.transpose(w, (2, 0, 1)))

    ys, xs = np.where(decoded[Channel.ENTITIES.value] == src.value)
    assert len(ys) == 1
    assert int(decoded[Channel.DIRECTION.value, ys[0], xs[0]]) == (
        Direction.EAST.value
    )
    assert int(decoded[Channel.ITEMS.value, ys[0], xs[0]]) == item_val


def test_round_trip_assembler_3x3_recipe_preserved():
    """A 3×3 assembler with a recipe must decode back to a 3×3
    footprint, every tile tagged with the recipe item."""
    from factorion import new_world

    w = new_world(width=5, height=5)
    asm = str2ent("assembling_machine_1")
    recipe_val = str2item("iron_gear_wheel").value
    for dx in range(3):
        for dy in range(3):
            w[1 + dx, 1 + dy, Channel.ENTITIES.value] = asm.value
            w[1 + dx, 1 + dy, Channel.ITEMS.value] = recipe_val
    decoded = _round_trip(np.transpose(w, (2, 0, 1)))

    asm_mask = decoded[Channel.ENTITIES.value] == asm.value
    assert int(asm_mask.sum()) == 9
    assert np.all(decoded[Channel.ITEMS.value][asm_mask] == recipe_val)
