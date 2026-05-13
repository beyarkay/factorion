"""Tests for the FROM_BLUEPRINT lesson kind and its augmentations.

FROM_BLUEPRINT samples a hand-authored blueprint from
``lesson_blueprints/``, decodes it via :func:`blueprint2world`, then
augments it by (a) substituting the recipe if the BP is a gears
factory, (b) optionally flipping horizontally/vertically, and (c)
translating to a random offset within the size×size world. The tests
here verify each augmentation in isolation and the end-to-end lesson
contract.
"""

import random

import numpy as np
import pytest
import torch

from factorion import (
    Channel,
    Direction,
    LessonKind,
    Misc,
    _count_removable_entity_units,
    _flip_world,
    _is_gears_factory,
    _substitute_gears_recipe,
    blank_entities,
    build_factory,
    new_world,
    str2ent,
    str2item,
)


# --- _flip_world -----------------------------------------------------------


def _make_test_world(W, H):
    """Build a tiny world with one entity of each major kind so flip
    tests can observe direction remaps from a controlled starting
    state."""
    w = torch.tensor(new_world(width=W, height=H)).permute(2, 0, 1)
    return w


def test_flip_horizontal_swaps_east_west():
    w = _make_test_world(5, 5)
    w[Channel.ENTITIES.value, 1, 2] = str2ent("transport_belt").value
    w[Channel.DIRECTION.value, 1, 2] = Direction.EAST.value

    flipped = _flip_world(w, axis=1)
    # Tile (1, 2) maps to (W-1-1, 2) = (3, 2).
    assert int(flipped[Channel.ENTITIES.value, 3, 2]) == (
        str2ent("transport_belt").value
    )
    assert int(flipped[Channel.DIRECTION.value, 3, 2]) == Direction.WEST.value


def test_flip_horizontal_preserves_north_south():
    w = _make_test_world(5, 5)
    w[Channel.ENTITIES.value, 1, 2] = str2ent("transport_belt").value
    w[Channel.DIRECTION.value, 1, 2] = Direction.NORTH.value

    flipped = _flip_world(w, axis=1)
    assert int(flipped[Channel.DIRECTION.value, 3, 2]) == (
        Direction.NORTH.value
    )


def test_flip_vertical_swaps_north_south():
    w = _make_test_world(5, 5)
    w[Channel.ENTITIES.value, 2, 1] = str2ent("transport_belt").value
    w[Channel.DIRECTION.value, 2, 1] = Direction.NORTH.value

    flipped = _flip_world(w, axis=2)
    # Tile (2, 1) maps to (2, H-1-1) = (2, 3).
    assert int(flipped[Channel.ENTITIES.value, 2, 3]) == (
        str2ent("transport_belt").value
    )
    assert int(flipped[Channel.DIRECTION.value, 2, 3]) == (
        Direction.SOUTH.value
    )


def test_flip_double_is_identity():
    """Flipping twice along the same axis restores the original."""
    w = _make_test_world(5, 5)
    w[Channel.ENTITIES.value, 1, 2] = str2ent("transport_belt").value
    w[Channel.DIRECTION.value, 1, 2] = Direction.EAST.value

    once = _flip_world(w, axis=1)
    twice = _flip_world(once, axis=1)
    assert torch.equal(twice, w)


def test_flip_underground_misc_unchanged():
    """Flipping does not swap UNDERGROUND_DOWN ↔ UNDERGROUND_UP; the
    descending end stays descending after a mirror, only its direction
    changes."""
    w = _make_test_world(5, 5)
    w[Channel.ENTITIES.value, 1, 2] = str2ent("underground_belt").value
    w[Channel.DIRECTION.value, 1, 2] = Direction.EAST.value
    w[Channel.MISC.value, 1, 2] = Misc.UNDERGROUND_DOWN.value

    flipped = _flip_world(w, axis=1)
    assert int(flipped[Channel.MISC.value, 3, 2]) == (
        Misc.UNDERGROUND_DOWN.value
    )


# --- recipe substitution ---------------------------------------------------


def _build_gears_world(W=5, H=5):
    """Synthesize a tiny gears factory: source@iron_plate, sink@gear,
    3x3 assembler with iron_gear_wheel recipe. Enough to satisfy
    :func:`_is_gears_factory`."""
    w = _make_test_world(W, H)
    plate = str2item("iron_plate").value
    gear = str2item("iron_gear_wheel").value
    asm = str2ent("assembling_machine_1").value

    w[Channel.ENTITIES.value, 0, 2] = str2ent("stack_inserter").value
    w[Channel.DIRECTION.value, 0, 2] = Direction.EAST.value
    w[Channel.ITEMS.value, 0, 2] = plate

    w[Channel.ENTITIES.value, 4, 2] = str2ent("bulk_inserter").value
    w[Channel.DIRECTION.value, 4, 2] = Direction.EAST.value
    w[Channel.ITEMS.value, 4, 2] = gear

    for dx in range(3):
        for dy in range(3):
            w[Channel.ENTITIES.value, 1 + dx, 1 + dy] = asm
            w[Channel.ITEMS.value, 1 + dx, 1 + dy] = gear
    return w


def test_is_gears_factory_recognizes_canonical_layout():
    assert _is_gears_factory(_build_gears_world()) is True


def test_is_gears_factory_rejects_non_gears_recipe():
    w = _build_gears_world()
    cable = str2item("copper_cable").value
    asm_mask = (
        w[Channel.ENTITIES.value] == str2ent("assembling_machine_1").value
    )
    w[Channel.ITEMS.value][asm_mask] = cable
    assert _is_gears_factory(w) is False


def test_substitute_gears_recipe_remaps_items_consistently():
    """After substitution, the (source, sink, assembler) ITEMS values
    should all reflect one specific 1-in-1-out recipe — sink+assembler
    share the produced-item ID, source has the consumed-item ID."""
    random.seed(42)
    w = _build_gears_world()
    out = _substitute_gears_recipe(w)

    src_id = str2ent("stack_inserter").value
    snk_id = str2ent("bulk_inserter").value
    asm_id = str2ent("assembling_machine_1").value
    ent = out[Channel.ENTITIES.value]
    itm = out[Channel.ITEMS.value]

    src_items = set(int(v) for v in itm[ent == src_id].tolist())
    snk_items = set(int(v) for v in itm[ent == snk_id].tolist())
    asm_items = set(int(v) for v in itm[ent == asm_id].tolist())

    assert len(src_items) == 1
    assert len(snk_items) == 1
    # sink and every assembler tile carry the same produced-item ID.
    assert snk_items == asm_items


def test_substitute_gears_recipe_does_not_mutate_input():
    w = _build_gears_world()
    snapshot = w.clone()
    _substitute_gears_recipe(w)
    assert torch.equal(w, snapshot)


# --- _count_removable_entity_units -----------------------------------------


def test_count_excludes_source_sink_and_empty():
    w = _make_test_world(5, 5)
    w[Channel.ENTITIES.value, 0, 0] = str2ent("stack_inserter").value
    w[Channel.ENTITIES.value, 4, 4] = str2ent("bulk_inserter").value
    w[Channel.ENTITIES.value, 2, 2] = str2ent("transport_belt").value
    assert _count_removable_entity_units(w) == 1


def test_count_treats_multitile_entity_as_one_unit():
    """A 3×3 assembler occupies 9 tiles but counts as one removable
    unit (matching `_remove_entities`)."""
    w = _make_test_world(5, 5)
    asm = str2ent("assembling_machine_1").value
    for dx in range(3):
        for dy in range(3):
            w[Channel.ENTITIES.value, 1 + dx, 1 + dy] = asm
    assert _count_removable_entity_units(w) == 1


# --- end-to-end FROM_BLUEPRINT lesson --------------------------------------


def test_from_blueprint_returns_factory_with_correct_shape():
    f = build_factory(size=15, kind=LessonKind.FROM_BLUEPRINT, seed=1)
    assert f is not None
    assert f.world_CWH.shape == (5, 15, 15)
    assert f.total_entities > 0


def test_from_blueprint_returns_none_when_no_blueprint_fits():
    """`size=3` is smaller than every fixture's bounding box, so no
    blueprint can be placed; the function must give up and return
    None rather than raise."""
    f = build_factory(size=3, kind=LessonKind.FROM_BLUEPRINT, seed=1)
    assert f is None


def test_from_blueprint_determinism_same_seed_same_factory():
    f1 = build_factory(size=15, kind=LessonKind.FROM_BLUEPRINT, seed=42)
    f2 = build_factory(size=15, kind=LessonKind.FROM_BLUEPRINT, seed=42)
    assert f1 is not None and f2 is not None
    assert torch.equal(f1.world_CWH, f2.world_CWH)
    assert f1.total_entities == f2.total_entities


def test_from_blueprint_different_seeds_can_produce_different_factories():
    """Sampling + augmentation diversity should yield distinct
    factories across seeds. With 6 fixtures and 4 random binary
    augmentation knobs, a few seed pairs sampled here are nearly
    certain to differ."""
    seeds = [1, 2, 3, 5, 7, 11, 13, 17]
    factories = [
        build_factory(size=15, kind=LessonKind.FROM_BLUEPRINT, seed=s)
        for s in seeds
    ]
    assert all(f is not None for f in factories)
    # At least two factories differ — be lenient about which.
    distinct = any(
        not torch.equal(factories[i].world_CWH, factories[j].world_CWH)
        for i in range(len(factories))
        for j in range(i + 1, len(factories))
    )
    assert distinct, "all seeds produced identical factories"


def test_from_blueprint_blanking_round_trip():
    """A FROM_BLUEPRINT factory must be blankable end-to-end."""
    f = build_factory(size=15, kind=LessonKind.FROM_BLUEPRINT, seed=3)
    assert f is not None
    partial, removed = blank_entities(f, num_missing_entities=5, seed=3)
    assert partial.shape == f.world_CWH.shape
    assert 0 < removed <= 5


def test_from_blueprint_world_always_has_source_and_sink():
    """Every fixture has at least one source and one sink, so any
    FROM_BLUEPRINT-generated world must too."""
    for seed in range(1, 8):
        f = build_factory(size=15, kind=LessonKind.FROM_BLUEPRINT, seed=seed)
        assert f is not None
        ent = f.world_CWH[Channel.ENTITIES.value]
        n_src = int((ent == str2ent("stack_inserter").value).sum())
        n_snk = int((ent == str2ent("bulk_inserter").value).sum())
        assert n_src >= 1, f"seed={seed}: no source in factory"
        assert n_snk >= 1, f"seed={seed}: no sink in factory"


def test_from_blueprint_substitutes_non_gears_recipe_sometimes():
    """Across many seeds, the recipe substitution should eventually
    pick a non-iron-gear-wheel recipe and surface it in the factory's
    ITEMS channel."""
    gear_id = str2item("iron_gear_wheel").value
    plate_id = str2item("iron_plate").value
    asm_id = str2ent("assembling_machine_1").value
    seen_non_gear = False
    for seed in range(1, 100):
        f = build_factory(size=15, kind=LessonKind.FROM_BLUEPRINT, seed=seed)
        if f is None:
            continue
        ent = f.world_CWH[Channel.ENTITIES.value]
        itm = f.world_CWH[Channel.ITEMS.value]
        asm_mask = ent == asm_id
        if not asm_mask.any():
            continue
        asm_items = set(int(v) for v in itm[asm_mask].tolist())
        if asm_items and gear_id not in asm_items and plate_id not in asm_items:
            seen_non_gear = True
            break
    assert seen_non_gear, "100 seeds never produced a non-gears recipe"
