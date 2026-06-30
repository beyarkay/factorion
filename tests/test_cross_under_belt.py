"""Python-side integration tests for the CROSS_UNDER_BELT lesson.

The lesson and its layout invariants (the obstruction cut, the underground
crossing, no orphan tiles, …) are generated and verified in Rust — see
``factory_gen.rs::tests``. What lives here is only what touches *Python*: that
``blank_entities`` honours the lesson's ``protected_positions`` contract — the
obstruction survives a full blank while the crossing's belts are removable.
"""

import pytest

from factorion import (
    Channel,
    LessonKind,
    blank_entities,
    build_factory,
    str2ent,
)
from helpers import rs_throughput


def _build(seed, size=12):
    factory = build_factory(size=size, kind=LessonKind.CROSS_UNDER_BELT, seed=seed)
    assert factory is not None, f"seed={seed} failed to build"
    return factory


@pytest.mark.parametrize("size", [5, 6, 8, 10, 12])
def test_blank_entities_shape_and_throughput(size):
    # The solved factory (nothing missing) round-trips through blank_entities.
    world, min_ent = blank_entities(_build(7, size=size), num_missing_entities=0)
    assert world.shape == (len(Channel), size, size)
    assert min_ent is not None
    tp, _ = rs_throughput(world.permute(1, 2, 0))
    assert tp > 0


@pytest.mark.parametrize("num_missing", [1, 2, 5, float("inf")])
@pytest.mark.parametrize("seed", range(10))
def test_sources_and_sinks_survive_blanking(num_missing, seed):
    world, _ = blank_entities(_build(seed, size=8), num_missing_entities=num_missing)
    ent = world[Channel.ENTITIES.value]
    assert (ent == str2ent("source").value).sum().item() == 2
    assert (ent == str2ent("sink").value).sum().item() == 2


@pytest.mark.parametrize("seed", range(20))
def test_obstruction_survives_full_blank(seed):
    f = _build(seed, size=8)
    solved = f.world_CWH
    world, _ = blank_entities(f, num_missing_entities=float("inf"))
    ent = world[Channel.ENTITIES.value]
    # Every protected (obstruction) tile keeps its entity...
    for x, y in f.protected_positions:
        assert int(ent[x, y].item()) == int(
            solved[Channel.ENTITIES.value, x, y].item()
        )
    # ...while the crossing's underground belts are removable (not protected).
    assert (ent == str2ent("underground_belt").value).sum().item() == 0
    tp, _ = rs_throughput(world.permute(1, 2, 0))
    assert tp > 0
