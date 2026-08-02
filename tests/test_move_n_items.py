"""Python-side integration tests for the MOVE_N_ITEMS lessons.

The layout invariants (n lines, distinct items, full-belt delivery on every
line, the cheapest routing order) are generated and verified in Rust — see
``factory_gen.rs::tests``. What lives here is only what touches *Python*: that
a full blank strips every belt back to the bare markers, which is what makes
the lesson a multi-route rebuild rather than a fill-in-the-gap.
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

MOVE_N_KINDS = [(LessonKind[f"MOVE_{n}_ITEMS"], n) for n in range(1, 6)]


def _build(kind, seed, size=11):
    factory = build_factory(size=size, kind=kind, seed=seed)
    assert factory is not None, f"{kind.name} seed={seed} failed to build"
    return factory


@pytest.mark.parametrize("kind,n", MOVE_N_KINDS)
def test_solved_factory_round_trips(kind, n):
    factory = _build(kind, 7)
    world, _ = blank_entities(factory, num_missing_entities=0)
    assert world.shape == (len(Channel), 11, 11)
    tp, _ = rs_throughput(world.permute(1, 2, 0))
    assert tp == pytest.approx(factory.max_throughput)


@pytest.mark.parametrize("kind,n", MOVE_N_KINDS)
@pytest.mark.parametrize("num_missing", [1, 3, float("inf")])
def test_markers_survive_blanking(kind, n, num_missing):
    world, _ = blank_entities(
        _build(kind, 3), num_missing_entities=num_missing
    )
    ent = world[Channel.ENTITIES.value]
    assert (ent == str2ent("source").value).sum().item() == n
    assert (ent == str2ent("sink").value).sum().item() == n


@pytest.mark.parametrize("kind,n", MOVE_N_KINDS)
@pytest.mark.parametrize("seed", range(5))
def test_full_blank_leaves_only_the_markers(kind, n, seed):
    factory = _build(kind, seed)
    world, removed = blank_entities(factory, num_missing_entities=float("inf"))
    assert not factory.protected_positions
    assert removed == factory.total_entities
    ent = world[Channel.ENTITIES.value]
    nonempty = int((ent != 0).sum().item())
    assert nonempty == 2 * n, "a full blank must leave the markers and nothing else"
    # With every belt gone, nothing is delivered — the policy rebuilds from here.
    tp, _ = rs_throughput(world.permute(1, 2, 0))
    assert tp == 0


@pytest.mark.parametrize("kind,n", MOVE_N_KINDS)
def test_each_line_carries_its_own_item(kind, n):
    factory = _build(kind, 11)
    ent = factory.world_CWH[Channel.ENTITIES.value]
    items = factory.world_CWH[Channel.ITEMS.value]
    sources = items[ent == str2ent("source").value].tolist()
    sinks = items[ent == str2ent("sink").value].tolist()
    assert len(sources) == n
    assert sorted(sources) == sorted(sinks)
    assert len(set(sources)) == n
