"""The throughput sim must never panic on partial factories.

During a rollout the agent can place ``empty`` on a *single* tile — including
one tile of a multi-tile entity (a splitter or an assembler), leaving a partial
entity behind. ``simulate_throughput`` (the Rust engine) once panicked with an
index-out-of-bounds on exactly such a world: a splitter with its anchor tile
deleted, sitting at the grid edge, whose recomputed 2-wide footprint ran
off-grid.

These end-to-end tests drive the real pipeline — build every lesson's solved
factory, apply the agent's single-tile delete to each occupied tile, and
simulate — asserting it never raises. This guards the whole class of
partial/malformed worlds the policy can produce, not just the one layout in the
Rust regression test.
"""

import pytest

from helpers import (
    Channel,
    LessonKind,
    build_factory,
    items,
    rs_throughput,
)

# Channels the env clears when the agent deletes a tile (footprint is left).
_DELETE_CHANNELS = (Channel.ENTITIES, Channel.DIRECTION, Channel.ITEMS, Channel.MISC)

_SPLITTER_ID = next(v for v, it in items.items() if it.name == "splitter")


@pytest.mark.parametrize("kind", list(LessonKind))
def test_single_tile_delete_never_panics(kind):
    """Deleting any one tile of any lesson's solved factory must not panic.

    Multi-tile entities (splitters, assemblers) are the interesting case: a
    single-tile delete leaves a partial entity that the graph builder treats as
    a fresh anchor.
    """
    for seed in range(8):
        factory = build_factory(size=11, kind=kind, seed=seed)
        if factory is None:  # rejection sampling can fail for a given seed
            continue
        solved_CWH = factory.world_CWH
        ent_layer = solved_CWH[Channel.ENTITIES.value]
        for x, y in (ent_layer != 0).nonzero(as_tuple=False).tolist():
            partial = solved_CWH.clone()
            for ch in _DELETE_CHANNELS:
                partial[ch.value, x, y] = 0
            # Rust panics surface as pyo3_runtime.PanicException; any exception
            # here is a failure.
            rs_throughput(partial.permute(1, 2, 0))


def test_splitter_partial_at_edge_never_panics():
    """The exact crash class, reached through the real generator: a
    SPLITTER_SPLIT / SPLITTER_MERGE factory with a single splitter tile deleted.
    """
    for kind in (LessonKind.SPLITTER_SPLIT, LessonKind.SPLITTER_MERGE):
        for seed in range(16):
            factory = build_factory(size=11, kind=kind, seed=seed)
            if factory is None:
                continue
            solved_CWH = factory.world_CWH
            ent_layer = solved_CWH[Channel.ENTITIES.value]
            for x, y in (ent_layer == _SPLITTER_ID).nonzero(as_tuple=False).tolist():
                partial = solved_CWH.clone()
                for ch in _DELETE_CHANNELS:
                    partial[ch.value, x, y] = 0
                rs_throughput(partial.permute(1, 2, 0))
