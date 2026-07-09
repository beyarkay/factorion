"""A generated sink must be a pure consumer — it must not feed back into
the factory.

FAILING TEST documenting a lesson-generator bug found by the Factorio
parity harness (issue #261). `build_factory` can emit a degenerate factory
whose sink is placed facing an occupied tile — e.g. MOVE_ONE_ITEM seed 99
puts the sink directly above the source, facing north *into* it, forming a
source→…→sink→source loop.

The Factorion engine models a sink as an infinite consumer, so it scores
such a factory as a normal 15/s belt. But the layout is invalid: in real
Factorio a belt on the sink tile would output its items back into the
factory (the parity harness measures an impossible 135/s there). The
generator should never produce it.

This test scans the generators and asserts no sink's output tile is
occupied. It FAILS until `build_factory` rejects (or repairs) sink-loop
layouts.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorion import (  # noqa: E402
    Channel,
    Direction,
    DIR_TO_DELTA,
    LessonKind,
    build_factory,
    render_factory,
    str2ent,
)

SIZE = 11
SEEDS = range(300)


def _sink_loop_violations(kind, seed):
    """Return [(sink_xy, output_xy, occupying_entity_id), ...] for any sink
    whose output tile (the tile it faces) is occupied — i.e. it is not a
    pure consumer."""
    factory = build_factory(SIZE, kind, seed=seed)
    if factory is None:
        return []
    world = np.asarray(factory.world_CWH)
    ent = world[Channel.ENTITIES.value]
    dir_ch = world[Channel.DIRECTION.value]
    snk = str2ent("sink").value
    w, h = ent.shape
    out = []
    for x in range(w):
        for y in range(h):
            if int(ent[x, y]) != snk:
                continue
            d = int(dir_ch[x, y])
            if d == 0:
                continue
            dx, dy = DIR_TO_DELTA[Direction(d)]
            ox, oy = x + dx, y + dy
            if 0 <= ox < w and 0 <= oy < h and int(ent[ox, oy]) != 0:
                out.append(((x, y), (ox, oy), int(ent[ox, oy])))
    return out


def test_generated_sinks_are_pure_consumers():
    violations = []
    for kind in LessonKind:
        for seed in SEEDS:
            for sink_xy, out_xy, ent_id in _sink_loop_violations(kind, seed):
                violations.append((kind.name, seed, sink_xy, out_xy, ent_id))

    if violations:
        # Show the first offending factory in full for the fixer.
        kind0, seed0, *_ = violations[0]
        factory = build_factory(SIZE, LessonKind[kind0], seed=seed0)
        grid = render_factory(factory) if factory is not None else "(None)"
        lines = [
            f"{len(violations)} generated factory/-ies have a sink that feeds "
            "back into the factory (sink is not a pure consumer):",
        ]
        for kind, seed, sink_xy, out_xy, ent_id in violations[:20]:
            lines.append(
                f"  {kind} seed={seed}: sink {sink_xy} outputs into {out_xy} "
                f"(entity id {ent_id})"
            )
        lines.append(f"\nfirst offender ({kind0} seed={seed0}):\n{grid}")
        pytest.fail("\n".join(lines))
