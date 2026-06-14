"""MOVE_ONE_ITEM factories must actually carry throughput.

`build_factory` chose a random belt path via `random.choice(paths)` WITHOUT
validating it. `find_belt_paths_with_source_sink_orient` can return
geometrically-connected but functionally-broken paths when the random source/sink
placement is degenerate (too close, adjacent, or facing off-grid), so the
generator emitted ~5-13% (size-dependent) MOVE_ONE_ITEM factories that simulate to
ZERO throughput. Those are broken training targets — no policy can complete them.

The generator must reject-and-resample any factory whose throughput is zero. This
is the regression test for that fix.
"""

import pytest

from helpers import (
    LessonKind,
    blank_entities,
    build_factory,
    rs_throughput,
)


@pytest.mark.parametrize("size", [5, 8, 11, 12])
def test_move_one_item_every_factory_has_throughput(size):
    """Across many seeds, EVERY generated MOVE_ONE_ITEM factory must carry
    positive throughput. Before the fix ~5-13% (size-dependent) were zero."""
    broken = []
    n_built = 0
    for seed in range(120):
        factory = build_factory(size=size, kind=LessonKind.MOVE_ONE_ITEM, seed=seed)
        if factory is None:
            continue
        n_built += 1
        world, _ = blank_entities(factory, num_missing_entities=0)
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        if tp <= 0:
            broken.append(seed)
    assert n_built > 0, f"size={size}: build_factory never returned a factory"
    assert not broken, (
        f"size={size}: build_factory emitted {len(broken)}/{n_built} "
        f"MOVE_ONE_ITEM factories with zero throughput (seeds {broken}); the "
        f"generator must reject-and-resample broken factories"
    )


@pytest.mark.parametrize("seed", range(40))
def test_move_one_item_size11_seed_has_throughput(seed):
    """Per-seed granularity at size 11 (matches the SPLITTER_SPLIT test style)."""
    factory = build_factory(size=11, kind=LessonKind.MOVE_ONE_ITEM, seed=seed)
    assert factory is not None
    world, _ = blank_entities(factory, num_missing_entities=0)
    tp, _ = rs_throughput(world.permute(1, 2, 0))
    assert tp > 0, f"seed={seed}: MOVE_ONE_ITEM factory has zero throughput"
