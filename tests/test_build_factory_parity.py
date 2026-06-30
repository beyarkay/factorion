"""Parity tests: the Rust port of ``build_factory`` must produce a
byte-identical factory to the Python implementation for the same
``(size, kind, seed)``.

After the single-RNG refactor, Python's ``build_factory`` draws every bit of
layout randomness from CPython's ``random`` module, and ``factorion_rs``
reimplements that generator exactly (``src/pyrandom.rs``) plus a line-for-line
port of the placement logic (``src/factory_gen.rs``). These tests fuzz
thousands of seeds and assert the two agree on every channel of every tile,
the removable-entity count, and the protected positions.

The headline goal is to speed up RL rollouts by building factories in Rust;
these tests are the safety net that the port is faithful before it is wired
into the hot path.
"""

import numpy as np
import pytest

import factorion_rs
from factorion import LessonKind, build_factory

# Lesson kinds whose Rust port is complete. Extend this as kinds land; the
# fuzz tests below run over exactly these, and ``test_unported_kinds_raise``
# asserts everything else still reports as not-yet-ported.
PORTED_KINDS = [
    LessonKind.MOVE_ONE_ITEM,
    LessonKind.MOVE_ONE_ITEM_CHAOS,
    LessonKind.SPLITTER_SPLIT,
    LessonKind.SPLITTER_MERGE,
    LessonKind.ASSEMBLE_1IN_1OUT,
    LessonKind.MOVE_VIA_UG_BELT,
    LessonKind.ASSEMBLE_2IN_1OUT,
]

ALL_KINDS = list(LessonKind)


def _py_world_whc(factory):
    """Python ``Factory.world_CWH`` (C, W, H) → (W, H, C) int64 ndarray,
    matching the shape the Rust port returns."""
    return factory.world_CWH.permute(1, 2, 0).to(int).numpy()


def assert_parity(size, kind, seed, *, random_item=True, max_entities=float("inf")):
    """Build the same factory in Python and Rust and assert they match
    exactly (or are both ``None``). Returns ``"built"`` / ``"none"`` so
    callers can assert coverage (that fuzzing actually exercised the
    success path, not only rejections)."""
    py = build_factory(
        size=size,
        kind=kind,
        seed=seed,
        random_item=random_item,
        max_entities=max_entities,
    )
    rs = factorion_rs.build_factory(
        size, kind.value, seed, random_item, max_entities
    )

    if py is None:
        assert rs is None, (
            f"kind={kind.name} seed={seed} size={size}: "
            f"Python returned None but Rust built a factory"
        )
        return "none"

    assert rs is not None, (
        f"kind={kind.name} seed={seed} size={size}: "
        f"Python built a factory but Rust returned None"
    )
    rs_world, rs_total, rs_protected = rs
    py_world = _py_world_whc(py)

    if not np.array_equal(py_world, rs_world):
        diffs = np.argwhere(py_world != rs_world)
        first = diffs[:8].tolist()
        raise AssertionError(
            f"kind={kind.name} seed={seed} size={size}: world mismatch at "
            f"(x,y,c)={first} (py={[int(py_world[tuple(d)]) for d in first]} "
            f"rs={[int(rs_world[tuple(d)]) for d in first]})"
        )
    assert rs_total == py.total_entities, (
        f"kind={kind.name} seed={seed} size={size}: total_entities "
        f"{rs_total} != {py.total_entities}"
    )
    assert set(map(tuple, rs_protected)) == set(py.protected_positions), (
        f"kind={kind.name} seed={seed} size={size}: protected positions "
        f"{sorted(map(tuple, rs_protected))} != {sorted(py.protected_positions)}"
    )
    return "built"


# ── MOVE_ONE_ITEM ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [5, 8, 10, 12, 15])
def test_move_one_item_fuzz_sizes(size):
    """Across 400 seeds per grid size, every Rust factory equals Python's,
    and the success path is actually exercised."""
    built = 0
    for seed in range(400):
        if assert_parity(size, LessonKind.MOVE_ONE_ITEM, seed) == "built":
            built += 1
    assert built > 300, f"size={size}: only {built}/400 seeds built a factory"


def test_move_one_item_fuzz_many_seeds():
    """A deep sweep on a single size to stress the RNG/path enumeration."""
    for seed in range(3000):
        assert_parity(10, LessonKind.MOVE_ONE_ITEM, seed)


def test_move_one_item_fixed_recipe():
    """``random_item=False`` pins the electronic-circuit item; parity must
    hold on that branch too."""
    for seed in range(500):
        assert_parity(10, LessonKind.MOVE_ONE_ITEM, seed, random_item=False)


@pytest.mark.parametrize("max_entities", [3.0, 5.0, 8.0])
def test_move_one_item_max_entities(max_entities):
    """The ``max_entities`` path filter changes which paths survive (and can
    push factories to ``None``); parity must track that exactly."""
    for seed in range(500):
        assert_parity(
            10, LessonKind.MOVE_ONE_ITEM, seed, max_entities=max_entities
        )


# ── MOVE_ONE_ITEM_CHAOS ──────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [5, 8, 10, 12, 15])
def test_move_one_item_chaos_fuzz_sizes(size):
    """The chaos variant routes through a random waypoint and protects the
    source→waypoint stub; parity must hold on the world, the count, and the
    protected positions."""
    built = 0
    for seed in range(400):
        if assert_parity(size, LessonKind.MOVE_ONE_ITEM_CHAOS, seed) == "built":
            built += 1
    assert built > 250, f"size={size}: only {built}/400 seeds built"


def test_move_one_item_chaos_fuzz_many_seeds():
    for seed in range(3000):
        assert_parity(10, LessonKind.MOVE_ONE_ITEM_CHAOS, seed)


def test_move_one_item_chaos_fixed_recipe_and_caps():
    for seed in range(500):
        assert_parity(
            10, LessonKind.MOVE_ONE_ITEM_CHAOS, seed, random_item=False
        )
    for seed in range(500):
        assert_parity(
            10, LessonKind.MOVE_ONE_ITEM_CHAOS, seed, max_entities=6.0
        )


# ── SPLITTER_SPLIT ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [5, 8, 10, 12, 15])
def test_splitter_split_fuzz_sizes(size):
    """1 source → splitter → 2 sinks. Exercises the splitter geometry,
    random.sample over the free cells (both pool and set strategies across
    sizes), the dual sink-assignment search, and the protected splitter
    tiles."""
    built = 0
    for seed in range(400):
        if assert_parity(size, LessonKind.SPLITTER_SPLIT, seed) == "built":
            built += 1
    assert built > 150, f"size={size}: only {built}/400 seeds built"


def test_splitter_split_fuzz_many_seeds():
    for seed in range(3000):
        assert_parity(10, LessonKind.SPLITTER_SPLIT, seed)


def test_splitter_split_fixed_recipe_and_caps():
    for seed in range(500):
        assert_parity(10, LessonKind.SPLITTER_SPLIT, seed, random_item=False)
    for seed in range(500):
        assert_parity(10, LessonKind.SPLITTER_SPLIT, seed, max_entities=8.0)


# ── SPLITTER_MERGE ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [5, 8, 10, 12, 15])
def test_splitter_merge_fuzz_sizes(size):
    """2 sources → splitter → 1 sink. Same splitter machinery as SPLIT but a
    different path topology (two inbound, one outbound with its own fallback)."""
    built = 0
    for seed in range(400):
        if assert_parity(size, LessonKind.SPLITTER_MERGE, seed) == "built":
            built += 1
    assert built > 150, f"size={size}: only {built}/400 seeds built"


def test_splitter_merge_fuzz_many_seeds():
    for seed in range(3000):
        assert_parity(10, LessonKind.SPLITTER_MERGE, seed)


def test_splitter_merge_fixed_recipe_and_caps():
    for seed in range(500):
        assert_parity(10, LessonKind.SPLITTER_MERGE, seed, random_item=False)
    for seed in range(500):
        assert_parity(10, LessonKind.SPLITTER_MERGE, seed, max_entities=8.0)


# ── ASSEMBLE_1IN_1OUT ────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [5, 8, 10, 12, 15])
def test_assemble_1in1out_fuzz_sizes(size):
    """source → inserter → 3×3 assembler (recipe) → inserter → sink. Exercises
    the recipe choice, assembler anchoring, perimeter-slot sampling, and the
    inserter/belt routing. The recipe item lands in the ITEMS channel of the
    assembler tiles — a path that pure-belt lessons never touch."""
    built = 0
    for seed in range(400):
        if assert_parity(size, LessonKind.ASSEMBLE_1IN_1OUT, seed) == "built":
            built += 1
    assert built > 150, f"size={size}: only {built}/400 seeds built"


def test_assemble_1in1out_fuzz_many_seeds():
    for seed in range(3000):
        assert_parity(10, LessonKind.ASSEMBLE_1IN_1OUT, seed)


def test_assemble_1in1out_caps():
    for seed in range(500):
        assert_parity(10, LessonKind.ASSEMBLE_1IN_1OUT, seed, max_entities=10.0)


# ── MOVE_VIA_UG_BELT ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("size", [5, 8, 10, 12, 15])
def test_move_via_ug_belt_fuzz_sizes(size):
    """A wall (1..4 thick) spans the grid; an underground-belt pair tunnels
    through it. Exercises the MISC channel (UG up/down), the FOOTPRINT wall
    barrier, and side-restricted belt routing — none of which the other
    lessons touch."""
    built = 0
    for seed in range(400):
        if assert_parity(size, LessonKind.MOVE_VIA_UG_BELT, seed) == "built":
            built += 1
    assert built > 150, f"size={size}: only {built}/400 seeds built"


def test_move_via_ug_belt_fuzz_many_seeds():
    for seed in range(3000):
        assert_parity(10, LessonKind.MOVE_VIA_UG_BELT, seed)


def test_move_via_ug_belt_fixed_recipe_and_caps():
    for seed in range(500):
        assert_parity(10, LessonKind.MOVE_VIA_UG_BELT, seed, random_item=False)
    for seed in range(500):
        assert_parity(10, LessonKind.MOVE_VIA_UG_BELT, seed, max_entities=8.0)


# ── ASSEMBLE_2IN_1OUT ────────────────────────────────────────────────────────


# 2-in-1-out is the largest lesson (3×3 assembler + 2 sources + sink + 3
# inserters + belts), so a 5×5 grid never fits it — both impls return None for
# every seed (parity still holds, but the success path isn't exercised). Use
# sizes that actually build so the "built > N" guard is meaningful.
@pytest.mark.parametrize("size", [8, 10, 12, 14, 16])
def test_assemble_2in1out_fuzz_sizes(size):
    """2 sources → 2 input inserters → 3×3 assembler → output inserter → sink.
    Exercises the 2-in-1-out recipe pool, the ingredient shuffle (which source
    gets which item), the 3-slot perimeter sample, and three belt paths."""
    built = 0
    for seed in range(400):
        if assert_parity(size, LessonKind.ASSEMBLE_2IN_1OUT, seed) == "built":
            built += 1
    assert built > 100, f"size={size}: only {built}/400 seeds built"


def test_assemble_2in1out_tight_grid_parity():
    """On a 5×5 grid the lesson never fits — assert that's a *parity* fact
    (both impls return None) rather than a silent gap."""
    for seed in range(200):
        assert assert_parity(5, LessonKind.ASSEMBLE_2IN_1OUT, seed) == "none"


def test_assemble_2in1out_fuzz_many_seeds():
    for seed in range(3000):
        assert_parity(10, LessonKind.ASSEMBLE_2IN_1OUT, seed)


def test_assemble_2in1out_caps():
    for seed in range(500):
        assert_parity(10, LessonKind.ASSEMBLE_2IN_1OUT, seed, max_entities=12.0)


# ── progress guard ───────────────────────────────────────────────────────────


def test_all_kinds_are_ported():
    """Every LessonKind is covered by the Rust port (and by PORTED_KINDS)."""
    assert set(PORTED_KINDS) == set(ALL_KINDS)


def test_every_kind_builds_via_rust():
    """Smoke: each kind builds at least one factory through the Rust path."""
    for kind in ALL_KINDS:
        built = any(
            factorion_rs.build_factory(12, kind.value, seed, True, float("inf"))
            is not None
            for seed in range(40)
        )
        assert built, f"{kind.name}: Rust built nothing in 40 seeds"


def test_unknown_kind_raises():
    with pytest.raises(ValueError):
        factorion_rs.build_factory(10, 999, 0, True, float("inf"))


def test_from_blueprint_kind_is_removed():
    """FROM_BLUEPRINT (the old kind 8) is no longer a LessonKind and the Rust
    engine rejects its value — the lesson was removed rather than ported (it
    would have required shipping a Factorio blueprint decoder)."""
    assert not hasattr(LessonKind, "FROM_BLUEPRINT")
    with pytest.raises(ValueError):
        factorion_rs.build_factory(10, 8, 0, True, float("inf"))
