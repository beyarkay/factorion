"""Tests for the ASCII grid renderer exposed as factorion_rs.render_factory
(and the factorion.render_factory wrapper)."""

import pytest

from helpers import (
    Channel,
    LessonKind,
    build_factory,
    entities,
    render_factory,
)

# Entity value → grid character (mirrors ENTITY_CHARS in factorion_rs/src/render.rs).
ENTITY_CHAR = {
    "transport_belt": "b",
    "inserter": "i",
    "assembling_machine_1": "a",
    "stone_furnace": "f",
    "splitter": "Y",
    "underground_belt": "d",  # down; up renders as 'u' (handled via misc below)
    "stack_inserter": "S",  # source
    "bulk_inserter": "K",  # sink
}
DIR_CHAR = {0: ".", 1: "^", 2: ">", 3: "v", 4: "<"}


def _grid_rows(world_CWH):
    return render_factory(world_CWH).split("\n")


class TestRenderFactoryShape:
    @pytest.mark.parametrize("size", [5, 8, 10])
    def test_row_and_column_count(self, size):
        f = build_factory(size=size, kind=LessonKind.ASSEMBLE_1_INGREDIENT, seed=3)
        assert f is not None
        rows = _grid_rows(f)
        assert len(rows) == size, f"expected {size} rows, got {len(rows)}"
        # Each non-trailing-trimmed row encodes `size` two-char tiles separated
        # by one filler char, so full width is 3*size - 1.
        for r in rows:
            assert len(r) <= 3 * size - 1

    def test_accepts_factory_or_tensor(self):
        f = build_factory(size=8, kind=LessonKind.ASSEMBLE_1_INGREDIENT, seed=1)
        assert f is not None
        from_factory = render_factory(f)
        from_tensor = render_factory(f.world_CWH)
        assert from_factory == from_tensor


class TestRenderFactoryContent:
    @pytest.mark.parametrize("seed", range(20))
    def test_single_tile_entities_render_correctly(self, seed):
        """Every single-tile entity (source/sink/belt/inserter) renders as its
        registry char + direction marker at its (x, y) position."""
        f = build_factory(size=10, kind=LessonKind.ASSEMBLE_1_INGREDIENT, seed=seed)
        assert f is not None
        world = f.world_CWH
        ent = world[Channel.ENTITIES.value]
        dirs = world[Channel.DIRECTION.value]
        rows = _grid_rows(world)
        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                v = ent[x, y].item()
                if v == 0:
                    continue
                name = entities[v].name
                # Multi-tile entities (assembler, furnace) have their own
                # box rendering.
                if name not in ENTITY_CHAR or name in (
                    "assembling_machine_1",
                    "stone_furnace",
                ):
                    continue
                c0 = rows[y][3 * x]
                c1 = rows[y][3 * x + 1]
                assert c0 == ENTITY_CHAR[name], (
                    f"seed={seed}: tile ({x},{y}) entity {name} rendered "
                    f"as {c0!r}, expected {ENTITY_CHAR[name]!r}"
                )
                assert c1 == DIR_CHAR[dirs[x, y].item()], (
                    f"seed={seed}: tile ({x},{y}) dir rendered as {c1!r}"
                )

    @pytest.mark.parametrize("seed", range(10))
    def test_assembler_box_present(self, seed):
        """The 3×3 assembler renders as a bordered box: a contiguous 3×3 of 'a'
        body characters with a blank interior centre."""
        f = build_factory(size=10, kind=LessonKind.ASSEMBLE_1_INGREDIENT, seed=seed)
        assert f is not None
        world = f.world_CWH
        ent = world[Channel.ENTITIES.value]
        asm_val = next(v for v, e in entities.items() if e.name == "assembling_machine_1")
        asm = [
            (x, y)
            for x in range(world.shape[1])
            for y in range(world.shape[2])
            if ent[x, y].item() == asm_val
        ]
        ax = min(x for x, _ in asm)
        ay = min(y for _, y in asm)
        rows = _grid_rows(world)
        # Corner of the box is 'a'; the centre interior tile is blank.
        assert rows[ay][3 * ax] == "a"
        assert rows[ay + 1][3 * (ax + 1)] == " "

    def test_empty_tiles_render_as_dots(self):
        f = build_factory(size=10, kind=LessonKind.ASSEMBLE_1_INGREDIENT, seed=0)
        assert f is not None
        world = f.world_CWH
        ent = world[Channel.ENTITIES.value]
        rows = _grid_rows(world)
        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                if ent[x, y].item() == 0:
                    assert rows[y][3 * x] == "." and rows[y][3 * x + 1] == ".", (
                        f"empty tile ({x},{y}) did not render as '..'"
                    )


class TestRenderFactoryDeterminism:
    @pytest.mark.parametrize("kind", list(LessonKind))
    def test_same_factory_same_render(self, kind):
        f1 = build_factory(size=11, kind=kind, seed=5)
        f2 = build_factory(size=11, kind=kind, seed=5)
        if f1 is None or f2 is None:
            pytest.skip(f"{kind.name} did not build at this size/seed")
        assert render_factory(f1) == render_factory(f2)
