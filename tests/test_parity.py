"""Parity tests between Python funge_throughput and Rust simulate_throughput.

Tests verify that the Rust implementation produces the same results as the
existing Python implementation for identical factory layouts.
"""

import random

import numpy as np
import pytest
import torch

import factorion_rs

from helpers import (
    Channel,
    Direction,
    LessonKind,
    Misc,
    compare_throughput,
    entities,
    generate_lesson,
    make_world,
    rs_throughput,
    py_throughput_safe,
    set_entity,
    set_splitter,
)


# ── Deterministic parity tests ───────────────────────────────────────────────


class TestEmptyWorld:
    def test_empty_small(self):
        """Empty world. Python crashes on this (count=0*0=0); verify Rust returns (0,0)."""
        world = make_world(3)
        rs_tp, rs_unreachable = rs_throughput(world)
        assert rs_tp == 0.0
        assert rs_unreachable == 0

    def test_empty_large(self):
        world = make_world(10)
        rs_tp, rs_unreachable = rs_throughput(world)
        assert rs_tp == 0.0
        assert rs_unreachable == 0


class TestSimpleBeltChains:
    def test_source_belt_sink(self):
        """Source -> Belt -> Sink (east)."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_source_belt_belt_sink(self):
        """Source -> Belt -> Belt -> Sink (east)."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_source_belt_chain_south(self):
        """Vertical belt chain going south."""
        world = make_world(5)
        set_entity(world, 2, 0, "stack_inserter", Direction.SOUTH, "iron_plate")
        set_entity(world, 2, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 2, 2, "transport_belt", Direction.SOUTH)
        set_entity(world, 2, 3, "bulk_inserter", Direction.SOUTH, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_source_belt_chain_north(self):
        """Vertical belt chain going north."""
        world = make_world(5)
        set_entity(world, 2, 4, "stack_inserter", Direction.NORTH, "iron_plate")
        set_entity(world, 2, 3, "transport_belt", Direction.NORTH)
        set_entity(world, 2, 2, "transport_belt", Direction.NORTH)
        set_entity(world, 2, 1, "bulk_inserter", Direction.NORTH, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_source_belt_chain_west(self):
        """Horizontal belt chain going west."""
        world = make_world(5)
        set_entity(world, 4, 2, "stack_inserter", Direction.WEST, "copper_plate")
        set_entity(world, 3, 2, "transport_belt", Direction.WEST)
        set_entity(world, 2, 2, "transport_belt", Direction.WEST)
        set_entity(world, 1, 2, "bulk_inserter", Direction.WEST, "copper_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)


class TestInserterChains:
    def test_source_inserter_belt_sink(self):
        """Source -> Inserter -> Belt -> Sink. Inserter is the bottleneck."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "inserter", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(0.86, abs=1e-6)


class TestUndergroundBelts:
    def test_underground_bypass(self):
        """Source -> Belt -> Underground(down) ... Underground(up) -> Belt -> Sink."""
        world = make_world(8)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(
            world, 2, 0, "underground_belt", Direction.EAST, misc=Misc.UNDERGROUND_DOWN.value
        )
        set_entity(
            world, 5, 0, "underground_belt", Direction.EAST, misc=Misc.UNDERGROUND_UP.value
        )
        set_entity(world, 6, 0, "transport_belt", Direction.EAST)
        set_entity(world, 7, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        compare_throughput(world)


class TestDisconnected:
    def test_source_and_sink_not_connected(self):
        """Source and sink exist but with no path between them."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 4, 4, "bulk_inserter", Direction.EAST, "copper_cable")
        compare_throughput(world)

    def test_lone_belt(self):
        """A single belt with no source or sink."""
        world = make_world(5)
        set_entity(world, 2, 2, "transport_belt", Direction.EAST)
        compare_throughput(world)

    def test_disconnected_belt_between_source_sink(self):
        """Source, sink, and an unrelated belt."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        # Disconnected belt
        set_entity(world, 4, 4, "transport_belt", Direction.NORTH)
        compare_throughput(world)


class TestOpposingBelts:
    def test_opposing_belts_no_connection(self):
        """Two belts facing each other should not connect."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.WEST)
        set_entity(world, 3, 0, "bulk_inserter", Direction.WEST, "copper_cable")
        compare_throughput(world)


class TestEdgeCases:
    def test_source_only(self):
        """Only a source, no sink. Python crashes on this; verify Rust returns (0, 1)."""
        world = make_world(3)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        rs_tp, rs_unreachable = rs_throughput(world)
        assert rs_tp == 0.0
        assert rs_unreachable == 1

    def test_sink_only(self):
        """Only a sink, no source."""
        world = make_world(3)
        set_entity(world, 0, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        compare_throughput(world)

    def test_source_directly_adjacent_to_sink(self):
        """Source drops directly onto sink. Python returns inf (bug); Rust should return 0."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        # Python funge_throughput asserts inf < inf here, which is a known bug.
        # Both should logically return 0 since there's no belt between them,
        # but the source-sink inserter connection produces inf throughput.
        # Verify Rust handles this gracefully.
        rs_tp, rs_unreachable = rs_throughput(world)
        # The Rust implementation should return 0 since lib.rs checks for inf
        assert rs_tp == 0.0


# ── Generated lesson parity tests ────────────────────────────────────────────


class TestGeneratedLessons:
    """Test parity on worlds generated by generate_lesson with fixed seeds."""

    @pytest.mark.parametrize("seed", range(20))
    def test_lesson_seed(self, seed):
        try:
            result = generate_lesson(
                size=5,
                kind=LessonKind.MOVE_ONE_ITEM,
                num_missing_entities=0,
                seed=seed,
            )
            world_CWH = result[0]
            world_WHC = world_CWH.permute(1, 2, 0).to(torch.int64)
            compare_throughput(world_WHC, tolerance=0.1)
        except (ValueError, RuntimeError) as e:
            pytest.skip(f"Seed {seed} failed to generate lesson: {e}")


# ── Fuzz testing ─────────────────────────────────────────────────────────────


class TestFuzz:
    """Fuzz test: generate random factory layouts and check parity."""

    # All 1x1 entity IDs. Multi-tile entities (assembler=3, splitter=7) are
    # excluded because random single-cell placement is invalid for them.
    ENTITY_VALUES = [eid for eid, e in entities.items() if e.width == 1 and e.height == 1]
    DIRECTION_VALUES = [0, 1, 2, 3, 4]
    ITEM_VALUES = [0, 1, 2, 3, 4]
    MISC_VALUES = [0, 1, 2]

    # Multi-tile entity specs: (entity_id, width, height)
    MULTI_TILE_ENTITIES = [
        (eid, e.width, e.height)
        for eid, e in entities.items()
        if e.width > 1 or e.height > 1
    ]

    @staticmethod
    def _random_world(size, rng, density=0.3, include_multi_tile=False):
        """Create a random world with given entity density."""
        world = torch.zeros((size, size, len(Channel)), dtype=torch.int64)
        world[:, :, Channel.FOOTPRINT.value] = 1
        occupied = set()
        for x in range(size):
            for y in range(size):
                if (x, y) in occupied:
                    continue
                if rng.random() < density:
                    # Occasionally place a multi-tile entity
                    if (
                        include_multi_tile
                        and TestFuzz.MULTI_TILE_ENTITIES
                        and rng.random() < 0.15
                    ):
                        eid, w, h = rng.choice(TestFuzz.MULTI_TILE_ENTITIES)
                        direction = rng.choice([1, 2, 3, 4])
                        tiles = factorion_rs.py_entity_tiles(x, y, direction, w, h)
                        if tiles is None:
                            continue
                        # Check all tiles in bounds and unoccupied
                        all_ok = all(
                            0 <= tx < size and 0 <= ty < size and (tx, ty) not in occupied
                            for tx, ty in tiles
                        )
                        if all_ok:
                            for tx, ty in tiles:
                                world[tx, ty, 0] = eid
                                world[tx, ty, 1] = direction
                                occupied.add((tx, ty))
                    else:
                        entity = rng.choice(TestFuzz.ENTITY_VALUES)
                        direction = rng.choice(TestFuzz.DIRECTION_VALUES)
                        item = rng.choice(TestFuzz.ITEM_VALUES)
                        misc = rng.choice(TestFuzz.MISC_VALUES)
                        world[x, y, 0] = entity
                        world[x, y, 1] = direction
                        world[x, y, 2] = item
                        world[x, y, 3] = misc
                        occupied.add((x, y))
        return world

    @pytest.mark.parametrize("seed", range(100))
    def test_fuzz_random_layout(self, seed):
        """Generate a random factory and verify both implementations agree."""
        rng = random.Random(seed)
        size = rng.randint(3, 8)
        density = rng.uniform(0.1, 0.5)
        world = self._random_world(size, rng, density)

        try:
            py_tp, py_unreachable = py_throughput_safe(world)
        except AssertionError:
            pytest.skip(f"Python calc_throughput assertion failed on seed {seed}")

        rs_tp, rs_unreachable = rs_throughput(world)

        assert abs(py_tp - rs_tp) <= 0.1, (
            f"Seed {seed}: throughput mismatch: Python={py_tp}, Rust={rs_tp}"
        )
        # Only check unreachable count when Python didn't return 0 due to cycles.
        # When Python detects cycles via nx.simple_cycles it returns unreachable=0,
        # but Rust may not detect the same cycles (e.g. self-loops from invalid
        # entities like underground belts with direction=NONE), leading to different
        # unreachable counts. Throughput still matches (both return 0).
        if py_tp != 0.0 or py_unreachable != 0:
            assert py_unreachable == rs_unreachable, (
                f"Seed {seed}: unreachable mismatch: Python={py_unreachable}, Rust={rs_unreachable}"
            )

    @pytest.mark.parametrize("seed", range(50))
    def test_fuzz_with_multi_tile(self, seed):
        """Random layouts including correctly-placed multi-tile entities."""
        rng = random.Random(seed + 2000)
        size = rng.randint(5, 10)
        world = self._random_world(size, rng, density=0.3, include_multi_tile=True)

        try:
            py_tp, py_unreachable = py_throughput_safe(world)
        except AssertionError:
            pytest.skip(f"Python calc_throughput assertion failed on seed {seed}")

        rs_tp, rs_unreachable = rs_throughput(world)

        assert abs(py_tp - rs_tp) <= 0.1, (
            f"Seed {seed}: throughput mismatch: Python={py_tp}, Rust={rs_tp}"
        )
        if py_tp != 0.0 or py_unreachable != 0:
            assert py_unreachable == rs_unreachable, (
                f"Seed {seed}: unreachable mismatch: Python={py_unreachable}, Rust={rs_unreachable}"
            )

    @pytest.mark.parametrize("seed", range(50))
    def test_fuzz_belt_only_layouts(self, seed):
        """Random layouts using only belts, sources, and sinks."""
        rng = random.Random(seed + 1000)
        size = rng.randint(3, 7)
        world = make_world(size)

        # Place source
        sx, sy = rng.randint(0, size - 1), rng.randint(0, size - 1)
        sdir = rng.choice([1, 2, 3, 4])
        world[sx, sy, 0] = 6  # source
        world[sx, sy, 1] = sdir
        world[sx, sy, 2] = 1  # copper_cable

        # Place sink
        for _ in range(10):
            bx, by = rng.randint(0, size - 1), rng.randint(0, size - 1)
            if (bx, by) != (sx, sy):
                break
        bdir = rng.choice([1, 2, 3, 4])
        world[bx, by, 0] = 5  # sink
        world[bx, by, 1] = bdir
        world[bx, by, 2] = 1  # copper_cable

        # Fill some belts
        for x in range(size):
            for y in range(size):
                if (x, y) in ((sx, sy), (bx, by)):
                    continue
                if rng.random() < 0.4:
                    world[x, y, 0] = 1  # transport_belt
                    world[x, y, 1] = rng.choice([1, 2, 3, 4])

        try:
            py_tp, py_unreachable = py_throughput_safe(world)
        except AssertionError:
            pytest.skip(f"Python calc_throughput assertion failed on seed {seed}")

        rs_tp, rs_unreachable = rs_throughput(world)

        assert abs(py_tp - rs_tp) <= 0.1, (
            f"Seed {seed}: throughput mismatch: Python={py_tp}, Rust={rs_tp}"
        )
        assert py_unreachable == rs_unreachable, (
            f"Seed {seed}: unreachable mismatch: Python={py_unreachable}, Rust={rs_unreachable}"
        )


# ── Splitter parity tests ──────────────────────────────────────────────────

# Helper: for a splitter facing `d` at anchor (ax, ay), compute the positions
# of left/right input cells (behind) and left/right output cells (ahead).
def _splitter_io_cells(ax, ay, d):
    """Returns (tiles, input_cells, output_cells) for a splitter."""
    tiles = factorion_rs.py_entity_tiles(ax, ay, d.value, 2, 1)
    assert tiles is not None
    dx, dy = {
        Direction.EAST: (1, 0), Direction.WEST: (-1, 0),
        Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1),
    }[d]
    input_cells = [(t[0] - dx, t[1] - dy) for t in tiles]
    output_cells = [(t[0] + dx, t[1] + dy) for t in tiles]
    return tiles, input_cells, output_cells


class TestSplitterExhaustive:
    """Test every combination of direction x inputs x outputs."""

    DIRS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    # Subsets of {left, right} for inputs and outputs
    SUBSETS = [(), (0,), (1,), (0, 1)]

    @pytest.mark.parametrize("d", DIRS, ids=lambda d: d.name)
    @pytest.mark.parametrize("inputs", SUBSETS, ids=["no_in", "left_in", "right_in", "both_in"])
    @pytest.mark.parametrize("outputs", SUBSETS, ids=["no_out", "left_out", "right_out", "both_out"])
    def test_splitter_combination(self, d, inputs, outputs):
        """For each direction x input subset x output subset, verify parity."""
        world = make_world(10)
        ax, ay = 4, 4  # anchor with room in all directions
        set_splitter(world, ax, ay, d)

        tiles, in_cells, out_cells = _splitter_io_cells(ax, ay, d)

        # Place source→belt on each active input
        for idx in inputs:
            bx, by = in_cells[idx]
            # belt behind the input cell
            dx, dy = {
                Direction.EAST: (1, 0), Direction.WEST: (-1, 0),
                Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1),
            }[d]
            sx, sy = bx - dx, by - dy
            if 0 <= sx < 10 and 0 <= sy < 10:
                set_entity(world, sx, sy, "source", d, "copper_cable")
            set_entity(world, bx, by, "transport_belt", d)

        # Place belt→sink on each active output
        for idx in outputs:
            bx, by = out_cells[idx]
            dx, dy = {
                Direction.EAST: (1, 0), Direction.WEST: (-1, 0),
                Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1),
            }[d]
            kx, ky = bx + dx, by + dy
            set_entity(world, bx, by, "transport_belt", d)
            if 0 <= kx < 10 and 0 <= ky < 10:
                set_entity(world, kx, ky, "sink", d, "copper_cable")

        # Run both implementations and check parity
        tp, unreachable = compare_throughput(world)

        n_in = len(inputs)
        n_out = len(outputs)
        if n_in == 0 or n_out == 0:
            assert tp == 0.0, f"No path expected, got tp={tp}"
        else:
            # Each input belt contributes 15 i/s. Splitter capacity = 30 i/s
            # (2 lanes). Split evenly among outputs, each capped at 15 by
            # the output belt.
            splitter_out = min(n_in * 15.0, 30.0) / n_out
            expected_per_output = min(splitter_out, 15.0)
            expected_total = expected_per_output * n_out
            assert abs(tp - expected_total) < 1e-6, (
                f"dir={d.name} in={inputs} out={outputs}: "
                f"expected {expected_total}, got {tp}"
            )


class TestSplitterChaining:
    """Test splitter→belt→splitter chains."""

    def test_two_splitters_in_series(self):
        """Source → Belt → Splitter → Belt → Splitter → 2x(Belt → Sink)."""
        world = make_world(10, 3)
        set_entity(world, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_splitter(world, 2, 0, Direction.EAST)
        set_entity(world, 3, 0, "transport_belt", Direction.EAST)
        set_splitter(world, 4, 0, Direction.EAST)
        set_entity(world, 5, 0, "transport_belt", Direction.EAST)
        set_entity(world, 5, 1, "transport_belt", Direction.EAST)
        set_entity(world, 6, 0, "sink", Direction.EAST, "copper_cable")
        set_entity(world, 6, 1, "sink", Direction.EAST, "copper_cable")

        tp, unreachable = compare_throughput(world)
        # First splitter: 1 input, 1 output (passthrough) = 15.0
        # Second splitter: 1 input, 2 outputs = 7.5 each = 15.0 total
        assert abs(tp - 15.0) < 1e-6, f"Expected 15.0, got {tp}"

    def test_split_then_merge(self):
        """Source → Splitter → 2x Belt → Splitter → Belt → Sink."""
        world = make_world(10, 3)
        set_entity(world, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_splitter(world, 2, 0, Direction.EAST)
        set_entity(world, 3, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 1, "transport_belt", Direction.EAST)
        set_splitter(world, 4, 0, Direction.EAST)
        set_entity(world, 5, 0, "transport_belt", Direction.EAST)
        set_entity(world, 6, 0, "sink", Direction.EAST, "copper_cable")

        tp, unreachable = compare_throughput(world)
        # Split 15 into 7.5+7.5, merge back: 15.0 total, 1 output = 15.0
        assert abs(tp - 15.0) < 1e-6, f"Expected 15.0, got {tp}"

    def test_splitter_fan_out_to_two_splitters(self):
        """Splitter1 outputs each feed a separate downstream splitter via belts.

        Layout (east-facing, y increases downward):
          Source → Belt ─┐
          Source → Belt ─┤ Splitter1 ─ Belt ─┐ Splitter2 ─ Belt → Sink
                         └───────── Belt ────┤ Splitter3 ─ Belt → Sink
        Splitter1 at (2,0), outputs at (3,0)/(3,1)
        Belt at (3,0) → Splitter2 at (4,0)  (left input only)
        Belt at (3,1) → Splitter3 at (4,2)? No, need alignment.
        Easier: route (3,1) south then east.
        """
        world = make_world(10, 6)
        # Two sources into splitter1
        set_entity(world, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(world, 0, 1, "source", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 1, 1, "transport_belt", Direction.EAST)
        set_splitter(world, 2, 0, Direction.EAST)  # tiles (2,0)/(2,1)

        # Left output (3,0) → belt → splitter2
        set_entity(world, 3, 0, "transport_belt", Direction.EAST)
        set_splitter(world, 4, 0, Direction.EAST)  # tiles (4,0)/(4,1)
        set_entity(world, 5, 0, "transport_belt", Direction.EAST)
        set_entity(world, 6, 0, "sink", Direction.EAST, "copper_cable")

        # Right output (3,1) → belt south → belt east → splitter3
        set_entity(world, 3, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 3, 2, "transport_belt", Direction.SOUTH)
        set_entity(world, 3, 3, "transport_belt", Direction.EAST)
        set_entity(world, 4, 3, "transport_belt", Direction.EAST)
        set_splitter(world, 5, 3, Direction.EAST)  # tiles (5,3)/(5,4)
        set_entity(world, 6, 3, "transport_belt", Direction.EAST)
        set_entity(world, 7, 3, "sink", Direction.EAST, "copper_cable")

        tp, _ = compare_throughput(world)
        # 2 inputs (30 total) → splitter1 (30 cap, 2 outputs = 15 each)
        # Each downstream splitter: 15 in, 1 output = 15 each
        # Total at sinks = 30
        assert abs(tp - 30.0) < 1e-6, f"Expected 30.0, got {tp}"

    def test_splitter_fan_in(self):
        """4 sources → 2 splitters → 1 splitter → sink.

        Layout (east-facing):
          A → belt ─┐ S1 ─ belt ─┐
          B → belt ─┘             ├─ S3 ─ belt → Sink
          C → belt ─┐ S2 ─ belt ─┘
          D → belt ─┘

        S1: 2 in (30), 1 out = 30 → belt caps at 15
        S2: 2 in (30), 1 out = 30 → belt caps at 15
        S3: 2 in (15+15=30), 1 out = 30 → belt caps at 15
        Sink = 15
        """
        world = make_world(10, 8)

        # Sources A,B → belts → S1
        set_entity(world, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(world, 0, 1, "source", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 1, 1, "transport_belt", Direction.EAST)
        set_splitter(world, 2, 0, Direction.EAST)  # tiles (2,0)/(2,1)

        # Sources C,D → belts → S2
        set_entity(world, 0, 4, "source", Direction.EAST, "copper_cable")
        set_entity(world, 0, 5, "source", Direction.EAST, "copper_cable")
        set_entity(world, 1, 4, "transport_belt", Direction.EAST)
        set_entity(world, 1, 5, "transport_belt", Direction.EAST)
        set_splitter(world, 2, 4, Direction.EAST)  # tiles (2,4)/(2,5)

        # S1 single output (top lane) → belt east → route down to S3
        set_entity(world, 3, 0, "transport_belt", Direction.EAST)
        set_entity(world, 4, 0, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 2, "transport_belt", Direction.EAST)

        # S2 single output (top lane) → belt east → route up to S3
        set_entity(world, 3, 4, "transport_belt", Direction.EAST)
        set_entity(world, 4, 4, "transport_belt", Direction.NORTH)
        set_entity(world, 4, 3, "transport_belt", Direction.EAST)

        # S3: inputs from (4,2) and (4,3) → anchor at (5,2)
        set_splitter(world, 5, 2, Direction.EAST)  # tiles (5,2)/(5,3)

        # S3 single output → belt → sink
        set_entity(world, 6, 2, "transport_belt", Direction.EAST)
        set_entity(world, 7, 2, "sink", Direction.EAST, "copper_cable")

        tp, _ = compare_throughput(world)
        # Each intermediate belt caps at 15, so S3 gets 15+15=30 in,
        # 1 output = 30, but output belt caps at 15.
        assert abs(tp - 15.0) < 1e-6, f"Expected 15.0, got {tp}"


class TestSplitterSecondaryTile:
    """Test that belts feeding into a splitter's secondary tile behave correctly."""

    def test_belt_into_secondary_tile_no_connection(self):
        """A belt pointing at the splitter's secondary tile from the side
        should NOT connect (secondary tile is not a separate graph node)."""
        world = make_world(10)
        set_splitter(world, 4, 4, Direction.EAST)
        # Place a belt pointing south into (4, 5) — the secondary tile
        set_entity(world, 4, 3, "source", Direction.SOUTH, "copper_cable")
        set_entity(world, 4, 5, "transport_belt", Direction.SOUTH)
        # The belt at (4,5) overlaps the splitter secondary tile, but
        # since the secondary is skipped in graph building, this belt
        # replaces it. The source→belt edge exists but goes nowhere useful.
        # Just verify parity (no crash, both agree on throughput).
        compare_throughput(world)


# ── Edge case tests ─────────────────────────────────────────────────────────

DIRS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]


class TestSplitterEdgeCases:
    @pytest.mark.parametrize("d", DIRS, ids=lambda d: d.name)
    def test_splitter_at_grid_edge(self, d):
        """Splitter placed at the grid boundary where some IO cells are OOB."""
        world = make_world(6)
        # Place at edge so one input or output is out of bounds
        if d == Direction.EAST:
            ax, ay = 5, 0  # output cells at (6,0)/(6,1) — OOB
        elif d == Direction.WEST:
            ax, ay = 0, 0  # output cells at (-1,0)/(-1,1) — OOB
        elif d == Direction.NORTH:
            ax, ay = 0, 0  # output cells at (0,-1)/(1,-1) — OOB
        elif d == Direction.SOUTH:
            ax, ay = 0, 5  # output cells at (0,6)/(1,6) — OOB
        set_splitter(world, ax, ay, d)
        # Should not crash, just produce zero throughput
        tp, _ = compare_throughput(world)
        assert tp == 0.0

    def test_adjacent_splitters_no_belt(self):
        """Two splitters placed directly adjacent (no belt between) → zero throughput.

        Splitter connections only accept belt-like entities, so direct
        splitter→splitter produces no edges.
        """
        world = make_world(8)
        set_entity(world, 0, 0, "source", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_splitter(world, 2, 0, Direction.EAST)  # tiles (2,0)/(2,1)
        # Splitter2 directly adjacent — its input cells are at (2,0)/(2,1)
        # which are splitter1's tiles. Splitter connections reject non-belt entities.
        set_splitter(world, 3, 0, Direction.EAST)  # tiles (3,0)/(3,1)
        set_entity(world, 4, 0, "transport_belt", Direction.EAST)
        set_entity(world, 5, 0, "sink", Direction.EAST, "copper_cable")

        tp, _ = compare_throughput(world)
        # No connection between the two splitters
        assert tp == 0.0

    @pytest.mark.parametrize("d", DIRS, ids=lambda d: d.name)
    def test_underground_belt_into_splitter(self, d):
        """Underground belt (up) → splitter input should connect."""
        world = make_world(10)
        ax, ay = 4, 4
        dx, dy = {
            Direction.EAST: (1, 0), Direction.WEST: (-1, 0),
            Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1),
        }[d]

        tiles = factorion_rs.py_entity_tiles(ax, ay, d.value, 2, 1)
        # Place underground belt (up) behind the first input cell
        in_x, in_y = tiles[0][0] - dx, tiles[0][1] - dy
        ub_x, ub_y = in_x - dx, in_y - dy
        # Source → underground_down → ... → underground_up → splitter → belt → sink
        src_x, src_y = ub_x - dx, ub_y - dy
        if not all(0 <= c < 10 for c in [src_x, src_y, ub_x, ub_y, in_x, in_y]):
            pytest.skip("Not enough room for this direction")

        set_entity(world, src_x, src_y, "source", d, "copper_cable")
        set_entity(world, ub_x, ub_y, "underground_belt", d, misc=1)  # down
        set_entity(world, in_x, in_y, "underground_belt", d, misc=2)  # up
        set_splitter(world, ax, ay, d)
        # Output: belt → sink
        out_x, out_y = tiles[0][0] + dx, tiles[0][1] + dy
        sink_x, sink_y = out_x + dx, out_y + dy
        if not all(0 <= c < 10 for c in [out_x, out_y, sink_x, sink_y]):
            pytest.skip("Not enough room for output in this direction")
        set_entity(world, out_x, out_y, "transport_belt", d)
        set_entity(world, sink_x, sink_y, "sink", d, "copper_cable")

        tp, _ = compare_throughput(world)
        assert tp > 0, f"Expected nonzero throughput for underground→splitter ({d.name})"

    @pytest.mark.parametrize("d", DIRS, ids=lambda d: d.name)
    def test_splitter_chaining_all_directions(self, d):
        """Splitter → belt → splitter chain in every direction."""
        world = make_world(12)
        dx, dy = {
            Direction.EAST: (1, 0), Direction.WEST: (-1, 0),
            Direction.NORTH: (0, -1), Direction.SOUTH: (0, 1),
        }[d]
        # Start from center, lay out: source → belt → S1 → belt → S2 → belt → sink
        cx, cy = 5, 5
        pos = (cx - 5 * dx, cy - 5 * dy)
        if not (0 <= pos[0] < 12 and 0 <= pos[1] < 12):
            pos = (cx, cy)

        x, y = 2, 5  # fixed start that works for all dirs with enough room
        if d == Direction.EAST:
            x, y = 1, 4
        elif d == Direction.WEST:
            x, y = 10, 4
        elif d == Direction.NORTH:
            x, y = 4, 10
        elif d == Direction.SOUTH:
            x, y = 4, 1

        set_entity(world, x, y, "source", d, "copper_cable")
        x, y = x + dx, y + dy
        set_entity(world, x, y, "transport_belt", d)
        x, y = x + dx, y + dy
        set_splitter(world, x, y, d)
        # Skip over splitter's depth (1 tile along flow)
        x, y = x + dx, y + dy
        set_entity(world, x, y, "transport_belt", d)
        x, y = x + dx, y + dy
        set_splitter(world, x, y, d)
        x, y = x + dx, y + dy
        set_entity(world, x, y, "transport_belt", d)
        x, y = x + dx, y + dy
        set_entity(world, x, y, "sink", d, "copper_cable")

        tp, _ = compare_throughput(world)
        assert abs(tp - 15.0) < 1e-6, f"Expected 15.0 for {d.name} chain, got {tp}"

    def test_splitter_direction_none(self):
        """Splitter with Direction.NONE should produce zero throughput."""
        world = make_world(6)
        # Manually place splitter tiles with NONE direction (can't use set_splitter)
        set_entity(world, 2, 2, "splitter", Direction.NONE)
        set_entity(world, 2, 3, "splitter", Direction.NONE)
        set_entity(world, 0, 2, "source", Direction.EAST, "copper_cable")
        set_entity(world, 1, 2, "transport_belt", Direction.EAST)
        set_entity(world, 4, 2, "sink", Direction.EAST, "copper_cable")
        tp, _ = compare_throughput(world)
        assert tp == 0.0
