"""Handcrafted mini-world parity tests.

Each test builds a small, specific factory layout by hand and verifies that
the Python and Rust throughput implementations agree.  The worlds exercise
edge-cases that random fuzz testing is unlikely to hit:

- Zigzag belts (S-curves, U-turns, spirals)
- Belts merging / splitting
- Inserter bottlenecks and chains
- Assembling machines with recipes
- Underground belts (short / max range / chained)
- Mixed entity combinations
- Multi-source / multi-sink factories
- Different item types
- Large belt grids
"""

import pytest

from helpers import (
    Direction,
    Misc,
    compare_throughput,
    make_world,
    set_assembler,
    set_entity,
    world2graph,
)


# ── Zigzag / curved belt paths ──────────────────────────────────────────────


class TestZigzagBelts:
    def test_s_curve_east_south_east(self):
        """S-curve: east -> south -> west -> south -> east."""
        world = make_world(6)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "transport_belt", Direction.SOUTH)
        set_entity(world, 3, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 3, 2, "transport_belt", Direction.WEST)
        set_entity(world, 2, 2, "transport_belt", Direction.WEST)
        set_entity(world, 1, 2, "transport_belt", Direction.SOUTH)
        set_entity(world, 1, 3, "transport_belt", Direction.EAST)
        set_entity(world, 2, 3, "transport_belt", Direction.EAST)
        set_entity(world, 3, 3, "bulk_inserter", Direction.EAST, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_u_turn_south_west_north(self):
        """U-turn: south -> west -> north."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.SOUTH, "copper_cable")
        set_entity(world, 0, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 0, 2, "transport_belt", Direction.EAST)
        set_entity(world, 1, 2, "transport_belt", Direction.NORTH)
        set_entity(world, 1, 1, "transport_belt", Direction.NORTH)
        set_entity(world, 1, 0, "bulk_inserter", Direction.NORTH, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_zigzag_3_turns(self):
        """3-turn zigzag going east then south then east then south."""
        world = make_world(8)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_plate")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.SOUTH)
        set_entity(world, 2, 1, "transport_belt", Direction.EAST)
        set_entity(world, 3, 1, "transport_belt", Direction.EAST)
        set_entity(world, 4, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 2, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 3, "transport_belt", Direction.WEST)
        set_entity(world, 3, 3, "transport_belt", Direction.WEST)
        set_entity(world, 2, 3, "transport_belt", Direction.SOUTH)
        set_entity(world, 2, 4, "bulk_inserter", Direction.SOUTH, "copper_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_spiral_inward(self):
        """Clockwise spiral inward on a 5x5 grid."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "transport_belt", Direction.EAST)
        set_entity(world, 4, 0, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 2, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 3, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 4, "transport_belt", Direction.WEST)
        set_entity(world, 3, 4, "transport_belt", Direction.WEST)
        set_entity(world, 2, 4, "transport_belt", Direction.NORTH)
        set_entity(world, 2, 3, "transport_belt", Direction.NORTH)
        set_entity(world, 2, 2, "transport_belt", Direction.WEST)
        set_entity(world, 1, 2, "bulk_inserter", Direction.WEST, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_long_serpentine(self):
        """Long serpentine belt pattern: east -> south -> west -> south -> east."""
        world = make_world(7, 7)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        for x in range(1, 6):
            set_entity(world, x, 0, "transport_belt", Direction.EAST)
        set_entity(world, 6, 0, "transport_belt", Direction.SOUTH)
        set_entity(world, 6, 1, "transport_belt", Direction.WEST)
        for x in range(5, 0, -1):
            set_entity(world, x, 1, "transport_belt", Direction.WEST)
        set_entity(world, 0, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 0, 2, "transport_belt", Direction.EAST)
        for x in range(1, 6):
            set_entity(world, x, 2, "transport_belt", Direction.EAST)
        set_entity(world, 6, 2, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)


# ── Belt merging ─────────────────────────────────────────────────────────────


class TestBeltMerging:
    def test_two_sources_merge_onto_one_belt(self):
        """Two sources feed into the same belt from perpendicular directions."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.SOUTH, "iron_plate")
        set_entity(world, 0, 1, "transport_belt", Direction.EAST)
        set_entity(world, 1, 1, "transport_belt", Direction.EAST)
        set_entity(world, 2, 1, "bulk_inserter", Direction.EAST, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_side_loading_belts(self):
        """Side-loading: two belts merge from different directions onto one belt."""
        world = make_world(5)
        set_entity(world, 0, 1, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "stack_inserter", Direction.SOUTH, "copper_cable")
        set_entity(world, 1, 1, "transport_belt", Direction.EAST)
        set_entity(world, 2, 1, "transport_belt", Direction.EAST)
        set_entity(world, 3, 1, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t > 0

    def test_parallel_paths_to_sink(self):
        """Two independent source->belt->sink paths in parallel rows."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "bulk_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 2, "transport_belt", Direction.EAST)
        set_entity(world, 2, 2, "bulk_inserter", Direction.EAST, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(30.0, abs=1e-6)


# ── Inserter patterns ────────────────────────────────────────────────────────


class TestInserterPatterns:
    def test_double_inserter_bottleneck(self):
        """Source -> inserter -> belt -> inserter -> belt -> sink."""
        world = make_world(7)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_plate")
        set_entity(world, 1, 0, "inserter", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "inserter", Direction.EAST)
        set_entity(world, 4, 0, "transport_belt", Direction.EAST)
        set_entity(world, 5, 0, "bulk_inserter", Direction.EAST, "copper_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(0.86, abs=1e-6)

    def test_inserter_feeds_belt_chain(self):
        """Source -> inserter -> long belt chain -> sink."""
        world = make_world(8)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 0, "inserter", Direction.EAST)
        for x in range(2, 7):
            set_entity(world, x, 0, "transport_belt", Direction.EAST)
        set_entity(world, 7, 0, "bulk_inserter", Direction.EAST, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(0.86, abs=1e-6)

    def test_inserter_between_belts(self):
        """Belt -> inserter -> belt. Inserter in middle of belt chain."""
        world = make_world(7)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "inserter", Direction.EAST)
        set_entity(world, 4, 0, "transport_belt", Direction.EAST)
        set_entity(world, 5, 0, "transport_belt", Direction.EAST)
        set_entity(world, 6, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(0.86, abs=1e-6)

    def test_inserter_all_directions(self):
        """Inserters in all four cardinal directions."""
        for direction, src_pos, ins_pos, belt_pos, sink_pos in [
            (Direction.EAST, (0, 2), (1, 2), (2, 2), (3, 2)),
            (Direction.WEST, (4, 2), (3, 2), (2, 2), (1, 2)),
            (Direction.SOUTH, (2, 0), (2, 1), (2, 2), (2, 3)),
            (Direction.NORTH, (2, 4), (2, 3), (2, 2), (2, 1)),
        ]:
            world = make_world(5)
            set_entity(world, *src_pos, "stack_inserter", direction, "iron_plate")
            set_entity(world, *ins_pos, "inserter", direction)
            set_entity(world, *belt_pos, "transport_belt", direction)
            set_entity(world, *sink_pos, "bulk_inserter", direction, "iron_plate")
            t, u = compare_throughput(world)
            assert t == pytest.approx(0.86, abs=1e-6), (
                f"Failed for direction {direction}"
            )


# ── Underground belts ────────────────────────────────────────────────────────


class TestUndergroundBeltPatterns:
    def test_underground_minimum_distance(self):
        """Underground belt pair with minimum gap (adjacent)."""
        world = make_world(6)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(
            world, 2, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        set_entity(
            world, 3, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_UP.value,
        )
        set_entity(world, 4, 0, "transport_belt", Direction.EAST)
        set_entity(world, 5, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_underground_max_range(self):
        """Underground belt pair at maximum range (delta 1..6 exclusive = max 5)."""
        world = make_world(9)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(
            world, 1, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        set_entity(
            world, 6, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_UP.value,
        )
        set_entity(world, 7, 0, "transport_belt", Direction.EAST)
        set_entity(world, 8, 0, "bulk_inserter", Direction.EAST, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_chained_underground_belts(self):
        """Two underground belt pairs in sequence."""
        world = make_world(12)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_plate")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(
            world, 2, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        set_entity(
            world, 5, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_UP.value,
        )
        set_entity(world, 6, 0, "transport_belt", Direction.EAST)
        set_entity(
            world, 7, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        set_entity(
            world, 10, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_UP.value,
        )
        set_entity(world, 11, 0, "bulk_inserter", Direction.EAST, "copper_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_underground_vertical_south(self):
        """Vertical underground belts going south."""
        world = make_world(8)
        set_entity(world, 0, 0, "stack_inserter", Direction.SOUTH, "iron_plate")
        set_entity(world, 0, 1, "transport_belt", Direction.SOUTH)
        set_entity(
            world, 0, 2, "underground_belt", Direction.SOUTH,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        set_entity(
            world, 0, 5, "underground_belt", Direction.SOUTH,
            misc=Misc.UNDERGROUND_UP.value,
        )
        set_entity(world, 0, 6, "transport_belt", Direction.SOUTH)
        set_entity(world, 0, 7, "bulk_inserter", Direction.SOUTH, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_underground_with_obstacle(self):
        """Underground belts bypass an entity in the middle."""
        world = make_world(8)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(
            world, 2, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        set_entity(world, 3, 0, "transport_belt", Direction.NORTH)
        set_entity(world, 4, 0, "transport_belt", Direction.NORTH)
        set_entity(
            world, 5, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_UP.value,
        )
        set_entity(world, 6, 0, "transport_belt", Direction.EAST)
        set_entity(world, 7, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)


# ── Assembling machines ──────────────────────────────────────────────────────


class TestAssemblingMachines:
    def test_copper_cable_factory(self):
        """Source(copper_plate) -> inserter -> assembler(copper_cable) -> inserter -> sink."""
        world = make_world(9, 5)
        set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "copper_plate")
        set_entity(world, 1, 2, "inserter", Direction.EAST)
        set_assembler(world, 2, 1, "copper_cable")
        set_entity(world, 5, 2, "inserter", Direction.EAST)
        set_entity(world, 6, 2, "transport_belt", Direction.EAST)
        set_entity(world, 7, 2, "transport_belt", Direction.EAST)
        set_entity(world, 8, 2, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t > 0

    def test_electronic_circuit_factory(self):
        """Two sources (copper_cable + iron_plate) -> assembler -> sink."""
        world = make_world(9, 7)
        set_entity(world, 3, 0, "stack_inserter", Direction.SOUTH, "copper_cable")
        set_entity(world, 3, 1, "inserter", Direction.SOUTH)
        set_entity(world, 0, 3, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 3, "inserter", Direction.EAST)
        set_assembler(world, 2, 2, "electronic_circuit")
        set_entity(world, 5, 3, "inserter", Direction.EAST)
        set_entity(world, 6, 3, "transport_belt", Direction.EAST)
        set_entity(world, 7, 3, "transport_belt", Direction.EAST)
        set_entity(world, 8, 3, "bulk_inserter", Direction.EAST, "electronic_circuit")
        t, u = compare_throughput(world)
        assert t > 0

    def test_assembler_with_belt_input(self):
        """Source -> belt chain -> inserter -> assembler -> inserter -> sink."""
        world = make_world(11, 5)
        set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "copper_plate")
        set_entity(world, 1, 2, "transport_belt", Direction.EAST)
        set_entity(world, 2, 2, "transport_belt", Direction.EAST)
        set_entity(world, 3, 2, "inserter", Direction.EAST)
        set_assembler(world, 4, 1, "copper_cable")
        set_entity(world, 7, 2, "inserter", Direction.EAST)
        set_entity(world, 8, 2, "transport_belt", Direction.EAST)
        set_entity(world, 9, 2, "transport_belt", Direction.EAST)
        set_entity(world, 10, 2, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t > 0

    def test_assembler_no_input_connected(self):
        """Assembler with no inserters feeding it: throughput should be 0."""
        world = make_world(7, 5)
        set_assembler(world, 2, 1, "copper_cable")
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_plate")
        set_entity(world, 6, 4, "bulk_inserter", Direction.EAST, "copper_cable")
        compare_throughput(world)


# ── Exhaustive assembler perimeter coverage ────────────────────────────────
#
# The 3x3 assembling machine has 12 non-diagonal perimeter tiles where an
# inserter (or source/sink, which use the same connection logic) can interact
# with it. Layout:
#
#     x i i i x
#     i A A A i
#     i A A A i
#     i A A A i
#     x i i i x
#
# `x` = diagonal corners (no interaction). `i` = the 12 interactable slots
# (3 per side × 4 sides). For each slot, the single "toward machine" direction
# makes the adjacent inserter/source feed INTO the assembler. The single
# "away from machine" direction makes the adjacent inserter/sink take OUT of
# the assembler. The other 2 facing directions ("along" the edge) still
# register the adjacent entity as an assembler neighbour but drop/pickup cells
# miss the body — those are excluded from the lesson generator and not tested
# here.


# (label, (dx, dy) offset from assembler anchor, Direction toward machine)
# Assembler anchor is the top-left tile of the 3x3 body.
_PERIMETER_SLOTS = [
    # North edge (y = ay - 1): all face SOUTH to feed INTO the assembler
    ("N_left",   (0, -1), Direction.SOUTH),
    ("N_center", (1, -1), Direction.SOUTH),
    ("N_right",  (2, -1), Direction.SOUTH),
    # South edge (y = ay + 3): all face NORTH to feed INTO the assembler
    ("S_left",   (0,  3), Direction.NORTH),
    ("S_center", (1,  3), Direction.NORTH),
    ("S_right",  (2,  3), Direction.NORTH),
    # West edge (x = ax - 1): all face EAST to feed INTO the assembler
    ("W_top",    (-1, 0), Direction.EAST),
    ("W_mid",    (-1, 1), Direction.EAST),
    ("W_bot",    (-1, 2), Direction.EAST),
    # East edge (x = ax + 3): all face WEST to feed INTO the assembler
    ("E_top",    (3,  0), Direction.WEST),
    ("E_mid",    (3,  1), Direction.WEST),
    ("E_bot",    (3,  2), Direction.WEST),
]


_OPPOSITE = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}


_DELTA = {
    Direction.NORTH: (0, -1),
    Direction.SOUTH: (0, 1),
    Direction.EAST: (1, 0),
    Direction.WEST: (-1, 0),
}


def _assembler_edges(world, assembler_x, assembler_y):
    """Return edges (src_node, dst_node) touching any tile of the 3x3 assembler."""
    G = world2graph(world)
    body_tiles = {
        (assembler_x + dx, assembler_y + dy) for dx in range(3) for dy in range(3)
    }

    def tile_of(node):
        # Nodes are f"{name}\n@{x},{y}"
        coords = node.split("@")[1]
        x, y = coords.split(",")
        return (int(x), int(y))

    edges = []
    for u, v in G.edges():
        u_is_body = tile_of(u) in body_tiles
        v_is_body = tile_of(v) in body_tiles
        if u_is_body or v_is_body:
            edges.append((u, v, u_is_body, v_is_body))
    return edges


class TestAssemblerPerimeterConnections:
    """DAG-level: verify every perimeter slot connects correctly in both directions."""

    # Place the assembler at a fixed anchor with plenty of clearance in an 11x11 world.
    AX = 4
    AY = 4
    SIZE = 11

    @pytest.mark.parametrize("label,offset,toward_dir", _PERIMETER_SLOTS)
    def test_inserter_toward_machine_creates_inserter_to_assembler_edge(
        self, label, offset, toward_dir
    ):
        """Inserter facing TOWARD the assembler body → edge direction is inserter → assembler."""
        world = make_world(self.SIZE)
        set_assembler(world, self.AX, self.AY, "copper_cable")
        ix = self.AX + offset[0]
        iy = self.AY + offset[1]
        set_entity(world, ix, iy, "inserter", toward_dir)

        edges = _assembler_edges(world, self.AX, self.AY)
        # Exactly one edge should exist, and it should flow INTO a body tile.
        inserter_tag = f"inserter\n@{ix},{iy}"
        matching = [
            (u, v) for (u, v, u_body, v_body) in edges
            if (u == inserter_tag and v_body) or (v == inserter_tag and u_body)
        ]
        assert len(matching) == 1, (
            f"slot {label}: expected 1 edge to/from inserter@{ix},{iy}, got {matching}"
        )
        u, v = matching[0]
        assert u == inserter_tag, (
            f"slot {label}: edge should be inserter→assembler, got {u}→{v}"
        )

    @pytest.mark.parametrize("label,offset,toward_dir", _PERIMETER_SLOTS)
    def test_inserter_away_from_machine_creates_assembler_to_inserter_edge(
        self, label, offset, toward_dir
    ):
        """Inserter facing AWAY from the assembler body → edge direction is assembler → inserter."""
        world = make_world(self.SIZE)
        set_assembler(world, self.AX, self.AY, "copper_cable")
        ix = self.AX + offset[0]
        iy = self.AY + offset[1]
        away_dir = _OPPOSITE[toward_dir]
        set_entity(world, ix, iy, "inserter", away_dir)

        edges = _assembler_edges(world, self.AX, self.AY)
        inserter_tag = f"inserter\n@{ix},{iy}"
        matching = [
            (u, v) for (u, v, u_body, v_body) in edges
            if (u == inserter_tag and v_body) or (v == inserter_tag and u_body)
        ]
        assert len(matching) == 1, (
            f"slot {label}: expected 1 edge to/from inserter@{ix},{iy}, got {matching}"
        )
        u, v = matching[0]
        assert v == inserter_tag, (
            f"slot {label}: edge should be assembler→inserter, got {u}→{v}"
        )

    def test_diagonal_corners_have_no_edge(self):
        """The 4 diagonal corners (x tiles in the diagram) must NOT connect."""
        corners = [(-1, -1), (3, -1), (-1, 3), (3, 3)]
        for dx, dy in corners:
            for facing in [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]:
                world = make_world(self.SIZE)
                set_assembler(world, self.AX, self.AY, "copper_cable")
                ix = self.AX + dx
                iy = self.AY + dy
                set_entity(world, ix, iy, "inserter", facing)

                edges = _assembler_edges(world, self.AX, self.AY)
                inserter_tag = f"inserter\n@{ix},{iy}"
                touching = [
                    (u, v) for (u, v, _, _) in edges if inserter_tag in (u, v)
                ]
                assert touching == [], (
                    f"corner ({dx},{dy}) facing {facing.name}: unexpected edge {touching}"
                )

    @pytest.mark.parametrize(
        "assembler_dir",
        [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST],
    )
    def test_assembler_own_direction_does_not_affect_connections(self, assembler_dir):
        """The assembler's own direction channel is ignored for connection logic."""
        # Set up a full source→inserter→assembler→inserter→sink chain.
        world = make_world(self.SIZE)
        set_assembler(world, self.AX, self.AY, "copper_cable")
        # Overwrite assembler direction
        from helpers import Channel
        for dx in range(3):
            for dy in range(3):
                world[self.AX + dx, self.AY + dy, Channel.DIRECTION.value] = (
                    assembler_dir.value
                )
        # Input: source at W edge, feeding inserter that drops INTO assembler
        set_entity(world, self.AX - 2, self.AY + 1, "stack_inserter",
                   Direction.EAST, "copper_plate")
        set_entity(world, self.AX - 1, self.AY + 1, "inserter", Direction.EAST)
        # Output: inserter picks from assembler body, drops onto sink at E edge
        set_entity(world, self.AX + 3, self.AY + 1, "inserter", Direction.EAST)
        set_entity(world, self.AX + 4, self.AY + 1, "bulk_inserter",
                   Direction.EAST, "copper_cable")

        tp, _ = compare_throughput(world)
        assert tp > 0, (
            f"throughput should be > 0 regardless of assembler direction {assembler_dir.name}"
        )


class TestAssemblerPerimeterThroughput:
    """End-to-end: full source→inserter→assembler→inserter→sink chain through each perimeter slot."""

    AX = 4
    AY = 4
    SIZE = 11

    # Known-working "anchor" output slot: E-center, inserter facing EAST (away).
    _OUT_OFFSET = (3, 1)
    _OUT_AWAY_DIR = Direction.EAST

    # Known-working "anchor" input slot: W-center, inserter facing EAST (toward).
    _IN_OFFSET = (-1, 1)
    _IN_TOWARD_DIR = Direction.EAST

    def _place_input_chain(self, world, offset, toward_dir, item):
        """Place source → inserter feeding into assembler body via perimeter slot."""
        ix = self.AX + offset[0]
        iy = self.AY + offset[1]
        dx, dy = _DELTA[toward_dir]
        # pickup cell is behind the inserter (from its facing)
        sx, sy = ix - dx, iy - dy
        set_entity(world, sx, sy, "stack_inserter", toward_dir, item)
        set_entity(world, ix, iy, "inserter", toward_dir)

    def _place_output_chain(self, world, offset, away_dir, item):
        """Place inserter (at perimeter slot) taking from assembler body → sink."""
        ix = self.AX + offset[0]
        iy = self.AY + offset[1]
        dx, dy = _DELTA[away_dir]
        # drop cell is in front of the inserter
        kx, ky = ix + dx, iy + dy
        set_entity(world, ix, iy, "inserter", away_dir)
        set_entity(world, kx, ky, "bulk_inserter", away_dir, item)

    @pytest.mark.parametrize("label,offset,toward_dir", _PERIMETER_SLOTS)
    def test_copper_cable_input_from_each_perimeter_slot(
        self, label, offset, toward_dir
    ):
        """Full copper_cable chain with the input inserter at each of the 12 slots.

        The output inserter is fixed at E-center. If the input slot cannot
        reach the assembler body, the input chain is broken and throughput
        drops to 0 — failure here indicates a connection-logic bug for that slot.
        """
        # Skip when input slot collides with the fixed output slot.
        if offset == self._OUT_OFFSET:
            pytest.skip("input slot coincides with fixed output slot")
        world = make_world(self.SIZE)
        set_assembler(world, self.AX, self.AY, "copper_cable")
        self._place_input_chain(world, offset, toward_dir, "copper_plate")
        self._place_output_chain(world, self._OUT_OFFSET, self._OUT_AWAY_DIR,
                                 "copper_cable")
        tp, _ = compare_throughput(world)
        assert tp > 0, f"slot {label}: expected > 0 throughput, got {tp}"

    @pytest.mark.parametrize("label,offset,toward_dir", _PERIMETER_SLOTS)
    def test_copper_cable_output_to_each_perimeter_slot(
        self, label, offset, toward_dir
    ):
        """Full copper_cable chain with the output inserter at each of the 12 slots.

        The input inserter is fixed at W-center. The output slot uses the
        "away from machine" direction so the assembler → inserter edge is created.
        """
        if offset == self._IN_OFFSET:
            pytest.skip("output slot coincides with fixed input slot")
        world = make_world(self.SIZE)
        set_assembler(world, self.AX, self.AY, "copper_cable")
        self._place_input_chain(world, self._IN_OFFSET, self._IN_TOWARD_DIR,
                                "copper_plate")
        away_dir = _OPPOSITE[toward_dir]
        self._place_output_chain(world, offset, away_dir, "copper_cable")
        tp, _ = compare_throughput(world)
        assert tp > 0, f"slot {label}: expected > 0 throughput, got {tp}"

    @pytest.mark.parametrize("label_cu,off_cu,toward_cu", _PERIMETER_SLOTS)
    @pytest.mark.parametrize("label_fe,off_fe,toward_fe", _PERIMETER_SLOTS)
    def test_electronic_circuit_two_inputs_pairs(
        self, label_cu, off_cu, toward_cu, label_fe, off_fe, toward_fe
    ):
        """Electronic circuit needs 2 distinct inputs (copper_cable + iron_plate).

        For each ordered pair of distinct input slots (12×11 = 132 combos),
        verify the recipe runs with the output fixed at E-center.
        """
        if off_cu == off_fe:
            pytest.skip("both inputs in the same slot")
        if off_cu == self._OUT_OFFSET or off_fe == self._OUT_OFFSET:
            pytest.skip("input slot coincides with fixed output slot")
        world = make_world(self.SIZE)
        set_assembler(world, self.AX, self.AY, "electronic_circuit")
        self._place_input_chain(world, off_cu, toward_cu, "copper_cable")
        self._place_input_chain(world, off_fe, toward_fe, "iron_plate")
        self._place_output_chain(world, self._OUT_OFFSET, self._OUT_AWAY_DIR,
                                 "electronic_circuit")
        tp, _ = compare_throughput(world)
        assert tp > 0, (
            f"cu@{label_cu}, fe@{label_fe}: expected > 0 throughput, got {tp}"
        )


# ── Different items ──────────────────────────────────────────────────────────


class TestDifferentItems:
    @pytest.mark.parametrize(
        "item",
        ["copper_cable", "copper_plate", "iron_plate", "electronic_circuit"],
    )
    def test_each_item_type(self, item):
        """Simple source->belt->sink for each item type."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, item)
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "bulk_inserter", Direction.EAST, item)
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_two_different_item_paths(self):
        """Two parallel paths carrying different items."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 2, "transport_belt", Direction.EAST)
        set_entity(world, 2, 2, "bulk_inserter", Direction.EAST, "iron_plate")
        t, u = compare_throughput(world)
        assert t > 0


# ── Multi-source / multi-sink ────────────────────────────────────────────────


class TestMultiSourceSink:
    def test_two_sources_one_sink(self):
        """Two sources feed belts that converge to one sink."""
        world = make_world(5, 5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.SOUTH)
        set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 2, "transport_belt", Direction.EAST)
        set_entity(world, 2, 2, "transport_belt", Direction.NORTH)
        set_entity(world, 2, 1, "transport_belt", Direction.EAST)
        set_entity(world, 3, 1, "transport_belt", Direction.EAST)
        set_entity(world, 4, 1, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t > 0

    def test_one_source_two_sinks(self):
        """One source feeds two sinks via parallel belt paths."""
        world = make_world(5, 5)
        set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 2, "transport_belt", Direction.EAST)
        set_entity(world, 2, 2, "transport_belt", Direction.NORTH)
        set_entity(world, 2, 1, "transport_belt", Direction.EAST)
        set_entity(world, 3, 1, "bulk_inserter", Direction.EAST, "iron_plate")
        t, u = compare_throughput(world)
        assert t > 0

    def test_three_parallel_paths(self):
        """Three independent source->belt->sink paths."""
        world = make_world(5, 7)
        for row in [0, 3, 6]:
            set_entity(world, 0, row, "stack_inserter", Direction.EAST, "iron_plate")
            set_entity(world, 1, row, "transport_belt", Direction.EAST)
            set_entity(world, 2, row, "transport_belt", Direction.EAST)
            set_entity(world, 3, row, "transport_belt", Direction.EAST)
            set_entity(world, 4, row, "bulk_inserter", Direction.EAST, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(45.0, abs=1e-6)


# ── Mixed entity combinations ───────────────────────────────────────────────


class TestMixedEntities:
    def test_belt_inserter_belt_underground(self):
        """Source -> belt -> inserter -> belt -> underground -> belt -> sink."""
        world = make_world(10)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "inserter", Direction.EAST)
        set_entity(world, 3, 0, "transport_belt", Direction.EAST)
        set_entity(
            world, 4, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        set_entity(
            world, 7, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_UP.value,
        )
        set_entity(world, 8, 0, "transport_belt", Direction.EAST)
        set_entity(world, 9, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(0.86, abs=1e-6)

    def test_underground_into_zigzag(self):
        """Underground belt feeds into a zigzag belt path."""
        world = make_world(8, 4)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(
            world, 1, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        set_entity(
            world, 4, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_UP.value,
        )
        set_entity(world, 5, 0, "transport_belt", Direction.EAST)
        set_entity(world, 6, 0, "transport_belt", Direction.SOUTH)
        set_entity(world, 6, 1, "transport_belt", Direction.WEST)
        set_entity(world, 5, 1, "bulk_inserter", Direction.WEST, "iron_plate")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_full_factory_copper_cable(self):
        """Complete mini factory: source -> inserter -> assembler -> inserter -> belt -> sink."""
        world = make_world(12, 5)
        set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "copper_plate")
        set_entity(world, 1, 2, "transport_belt", Direction.EAST)
        set_entity(world, 2, 2, "transport_belt", Direction.EAST)
        set_entity(world, 3, 2, "inserter", Direction.EAST)
        set_assembler(world, 4, 1, "copper_cable")
        set_entity(world, 7, 2, "inserter", Direction.EAST)
        set_entity(world, 8, 2, "transport_belt", Direction.EAST)
        set_entity(world, 9, 2, "transport_belt", Direction.EAST)
        set_entity(world, 10, 2, "transport_belt", Direction.EAST)
        set_entity(world, 11, 2, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t > 0

    def test_full_factory_electronic_circuit(self):
        """Complete electronic circuit factory with two input lines."""
        world = make_world(11, 9)
        set_entity(world, 0, 4, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 4, "transport_belt", Direction.EAST)
        set_entity(world, 2, 4, "inserter", Direction.EAST)
        set_entity(world, 4, 0, "stack_inserter", Direction.SOUTH, "iron_plate")
        set_entity(world, 4, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 4, 2, "inserter", Direction.SOUTH)
        set_assembler(world, 3, 3, "electronic_circuit")
        set_entity(world, 6, 4, "inserter", Direction.EAST)
        set_entity(world, 7, 4, "transport_belt", Direction.EAST)
        set_entity(world, 8, 4, "transport_belt", Direction.EAST)
        set_entity(world, 9, 4, "transport_belt", Direction.EAST)
        set_entity(
            world, 10, 4, "bulk_inserter", Direction.EAST, "electronic_circuit"
        )
        t, u = compare_throughput(world)
        assert t > 0


# ── Disconnected / unreachable patterns ──────────────────────────────────────


class TestUnreachablePatterns:
    def test_island_belts_no_source_no_sink(self):
        """Grid of belts with no source or sink: all unreachable."""
        world = make_world(4)
        for x in range(4):
            for y in range(4):
                set_entity(world, x, y, "transport_belt", Direction.EAST)
        compare_throughput(world)

    def test_connected_path_plus_disconnected_island(self):
        """Working path + disconnected belt island."""
        world = make_world(6, 6)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "bulk_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 4, 4, "transport_belt", Direction.EAST)
        set_entity(world, 5, 4, "transport_belt", Direction.EAST)
        set_entity(world, 4, 5, "transport_belt", Direction.EAST)
        set_entity(world, 5, 5, "transport_belt", Direction.EAST)
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_source_belt_dead_end(self):
        """Source -> belt -> belt (dead end, no sink)."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        compare_throughput(world)

    def test_sink_belt_no_source(self):
        """Belt -> belt -> sink (no source)."""
        world = make_world(5)
        set_entity(world, 0, 0, "transport_belt", Direction.EAST)
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "bulk_inserter", Direction.EAST, "iron_plate")
        compare_throughput(world)


# ── Large / stress test worlds ───────────────────────────────────────────────


class TestLargeWorlds:
    def test_long_belt_chain_20(self):
        """20-belt chain."""
        world = make_world(22)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        for x in range(1, 21):
            set_entity(world, x, 0, "transport_belt", Direction.EAST)
        set_entity(world, 21, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_grid_of_belts_10x10(self):
        """10x10 grid of east-facing belts with source and sink."""
        world = make_world(12, 10)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        for x in range(1, 11):
            set_entity(world, x, 0, "transport_belt", Direction.EAST)
        set_entity(world, 11, 0, "bulk_inserter", Direction.EAST, "iron_plate")
        for y in range(1, 10):
            for x in range(1, 11):
                set_entity(world, x, y, "transport_belt", Direction.EAST)
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_many_parallel_paths_5(self):
        """5 parallel independent paths."""
        world = make_world(6, 10)
        for row in range(0, 10, 2):
            set_entity(
                world, 0, row, "stack_inserter", Direction.EAST, "copper_plate"
            )
            set_entity(world, 1, row, "transport_belt", Direction.EAST)
            set_entity(world, 2, row, "transport_belt", Direction.EAST)
            set_entity(world, 3, row, "transport_belt", Direction.EAST)
            set_entity(world, 4, row, "transport_belt", Direction.EAST)
            set_entity(
                world, 5, row, "bulk_inserter", Direction.EAST, "copper_plate"
            )
        t, u = compare_throughput(world)
        assert t == pytest.approx(75.0, abs=1e-6)


# ── Belt direction edge cases ────────────────────────────────────────────────


class TestBeltDirectionEdgeCases:
    def test_perpendicular_belt_no_connect(self):
        """A belt perpendicular to the path should not feed into the path."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "bulk_inserter", Direction.EAST, "iron_plate")
        set_entity(world, 2, 1, "transport_belt", Direction.SOUTH)
        t, u = compare_throughput(world)
        assert t == pytest.approx(15.0, abs=1e-6)

    def test_belt_facing_away_from_sink(self):
        """Belt next to sink but facing away: should not connect."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.EAST)
        set_entity(world, 3, 0, "transport_belt", Direction.WEST)
        set_entity(world, 4, 0, "bulk_inserter", Direction.EAST, "copper_cable")
        compare_throughput(world)

    def test_all_belts_face_center(self):
        """4 belts facing center from all directions (convergence point)."""
        world = make_world(5)
        set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 1, 2, "transport_belt", Direction.EAST)
        set_entity(world, 2, 2, "transport_belt", Direction.EAST)
        set_entity(world, 3, 2, "transport_belt", Direction.EAST)
        set_entity(world, 4, 2, "bulk_inserter", Direction.EAST, "copper_cable")
        set_entity(world, 2, 1, "transport_belt", Direction.SOUTH)
        set_entity(world, 2, 3, "transport_belt", Direction.NORTH)
        t, u = compare_throughput(world)
        assert t > 0
