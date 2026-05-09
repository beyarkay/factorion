"""Lane-aware throughput tests.

The Rust implementation models double-sided belts: each transport belt and
underground belt carries TWO independent lanes (port = left of travel
direction, starboard = right). Per-lane flow caps at 7.5 i/s, so a
saturated belt total is 15 i/s. The Python implementation does not know
about lanes, so these tests bypass `compare_throughput` and call only
`rs_throughput` directly.

Every scenario is parametrised over the 8 symmetry variants of the plane
(4 rotations × 2 mirrors). The expected throughput is invariant under
rotation and mirror, so a single assertion per scenario should hold for
all 8 variants. If a lane-handling bug is direction-asymmetric, one of
the variants will fail.
"""

import itertools

import pytest

from helpers import (
    Direction,
    Misc,
    make_world,
    rs_throughput,
    set_entity,
)


# ── Symmetry helpers ─────────────────────────────────────────────────────────

ALL_DIRS = [
    Direction.NORTH,
    Direction.EAST,
    Direction.SOUTH,
    Direction.WEST,
]


def rotate_pos(x, y, size, n):
    """Rotate (x, y) by n × 90° clockwise around the centre of a `size×size` grid.

    Clockwise on screen-coords (y grows downward): (x, y) → (size-1-y, x).
    """
    for _ in range(n % 4):
        x, y = size - 1 - y, x
    return x, y


def rotate_dir(d, n):
    """Rotate a Direction by n × 90° clockwise. Direction.NONE is unchanged."""
    if d == Direction.NONE:
        return d
    seq = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]
    return seq[(seq.index(d) + n) % 4]


def mirror_pos_x(x, y, size):
    """Mirror (x, y) across the vertical axis (swaps left/right)."""
    return size - 1 - x, y


def mirror_dir_x(d):
    """Mirror a Direction across the vertical axis: E↔W, N and S unchanged."""
    return {Direction.EAST: Direction.WEST, Direction.WEST: Direction.EAST}.get(d, d)


def build_world(size, layout, rotation=0, mirror=False):
    """Build a square world of side `size` from a canonical layout.

    `layout` is a list of `(x, y, entity_name, direction, item_name_or_None)`
    tuples in the canonical (East-flow) frame. `rotation` rotates every
    entry n × 90° clockwise; `mirror` mirrors across the vertical axis
    AFTER rotation.
    """
    world = make_world(size)
    for x, y, kind, d, item in layout:
        rx, ry = rotate_pos(x, y, size, rotation)
        rd = rotate_dir(d, rotation)
        if mirror:
            rx, ry = mirror_pos_x(rx, ry, size)
            rd = mirror_dir_x(rd)
        item_name = item if item is not None else "empty"
        set_entity(world, rx, ry, kind, rd, item_name)
    return world


# Pytest IDs that read better in the run output than raw `(0, False)` tuples.
SYMMETRY_VARIANTS = list(itertools.product(range(4), [False, True]))
SYMMETRY_IDS = [
    f"rot{r * 90:03d}{'-mirror' if m else ''}" for r, m in SYMMETRY_VARIANTS
]


# ── Tests ────────────────────────────────────────────────────────────────────


class TestLanes:
    """Lane-aware Rust throughput across the 8 symmetry variants."""

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_straight_chain_saturates_at_15(self, rot, mirror):
        """Source → 3 belts → Sink: both lanes saturate at 7.5 each → 15.0 total."""
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "iron_plate"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (2, 0, "transport_belt", Direction.EAST, None),
            (3, 0, "transport_belt", Direction.EAST, None),
            (4, 0, "bulk_inserter", Direction.EAST, "iron_plate"),
        ]
        world = build_world(5, layout, rot, mirror)
        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6)

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_lone_curve_preserves_both_lanes(self, rot, mirror):
        """Source → 2 belts → curve → 2 belts → Sink. Lone curve is lane-
        preserving — both belt lanes flow through the bend → 15.0 total.
        If the curve incorrectly side-loaded onto a single lane, total
        would drop to 7.5."""
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "iron_plate"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (2, 0, "transport_belt", Direction.EAST, None),
            (3, 0, "transport_belt", Direction.SOUTH, None),
            (3, 1, "transport_belt", Direction.SOUTH, None),
            (3, 2, "transport_belt", Direction.SOUTH, None),
            (3, 3, "bulk_inserter", Direction.SOUTH, "iron_plate"),
        ]
        world = build_world(5, layout, rot, mirror)
        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6)

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_t_junction_side_load(self, rot, mirror):
        """T-junction `>>v<<v K`: two perpendicular feeders side-load onto
        opposite lanes of the south-going belt. Each lane caps at 7.5 →
        15.0 total at sink."""
        layout = [
            # West-going feeder.
            (0, 0, "stack_inserter", Direction.EAST, "copper_cable"),
            (1, 0, "transport_belt", Direction.EAST, None),
            # The junction tile.
            (2, 0, "transport_belt", Direction.SOUTH, None),
            # East-going feeder (mirror of the first).
            (4, 0, "stack_inserter", Direction.WEST, "copper_cable"),
            (3, 0, "transport_belt", Direction.WEST, None),
            # Drain south.
            (2, 1, "transport_belt", Direction.SOUTH, None),
            (2, 2, "bulk_inserter", Direction.SOUTH, "copper_cable"),
        ]
        world = build_world(5, layout, rot, mirror)
        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6)

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_parallel_plus_perpendicular_junction(self, rot, mirror):
        """Junction where the dst has a parallel-back source AND a
        perpendicular feeder. The perpendicular detects the junction and
        side-loads; the parallel-back stays lane-preserving. Both lanes
        of the dst saturate → 15.0."""
        layout = [
            # Parallel-back chain (south-going).
            (2, 0, "stack_inserter", Direction.SOUTH, "copper_cable"),
            (2, 1, "transport_belt", Direction.SOUTH, None),
            # Junction.
            (2, 2, "transport_belt", Direction.SOUTH, None),
            # Perpendicular feeder (east-going) into the junction.
            (0, 2, "stack_inserter", Direction.EAST, "copper_cable"),
            (1, 2, "transport_belt", Direction.EAST, None),
            # Drain south.
            (2, 3, "bulk_inserter", Direction.SOUTH, "copper_cable"),
        ]
        world = build_world(5, layout, rot, mirror)
        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6)

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_inserter_in_line_drop_on_port_lane(self, rot, mirror):
        """Source → Inserter → Belt → Sink. Inserter drops in-line on the
        belt's PORT lane only → 0.86 i/s through the chain."""
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "iron_plate"),
            (1, 0, "inserter", Direction.EAST, None),
            (2, 0, "transport_belt", Direction.EAST, None),
            (3, 0, "bulk_inserter", Direction.EAST, "iron_plate"),
        ]
        world = build_world(5, layout, rot, mirror)
        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(0.86, abs=1e-6)

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_inserter_perpendicular_far_lane_drops_fill_both_lanes(
        self, rot, mirror
    ):
        """Two inserters drop on opposite sides of a single east-going
        belt. Each FAR-lane drop targets a different belt lane (one →
        port, the other → starboard), so the belt carries 0.86 + 0.86 =
        1.72 i/s through to the sink. If both inserters dropped on the
        same lane, total would be 0.86 (the other lane would stay
        empty). This isolates the FAR-vs-PORT lane choice in
        `inserter_drop_lane`."""
        layout = [
            # Two sources feed two inserters on opposite sides of (2, 2).
            (2, 0, "stack_inserter", Direction.SOUTH, "copper_cable"),
            (2, 1, "inserter", Direction.SOUTH, None),
            (2, 4, "stack_inserter", Direction.NORTH, "copper_cable"),
            (2, 3, "inserter", Direction.NORTH, None),
            # The east-going belt that receives the drops.
            (2, 2, "transport_belt", Direction.EAST, None),
            (3, 2, "transport_belt", Direction.EAST, None),
            (4, 2, "bulk_inserter", Direction.EAST, "copper_cable"),
        ]
        world = build_world(5, layout, rot, mirror)
        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(2 * 0.86, abs=1e-6)

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_underground_preserves_both_lanes(self, rot, mirror):
        """Source → belt → UG-down ... UG-up → belt → Sink. The tunnel
        is lane-preserving — both lanes survive the underground hop →
        15.0 throughput end-to-end."""
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "iron_plate"),
            (1, 0, "transport_belt", Direction.EAST, None),
        ]
        # Underground belts use Misc; can't pass via build_world's set_entity
        # since misc is not a layout field. Build the canonical world first
        # then layer on the tunnel + sink at canonical positions, applying
        # the same rotation/mirror.
        world = build_world(7, layout, rot, mirror)

        def place_canonical(x, y, kind, d, item="empty", misc=0):
            rx, ry = rotate_pos(x, y, 7, rot)
            rd = rotate_dir(d, rot)
            if mirror:
                rx, ry = mirror_pos_x(rx, ry, 7)
                rd = mirror_dir_x(rd)
            set_entity(world, rx, ry, kind, rd, item, misc)

        place_canonical(
            2, 0, "underground_belt", Direction.EAST, misc=Misc.UNDERGROUND_DOWN.value
        )
        place_canonical(
            4, 0, "underground_belt", Direction.EAST, misc=Misc.UNDERGROUND_UP.value
        )
        place_canonical(5, 0, "transport_belt", Direction.EAST)
        place_canonical(6, 0, "bulk_inserter", Direction.EAST, "iron_plate")

        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6)
