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
    def test_splitter_passes_lanes_through(self, rot, mirror):
        """Source → belt → splitter → belt → sink with one input + one
        output. Splitter pools per lane, the divisor is 1, the per-output-
        lane cap (7.5) is non-binding → 15.0 saturated total.

        The splitter is 2-wide perpendicular to its flow direction; its
        canonical-frame body tiles must be rotated/mirrored as a SET
        (not via the anchor alone) since `set_splitter` re-derives body
        tiles from anchor + direction and the resulting anchor depends
        on the post-rotation orientation.
        """
        from helpers import set_splitter

        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "iron_plate"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (3, 0, "transport_belt", Direction.EAST, None),
            (4, 0, "bulk_inserter", Direction.EAST, "iron_plate"),
        ]
        world = build_world(6, layout, rot, mirror)

        # Canonical splitter occupies (2,0) anchor + (2,1) body, direction East.
        # Rotate both cells, then pick the post-rotation anchor (min-x then min-y).
        canonical_tiles = [(2, 0), (2, 1)]
        rotated_tiles = []
        for tx, ty in canonical_tiles:
            rx, ry = rotate_pos(tx, ty, 6, rot)
            if mirror:
                rx, ry = mirror_pos_x(rx, ry, 6)
            rotated_tiles.append((rx, ry))
        anchor_x, anchor_y = min(rotated_tiles)
        rd = rotate_dir(Direction.EAST, rot)
        if mirror:
            rd = mirror_dir_x(rd)
        set_splitter(world, anchor_x, anchor_y, rd)

        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6)

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


# ── Snaking underground belt patterns ───────────────────────────────────────


class TestUndergroundPatterns:
    """Underground-belt edge cases: tunnel ranges, snaking pairs, and
    perpendicular feeds into a UG-down."""

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    @pytest.mark.parametrize("distance", [1, 2, 3, 4, 5])
    def test_single_ug_pair_at_each_valid_distance(self, rot, mirror, distance):
        """`S > d _..._ u > K` — Source → belt → UG-down → tunnel of
        `distance` underground cells → UG-up → belt → Sink. Within the
        UG range (1..=5) every distance saturates at 15.0."""
        # Layout in the canonical East frame:
        #   col 0: Source
        #   col 1: belt (so source's drop has a clean target)
        #   col 2: UG-down
        #   col 2 + distance: UG-up
        #   col 2 + distance + 1: belt
        #   col 2 + distance + 2: Sink
        size = max(8, 5 + distance)
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "copper_cable"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (2 + distance + 1, 0, "transport_belt", Direction.EAST, None),
            (2 + distance + 2, 0, "bulk_inserter", Direction.EAST, "copper_cable"),
        ]
        world = build_world(size, layout, rot, mirror)

        # Place the UG pair after rotation/mirror.
        def place_ug(x, y, d, misc):
            rx, ry = rotate_pos(x, y, size, rot)
            rd = rotate_dir(d, rot)
            if mirror:
                rx, ry = mirror_pos_x(rx, ry, size)
                rd = mirror_dir_x(rd)
            set_entity(world, rx, ry, "underground_belt", rd, misc=misc.value)

        place_ug(2, 0, Direction.EAST, Misc.UNDERGROUND_DOWN)
        place_ug(2 + distance, 0, Direction.EAST, Misc.UNDERGROUND_UP)

        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6), (
            f"distance={distance} expected 15.0, got {tp}"
        )

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_ug_pair_beyond_max_range_does_not_connect(self, rot, mirror):
        """Distance 6 is past the UG range (max 5) — no tunnel pair
        forms, so the source can't reach the sink. Throughput = 0."""
        size = 12
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "copper_cable"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (10, 0, "transport_belt", Direction.EAST, None),
            (11, 0, "bulk_inserter", Direction.EAST, "copper_cable"),
        ]
        world = build_world(size, layout, rot, mirror)

        def place_ug(x, y, d, misc):
            rx, ry = rotate_pos(x, y, size, rot)
            rd = rotate_dir(d, rot)
            if mirror:
                rx, ry = mirror_pos_x(rx, ry, size)
                rd = mirror_dir_x(rd)
            set_entity(world, rx, ry, "underground_belt", rd, misc=misc.value)

        # UG-down at col 2, UG-up at col 8 → distance 6 (out of range).
        place_ug(2, 0, Direction.EAST, Misc.UNDERGROUND_DOWN)
        place_ug(8, 0, Direction.EAST, Misc.UNDERGROUND_UP)

        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_chained_ug_pairs_with_belt_between(self, rot, mirror):
        """Two UG pairs back-to-back with a TB between the first UG-up
        and the second UG-down. The TB picks up the first UG-up's
        output (via TB's backward scan) and forwards into the second
        UG-down. Saturated → 15.0.

        Layout:
          col 0: S, 1: belt, 2: d, 3-4: empty, 5: u, 6: belt, 7: d,
          8-9: empty, 10: u, 11: belt, 12: K.
        """
        size = 14
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "copper_cable"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (6, 0, "transport_belt", Direction.EAST, None),
            (11, 0, "transport_belt", Direction.EAST, None),
            (12, 0, "bulk_inserter", Direction.EAST, "copper_cable"),
        ]
        world = build_world(size, layout, rot, mirror)

        def place_ug(x, y, d, misc):
            rx, ry = rotate_pos(x, y, size, rot)
            rd = rotate_dir(d, rot)
            if mirror:
                rx, ry = mirror_pos_x(rx, ry, size)
                rd = mirror_dir_x(rd)
            set_entity(world, rx, ry, "underground_belt", rd, misc=misc.value)

        place_ug(2, 0, Direction.EAST, Misc.UNDERGROUND_DOWN)
        place_ug(5, 0, Direction.EAST, Misc.UNDERGROUND_UP)
        place_ug(7, 0, Direction.EAST, Misc.UNDERGROUND_DOWN)
        place_ug(10, 0, Direction.EAST, Misc.UNDERGROUND_UP)

        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6), (
            f"chained UG pairs with belt between: expected 15.0, got {tp}"
        )

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_side_load_perpendicular_into_ug_down(self, rot, mirror):
        """East belts forward-feed a south-going UG-down (perpendicular).
        UG-down tunnels south to UG-up, which feeds a south belt to the
        sink. Lone curve into the UG-down (no other source) → lane-
        preserving → 15.0 throughput.

        Diagram (canonical, before rotation/mirror):
              col 0  col 1  col 2
          row 0:  S>     >>     d
          row 1:                .
          row 2:                u
          row 3:                v
          row 4:                K
        """
        size = 6
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "copper_cable"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (2, 3, "transport_belt", Direction.SOUTH, None),
            (2, 4, "bulk_inserter", Direction.SOUTH, "copper_cable"),
        ]
        world = build_world(size, layout, rot, mirror)

        def place_ug(x, y, d, misc):
            rx, ry = rotate_pos(x, y, size, rot)
            rd = rotate_dir(d, rot)
            if mirror:
                rx, ry = mirror_pos_x(rx, ry, size)
                rd = mirror_dir_x(rd)
            set_entity(world, rx, ry, "underground_belt", rd, misc=misc.value)

        # UG-down (south) at (2,0) — receives the perpendicular feed from
        # the east belt at (1,0). UG-up (south) at (2,2) is its tunnel exit.
        place_ug(2, 0, Direction.SOUTH, Misc.UNDERGROUND_DOWN)
        place_ug(2, 2, Direction.SOUTH, Misc.UNDERGROUND_UP)

        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(15.0, abs=1e-6), (
            f"perpendicular feed into UG-down: expected 15.0, got {tp}"
        )

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_adjacent_ug_pairs_no_belt_between(self, rot, mirror):
        """`S d u d u K` — no TB between the first UG-up and the second
        UG-down (they sit at adjacent tiles). The current implementation
        does NOT wire UG-up directly to UG-down (only TB::connections
        scans backward for UG-up sources, and UG-down has no backward
        scan), so this layout breaks the chain and throughput = 0.

        FLAGGED FOR USER: should UG-up forward-feed an adjacent UG-down?
        Pinning the current behaviour here so the answer is explicit;
        update the expected if/when we fix it.
        """
        size = 8
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "copper_cable"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (6, 0, "transport_belt", Direction.EAST, None),
            (7, 0, "bulk_inserter", Direction.EAST, "copper_cable"),
        ]
        world = build_world(size, layout, rot, mirror)

        def place_ug(x, y, d, misc):
            rx, ry = rotate_pos(x, y, size, rot)
            rd = rotate_dir(d, rot)
            if mirror:
                rx, ry = mirror_pos_x(rx, ry, size)
                rd = mirror_dir_x(rd)
            set_entity(world, rx, ry, "underground_belt", rd, misc=misc.value)

        # `Sd u d u K` (cols 2,3,4,5 are UG d/u/d/u with no TB between).
        place_ug(2, 0, Direction.EAST, Misc.UNDERGROUND_DOWN)
        place_ug(3, 0, Direction.EAST, Misc.UNDERGROUND_UP)
        place_ug(4, 0, Direction.EAST, Misc.UNDERGROUND_DOWN)
        place_ug(5, 0, Direction.EAST, Misc.UNDERGROUND_UP)

        tp, _ = rs_throughput(world)
        # Current behaviour: UG-up → UG-down has no edge in the graph.
        assert tp == pytest.approx(0.0, abs=1e-6), (
            f"adjacent UG pairs (no TB between): current behaviour expects 0, got {tp}"
        )

    @pytest.mark.parametrize("rot,mirror", SYMMETRY_VARIANTS, ids=SYMMETRY_IDS)
    def test_inserter_lifts_between_ug_pairs(self, rot, mirror):
        """`S d u I d u K` — inserter bridges from the first UG-up to
        the second UG-down. Inserter caps at 0.86 i/s → bottleneck.
        Verifies that an inserter can pick up from a UG-up (lane-aware
        source: dual pickup edges combining both lanes into the
        inserter) and drop on a UG-down (in-line drop on PORT lane).
        """
        size = 9
        # cols: 0:S, 1:belt, 2:d, 3:u, 4:I (faces east), 5:d, 6:u, 7:belt, 8:K
        layout = [
            (0, 0, "stack_inserter", Direction.EAST, "copper_cable"),
            (1, 0, "transport_belt", Direction.EAST, None),
            (4, 0, "inserter", Direction.EAST, None),
            (7, 0, "transport_belt", Direction.EAST, None),
            (8, 0, "bulk_inserter", Direction.EAST, "copper_cable"),
        ]
        world = build_world(size, layout, rot, mirror)

        def place_ug(x, y, d, misc):
            rx, ry = rotate_pos(x, y, size, rot)
            rd = rotate_dir(d, rot)
            if mirror:
                rx, ry = mirror_pos_x(rx, ry, size)
                rd = mirror_dir_x(rd)
            set_entity(world, rx, ry, "underground_belt", rd, misc=misc.value)

        place_ug(2, 0, Direction.EAST, Misc.UNDERGROUND_DOWN)
        place_ug(3, 0, Direction.EAST, Misc.UNDERGROUND_UP)
        place_ug(5, 0, Direction.EAST, Misc.UNDERGROUND_DOWN)
        place_ug(6, 0, Direction.EAST, Misc.UNDERGROUND_UP)

        tp, _ = rs_throughput(world)
        assert tp == pytest.approx(0.86, abs=1e-6), (
            f"inserter bridge between UG pairs: expected 0.86, got {tp}"
        )
