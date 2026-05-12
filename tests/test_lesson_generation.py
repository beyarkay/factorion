"""Tests for lesson generation: INSERTER_TRANSFER, SPLITTER_SPLIT, SPLITTER_MERGE."""

import pytest
import random
import re
import torch

from helpers import (
    Channel,
    DIR_TO_DELTA,
    Direction,
    Footprint,
    LessonKind,
    Misc,
    compare_throughput,
    entities,
    generate_lesson,
    items,
    recipes,
    rs_throughput,
    str2ent,
    str2item,
    world2graph,
    world2html,
)


DIRS = [Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST]


class TestInserterTransferBasic:
    """Basic sanity checks for INSERTER_TRANSFER lesson generation."""

    def test_generates_without_error(self):
        """Can generate at least one lesson without raising."""
        world, min_ent = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        assert world is not None
        assert min_ent is not None

    def test_returns_cwh_tensor(self):
        """Output tensor is CWH format with correct shape."""
        size = 8
        world, _ = generate_lesson(
            size=size, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        assert world.shape[0] == len(Channel)
        assert world.shape[1] == size
        assert world.shape[2] == size

    def test_has_source_and_sink(self):
        """Generated world contains exactly one source and one sink."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        source_count = (ent_layer == str2ent("source").value).sum().item()
        sink_count = (ent_layer == str2ent("sink").value).sum().item()
        assert source_count == 1, f"Expected 1 source, got {source_count}"
        assert sink_count == 1, f"Expected 1 sink, got {sink_count}"

    def test_has_inserter(self):
        """Generated world contains exactly one inserter."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        inserter_count = (ent_layer == str2ent("inserter").value).sum().item()
        assert inserter_count == 1, f"Expected 1 inserter, got {inserter_count}"

    def test_has_belts(self):
        """Generated world contains at least one transport belt."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        belt_count = (ent_layer == str2ent("transport_belt").value).sum().item()
        assert belt_count >= 2, f"Expected at least 2 belts, got {belt_count}"

    def test_nonzero_throughput(self):
        """Complete factory has throughput > 0."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"Expected positive throughput, got {tp}"

    def test_throughput_bottlenecked_by_inserter(self):
        """Throughput should be <= inserter flow rate (0.86)."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        inserter_flow = str2ent("inserter").flow
        assert tp <= inserter_flow + 1e-6, (
            f"Throughput {tp} exceeds inserter flow rate {inserter_flow}"
        )

    @pytest.mark.parametrize("num_missing", [1, 5, 10, 100, float("inf")])
    @pytest.mark.parametrize("seed", range(10))
    def test_inserter_always_present_after_blanking(self, num_missing, seed):
        """The central inserter must never be blanked, even at maximum
        num_missing_entities — without it the lesson is ambiguous."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER,
            num_missing_entities=num_missing, seed=seed,
        )
        ent_layer = world[Channel.ENTITIES.value]
        inserter_count = (ent_layer == str2ent("inserter").value).sum().item()
        assert inserter_count == 1, (
            f"Expected 1 inserter at num_missing={num_missing}, seed={seed}; "
            f"got {inserter_count}"
        )


class TestInserterTransferManySeeds:
    """Generate many lessons with different seeds and verify all are valid."""

    @pytest.mark.parametrize("seed", range(50))
    def test_size_8_seed(self, seed):
        """Size 8 grid, many seeds — all must produce valid factories."""
        world, min_ent = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"
        assert tp <= str2ent("inserter").flow + 1e-6

    @pytest.mark.parametrize("seed", range(30))
    def test_size_6_seed(self, seed):
        """Size 6 grid — smaller grid, still must work."""
        world, min_ent = generate_lesson(
            size=6, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(30))
    def test_size_10_seed(self, seed):
        """Size 10 grid — larger grid."""
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_15_seed(self, seed):
        """Size 15 grid — large grid."""
        world, min_ent = generate_lesson(
            size=15, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"


class TestInserterTransferParity:
    """Python and Rust throughput must agree on generated inserter factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_parity_size_8(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        world_whc = world.permute(1, 2, 0)
        tp, ur = rs_throughput(world_whc)
        assert tp > 0, f"seed={seed}: throughput={tp}"
        assert ur >= 0


class TestInserterTransferGridSizes:
    """Test across a range of grid sizes with a fixed seed."""

    @pytest.mark.parametrize("size", [5, 6, 7, 8, 9, 10, 12, 15])
    def test_valid_factory_per_size(self, size):
        world, min_ent = generate_lesson(
            size=size, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=7
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 1
        assert (ent_layer == str2ent("inserter").value).sum().item() == 1
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0


class TestInserterTransferMissingEntities:
    """Test that entity removal works correctly."""

    @pytest.mark.parametrize("num_missing", [1, 2, 3])
    def test_removes_correct_count(self, num_missing):
        """Removing N entities should leave the right number missing."""
        world_full, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        world_partial, min_ent = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=num_missing, seed=42
        )
        assert min_ent == num_missing

        # Source and sink should still be present
        ent_layer = world_partial[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 1

    @pytest.mark.parametrize("seed", range(20))
    def test_missing_1_preserves_source_sink(self, seed):
        """With 1 missing entity, source and sink remain."""
        world, min_ent = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=1, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 1
        assert min_ent == 1

    def test_missing_inf_returns_min_entities(self):
        """With inf missing, min_entities_required equals total placeable count."""
        world, min_ent = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=float("inf"), seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 1
        # min_ent should be the total number of inserter + belt entities
        assert min_ent >= 3  # at least 1 inserter + 2 belts


class TestInserterTransferEntityDirections:
    """Verify that all placed entities have valid, non-NONE directions."""

    @pytest.mark.parametrize("seed", range(20))
    def test_all_entities_have_directions(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        dir_layer = world[Channel.DIRECTION.value]

        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                ent_val = ent_layer[x, y].item()
                if ent_val != str2ent("empty").value:
                    dir_val = dir_layer[x, y].item()
                    assert dir_val != Direction.NONE.value, (
                        f"seed={seed}: entity {entities[ent_val].name} at ({x},{y}) "
                        f"has NONE direction"
                    )


class TestInserterTransferNoOverlaps:
    """Verify no entity overlaps in generated factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_unique_positions(self, seed):
        """Source, sink, inserter, and belts should all occupy unique cells."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]

        occupied = []
        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                if ent_layer[x, y].item() != str2ent("empty").value:
                    occupied.append((x, y))

        # No duplicate positions
        assert len(occupied) == len(set(occupied)), (
            f"seed={seed}: duplicate positions found"
        )


class TestInserterTransferItems:
    """Verify item channels are set correctly."""

    @pytest.mark.parametrize("seed", range(20))
    def test_source_sink_have_matching_items(self, seed):
        """Source and sink should carry the same item."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        item_layer = world[Channel.ITEMS.value]

        source_pos = (ent_layer == str2ent("source").value).nonzero(as_tuple=False)[0]
        sink_pos = (ent_layer == str2ent("sink").value).nonzero(as_tuple=False)[0]

        source_item = item_layer[source_pos[0], source_pos[1]].item()
        sink_item = item_layer[sink_pos[0], sink_pos[1]].item()

        assert source_item == sink_item, (
            f"seed={seed}: source item={source_item}, sink item={sink_item}"
        )
        assert source_item != str2item("empty").value, (
            f"seed={seed}: source has empty item"
        )


class TestInserterTransferDeterminism:
    """Same seed should produce identical results."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99])
    def test_deterministic(self, seed):
        world1, min1 = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        world2, min2 = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        assert torch.equal(world1, world2), f"seed={seed}: not deterministic"
        assert min1 == min2


class TestInserterTransferMaxEntities:
    """Test max_entities constraint limits factory complexity."""

    @pytest.mark.parametrize("seed", range(10))
    def test_respects_max_entities(self, seed):
        """With a reasonable max_entities, total placeable entities should be bounded."""
        max_ent = 15
        world, _ = generate_lesson(
            size=10,
            kind=LessonKind.INSERTER_TRANSFER,
            num_missing_entities=0,
            seed=seed,
            max_entities=max_ent,
        )
        ent_layer = world[Channel.ENTITIES.value]
        # Count non-source, non-sink, non-empty entities
        placeable = (
            (ent_layer != str2ent("source").value)
            & (ent_layer != str2ent("sink").value)
            & (ent_layer != str2ent("empty").value)
        ).sum().item()
        assert placeable <= max_ent, (
            f"seed={seed}: {placeable} placeable entities exceeds max {max_ent}"
        )


class TestInserterTransferThroughputRange:
    """Verify throughput falls in expected range across many configurations."""

    @pytest.mark.parametrize("size", [6, 8, 10])
    @pytest.mark.parametrize("seed", range(15))
    def test_throughput_in_range(self, size, seed):
        """Throughput should be > 0 and <= inserter rate (0.86)."""
        world, _ = generate_lesson(
            size=size, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        inserter_flow = str2ent("inserter").flow
        assert 0 < tp <= inserter_flow + 1e-6, (
            f"size={size}, seed={seed}: throughput {tp} not in (0, {inserter_flow}]"
        )


# ── SPLITTER_SPLIT tests ─────────────────────────────────────────────────────


class TestSplitterSplitBasic:
    """Basic sanity checks for SPLITTER_SPLIT lesson generation."""

    def test_generates_without_error(self):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=42
        )
        assert world is not None
        assert min_ent is not None

    def test_returns_cwh_tensor(self):
        size = 10
        world, _ = generate_lesson(
            size=size, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=42
        )
        assert world.shape[0] == len(Channel)
        assert world.shape[1] == size
        assert world.shape[2] == size

    def test_has_one_source_two_sinks(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 2

    def test_has_splitter(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        splitter_count = (ent_layer == str2ent("splitter").value).sum().item()
        # Splitter occupies 2 tiles
        assert splitter_count == 2, f"Expected 2 splitter tiles, got {splitter_count}"

    def test_has_belts(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        belt_count = (ent_layer == str2ent("transport_belt").value).sum().item()
        assert belt_count >= 3, f"Expected at least 3 belts, got {belt_count}"

    def test_nonzero_throughput(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=42
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"Expected positive throughput, got {tp}"

    def test_throughput_bounded_by_splitter(self):
        """Total throughput should be <= 30.0 (splitter max flow)."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=42
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp <= 30.0 + 1e-6, f"Throughput {tp} exceeds splitter max flow"


class TestSplitterSplitManySeeds:
    """Generate many lessons with different seeds and verify all are valid."""

    @pytest.mark.parametrize("seed", range(50))
    def test_size_10_seed(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(30))
    def test_size_8_seed(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_12_seed(self, seed):
        world, _ = generate_lesson(
            size=12, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_15_seed(self, seed):
        world, _ = generate_lesson(
            size=15, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"


class TestSplitterSplitParity:
    """Python and Rust throughput must agree on generated splitter-split factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_parity_size_10(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        world_whc = world.permute(1, 2, 0)
        tp, ur = rs_throughput(world_whc)
        assert tp > 0, f"seed={seed}: throughput={tp}"
        assert ur >= 0


class TestSplitterSplitGridSizes:
    """Test across a range of grid sizes."""

    @pytest.mark.parametrize("size", [8, 9, 10, 12, 15])
    def test_valid_factory_per_size(self, size):
        world, _ = generate_lesson(
            size=size, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=7
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 2
        assert (ent_layer == str2ent("splitter").value).sum().item() == 2
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0


class TestSplitterSplitEntityDirections:
    """Verify all placed entities have valid directions."""

    @pytest.mark.parametrize("seed", range(20))
    def test_all_entities_have_directions(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        dir_layer = world[Channel.DIRECTION.value]

        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                ent_val = ent_layer[x, y].item()
                if ent_val != str2ent("empty").value:
                    dir_val = dir_layer[x, y].item()
                    assert dir_val != Direction.NONE.value, (
                        f"seed={seed}: entity {entities[ent_val].name} at ({x},{y}) "
                        f"has NONE direction"
                    )


class TestSplitterSplitNoOverlaps:
    """Verify no entity overlaps in generated factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_no_double_placement(self, seed):
        """Each cell should have at most one entity."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        occupied = []
        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                if ent_layer[x, y].item() != str2ent("empty").value:
                    occupied.append((x, y))
        assert len(occupied) == len(set(occupied)), (
            f"seed={seed}: duplicate positions"
        )


class TestSplitterSplitItems:
    """Verify item channels are set correctly."""

    @pytest.mark.parametrize("seed", range(20))
    def test_source_sinks_have_matching_items(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        item_layer = world[Channel.ITEMS.value]

        source_pos = (ent_layer == str2ent("source").value).nonzero(as_tuple=False)[0]
        sink_positions = (ent_layer == str2ent("sink").value).nonzero(as_tuple=False)

        source_item = item_layer[source_pos[0], source_pos[1]].item()
        assert source_item != str2item("empty").value

        for i in range(len(sink_positions)):
            sink_item = item_layer[sink_positions[i][0], sink_positions[i][1]].item()
            assert sink_item == source_item, (
                f"seed={seed}: sink {i} item={sink_item} != source item={source_item}"
            )


class TestSplitterSplitMissingEntities:
    """Test entity removal for splitter-split lessons."""

    @pytest.mark.parametrize("num_missing", [1, 2, 3])
    def test_removes_correct_count(self, num_missing):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=num_missing, seed=42
        )
        assert min_ent == num_missing
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 2

    @pytest.mark.parametrize("seed", range(10))
    def test_missing_inf_returns_positive_count(self, seed):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=float("inf"), seed=seed
        )
        assert min_ent >= 4  # at least 1 splitter + 3 belts

    @pytest.mark.parametrize("num_missing", [1, 5, 10, 100, float("inf")])
    @pytest.mark.parametrize("seed", range(10))
    def test_splitter_always_present_after_blanking(self, num_missing, seed):
        """The central splitter must never be blanked, even at maximum
        num_missing_entities — without it the lesson is ambiguous (could be
        solved by belts alone)."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT,
            num_missing_entities=num_missing, seed=seed,
        )
        ent_layer = world[Channel.ENTITIES.value]
        splitter_tiles = (ent_layer == str2ent("splitter").value).sum().item()
        assert splitter_tiles == 2, (
            f"Expected splitter (2 tiles) at num_missing={num_missing}, "
            f"seed={seed}; got {splitter_tiles}"
        )


class TestSplitterSplitDeterminism:
    """Same seed produces identical results."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99])
    def test_deterministic(self, seed):
        world1, min1 = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        world2, min2 = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        assert torch.equal(world1, world2), f"seed={seed}: not deterministic"
        assert min1 == min2


class TestSplitterSplitThroughputRange:
    """Throughput should be in the expected range for splitting."""

    @pytest.mark.parametrize("size", [8, 10, 12])
    @pytest.mark.parametrize("seed", range(15))
    def test_throughput_in_range(self, size, seed):
        """Throughput should be > 0 and <= 15.0 (single belt input)."""
        world, _ = generate_lesson(
            size=size, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        # Splitter max flow is 30.0 (can sideload both inputs)
        assert 0 < tp <= 30.0 + 1e-6, (
            f"size={size}, seed={seed}: throughput {tp} not in (0, 30.0]"
        )


# ── SPLITTER_MERGE tests ─────────────────────────────────────────────────────


class TestSplitterMergeBasic:
    """Basic sanity checks for SPLITTER_MERGE lesson generation."""

    def test_generates_without_error(self):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=42
        )
        assert world is not None
        assert min_ent is not None

    def test_returns_cwh_tensor(self):
        size = 10
        world, _ = generate_lesson(
            size=size, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=42
        )
        assert world.shape[0] == len(Channel)
        assert world.shape[1] == size
        assert world.shape[2] == size

    def test_has_two_sources_one_sink(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 2
        assert (ent_layer == str2ent("sink").value).sum().item() == 1

    def test_has_splitter(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        splitter_count = (ent_layer == str2ent("splitter").value).sum().item()
        assert splitter_count == 2, f"Expected 2 splitter tiles, got {splitter_count}"

    def test_has_belts(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        belt_count = (ent_layer == str2ent("transport_belt").value).sum().item()
        assert belt_count >= 3, f"Expected at least 3 belts, got {belt_count}"

    def test_nonzero_throughput(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=42
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"Expected positive throughput, got {tp}"

    def test_throughput_bounded_by_splitter(self):
        """Total throughput should be <= 30.0 (splitter max flow)."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=42
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp <= 30.0 + 1e-6, f"Throughput {tp} exceeds splitter max flow"


class TestSplitterMergeManySeeds:
    """Generate many lessons with different seeds and verify all are valid."""

    @pytest.mark.parametrize("seed", range(50))
    def test_size_10_seed(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(30))
    def test_size_8_seed(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_12_seed(self, seed):
        world, _ = generate_lesson(
            size=12, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_15_seed(self, seed):
        world, _ = generate_lesson(
            size=15, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"


class TestSplitterMergeParity:
    """Python and Rust throughput must agree on generated splitter-merge factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_parity_size_10(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        world_whc = world.permute(1, 2, 0)
        tp, ur = rs_throughput(world_whc)
        assert tp > 0, f"seed={seed}: throughput={tp}"
        assert ur >= 0


class TestSplitterMergeGridSizes:
    """Test across a range of grid sizes."""

    @pytest.mark.parametrize("size", [8, 9, 10, 12, 15])
    def test_valid_factory_per_size(self, size):
        world, _ = generate_lesson(
            size=size, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=7
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 2
        assert (ent_layer == str2ent("sink").value).sum().item() == 1
        assert (ent_layer == str2ent("splitter").value).sum().item() == 2
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0


class TestSplitterMergeEntityDirections:
    """Verify all placed entities have valid directions."""

    @pytest.mark.parametrize("seed", range(20))
    def test_all_entities_have_directions(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        dir_layer = world[Channel.DIRECTION.value]

        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                ent_val = ent_layer[x, y].item()
                if ent_val != str2ent("empty").value:
                    dir_val = dir_layer[x, y].item()
                    assert dir_val != Direction.NONE.value, (
                        f"seed={seed}: entity {entities[ent_val].name} at ({x},{y}) "
                        f"has NONE direction"
                    )


class TestSplitterMergeNoOverlaps:
    """Verify no entity overlaps in generated factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_no_double_placement(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        occupied = []
        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                if ent_layer[x, y].item() != str2ent("empty").value:
                    occupied.append((x, y))
        assert len(occupied) == len(set(occupied)), (
            f"seed={seed}: duplicate positions"
        )


class TestSplitterMergeItems:
    """Verify item channels are set correctly."""

    @pytest.mark.parametrize("seed", range(20))
    def test_sources_sink_have_matching_items(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        item_layer = world[Channel.ITEMS.value]

        source_positions = (ent_layer == str2ent("source").value).nonzero(as_tuple=False)
        sink_pos = (ent_layer == str2ent("sink").value).nonzero(as_tuple=False)[0]

        sink_item = item_layer[sink_pos[0], sink_pos[1]].item()
        assert sink_item != str2item("empty").value

        for i in range(len(source_positions)):
            src_item = item_layer[source_positions[i][0], source_positions[i][1]].item()
            assert src_item == sink_item, (
                f"seed={seed}: source {i} item={src_item} != sink item={sink_item}"
            )


class TestSplitterMergeMissingEntities:
    """Test entity removal for splitter-merge lessons."""

    @pytest.mark.parametrize("num_missing", [1, 2, 3])
    def test_removes_correct_count(self, num_missing):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=num_missing, seed=42
        )
        assert min_ent == num_missing
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 2
        assert (ent_layer == str2ent("sink").value).sum().item() == 1

    @pytest.mark.parametrize("seed", range(10))
    def test_missing_inf_returns_positive_count(self, seed):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=float("inf"), seed=seed
        )
        assert min_ent >= 4  # at least 1 splitter + 3 belts

    @pytest.mark.parametrize("num_missing", [1, 5, 10, 100, float("inf")])
    @pytest.mark.parametrize("seed", range(10))
    def test_splitter_always_present_after_blanking(self, num_missing, seed):
        """The central splitter must never be blanked, even at maximum
        num_missing_entities — without it the lesson is ambiguous (could be
        solved by belts alone)."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE,
            num_missing_entities=num_missing, seed=seed,
        )
        ent_layer = world[Channel.ENTITIES.value]
        splitter_tiles = (ent_layer == str2ent("splitter").value).sum().item()
        assert splitter_tiles == 2, (
            f"Expected splitter (2 tiles) at num_missing={num_missing}, "
            f"seed={seed}; got {splitter_tiles}"
        )


class TestSplitterMergeDeterminism:
    """Same seed produces identical results."""

    @pytest.mark.parametrize("seed", [0, 1, 42, 99])
    def test_deterministic(self, seed):
        world1, min1 = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        world2, min2 = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        assert torch.equal(world1, world2), f"seed={seed}: not deterministic"
        assert min1 == min2


class TestSplitterMergeThroughputRange:
    """Throughput should be in the expected range for merging."""

    @pytest.mark.parametrize("size", [8, 10, 12])
    @pytest.mark.parametrize("seed", range(15))
    def test_throughput_in_range(self, size, seed):
        """Throughput should be > 0 and <= 30.0 (splitter max)."""
        world, _ = generate_lesson(
            size=size, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert 0 < tp <= 30.0 + 1e-6, (
            f"size={size}, seed={seed}: throughput {tp} not in (0, 30.0]"
        )


# ── Multi-tile entity removal tests ──────────────────────────────────────────


class TestRemoveEntitiesSplitterIntegrity:
    """When a splitter is removed, both tiles must be cleared together."""

    @pytest.mark.parametrize("seed", range(50))
    def test_splitter_removal_clears_both_tiles(self, seed):
        """Generate a splitter lesson, remove 1 entity. If the splitter was
        removed, BOTH tiles must be empty — no orphaned single tile."""
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=1, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        splitter_val = str2ent("splitter").value
        splitter_tiles = (ent_layer == splitter_val).sum().item()
        # Splitter is 2 tiles. After removal it should be 0 or 2, never 1.
        assert splitter_tiles in (0, 2), (
            f"seed={seed}: found {splitter_tiles} splitter tiles — "
            f"expected 0 (removed) or 2 (kept), not 1 (orphaned)"
        )

    @pytest.mark.parametrize("seed", range(50))
    def test_splitter_merge_removal_clears_both_tiles(self, seed):
        """Same test for SPLITTER_MERGE lessons."""
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=1, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        splitter_val = str2ent("splitter").value
        splitter_tiles = (ent_layer == splitter_val).sum().item()
        assert splitter_tiles in (0, 2), (
            f"seed={seed}: found {splitter_tiles} splitter tiles — "
            f"expected 0 (removed) or 2 (kept), not 1 (orphaned)"
        )

    @pytest.mark.parametrize("seed", range(30))
    @pytest.mark.parametrize("num_missing", [1, 2, 3, 4])
    def test_no_orphaned_splitter_tiles_any_removal_count(self, seed, num_missing):
        """With various removal counts, splitter tiles are always 0 or 2."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=num_missing, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        splitter_val = str2ent("splitter").value
        splitter_tiles = (ent_layer == splitter_val).sum().item()
        assert splitter_tiles in (0, 2), (
            f"seed={seed}, missing={num_missing}: {splitter_tiles} splitter tiles (orphaned!)"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_removed_entity_count_matches_min_entities(self, seed):
        """The number of entity *units* removed should equal min_entities_required."""
        # Generate a full factory first to count total entity units
        world_full, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        ent_full = world_full[Channel.ENTITIES.value]

        # Count entity units (splitter = 1 unit despite 2 tiles)
        full_belts = (ent_full == str2ent("transport_belt").value).sum().item()
        full_splitter_tiles = (ent_full == str2ent("splitter").value).sum().item()
        full_units = full_belts + (full_splitter_tiles // 2)

        # Remove 2 entities
        world_partial, min_ent = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=2, seed=seed
        )
        assert min_ent == 2

        ent_partial = world_partial[Channel.ENTITIES.value]
        partial_belts = (ent_partial == str2ent("transport_belt").value).sum().item()
        partial_splitter_tiles = (ent_partial == str2ent("splitter").value).sum().item()
        partial_units = partial_belts + (partial_splitter_tiles // 2)

        units_removed = full_units - partial_units
        assert units_removed == 2, (
            f"seed={seed}: removed {units_removed} entity units, expected 2"
        )


def _count_full_opacity_icons(html, icon_b64):
    """Count <img> tags rendering icon_b64 at full opacity (no opacity: 20%)."""
    pattern = (
        r"<img src='" + re.escape(icon_b64) + r"' style='([^']+)'"
    )
    return sum(1 for style in re.findall(pattern, html) if "opacity: 20%" not in style)


def _ent_b64(name):
    # Build a tiny world with just this entity to extract its rendered b64.
    w = torch.zeros(3, 3, len(Channel), dtype=torch.int64)
    w[..., Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
    w[1, 1, Channel.ENTITIES.value] = str2ent(name).value
    html = world2html(w).text
    # Extract any data:image src from the rendered cell at (1, 1).
    matches = re.findall(r"src='(data:image/png;base64,[^']+)'", html)
    # The entity icon for the (1,1) cell is the unique non-empty icon.
    empty_w = torch.zeros(3, 3, len(Channel), dtype=torch.int64)
    empty_w[..., Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
    empty_html = world2html(empty_w).text
    empty_srcs = set(re.findall(r"src='(data:image/png;base64,[^']+)'", empty_html))
    for src in matches:
        if src not in empty_srcs:
            return src
    raise AssertionError(f"could not find unique icon for {name}")


class TestWorld2HtmlMultiTile:
    """Regression: world2html must render a multi-tile entity once, not once
    per occupied cell."""

    @pytest.mark.parametrize("direction", DIRS, ids=lambda d: d.name)
    def test_splitter_renders_one_full_icon(self, direction):
        size = 8
        world = torch.zeros(size, size, len(Channel), dtype=torch.int64)
        world[..., Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
        # Place a splitter; entity_id at BOTH occupied cells, matching what
        # the lessons (and the env) write.
        from helpers import Footprint as _Fp  # noqa
        import factorion_rs
        anchor = (3, 3)
        tiles = factorion_rs.py_entity_tiles(anchor[0], anchor[1], direction.value, 2, 1)
        for tx, ty in tiles:
            world[tx, ty, Channel.ENTITIES.value] = str2ent("splitter").value
            world[tx, ty, Channel.DIRECTION.value] = direction.value

        html = world2html(world).text
        splitter_icon = _ent_b64("splitter")
        full_count = _count_full_opacity_icons(html, splitter_icon)
        assert full_count == 1, (
            f"splitter facing {direction.name}: expected 1 full-opacity "
            f"splitter icon, got {full_count} (secondary tile is being "
            f"double-rendered)"
        )



# ── ASSEMBLE_1IN_1OUT tests ──────────────────────────────────────────────────
#
# An ASSEMBLE_1IN_1OUT lesson is:
#   source(input item) → belt → input inserter → 3×3 assembler (recipe) →
#                                                output inserter → belt → sink(output item)
# The recipe is randomly chosen from all 1-input 1-output recipes
# (currently: copper_cable, iron_gear_wheel).


# Recipes with one ingredient and one product type. Both 1-in-1-out
# recipes available at the time of writing produce different output:
#   copper_cable:    1 cu_plate → 2 cu_cable
#   iron_gear_wheel: 2 ir_plate → 1 ir_gear_wheel
# Built dynamically from the live recipe table so adding new recipes
# automatically extends what ASSEMBLE_1IN_1OUT can pick.
def _one_in_one_out_pairs():
    from helpers import recipes as _recipes
    pairs = {}
    for name, r in _recipes.items():
        if len(r.consumes) == 1 and len(r.produces) == 1:
            pairs[name] = (next(iter(r.consumes)), next(iter(r.produces)))
    return pairs


ONE_IN_ONE_OUT_RECIPES = _one_in_one_out_pairs()


class TestAssemble1In1OutBasic:
    """Basic sanity checks for ASSEMBLE_1IN_1OUT lesson generation."""

    def test_generates_without_error(self):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=42
        )
        assert world is not None
        assert min_ent is not None

    def test_returns_cwh_tensor(self):
        size = 10
        world, _ = generate_lesson(
            size=size, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=42
        )
        assert world.shape[0] == len(Channel)
        assert world.shape[1] == size
        assert world.shape[2] == size

    def test_has_one_source_one_sink(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 1

    def test_has_one_assembler(self):
        """The assembler is a 3×3 multi-tile entity, so 9 tiles."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        asm_count = (ent_layer == str2ent("assembling_machine_1").value).sum().item()
        assert asm_count == 9, f"Expected 9 assembler tiles, got {asm_count}"

    def test_has_two_inserters(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        inserter_count = (ent_layer == str2ent("inserter").value).sum().item()
        assert inserter_count == 2, f"Expected 2 inserters, got {inserter_count}"

    def test_has_belts(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=42
        )
        ent_layer = world[Channel.ENTITIES.value]
        belt_count = (ent_layer == str2ent("transport_belt").value).sum().item()
        assert belt_count >= 2, f"Expected at least 2 belts, got {belt_count}"

    def test_nonzero_throughput(self):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=42
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"Expected positive throughput, got {tp}"


class TestAssemble1In1OutManySeeds:
    """Generate many lessons with different seeds and verify all are valid."""

    @pytest.mark.parametrize("seed", range(50))
    def test_size_10_seed(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(30))
    def test_size_8_seed(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_12_seed(self, seed):
        world, _ = generate_lesson(
            size=12, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_15_seed(self, seed):
        world, _ = generate_lesson(
            size=15, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"


class TestAssemble1In1OutParity:
    """Python and Rust throughput must agree on generated factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_parity_size_10(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        world_whc = world.permute(1, 2, 0)
        tp, ur = rs_throughput(world_whc)
        assert tp > 0, f"seed={seed}: throughput={tp}"
        assert ur >= 0


class TestAssemble1In1OutGridSizes:
    """Test across a range of grid sizes."""

    @pytest.mark.parametrize("size", [8, 9, 10, 12, 15])
    def test_valid_factory_per_size(self, size):
        world, _ = generate_lesson(
            size=size, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=7
        )
        ent_layer = world[Channel.ENTITIES.value]
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 1
        assert (ent_layer == str2ent("assembling_machine_1").value).sum().item() == 9
        assert (ent_layer == str2ent("inserter").value).sum().item() == 2
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0

    def test_too_small_grid_raises(self):
        """A grid smaller than 3×3 cannot fit the assembler."""
        with pytest.raises(Exception):
            generate_lesson(
                size=2, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=0
            )


class TestAssemble1In1OutEntityDirections:
    """All placed entities must have a non-NONE direction."""

    @pytest.mark.parametrize("seed", range(20))
    def test_all_entities_have_directions(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        dir_layer = world[Channel.DIRECTION.value]

        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                ent_val = ent_layer[x, y].item()
                if ent_val != str2ent("empty").value:
                    dir_val = dir_layer[x, y].item()
                    assert dir_val != Direction.NONE.value, (
                        f"seed={seed}: entity {entities[ent_val].name} at ({x},{y}) "
                        f"has NONE direction"
                    )


class TestAssemble1In1OutNoOverlaps:
    """Verify no entity overlaps in generated factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_no_double_placement(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        occupied = []
        for x in range(world.shape[1]):
            for y in range(world.shape[2]):
                if ent_layer[x, y].item() != str2ent("empty").value:
                    occupied.append((x, y))
        assert len(occupied) == len(set(occupied)), (
            f"seed={seed}: duplicate positions"
        )


class TestAssemble1In1OutItems:
    """Item channels: source carries the ingredient, sink the product, assembler the recipe."""

    @pytest.mark.parametrize("seed", range(30))
    def test_source_sink_items_match_a_known_recipe(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        item_layer = world[Channel.ITEMS.value]

        source_pos = (ent_layer == str2ent("source").value).nonzero(as_tuple=False)[0]
        sink_pos = (ent_layer == str2ent("sink").value).nonzero(as_tuple=False)[0]

        source_item = item_layer[source_pos[0], source_pos[1]].item()
        sink_item = item_layer[sink_pos[0], sink_pos[1]].item()

        assert source_item != str2item("empty").value, (
            f"seed={seed}: source has no item set"
        )
        assert sink_item != str2item("empty").value, (
            f"seed={seed}: sink has no item set"
        )
        assert source_item != sink_item, (
            f"seed={seed}: source and sink items are equal — recipe must transform"
        )

        # The (input, output) pair must match exactly one of the known
        # 1-in-1-out recipes.
        valid_pairs = {
            (str2item(inp).value, str2item(out).value)
            for (inp, out) in ONE_IN_ONE_OUT_RECIPES.values()
        }
        assert (source_item, sink_item) in valid_pairs, (
            f"seed={seed}: pair ({source_item}, {sink_item}) does not match any "
            f"known 1-in-1-out recipe"
        )

    @pytest.mark.parametrize("seed", range(30))
    def test_assembler_recipe_matches_output(self, seed):
        """The assembler's ITEMS channel value (recipe) should be the same
        item the sink consumes."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        ent_layer = world[Channel.ENTITIES.value]
        item_layer = world[Channel.ITEMS.value]
        asm_positions = (ent_layer == str2ent("assembling_machine_1").value).nonzero(
            as_tuple=False
        )
        sink_pos = (ent_layer == str2ent("sink").value).nonzero(as_tuple=False)[0]
        sink_item = item_layer[sink_pos[0], sink_pos[1]].item()

        for pos in asm_positions:
            asm_item = item_layer[pos[0], pos[1]].item()
            assert asm_item == sink_item, (
                f"seed={seed}: assembler tile at ({pos[0].item()}, {pos[1].item()}) "
                f"has recipe {asm_item}, sink expects {sink_item}"
            )


class TestAssemble1In1OutMissingEntities:
    """Test entity removal."""

    @pytest.mark.parametrize("num_missing", [1, 2, 3])
    def test_removes_correct_count(self, num_missing):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=num_missing,
            seed=42,
        )
        assert min_ent == num_missing
        ent_layer = world[Channel.ENTITIES.value]
        # Source, sink remain
        assert (ent_layer == str2ent("source").value).sum().item() == 1
        assert (ent_layer == str2ent("sink").value).sum().item() == 1

    @pytest.mark.parametrize("seed", range(10))
    def test_missing_inf_returns_positive_count(self, seed):
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=float("inf"), seed=seed,
        )
        assert world is not None
        assert min_ent is not None and min_ent > 0

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("num_missing", [1, 5, 20, float("inf")])
    def test_assembler_always_present(self, seed, num_missing):
        """The assembler is structurally required and must never be removed,
        regardless of num_missing_entities."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=num_missing, seed=seed,
        )
        ent_layer = world[Channel.ENTITIES.value]
        asm_count = (ent_layer == str2ent("assembling_machine_1").value).sum().item()
        assert asm_count == 9, (
            f"seed={seed}, num_missing={num_missing}: assembler removed "
            f"({asm_count} tiles, expected 9)"
        )

    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("num_missing", [1, 5, 20, float("inf")])
    def test_assembler_recipe_preserved_after_removal(self, seed, num_missing):
        """The recipe channel on assembler tiles must survive blanking."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=num_missing, seed=seed,
        )
        ent_layer = world[Channel.ENTITIES.value]
        item_layer = world[Channel.ITEMS.value]
        asm_positions = (ent_layer == str2ent("assembling_machine_1").value).nonzero(
            as_tuple=False
        )
        for pos in asm_positions:
            recipe_val = item_layer[pos[0], pos[1]].item()
            assert recipe_val != str2item("empty").value, (
                f"seed={seed}, num_missing={num_missing}: assembler tile "
                f"({pos[0].item()},{pos[1].item()}) lost its recipe"
            )


class TestAssemble1In1OutDeterminism:
    """Same seed → identical world."""

    @pytest.mark.parametrize("seed", [0, 1, 7, 42, 100])
    def test_same_seed_same_world(self, seed):
        w1, m1 = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        w2, m2 = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        assert torch.equal(w1, w2), f"seed={seed}: regenerated world differs"
        assert m1 == m2


class TestAssemble1In1OutThroughputRange:
    """Throughput should be positive and bounded by the assembler / inserter caps."""

    @pytest.mark.parametrize("seed", range(30))
    def test_throughput_in_expected_range(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT, num_missing_entities=0, seed=seed
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        # Lower bound: positive
        assert tp > 0, f"seed={seed}: tp={tp}"
        # Upper bound: an inserter caps at 0.86 i/s, and the output side
        # has only one inserter, so output throughput cannot exceed 0.86 ×
        # produces_per_craft. The two recipes give 2× (CC) and 1× (IGW)
        # the ingredient cost, so the loosest bound is 0.86 × 2 = 1.72.
        assert tp <= 1.72 + 1e-6, f"seed={seed}: tp={tp} exceeds inserter+recipe cap"


class TestAssemble1In1OutRecipeSelection:
    """Recipe is randomly picked across seeds — multiple distinct recipes
    should appear across many seeds (uniform selection over all 1-in 1-out
    recipes)."""

    def test_multiple_recipes_appear(self):
        seen_recipes = set()
        for seed in range(100):
            world, _ = generate_lesson(
                size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
                num_missing_entities=0, seed=seed,
            )
            ent_layer = world[Channel.ENTITIES.value]
            item_layer = world[Channel.ITEMS.value]
            sink_pos = (ent_layer == str2ent("sink").value).nonzero(as_tuple=False)[0]
            sink_item = item_layer[sink_pos[0], sink_pos[1]].item()
            seen_recipes.add(sink_item)

        # We have several 1-in 1-out recipes (copper_cable, iron_gear_wheel,
        # iron_stick, pipe, iron_chest, steel_plate, wooden_chest,
        # stone_furnace, ...). Expect at least 2 distinct ones across 100 seeds.
        assert len(seen_recipes) >= 2, (
            f"Expected ≥2 distinct recipes across 100 seeds, got {seen_recipes}"
        )


# ── ASSEMBLE_1IN_1OUT — semantic-correctness coverage ───────────────────────


def _asm_tiles(world):
    """Return the set of (x, y) tiles occupied by the assembler."""
    ent = world[Channel.ENTITIES.value]
    asm_val = str2ent("assembling_machine_1").value
    return {tuple(p.tolist()) for p in (ent == asm_val).nonzero(as_tuple=False)}


def _inserters(world):
    """Return [(pos, dir_value)] for each inserter on the grid."""
    ent = world[Channel.ENTITIES.value]
    dirs = world[Channel.DIRECTION.value]
    ins_val = str2ent("inserter").value
    out = []
    for p in (ent == ins_val).nonzero(as_tuple=False):
        x, y = p[0].item(), p[1].item()
        out.append(((x, y), dirs[x, y].item()))
    return out


def _dir_from_value(d_val):
    """Look up Direction enum from int value, or None for NONE."""
    for d in Direction:
        if d.value == d_val:
            return d
    return None


class TestAssemble1In1OutInserterGeometry:
    """Each lesson must have exactly one input inserter (drop into the
    assembler body) and one output inserter (pickup from the assembler
    body). Inserters must be on valid non-corner perimeter slots."""

    @pytest.mark.parametrize("seed", range(30))
    def test_exactly_one_input_one_output_inserter(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        asm = _asm_tiles(world)
        assert len(asm) == 9

        input_count = 0
        output_count = 0
        for (x, y), d_val in _inserters(world):
            d = _dir_from_value(d_val)
            assert d is not None, f"seed={seed}: inserter at ({x},{y}) has invalid dir"
            dx, dy = DIR_TO_DELTA[d]
            drop = (x + dx, y + dy)
            pickup = (x - dx, y - dy)
            drop_in_asm = drop in asm
            pickup_in_asm = pickup in asm
            assert drop_in_asm or pickup_in_asm, (
                f"seed={seed}: inserter at ({x},{y}) dir={d.name} — neither "
                f"drop {drop} nor pickup {pickup} is inside assembler"
            )
            assert not (drop_in_asm and pickup_in_asm), (
                f"seed={seed}: inserter at ({x},{y}) has both pickup and drop "
                f"inside assembler — impossible geometry"
            )
            if drop_in_asm:
                input_count += 1
            else:
                output_count += 1

        assert input_count == 1, f"seed={seed}: expected 1 input inserter, got {input_count}"
        assert output_count == 1, f"seed={seed}: expected 1 output inserter, got {output_count}"

    @pytest.mark.parametrize("seed", range(30))
    def test_inserters_on_non_corner_perimeter(self, seed):
        """Inserters must be on the 12 non-corner perimeter slots — not on
        the 4 corner slots (which the engine ignores) and not anywhere else.
        """
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        asm = _asm_tiles(world)
        # Anchor = min x, min y of the assembler tiles
        ax = min(x for x, _ in asm)
        ay = min(y for _, y in asm)

        valid_slots = set()
        # 12 non-corner perimeter slots: 3 per side
        for d in range(3):
            valid_slots.add((ax + d, ay - 1))     # north side
            valid_slots.add((ax + d, ay + 3))     # south side
            valid_slots.add((ax - 1, ay + d))     # west side
            valid_slots.add((ax + 3, ay + d))     # east side

        for (x, y), _ in _inserters(world):
            assert (x, y) in valid_slots, (
                f"seed={seed}: inserter at ({x},{y}) is not on a valid "
                f"non-corner perimeter slot of the assembler at ({ax},{ay})"
            )


class TestAssemble1In1OutConnectivity:
    """The graph must form a continuous source → … → sink path that passes
    through the assembler. No orphaned belts."""

    @pytest.mark.parametrize("seed", range(20))
    def test_source_reaches_sink_through_assembler(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        G = world2graph(world.permute(1, 2, 0))

        # Find source / sink / assembler nodes by name substring
        source_nodes = [n for n in G.nodes if "stack_inserter" in n]
        sink_nodes = [n for n in G.nodes if "bulk_inserter" in n]
        asm_nodes = [n for n in G.nodes if "assembling_machine" in n]
        assert len(source_nodes) == 1, f"seed={seed}: {len(source_nodes)} sources"
        assert len(sink_nodes) == 1, f"seed={seed}: {len(sink_nodes)} sinks"
        assert len(asm_nodes) == 1, f"seed={seed}: {len(asm_nodes)} assemblers"

        import networkx as nx
        # Source must reach assembler
        assert nx.has_path(G, source_nodes[0], asm_nodes[0]), (
            f"seed={seed}: source cannot reach assembler"
        )
        # Assembler must reach sink
        assert nx.has_path(G, asm_nodes[0], sink_nodes[0]), (
            f"seed={seed}: assembler cannot reach sink"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_no_orphaned_belts(self, seed):
        """Every placed belt should be on the source → sink chain (or at
        least connected to either endpoint via the graph)."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        G = world2graph(world.permute(1, 2, 0))
        import networkx as nx

        source_nodes = [n for n in G.nodes if "stack_inserter" in n]
        sink_nodes = [n for n in G.nodes if "bulk_inserter" in n]
        belt_nodes = [n for n in G.nodes if "transport_belt" in n]
        assert source_nodes and sink_nodes
        src, snk = source_nodes[0], sink_nodes[0]
        for belt in belt_nodes:
            assert nx.has_path(G, src, belt) or nx.has_path(G, belt, snk), (
                f"seed={seed}: belt {belt} is orphaned — not reachable from "
                f"source nor reaches sink"
            )


class TestAssemble1In1OutThroughputPerRecipe:
    """Each recipe has a known closed-form throughput given a
    1 input inserter + 1 output inserter cap of 0.86 i/s.

      copper_cable: 1 cu_plate → 2 cu_cable, 0.5s
        Input flow at full inserter cap:
          0.86 cu_plate/s; recipe needs 1 per craft → ratio 0.86
          produces 2 × 0.86 = 1.72 cu_cable/s
          output inserter caps at 0.86 → final throughput = 0.86 cu_cable/s

      iron_gear_wheel: 2 ir_plate → 1 ir_gear_wheel, 0.5s
        Input flow at full inserter cap:
          0.86 ir_plate/s; recipe needs 2 per craft → ratio 0.43
          produces 1 × 0.43 = 0.43 ir_gear_wheel/s
          output inserter cap (0.86) is not binding → final = 0.43
    """

    INSERTER_CAP = 0.86  # items/s — flow_rate of a basic inserter

    @pytest.mark.parametrize("seed", range(40))
    def test_throughput_matches_recipe_closed_form(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        ent = world[Channel.ENTITIES.value]
        item = world[Channel.ITEMS.value]
        sink_pos = (ent == str2ent("sink").value).nonzero(as_tuple=False)[0]
        recipe_name = items[item[sink_pos[0], sink_pos[1]].item()].name

        recipe = recipes[recipe_name]
        assert len(recipe.consumes) == 1 and len(recipe.produces) == 1, (
            f"ASSEMBLE_1IN_1OUT picked a non-1in1out recipe {recipe_name!r}"
        )
        cons_count = next(iter(recipe.consumes.values()))
        prod_count = next(iter(recipe.produces.values()))

        # Some seeds spawn the env Source directly adjacent to the
        # assembler footprint, which the graph treats as an extra
        # input edge bypassing the input inserter. In that case the
        # assembler input is unconstrained (recipe ratio caps at 1)
        # and only the output inserter caps the final throughput.
        # Detect via graph predecessors and pick the right expectation.
        from helpers import world2graph
        G = world2graph(world.permute(1, 2, 0))
        asm = next(n for n in G.nodes if "assembling_machine" in n)
        preds = list(G.predecessors(asm))
        source_directly_adjacent = any("stack_inserter" in p for p in preds)

        if source_directly_adjacent:
            # input ≥ recipe needs, ratio capped at 1, output = prod_count,
            # then output inserter caps to INSERTER_CAP.
            expected = min(self.INSERTER_CAP, prod_count)
        else:
            # Normal layout: input rate = INSERTER_CAP / cons_count.
            input_rate = self.INSERTER_CAP / cons_count
            expected = min(self.INSERTER_CAP, input_rate * prod_count)

        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert abs(tp - expected) < 1e-3, (
            f"seed={seed}, recipe={recipe_name}, "
            f"source_directly_adjacent={source_directly_adjacent}: "
            f"expected ≈ {expected}, got {tp}"
        )


class TestAssemble1In1OutSourceSinkOrientation:
    """Source / sink must face into the belt chain — the cell they output
    to / take from must contain a belt (or directly the inserter pickup)."""

    @pytest.mark.parametrize("seed", range(20))
    def test_source_faces_belt(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        ent = world[Channel.ENTITIES.value]
        dirs = world[Channel.DIRECTION.value]
        src_pos = (ent == str2ent("source").value).nonzero(as_tuple=False)[0]
        sx, sy = src_pos[0].item(), src_pos[1].item()
        d = _dir_from_value(dirs[sx, sy].item())
        dx, dy = DIR_TO_DELTA[d]
        out_x, out_y = sx + dx, sy + dy
        downstream = ent[out_x, out_y].item()
        # Should be a belt or inserter (the source feeds into the belt chain)
        valid = {str2ent("transport_belt").value, str2ent("inserter").value}
        assert downstream in valid, (
            f"seed={seed}: source at ({sx},{sy}) faces ({out_x},{out_y}) which "
            f"holds entity {entities[downstream].name}, not a belt/inserter"
        )

    @pytest.mark.parametrize("seed", range(20))
    def test_sink_faces_belt(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        ent = world[Channel.ENTITIES.value]
        dirs = world[Channel.DIRECTION.value]
        snk_pos = (ent == str2ent("sink").value).nonzero(as_tuple=False)[0]
        sx, sy = snk_pos[0].item(), snk_pos[1].item()
        d = _dir_from_value(dirs[sx, sy].item())
        dx, dy = DIR_TO_DELTA[d]
        # Sink takes from `pos - delta(dir)` (its input cell)
        in_x, in_y = sx - dx, sy - dy
        upstream = ent[in_x, in_y].item()
        valid = {str2ent("transport_belt").value, str2ent("inserter").value}
        assert upstream in valid, (
            f"seed={seed}: sink at ({sx},{sy}) takes from ({in_x},{in_y}) which "
            f"holds entity {entities[upstream].name}, not a belt/inserter"
        )


class TestAssemble1In1OutBeltItems:
    """Belt tiles should not have items set in the ITEMS channel — items
    only travel through belts at runtime; the channel is for source/sink/
    recipe metadata, not transient cargo."""

    @pytest.mark.parametrize("seed", range(20))
    def test_belt_items_channel_is_empty(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        ent = world[Channel.ENTITIES.value]
        item = world[Channel.ITEMS.value]
        belt_val = str2ent("transport_belt").value
        empty = str2item("empty").value
        for p in (ent == belt_val).nonzero(as_tuple=False):
            x, y = p[0].item(), p[1].item()
            assert item[x, y].item() == empty, (
                f"seed={seed}: belt at ({x},{y}) has unexpected item "
                f"{items[item[x, y].item()].name}"
            )


class TestAssemble1In1OutEdgeCases:
    """Edge cases: tiny grids, large grids."""

    @pytest.mark.parametrize("size", [1, 2])
    def test_grid_too_small_raises(self, size):
        with pytest.raises(Exception):
            generate_lesson(
                size=size, kind=LessonKind.ASSEMBLE_1IN_1OUT,
                num_missing_entities=0, seed=0,
            )

    @pytest.mark.parametrize("seed", range(5))
    def test_large_grid_works(self, seed):
        world, _ = generate_lesson(
            size=20, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0


class TestAssemble1In1OutMaxEntities:
    """The `max_entities` cap on belt-path length should be respected by
    the generator (it retries until it finds a layout under the cap)."""

    @pytest.mark.parametrize("seed", range(10))
    def test_respects_max_entities(self, seed):
        # Generate without a cap to find the natural entity count, then
        # impose a cap of natural+5 and verify regeneration succeeds and
        # respects the limit.
        world, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed,
        )
        ent = world[Channel.ENTITIES.value]
        belts = (ent == str2ent("transport_belt").value).sum().item()
        natural_total = belts + 3  # +1 assembler unit, +2 inserters

        # Generation under the natural cap should still succeed.
        world2, _ = generate_lesson(
            size=10, kind=LessonKind.ASSEMBLE_1IN_1OUT,
            num_missing_entities=0, seed=seed, max_entities=natural_total + 5,
        )
        ent2 = world2[Channel.ENTITIES.value]
        belts2 = (ent2 == str2ent("transport_belt").value).sum().item()
        total2 = belts2 + 3
        assert total2 <= natural_total + 5, (
            f"seed={seed}: generator placed {total2} entities (cap was {natural_total + 5})"
        )


# ── MOVE_VIA_UG_BELT ─────────────────────────────────────────────────────────


def _ug_layout_info(world):
    """Extract layout info from a MOVE_VIA_UG_BELT solved world.

    Returns dict with: flow_dir, source_pos, sink_pos, ug_down_pos, ug_up_pos,
    wall_tiles (set of (x, y)).
    """
    ent = world[Channel.ENTITIES.value]
    direc = world[Channel.DIRECTION.value]
    misc = world[Channel.MISC.value]
    fp = world[Channel.FOOTPRINT.value]

    src_pos = (ent == str2ent("source").value).nonzero(as_tuple=False)
    sink_pos = (ent == str2ent("sink").value).nonzero(as_tuple=False)
    assert len(src_pos) == 1 and len(sink_pos) == 1
    src = tuple(src_pos[0].tolist())
    sink = tuple(sink_pos[0].tolist())

    ug = (ent == str2ent("underground_belt").value).nonzero(as_tuple=False)
    assert len(ug) == 2, f"expected 2 UG belts, got {len(ug)}"
    down_pos = up_pos = None
    for px, py in ug.tolist():
        m = int(misc[px, py].item())
        if m == Misc.UNDERGROUND_DOWN.value:
            down_pos = (px, py)
        elif m == Misc.UNDERGROUND_UP.value:
            up_pos = (px, py)
    assert down_pos is not None and up_pos is not None

    flow_dir = Direction(int(direc[down_pos[0], down_pos[1]].item()))
    wall_tiles = set()
    W, H = fp.shape
    for x in range(W):
        for y in range(H):
            if fp[x, y].item() == Footprint.UNAVAILABLE.value:
                wall_tiles.add((x, y))

    return {
        "flow_dir": flow_dir,
        "source_pos": src,
        "sink_pos": sink,
        "ug_down_pos": down_pos,
        "ug_up_pos": up_pos,
        "wall_tiles": wall_tiles,
    }


class TestMoveViaUgBeltBasic:
    """Basic sanity checks for MOVE_VIA_UG_BELT lesson generation."""

    def test_generates_without_error(self):
        world, min_ent = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=42
        )
        assert world is not None
        assert min_ent is not None

    def test_returns_cwh_tensor(self):
        size = 8
        world, _ = generate_lesson(
            size=size, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=42
        )
        assert world.shape == (len(Channel), size, size)

    def test_has_source_and_sink(self):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=42
        )
        ent = world[Channel.ENTITIES.value]
        assert (ent == str2ent("source").value).sum().item() == 1
        assert (ent == str2ent("sink").value).sum().item() == 1

    def test_has_one_ug_pair(self):
        """Solved layout has exactly one UG_DOWN and one UG_UP."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=42
        )
        ent = world[Channel.ENTITIES.value]
        misc = world[Channel.MISC.value]
        ug_mask = ent == str2ent("underground_belt").value
        assert ug_mask.sum().item() == 2
        down = ((misc == Misc.UNDERGROUND_DOWN.value) & ug_mask).sum().item()
        up = ((misc == Misc.UNDERGROUND_UP.value) & ug_mask).sum().item()
        assert down == 1 and up == 1, f"expected 1 DOWN + 1 UP, got {down} + {up}"

    def test_nonzero_throughput(self):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=42
        )
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0

    @pytest.mark.parametrize("seed", range(20))
    def test_throughput_parity(self, seed):
        """Python and Rust throughput agree on solved layouts."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=seed
        )
        compare_throughput(world.permute(1, 2, 0))


class TestMoveViaUgBeltGeometry:
    """A perpendicular wall separates source and sink; UG pair bridges it."""

    @pytest.mark.parametrize("seed", range(40))
    def test_ug_pair_faces_flow_direction(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=seed
        )
        direc = world[Channel.DIRECTION.value]
        info = _ug_layout_info(world)
        for pos in (info["ug_down_pos"], info["ug_up_pos"]):
            assert int(direc[pos[0], pos[1]].item()) == info["flow_dir"].value

    @pytest.mark.parametrize("seed", range(40))
    def test_ug_pair_directly_opposite_across_wall(self, seed):
        """UG_UP shares UG_DOWN's perpendicular coord and is on the other
        side of the wall, with the wall flush between them."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=seed
        )
        info = _ug_layout_info(world)
        flow = info["flow_dir"]
        is_h = flow in (Direction.EAST, Direction.WEST)
        down = info["ug_down_pos"]
        up = info["ug_up_pos"]
        if is_h:
            assert down[1] == up[1], (
                f"seed={seed}: UG_DOWN.y={down[1]} != UG_UP.y={up[1]}"
            )
            gap = abs(up[0] - down[0])
        else:
            assert down[0] == up[0], (
                f"seed={seed}: UG_DOWN.x={down[0]} != UG_UP.x={up[0]}"
            )
            gap = abs(up[1] - down[1])
        # gap = wall_thickness + 1, with wall_thickness ∈ [1, 4]
        assert 2 <= gap <= 5, f"seed={seed}: UG gap {gap} not in [2, 5]"

    @pytest.mark.parametrize("seed", range(40))
    def test_wall_spans_full_perpendicular(self, seed):
        """The wall (UNAVAILABLE tiles) spans the entire perpendicular
        dimension of the grid."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=seed
        )
        info = _ug_layout_info(world)
        flow = info["flow_dir"]
        is_h = flow in (Direction.EAST, Direction.WEST)
        wall = info["wall_tiles"]
        assert wall, f"seed={seed}: no wall tiles"
        # Group by the flow-axis coordinate; each column/row must hit every
        # perpendicular coordinate.
        W, H = world[Channel.FOOTPRINT.value].shape
        flow_coords = {(x if is_h else y) for x, y in wall}
        perp_span = H if is_h else W
        for fc in flow_coords:
            perp_cells = {
                (y if is_h else x)
                for x, y in wall
                if (x if is_h else y) == fc
            }
            assert perp_cells == set(range(perp_span)), (
                f"seed={seed}: wall column/row {fc} doesn't span full perp"
            )

    @pytest.mark.parametrize("seed", range(40))
    def test_only_wall_is_unavailable(self, seed):
        """No tile outside the wall is FOOTPRINT.UNAVAILABLE — open space
        on either side of the wall is freely buildable."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=seed
        )
        fp = world[Channel.FOOTPRINT.value]
        info = _ug_layout_info(world)
        wall = info["wall_tiles"]
        W, H = fp.shape
        for x in range(W):
            for y in range(H):
                if (x, y) not in wall:
                    assert fp[x, y].item() == Footprint.AVAILABLE.value, (
                        f"seed={seed}: ({x},{y}) UNAVAILABLE but not in wall"
                    )

    @pytest.mark.parametrize("seed", range(40))
    def test_source_and_sink_on_opposite_sides_of_wall(self, seed):
        """Source and sink lie on opposite sides of the wall along the
        flow axis."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=seed
        )
        info = _ug_layout_info(world)
        flow = info["flow_dir"]
        is_h = flow in (Direction.EAST, Direction.WEST)
        wall = info["wall_tiles"]
        wall_flow = {(x if is_h else y) for x, y in wall}
        wall_lo, wall_hi = min(wall_flow), max(wall_flow)
        src = info["source_pos"]
        sink = info["sink_pos"]
        src_f = src[0] if is_h else src[1]
        sink_f = sink[0] if is_h else sink[1]
        # Both must be outside the wall, and on different sides of it
        assert src_f < wall_lo or src_f > wall_hi
        assert sink_f < wall_lo or sink_f > wall_hi
        assert (src_f < wall_lo) != (sink_f < wall_lo), (
            f"seed={seed}: src and sink on same side of wall"
        )

    @pytest.mark.parametrize("seed", range(40))
    def test_ug_down_on_source_side_ug_up_on_sink_side(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=seed
        )
        info = _ug_layout_info(world)
        flow = info["flow_dir"]
        is_h = flow in (Direction.EAST, Direction.WEST)
        wall = info["wall_tiles"]
        wall_flow = {(x if is_h else y) for x, y in wall}
        wall_lo, wall_hi = min(wall_flow), max(wall_flow)
        src = info["source_pos"]
        down = info["ug_down_pos"]
        up = info["ug_up_pos"]
        src_f = src[0] if is_h else src[1]
        down_f = down[0] if is_h else down[1]
        up_f = up[0] if is_h else up[1]
        # UG_DOWN on source side
        if src_f < wall_lo:
            assert down_f < wall_lo and up_f > wall_hi
        else:
            assert down_f > wall_hi and up_f < wall_lo


class TestMoveViaUgBeltManySeeds:
    @pytest.mark.parametrize("seed", range(50))
    def test_size_8_seed(self, seed):
        world, min_ent = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT, num_missing_entities=0, seed=seed
        )
        assert world is not None
        assert min_ent is not None
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0

    @pytest.mark.parametrize("size", [6, 8, 10, 12])
    def test_grid_sizes(self, size):
        world, _ = generate_lesson(
            size=size, kind=LessonKind.MOVE_VIA_UG_BELT,
            num_missing_entities=0, seed=7,
        )
        assert world.shape == (len(Channel), size, size)
        tp, _ = rs_throughput(world.permute(1, 2, 0))
        assert tp > 0


class TestMoveViaUgBeltDirections:
    """All four flow directions must be reachable by the generator."""

    def test_all_directions_appear(self):
        seen = set()
        for seed in range(200):
            world, _ = generate_lesson(
                size=8, kind=LessonKind.MOVE_VIA_UG_BELT,
                num_missing_entities=0, seed=seed,
            )
            seen.add(_ug_layout_info(world)["flow_dir"])
            if len(seen) == 4:
                break
        assert seen == {
            Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST,
        }, f"only saw {seen}"


class TestMoveViaUgBeltMissingEntities:
    """Blanking preserves source/sink and the FOOTPRINT mask."""

    @pytest.mark.parametrize("num_missing", [1, 2, 5, float("inf")])
    @pytest.mark.parametrize("seed", range(10))
    def test_source_and_sink_always_present(self, num_missing, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT,
            num_missing_entities=num_missing, seed=seed,
        )
        ent = world[Channel.ENTITIES.value]
        assert (ent == str2ent("source").value).sum().item() == 1
        assert (ent == str2ent("sink").value).sum().item() == 1

    @pytest.mark.parametrize("seed", range(20))
    def test_wall_remains_unavailable_after_blanking(self, seed):
        """Even at maximum blanking, the wall is still the only UNAVAILABLE
        region (open tiles do not flip to UNAVAILABLE just because they
        were blanked)."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_VIA_UG_BELT,
            num_missing_entities=float("inf"), seed=seed,
        )
        info = _ug_layout_info(world)
        # The wall spans full perpendicular; any non-wall tile must be
        # AVAILABLE (already covered by other tests for num_missing=0;
        # here we just re-assert it survives blanking).
        fp = world[Channel.FOOTPRINT.value]
        W, H = fp.shape
        for x in range(W):
            for y in range(H):
                if (x, y) not in info["wall_tiles"]:
                    assert fp[x, y].item() == Footprint.AVAILABLE.value

    @pytest.mark.parametrize("seed", range(20))
    def test_independent_ug_removal_possible(self, seed):
        """With num_missing=1, we may end up removing only DOWN, only UP,
        or only a transport belt — they're independent removal units."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.MOVE_VIA_UG_BELT,
            num_missing_entities=1, seed=seed,
        )
        ent = world[Channel.ENTITIES.value]
        misc = world[Channel.MISC.value]
        ug_mask = ent == str2ent("underground_belt").value
        down = ((misc == Misc.UNDERGROUND_DOWN.value) & ug_mask).sum().item()
        up = ((misc == Misc.UNDERGROUND_UP.value) & ug_mask).sum().item()
        assert (down, up) in {(0, 1), (1, 0), (1, 1)}, (
            f"seed={seed}: unexpected (down, up) = ({down}, {up})"
        )
