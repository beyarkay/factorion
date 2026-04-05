"""Tests for lesson generation: INSERTER_TRANSFER, SPLITTER_SPLIT, SPLITTER_MERGE."""

import pytest
import random
import torch

from helpers import (
    Channel,
    Direction,
    LessonKind,
    Misc,
    compare_throughput,
    entities,
    generate_lesson,
    py_throughput_safe,
    rs_throughput,
    str2ent,
    str2item,
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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"Expected positive throughput, got {tp}"

    def test_throughput_bottlenecked_by_inserter(self):
        """Throughput should be <= inserter flow rate (0.86)."""
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=42
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        inserter_flow = str2ent("inserter").flow
        assert tp <= inserter_flow + 1e-6, (
            f"Throughput {tp} exceeds inserter flow rate {inserter_flow}"
        )


class TestInserterTransferManySeeds:
    """Generate many lessons with different seeds and verify all are valid."""

    @pytest.mark.parametrize("seed", range(50))
    def test_size_8_seed(self, seed):
        """Size 8 grid, many seeds — all must produce valid factories."""
        world, min_ent = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"
        assert tp <= str2ent("inserter").flow + 1e-6

    @pytest.mark.parametrize("seed", range(30))
    def test_size_6_seed(self, seed):
        """Size 6 grid — smaller grid, still must work."""
        world, min_ent = generate_lesson(
            size=6, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(30))
    def test_size_10_seed(self, seed):
        """Size 10 grid — larger grid."""
        world, min_ent = generate_lesson(
            size=10, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_15_seed(self, seed):
        """Size 15 grid — large grid."""
        world, min_ent = generate_lesson(
            size=15, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"


class TestInserterTransferParity:
    """Python and Rust throughput must agree on generated inserter factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_parity_size_8(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.INSERTER_TRANSFER, num_missing_entities=0, seed=seed
        )
        world_whc = world.permute(1, 2, 0)
        py_tp, py_ur = py_throughput_safe(world_whc)
        rs_tp, rs_ur = rs_throughput(world_whc)
        assert abs(py_tp - rs_tp) < 1e-6, (
            f"seed={seed}: Python={py_tp}, Rust={rs_tp}"
        )
        assert py_ur == rs_ur, (
            f"seed={seed}: unreachable Python={py_ur}, Rust={rs_ur}"
        )


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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"Expected positive throughput, got {tp}"

    def test_throughput_bounded_by_splitter(self):
        """Total throughput should be <= 30.0 (splitter max flow)."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=42
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp <= 30.0 + 1e-6, f"Throughput {tp} exceeds splitter max flow"


class TestSplitterSplitManySeeds:
    """Generate many lessons with different seeds and verify all are valid."""

    @pytest.mark.parametrize("seed", range(50))
    def test_size_10_seed(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(30))
    def test_size_8_seed(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_12_seed(self, seed):
        world, _ = generate_lesson(
            size=12, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_15_seed(self, seed):
        world, _ = generate_lesson(
            size=15, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"


class TestSplitterSplitParity:
    """Python and Rust throughput must agree on generated splitter-split factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_parity_size_10(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_SPLIT, num_missing_entities=0, seed=seed
        )
        world_whc = world.permute(1, 2, 0)
        py_tp, py_ur = py_throughput_safe(world_whc)
        rs_tp, rs_ur = rs_throughput(world_whc)
        assert abs(py_tp - rs_tp) < 1e-6, (
            f"seed={seed}: Python={py_tp}, Rust={rs_tp}"
        )
        assert py_ur == rs_ur, (
            f"seed={seed}: unreachable Python={py_ur}, Rust={rs_ur}"
        )


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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"Expected positive throughput, got {tp}"

    def test_throughput_bounded_by_splitter(self):
        """Total throughput should be <= 30.0 (splitter max flow)."""
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=42
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp <= 30.0 + 1e-6, f"Throughput {tp} exceeds splitter max flow"


class TestSplitterMergeManySeeds:
    """Generate many lessons with different seeds and verify all are valid."""

    @pytest.mark.parametrize("seed", range(50))
    def test_size_10_seed(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(30))
    def test_size_8_seed(self, seed):
        world, _ = generate_lesson(
            size=8, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_12_seed(self, seed):
        world, _ = generate_lesson(
            size=12, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"

    @pytest.mark.parametrize("seed", range(20))
    def test_size_15_seed(self, seed):
        world, _ = generate_lesson(
            size=15, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
        assert tp > 0, f"seed={seed}: throughput is {tp}"


class TestSplitterMergeParity:
    """Python and Rust throughput must agree on generated splitter-merge factories."""

    @pytest.mark.parametrize("seed", range(30))
    def test_parity_size_10(self, seed):
        world, _ = generate_lesson(
            size=10, kind=LessonKind.SPLITTER_MERGE, num_missing_entities=0, seed=seed
        )
        world_whc = world.permute(1, 2, 0)
        py_tp, py_ur = py_throughput_safe(world_whc)
        rs_tp, rs_ur = rs_throughput(world_whc)
        assert abs(py_tp - rs_tp) < 1e-6, (
            f"seed={seed}: Python={py_tp}, Rust={rs_tp}"
        )
        assert py_ur == rs_ur, (
            f"seed={seed}: unreachable Python={py_ur}, Rust={rs_ur}"
        )


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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
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
        tp, _ = py_throughput_safe(world.permute(1, 2, 0))
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
