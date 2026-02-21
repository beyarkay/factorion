"""Parity tests between Python funge_throughput and Rust simulate_throughput."""

import numpy as np
import pytest

import factorion_rs


# Channel indices (matching factorion.py Channel enum)
CH_ENTITIES = 0
CH_DIRECTION = 1
CH_ITEMS = 2
CH_MISC = 3

# Entity values (matching factorion.py entities dict)
ENT_EMPTY = 0
ENT_TRANSPORT_BELT = 1
ENT_INSERTER = 2
ENT_ASSEMBLING_MACHINE_1 = 3
ENT_UNDERGROUND_BELT = 4
ENT_SINK = 5  # bulk_inserter
ENT_SOURCE = 6  # stack_inserter

# Direction values
DIR_NONE = 0
DIR_NORTH = 1
DIR_EAST = 2
DIR_SOUTH = 3
DIR_WEST = 4

# Item values
ITEM_EMPTY = 0
ITEM_COPPER_CABLE = 1
ITEM_COPPER_PLATE = 2
ITEM_IRON_PLATE = 3
ITEM_ELECTRONIC_CIRCUIT = 4

# Misc values
MISC_NONE = 0
MISC_UNDERGROUND_DOWN = 1
MISC_UNDERGROUND_UP = 2


def make_world(width, height):
    """Create an empty WHC world tensor."""
    return np.zeros((width, height, 4), dtype=np.int64)


def place_entity(world, x, y, entity, direction=DIR_NONE, item=ITEM_EMPTY, misc=MISC_NONE):
    """Place an entity at (x, y) in the world."""
    world[x, y, CH_ENTITIES] = entity
    world[x, y, CH_DIRECTION] = direction
    world[x, y, CH_ITEMS] = item
    world[x, y, CH_MISC] = misc


class TestEmptyWorld:
    def test_empty_world_returns_zero(self):
        w = make_world(5, 5)
        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert throughput == 0.0
        assert unreachable == 0

    def test_1x1_empty(self):
        w = make_world(1, 1)
        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert throughput == 0.0


class TestSimpleBeltChain:
    def test_source_belt_sink(self):
        """Source → Belt → Sink: throughput limited by belt (15.0)."""
        w = make_world(3, 1)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 1, 0, ENT_TRANSPORT_BELT, DIR_EAST)
        place_entity(w, 2, 0, ENT_SINK, DIR_EAST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert abs(throughput - 15.0) < 1e-9
        assert unreachable == 0

    def test_source_belt_belt_sink(self):
        """Source → Belt → Belt → Sink."""
        w = make_world(4, 1)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 1, 0, ENT_TRANSPORT_BELT, DIR_EAST)
        place_entity(w, 2, 0, ENT_TRANSPORT_BELT, DIR_EAST)
        place_entity(w, 3, 0, ENT_SINK, DIR_EAST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert abs(throughput - 15.0) < 1e-9

    def test_source_belt_chain_long(self):
        """Source → 8 belts → Sink."""
        w = make_world(10, 1)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        for i in range(1, 9):
            place_entity(w, i, 0, ENT_TRANSPORT_BELT, DIR_EAST)
        place_entity(w, 9, 0, ENT_SINK, DIR_EAST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert abs(throughput - 15.0) < 1e-9

    def test_vertical_belt_chain_south(self):
        """Vertical chain going south."""
        w = make_world(1, 4)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_SOUTH, ITEM_COPPER_CABLE)
        place_entity(w, 0, 1, ENT_TRANSPORT_BELT, DIR_SOUTH)
        place_entity(w, 0, 2, ENT_TRANSPORT_BELT, DIR_SOUTH)
        place_entity(w, 0, 3, ENT_SINK, DIR_SOUTH, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert abs(throughput - 15.0) < 1e-9


class TestInserterChain:
    def test_source_inserter_belt_sink(self):
        """Source → Inserter → Belt → Sink: throughput limited by inserter (0.86)."""
        w = make_world(4, 1)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 1, 0, ENT_INSERTER, DIR_EAST)
        place_entity(w, 2, 0, ENT_TRANSPORT_BELT, DIR_EAST)
        place_entity(w, 3, 0, ENT_SINK, DIR_EAST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert abs(throughput - 0.86) < 1e-9


class TestUndergroundBelt:
    def test_underground_bypass(self):
        """Source → Belt → Underground(down) ... Underground(up) → Belt → Sink."""
        w = make_world(7, 1)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 1, 0, ENT_TRANSPORT_BELT, DIR_EAST)
        place_entity(w, 2, 0, ENT_UNDERGROUND_BELT, DIR_EAST, misc=MISC_UNDERGROUND_DOWN)
        # Tiles 3, 4 are empty (underground)
        place_entity(w, 5, 0, ENT_UNDERGROUND_BELT, DIR_EAST, misc=MISC_UNDERGROUND_UP)
        place_entity(w, 6, 0, ENT_SINK, DIR_EAST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        # Underground belt has same throughput as transport belt (15.0)
        assert abs(throughput - 15.0) < 1e-9


class TestDisconnected:
    def test_disconnected_source_sink(self):
        """Source and sink with no path between them."""
        w = make_world(5, 5)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 4, 4, ENT_SINK, DIR_EAST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        # No connection → 0 throughput
        assert throughput == 0.0 or not any(v > 0 for v in [throughput])

    def test_disconnected_with_lone_belt(self):
        """Source, sink, and a disconnected belt."""
        w = make_world(5, 5)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 4, 4, ENT_SINK, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 2, 2, ENT_TRANSPORT_BELT, DIR_EAST)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert throughput == 0.0 or not any(v > 0 for v in [throughput])
        assert unreachable >= 1  # at least the disconnected belt


class TestSourceOnly:
    def test_source_no_sink(self):
        w = make_world(3, 1)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert throughput == 0.0

    def test_sink_no_source(self):
        w = make_world(3, 1)
        place_entity(w, 0, 0, ENT_SINK, DIR_EAST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        assert throughput == 0.0


class TestBeltDirection:
    def test_opposing_belts_no_connection(self):
        """Two belts facing each other should not connect."""
        w = make_world(4, 1)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 1, 0, ENT_TRANSPORT_BELT, DIR_EAST)
        place_entity(w, 2, 0, ENT_TRANSPORT_BELT, DIR_WEST)
        place_entity(w, 3, 0, ENT_SINK, DIR_WEST, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        # Opposing belts break the chain
        assert throughput == 0.0 or unreachable > 0


class TestLCorner:
    def test_belt_corner(self):
        """Source(east) → Belt(east) → Belt(south) → Sink(south).

        The east belt feeds into the south belt (belt ahead is south-facing).
        This should work since they're not opposing.
        """
        w = make_world(3, 3)
        place_entity(w, 0, 0, ENT_SOURCE, DIR_EAST, ITEM_COPPER_CABLE)
        place_entity(w, 1, 0, ENT_TRANSPORT_BELT, DIR_EAST)
        # Belt at (2,0) turns south
        place_entity(w, 2, 0, ENT_TRANSPORT_BELT, DIR_SOUTH)
        place_entity(w, 2, 1, ENT_TRANSPORT_BELT, DIR_SOUTH)
        place_entity(w, 2, 2, ENT_SINK, DIR_SOUTH, ITEM_COPPER_CABLE)

        throughput, unreachable = factorion_rs.simulate_throughput(w)
        # The east belt should connect to the south belt since south is not opposing east
        # Then the chain continues south to the sink
        assert abs(throughput - 15.0) < 1e-9
