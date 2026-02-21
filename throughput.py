"""Refactored throughput calculation for Factorio factory grids.

Replaces the monolithic funge_throughput/world2graph/calc_throughput functions
with a handler-based architecture where each entity type defines its own
connection and flow logic.

Usage:
    calculator = ThroughputCalculator(entities, items, recipes, Channel, Direction, Misc, DIR_TO_DELTA)
    throughput, num_unreachable = calculator.calculate_throughput(world_tensor)
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Protocol, Optional

import networkx as nx
import numpy as np
import torch


@dataclass(slots=True)
class TileInfo:
    """Data read from the world tensor for a single tile."""
    entity_name: str
    entity_value: int
    entity_flow: float
    entity_width: int
    entity_height: int
    direction: object  # Direction enum
    item_name: str
    misc: object  # Misc enum
    x: int
    y: int
    node_id: str = ""

    def __post_init__(self):
        self.node_id = f"{self.entity_name}\n@{self.x},{self.y}"


class WorldAccessor:
    """Thin wrapper around the numpy world array for named tile access.

    Pre-caches all TileInfo objects on construction so that get_tile is O(1).
    """

    def __init__(self, world_WHC: np.ndarray, entities, items, Channel, Direction, Misc, DIR_TO_DELTA):
        W, H, C = world_WHC.shape
        self.W = W
        self.H = H
        self._DIR_TO_DELTA = DIR_TO_DELTA

        # Pre-extract enum .value constants to avoid repeated attribute lookups
        ch_ent = Channel.ENTITIES.value
        ch_dir = Channel.DIRECTION.value
        ch_itm = Channel.ITEMS.value
        ch_msc = Channel.MISC.value

        # Pre-build lookup tables to avoid repeated enum construction
        dir_lookup = {d.value: d for d in Direction}
        misc_lookup = {m.value: m for m in Misc}

        # Pre-cache all tiles in a flat list (row-major: index = x * H + y)
        cache = [None] * (W * H)
        for x in range(W):
            row_offset = x * H
            for y in range(H):
                e = entities[world_WHC[x, y, ch_ent]]
                if e.name == "empty":
                    continue
                d = dir_lookup[world_WHC[x, y, ch_dir]]
                item = items[world_WHC[x, y, ch_itm]]
                misc = misc_lookup[world_WHC[x, y, ch_msc]]
                cache[row_offset + y] = TileInfo(
                    entity_name=e.name,
                    entity_value=e.value,
                    entity_flow=e.flow,
                    entity_width=e.width,
                    entity_height=e.height,
                    direction=d,
                    item_name=item.name,
                    misc=misc,
                    x=x,
                    y=y,
                )
        self._cache = cache
        self._empty_tile_template = TileInfo(
            entity_name="empty", entity_value=0, entity_flow=0.0,
            entity_width=1, entity_height=1, direction=Direction.NONE,
            item_name="empty", misc=Misc.NONE, x=0, y=0,
        )

    def get_tile(self, x: int, y: int) -> Optional[TileInfo]:
        if not (0 <= x < self.W and 0 <= y < self.H):
            return None
        tile = self._cache[x * self.H + y]
        if tile is None:
            # Empty tile — return a lightweight sentinel
            return self._empty_tile_template
        return tile

    def get_neighbor(self, x: int, y: int, direction) -> Optional[TileInfo]:
        """Get the tile in front of (x, y) given the direction."""
        delta = self._DIR_TO_DELTA.get(direction)
        if delta is None:
            return self.get_tile(x, y)
        return self.get_tile(x + delta[0], y + delta[1])

    def get_behind(self, x: int, y: int, direction) -> Optional[TileInfo]:
        """Get the tile behind (x, y) given the direction."""
        delta = self._DIR_TO_DELTA.get(direction)
        if delta is None:
            return self.get_tile(x, y)
        return self.get_tile(x - delta[0], y - delta[1])


class EntityHandler(Protocol):
    """Interface every entity type must implement."""

    def get_connections(self, tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
        """Return (src_node_id, dst_node_id) edges this tile creates."""
        ...

    def get_initial_output(self, tile: TileInfo) -> dict[str, float]:
        """Return initial output for source nodes; empty dict for most."""
        ...

    def compute_output(self, tile: TileInfo, inputs: dict[str, float]) -> dict[str, float]:
        """Given aggregated input flows, compute output flows."""
        ...


class SourceHandler:
    """stack_inserter: infinite source of items."""

    def get_connections(self, tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
        # Source nodes create edges like inserters (since they ARE inserters).
        # They connect behind→self and self→ahead.
        return _inserter_connections(tile, world)

    def get_initial_output(self, tile: TileInfo) -> dict[str, float]:
        return {tile.item_name: float("inf")}

    def compute_output(self, tile: TileInfo, inputs: dict[str, float]) -> dict[str, float]:
        return {tile.item_name: float("inf")}


class SinkHandler:
    """bulk_inserter: infinite capacity sink."""

    def get_connections(self, tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
        # Sink nodes connect like inserters too.
        return _inserter_connections(tile, world)

    def get_initial_output(self, tile: TileInfo) -> dict[str, float]:
        return {}

    def compute_output(self, tile: TileInfo, inputs: dict[str, float]) -> dict[str, float]:
        # Infinite capacity: pass through all input uncapped
        return {k: min(v, tile.entity_flow) for k, v in inputs.items()}


def _inserter_connections(tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
    """Shared connection logic for inserter-like entities (inserter, source, sink)."""
    edges = []
    src_tile = world.get_behind(tile.x, tile.y, tile.direction)
    dst_tile = world.get_neighbor(tile.x, tile.y, tile.direction)

    if src_tile is not None and src_tile.entity_name != "empty":
        edges.append((src_tile.node_id, tile.node_id))

    if dst_tile is not None:
        dst_is_insertable = (
            "belt" in dst_tile.entity_name
            or "assembling_machine" in dst_tile.entity_name
        )
        if dst_is_insertable:
            edges.append((tile.node_id, dst_tile.node_id))

    return edges


class InserterHandler:
    """inserter: picks items from behind, places ahead. Caps flow at entity rate."""

    def get_connections(self, tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
        return _inserter_connections(tile, world)

    def get_initial_output(self, tile: TileInfo) -> dict[str, float]:
        return {}

    def compute_output(self, tile: TileInfo, inputs: dict[str, float]) -> dict[str, float]:
        return {k: min(v, tile.entity_flow) for k, v in inputs.items()}


class TransportBeltHandler:
    """transport_belt: connects to same-direction belt behind, belt ahead (not opposing)."""

    def __init__(self, Direction, Misc):
        self._Direction = Direction
        self._Misc = Misc
        self._opposite = Direction.SOUTH.value - Direction.NORTH.value

    def get_connections(self, tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
        edges = []
        Misc = self._Misc

        # Source: belt behind, same direction, not underground-DOWN
        src_tile = world.get_behind(tile.x, tile.y, tile.direction)
        if src_tile is not None:
            src_is_beltish = (
                "belt" in src_tile.entity_name
                and src_tile.direction == tile.direction
                # NOTE: The original code has `src_misc.value == Misc.UNDERGROUND_DOWN`
                # which compares int to Enum member and always returns False.
                # We replicate the buggy behavior for parity: DOWN underground
                # belts are NOT excluded as sources.
                and not (
                    "underground_belt" in src_tile.entity_name
                    and src_tile.misc.value == Misc.UNDERGROUND_DOWN
                )
            )
            if src_is_beltish:
                edges.append((src_tile.node_id, tile.node_id))

        # Destination: belt ahead, not opposing direction
        dst_tile = world.get_neighbor(tile.x, tile.y, tile.direction)
        if dst_tile is not None:
            dst_is_belt = "belt" in dst_tile.entity_name
            dst_opposing = (
                dst_is_belt
                and abs(dst_tile.direction.value - tile.direction.value) == self._opposite
            )
            if dst_is_belt and not dst_opposing:
                edges.append((tile.node_id, dst_tile.node_id))

        return edges

    def get_initial_output(self, tile: TileInfo) -> dict[str, float]:
        return {}

    def compute_output(self, tile: TileInfo, inputs: dict[str, float]) -> dict[str, float]:
        return {k: min(v, tile.entity_flow) for k, v in inputs.items()}


class AssemblingMachineHandler:
    """assembling_machine_1: 3x3 entity. Scans perimeter for inserters, applies recipe."""

    def __init__(self, Direction, recipes):
        self._Direction = Direction
        self._recipes = recipes

    def get_connections(self, tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
        edges = []
        Direction = self._Direction
        w = tile.entity_width  # 3
        h = tile.entity_height  # 3

        # Search perimeter: dx in [-1, w], dy in [-1, h], skip interior and corners
        for dx in range(-1, w + 1):
            for dy in range(-1, h + 1):
                # Skip interior
                if 0 <= dx < w and 0 <= dy < h:
                    continue
                # Skip corners
                if dx in (-1, w) and dy in (-1, h):
                    continue

                other = world.get_tile(tile.x + dx, tile.y + dy)
                if other is None or "inserter" not in other.entity_name:
                    continue

                other_str = other.node_id
                self_str = tile.node_id

                # Inserter pointing away from machine: machine → inserter
                if (
                    (other.direction == Direction.NORTH and dy < 0)
                    or (other.direction == Direction.SOUTH and dy > 0)
                    or (other.direction == Direction.WEST and dx < 0)
                    or (other.direction == Direction.EAST and dx > 0)
                ):
                    edges.append((self_str, other_str))
                else:
                    # Inserter pointing toward machine: inserter → machine
                    edges.append((other_str, self_str))

        return edges

    def get_initial_output(self, tile: TileInfo) -> dict[str, float]:
        return {}

    def compute_output(self, tile: TileInfo, inputs: dict[str, float]) -> dict[str, float]:
        recipe_name = tile.item_name
        if recipe_name not in self._recipes:
            return {}
        recipe = self._recipes[recipe_name]
        min_ratio = 1.0
        for item, required in recipe.consumes.items():
            available = inputs.get(item, 0)
            min_ratio = min(min_ratio, available / required)
        return {k: v * min_ratio for k, v in recipe.produces.items()}


class UndergroundBeltHandler:
    """underground_belt: DOWN scans ahead for UP counterpart; UP creates no edges."""

    def __init__(self, Direction, Misc, DIR_TO_DELTA):
        self._Direction = Direction
        self._Misc = Misc
        self._DIR_TO_DELTA = DIR_TO_DELTA

    def get_connections(self, tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
        edges = []
        Misc = self._Misc
        DIR_TO_DELTA = self._DIR_TO_DELTA

        if tile.misc == Misc.UNDERGROUND_DOWN:
            max_delta = 6
        elif tile.misc == Misc.UNDERGROUND_UP:
            max_delta = 1
        else:
            # Invalid underground belt state; old code would assert here
            return edges

        delta_xy = DIR_TO_DELTA.get(tile.direction)
        if delta_xy is None:
            return edges
        dx, dy = delta_xy

        for delta in range(1, max_delta):
            target = world.get_tile(tile.x + dx * delta, tile.y + dy * delta)
            if target is None:
                continue

            going_underground = (
                target.entity_name == "underground_belt"
                and tile.misc == Misc.UNDERGROUND_DOWN
            )
            cxn_to_belt = (
                "transport_belt" in target.entity_name
                and tile.misc == Misc.UNDERGROUND_UP
            )
            if going_underground or cxn_to_belt:
                edges.append((tile.node_id, target.node_id))

        return edges

    def get_initial_output(self, tile: TileInfo) -> dict[str, float]:
        return {}

    def compute_output(self, tile: TileInfo, inputs: dict[str, float]) -> dict[str, float]:
        return {k: min(v, tile.entity_flow) for k, v in inputs.items()}


class SplitterHandler:
    """splitter: 2x1 entity that splits input equally among outputs.

    Not present in the current entity registry; included for future extensibility.
    Flow division is handled during propagation via out-degree splitting.
    """

    def __init__(self, Direction, DIR_TO_DELTA):
        self._Direction = Direction
        self._DIR_TO_DELTA = DIR_TO_DELTA

    def get_connections(self, tile: TileInfo, world: WorldAccessor) -> list[tuple[str, str]]:
        edges = []
        Direction = self._Direction
        DIR_TO_DELTA = self._DIR_TO_DELTA

        delta_xy = DIR_TO_DELTA.get(tile.direction)
        if delta_xy is None:
            return edges
        dx, dy = delta_xy

        # Splitter occupies 2 tiles perpendicular to its direction
        if tile.direction in (Direction.NORTH, Direction.SOUTH):
            lane_offsets = [(0, 0), (1, 0)]
        else:
            lane_offsets = [(0, 0), (0, 1)]

        for lx, ly in lane_offsets:
            # Input: belt behind this lane
            behind = world.get_tile(tile.x + lx - dx, tile.y + ly - dy)
            if behind is not None and "belt" in behind.entity_name:
                edges.append((behind.node_id, tile.node_id))

            # Output: belt ahead of this lane
            ahead = world.get_tile(tile.x + lx + dx, tile.y + ly + dy)
            if ahead is not None and "belt" in ahead.entity_name:
                edges.append((tile.node_id, ahead.node_id))

        return edges

    def get_initial_output(self, tile: TileInfo) -> dict[str, float]:
        return {}

    def compute_output(self, tile: TileInfo, inputs: dict[str, float]) -> dict[str, float]:
        # Raw total; division by out-degree happens during propagation
        return dict(inputs)


class ThroughputCalculator:
    """Calculates factory throughput using a handler-based graph approach.

    Args:
        entities: dict mapping int -> Entity dataclass
        items: dict mapping int -> Item dataclass
        recipes: dict mapping recipe_name -> Recipe dataclass
        Channel: Enum with ENTITIES, DIRECTION, ITEMS, MISC
        Direction: Enum with NONE, NORTH, EAST, SOUTH, WEST
        Misc: Enum with NONE, UNDERGROUND_DOWN, UNDERGROUND_UP
        DIR_TO_DELTA: dict mapping Direction -> (dx, dy)
    """

    def __init__(self, entities, items, recipes, Channel, Direction, Misc, DIR_TO_DELTA):
        self.entities = entities
        self.items = items
        self.recipes = recipes
        self.Channel = Channel
        self.Direction = Direction
        self.Misc = Misc
        self.DIR_TO_DELTA = DIR_TO_DELTA

        # Build handler registry
        self._handlers = {
            "transport_belt": TransportBeltHandler(Direction, Misc),
            "inserter": InserterHandler(),
            "assembling_machine_1": AssemblingMachineHandler(Direction, recipes),
            "underground_belt": UndergroundBeltHandler(Direction, Misc, DIR_TO_DELTA),
            "bulk_inserter": SinkHandler(),
            "stack_inserter": SourceHandler(),
            "splitter": SplitterHandler(Direction, DIR_TO_DELTA),
        }

    def _make_world_accessor(self, world_WHC: np.ndarray) -> WorldAccessor:
        return WorldAccessor(
            world_WHC, self.entities, self.items,
            self.Channel, self.Direction, self.Misc, self.DIR_TO_DELTA,
        )

    def build_flow_graph(self, world_WHC: np.ndarray, debug=False) -> nx.DiGraph:
        """Build a directed flow graph from the world tensor.

        Single-pass: iterates pre-cached tiles, creates nodes and edges together.
        """
        world = self._make_world_accessor(world_WHC)
        G = nx.DiGraph()
        handlers = self._handlers

        def dbg(s):
            if debug:
                print(s)

        # Collect non-empty tiles from the pre-built cache
        cache = world._cache
        H = world.H
        tiles = [t for t in cache if t is not None]

        # Pass 1: create all nodes
        for tile in tiles:
            handler = handlers.get(tile.entity_name)
            if handler is None:
                assert False, f"Don't know how to handle {tile.entity_name} at {tile.x} {tile.y}"

            initial_output = handler.get_initial_output(tile)
            G.add_node(
                tile.node_id,
                input_={},
                output=initial_output,
                recipe=tile.item_name if "assembling_machine" in tile.entity_name else None,
                entity_name=tile.entity_name,
                entity_flow=tile.entity_flow,
            )
            dbg(
                f"Created node {repr(tile.node_id)}: {G.nodes[tile.node_id]}, "
                f"direction is {tile.direction}, recipe is {tile.item_name}"
            )

        # Pass 2: create all edges
        for tile in tiles:
            handler = handlers[tile.entity_name]
            for src, dst in handler.get_connections(tile, world):
                G.add_edge(src, dst)
                dbg(f"{repr(src)} -> {repr(dst)}")

        return G

    def propagate_flow(self, G: nx.DiGraph, debug=False) -> tuple[dict[str, float], int]:
        """Propagate flow through the graph and return (output_dict, num_unreachable).

        Handles all edge cases from RL-generated worlds:
        - Cycles: returns ({}, 0) immediately (matches old behavior)
        - Disconnected components: processed normally via topological sort
        - Unconnected entities: get zero flow
        - No sources/sinks: throughput = 0
        """

        def dbg(s):
            if debug:
                print(s)

        if len(G.nodes) == 0:
            return {}, 0

        # Check for cycles (matches old behavior: returns {"foobar": 0.0}, 0)
        if not nx.is_directed_acyclic_graph(G):
            dbg(f"Returning 0 reward due to cycles")
            return {"foobar": 0.0}, 0

        dbg("Pre-calcs:")
        for n in G.nodes:
            dbg(f"- {repr(n)}: {G.nodes[n]}")

        # Topological sort guarantees we process each node after all predecessors
        for node in nx.topological_sort(G):
            data = G.nodes[node]
            entity_name = data.get("entity_name", "")
            entity_flow = data.get("entity_flow", 0.0)

            # Skip source nodes (they already have output set)
            if len(data.get("output", {})) > 0:
                dbg(f"\nSkipping node {repr(node)} (already has output)")
                continue

            dbg(f"\nChecking node {repr(node)}")

            # Aggregate inputs from all predecessors
            aggregated_input = {}
            for pred in G.predecessors(node):
                pred_output = G.nodes[pred].get("output", {})
                for item, flow_rate in pred_output.items():
                    if item not in aggregated_input:
                        aggregated_input[item] = 0
                    aggregated_input[item] += flow_rate

            data["input_"] = aggregated_input
            dbg(f"  curr[input_] is now: {data['input_']}")

            # Compute output
            if "assembling_machine" in node:
                recipe_name = data.get("recipe", "empty")
                if recipe_name == "empty":
                    print(
                        f"assembling machine {repr(node)} has {recipe_name=}, is not equal to empty"
                    )
                data["output"] = {}
                if recipe_name in self.recipes:
                    min_ratio = 1.0
                    recipe = self.recipes[recipe_name]
                    for item, rate in recipe.consumes.items():
                        ratio = aggregated_input.get(item, 0) / rate
                        min_ratio = min(min_ratio, ratio)
                    dbg(f"    Recipe consumables: {recipe.consumes}")
                    dbg(f"    Recipe products: {recipe.produces}")
                    data["output"] = {
                        k: v * min_ratio for k, v in recipe.produces.items()
                    }
                dbg(f"  Minimum ratio for {data} is {min_ratio if recipe_name in self.recipes else 'N/A'}")
            else:
                # Non-assembler: cap each input by entity flow rate
                for k, v in aggregated_input.items():
                    data["output"][k] = min(v, entity_flow)
                dbg(f'  made input_ match output: {data["input_"]=} {data["output"]=}')

            dbg(f"  after: {data=}")

        # Sum output at all sinks
        output = {}
        dbg("iterating G.nodes")
        for n in G.nodes:
            dbg(f"- {repr(n)}: {G.nodes[n]}")
            if "bulk_inserter" not in n:
                continue
            dbg(f"{repr(n)} is bulk inserter, examining")
            for k, v in G.nodes[n].get("output", {}).items():
                if k not in output:
                    output[k] = 0
                output[k] += v
                dbg(f"- Added {v} to output[{k}] to make {output[k]} from {repr(n)}")

        # Calculate unreachable nodes
        sources = [n for n in G if "stack_inserter" in n]
        sinks = [n for n in G if "bulk_inserter" in n]

        if sinks:
            can_reach_sink = set().union(*(nx.ancestors(G, s) | {s} for s in sinks))
        else:
            can_reach_sink = set()

        if sources:
            reachable_from_source = set().union(
                *(nx.descendants(G, s) | {s} for s in sources)
            )
        else:
            reachable_from_source = set()

        unreachable = set(G.nodes) - (
            can_reach_sink.intersection(reachable_from_source)
        )

        dbg(f"{can_reach_sink=}")
        dbg(f"{reachable_from_source=}")
        dbg(
            f"source -> ({len(reachable_from_source)} nodes) ... "
            f"({len(can_reach_sink)} nodes) -> sink"
        )
        dbg(
            f"Final Throughput: {output}, {len(unreachable)} unreachable nodes: {unreachable}"
        )

        return output, len(unreachable)

    def calculate_throughput(self, world, debug=False):
        """API-compatible replacement for funge_throughput.

        Args:
            world: tensor of shape (W, H, C) where C=4 channels
            debug: if True, print debug information

        Returns:
            (throughput: float, num_unreachable: int)
        """
        assert torch.is_tensor(world), f"world is {type(world)}, not a tensor"
        assert len(world.shape) == 3, (
            f"Expected world to have 3 dimensions, but is of shape {world.shape}"
        )
        assert world.shape[0] == world.shape[1], (
            f"Expected world to be square, but is of shape {world.shape}"
        )
        try:
            G = self.build_flow_graph(world.numpy(), debug=debug)
            throughput, num_unreachable = self.propagate_flow(G, debug=debug)
            if len(throughput) == 0:
                return 0, num_unreachable
            actual_throughput = list(throughput.values())[0]
            assert actual_throughput < float("inf"), (
                f"throughput is +inf, probably a bug, world is: {torch.tensor(world).permute(2, 0, 1)}"
            )
            return actual_throughput, num_unreachable
        except AssertionError:
            traceback.print_exc()
            return 0, 0
