"""Shared test fixtures and utilities for throughput tests."""

import os
import sys
from typing import Any

import numpy as np
import torch

# Disable wandb before importing factorion
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion_rs  # noqa: E402
from factorion import (  # noqa: E402
    DIR_TO_DELTA,
    Channel,
    Direction,
    Factory,
    Footprint,
    LessonKind,
    Misc,
    blank_entities,
    build_factory,
    build_graph_nx,
    entities,
    find_belt_path,
    find_belt_paths_with_source_sink_orient,
    items,
    recipes,
    render_factory,
    str2ent,
    str2item,
    world2html,
)


# ── Test model size ──────────────────────────────────────────────────────────
# A CPU forward through the production attention stack (attn_dim 192, 12 heads,
# 4 layers) costs more than everything else a model test does, and these tests
# exercise the code path rather than the capacity. The two spellings must stay
# in sync: tests save a checkpoint through the AgentCNN kwargs and load it back
# through a SharedArgs-configured model.
_TINY_ATTN: dict[str, Any] = dict(attn_dim=32, attn_heads=4, attn_layers=1)
TINY_ARCH: dict[str, Any] = dict(layers=(16, 16, 16), **_TINY_ATTN)
TINY_ARCH_ARGS: dict[str, Any] = dict(layer1=16, layer2=16, layer3=16, **_TINY_ATTN)


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_world(w, h=None):
    """Create an empty square WHC world tensor.

    Several lesson generators assume a square grid, so we use max(w, h).
    """
    if h is None:
        h = w
    size = max(w, h)
    world = torch.zeros((size, size, len(Channel)), dtype=torch.int64)
    world[:, :, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
    return world


def set_entity(world, x, y, entity_name, direction, item_name="empty", misc=0):
    """Place an entity on the world tensor."""
    world[x, y, Channel.ENTITIES.value] = str2ent(entity_name).value
    world[x, y, Channel.DIRECTION.value] = direction.value
    world[x, y, Channel.ITEMS.value] = str2item(item_name).value
    world[x, y, Channel.MISC.value] = misc


def set_assembler(world, x, y, recipe_name):
    """Place a 3x3 assembling machine with its top-left corner at (x, y)."""
    for dx in range(3):
        for dy in range(3):
            world[x + dx, y + dy, Channel.ENTITIES.value] = str2ent(
                "assembling_machine_1"
            ).value
            world[x + dx, y + dy, Channel.DIRECTION.value] = Direction.NORTH.value
            world[x + dx, y + dy, Channel.ITEMS.value] = str2item(recipe_name).value


def set_multi_tile_entity(world, entity_name, x, y, direction, item_name="empty"):
    """Place a multi-tile entity at anchor (x, y), filling all occupied tiles.

    Uses entity_tiles to compute the footprint. Raises ValueError if any
    tile would overlap an existing non-empty entity.
    """
    ent = str2ent(entity_name)
    tiles = factorion_rs.py_entity_tiles(x, y, direction.value, ent.width, ent.height)
    if tiles is None:
        raise ValueError(
            f"Cannot place {entity_name} with direction {direction}"
        )
    for tx, ty in tiles:
        existing = world[tx, ty, Channel.ENTITIES.value].item()
        if existing != str2ent("empty").value:
            raise ValueError(
                f"Cannot place {entity_name} at ({x},{y}): tile ({tx},{ty}) "
                f"already has entity {entities[existing].name}"
            )
    for tx, ty in tiles:
        set_entity(world, tx, ty, entity_name, direction, item_name)


def set_splitter(world, x, y, direction, item_name="empty"):
    """Place a splitter at anchor (x, y), filling all occupied tiles."""
    set_multi_tile_entity(world, "splitter", x, y, direction, item_name)


def build_factory_graph(world_WHC):
    """Build a factory's connection graph as a networkx ``DiGraph``.

    The single indirection point for graph construction across the test
    suite. Nodes are named ``f"{entity_char}@{x},{y}"`` (plus a ``:L``/``:R``
    lane suffix for belt-ish lane nodes) and edges follow
    the engine's entity-connection rules. Delegates to the Rust engine via
    ``factorion.build_graph_nx`` (issue #178).
    """
    return build_graph_nx(world_WHC)


def rs_throughput(world):
    """`(throughput, unreachable)` via the Rust engine's Python bindings."""
    score, unreachable, _flow = factorion_rs.simulate_throughput(
        world.numpy().astype(np.int64)
    )
    return score, unreachable
