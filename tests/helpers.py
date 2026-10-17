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
    items,
    recipes,
    render_factory,
    str2ent,
    str2item,
    world2html,
)


# ── Test model size ──────────────────────────────────────────────────────────
# These tests exercise the denoiser's code paths, not its capacity; a CPU
# forward through the production size costs more than the rest of a test.
TINY: dict[str, Any] = dict(width=16, levels=2, blocks=1, attn_layers=1, attn_heads=4)


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
    """Throughput via the Rust engine's Python bindings."""
    return factorion_rs.simulate_throughput(world.numpy().astype(np.int64))


def power(world):
    """A hand-built WHC `world` with medium poles powering its inserters and
    assemblers, as the lesson generators place them. Rows and columns of free
    tiles are added along its bottom and right edges (keeping every
    coordinate) until the poles fit."""
    for i in range(8):
        powered = factorion_rs.py_place_poles(world.numpy().astype(np.int64))
        if powered is not None:
            return torch.from_numpy(powered)
        axis = 1 - i % 2
        shape = list(world.shape)
        shape[axis] = 1
        pad = torch.zeros(shape, dtype=world.dtype)
        pad[..., Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
        world = torch.cat([world, pad], dim=axis)
    raise ValueError("no room to power this world")
