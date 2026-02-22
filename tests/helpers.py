"""Shared test fixtures and utilities for throughput parity tests."""

import os
import sys

import numpy as np
import torch

# Disable wandb before importing factorion
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion  # noqa: E402
import factorion_rs  # noqa: E402

# ── Extract marimo cell objects ──────────────────────────────────────────────

_, _objs = factorion.datatypes.run()
_, _fns = factorion.functions.run()

Channel = _objs["Channel"]
Direction = _objs["Direction"]
Misc = _objs["Misc"]
LessonKind = _objs["LessonKind"]
entities = _objs["entities"]
items = _objs["items"]

generate_lesson = _fns["generate_lesson"]
str2ent = _fns["str2ent"]
str2item = _fns["str2item"]
world2graph = _fns["world2graph"]
calc_throughput_py = _fns["calc_throughput"]


# ── Helpers ──────────────────────────────────────────────────────────────────


def make_world(w, h=None):
    """Create an empty square WHC world tensor.

    Python's world2graph requires square worlds, so we use max(w, h).
    """
    if h is None:
        h = w
    size = max(w, h)
    return torch.zeros((size, size, 4), dtype=torch.int64)


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


def py_throughput_safe(world):
    """Call the Python throughput pipeline (world2graph + calc_throughput) directly.

    This avoids funge_throughput's breakpoint() on assertion errors.
    Returns (throughput_value, num_unreachable) matching funge_throughput's contract.
    """
    G = world2graph(world)
    throughput_dict, num_unreachable = calc_throughput_py(G)
    if len(throughput_dict) == 0:
        return 0.0, num_unreachable
    actual = list(throughput_dict.values())[0]
    if actual == float("inf"):
        # Python funge_throughput would assert here; treat as 0
        return 0.0, num_unreachable
    return float(actual), num_unreachable


def rs_throughput(world):
    """Rust throughput via Python bindings."""
    return factorion_rs.simulate_throughput(world.numpy().astype(np.int64))


def compare_throughput(world, tolerance=1e-6):
    """Run both implementations and assert they match."""
    py_tp, py_unreachable = py_throughput_safe(world)
    rs_tp, rs_unreachable = rs_throughput(world)
    assert abs(py_tp - rs_tp) <= tolerance, (
        f"Throughput mismatch: Python={py_tp}, Rust={rs_tp}"
    )
    assert py_unreachable == rs_unreachable, (
        f"Unreachable mismatch: Python={py_unreachable}, Rust={rs_unreachable}"
    )
    return py_tp, py_unreachable
