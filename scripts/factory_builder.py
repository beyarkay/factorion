"""Interactive factory builder.

Spins up a tiny local HTTP server that serves a drag-and-drop UI for
designing a factory and visualising the flow graph it produces.

The browser POSTs the grid to the server, which builds the flow graph via
the Rust engine (``factorion_rs.py_build_graph``) for visualisation and runs
``factorion_rs.simulate_throughput`` for throughput, then returns a rendered
graph image.

Usage:
    uv run python scripts/factory_builder.py
    # then open http://localhost:8765
"""

from __future__ import annotations

import base64
import functools
import io
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Iterator, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402
import yaml  # noqa: E402

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion_rs  # noqa: E402
from factorion import (  # noqa: E402
    Channel,
    Direction,
    Factory,
    Footprint,
    LessonKind,
    Misc,
    blank_entities,
    build_factory,
    build_graph_nx,
    ent_str2b64img,
    entities,
    items,
    new_world,
    plot_flow_network,
    render_factory,
)
from denoise import (  # noqa: E402
    CH_PART,
    DenoiserUNet,
    corrupt,
    free_mask,
    generate,
    legalize,
    load_denoiser,
    noise_tokens,
    tokens_to_state,
    world_to_state,
)
from ppo import _resolve_wandb_checkpoint, apply_placement_action  # noqa: E402
from training_config import SftArgs  # noqa: E402


# Order shown in the palette and dropdowns.
PLACEABLE_ENTITIES = [
    "transport_belt",
    "underground_belt",
    "splitter",
    "inserter",
    "long_handed_inserter",
    "stack_inserter",
    "bulk_inserter",
    "assembling_machine_1",
]
NON_PLACEABLE_ITEMS = [
    "copper_cable",
    "copper_plate",
    "iron_plate",
    "electronic_circuit",
    "iron_gear_wheel",
]
# 10-slot hotbar mapped to keys 1..9,0 (key '0' is the last slot).
# `None` means the slot is unbound and pressing that key is a no-op.
# "empty" is the eraser.
HOTBAR = [
    "transport_belt",
    "underground_belt",
    "splitter",
    "inserter",
    "stack_inserter",
    "bulk_inserter",
    "assembling_machine_1",
    "long_handed_inserter",
    None,
    "empty",
]
# Display labels for slots whose canonical entity name differs from how
# they're shown to the user. `stack_inserter` / `bulk_inserter` are the
# Python-facing names for `Item::Source` / `Item::Sink` (see
# factorion_rs/src/types.rs); the UI says what they actually are.
DISPLAY_NAME = {
    "stack_inserter": "source",
    "bulk_inserter": "sink",
}


def _stroke_icon(path: str) -> str:
    """A one-path inline SVG that inherits its colour from the text around it."""
    return (
        f'<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" '
        f'fill="none" stroke="currentColor" stroke-width="2.2" '
        f'stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></svg>'
    )


# The throughput verdict. Inline SVG for the same reason as the copy icon
# below: a glyph the viewer's font doesn't cover would render as a blank box,
# and this one carries the whole answer.
OK_ICON = _stroke_icon("M3 8.5l3.5 3.5L13 4.5")
BAD_ICON = _stroke_icon("M4 4l8 8M12 4l-8 8")

# Inline rather than an emoji: the button carries no text, so a glyph the
# user's font happens not to cover would leave it blank. `currentColor` keeps
# it in step with whatever the button inherits.
COPY_ICON = (
    '<svg viewBox="0 0 16 16" width="11" height="11" aria-hidden="true" '
    'fill="none" stroke="currentColor" stroke-width="1.4">'
    '<rect x="5.9" y="5.9" width="8.4" height="8.4" rx="1.5"/>'
    '<path d="M10.6 3.2H3.2a1.5 1.5 0 0 0-1.5 1.5v7.4"/></svg>'
)
DIRECTIONS = ["NONE", "NORTH", "EAST", "SOUTH", "WEST"]
MISC_VALUES = ["NONE", "UNDERGROUND_DOWN", "UNDERGROUND_UP"]

# Keyboard + mouse cheatsheet shown by the [?] popover. Kept as data so
# the markup is built once and the lines stay easy to edit.
HELP_LINES = [
    "Hotbar: 1–9, 0",
    "Place: click / drag slot onto tile",
    "Select / edit: click an empty tile",
    "Rotate selected: r (cw), R (ccw)",
    "Clear selected: Delete / Backspace / right-click",
    "Deselect hotbar: Esc",
    "Eyedropper: z picks up the hovered entity (rotation, recipe); click places it, r rotates it, Esc drops it",
    "Generate lesson: g (set 'entities to clear' to blank N first)",
    "Denoise: tap a for one step, hold a to run the chain to its fixed point; a after that starts a new chain",
    "Resize / clear grid: c",
    "Undo / redo: Ctrl+Z / Ctrl+Shift+Z (or Ctrl+Y)",
    "Scan seeds tab: rebuild N blanked seeds at once, click a result to open it",
    "Scan filter: / focuses it, e.g. lesson=A,B thput>=0.5 converged=false steps<=12; Esc clears",
]


@dataclass
class Args:
    port: int = 8765
    """port for the local HTTP server"""
    size: int = 15
    """default grid size"""
    checkpoint: Optional[str] = None
    """path to a trained denoiser checkpoint (`sft.py`'s .pt). If set, `a`
    denoises the grid one step at a time."""
    wandb_run: Optional[str] = "dr17bmik"
    """W&B run id (or full path 'entity/project/run_id'). The run's most
    recent model-type artifact is downloaded to /tmp/factorion-checkpoints
    and loaded. Mutually exclusive with --checkpoint."""
    wandb_project: str = "factorion"
    """W&B project to look in when --wandb-run is a bare id."""
    wandb_entity: Optional[str] = None
    """W&B entity (team or user). None = your default entity."""


def build_world(grid: list[list[dict]]) -> torch.Tensor:
    """Convert a JSON grid (rows of {entity, direction, item, misc,
    footprint} dicts) into a world tensor in WHC layout."""
    h = len(grid)
    w = len(grid[0]) if h else 0
    if w != h:
        raise ValueError(f"grid must be square, got {w}x{h}")
    world = new_world(width=w, height=h)
    name_to_value = {it.name: it.value for it in items.values()}
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            ent_name = cell.get("entity") or "empty"
            world[x, y, Channel.ENTITIES.value] = name_to_value.get(
                ent_name, name_to_value["empty"]
            )
            world[x, y, Channel.DIRECTION.value] = Direction[
                cell.get("direction", "NONE")
            ].value
            item_name = cell.get("item") or "empty"
            world[x, y, Channel.ITEMS.value] = name_to_value.get(
                item_name, name_to_value["empty"]
            )
            world[x, y, Channel.MISC.value] = Misc[cell.get("misc", "NONE")].value
            footprint = cell.get("footprint", "AVAILABLE")
            world[x, y, Channel.FOOTPRINT.value] = Footprint[footprint].value
    return torch.tensor(world)


def world_CWH_to_grid(world_CWH: torch.Tensor) -> list[list[dict]]:
    """Inverse of :func:`build_world`: convert a (C, W, H) world tensor
    into the JSON grid format the JS frontend uses
    (``grid[y][x] = {entity, direction, item, misc, footprint}``)."""
    _, W, H = world_CWH.shape
    name_for_value = {it.value: it.name for it in items.values()}
    dir_for_value = {d.value: d.name for d in Direction}
    misc_for_value = {m.value: m.name for m in Misc}
    footprint_for_value = {f.value: f.name for f in Footprint}
    rows: list[list[dict]] = []
    for y in range(H):
        row: list[dict] = []
        for x in range(W):
            ent_v = int(world_CWH[Channel.ENTITIES.value, x, y].item())
            dir_v = int(world_CWH[Channel.DIRECTION.value, x, y].item())
            item_v = int(world_CWH[Channel.ITEMS.value, x, y].item())
            misc_v = int(world_CWH[Channel.MISC.value, x, y].item())
            foot_v = int(world_CWH[Channel.FOOTPRINT.value, x, y].item())
            row.append({
                "entity": name_for_value.get(ent_v, "empty"),
                "direction": dir_for_value.get(dir_v, "NONE"),
                "item": name_for_value.get(item_v, "empty"),
                "misc": misc_for_value.get(misc_v, "NONE"),
                "footprint": footprint_for_value.get(foot_v, "AVAILABLE"),
            })
        rows.append(row)
    return rows


def _apply_prediction(grid: list[list[dict]], prediction: dict) -> dict:
    """Apply a model prediction through the rollout environment's placement path.

    The browser intentionally does not know entity dimensions or footprint
    rules. It sends the current grid and predicted action here; this function
    converts their names to action IDs and delegates the actual validation and
    mutation to :func:`ppo.apply_placement_action`, the same function used by
    :meth:`ppo.FactorioEnv.step`.
    """
    world_CWH = build_world(grid).permute(2, 0, 1).contiguous()
    name_to_value = {it.name: it.value for it in items.values()}
    action = {
        "xy": (int(prediction["x"]), int(prediction["y"])),
        "entity": name_to_value[prediction["entity"]],
        "direction": Direction[prediction["direction"]].value,
        "item": name_to_value[prediction["item"]],
        "misc": Misc[prediction["misc"]].value,
        "eot": 0,
    }
    is_invalid, invalid_reason, _placed_action = apply_placement_action(
        world_CWH,
        action,
        source_id=name_to_value["stack_inserter"],
        sink_id=name_to_value["bulk_inserter"],
    )
    return {
        "applied": not is_invalid,
        "invalid_reason": invalid_reason,
        "grid": world_CWH_to_grid(world_CWH),
    }


_MARKERS = ("stack_inserter", "bulk_inserter")
_ASSEMBLER = "assembling_machine_1"


def _footprint(x: int, y: int, cell: dict) -> list[tuple[int, int]]:
    proto = next(it for it in items.values() if it.name == cell["entity"])
    tiles = factorion_rs.py_entity_tiles(
        x, y, Direction[cell["direction"]].value, proto.width, proto.height
    )
    return [tuple(t) for t in tiles] if tiles else [(x, y)]


def _unit_anchor(grid: list[list[dict]], x: int, y: int) -> tuple[int, int]:
    """The tile the entity covering (x, y) was placed from."""
    world_WHC = np.ascontiguousarray(build_world(grid).numpy(), dtype=np.int64)
    unit = {tuple(t) for t in factorion_rs.py_unit_tiles(world_WHC, [(x, y)])}
    for ux, uy in unit:
        tiles = _footprint(ux, uy, grid[uy][ux])
        if tiles[0] == (ux, uy) and set(tiles) == unit:
            return ux, uy
    return x, y


def _place_by_hand(grid: list[list[dict]], x: int, y: int, cell: dict) -> dict:
    """Place `cell` at (x, y) through the rollout placement path, so multi-tile
    entities land and clear whole. Hand edits get two allowances that path
    refuses: 1x1 source/sink markers are written directly, and an assembler may
    be placed before its recipe is chosen."""
    cell = {k: cell[k] for k in ("entity", "direction", "item", "misc")}
    if cell["entity"] == "empty":
        cell.update(direction="NONE", item="empty", misc="NONE")
    else:
        if cell["direction"] == "NONE" and cell["entity"] != _ASSEMBLER:
            cell["direction"] = "EAST"
        if cell["entity"] != "underground_belt":
            cell["misc"] = "NONE"
        elif cell["misc"] == "NONE":
            cell["misc"] = "UNDERGROUND_DOWN"
    if cell["entity"] in _MARKERS or grid[y][x]["entity"] in _MARKERS:
        if grid[y][x]["entity"] not in _MARKERS:
            cleared = _place_by_hand(grid, x, y, {**cell, "entity": "empty"})
            if not cleared["applied"]:
                return cleared
            grid = cleared["grid"]
        grid = [[dict(c) for c in row] for row in grid]
        grid[y][x].update(cell)
        return {"applied": True, "invalid_reason": None, "grid": grid}
    recipe_later = cell["entity"] == _ASSEMBLER and cell["item"] == "empty"
    any_recipe = next(iter(factorion_rs.py_recipes()))
    result = _apply_prediction(
        grid, {"x": x, "y": y, **cell, "item": any_recipe if recipe_later else cell["item"]}
    )
    if recipe_later and result["applied"]:
        for tx, ty in _footprint(x, y, cell):
            result["grid"][ty][tx]["item"] = "empty"
    return result


def _hand_edit(grid: list[list[dict]], ops: list[dict]) -> dict:
    """Apply hand-edit ops in order, all or nothing. An op either places a
    `cell` at (x, y), or sets one `field` of whichever entity covers (x, y),
    re-placing it whole from its anchor."""
    for op in ops:
        x, y = int(op["x"]), int(op["y"])
        if "field" in op:
            x, y = _unit_anchor(grid, x, y)
            cell = {**grid[y][x], op["field"]: op["value"]}
        else:
            cell = op["cell"]
        result = _place_by_hand(grid, x, y, cell)
        if not result["applied"]:
            return {**result, "grid": None}
        grid = result["grid"]
    return {"applied": True, "invalid_reason": None, "grid": grid}


# Cap retries so a misconfigured (size, kind) pair fails fast with a
# clear error rather than spinning forever. build_factory's rejection
# sampler usually succeeds in a handful of tries; 200 is generous.
_LESSON_RETRY_BUDGET = 200


def _build_with_retry(kind: LessonKind, size: int, seed: int) -> tuple[Factory, int]:
    """Build a factory of `kind`, returning it with the seed that produced it.

    `build_factory` returns None when its random layout search doesn't find
    a valid configuration; we follow the docstring's recommended retry
    idiom of advancing the seed and trying again."""
    attempt_seed = int(seed)
    for _ in range(_LESSON_RETRY_BUDGET):
        factory = build_factory(size=size, kind=kind, seed=attempt_seed)
        if factory is not None:
            return factory, attempt_seed
        attempt_seed += 1
    raise RuntimeError(
        f"build_factory returned None for {_LESSON_RETRY_BUDGET} consecutive "
        f"seeds (kind={kind.name}, size={size}, start_seed={seed}) — the "
        f"grid may be too small for this lesson."
    )


def _load_lesson(
    kind_name: str, seed: int, size: int, num_missing_entities: int = 0
) -> dict:
    """Build a complete factory of the given lesson kind + seed and
    return its grid in the JSON format the frontend expects.

    When ``num_missing_entities > 0`` the factory is handed to
    :func:`blank_entities` — the *same* removal path SFT uses to turn a
    solved factory into a (partial, completion) training pair. It applies
    each lesson's own ``protected_positions`` rules and removes whole
    multi-tile entities (splitters, assemblers) as single units, so the
    UI gets a partially-completed factory in exactly the shape the model
    was trained to complete. ``num_missing_entities=0`` (the default)
    leaves the factory fully generated.

    The seed that actually produced the factory is returned as
    ``used_seed`` so the UI can show it; ``next_seed = used_seed + 1`` is
    what the UI advances to so repeated clicks produce distinct
    variants."""
    try:
        kind = LessonKind[kind_name]
    except KeyError as e:
        raise ValueError(f"unknown lesson kind: {kind_name!r}") from e
    num_missing_entities = max(0, int(num_missing_entities))
    factory, used_seed = _build_with_retry(kind, size, seed)
    # Blank entities with the factory's own seed so repeated clicks at the
    # same (kind, seed, N) are reproducible. N=0 removes nothing → the
    # partial world is the full factory.
    partial_CWH, num_removed = blank_entities(
        factory, num_missing_entities=num_missing_entities, seed=used_seed
    )
    return {
        "size": size,
        "grid": world_CWH_to_grid(partial_CWH),
        "used_seed": used_seed,
        "next_seed": used_seed + 1,
        "total_entities": int(factory.total_entities),
        "num_removed": int(num_removed),
    }


@functools.lru_cache(maxsize=4)
def _lesson_previews(size: int) -> dict:
    """Every lesson's seed-0 reference at `size`, for the page's hover previews."""
    previews = {}
    for kind in LessonKind:
        try:
            previews[kind.name] = _load_lesson(kind.name, 0, size)
        except Exception as e:
            previews[kind.name] = {"error": f"{type(e).__name__}: {e}"}
    return previews


def render_graph_png(grid: list[list[dict]]) -> dict:
    """Build the world, construct its graph via the Rust engine, and return a
    base64 PNG plus text describing the nodes/edges/throughput."""
    world = build_world(grid)
    G = build_graph_nx(world)
    if len(G.nodes) == 0:
        return {
            "png": "",
            "info": "No entities placed yet — drop something onto the grid.",
            "edges": [],
        }

    fig = plt.figure(figsize=(max(6, len(G.nodes) ** 0.5 * 3),
                              max(4, len(G.nodes) ** 0.5 * 3)))
    try:
        plot_flow_network(G)
        # plot_flow_network calls plt.show() which is a no-op under Agg;
        # the active figure is still the one we just created via
        # plot_flow_network's plt.figure(...) call. Use plt.gcf() so we
        # capture whichever figure is current.
        buf = io.BytesIO()
        plt.gcf().savefig(buf, format="png", bbox_inches="tight", dpi=110)
        png_b64 = base64.b64encode(buf.getvalue()).decode()
    finally:
        plt.close("all")

    edges = [[u, v] for u, v in G.edges]
    info = f"{len(G.nodes)} nodes · {len(edges)} edges"
    return {"png": png_b64, "info": info, "edges": edges}


def _throughput(world_WHC: torch.Tensor) -> dict:
    """Simulate a world's throughput, as fields to merge into a response.

    Split out of :func:`render_graph_png` — where the number used to arrive
    bundled with the graph image — because it costs ~0.1 ms against that
    function's ~1 s of matplotlib. At that price every response that has
    already built the world carries it, so the readout never waits on a
    request of its own.

    ``thput`` is ``None`` when there is no meaningful answer — an empty grid,
    or a failed simulation — so the UI can stay neutral instead of reporting
    a factory as blocked.
    """
    arr = world_WHC.numpy().astype(np.int64)
    if not arr[:, :, Channel.ENTITIES.value].any():
        return {"thput": None, "note": "nothing placed yet"}
    try:
        thput, unreachable = factorion_rs.simulate_throughput(arr)
    except Exception as e:
        return {"thput": None, "note": f"throughput failed: {e}"}
    return {"thput": float(thput), "unreachable": int(unreachable)}


class _Flow(dict):
    """A mapping the fixtures write inline (``- { x: 0, y: 0, item: … }``)."""


class _Block(str):
    """A string the fixtures write as a literal block scalar (``factory: |``)."""


# Scoped to its own Dumper: `yaml.add_representer` would mutate the global one
# for everything that imports this module.
class _FixtureDumper(yaml.Dumper):
    pass


_FixtureDumper.add_representer(
    _Flow,
    lambda d, x: d.represent_mapping("tag:yaml.org,2002:map", x, flow_style=True),
)
_FixtureDumper.add_representer(
    _Block,
    lambda d, x: d.represent_scalar("tag:yaml.org,2002:str", x, style="|"),
)


_REPO_ROOT = Path(__file__).resolve().parent.parent


def _git_commit(root: Path = _REPO_ROOT) -> Optional[str]:
    """The checkout's short sha, `+dirty` when the tree has uncommitted
    changes. ``None`` when git is missing or `root` isn't a repo — plenty of
    environments are neither, and a fixture is still worth copying."""
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()

    try:
        sha = run("rev-parse", "--short", "HEAD")
        dirty = bool(run("status", "--porcelain"))
    except (OSError, subprocess.SubprocessError):
        return None
    return f"commit {sha}{' +dirty' if dirty else ''}" if sha else None


def _provenance(source: Optional[str]) -> Optional[str]:
    """One line on where a copied factory came from: what produced it, the
    checkout it was produced at, and when. Every part is best-effort, so an
    environment missing git or a usable clock simply contributes less;
    ``None`` when nothing at all is known."""
    try:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, ValueError):
        stamp = None
    known = [part for part in (source, _git_commit(), stamp) if part]
    if not known:
        return None
    return "Captured from the factory builder — " + ", ".join(known) + "."


def factory_yaml(grid: list[list[dict]], source: Optional[str] = None) -> str:
    """Serialise a grid as a textual test fixture, ready to paste into
    ``factorion_rs/tests/factories/*.yaml``.

    ``source`` describes what produced the factory (e.g. ``"MOVE_ONE_ITEM
    seed 4"``); it is folded into a generated ``description:`` alongside the
    commit and timestamp. The caller owns that wording because only the page
    knows whether a grid is a generator's output, a model's rebuild, or a
    hand-edit of either.

    The `factory:` block comes from the canonical renderer and `throughput:`
    from the engine's own per-sink deliveries, so the fixture asserts what the
    engine computes for this exact world. The FOOTPRINT channel has no fixture
    form and is dropped — it gates placement legality, not throughput. Nor does
    a sink with no item bound, so those are left out of `throughput:`.
    """
    world_WHC = build_world(grid)
    protos = {it.name: it for it in items.values()}

    # One binding per entity: `items:` resolves a coordinate to the whole
    # footprint, so a 3x3 assembler needs one line rather than nine. Reading
    # order reaches a multi-tile entity's anchor first, so the rest of its
    # footprint is always still ahead of the walk.
    claimed: set[tuple[int, int]] = set()
    bindings: list[_Flow] = []
    for y, row in enumerate(grid):
        for x, cell in enumerate(row):
            ent_name = cell.get("entity") or "empty"
            if ent_name == "empty" or (x, y) in claimed:
                continue
            proto = protos[ent_name]
            tiles = factorion_rs.py_entity_tiles(
                x, y,
                Direction[cell.get("direction", "NONE")].value,
                proto.width, proto.height,
            )
            claimed.update(map(tuple, tiles or []))
            item_name = cell.get("item") or "empty"
            if item_name != "empty":
                bindings.append(_Flow(x=x, y=y, item=item_name))

    throughput = [
        _Flow(item=item, per_second=rate)
        for _x, _y, item, rate in factorion_rs.py_sink_deliveries(
            world_WHC.numpy().astype(np.int64)
        )
        if item is not None
    ]
    doc: dict = {}
    note = _provenance(source)
    if note:
        doc["description"] = _Block(note + "\n")
    if bindings:
        doc["items"] = bindings
    if throughput:
        doc["throughput"] = throughput
    doc["factory"] = _Block(render_factory(world_WHC.permute(2, 0, 1)) + "\n")
    # `width` off so a long flow mapping is never wrapped mid-entry;
    # `allow_unicode` so a non-ASCII description keeps its block style instead
    # of being escaped into a quoted scalar.
    return yaml.dump(
        doc, Dumper=_FixtureDumper, sort_keys=False,
        width=10**9, allow_unicode=True,
    )


# ── Model inference ──────────────────────────────────────────────────────────

_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
_MODEL: Optional[DenoiserUNet] = None
_CHECKPOINT_PATH: Optional[str] = None
# How the current checkpoint was loaded. Surfaced to the UI so it can
# render meaningful info (e.g. wandb run id + link) instead of the
# anonymous /tmp download path. Either {"kind": "local", "path": "..."}
# or {"kind": "wandb", "run_id", "run_url", "run_name", "artifact"}.
_CHECKPOINT_SOURCE: Optional[dict] = None
# The loaded run's generation schedule, so chains here match its val/ metrics.
_GEN = SftArgs()


def _load_checkpoint(path: str, config: Optional[dict] = None) -> None:
    """Load a denoiser checkpoint, generating with its run's `config` where
    one is known. Does NOT touch _CHECKPOINT_SOURCE — the caller knows the
    provenance (local vs wandb) and sets it after this returns."""
    global _MODEL, _CHECKPOINT_PATH, _GEN
    _MODEL = load_denoiser(path, _DEVICE).eval()
    _CHECKPOINT_PATH = path
    gen = ("gen_steps", "gen_refine", "gen_temperature")
    _GEN = SftArgs(**{k: v for k, v in (config or {}).items() if k in gen})
    print(f"Loaded checkpoint {path} ({_MODEL.arch}, device={_DEVICE})")


def _model() -> DenoiserUNet:
    if _MODEL is None:
        raise RuntimeError("no checkpoint loaded — pass --checkpoint at launch")
    return _MODEL


def _model_info() -> dict:
    """Snapshot of the currently-loaded model for the UI's status line.
    Returns `loaded: False` when nothing is loaded yet, so the UI can
    render "(none loaded)" without special-casing on the JS side."""
    if _MODEL is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "path": _CHECKPOINT_PATH,
        "source": _CHECKPOINT_SOURCE,
        "arch": _MODEL.arch,
        "params": sum(p.numel() for p in _MODEL.parameters()),
        "device": str(_DEVICE),
    }


def _swap_model(value: str, project: str, entity: Optional[str]) -> dict:
    """Resolve `value` to a local .pt path, load it, return new model
    info. Called by the /load_model POST endpoint so the user can
    switch models without restarting the server.

    Auto-detects local-vs-wandb: an existing path on disk is loaded as
    local; otherwise the value is treated as a wandb run id. If both
    fail the error from the wandb resolver bubbles up (wandb's error
    is usually the more informative one — "no such file" tells the
    user nothing they don't already know)."""
    global _CHECKPOINT_SOURCE
    value = (value or "").strip()
    if not value:
        raise ValueError("value cannot be empty")
    if Path(value).exists():
        path = value
        source: dict = {"kind": "local", "path": value}
    else:
        path, source = _resolve_wandb_checkpoint(value, project, entity)
    _load_checkpoint(path, source.get("config"))
    _CHECKPOINT_SOURCE = source
    return _model_info()


def _legalize(model: DenoiserUNet, cond: torch.Tensor, toks, conf: torch.Tensor) -> list[np.ndarray]:
    """Each chain's tokens as a valid (C, W, H) world."""
    typ, rec, part = (t.cpu().numpy() for t in toks)
    cond_np, conf_np = cond.cpu().numpy().astype(np.int64), conf.float().cpu().numpy()
    return [
        legalize(cond_np[i, :CH_PART], typ[i], rec[i], part[i] if model.io == "token" else None, conf_np[i])
        for i in range(len(cond_np))
    ]


@torch.no_grad()
def _denoise_step(grid: list[list[dict]], chain: Optional[dict]) -> dict:
    """Advance one chain of `denoise.generate` by one model call.

    The server keeps no state: `chain` is the previous response's ``chain``
    (the tokens the next call reads, and how many steps it has taken), or
    None to start a fresh one from noise around the grid's markers and
    walls. The response's ``grid`` is the step's clean-factory prediction,
    legalized; ``done`` once the greedy refinement reaches a fixed point or
    the step budget runs out."""
    model = _model()
    steps, refine = _GEN.gen_steps, _GEN.gen_refine
    world_CWH = build_world(grid).permute(2, 0, 1).numpy()
    cond = torch.as_tensor(world_to_state(world_CWH), device=_DEVICE)[None]
    free = free_mask(cond)
    if chain is None:
        step, toks = 0, noise_tokens(model.noise_prior, free.shape)
    else:
        step = int(chain["step"])
        toks = tuple(torch.tensor(chain[k], device=_DEVICE)[None] for k in ("type", "recipe", "part"))
    ancestral = step < steps
    logits = model(tokens_to_state(cond, free, *toks))
    new = model.pick(logits, _GEN.gen_temperature if ancestral else 0.0)
    converged = not ancestral and all(bool(((a == b) | ~free).all()) for a, b in zip(new, toks))
    conf = torch.where(free, model.conf(logits, new), 1.0)
    world = _legalize(model, cond, new, conf)[0]
    nxt = new
    t_next = 1.0 - (step + 1) / steps
    if ancestral and t_next > 0:
        nxt = corrupt(new, free, torch.tensor([t_next], device=_DEVICE), model.noise_prior)
    return {
        "grid": world_CWH_to_grid(torch.from_numpy(world)),
        "chain": {"step": step + 1, **{k: t[0].tolist() for k, t in zip(("type", "recipe", "part"), nxt)}},
        "phase": "denoise" if ancestral else "refine",
        "noise": max(t_next, 0.0) if ancestral else 0.0,
        "converged": converged,
        "done": converged or step + 1 >= steps + refine,
        "conf": conf[0].T.float().cpu().numpy().round(3).tolist(),
        **_throughput(torch.from_numpy(world.transpose(1, 2, 0))),
    }


# ── Batched seed scan ────────────────────────────────────────────────────────

SCAN_BATCH_CHAINS = 256
"""Chains denoised per batched `generate` call. A scan larger than one batch
runs as back-to-back groups, so memory is flat in the count, results stream
as each group finishes, and Stop cancels between groups."""


def _num_entities(world_CWH: np.ndarray) -> int:
    ent = world_CWH[Channel.ENTITIES.value]
    area = {it.value: it.width * it.height for it in items.values() if it.name not in ("empty", *_MARKERS)}
    return round(sum(1 / area[v] for v in ent.ravel().tolist() if v in area))


def _thput_raw(world_CWH: np.ndarray) -> float:
    return _throughput(torch.from_numpy(world_CWH.transpose(1, 2, 0))).get("thput") or 0.0


@torch.no_grad()
def _batch_rollout(kinds: list[LessonKind], seeds: list[int], size: int, best_of: int) -> Iterator[dict]:
    """Rebuild each (kind, seed)'s factory from its markers alone, yielding an
    event per factory: `best_of` chains denoise it from pure noise, as
    `denoise.run_denoise_eval` scores it, and it shows its highest-throughput
    chain."""
    model = _model()
    yield {"type": "start", "n": len(seeds)}
    per_group = max(1, SCAN_BATCH_CHAINS // best_of)
    for base in range(0, len(seeds), per_group):
        built = [
            (kind, *_build_with_retry(kind, size, seed))
            for kind, seed in zip(kinds[base : base + per_group], seeds[base : base + per_group])
        ]
        blanks = [world_to_state(blank_entities(factory)[0].numpy()) for _, factory, _ in built]
        cond = torch.as_tensor(np.stack(blanks), device=_DEVICE).repeat_interleave(best_of, 0)
        toks, conf, converged, n_refine = generate(
            model, cond, _GEN.gen_steps, _GEN.gen_refine, _GEN.gen_temperature
        )
        worlds = _legalize(model, cond, toks, conf)
        raws = [_thput_raw(w) for w in worlds]
        for i, (kind, factory, seed) in enumerate(built):
            j = max(range(i * best_of, (i + 1) * best_of), key=lambda j: raws[j])
            raw, world = raws[j], worlds[j]
            yield {
                "type": "result",
                "index": base + i,
                "kind": kind.name,
                "seed": seed,
                "size": size,
                "steps": _GEN.gen_steps + int(n_refine[j]),
                "stopped_by": "converged" if converged[j] else "max_steps",
                "thput_normed": raw / factory.max_throughput if factory.max_throughput > 0 else 0.0,
                "thput_raw": raw,
                "max_throughput": float(factory.max_throughput),
                "num_placed_entities": _num_entities(world),
                "grid": world_CWH_to_grid(torch.from_numpy(world)),
                "solved_grid": world_CWH_to_grid(factory.world_CWH),
            }
        yield {"type": "progress", "step": base + len(built)}
    yield {"type": "done"}


def _batch_rollout_request(payload: dict) -> Iterator[dict]:
    """Turn a /batch_rollout POST body into a stream of scan events.

    Errors are yielded rather than raised: the response headers are already
    on the wire by the time this generator runs, so the browser can only be
    told about a failure in-band.
    """
    try:
        size = int(payload.get("size", 15))
        count = max(1, int(payload.get("count", 10)))
        start_seed = int(payload.get("seed", 0))
        chosen = [LessonKind[name] for name in payload.get("kinds", [])]
        if not chosen:
            raise ValueError("no lessons selected")
        # Kinds cycle before seeds so a stopped scan still covers every lesson.
        kinds = [kind for _ in range(count) for kind in chosen]
        seeds = [start_seed + i for i in range(count) for _ in chosen]
        yield from _batch_rollout(
            kinds=kinds, seeds=seeds, size=size, best_of=max(1, int(payload.get("best_of", 1)))
        )
    except Exception as e:
        traceback.print_exc()
        yield {"type": "error", "error": f"{type(e).__name__}: {e}"}


# Cache palette icons so the page payload stays small per cell.
def _icon_b64(name: str) -> str:
    try:
        return ent_str2b64img(name)
    except Exception:
        return ""


PALETTE_ICONS = {n: _icon_b64(n) for n in PLACEABLE_ENTITIES + ["empty"]}
# Generated lessons can put *any* item in the ITEMS channel (recipe /
# filter), so cache an icon for every known item — not just the curated
# few in NON_PLACEABLE_ITEMS. _icon_b64 already silently returns "" for
# items without a PNG asset.
ITEM_ICONS = {
    it.name: _icon_b64(it.name)
    for it in items.values()
    if it.name != "empty" and it.name not in PALETTE_ICONS
}
# Grid cells reference icons by class, not by inlining the data URI. A scan
# renders up to 128 grids at once and each ~10 KB URI would be repeated per
# occupied cell — ~300 KB of HTML per grid instead of ~5 KB.
ICON_CSS = "\n".join(
    f".ic-{name}{{background-image:url({uri})}}"
    for name, uri in {**PALETTE_ICONS, **ITEM_ICONS}.items()
    if uri
)


def render_index(default_size: int) -> str:
    def _hotbar_slot(idx: int, name: str | None) -> str:
        key_label = str((idx + 1) % 10)
        if name is None:
            return (
                f'<div class="hb-slot empty-slot" data-slot="{idx}" '
                f'title="(unbound)"><div class="hb-key">{key_label}</div></div>'
            )
        if name == "empty":
            return (
                f'<div class="hb-slot eraser" data-slot="{idx}" '
                f'data-entity="empty" draggable="true" title="eraser">'
                f'<div class="hb-key">{key_label}</div>'
                f'<img draggable="false" src="{PALETTE_ICONS["empty"]}" alt="eraser">'
                f'<div class="hb-name">eraser</div></div>'
            )
        display = DISPLAY_NAME.get(name, name)
        return (
            f'<div class="hb-slot" data-slot="{idx}" data-entity="{name}" '
            f'draggable="true" title="{display} ({name})">'
            f'<div class="hb-key">{key_label}</div>'
            f'<img draggable="false" src="{PALETTE_ICONS[name]}" alt="{display}">'
            f'<div class="hb-name">{display}</div></div>'
        )

    hotbar_html = "".join(_hotbar_slot(i, n) for i, n in enumerate(HOTBAR))
    item_options = "".join(
        f'<option value="{n}">{n}</option>'
        for n in (["empty"] + PLACEABLE_ENTITIES + NON_PLACEABLE_ITEMS)
    )
    # The recipe/filter dropdown can hold any item, not just the curated
    # palette set — generated lessons routinely set obscure recipes
    # (e.g. burner_mining_drill) and we want them visible + editable.
    all_item_names = sorted(
        {it.name for it in items.values() if it.name != "empty"}
    )
    all_item_options = "".join(
        f'<option value="{n}">{n}</option>'
        for n in (["empty"] + all_item_names)
    )
    direction_options = "".join(
        f'<option value="{d}">{d}</option>' for d in DIRECTIONS
    )
    misc_options = "".join(
        f'<option value="{m}">{m}</option>' for m in MISC_VALUES
    )
    lesson_options = "".join(
        f'<option value="{k.name}">{k.name}</option>' for k in LessonKind
    )
    scan_kind_checkboxes = "".join(
        f'<label data-lesson="{k.name}"><input type="checkbox" value="{k.name}" checked> '
        f'{k.name}</label>'
        for k in LessonKind
    )

    # Model loader is collapsed by default — switching models is rare
    # compared to using the prediction; surface the active model
    # inline and tuck the form into a <details>.
    model_panel_html = (
        '<div class="model-panel">'
        '<h3 style="margin-top:0.8em;">'
        '<span class="swatch"></span>Model'
        '</h3>'
        '<div class="model-current help" id="model-current">checking…</div>'
        '<div class="model-loader">'
        '<label>switch model'
        '  <input id="model-value" type="text" placeholder="sft_local.pt or run_id">'
        '</label>'
        '<div class="model-buttons">'
        '<button id="model-load" title="Local path if the file exists, else wandb run id">'
        'Load model</button>'
        '</div>'
        '<div id="model-load-status" class="help"></div>'
        '</div>'
        '<h3 style="margin-top:0.8em;">Denoise</h3>'
        '<div id="model-info" class="help">(no chain yet)</div>'
        '<div class="model-buttons">'
        '<button id="model-apply" title="Run one denoising step: a fresh chain starts from noise around the markers">'
        'Denoise step <span class="kbd">a</span></button>'
        '</div>'
        '</div>'
    )

    # [?] help: a click-to-toggle popover. The previous version leaned on
    # the native `title` tooltip, which many browsers render unreliably (or
    # not at all) — so the shortcuts list is now real DOM that always shows.
    help_html = (
        '<span class="help-wrap">'
        '<span class="kbd-help" id="help-toggle" tabindex="0" role="button" '
        'aria-expanded="false" aria-label="Show keyboard and mouse shortcuts" '
        'title="Keyboard &amp; mouse shortcuts (click)">[?]</span>'
        '<div class="help-popover" id="help-popover" hidden>'
        + "<br>".join(HELP_LINES)
        + "</div></span>"
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Factory builder</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 0.6em; color: #222; }}
  h1 {{ margin: 0 0 0.3em; font-size: 1.3em; }}
  .layout {{ display: grid; grid-template-columns: 1fr 260px; gap: 0.8em; }}
  .panel {{ border: 1px solid #ccc; border-radius: 6px; padding: 0.5em; background: #fafafa; }}
  .panel h3 {{ margin: 0 0 0.3em; font-size: 0.85em; text-transform: uppercase; color: #555; }}
  .hotbar {{
    display: flex; gap: 0.25em; flex-wrap: wrap;
    user-select: none; -webkit-user-select: none;
  }}
  .hb-slot {{
    position: relative; width: 52px; height: 66px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 2px;
    border: 2px solid #ddd; border-radius: 4px; background: white;
    cursor: grab; font-size: 0.65em; padding: 2px;
    user-select: none; -webkit-user-select: none;
  }}
  .hb-slot img {{ width: 26px; height: 26px; pointer-events: none; }}
  .hb-slot .hb-key {{
    position: absolute; top: 1px; left: 3px;
    font-size: 0.7em; font-weight: bold; color: #888;
  }}
  .hb-slot .hb-name {{
    font-size: 0.65em; line-height: 1; text-align: center;
    word-break: break-all; pointer-events: none;
  }}
  .hb-slot.active {{ border-color: #28c850; background: #e8ffe8; }}
  .cursor-ghost {{ position: absolute; inset: 0; opacity: 0.5; pointer-events: none; }}
  .hb-slot.eraser {{ background: #fee; }}
  .hb-slot.empty-slot {{
    background: #f4f4f4; cursor: default; color: #bbb;
  }}
  .grid-wrap {{ display: flex; flex-direction: column; align-items: flex-start; gap: 0.4em; min-width: 0; }}
  .grid-graph-row {{
    display: flex; gap: 0.8em; align-items: flex-start;
    width: 100%; min-width: 0;
  }}
  .grid-graph-row > .grid-col {{
    flex: 0 0 auto; display: flex; flex-direction: column; gap: 0.4em;
  }}
  /* The verdict sits above the grid because it is what the user is waiting
     for after a build, and the flag reads before the digits do. */
  .thput {{
    display: flex; align-items: center; gap: 0.55em;
    font-family: monospace; font-size: 0.92em; min-height: 1.5em;
    padding: 0.4em 0.6em;
    border: 1px solid #ddd; border-radius: 5px; background: #fafafa;
  }}
  /* Tints the whole card with the verdict, so it reads from the corner of
     the eye without competing with the grid for attention. */
  .thput.ok {{ border-color: #9ccfad; background: #f6fbf7; }}
  .thput.bad {{ border-color: #e6adad; background: #fdf7f7; }}
  .thput-icon {{ display: inline-flex; }}
  .thput-icon.ok {{ color: #1a7f37; }}
  .thput-icon.bad {{ color: #c62828; }}
  .thput-value {{ font-weight: bold; }}
  .thput-sub {{ color: #888; font-size: 0.9em; }}
  .spinner {{
    width: 11px; height: 11px; flex: 0 0 auto;
    border: 2px solid #d8d8d8; border-top-color: #666;
    border-radius: 50%; animation: spin 0.7s linear infinite;
  }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  .graph-view {{ flex: 1 1 0; min-width: 0; }}
  .graph-view h3 {{ margin: 0 0 0.3em; font-size: 0.85em; text-transform: uppercase; color: #555; }}
  .controls {{ display: flex; gap: 0.4em; flex-wrap: wrap; align-items: center; }}
  .controls input[type=number] {{ width: 4em; }}
  table.grid {{ border-collapse: collapse; }}
  table.grid td {{
    width: 44px; height: 44px; border: 1px solid #bbb; padding: 0;
    position: relative; background: #fff;
  }}
  table.grid td.selected {{ outline: 2px solid #28c850; outline-offset: -2px; }}
  table.grid td.unavailable {{
    background:
      repeating-linear-gradient(45deg,
        rgba(80,80,80,0.12), rgba(80,80,80,0.12) 2px,
        transparent 2px, transparent 8px),
      repeating-linear-gradient(-45deg,
        rgba(80,80,80,0.12), rgba(80,80,80,0.12) 2px,
        transparent 2px, transparent 8px);
  }}
  .cell-inner {{ position: relative; width: 100%; height: 100%; }}
  .cell-inner .ent, .cell-inner .itm {{
    position: absolute; background-size: contain;
    background-repeat: no-repeat; background-position: center;
  }}
  .cell-inner .ent {{ top: 10%; left: 10%; width: 60%; height: 60%; }}
  .cell-inner .itm {{ bottom: 4%; right: 4%; width: 30%; height: 30%; }}
  .cell-inner .arrow {{
    position: absolute; bottom: -1px; left: 2px;
    font-size: 13px; line-height: 13px; color: #222;
  }}
  .cell-inner .misc {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-weight: bold; color: white; text-shadow: 0 0 2px black; font-size: 14px;
  }}
  .cell-inner .xy {{ position: absolute; top: 0; left: 1px; font-size: 7px; opacity: 0.5; }}
  .cell-inner .p-badge {{
    position: absolute; bottom: 0; right: 2px;
    font-size: 10px; font-weight: bold; line-height: 1;
    text-shadow: 0 0 1px white, 0 0 1px white;
    pointer-events: none;
  }}
  .editor label {{ display: block; font-size: 0.8em; margin-top: 0.4em; }}
  .editor select, .editor button {{ width: 100%; padding: 0.2em; }}
  .out-img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .info {{ font-family: monospace; font-size: 0.85em; margin: 0.4em 0; }}
  .help {{ font-size: 0.85em; color: #555; }}
  pre.edges {{
    font-size: 0.75em; max-height: 240px; overflow: auto;
    background: #222; color: #cfc; padding: 0.5em; border-radius: 4px;
  }}
  .model-panel {{ margin-top: 0.8em; }}
  .model-panel pre {{
    font-size: 0.78em; background: #f4f4f4; padding: 0.5em;
    border-radius: 4px; margin: 0.3em 0;
    /* `pre` (not pre-wrap) + overflow-x:auto so long top-p lines
       scroll horizontally instead of wrapping mid-token and breaking
       the column alignment. */
    white-space: pre; overflow-x: auto;
  }}
  .model-panel .model-buttons {{
    display: flex; gap: 0.4em; flex-wrap: wrap; margin-top: 0.3em;
  }}
  .model-panel .swatch {{
    display: inline-block; width: 0.8em; height: 0.8em; border: 1px solid #082466;
    box-shadow: inset 0 0 0 2px #0d47a1;
    vertical-align: middle; margin-right: 0.25em;
  }}
  .copy-yaml {{
    display: inline-flex; align-items: center; justify-content: center;
    /* Fixed box so the ✓/✗ feedback can't resize the button mid-click. */
    min-width: 2.1em; height: 1.6em; vertical-align: middle;
    font-weight: normal; font-size: 12px; padding: 0;
    margin-left: 0.4em; cursor: pointer;
  }}
  .help-wrap {{ position: relative; display: inline-block; }}
  .kbd-help {{
    display: inline-block; font-size: 0.6em; font-weight: normal;
    color: #555; background: #eee; border: 1px solid #bbb;
    border-radius: 4px; padding: 0 0.4em; vertical-align: middle;
    margin-left: 0.5em; cursor: pointer; user-select: none;
  }}
  .kbd-help:hover, .kbd-help:focus {{ background: #fff5cc; outline: none; }}
  .help-popover {{
    position: absolute; top: 1.9em; left: 0; z-index: 50;
    width: max-content; max-width: 360px;
    background: #fffbe6; border: 1px solid #d8c97a; border-radius: 6px;
    padding: 0.5em 0.75em; font-size: 0.8rem; font-weight: normal;
    color: #333; line-height: 1.65; text-align: left;
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.18);
  }}
  /* The UA stylesheet's [hidden] {{display:none}} loses to .help-popover
     (equal specificity, author rule wins), so re-assert it explicitly. */
  .help-popover[hidden] {{ display: none; }}
  .sel-coord {{ font-family: monospace; color: #888; font-weight: normal; }}
  .kbd {{
    display: inline-block; min-width: 1em; padding: 0 0.3em;
    margin-left: 0.3em; font-size: 0.8em; font-family: monospace;
    color: #333; background: #f4f4f4; border: 1px solid #bbb;
    border-radius: 3px; line-height: 1.2;
  }}
  .action-row button {{ display: inline-flex; align-items: center; }}
  details summary {{
    cursor: pointer; font-size: 0.85em; color: #555;
    user-select: none; padding: 0.2em 0;
  }}
  details.edges-details {{ margin-top: 0.4em; }}
  .tabs {{ display: flex; gap: 0.3em; margin-bottom: 0.5em; }}
  .tabs button {{
    padding: 0.3em 0.9em; font-size: 0.85em; cursor: pointer;
    border: 1px solid #ccc; border-radius: 5px 5px 0 0;
    background: #eee; color: #555; border-bottom-color: #ccc;
  }}
  .tabs button.active {{
    background: #fff; color: #111; font-weight: bold; border-bottom-color: #fff;
  }}
  .scan-kinds {{ columns: 20em; column-gap: 1.5em; font-size: 0.78em; margin: 0.4em 0; }}
  .scan-kinds label {{ display: block; white-space: nowrap; }}
  .scan-summary {{
    font-family: monospace; font-size: 0.85em; margin: 0.5em 0;
    padding: 0.4em 0.6em; background: #f4f4f4; border-radius: 4px;
  }}
  .scan-stats {{ margin: 0.5em 0; }}
  .scan-stats table {{ border-collapse: collapse; font-size: 0.78em; }}
  .scan-stats th, .scan-stats td {{
    padding: 0.12em 0.9em 0.12em 0; text-align: right;
    font-family: monospace; font-weight: normal;
  }}
  .scan-stats th {{ color: #666; border-bottom: 1px solid #ddd; }}
  .scan-stats th:first-child, .scan-stats td:first-child {{ text-align: left; }}
  .scan-stats td.kind {{ font-family: inherit; }}
  .scan-stats a, .scan-summary a, .scan-card .hd a, .scan-card .sub a {{ color: inherit; }}
  .scan-stats tr.active {{ background: #e8f0ff; }}
  .scan-filter {{
    margin: 0.4em 0; padding: 0.35em 0.6em; font-size: 0.85em;
    border: 1px solid transparent; border-radius: 4px;
  }}
  .scan-filter.active {{ background: #fff4d6; border-color: #e0b84a; }}
  .scan-filter input {{ font-family: monospace; }}
  .scan-filter a {{ font-weight: bold; margin-left: 0.6em; }}
  .scan-filter .err {{ color: #b00020; }}
  .popup {{
    position: absolute; z-index: 20; background: #fff; padding: 0.4em;
    border: 1px solid #999; border-radius: 5px; font-size: 0.8em;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18); pointer-events: none;
  }}
  .popup .hd {{ font-weight: bold; }}
  .popup .sub {{ color: #666; margin-bottom: 0.3em; }}
  .scan-results {{ display: flex; flex-wrap: wrap; gap: 0.6em; }}
  .scan-card {{
    border: 1px solid #ccc; border-left-width: 5px; border-radius: 5px;
    padding: 0.35em; background: #fff; cursor: pointer;
    flex: 0 0 auto; width: max-content;
  }}
  .scan-card:hover {{ background: #f6fbff; border-color: #0d47a1; }}
  .scan-card .hd {{ font-size: 0.72em; font-weight: bold; }}
  .scan-card .sub {{ font-size: 0.7em; color: #666; font-family: monospace; }}
  /* Width 0 keeps the text out of the card's max-content width, so every
     card is exactly as wide as its grids and the gallery tiles into
     aligned columns; min-width then wraps the text back to that width
     instead of the longest lesson name stretching the card. */
  .scan-card .hd, .scan-card .sub {{ width: 0; min-width: 100%; }}
  .scan-card .pair {{ display: flex; gap: 0.4em; align-items: flex-start; }}
  .scan-card figure {{ margin: 0.2em 0 0; }}
  .scan-card figcaption {{
    font-size: 0.62em; color: #888; text-align: center;
  }}
  table.mini {{ border-collapse: collapse; }}
  table.mini td {{
    width: 22px; height: 22px; border: 1px solid #e2e2e2; padding: 0;
    position: relative; background: #fff;
  }}
  table.mini td.unavailable {{ background: #e8e8e8; }}
  table.mini .arrow {{ font-size: 9px; line-height: 9px; }}
  table.mini .misc {{ font-size: 9px; }}
{ICON_CSS}
</style></head><body>

<h1>Factory builder
  {help_html}
</h1>

<div class="tabs" id="tabs">
  <button class="active" data-tab="build">Build</button>
  <button data-tab="scan">Scan seeds</button>
</div>

<div id="tab-build">
<div class="layout">

  <div class="grid-wrap">
    <div class="hotbar" id="hotbar">{hotbar_html}</div>
    <div class="help" id="cursor-info"></div>
    <div class="controls">
      <label>size <input id="size" type="number" min="2" max="20" value="{default_size}"></label>
      <button id="resize" title="Resize the grid and clear all cells">resize / clear <span class="kbd">c</span></button>
      <button id="export" title="Copy {{size, grid}} JSON to clipboard">copy state JSON</button>
      <span id="edit-status" class="help"></span>
    </div>
    <div class="controls action-row">
      <label>lesson
        <select id="lesson-kind">{lesson_options}</select>
      </label>
      <label>seed <input id="lesson-seed" type="number" value="0" step="1"></label>
      <label title="Remove this many entity units from the generated factory using the lesson's own removal rules (0 = fully generated)">
        entities to clear
        <input id="lesson-clear" type="number" min="0" value="0" step="1">
      </label>
      <button id="lesson-generate" title="Build a factory of the chosen lesson kind at the given seed (optionally clearing N entities), then bump the seed for the next click">
        Generate lesson <span class="kbd">g</span>
      </button>
      <span id="lesson-status" class="help"></span>
    </div>
    <div class="grid-graph-row">
      <div class="grid-col">
        <div class="thput" id="thput"></div>
        <div id="grid-host"></div>
      </div>
      <div class="graph-view">
        <h3>Graph
          <button class="copy-yaml" id="copy-yaml"
                  title="Copy this factory as a YAML test fixture">{COPY_ICON}</button>
          <button class="copy-yaml copy-ascii" id="copy-ascii"
                  title="Copy this factory as a two-character text render (render_factory)">txt</button>
        </h3>
        <div class="info" id="info"></div>
        <img id="out-img" class="out-img" alt="" style="display:none">
        <details class="edges-details">
          <summary>graph edges</summary>
          <pre class="edges" id="edges"></pre>
        </details>
      </div>
    </div>
  </div>

  <div class="panel editor" id="editor">
    <h3>Selected cell <span id="sel-info" class="sel-coord"></span></h3>
    <label>entity
      <select id="ed-entity">{item_options}</select>
    </label>
    <label>direction
      <select id="ed-direction">{direction_options}</select>
    </label>
    <label>item (recipe / filter)
      <select id="ed-item">{all_item_options}</select>
    </label>
    <label>misc
      <select id="ed-misc">{misc_options}</select>
    </label>
    <label>footprint
      <select id="ed-footprint">
        <option value="AVAILABLE">AVAILABLE</option>
        <option value="UNAVAILABLE">UNAVAILABLE</option>
      </select>
    </label>
    <button id="clear-cell" style="margin-top:0.6em;">clear cell</button>

    {model_panel_html}
  </div>

</div>
</div>

<div id="tab-scan" hidden>
  <div class="controls action-row">
    <label title="How many seeds of each ticked lesson to add per click">
      seeds per lesson <input id="scan-count" type="number" min="1" value="50">
    </label>
    <label title="Seed the next batch starts from. Advances by the seed count after each run, so repeated clicks keep drawing fresh factories.">
      start seed <input id="scan-seed" type="number" value="0" step="1">
    </label>
    <label title="Chains denoised per factory; the gallery shows the one with the highest throughput, as val/thput@N scores it">
      best of <input id="scan-best-of" type="number" min="1" value="8" style="width:4.5em">
    </label>
    <label title="Also draw the generator's own solution next to each rebuild">
      <input id="scan-ref" type="checkbox"> show reference
    </label>
    <label>sort
      <select id="scan-sort">
        <option value="worst" selected>worst first</option>
        <option value="best">best first</option>
        <option value="seed">run order</option>
      </select>
    </label>
    <button id="scan-run" title="Blank each seed's factory to its markers, denoise it from noise, and add every final factory to the gallery">
      Run scan
    </button>
    <button id="scan-stop" disabled>Stop</button>
    <button id="scan-clear-results" title="Throw away everything scanned so far">clear</button>
  </div>
  <div class="controls">
    lessons
    <button type="button" id="scan-kinds-all">all</button>
    <button type="button" id="scan-kinds-none">none</button>
  </div>
  <div class="scan-kinds" id="scan-kinds">{scan_kind_checkboxes}</div>
  <div class="scan-summary" id="scan-summary">no scan yet</div>
  <div class="scan-stats" id="scan-stats"></div>
  <div class="scan-filter" id="scan-filter">
    <label title="Space-separated clauses that must all hold. Operators: = != &lt; &lt;= &gt; &gt;=. Fields: lesson, thput, raw, steps, seed, placed, converged. A comma list after = means any of.">
      filter <input id="scan-q" size="60" spellcheck="false"
        placeholder="lesson=A,B thput>=0.5 converged=false steps<=12">
    </label>
    <span id="scan-filter-status"></span>
  </div>
  <div class="scan-results" id="scan-results"></div>
</div>

<script>
const HOTBAR = {json.dumps(HOTBAR)};
const DIR_ARROW = {{ NONE: '', NORTH: '↑', EAST: '→', SOUTH: '↓', WEST: '←' }};
const MISC_GLYPH = {{ NONE: '', UNDERGROUND_DOWN: '▼', UNDERGROUND_UP: '▲' }};
const DIR_CYCLE = ['NORTH', 'EAST', 'SOUTH', 'WEST'];
// Tiles the model is less sure of than this get a confidence badge: they are
// the ones the next step is likely to change.
const UNSURE_BELOW = 0.9;
const OK_ICON = '{OK_ICON}';
const BAD_ICON = '{BAD_ICON}';
// `modelLoaded` is set by refreshModelInfo() at startup and after each
// successful /load_model call. Denoise calls are gated on it so we
// don't pester the server with /denoise when no checkpoint is loaded.
let modelLoaded = false;

let SIZE = {default_size};
let grid = [];           // grid[y][x] = cell dict
let selected = null;     // {{x, y}} or null
let activeHotbar = null; // 0..9 or null
let cursor = null;   // an entity picked up with z: {{entity, direction, item, misc}}
let hovered = null;  // the grid cell under the mouse, {{x, y}}
let denoised = null;     // last /denoise response, while `grid` is still its output
let gridSource = null;   // what produced `grid`, or null if hand-built
let gridSnapshot = '';   // `grid` as adopted, to detect later edits
let autoApplying = false;
let autoApplyGeneration = 0;
let autoApplyFrame = null;
let autoApplyHoldTimer = null;
let applyKeyHeld = false;

function emptyCell() {{
  return {{
    entity: 'empty', direction: 'NONE', item: 'empty',
    misc: 'NONE', footprint: 'AVAILABLE',
  }};
}}

function newGrid(n) {{
  const g = [];
  for (let y = 0; y < n; y++) {{
    const row = [];
    for (let x = 0; x < n; x++) row.push(emptyCell());
    g.push(row);
  }}
  return g;
}}

// Color for the per-tile p% badge. Saturated orange at high p,
// desaturates and shifts toward yellow as confidence drops, and lands
// at neutral grey for near-zero — the eye reads "strong / hedging /
// noise" before reading the digits.
function pBadgeColor(p) {{
  const q = Math.max(0, Math.min(p, 1));
  const hue = 25 + (1 - q) * 25;            // 25° orange -> 50° yellow
  const sat = Math.round(Math.min(q * 1.5, 1) * 90);  // 0% at p=0 -> 90% at p≥0.67
  // Slightly darker than 50% so the digits stay legible as foreground
  // text on a (usually) white cell background.
  return `hsl(${{hue}}, ${{sat}}%, 42%)`;
}}

// One cell's glyph layer, shared by the interactive grid and the scan
// gallery so the two can never draw the same world differently.
function cellGlyphs(c) {{
  let html = '';
  if (c.entity && c.entity !== 'empty')
    html += `<i class="ent ic-${{c.entity}}"></i>`;
  if (c.item && c.item !== 'empty')
    html += `<i class="itm ic-${{c.item}}"></i>`;
  const arrow = DIR_ARROW[c.direction] || '';
  if (arrow) html += `<div class="arrow">${{arrow}}</div>`;
  const m = MISC_GLYPH[c.misc] || '';
  if (m) html += `<div class="misc">${{m}}</div>`;
  return html;
}}

// The server keeps no chain state, so the chain is only valid while `grid`
// is exactly what its last step returned: any other edit starts a new one.
function currentChain() {{
  if (denoised && JSON.stringify(grid) !== denoised.gridJson) denoised = null;
  return denoised;
}}

function renderGrid() {{
  const host = document.getElementById('grid-host');
  const conf = currentChain() ? denoised.conf : null;
  const tbl = document.createElement('table');
  tbl.className = 'grid';
  for (let y = 0; y < SIZE; y++) {{
    const tr = document.createElement('tr');
    for (let x = 0; x < SIZE; x++) {{
      const td = document.createElement('td');
      td.dataset.x = x; td.dataset.y = y;
      const c = grid[y][x];
      if (c.footprint === 'UNAVAILABLE') td.classList.add('unavailable');
      if (selected && selected.x === x && selected.y === y) td.classList.add('selected');

      const inner = document.createElement('div');
      inner.className = 'cell-inner';
      let html = `<div class="xy">${{x}},${{y}}</div>` + cellGlyphs(c);
      const p = conf ? conf[y][x] : 1;
      if (p < UNSURE_BELOW) {{
        const pct = p >= 0.01 ? Math.round(p * 100) + '%' : '<1%';
        html += `<div class="p-badge" style="color:${{pBadgeColor(1 - p)}}">${{pct}}</div>`;
      }}
      inner.innerHTML = html;
      if (cursor && hovered && hovered.x === x && hovered.y === y) {{
        inner.appendChild(cursorGhost());
      }}
      td.appendChild(inner);

      td.addEventListener('mouseenter', () => {{
        hovered = {{ x, y }};
        if (cursor) inner.appendChild(cursorGhost());
      }});
      td.addEventListener('mouseleave', () => {{
        if (hovered && hovered.x === x && hovered.y === y) hovered = null;
        const ghost = inner.querySelector('.cursor-ghost');
        if (ghost) ghost.remove();
      }});
      td.addEventListener('click', () => {{
        selected = {{x, y}};
        if (cursor) {{
          handEdit([{{ x, y, cell: cursor }}]);
          return;
        }}
        if (activeHotbar !== null) {{
          const ent = HOTBAR[activeHotbar];
          if (ent !== null) {{
            placeEntity(x, y, ent);
            return;
          }}
        }}
        renderGrid(); syncEditor();
      }});
      td.addEventListener('contextmenu', (ev) => {{
        ev.preventDefault();
        handEdit([{{ x, y, cell: EMPTY }}]);
      }});
      // Populated cells are draggable: dragging one onto another tile
      // moves the *entire* cell state (entity, direction, item, misc,
      // footprint) and clears the source. Hotbar and tile drags share
      // the text/plain MIME via a {{kind}}-tagged JSON payload.
      if (c.entity !== 'empty') {{
        td.draggable = true;
        td.addEventListener('dragstart', (ev) => {{
          ev.dataTransfer.setData(
            'text/plain',
            JSON.stringify({{ kind: 'tile', from: {{ x, y }} }}),
          );
          ev.dataTransfer.effectAllowed = 'move';
        }});
      }}
      td.addEventListener('dragover', (ev) => ev.preventDefault());
      td.addEventListener('drop', (ev) => {{
        ev.preventDefault();
        const raw = ev.dataTransfer.getData('text/plain');
        if (!raw) return;
        let payload;
        try {{ payload = JSON.parse(raw); }} catch (_) {{ return; }}
        if (payload.kind === 'palette') {{
          placeEntity(x, y, payload.entity);
        }} else if (payload.kind === 'tile') {{
          const fx = payload.from.x, fy = payload.from.y;
          if (fx === x && fy === y) return;
          selected = {{ x, y }};
          handEdit([{{ x: fx, y: fy, cell: EMPTY }}, {{ x, y, cell: grid[fy][fx] }}]);
        }}
      }});
      tr.appendChild(td);
    }}
    tbl.appendChild(tr);
  }}
  host.replaceChildren(tbl);
}}

function syncEditor() {{
  const info = document.getElementById('sel-info');
  if (!selected) {{ info.textContent = ''; return; }}
  const c = grid[selected.y][selected.x];
  info.textContent = `(${{selected.x}}, ${{selected.y}})`;
  document.getElementById('ed-entity').value = c.entity;
  document.getElementById('ed-direction').value = c.direction;
  document.getElementById('ed-item').value = c.item;
  document.getElementById('ed-misc').value = c.misc;
  document.getElementById('ed-footprint').value = c.footprint;
}}

function bindEditor() {{
  const map = {{
    'ed-entity': 'entity', 'ed-direction': 'direction',
    'ed-item': 'item', 'ed-misc': 'misc', 'ed-footprint': 'footprint',
  }};
  for (const [id, field] of Object.entries(map)) {{
    document.getElementById(id).addEventListener('change', (ev) => {{
      if (!selected) return;
      if (field === 'footprint') {{
        grid[selected.y][selected.x].footprint = ev.target.value;
        renderGrid();
        scheduleCompute();
        return;
      }}
      handEdit([{{ x: selected.x, y: selected.y, field, value: ev.target.value }}]);
    }});
  }}
  document.getElementById('clear-cell').addEventListener('click', clearSelected);
}}

function renderHotbar() {{
  document.querySelectorAll('.hb-slot').forEach(el => {{
    const idx = parseInt(el.dataset.slot, 10);
    el.classList.toggle('active', idx === activeHotbar);
  }});
}}

function setActiveHotbar(idx) {{
  if (idx !== null && HOTBAR[idx] === null) return;
  activeHotbar = (activeHotbar === idx) ? null : idx;
  renderHotbar();
  if (cursor) setCursor(null);
}}

function cursorGhost() {{
  const ghost = document.createElement('div');
  ghost.className = 'cursor-ghost';
  ghost.innerHTML = cellGlyphs(cursor);
  return ghost;
}}

function setCursor(cell) {{
  cursor = cell;
  document.getElementById('cursor-info').textContent = cursor
    ? 'holding ' + cursor.entity + ' ' + (DIR_ARROW[cursor.direction] || '') +
      (cursor.item !== 'empty' ? ' · ' + cursor.item : '') +
      ' — click to place, r rotates, Esc drops'
    : '';
  renderGrid();
}}

// Factorio's pipette: pick up the hovered entity, rotation and recipe included.
function pickHovered() {{
  if (!hovered) return;
  const c = grid[hovered.y][hovered.x];
  if (activeHotbar !== null) {{ activeHotbar = null; renderHotbar(); }}
  setCursor(c.entity === 'empty' ? null : {{
    entity: c.entity, direction: c.direction, item: c.item, misc: c.misc,
  }});
}}

function rotateCursor(cw) {{
  const i = DIR_CYCLE.indexOf(cursor.direction);
  if (i < 0) return;
  setCursor({{ ...cursor, direction: DIR_CYCLE[(i + (cw ? 1 : -1) + 4) % 4] }});
}}

function bindHotbar() {{
  document.querySelectorAll('.hb-slot').forEach(el => {{
    const idx = parseInt(el.dataset.slot, 10);
    if (HOTBAR[idx] === null) return;
    el.addEventListener('dragstart', (ev) => {{
      ev.dataTransfer.setData(
        'text/plain',
        JSON.stringify({{ kind: 'palette', entity: el.dataset.entity }}),
      );
    }});
    el.addEventListener('click', () => setActiveHotbar(idx));
  }});
}}

const EMPTY = {{ entity: 'empty', direction: 'NONE', item: 'empty', misc: 'NONE' }};

// The server owns entity footprints, so every hand edit goes through it and
// multi-tile entities are placed and removed whole. Edits queue so each one
// sees the grid the previous one produced.
let editQueue = Promise.resolve();
function handEdit(ops) {{
  editQueue = editQueue.then(async () => {{
    const status = document.getElementById('edit-status');
    try {{
      const resp = await fetch('/hand_edit', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/json' }},
        body: JSON.stringify({{ grid, ops }}),
      }});
      const data = await resp.json();
      if (data.error || !data.applied) {{
        status.textContent = 'edit rejected: ' +
          (data.error || data.invalid_reason || 'invalid placement');
        syncEditor();
        return;
      }}
      status.textContent = '';
      grid = data.grid;
      renderGrid(); syncEditor();
      scheduleCompute();
    }} catch (e) {{
      status.textContent = 'edit failed: ' + e;
    }}
  }});
  return editQueue;
}}

function placeEntity(x, y, ent) {{
  const old = grid[y][x];
  const direction = ent === 'assembling_machine_1' ? 'NONE'
    : (old.direction !== 'NONE' ? old.direction : 'EAST');
  selected = {{x, y}};
  handEdit([{{ x, y, cell: {{ ...EMPTY, entity: ent, direction }} }}]);
}}

function rotateSelected(cw) {{
  if (!selected) return;
  const c = grid[selected.y][selected.x];
  if (c.entity === 'empty' || c.entity === 'assembling_machine_1') return;
  const i = DIR_CYCLE.indexOf(c.direction);
  const direction = i < 0
    ? (cw ? 'NORTH' : 'WEST')
    : DIR_CYCLE[(i + (cw ? 1 : -1) + 4) % 4];
  handEdit([{{ x: selected.x, y: selected.y, field: 'direction', value: direction }}]);
}}

function clearSelected() {{
  if (!selected) return;
  handEdit([{{ x: selected.x, y: selected.y, cell: EMPTY }}]);
}}

// The verdict above the grid. The flag is the whole point: it answers "did my
// factory work?" before anyone has to know what a good number looks like.
const THPUT_LABEL = '<span>factory throughput:</span>';

function showThputPending() {{
  const el = document.getElementById('thput');
  el.className = 'thput';
  el.innerHTML = THPUT_LABEL +
    '<span class="spinner"></span><span class="thput-sub">calculating…</span>';
}}

function showThput(data) {{
  const el = document.getElementById('thput');
  if (data.thput === null || data.thput === undefined) {{
    el.className = 'thput';
    el.innerHTML = THPUT_LABEL +
      '<span class="thput-sub">' + escHtml(data.note || 'unavailable') + '</span>';
    return;
  }}
  const flowing = data.thput > 0;
  el.className = flowing ? 'thput ok' : 'thput bad';
  el.innerHTML = THPUT_LABEL +
    '<span class="thput-icon ' + (flowing ? 'ok' : 'bad') + '">' +
      (flowing ? OK_ICON : BAD_ICON) + '</span>' +
    '<span class="thput-value">' + data.thput.toFixed(2) +
      ' items per second</span>' +
    (flowing && data.unreachable
      ? '<span class="thput-sub">' + data.unreachable + ' unreachable</span>'
      : '');
}}

async function computeThroughput() {{
  try {{
    const resp = await fetch('/throughput', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ grid }}),
    }});
    showThput(await resp.json());
  }} catch (e) {{
    showThput({{ thput: null, note: 'throughput failed: ' + e }});
  }}
}}

let _thputTimer = null;
let _graphTimer = null;
function cancelCompute() {{
  clearTimeout(_thputTimer);
  clearTimeout(_graphTimer);
}}
// Every grid change ends in scheduleCompute, so recording history here covers
// every edit path, and a held-a run (which defers it) becomes one undo step.
const HISTORY_LIMIT = 500;
let undoStack = [];
let redoStack = [];
let historyState = null;
function recordHistory() {{
  const state = JSON.stringify({{ size: SIZE, grid }});
  if (state === historyState) return;
  if (historyState !== null) {{
    undoStack.push(historyState);
    if (undoStack.length > HISTORY_LIMIT) undoStack.shift();
    redoStack = [];
  }}
  historyState = state;
}}
function stepHistory(from, to) {{
  if (!from.length || autoApplying) return;
  to.push(historyState);
  historyState = from.pop();
  const state = JSON.parse(historyState);
  SIZE = state.size;
  grid = state.grid;
  if (selected && (selected.x >= SIZE || selected.y >= SIZE)) selected = null;
  document.getElementById('size').value = SIZE;
  renderGrid(); syncEditor();
  scheduleCompute();
}}

function scheduleCompute() {{
  recordHistory();
  cancelCompute();
  // Whatever number is on screen belongs to a grid that no longer exists.
  showThputPending();
  if (autoApplying) return;
  // The throughput costs ~0.1 ms, so refresh it almost immediately.
  // Matplotlib graph rendering is ~1s and shares the server thread, so only
  // do it after the user has actually paused — the readout must never queue
  // behind it.
  _thputTimer = setTimeout(computeThroughput, 25);
  _graphTimer = setTimeout(computeGraph, 1000);
}}

function denoiseStatus(data) {{
  const step = data.chain.step;
  if (data.converged) return 'converged: step ' + step + ' reproduced its input';
  if (data.done) return 'stopped after ' + step + ' steps without converging';
  return 'step ' + step + ' · ' + (data.phase === 'denoise'
    ? 're-noised to ' + data.noise.toFixed(2) + ' for the next step'
    : 'refining toward a fixed point');
}}

// One model call, continuing the current chain, or starting a fresh one from
// noise once the last has finished.
async function denoiseStep() {{
  const chain = currentChain();
  const resp = await fetch('/denoise', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ grid, chain: chain && !chain.done ? chain.chain : null }}),
  }});
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
  grid = data.grid;
  denoised = {{ ...data, gridJson: JSON.stringify(grid) }};
  const info = document.getElementById('model-info');
  if (info) info.textContent = denoiseStatus(data);
  return data;
}}

function denoiseFailed(e) {{
  const info = document.getElementById('model-info');
  if (info) info.textContent = 'denoise failed: ' + e;
}}

async function applyStep() {{
  if (!modelLoaded) return;
  try {{
    await denoiseStep();
    renderGrid(); syncEditor();
    scheduleCompute();
  }} catch (e) {{
    denoiseFailed(e);
  }}
}}

function renderAutoApplyFrame() {{
  if (autoApplyFrame !== null) return;
  autoApplyFrame = requestAnimationFrame(() => {{
    autoApplyFrame = null;
    renderGrid();
    syncEditor();
  }});
}}

async function runAutoApply(generation) {{
  try {{
    while (autoApplying && generation === autoApplyGeneration) {{
      const data = await denoiseStep();
      if (generation !== autoApplyGeneration) {{
        // Released mid-request: stopAutoApply already settled the old grid.
        renderGrid(); syncEditor(); scheduleCompute();
        break;
      }}
      renderAutoApplyFrame();
      showThput(data);
      if (data.done) autoApplying = false;
    }}
  }} catch (e) {{
    autoApplying = false;
    denoiseFailed(e);
  }}
}}

function startAutoApply() {{
  if (autoApplying || !modelLoaded) return;
  cancelCompute();
  autoApplying = true;
  autoApplyGeneration += 1;
  runAutoApply(autoApplyGeneration);
}}

function stopAutoApply() {{
  if (!autoApplying) return;
  autoApplying = false;
  autoApplyGeneration += 1;
  if (autoApplyFrame !== null) {{
    cancelAnimationFrame(autoApplyFrame);
    autoApplyFrame = null;
  }}
  renderGrid();
  syncEditor();
  // Leave the PNG until one second of idle so it can never throttle the
  // held-key loop.
  scheduleCompute();
}}

function beginApplyKey() {{
  if (!modelLoaded) return;
  applyKeyHeld = true;
  cancelCompute();
  // A tap is one step; only a key still down after it enters the loop.
  const firstStep = denoiseStep().then((data) => {{
    if (applyKeyHeld) {{
      renderAutoApplyFrame();
      showThput(data);
    }} else {{
      renderGrid(); syncEditor(); scheduleCompute();
    }}
    return !data.done;
  }}, (e) => {{
    denoiseFailed(e);
    return false;
  }});
  clearTimeout(autoApplyHoldTimer);
  autoApplyHoldTimer = setTimeout(async () => {{
    autoApplyHoldTimer = null;
    const more = await firstStep;
    if (applyKeyHeld && more) startAutoApply();
  }}, 120);
}}

function endApplyKey() {{
  applyKeyHeld = false;
  clearTimeout(autoApplyHoldTimer);
  autoApplyHoldTimer = null;
  if (autoApplying) {{
    stopAutoApply();
    return;
  }}
  if (autoApplyFrame !== null) {{
    cancelAnimationFrame(autoApplyFrame);
    autoApplyFrame = null;
  }}
  renderGrid();
  syncEditor();
  scheduleCompute();
}}

async function computeGraph() {{
  document.getElementById('info').textContent = 'computing…';
  const resp = await fetch('/graph', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ grid }}),
  }});
  const data = await resp.json();
  if (data.error) {{
    document.getElementById('info').textContent = 'error: ' + data.error;
    document.getElementById('out-img').style.display = 'none';
    document.getElementById('edges').textContent = '';
    return;
  }}
  document.getElementById('info').textContent = data.info || '';
  const img = document.getElementById('out-img');
  if (data.png) {{
    img.src = 'data:image/png;base64,' + data.png;
    img.style.display = 'block';
  }} else {{
    img.style.display = 'none';
  }}
  const edges = document.getElementById('edges');
  if (data.edges && data.edges.length) {{
    edges.textContent = data.edges.map(e => e[0] + '  →  ' + e[1]).join('\\n');
  }} else {{
    edges.textContent = '(no edges)';
  }}
}}

// Serialise `g` server-side (the renderer and the throughput engine both live
// there) and put the fixture on the clipboard. The icon doubles as the status
// readout — there is nowhere else on a scan card to put one.
async function copyFromServer(path, body, field, btn) {{
  const icon = btn.innerHTML;
  try {{
    const resp = await fetch(path, {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
    }});
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    await navigator.clipboard.writeText(data[field]);
    console.log(data[field]);
    btn.textContent = '✓';
  }} catch (e) {{
    btn.textContent = '✗';
    console.error(e);
  }}
  setTimeout(() => {{ btn.innerHTML = icon; }}, 1500);
}}
const copyYaml = (g, btn, source) =>
  copyFromServer('/factory_yaml', {{ grid: g, source }}, 'yaml', btn);
const copyRender = (g, btn) => copyFromServer('/render', {{ grid: g }}, 'text', btn);
// What produced the grid on screen, for the fixture's provenance note.
// Diffing against the snapshot taken when the grid was adopted is what keeps
// a hand-edited factory from claiming to be that seed's own output; every
// edit path already funnels through a wholesale swap or mutates `grid`, so
// one comparison covers them all.
function buildSource() {{
  if (!gridSource) return null;
  return JSON.stringify(grid) === gridSnapshot
    ? gridSource
    : gridSource + ', hand-edited';
}}
document.getElementById('copy-yaml').addEventListener('click', (ev) =>
  copyYaml(grid, ev.currentTarget, buildSource()));
document.getElementById('copy-ascii').addEventListener('click', (ev) =>
  copyRender(grid, ev.currentTarget));

document.getElementById('model-apply').addEventListener('click', applyStep);
document.getElementById('model-load').addEventListener('click', loadModel);

// Minimal HTML escape so we can safely inject artifact names + run
// URLs into innerHTML. Strings come from the wandb API, which is
// reasonably trusted, but escaping is cheap insurance against weird
// characters in run names.
function escHtml(s) {{
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

async function refreshModelInfo() {{
  const cur = document.getElementById('model-current');
  try {{
    const resp = await fetch('/model_info');
    const data = await resp.json();
    if (data.loaded) {{
      modelLoaded = true;
      const a = data.arch;
      const shape = 'w' + a.width + 'x' + a.levels + ' attn' + a.attn_layers + ' ' +
        (data.params / 1e6).toFixed(1) + 'M params ' + data.device;
      const src = data.source || {{}};
      if (src.kind === 'wandb') {{
        // Show the run id directly — that's what the user types into
        // the switch form, so keeping the displayed identifier the
        // same as the input format avoids a translation step.
        cur.innerHTML = 'loaded wandb <a href="' + escHtml(src.run_url) +
          '" target="_blank">' + escHtml(src.run_id) + '</a>' +
          '  (' + escHtml(shape) + ')';
      }} else {{
        const path = (src && src.path) || data.path;
        cur.textContent = 'loaded: ' + path + '  (' + shape + ')';
      }}
    }} else {{
      modelLoaded = false;
      cur.textContent = '(none loaded — paste a path or wandb run id below)';
    }}
  }} catch (e) {{
    cur.textContent = 'model_info failed: ' + e;
  }}
}}

async function loadModel() {{
  const value = document.getElementById('model-value').value;
  const status = document.getElementById('model-load-status');
  const btn = document.getElementById('model-load');
  // wandb downloads can take seconds — disable + show a status so the
  // user doesn't double-click and queue a second download.
  btn.disabled = true;
  status.textContent = 'loading…';
  try {{
    const resp = await fetch('/load_model', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ value }}),
    }});
    const data = await resp.json();
    if (data.error) {{
      status.textContent = 'error: ' + data.error;
      return;
    }}
    status.textContent = 'loaded ✓';
    await refreshModelInfo();
    denoised = null;
    renderGrid();
  }} catch (e) {{
    status.textContent = 'load failed: ' + e;
  }} finally {{
    btn.disabled = false;
  }}
}}
// Adopt a whole grid as the Build tab's working state. The stale selection
// has to be dropped along with the old grid, so every wholesale swap goes
// through here.
function adoptGrid(size, cells, source) {{
  SIZE = size;
  grid = cells;
  selected = null;
  gridSource = source || null;
  gridSnapshot = JSON.stringify(grid);
  document.getElementById('size').value = SIZE;
  renderGrid(); syncEditor();
  scheduleCompute();
}}

async function generateLesson(fromHash = false) {{
  const kind = document.getElementById('lesson-kind').value;
  const seed = parseInt(document.getElementById('lesson-seed').value, 10);
  // `entities to clear` is optional — a blank / non-numeric / negative
  // value means "fully generated" (clear nothing).
  const clearRaw = parseInt(document.getElementById('lesson-clear').value, 10);
  const numMissing = Number.isFinite(clearRaw) && clearRaw > 0 ? clearRaw : 0;
  const status = document.getElementById('lesson-status');
  const btn = document.getElementById('lesson-generate');
  if (!Number.isFinite(seed)) {{ status.textContent = 'invalid seed'; return; }}
  btn.disabled = true;
  status.textContent = 'building…';
  try {{
    const resp = await fetch('/load_lesson', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ kind, seed, size: SIZE, num_missing_entities: numMissing }}),
    }});
    const data = await resp.json();
    if (data.error) {{ status.textContent = 'error: ' + data.error; return; }}
    adoptGrid(data.size, data.grid, kind + ' seed ' + data.used_seed);
    const href = buildHref(kind, data.used_seed, data.size, numMissing);
    loadedBuildHash = tabHash.build = href;
    history[fromHash ? 'replaceState' : 'pushState'](null, '', href);
    document.getElementById('lesson-seed').value = data.next_seed;
    // `num_removed` is what blank_entities actually cleared, which can be
    // less than requested when the lesson protects most of its entities.
    const cleared = data.num_removed
      ? ', cleared ' + data.num_removed + '/' + numMissing
      : '';
    status.textContent =
      'built ' + kind + ' (seed=' + data.used_seed +
      ', ' + data.total_entities + ' entities' + cleared + ')';
  }} catch (e) {{
    status.textContent = 'failed: ' + e;
  }} finally {{
    btn.disabled = false;
  }}
}}
document.getElementById('lesson-generate').addEventListener('click', () => generateLesson());

document.getElementById('resize').addEventListener('click', () => {{
  const n = parseInt(document.getElementById('size').value, 10);
  if (!Number.isFinite(n) || n < 2 || n > 20) return;
  adoptGrid(n, newGrid(n), null);
}});
document.getElementById('export').addEventListener('click', async () => {{
  const text = JSON.stringify({{ size: SIZE, grid }}, null, 2);
  try {{ await navigator.clipboard.writeText(text); }} catch (_) {{}}
  console.log(text);
  alert('state copied to clipboard (also logged to console)');
}});

document.addEventListener('keydown', (ev) => {{
  const t = ev.target;
  const tag = t && t.tagName;
  if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
  if ((ev.metaKey || ev.ctrlKey) && !ev.altKey) {{
    const k = ev.key.toLowerCase();
    if (k === 'z' && !ev.shiftKey) stepHistory(undoStack, redoStack);
    else if ((k === 'z' && ev.shiftKey) || k === 'y') stepHistory(redoStack, undoStack);
    else return;
    ev.preventDefault();
    return;
  }}
  if (ev.altKey) return;
  if (/^[0-9]$/.test(ev.key)) {{
    const n = parseInt(ev.key, 10);
    const idx = (n === 0) ? 9 : n - 1;
    setActiveHotbar(idx);
    ev.preventDefault();
    return;
  }}
  if (ev.key === 'r' || ev.key === 'R') {{
    const cw = ev.key === 'r';
    if (cursor) rotateCursor(cw); else rotateSelected(cw);
    ev.preventDefault();
    return;
  }}
  if (ev.key === 'z') {{ pickHovered(); ev.preventDefault(); return; }}
  if (ev.key === '/' && !document.getElementById('tab-scan').hidden) {{
    document.getElementById('scan-q').focus();
    ev.preventDefault();
    return;
  }}
  if (ev.key === 'Delete' || ev.key === 'Backspace') {{
    clearSelected(); ev.preventDefault(); return;
  }}
  if (ev.key === 'g') {{ generateLesson(); ev.preventDefault(); return; }}
  if (ev.key === 'a') {{
    if (!ev.repeat) beginApplyKey();
    ev.preventDefault();
    return;
  }}
  if (ev.key === 'c') {{
    document.getElementById('resize').click(); ev.preventDefault(); return;
  }}
  if (ev.key === 'Escape') {{
    const pop = document.getElementById('help-popover');
    if (pop && !pop.hidden) {{
      pop.hidden = true;
      const tgl = document.getElementById('help-toggle');
      if (tgl) tgl.setAttribute('aria-expanded', 'false');
      return;
    }}
    const filtering = scanFilter.clauses.length || scanFilter.error;
    if (!document.getElementById('tab-scan').hidden && filtering) location.hash = '#scan';
    else if (cursor) setCursor(null);
    else if (activeHotbar !== null) setActiveHotbar(activeHotbar);
    return;
  }}
}});

document.addEventListener('keyup', (ev) => {{
  if (ev.key === 'a') {{
    endApplyKey();
    ev.preventDefault();
  }}
}});

function cancelApplyKeyIfActive() {{
  if (applyKeyHeld || autoApplying || autoApplyHoldTimer !== null) endApplyKey();
}}
// A lost keyup (for example, switching windows while holding a) must not
// leave the request pump running in the background.
window.addEventListener('blur', cancelApplyKeyIfActive);
document.addEventListener('visibilitychange', () => {{
  if (document.hidden) cancelApplyKeyIfActive();
}});

// [?] help popover: click the badge to toggle the shortcuts list,
// click anywhere outside (or Esc, handled in the global keydown) to
// close. Replaces the old native `title` tooltip, which didn't render.
function bindHelp() {{
  const toggle = document.getElementById('help-toggle');
  const pop = document.getElementById('help-popover');
  if (!toggle || !pop) return;
  function setOpen(open) {{
    pop.hidden = !open;
    toggle.setAttribute('aria-expanded', open ? 'true' : 'false');
  }}
  toggle.addEventListener('click', (ev) => {{
    ev.stopPropagation();
    setOpen(pop.hidden);
  }});
  toggle.addEventListener('keydown', (ev) => {{
    if (ev.key === 'Enter' || ev.key === ' ') {{
      ev.preventDefault();
      setOpen(pop.hidden);
    }}
  }});
  // A click anywhere outside the badge/popover dismisses it.
  document.addEventListener('click', (ev) => {{
    if (!pop.hidden && ev.target !== toggle && !pop.contains(ev.target)) {{
      setOpen(false);
    }}
  }});
}}

// ── Scan tab ────────────────────────────────────────────────────────────
// A scan blanks N factories to their markers, denoises each from noise, and
// lays the finished factories out side by side.
let scanResults = [];
let scanAbort = null;
let scanIndexBase = 0;

function switchTab(name) {{
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  document.getElementById('tab-build').hidden = (name !== 'build');
  document.getElementById('tab-scan').hidden = (name !== 'scan');
}}

// The URL hash holds the navigable state, so views can be linked, bookmarked
// and walked back through: #build?lesson=K&seed=N&size=S&clear=C regenerates a
// lesson, and #scan?CLAUSE&CLAUSE filters the gallery.
const tabHash = {{ build: '#build', scan: '#scan' }};
let loadedBuildHash = null;

// Gallery filter: space-separated clauses that must all hold, e.g.
// `lesson=A,B thput>=0.5 converged=false steps<=12`. A comma list after = or !=
// means any of; in the URL the clauses are joined with & instead of spaces.
const FILTER_FIELDS = {{
  lesson: r => r.kind, thput: r => r.thput_normed, raw: r => r.thput_raw,
  steps: r => r.steps, seed: r => r.seed, placed: r => r.num_placed_entities,
  converged: r => r.stopped_by === 'converged',
}};
const CLAUSE = /^([a-z]+)(<=|>=|!=|=|<|>)(.+)$/;
let scanFilter = {{ text: '', clauses: [], error: null }};

function parseFilter(text) {{
  const clauses = [];
  for (const tok of text.split(/\\s+/).filter(Boolean)) {{
    const m = CLAUSE.exec(tok);
    if (!m) {{
      return {{ text, clauses: [], error: `can't read "${{tok}}"; want field, operator, value, like thput>=0.5` }};
    }}
    if (!(m[1] in FILTER_FIELDS)) {{
      return {{
        text, clauses: [],
        error: `no field "${{m[1]}}"; fields are ${{Object.keys(FILTER_FIELDS).join(', ')}}`,
      }};
    }}
    const values = m[3].split(',').map(v =>
      /^true$/i.test(v) ? true : /^false$/i.test(v) ? false
        : v !== '' && !isNaN(v) ? Number(v) : v.toUpperCase());
    clauses.push({{ field: m[1], op: m[2], values }});
  }}
  return {{ text, clauses, error: null }};
}}

function clauseHolds(r, {{ field, op, values }}) {{
  let v = FILTER_FIELDS[field](r);
  if (typeof v === 'string') v = v.toUpperCase();
  if (op === '=') return values.includes(v);
  if (op === '!=') return !values.includes(v);
  const x = values[0];
  return op === '<' ? v < x : op === '<=' ? v <= x : op === '>' ? v > x : v >= x;
}}

function inScanFilter(r) {{
  return scanFilter.clauses.every(c => clauseHolds(r, c));
}}

function scanHash(query) {{
  const clauses = query.split(/\\s+/).filter(Boolean);
  return '#scan' + (clauses.length ? '?' + clauses.join('&') : '');
}}

function scanHref(query) {{
  const href = scanHash(query);
  // Following a link to the view already shown clears the filter instead.
  return href === decodeURIComponent(location.hash) ? '#scan' : href;
}}

function setScanFilter(text) {{
  scanFilter = parseFilter(text);
  const input = document.getElementById('scan-q');
  if (document.activeElement !== input) input.value = text;
  renderScan();
  scanSummary();
}}

function renderScanFilter() {{
  const bar = document.getElementById('scan-filter');
  const status = document.getElementById('scan-filter-status');
  bar.classList.toggle('active', scanFilter.clauses.length > 0);
  if (scanFilter.error) {{
    status.innerHTML = `<span class="err">${{escHtml(scanFilter.error)}}</span>`;
  }} else if (scanFilter.clauses.length) {{
    const shown = scanResults.filter(inScanFilter).length;
    status.innerHTML = `showing ${{shown}} of ${{scanResults.length}}` +
      ` <a href="#scan">✕ show all</a> <span class="help">(or Esc)</span>`;
  }} else {{
    status.innerHTML = '';
  }}
}}

function buildHref(kind, seed, size, clear) {{
  return '#build?' + new URLSearchParams({{ lesson: kind, seed, size, clear }});
}}

function applyHash() {{
  const [tab, query] = location.hash.slice(1).split('?');
  const params = new URLSearchParams(query || '');
  if (tab === 'scan') {{
    tabHash.scan = location.hash;
    switchTab('scan');
    setScanFilter(decodeURIComponent(query || '').split('&').join(' '));
    return;
  }}
  switchTab('build');
  if (tab !== 'build') return;
  tabHash.build = location.hash;
  if (params.get('lesson') && location.hash !== loadedBuildHash) {{
    document.getElementById('lesson-kind').value = params.get('lesson');
    document.getElementById('lesson-seed').value = params.get('seed') || 0;
    document.getElementById('lesson-clear').value = params.get('clear') || 0;
    const size = parseInt(params.get('size'), 10);
    if (size >= 2 && size <= 20) SIZE = size;
    generateLesson(true);
  }}
}}
window.addEventListener('hashchange', applyHash);

// Red at 0, green at 1.
function fracColor(t) {{
  const q = Math.max(0, Math.min(t, 1));
  return `hsl(${{Math.round(q * 120)}}, 70%, 45%)`;
}}

function miniGrid(g) {{
  let html = '<table class="mini">';
  for (const row of g) {{
    html += '<tr>';
    for (const c of row) {{
      const cls = c.footprint === 'UNAVAILABLE' ? ' class="unavailable"' : '';
      html += `<td${{cls}}><div class="cell-inner">${{cellGlyphs(c)}}</div></td>`;
    }}
    html += '</tr>';
  }}
  return html + '</table>';
}}

function scanCard(r, showRef) {{
  const card = document.createElement('div');
  card.className = 'scan-card';
  // Binary because what counts as a good thput differs by lesson. The border
  // is the fastest read in the gallery: a cluster of zeros registers first.
  card.style.borderLeftColor = fracColor(r.thput_normed > 0 ? 1 : 0);
  const stop = r.stopped_by === 'converged'
    ? 'converged at step ' + r.steps
    : 'not converged, ' + r.steps + ' steps';
  const head =
    `<div class="hd"><a href="${{scanHref('lesson=' + r.kind)}}" data-lesson="${{r.kind}}">` +
    `${{escHtml(r.kind)}}</a><button class="copy-yaml"` +
    ` title="Copy this factory as a YAML test fixture">{COPY_ICON}</button>` +
    `<button class="copy-yaml copy-ascii" title="Copy this factory as a two-character text render (render_factory)">txt</button></div>` +
    `<div class="sub">seed <a href="${{buildHref(r.kind, r.seed, r.size, 0)}}"` +
    ` title="Open this seed's reference factory in the Build tab">${{r.seed}}</a>` +
    ` · thput ${{r.thput_normed.toFixed(3)}}` +
    ` (${{r.thput_raw.toFixed(2)}} of ${{r.max_throughput.toFixed(2)}} i/s)</div>` +
    `<div class="sub">${{stop}} · ${{r.num_placed_entities}} placed</div>`;
  const built =
    `<figure><figcaption>built</figcaption>${{miniGrid(r.grid)}}</figure>`;
  const ref = showRef
    ? `<figure><figcaption>reference</figcaption>${{miniGrid(r.solved_grid)}}</figure>`
    : '';
  card.innerHTML = head + `<div class="pair">${{built}}${{ref}}</div>`;
  card.title = 'Open this factory in the Build tab';
  // The card holds what the model built from this lesson's markers, not the
  // generator's own solution, so the seed alone would misdescribe it.
  const source = r.kind + ' seed ' + r.seed + ', model rebuild';
  card.querySelector('.copy-yaml').addEventListener('click', (ev) => {{
    ev.stopPropagation();  // the card's own click adopts the grid instead
    copyYaml(r.grid, ev.currentTarget, source);
  }});
  card.querySelector('.copy-ascii').addEventListener('click', (ev) => {{
    ev.stopPropagation();
    copyRender(r.grid, ev.currentTarget);
  }});
  card.addEventListener('click', (ev) => {{
    if (ev.target.closest('a')) return;
    tabHash.build = '#build';
    history.pushState(null, '', '#build');
    switchTab('build');
    adoptGrid(
      r.size, r.grid.map(row => row.map(c => Object.assign({{}}, c))), source,
    );
    document.getElementById('lesson-kind').value = r.kind;
    document.getElementById('lesson-seed').value = r.seed;
    document.getElementById('lesson-status').textContent = source;
  }});
  return card;
}}

function renderScan() {{
  const host = document.getElementById('scan-results');
  const showRef = document.getElementById('scan-ref').checked;
  const mode = document.getElementById('scan-sort').value;
  const rows = scanResults.filter(inScanFilter);
  if (mode === 'worst') {{
    rows.sort((a, b) => a.thput_normed - b.thput_normed || a.index - b.index);
  }} else if (mode === 'best') {{
    rows.sort((a, b) => b.thput_normed - a.thput_normed || a.index - b.index);
  }} else {{
    rows.sort((a, b) => a.index - b.index);
  }}
  host.replaceChildren(...rows.map(r => scanCard(r, showRef)));
  renderScanStats();
  renderScanFilter();
}}

// "Matched the reference" with float slack.
const REFERENCE_THPUT = 0.999;

function scanSummary(status) {{
  const el = document.getElementById('scan-summary');
  const n = scanResults.length;
  if (!n) {{ el.textContent = status || 'no scan yet'; return; }}
  const t = scanResults.map(r => r.thput_normed);
  const mean = t.reduce((a, b) => a + b, 0) / n;
  const zeros = t.filter(v => v <= 0).length;
  const atRef = t.filter(v => v >= REFERENCE_THPUT).length;
  const converged = scanResults.filter(r => r.stopped_by === 'converged').length;
  const link = (query, text) => `<a href="${{scanHref(query)}}">${{text}}</a>`;
  el.innerHTML =
    `${{n}} done · mean thput ${{mean.toFixed(3)}} · ${{link('thput<=0', zeros + ' at zero')}} · ` +
    `${{link('thput>=' + REFERENCE_THPUT, atRef + ' ≥ reference thput')}} · ` +
    `${{link('converged=true', 'converged ' + converged + '/' + n)}}` +
    (status ? '  ·  ' + escHtml(status) : '');
}}

// Per-lesson breakdown. A mean over the whole scan hides the thing worth
// finding — that one lesson is at zero while the rest are fine — so every
// kind gets its own row, lowest fraction above zero first.
function renderScanStats() {{
  const host = document.getElementById('scan-stats');
  if (!scanResults.length) {{ host.replaceChildren(); return; }}
  const byKind = new Map();
  for (const r of scanResults) {{
    if (!byKind.has(r.kind)) byKind.set(r.kind, []);
    byKind.get(r.kind).push(r);
  }}
  const frac = (hits, n) =>
    `${{hits}}/${{n}} (${{Math.round((hits / n) * 100)}}%)`;
  const rows = [...byKind.entries()].map(([kind, rs]) => {{
    const n = rs.length;
    return {{
      kind,
      n,
      mean: rs.reduce((a, r) => a + r.thput_normed, 0) / n,
      nonzero: rs.filter(r => r.thput_normed > 0).length,
      atRef: rs.filter(r => r.thput_normed >= REFERENCE_THPUT).length,
      converged: rs.filter(r => r.stopped_by === 'converged').length,
    }};
  }});
  rows.sort((a, b) =>
    a.nonzero / a.n - b.nonzero / b.n || a.mean - b.mean ||
    a.kind.localeCompare(b.kind));
  const head =
    '<tr><th>lesson</th><th>runs</th><th>mean thput</th><th>thput = 0</th>' +
    '<th>thput &gt; 0</th><th>≥ reference thput</th><th>converged</th></tr>';
  const body = rows.map(row => {{
    const link = (query, text) =>
      `<a href="${{scanHref(('lesson=' + row.kind + ' ' + query).trim())}}">${{text}}</a>`;
    const active = scanFilter.clauses.some(c =>
      c.field === 'lesson' && c.op === '=' && c.values.includes(row.kind));
    return `<tr${{active ? ' class="active"' : ''}}>` +
      `<td class="kind" data-lesson="${{row.kind}}">${{link('', escHtml(row.kind))}}</td>` +
      `<td>${{row.n}}</td>` +
      `<td>${{row.mean.toFixed(3)}}</td>` +
      `<td>${{link('thput<=0', frac(row.n - row.nonzero, row.n))}}</td>` +
      `<td style="color:${{fracColor(row.nonzero / row.n)}}">` +
      `${{link('thput>0', frac(row.nonzero, row.n))}}</td>` +
      `<td>${{link('thput>=' + REFERENCE_THPUT, frac(row.atRef, row.n))}}</td>` +
      `<td>${{link('converged=true', frac(row.converged, row.n))}}</td></tr>`;
  }}).join('');
  host.innerHTML = `<table>${{head}}${{body}}</table>`;
}}

async function runScan() {{
  const runBtn = document.getElementById('scan-run');
  const stopBtn = document.getElementById('scan-stop');
  if (!modelLoaded) {{
    document.getElementById('scan-summary').textContent =
      'no model loaded — load one from the Build tab first';
    return;
  }}
  const kinds = checkedScanKinds();
  if (!kinds.length) {{
    document.getElementById('scan-summary').textContent = 'tick at least one lesson';
    return;
  }}
  const body = {{
    kinds,
    count: parseInt(document.getElementById('scan-count').value, 10) || 1,
    seed: parseInt(document.getElementById('scan-seed').value, 10) || 0,
    size: SIZE,
    best_of: parseInt(document.getElementById('scan-best-of').value, 10) || 1,
  }};
  // Each run appends. Indices are offset by the running total so they stay
  // unique across runs and keep the server's within-run ordering, which is
  // what the "run order" sort reads.
  const indexBase = scanIndexBase;
  scanIndexBase += body.count * kinds.length;
  const before = scanResults.length;
  scanAbort = new AbortController();
  runBtn.disabled = true;
  stopBtn.disabled = false;
  const started = performance.now();
  const showRef = document.getElementById('scan-ref').checked;
  let total = body.count * kinds.length;
  scanSummary('generating ' + total + ' factories at size ' + SIZE + '…');
  try {{
    const resp = await fetch('/batch_rollout', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify(body),
      signal: scanAbort.signal,
    }});
    // A status-coded failure has no NDJSON body to read, so it would
    // otherwise show as a scan that finished with nothing in it.
    if (!resp.ok) throw new Error('server returned HTTP ' + resp.status);
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    for (;;) {{
      const {{ done, value }} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {{ stream: true }});
      let nl;
      while ((nl = buf.indexOf('\\n')) >= 0) {{
        const line = buf.slice(0, nl);
        buf = buf.slice(nl + 1);
        if (!line.trim()) continue;
        const ev = JSON.parse(line);
        if (ev.type === 'start') {{
          total = ev.n;
          scanSummary('denoising ' + total + ' factories…');
        }} else if (ev.type === 'result') {{
          ev.index += indexBase;
          scanResults.push(ev);
          // Append rather than re-render: a full re-sort of N cards on
          // every arrival is O(N^2) table builds. renderScan() applies
          // the chosen sort once the stream ends.
          if (inScanFilter(ev)) {{
            document.getElementById('scan-results').appendChild(scanCard(ev, showRef));
          }}
          renderScanStats();
          renderScanFilter();
        }} else if (ev.type === 'progress') {{
          const secs = (performance.now() - started) / 1000;
          scanSummary(
            (scanResults.length - before) + '/' + total + ' finished · ' +
            secs.toFixed(1) + 's'
          );
        }} else if (ev.type === 'error') {{
          scanSummary('error: ' + ev.error);
        }}
      }}
    }}
    renderScan();
    // Advance the seed so the next click draws fresh factories rather than
    // rebuilding the same ones.
    document.getElementById('scan-seed').value = body.seed + body.count;
    const secs = (performance.now() - started) / 1000;
    scanSummary(
      'added ' + (scanResults.length - before) + ' in ' + secs.toFixed(1) + 's'
    );
  }} catch (e) {{
    if (e.name === 'AbortError') {{
      renderScan();
      scanSummary('stopped');
    }} else {{
      scanSummary('scan failed: ' + e);
    }}
  }} finally {{
    runBtn.disabled = false;
    stopBtn.disabled = true;
    scanAbort = null;
  }}
}}

function clearScan() {{
  scanResults = [];
  scanIndexBase = 0;
  document.getElementById('scan-results').replaceChildren();
  renderScanStats();
  renderScanFilter();
  scanSummary('');
}}

function checkedScanKinds() {{
  return [...document.querySelectorAll('#scan-kinds input:checked')]
    .map(el => el.value);
}}

function syncScanRunLabel() {{
  const c = parseInt(document.getElementById('scan-count').value, 10) || 1;
  document.getElementById('scan-run').textContent =
    'Run +' + c + ' × ' + checkedScanKinds().length;
}}

function setScanKinds(checked) {{
  document.querySelectorAll('#scan-kinds input').forEach(el => {{
    el.checked = checked;
  }});
  syncScanRunLabel();
}}

// Per-browser preferences. Storage can be unavailable (private windows, blocked
// site data), in which case the page starts from its defaults.
const PREFS_KEY = 'factory-builder-prefs';
const PREF_INPUTS = [
  'scan-count', 'scan-seed', 'scan-best-of', 'scan-ref', 'scan-sort',
  'lesson-kind', 'lesson-seed', 'lesson-clear',
];
function savePrefs() {{
  const prefs = {{ scanKinds: checkedScanKinds() }};
  for (const id of PREF_INPUTS) {{
    const el = document.getElementById(id);
    prefs[id] = el.type === 'checkbox' ? el.checked : el.value;
  }}
  try {{ localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); }} catch (_) {{}}
}}
function loadPrefs() {{
  let prefs = null;
  try {{ prefs = JSON.parse(localStorage.getItem(PREFS_KEY)); }} catch (_) {{}}
  if (!prefs) return;
  for (const id of PREF_INPUTS) {{
    if (!(id in prefs)) continue;
    const el = document.getElementById(id);
    if (el.type === 'checkbox') el.checked = prefs[id];
    else if (el.tagName !== 'SELECT' || [...el.options].some(o => o.value === prefs[id])) {{
      el.value = prefs[id];
    }}
  }}
  if (prefs.scanKinds) {{
    document.querySelectorAll('#scan-kinds input').forEach(el => {{
      el.checked = prefs.scanKinds.includes(el.value);
    }});
  }}
}}

// Gwern-style previews: hovering any lesson name shows that lesson's reference
// factory. Every preview is fetched once up front, since the single-threaded
// server can be busy with a scan for minutes at a time.
const lessonPreviews = new Map();  // size -> {{kind: preview}}, once loaded
let popupOwner = null;

async function loadLessonPreviews(size) {{
  if (lessonPreviews.has(size)) return;
  lessonPreviews.set(size, null);
  try {{
    const resp = await fetch('/lesson_previews', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ size }}),
    }});
    lessonPreviews.set(size, await resp.json());
  }} catch (_) {{
    lessonPreviews.delete(size);
  }}
  if (popupOwner) showLessonPopup(popupOwner);
}}

function showLessonPopup(el) {{
  const kind = el.dataset.lesson;
  const previews = lessonPreviews.get(SIZE);
  if (!previews) loadLessonPreviews(SIZE);
  const data = previews && previews[kind];
  const pop = document.getElementById('popup');
  pop.innerHTML = !data
    ? escHtml(kind) + ': loading preview…'
    : data.error
      ? escHtml(data.error)
      : `<div class="hd">${{escHtml(kind)}}</div>` +
        `<div class="sub">seed ${{data.used_seed}} reference · ` +
        `${{data.total_entities}} entities</div>${{miniGrid(data.grid)}}`;
  const rect = el.getBoundingClientRect();
  pop.style.left = (window.scrollX + rect.left) + 'px';
  pop.style.top = (window.scrollY + rect.bottom + 4) + 'px';
  pop.hidden = false;
}}

function hideLessonPopup() {{
  popupOwner = null;
  document.getElementById('popup').hidden = true;
}}

document.addEventListener('mouseover', (ev) => {{
  const el = ev.target.closest('[data-lesson]');
  if (el === popupOwner) return;
  if (!el) {{ hideLessonPopup(); return; }}
  popupOwner = el;
  showLessonPopup(el);
}});
document.addEventListener('mouseout', (ev) => {{
  if (!ev.relatedTarget) hideLessonPopup();
}});

function bindScan() {{
  document.querySelectorAll('#tabs button').forEach(b =>
    b.addEventListener('click', () => {{ location.hash = tabHash[b.dataset.tab]; }}));
  document.getElementById('scan-run').addEventListener('click', runScan);
  document.getElementById('scan-stop').addEventListener('click', () => {{
    if (scanAbort) scanAbort.abort();
  }});
  document.getElementById('scan-count').addEventListener('input', syncScanRunLabel);
  document.getElementById('scan-kinds').addEventListener('change', syncScanRunLabel);
  document.getElementById('scan-kinds-all').addEventListener('click', () => setScanKinds(true));
  document.getElementById('scan-kinds-none').addEventListener('click', () => setScanKinds(false));
  document.getElementById('scan-clear-results').addEventListener('click', clearScan);
  document.getElementById('scan-sort').addEventListener('change', renderScan);
  document.getElementById('scan-ref').addEventListener('change', renderScan);
  const q = document.getElementById('scan-q');
  q.addEventListener('input', () => {{
    const parsed = parseFilter(q.value);
    if (parsed.error) {{
      scanFilter = {{ ...scanFilter, error: parsed.error }};
      renderScanFilter();
      return;
    }}
    const hash = scanHash(q.value);
    history.replaceState(null, '', hash);
    tabHash.scan = hash;
    setScanFilter(q.value);
  }});
  q.addEventListener('keydown', (ev) => {{
    if (ev.key === 'Escape') {{ location.hash = '#scan'; q.value = ''; q.blur(); }}
    if (ev.key === 'Enter') q.blur();
  }});
  document.addEventListener('change', savePrefs);
  window.addEventListener('pagehide', savePrefs);
  syncScanRunLabel();
}}

grid = newGrid(SIZE);
recordHistory();
renderGrid();
bindHotbar();
bindEditor();
bindHelp();
loadPrefs();
bindScan();
refreshModelInfo();
computeThroughput();
applyHash();
loadLessonPreviews(SIZE);
</script>
<div class="popup" id="popup" hidden></div>
</body></html>"""


class _BuilderServer(HTTPServer):
    """HTTPServer that carries the builder's CLI defaults for the handler."""

    default_size: int
    wandb_project: str
    wandb_entity: str | None


class Handler(BaseHTTPRequestHandler):
    server: _BuilderServer  # _BuilderServer stashes the CLI defaults on itself
    raw_requestline: bytes  # set by handle_one_request; absent from the stubs
    server_version = "FactoryBuilder/0.1"

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[%s] %s\n" % (self.address_string(), format % args))

    def parse_request(self) -> bool:
        """Reject a TLS handshake without logging it as a mangled request.

        Bytes starting 0x16 0x03 are a TLS ClientHello: something reached
        this plain-HTTP server over https://, which browsers do on their own
        (Firefox's HTTPS-Only mode, HSTS, an extension). The default handler
        renders the handshake as a garbled HTTP/0.9 request line, which reads
        like a crash rather than a wrong-scheme connection.
        """
        if self.raw_requestline.startswith(b"\x16\x03"):
            self.log_message(
                "ignored an https:// connection — this server is plain HTTP"
            )
            self.close_connection = True
            return False
        return super().parse_request()

    def _stream_ndjson(self, events: Iterator[dict]) -> None:
        """Write one JSON object per line as the generator produces them.

        The response carries no Content-Length, so end-of-body is
        end-of-connection — hence the explicit `Connection: close`, which
        keeps the framing correct even if someone raises the handler's
        protocol_version to HTTP/1.1. It is also how the browser's Stop
        button works: aborting the fetch drops the socket, the next write
        raises, and abandoning the generator cancels the scan.
        """
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for event in events:
                self.wfile.write((json.dumps(event) + "\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write_body(body)

    def _write_body(self, body: bytes) -> None:
        """Write a response body, tolerating a client that has gone away.

        Reloading mid-load or hitting Stop drops the socket while the ~1MB
        page is still going out; that is routine, not something to dump a
        traceback over."""
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            body = render_index(self.server.default_size).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write_body(body)
            return
        if self.path == "/model_info":
            self._send_json(_model_info())
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path not in (
            "/graph",
            "/denoise",
            "/hand_edit",
            "/load_model",
            "/load_lesson",
            "/lesson_previews",
            "/batch_rollout",
            "/factory_yaml",
            "/render",
            "/throughput",
        ):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
            if self.path == "/batch_rollout":
                self._stream_ndjson(_batch_rollout_request(payload))
                return
            if self.path == "/graph":
                result = render_graph_png(payload["grid"])
            elif self.path == "/throughput":
                result = _throughput(build_world(payload["grid"]))
            elif self.path == "/factory_yaml":
                result = {
                    "yaml": factory_yaml(payload["grid"], payload.get("source"))
                }
            elif self.path == "/render":
                world_CWH = build_world(payload["grid"]).permute(2, 0, 1)
                result = {"text": render_factory(world_CWH)}
            elif self.path == "/denoise":
                result = _denoise_step(payload["grid"], payload.get("chain"))
            elif self.path == "/hand_edit":
                result = _hand_edit(payload["grid"], payload["ops"])
            elif self.path == "/lesson_previews":
                result = _lesson_previews(int(payload["size"]))
            elif self.path == "/load_lesson":
                result = _load_lesson(
                    kind_name=payload["kind"],
                    seed=int(payload["seed"]),
                    size=int(payload["size"]),
                    num_missing_entities=int(payload.get("num_missing_entities", 0)),
                )
            else:
                result = _swap_model(
                    value=payload.get("value", ""),
                    project=self.server.wandb_project,
                    entity=self.server.wandb_entity,
                )
        except Exception as e:
            traceback.print_exc()
            result = {"error": f"{type(e).__name__}: {e}"}
        self._send_json(result)


def main(args: Args) -> None:
    global _CHECKPOINT_SOURCE
    if args.checkpoint and args.wandb_run:
        raise SystemExit("pass either --checkpoint or --wandb-run, not both")
    if args.wandb_run:
        ckpt_path, source = _resolve_wandb_checkpoint(
            args.wandb_run, args.wandb_project, args.wandb_entity,
        )
        _load_checkpoint(ckpt_path, source.get("config"))
        _CHECKPOINT_SOURCE = source
    elif args.checkpoint:
        _load_checkpoint(args.checkpoint)
        _CHECKPOINT_SOURCE = {"kind": "local", "path": args.checkpoint}
    else:
        print("(no checkpoint — denoising disabled)")

    httpd = _BuilderServer(("127.0.0.1", args.port), Handler)
    httpd.default_size = args.size
    # Stashed on the server so the /load_model endpoint can use the same
    # defaults as the CLI when resolving wandb run ids.
    httpd.wandb_project = args.wandb_project
    httpd.wandb_entity = args.wandb_entity
    print(f"Serving factory builder on http://127.0.0.1:{args.port}")
    print("Press Ctrl-C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")


if __name__ == "__main__":
    main(tyro.cli(Args))
