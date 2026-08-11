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
import gymnasium as gym  # noqa: E402
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
from ppo import (  # noqa: E402
    AgentCNN,
    FactorioEnv,
    _resolve_wandb_checkpoint,
    apply_placement_action,
    make_env,
)


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
def _stroke_icon(path: str, width: float = 2.2) -> str:
    """A one-path inline SVG that inherits its colour from the text around it."""
    return (
        f'<svg viewBox="0 0 16 16" width="12" height="12" aria-hidden="true" '
        f'fill="none" stroke="currentColor" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round"><path d="{path}"/></svg>'
    )


# The throughput verdict. Inline SVG for the same reason as the copy icon
# below: a glyph the viewer's font doesn't cover would render as a blank box,
# and this one carries the whole answer.
OK_ICON = _stroke_icon("M3 8.5l3.5 3.5L13 4.5")
BAD_ICON = _stroke_icon("M4 4l8 8M12 4l-8 8")
FLOW_ICON = _stroke_icon("M2 8h9M8 4.5L11.5 8 8 11.5", width=1.7)

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
    "Select / edit: click an empty tile (no ghost)",
    "Apply a ghost prediction: click the ghosted tile",
    "Rotate selected: r (cw), R (ccw)",
    "Clear selected: Delete / Backspace / right-click",
    "Deselect hotbar: Esc",
    "Generate lesson: g (set 'entities to clear' to blank N first)",
    "Apply prediction: tap a once / hold a for fast autoregressive placement",
    "Resize / clear grid: c",
    "Scan seeds tab: rebuild N blanked seeds at once, click a result to open it",
]


@dataclass
class Args:
    port: int = 8765
    """port for the local HTTP server"""
    size: int = 11
    """default grid size"""
    checkpoint: Optional[str] = None
    """path to a trained SFT/PPO checkpoint (.pt). If set, the UI shows
    the model's predicted next placement and exposes an Apply button."""
    wandb_run: Optional[str] = "h76h80yb"
    """W&B run id (or full path 'entity/project/run_id'). The run's most
    recent model-type artifact is downloaded to /tmp/factorion-checkpoints
    and loaded. Mutually exclusive with --checkpoint. Defaults to run
    h76h80yb."""
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
# AgentCNN init reads sizes from a gym env, so we keep one cached per grid
# size — the user can resize the UI grid live and we rebuild lazily on
# first request for that size. The state dict is loaded with strict=False
# because the critic head's flat dim depends on W*H and we only use the
# action heads for inference.

_AGENT_DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
_AGENT_CACHE: dict[int, AgentCNN] = {}
_CHECKPOINT_STATE: Optional[dict] = None
_CHECKPOINT_PATH: Optional[str] = None
# How the current checkpoint was loaded. Surfaced to the UI so it can
# render meaningful info (e.g. wandb run id + link) instead of the
# anonymous /tmp download path. Either {"kind": "local", "path": "..."}
# or {"kind": "wandb", "run_id", "run_url", "run_name", "artifact"}.
_CHECKPOINT_SOURCE: Optional[dict] = None

# Reverse lookup: head index -> readable name. The entity head excludes the
# last two catalog entries (source/sink) but its index space starts at 0 and
# aligns with the first N-2 Item values, so entities[idx].name works as-is.
_ENT_NAMES = {ent.value: ent.name for ent in entities.values()}
_ITEM_NAMES = {it.value: it.name for it in items.values()}
_DIR_NAMES = {d.value: d.name for d in Direction}
_MISC_NAMES = {m.value: m.name for m in Misc}


def _encoder_arch(state) -> tuple[list[int], int]:
    """Infer the encoder architecture (per-layer channel widths, kernel size)
    from a checkpoint's conv weights. Filters by 4-D weight shape and sorts by
    layer index rather than hardcoding `encoder.0/2/4`, so any depth/kernel and
    interleaved non-conv layers (e.g. Dropout2d) reconstruct correctly — no
    sidecar, no assumption of exactly three layers."""
    conv_keys = sorted(
        (
            k
            for k, v in state.items()
            if k.startswith("encoder.") and k.endswith(".weight") and v.dim() == 4
        ),
        key=lambda k: int(k.split(".")[1]),
    )
    layers = [int(state[k].shape[0]) for k in conv_keys]
    kernel_size = int(state[conv_keys[0]].shape[-1])
    return layers, kernel_size


def _load_checkpoint(path: str) -> None:
    """Load the checkpoint .pt and stash it. Clears the per-size agent
    cache so subsequent /predict calls rebuild against the new
    weights. Does NOT touch _CHECKPOINT_SOURCE — the caller knows the
    provenance (local vs wandb) and sets it after this returns."""
    global _CHECKPOINT_STATE, _CHECKPOINT_PATH
    state = torch.load(path, map_location="cpu", weights_only=True)
    # ppo.py saves its agent *after* torch.compile, which prepends
    # "_orig_mod." to every parameter name; SFT checkpoints are saved
    # uncompiled (clean names). Strip the prefix so both load identically
    # — otherwise _encoder_arch finds zero conv keys and crashes, and the
    # critic/eot-head filtering below silently misses every key.
    state = {k.removeprefix("_orig_mod."): v for k, v in state.items()}
    _CHECKPOINT_STATE = state
    _CHECKPOINT_PATH = path
    _AGENT_CACHE.clear()
    layers, kernel_size = _encoder_arch(state)
    print(
        f"Loaded checkpoint {path} "
        f"(layers={layers}, kernel_size={kernel_size}, device={_AGENT_DEVICE})"
    )


def _model_info() -> dict:
    """Snapshot of the currently-loaded model for the UI's status line.
    Returns `loaded: False` when nothing is loaded yet, so the UI can
    render "(none loaded)" without special-casing on the JS side."""
    if _CHECKPOINT_STATE is None:
        return {"loaded": False}
    s = _CHECKPOINT_STATE
    layers, kernel_size = _encoder_arch(s)
    return {
        "loaded": True,
        "path": _CHECKPOINT_PATH,
        "source": _CHECKPOINT_SOURCE,
        "layers": layers,
        "kernel_size": kernel_size,
        "device": str(_AGENT_DEVICE),
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
        source = {"kind": "local", "path": value}
    else:
        path, source = _resolve_wandb_checkpoint(value, project, entity)
    _load_checkpoint(path)
    _CHECKPOINT_SOURCE = source
    return _model_info()


def _get_agent(size: int) -> AgentCNN:
    if _CHECKPOINT_STATE is None:
        raise RuntimeError("no checkpoint loaded — pass --checkpoint at launch")
    if size in _AGENT_CACHE:
        return _AGENT_CACHE[size]

    layers, kernel_size = _encoder_arch(_CHECKPOINT_STATE)

    env_id = "factorion/FactorioEnv-v0-fb"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point="ppo:FactorioEnv")
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, size, "fb")])
    try:
        agent = AgentCNN(envs, layers=layers, kernel_size=kernel_size)
    finally:
        envs.close()
    # Several tensors are grid-size-dependent, so their saved shapes are wrong
    # whenever the UI grid size != the training size. strict=False would *not*
    # save us — it ignores missing/unexpected keys but still raises on shape
    # mismatches, so we filter explicitly.
    #
    # critic_head: always dropped (the critic isn't called during inference,
    # so loading it is wasted work even on a size match).
    #
    # coord_grid: a deterministic (2, W, H) buffer the ctor already rebuilt for
    # this grid size, so it is never loaded from the checkpoint.
    #
    # eot_head (Linear(layers[-1]*W*H, 1)) and attn.pos_embed
    # ((1, W*H, attn_dim)): dropped on size mismatch (random init is fine —
    # the UI doesn't act on the eot signal), kept on a size match so the UI's
    # eot panel shows the real trained prediction. Pre-#103 checkpoints have no
    # eot_head keys at all → load_state_dict (strict=False) leaves the
    # freshly-initialised head in place.
    expected_flat = layers[-1] * size * size
    saved_eot_w = _CHECKPOINT_STATE.get("eot_head.1.weight")
    keep_eot = saved_eot_w is not None and saved_eot_w.shape[1] == expected_flat
    saved_pos = _CHECKPOINT_STATE.get("attn.pos_embed")
    keep_pos = saved_pos is not None and saved_pos.shape[1] == size * size
    drop_prefixes: tuple[str, ...] = ("critic_head.", "coord_grid")
    if not keep_eot:
        drop_prefixes = drop_prefixes + ("eot_head.",)
    if not keep_pos:
        drop_prefixes = drop_prefixes + ("attn.pos_embed",)
    filtered = {
        k: v for k, v in _CHECKPOINT_STATE.items()
        if not k.startswith(drop_prefixes)
    }
    missing, unexpected = agent.load_state_dict(filtered, strict=False)
    ignorable = {
        "critic_head.1.weight", "critic_head.1.bias",
        "eot_head.1.weight", "eot_head.1.bias",
        "coord_grid", "attn.pos_embed",
    }
    real_missing = [k for k in missing if k not in ignorable]
    real_unexpected = [k for k in unexpected if k not in ignorable]
    if real_missing or real_unexpected:
        print(
            f"[warn] state_dict mismatch at size={size}: "
            f"missing={real_missing} unexpected={real_unexpected}"
        )
    agent.to(_AGENT_DEVICE).eval()
    _AGENT_CACHE[size] = agent
    return agent


def _top_p_named(probs: torch.Tensor, names: dict, top_p: float = 0.95) -> tuple[list[dict], float]:
    """Sort `probs` (1-D) by descending probability, take entries until
    cumulative mass >= top_p, return them as [{name, p}, ...] plus the
    remaining "rest" mass. Useful for showing the model's distribution
    over discrete choices (entities, directions, items, misc)."""
    probs_1d = probs.flatten()
    sorted_p, sorted_i = torch.sort(probs_1d, descending=True)
    top: list[dict] = []
    cum = 0.0
    for p, i in zip(sorted_p.tolist(), sorted_i.tolist()):
        top.append({"name": names.get(i, str(i)), "p": float(p)})
        cum += float(p)
        if cum >= top_p:
            break
    return top, max(0.0, 1.0 - cum)


def _tile_top_p(probs: torch.Tensor, H: int, top_p: float = 0.95) -> tuple[list[dict], float]:
    """Same as _top_p_named but emits (x, y, p) entries for the tile
    head, since each entry is a 2-D coordinate rather than a named
    category."""
    sorted_p, sorted_i = torch.sort(probs.flatten(), descending=True)
    top: list[dict] = []
    cum = 0.0
    for p, i in zip(sorted_p.tolist(), sorted_i.tolist()):
        top.append({"x": int(i) // H, "y": int(i) % H, "p": float(p)})
        cum += float(p)
        if cum >= top_p:
            break
    return top, max(0.0, 1.0 - cum)


EOT_STOP_THRESHOLD = 0.5
"""EOT-head probability above which the UI treats the model as finished: the
hold-to-apply loop stops and no further placement is applied. Matches
`AgentCNN.eot_should_stop`'s default and SFT's `rollout_eot_threshold`, so
holding `a` reproduces what a greedy rollout would build."""


CANDIDATE_TILE_THRESHOLD = 0.01
"""Minimum p(tile) for a tile to appear as a ghost candidate in the UI.
Tiles with `p(tile) < CANDIDATE_TILE_THRESHOLD` are dropped — they're
too unlikely to be worth visualising, and the long tail is dominated by
the rest mass anyway."""


def _predict(grid: list[list[dict]]) -> dict:
    """Run the model on `grid` and return:
      * the argmax placement (drives the Apply button + tile border),
      * per-head top-p=0.95 distributions conditioned on the argmax
        tile (drives the side panel),
      * a candidates list — every tile with p(tile) > threshold, paired
        with its conditional argmax (entity / dir / item / misc) so the
        UI can render ghost overlays across the whole grid.

    The four per-tile heads (ent/dir/item/misc) are linear projections
    of the encoder feature at the chosen tile, so computing them for
    every tile is one batched matmul each — cheap."""
    world_WHC = build_world(grid)
    size = world_WHC.shape[0]
    agent = _get_agent(size)

    obs_CWH = world_WHC.permute(2, 0, 1).float().unsqueeze(0).to(_AGENT_DEVICE)
    W = obs_CWH.shape[2]
    H = obs_CWH.shape[3]

    with torch.no_grad():
        # Greedy prediction via the shared sampler; the argmax tile is the
        # "Apply" target, and logp_heads drives the side-panel top-p lists.
        out = agent.sample_action(obs_CWH, temperature=0.0, compute_value=False)
        heads = out["logp_heads"]
        eot_prob = float(out["eot_prob"][0].item())

        tile_probs = heads["tile"].exp()[0]
        tile_top, tile_rest = _tile_top_p(tile_probs, H)
        x = int(out["action"]["xy"][0, 0].item())
        y = int(out["action"]["xy"][0, 1].item())

        ent_top, ent_rest = _top_p_named(heads["entity"].exp()[0], _ENT_NAMES)
        dir_top, dir_rest = _top_p_named(heads["direction"].exp()[0], _DIR_NAMES)
        item_top, item_rest = _top_p_named(heads["item"].exp()[0], _ITEM_NAMES)
        misc_top, misc_rest = _top_p_named(heads["misc"].exp()[0], _MISC_NAMES)

        # Ghost overlays need the greedy per-head pick at EVERY tile, not just
        # the sampled one — a whole-grid matmul on the shared heads.
        encoded_BCWH, g_1G = agent.encode(obs_CWH)
        feats_all = encoded_BCWH[0].permute(1, 2, 0).reshape(W * H, -1)
        if g_1G is not None:
            feats_all = torch.cat([feats_all, g_1G.expand(W * H, -1)], dim=1)
        ent_pick = agent.ent_head(feats_all).argmax(dim=-1)
        dir_pick = agent.dir_head(feats_all).argmax(dim=-1)
        item_pick = agent.item_head(feats_all).argmax(dim=-1)
        misc_pick = agent.misc_head(feats_all).argmax(dim=-1)

        candidates: list[dict] = []
        mask = (tile_probs > CANDIDATE_TILE_THRESHOLD).nonzero(as_tuple=False).squeeze(-1).tolist()
        for t in mask:
            candidates.append({
                "x": t // H,
                "y": t % H,
                "p_tile": float(tile_probs[t].item()),
                "entity": _ENT_NAMES.get(int(ent_pick[t].item()), str(int(ent_pick[t].item()))),
                "direction": _DIR_NAMES.get(int(dir_pick[t].item()), str(int(dir_pick[t].item()))),
                "item": _ITEM_NAMES.get(int(item_pick[t].item()), str(int(item_pick[t].item()))),
                "misc": _MISC_NAMES.get(int(misc_pick[t].item()), str(int(misc_pick[t].item()))),
            })

    return {
        "x": x,
        "y": y,
        "entity": ent_top[0]["name"],
        "direction": dir_top[0]["name"],
        "item": item_top[0]["name"],
        "misc": misc_top[0]["name"],
        "tile_top": tile_top,
        "tile_rest": tile_rest,
        "entity_top": ent_top,
        "entity_rest": ent_rest,
        "direction_top": dir_top,
        "direction_rest": dir_rest,
        "item_top": item_top,
        "item_rest": item_rest,
        "misc_top": misc_top,
        "misc_rest": misc_rest,
        "candidates": candidates,
        "eot_prob": eot_prob,
        **_throughput(world_WHC),
    }


def _predict_action(grid: list[list[dict]]) -> dict:
    """Return only the greedy next placement, plus the EOT probability.

    This is the latency-sensitive path used while the user holds ``a``. The
    detailed predictor above additionally builds probability tables and ghost
    candidates for inspection, none of which an autoregressive placement needs
    — but both go through the shared sampler, so what the held key applies is
    exactly what the panel showed. Reading the heads directly instead skips the
    entity-conditional masks `sample_action` applies to direction / item / misc
    (which is how a belt ends up tagged with a recipe, or an assembler with
    none), and skips `eot_prob`, without which the loop cannot stop where a
    rollout would.
    """
    world_WHC = build_world(grid)
    agent = _get_agent(world_WHC.shape[0])
    obs_CWH = world_WHC.permute(2, 0, 1).float().unsqueeze(0).to(_AGENT_DEVICE)

    with torch.inference_mode():
        out = agent.sample_action(obs_CWH, temperature=0.0, compute_value=False)
        act = out["action"]
        x = int(act["xy"][0, 0].item())
        y = int(act["xy"][0, 1].item())
        entity = int(act["entity"][0].item())
        direction = int(act["direction"][0].item())
        item = int(act["item"][0].item())
        misc = int(act["misc"][0].item())
        eot_prob = float(out["eot_prob"][0].item())

    return {
        "x": x,
        "y": y,
        "entity": _ENT_NAMES.get(entity, str(entity)),
        "direction": _DIR_NAMES.get(direction, str(direction)),
        "item": _ITEM_NAMES.get(item, str(item)),
        "misc": _MISC_NAMES.get(misc, str(misc)),
        "eot_prob": eot_prob,
        # Rides along so the readout tracks a held-`a` build as it happens,
        # one placement behind, instead of going blank until the key is let go.
        **_throughput(world_WHC),
    }


# ── Batched seed scan ────────────────────────────────────────────────────────

ALL_KINDS_SENTINEL = "__ALL__"
ROLLOUT_BATCH_SIZE = 64
"""Rollouts advanced in lockstep per batched forward. Every slot stays in the
batch until the last of its group finishes, so a wider batch mostly buys stale
compute at the tail; a bigger scan runs as back-to-back groups instead. There
is deliberately no ceiling on the scan itself: groups are built and freed one
at a time so memory is flat in the count, results stream as they finish, and
Stop cancels — nothing a cap would protect against."""


def _reset_rollout_env(
    env: FactorioEnv,
    kind: LessonKind,
    seed: int,
    num_missing_entities: Optional[int],
) -> tuple[np.ndarray, int]:
    """Reset `env` onto (kind, seed) and return (obs, the seed that built).

    The seed is settled against `build_factory` first — `FactorioEnv.reset`
    turns a failed build into a bare RuntimeError it also raises for
    unrelated reasons, so probing here (as `ppo._build_eval_set` does)
    keeps a scan from swallowing a real error as an unlucky seed. Omitting
    ``num_missing_entities`` leaves the env's default of "blank
    everything", which is the point of the scan: rebuild from the lesson's
    protected tiles alone.
    """
    _factory, used_seed = _build_with_retry(kind, env.size, seed)
    options: dict = {"kind": kind}
    if num_missing_entities is not None:
        options["num_missing_entities"] = num_missing_entities
    obs, _info = env.reset(seed=used_seed, options=options)
    return obs, used_seed


def _rollout_result(
    index: int, env: FactorioEnv, seed: int, info: dict, terminated: bool
) -> dict:
    return {
        "type": "result",
        "index": index,
        "kind": env._kind.name,
        "seed": seed,
        "size": env.size,
        "steps": int(env.steps),
        "stopped_by": "eot" if terminated else "max_steps",
        "thput_normed": float(info.get("thput_normed", 0.0)),
        "thput_raw": float(info.get("thput_raw", 0.0)),
        "max_throughput": float(env._max_throughput),
        "num_placed_entities": int(info.get("num_placed_entities", 0)),
        "frac_reachable": float(info.get("frac_reachable", 0.0)),
        "invalid_actions": int(env.invalid_actions),
        "grid": world_CWH_to_grid(env._world_CWH),
        "solved_grid": world_CWH_to_grid(env._solved_world_CWH),
    }


def _batch_rollout(
    kinds: list[LessonKind],
    seeds: list[int],
    size: int,
    num_missing_entities: Optional[int],
    legal_mask: bool,
) -> Iterator[dict]:
    """Greedily rebuild one blanked factory per (kind, seed), yielding an
    event per finished rollout.

    Rollouts advance in lockstep through one batched ``sample_action``, in
    groups of ``ROLLOUT_BATCH_SIZE``; a scan larger than one group runs the
    groups back to back, so the wall clock stays linear in the count instead
    of the count being truncated to fit one batch. Slots that have already
    stopped stay in their group's batch and their outputs are discarded:
    compacting would change the batch shape every step, and mps re-plans
    kernels per shape, which measures far slower than the wasted slots
    (N=64: 5.6s fixed vs 29.8s compacted). ``sft.run_rollout_eval`` makes the
    same choice.

    Greedy argmax + the legal-tile mask mirror ``sft.run_rollout_eval``, so a
    scan's throughput numbers are the same quantity as ``eval/thput`` — except
    that here the EOT head really does end the episode, since the whole point
    is to see the factory the model considers finished.
    """
    agent = _get_agent(size)
    yield {"type": "start", "n": len(seeds)}

    step = 0
    for base in range(0, len(seeds), ROLLOUT_BATCH_SIZE):
        group = slice(base, base + ROLLOUT_BATCH_SIZE)
        envs: list[FactorioEnv] = []
        used_seeds: list[int] = []
        obs_list: list[np.ndarray] = []
        for kind, seed in zip(kinds[group], seeds[group]):
            env = FactorioEnv(size=size, idx=0)
            # Only the terminal throughput is ever shown, and the per-step
            # simulate_throughput otherwise dominates the rollout.
            env._full_diagnostics = False
            obs, used = _reset_rollout_env(env, kind, seed, num_missing_entities)
            envs.append(env)
            used_seeds.append(used)
            obs_list.append(obs)

        obs_NCWH = np.stack(obs_list)
        active = [True] * len(envs)
        with torch.no_grad():
            while any(active):
                batch = torch.as_tensor(
                    obs_NCWH, dtype=torch.float32, device=_AGENT_DEVICE
                )
                out = agent.sample_action(
                    batch, temperature=0.0, legal_mask=legal_mask,
                    eot_threshold=EOT_STOP_THRESHOLD, compute_value=False,
                )
                act = out["action"]
                xy = act["xy"].cpu().numpy()
                ent = act["entity"].reshape(-1).cpu().numpy()
                dirs = act["direction"].reshape(-1).cpu().numpy()
                item = act["item"].reshape(-1).cpu().numpy()
                misc = act["misc"].reshape(-1).cpu().numpy()
                eot = act["eot"].reshape(-1).cpu().numpy()
                step += 1
                for i, env in enumerate(envs):
                    if not active[i]:
                        continue
                    action = {
                        "xy": np.array([int(xy[i, 0]), int(xy[i, 1])], dtype=int),
                        "entity": int(ent[i]),
                        "direction": int(dirs[i]),
                        "item": int(item[i]),
                        "misc": int(misc[i]),
                        "eot": int(eot[i]),
                    }
                    next_obs, _reward, terminated, truncated, info = env.step(action)
                    obs_NCWH[i] = next_obs
                    if terminated or truncated:
                        active[i] = False
                        yield _rollout_result(
                            base + i, env, used_seeds[i], info, terminated
                        )
                yield {"type": "progress", "step": step}
    yield {"type": "done"}


def _batch_rollout_request(payload: dict) -> Iterator[dict]:
    """Turn a /batch_rollout POST body into a stream of scan events.

    Errors are yielded rather than raised: the response headers are already
    on the wire by the time this generator runs, so the browser can only be
    told about a failure in-band.
    """
    try:
        size = int(payload.get("size", 11))
        count = max(1, int(payload.get("count", 10)))
        start_seed = int(payload.get("seed", 0))
        kind_name = payload.get("kind") or ALL_KINDS_SENTINEL
        if kind_name == ALL_KINDS_SENTINEL:
            # Cycle kinds before seeds so a scan of N < len(LessonKind) is a
            # breadth-first sweep of setups, which is what "one of each" means.
            every = list(LessonKind)
            kinds = [every[i % len(every)] for i in range(count)]
            seeds = [start_seed + i // len(every) for i in range(count)]
        else:
            kinds = [LessonKind[kind_name]] * count
            seeds = [start_seed + i for i in range(count)]
        clear = payload.get("num_missing_entities")
        num_missing = None if clear in (None, "") else max(0, int(clear))
        yield from _batch_rollout(
            kinds=kinds,
            seeds=seeds,
            size=size,
            num_missing_entities=num_missing,
            legal_mask=bool(payload.get("legal_mask", True)),
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
    scan_lesson_options = (
        f'<option value="{ALL_KINDS_SENTINEL}">(every kind)</option>'
        + lesson_options
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
        '<h3 style="margin-top:0.8em;">Prediction</h3>'
        '<div id="model-info" class="help">(no prediction yet)</div>'
        '<pre id="model-action"></pre>'
        '<div class="model-buttons">'
        '<button id="model-apply" title="Apply the predicted placement at the highlighted tile">'
        'Apply prediction <span class="kbd">a</span></button>'
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
  .thput-icon {{ display: inline-flex; }}
  .thput-icon.ok {{ color: #1a7f37; }}
  .thput-icon.bad {{ color: #c62828; }}
  .thput-icon.flow {{ color: #1a7f37; opacity: 0.5; }}
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
  table.grid td.predicted {{
    box-shadow: inset 0 0 0 3px #0d47a1;
  }}
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
  /* "ghost" = a predicted placement drawn on top of an empty cell. One
     ghost per candidate tile (all tiles with p(tile) > threshold).
     Opacity is set inline per element so it can encode the model's
     confidence — solid for high p(tile), faded for marginal ones. The
     argmax tile additionally gets the dark-blue inset border via
     td.predicted so the top pick stays unambiguous. */
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
    <div class="controls">
      <label>size <input id="size" type="number" min="2" max="20" value="{default_size}"></label>
      <button id="resize" title="Resize the grid and clear all cells">resize / clear <span class="kbd">c</span></button>
      <button id="export" title="Copy {{size, grid}} JSON to clipboard">copy state JSON</button>
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
    <label>lesson
      <select id="scan-kind">{scan_lesson_options}</select>
    </label>
    <label title="How many rollouts to add per click. With '(every kind)' the kinds cycle first, so N = the number of kinds gives one seed of each.">
      rollouts <input id="scan-count" type="number" min="1" value="50">
    </label>
    <label title="Seed the next batch starts from. Advances by the rollout count after each run, so repeated clicks keep drawing fresh factories.">
      start seed <input id="scan-seed" type="number" value="0" step="1">
    </label>
    <label title="Entities to remove before the model rebuilds. Blank = remove everything the lesson allows (source, sink and reserved tiles always survive).">
      entities to clear <input id="scan-clear" type="number" min="0" placeholder="all" style="width:4.5em">
    </label>
    <label title="Restrict the tile head's argmax to empty, buildable cells — what sft.run_rollout_eval does. Off shows the raw head, including illegal proposals.">
      <input id="scan-mask" type="checkbox" checked> legal-tile mask
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
    <button id="scan-run" title="Blank each seed's factory, let the model rebuild it until its EOT head fires, and add every final factory to the gallery">
      Run scan
    </button>
    <button id="scan-stop" disabled>Stop</button>
    <button id="scan-clear-results" title="Throw away everything scanned so far">clear</button>
  </div>
  <div class="scan-summary" id="scan-summary">no scan yet</div>
  <div class="scan-stats" id="scan-stats"></div>
  <div class="scan-results" id="scan-results"></div>
</div>

<script>
const ALL_KINDS = '{ALL_KINDS_SENTINEL}';
const NUM_LESSON_KINDS = {len(list(LessonKind))};
const HOTBAR = {json.dumps(HOTBAR)};
const DIR_ARROW = {{ NONE: '', NORTH: '↑', EAST: '→', SOUTH: '↓', WEST: '←' }};
const MISC_GLYPH = {{ NONE: '', UNDERGROUND_DOWN: '▼', UNDERGROUND_UP: '▲' }};
const DIR_CYCLE = ['NORTH', 'EAST', 'SOUTH', 'WEST'];
const EOT_STOP_THRESHOLD = {EOT_STOP_THRESHOLD};
const OK_ICON = '{OK_ICON}';
const BAD_ICON = '{BAD_ICON}';
const FLOW_ICON = '{FLOW_ICON}';
// `modelLoaded` is set by refreshModelInfo() at startup and after each
// successful /load_model call. Prediction calls are gated on it so we
// don't pester the server with /predict when no checkpoint is loaded.
let modelLoaded = false;

let SIZE = {default_size};
let grid = [];           // grid[y][x] = cell dict
let selected = null;     // {{x, y}} or null
let activeHotbar = null; // 0..9 or null
let prediction = null;   // last /predict response (or null)
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

// The stop head is authoritative for every apply path in the UI, not just
// the visualisation: once it fires, a rollout is over, so the page must not
// place anything more either.
function eotStop(pred) {{
  return !!pred && pred.eot_prob > EOT_STOP_THRESHOLD;
}}

function eotStopMessage(pred) {{
  return 'model says done (eot ' + fmtPct(pred.eot_prob) + ')';
}}

function renderGrid() {{
  const host = document.getElementById('grid-host');
  // Build (x,y) -> candidate map once per render so the per-cell
  // ghost lookup is O(1).
  // When the model's EOT probability crosses 0.5 it's saying "I'm done
  // placing things" — the candidate ghosts would just be misleading
  // hallucinations of a forced placement, so suppress them.
  const candByXY = {{}};
  if (prediction && prediction.candidates && !eotStop(prediction)) {{
    for (const c of prediction.candidates) candByXY[c.x + ',' + c.y] = c;
  }}
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
      // The blue argmax border tracks the same suppression rule as the
      // ghost overlays: if the model says it's done (eot > 0.5), don't
      // visually nominate a "next placement" tile.
      if (
        prediction && prediction.x === x && prediction.y === y
        && !eotStop(prediction)
      ) td.classList.add('predicted');

      const inner = document.createElement('div');
      inner.className = 'cell-inner';
      let html = `<div class="xy">${{x}},${{y}}</div>` + cellGlyphs(c);
      // Ghost overlay: render every candidate tile (all tiles where
      // p(tile) > threshold) on top of empty cells, with opacity
      // proportional to the tile probability so the user can see what
      // the model is *considering*, not just the top pick. The argmax
      // tile additionally gets the dark-blue inset border (above).
      // Skip non-empty cells to keep the visualisation legible.
      const cand = candByXY[x + ',' + y];
      if (cand && c.entity === 'empty') {{
        // Map p in [0.05, 1.0] to opacity in [0.18, 0.95] so the
        // weakest visible ghost still has some presence and the
        // strongest reads as near-solid.
        const op = (0.18 + 0.77 * cand.p_tile).toFixed(2);
        if (cand.entity && cand.entity !== 'empty')
          html += `<i class="ent ghost ic-${{cand.entity}}" style="opacity:${{op}}"></i>`;
        if (cand.item && cand.item !== 'empty')
          html += `<i class="itm ghost ic-${{cand.item}}" style="opacity:${{op}}"></i>`;
        const garrow = DIR_ARROW[cand.direction] || '';
        if (garrow) html += `<div class="arrow ghost" style="opacity:${{op}}">${{garrow}}</div>`;
        const gmisc = MISC_GLYPH[cand.misc] || '';
        if (gmisc) html += `<div class="misc ghost" style="opacity:${{op}}">${{gmisc}}</div>`;
        // Percentage badge bottom-right: same data the ghost opacity
        // encodes, but as a precise number for tiles where the eye
        // can't tell a 60% ghost from a 75% ghost.
        const pct = cand.p_tile >= 0.01
          ? Math.round(cand.p_tile * 100) + '%'
          : '<1%';
        html += `<div class="p-badge" style="color:${{pBadgeColor(cand.p_tile)}}">${{pct}}</div>`;
      }}
      inner.innerHTML = html;
      td.appendChild(inner);

      td.addEventListener('click', () => {{
        selected = {{x, y}};
        if (activeHotbar !== null) {{
          const ent = HOTBAR[activeHotbar];
          if (ent !== null) {{
            placeEntity(x, y, ent);
            return;
          }}
        }}
        // Clicking a tile that's showing a ghost prediction applies it,
        // just like Apply but for *any* candidate — not only the blue
        // argmax. Gated on the same condition that drew the ghost here
        // (candidate present + cell empty) so the click does exactly
        // what the user sees. An active hotbar still wins (handled
        // above): an explicit palette pick overrides the suggestion.
        const cand = candByXY[x + ',' + y];
        if (cand && c.entity === 'empty') {{
          applyCandidate(cand);
          return;
        }}
        renderGrid(); syncEditor();
      }});
      td.addEventListener('contextmenu', (ev) => {{
        ev.preventDefault();
        grid[y][x] = emptyCell();
        renderGrid();
        if (selected && selected.x === x && selected.y === y) syncEditor();
        scheduleCompute();
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
          grid[y][x] = Object.assign({{}}, grid[fy][fx]);
          grid[fy][fx] = emptyCell();
          selected = {{ x, y }};
          renderGrid(); syncEditor();
          scheduleCompute();
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
      grid[selected.y][selected.x][field] = ev.target.value;
      renderGrid();
      scheduleCompute();
    }});
  }}
  document.getElementById('clear-cell').addEventListener('click', () => {{
    if (!selected) return;
    grid[selected.y][selected.x] = emptyCell();
    renderGrid(); syncEditor();
    scheduleCompute();
  }});
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

function placeEntity(x, y, ent) {{
  if (ent === 'empty') {{
    grid[y][x] = emptyCell();
  }} else {{
    grid[y][x].entity = ent;
    if (grid[y][x].direction === 'NONE') grid[y][x].direction = 'EAST';
  }}
  selected = {{x, y}};
  renderGrid(); syncEditor();
  scheduleCompute();
}}

function rotateSelected(cw) {{
  if (!selected) return;
  const c = grid[selected.y][selected.x];
  let i = DIR_CYCLE.indexOf(c.direction);
  if (i < 0) {{
    c.direction = cw ? 'NORTH' : 'WEST';
  }} else {{
    c.direction = DIR_CYCLE[(i + (cw ? 1 : -1) + 4) % 4];
  }}
  renderGrid(); syncEditor();
  scheduleCompute();
}}

function clearSelected() {{
  if (!selected) return;
  grid[selected.y][selected.x] = emptyCell();
  renderGrid(); syncEditor();
  scheduleCompute();
}}

// The verdict above the grid. The flag is the whole point: it answers "did my
// factory work?" before anyone has to know what a good number looks like.
const THPUT_LABEL = '<span>factory throughput:</span>';

function showThputPending() {{
  document.getElementById('thput').innerHTML = THPUT_LABEL +
    '<span class="spinner"></span><span class="thput-sub">calculating…</span>';
}}

function showThput(data) {{
  const el = document.getElementById('thput');
  if (data.thput === null || data.thput === undefined) {{
    el.innerHTML = THPUT_LABEL +
      '<span class="thput-sub">' + escHtml(data.note || 'unavailable') + '</span>';
    return;
  }}
  const flowing = data.thput > 0;
  el.innerHTML = THPUT_LABEL +
    '<span class="thput-icon ' + (flowing ? 'ok' : 'bad') + '">' +
      (flowing ? OK_ICON : BAD_ICON) + '</span>' +
    '<span class="thput-value">' + data.thput.toFixed(2) +
      ' items per second</span>' +
    (flowing ? '<span class="thput-icon flow">' + FLOW_ICON + '</span>' : '') +
    (flowing
      ? (data.unreachable
          ? '<span class="thput-sub">' + data.unreachable + ' unreachable</span>'
          : '')
      : '<span class="thput-sub">nothing reaches a sink</span>');
}}

// Every response that already built the world carries the throughput, so this
// is only for the case where nothing else will ask: no model loaded.
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

let _predictionTimer = null;
let _graphTimer = null;
function cancelCompute() {{
  clearTimeout(_predictionTimer);
  clearTimeout(_graphTimer);
}}
function scheduleCompute() {{
  cancelCompute();
  // Whatever number is on screen belongs to a grid that no longer exists.
  showThputPending();
  if (autoApplying) return;
  // Prediction is interactive; refresh it almost immediately, and the
  // throughput rides back with it. Matplotlib graph rendering is ~1s and
  // shares the server thread, so only do it after the user has actually
  // paused — the readout must never queue behind it.
  _predictionTimer = setTimeout(computePrediction, 25);
  _graphTimer = setTimeout(computeGraph, 1000);
}}

// Format a probability as a short percent string. Matches the user's
// preferred ".3%" style (no leading zero for sub-1% values) so the
// numbers stay compact even when the top-p tail gets long.
function fmtPct(p) {{
  const v = p * 100;
  if (v >= 10) return v.toFixed(1) + '%';
  if (v >= 1)  return v.toFixed(1) + '%';
  return v.toFixed(1).replace(/^0/, '') + '%';
}}

function fmtTopNamed(top, rest) {{
  const parts = top.map(t => t.name + ' (' + fmtPct(t.p) + ')');
  parts.push('rest (' + fmtPct(rest) + ')');
  return parts.join(', ');
}}

function fmtTopTile(top, rest) {{
  const parts = top.map(t => '(' + t.x + ',' + t.y + ') (' + fmtPct(t.p) + ')');
  parts.push('rest (' + fmtPct(rest) + ')');
  return parts.join(', ');
}}

async function computePrediction() {{
  if (!modelLoaded) {{
    prediction = null;
    const info = document.getElementById('model-info');
    if (info) info.textContent = '(no model loaded)';
    const out = document.getElementById('model-action');
    if (out) out.textContent = '';
    renderGrid();
    computeThroughput();
    return;
  }}
  const info = document.getElementById('model-info');
  const out = document.getElementById('model-action');
  if (info) info.textContent = 'predicting…';
  try {{
    const resp = await fetch('/predict', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ grid }}),
    }});
    const data = await resp.json();
    if (data.error) {{
      if (info) info.textContent = 'error: ' + data.error;
      if (out) out.textContent = '';
      prediction = null;
      renderGrid();
      showThput({{ thput: null, note: 'error: ' + data.error }});
      return;
    }}
    prediction = data;
    showThput(data);
    if (info) {{
      info.textContent = eotStop(data)
        ? eotStopMessage(data) + ' — no placement offered'
        : 'predicted next placement at (' + data.x + ', ' + data.y + ')';
    }}
    if (out) {{
      // Each line: "head:   cand1 (p1), cand2 (p2), ..., rest (R)".
      // The <pre> uses white-space:pre + overflow-x:auto so long top-p
      // lines scroll horizontally instead of wrapping.
      // EOT line: model's "I'm done" probability. The {{stop}} /
      // {{continue}} marker matches the threshold in
      // agent.eot_should_stop — a quick read for whether the model
      // would terminate an inference rollout right now.
      const eotPct = fmtPct(data.eot_prob);
      const eotMark = eotStop(data) ? '[stop]' : '[continue]';
      const lines = [
        '  eot:       ' + eotPct + ' ' + eotMark,
        '  tile:      ' + fmtTopTile(data.tile_top, data.tile_rest),
        '  entity:    ' + fmtTopNamed(data.entity_top, data.entity_rest),
        '  direction: ' + fmtTopNamed(data.direction_top, data.direction_rest),
        '  item:      ' + fmtTopNamed(data.item_top, data.item_rest),
        '  misc:      ' + fmtTopNamed(data.misc_top, data.misc_rest),
      ];
      out.textContent = lines.join('\\n');
    }}
    renderGrid();
  }} catch (e) {{
    if (info) info.textContent = 'predict failed: ' + e;
  }}
}}

async function requestFastPrediction() {{
  const resp = await fetch('/predict', {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json' }},
    body: JSON.stringify({{ grid, detail: 'action' }}),
  }});
  const data = await resp.json();
  if (data.error) throw new Error(data.error);
  return data;
}}

// Apply through the server-side placement function shared with
// FactorioEnv.step. The browser deliberately has no copy of multi-tile
// geometry or action-validity rules.
async function applyCandidate(cand, interactive = true) {{
  const {{ x, y, entity, direction, item, misc }} = cand;
  const info = document.getElementById('model-info');
  try {{
    const resp = await fetch('/apply_prediction', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ grid, prediction: {{
        x, y, entity, direction, item, misc,
      }} }}),
    }});
    const data = await resp.json();
    if (data.error) {{
      if (info) info.textContent = 'apply failed: ' + data.error;
      return false;
    }}
    if (!data.applied) {{
      if (info) {{
        info.textContent =
          'prediction rejected: ' + (data.invalid_reason || 'invalid action');
      }}
      return false;
    }}
    grid = data.grid;
    selected = {{ x, y }};
    prediction = null;
    if (interactive) {{
      renderGrid(); syncEditor();
      scheduleCompute();
    }}
    return true;
  }} catch (e) {{
    if (info) info.textContent = 'apply failed: ' + e;
    return false;
  }}
}}

function applyPrediction() {{
  if (!prediction) return;
  if (eotStop(prediction)) {{
    const info = document.getElementById('model-info');
    if (info) info.textContent = eotStopMessage(prediction) + ' — nothing applied';
    return;
  }}
  applyCandidate(prediction);
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
  const info = document.getElementById('model-info');
  let count = 0;
  const started = performance.now();
  try {{
    while (autoApplying && generation === autoApplyGeneration) {{
      const action = await requestFastPrediction();
      if (!autoApplying || generation !== autoApplyGeneration) break;
      showThput(action);
      if (eotStop(action)) {{
        autoApplying = false;
        if (info) {{
          info.textContent =
            'fast apply: stopped after ' + count + ' placements · ' +
            eotStopMessage(action);
        }}
        break;
      }}
      const applied = await applyCandidate(action, false);
      if (!applied) {{
        autoApplying = false;
        break;
      }}
      count += 1;
      renderAutoApplyFrame();
      if (info) {{
        const elapsed = Math.max((performance.now() - started) / 1000, 0.001);
        info.textContent =
          'fast apply: ' + count + ' placements · ' +
          (count / elapsed).toFixed(1) + '/s';
      }}
    }}
  }} catch (e) {{
    autoApplying = false;
    if (info) info.textContent = 'fast apply failed: ' + e;
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
  // Restore the rich probability/ghost view promptly; leave the PNG until
  // one second of idle so it can never throttle the held-key loop.
  scheduleCompute();
}}

function beginApplyKey() {{
  if (!modelLoaded) return;
  applyKeyHeld = true;
  cancelCompute();
  // Preserve tap-to-apply: consume the visible prediction once, then only
  // enter the continuous request pump if the key remains down.
  let firstApply = Promise.resolve(true);
  if (eotStop(prediction)) {{
    // Resolving false also keeps the hold from entering the request pump.
    firstApply = Promise.resolve(false);
    const info = document.getElementById('model-info');
    if (info) info.textContent = eotStopMessage(prediction) + ' — nothing applied';
  }} else if (prediction) {{
    firstApply = applyCandidate(prediction, false);
    firstApply.then((applied) => {{
      if (applied) renderAutoApplyFrame();
    }});
  }}
  clearTimeout(autoApplyHoldTimer);
  autoApplyHoldTimer = setTimeout(async () => {{
    autoApplyHoldTimer = null;
    const applied = await firstApply;
    if (applyKeyHeld && applied) startAutoApply();
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
async function copyYaml(g, btn, source) {{
  const icon = btn.innerHTML;
  try {{
    const resp = await fetch('/factory_yaml', {{
      method: 'POST',
      headers: {{ 'Content-Type': 'application/json' }},
      body: JSON.stringify({{ grid: g, source }}),
    }});
    const data = await resp.json();
    if (data.error) throw new Error(data.error);
    await navigator.clipboard.writeText(data.yaml);
    console.log(data.yaml);
    btn.textContent = '✓';
  }} catch (e) {{
    btn.textContent = '✗';
    console.error(e);
  }}
  setTimeout(() => {{ btn.innerHTML = icon; }}, 1500);
}}
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

document.getElementById('model-apply').addEventListener('click', applyPrediction);
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
      const shape = 'layers=' + (data.layers || []).join('-') +
        ' k=' + data.kernel_size + ' ' + data.device;
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
    // Recompute prediction with the new weights so the UI updates
    // immediately instead of waiting for the next grid edit.
    computePrediction();
  }} catch (e) {{
    status.textContent = 'load failed: ' + e;
  }} finally {{
    btn.disabled = false;
  }}
}}
// Adopt a whole grid as the Build tab's working state. The stale selection
// and prediction have to be dropped along with the old grid — otherwise the
// ghost overlays and argmax border are drawn over a factory they don't
// belong to — so every wholesale swap goes through here.
function adoptGrid(size, cells, source) {{
  SIZE = size;
  grid = cells;
  selected = null;
  prediction = null;
  gridSource = source || null;
  gridSnapshot = JSON.stringify(grid);
  document.getElementById('size').value = SIZE;
  renderGrid(); syncEditor();
  scheduleCompute();
}}

async function generateLesson() {{
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
document.getElementById('lesson-generate').addEventListener('click', generateLesson);

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
  if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
  if (/^[0-9]$/.test(ev.key)) {{
    const n = parseInt(ev.key, 10);
    const idx = (n === 0) ? 9 : n - 1;
    setActiveHotbar(idx);
    ev.preventDefault();
    return;
  }}
  if (ev.key === 'r') {{ rotateSelected(true); ev.preventDefault(); return; }}
  if (ev.key === 'R') {{ rotateSelected(false); ev.preventDefault(); return; }}
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
    if (activeHotbar !== null) setActiveHotbar(activeHotbar);
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
// A scan blanks N factories, lets the model rebuild each one until its EOT
// head fires, and lays the finished factories out side by side.
let scanResults = [];
let scanAbort = null;
let scanIndexBase = 0;

function switchTab(name) {{
  document.querySelectorAll('#tabs button').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === name));
  document.getElementById('tab-build').hidden = (name !== 'build');
  document.getElementById('tab-scan').hidden = (name !== 'scan');
}}

// The card's left border is the fastest read in the gallery — a cluster of
// failures registers before any text resolves.
function thputColor(t) {{
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
  card.style.borderLeftColor = thputColor(r.thput_normed);
  const stop = r.stopped_by === 'eot'
    ? 'stopped at ' + r.steps
    : 'no stop, ' + r.steps + ' steps';
  const head =
    `<div class="hd">${{escHtml(r.kind)}}<button class="copy-yaml"` +
    ` title="Copy this factory as a YAML test fixture">{COPY_ICON}</button></div>` +
    `<div class="sub">seed ${{r.seed}} · thput ${{r.thput_normed.toFixed(3)}}` +
    ` (${{r.thput_raw.toFixed(2)}} of ${{r.max_throughput.toFixed(2)}} i/s)</div>` +
    `<div class="sub">${{stop}} · ${{r.num_placed_entities}} placed · ` +
    `${{r.invalid_actions}} invalid · reach ${{Math.round(r.frac_reachable * 100)}}%</div>`;
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
  card.addEventListener('click', () => {{
    switchTab('build');
    adoptGrid(
      r.size, r.grid.map(row => row.map(c => Object.assign({{}}, c))), source,
    );
  }});
  return card;
}}

function renderScan() {{
  const host = document.getElementById('scan-results');
  const showRef = document.getElementById('scan-ref').checked;
  const mode = document.getElementById('scan-sort').value;
  const rows = scanResults.slice();
  if (mode === 'worst') {{
    rows.sort((a, b) => a.thput_normed - b.thput_normed || a.index - b.index);
  }} else if (mode === 'best') {{
    rows.sort((a, b) => b.thput_normed - a.thput_normed || a.index - b.index);
  }} else {{
    rows.sort((a, b) => a.index - b.index);
  }}
  host.replaceChildren(...rows.map(r => scanCard(r, showRef)));
  renderScanStats();
}}

// A factory counts as complete at thput 1.0 — the same "already done"
// test sft.run_rollout_eval scores the EOT head against.
const COMPLETE_THPUT = 0.999;

function scanSummary(status) {{
  const el = document.getElementById('scan-summary');
  const n = scanResults.length;
  if (!n) {{ el.textContent = status || 'no scan yet'; return; }}
  const t = scanResults.map(r => r.thput_normed);
  const mean = t.reduce((a, b) => a + b, 0) / n;
  const zeros = t.filter(v => v <= 0).length;
  const perfect = t.filter(v => v >= COMPLETE_THPUT).length;
  const eot = scanResults.filter(r => r.stopped_by === 'eot').length;
  el.textContent =
    `${{n}} done · mean thput ${{mean.toFixed(3)}} · ${{zeros}} at zero · ` +
    `${{perfect}} perfect · eot fired ${{eot}}/${{n}}` +
    (status ? '  ·  ' + status : '');
}}

// Per-lesson breakdown. A mean over the whole scan hides the thing worth
// finding — that one lesson is at zero while the rest are fine — so every
// kind gets its own row, worst mean first.
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
      complete: rs.filter(r => r.thput_normed >= COMPLETE_THPUT).length,
      eot: rs.filter(r => r.stopped_by === 'eot').length,
    }};
  }});
  rows.sort((a, b) => a.mean - b.mean || a.kind.localeCompare(b.kind));
  const head =
    '<tr><th>lesson</th><th>runs</th><th>mean thput</th>' +
    '<th>thput &gt; 0</th><th>complete</th><th>eot fired</th></tr>';
  const body = rows.map(row =>
    `<tr><td class="kind">${{escHtml(row.kind)}}</td><td>${{row.n}}</td>` +
    `<td style="color:${{thputColor(row.mean)}}">${{row.mean.toFixed(3)}}</td>` +
    `<td>${{frac(row.nonzero, row.n)}}</td><td>${{frac(row.complete, row.n)}}</td>` +
    `<td>${{frac(row.eot, row.n)}}</td></tr>`
  ).join('');
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
  const clearRaw = parseInt(document.getElementById('scan-clear').value, 10);
  const body = {{
    kind: document.getElementById('scan-kind').value,
    count: parseInt(document.getElementById('scan-count').value, 10) || 1,
    seed: parseInt(document.getElementById('scan-seed').value, 10) || 0,
    size: SIZE,
    legal_mask: document.getElementById('scan-mask').checked,
    num_missing_entities:
      Number.isFinite(clearRaw) && clearRaw >= 0 ? clearRaw : null,
  }};
  // Each run appends. Indices are offset by the running total so they stay
  // unique across runs and keep the server's within-run ordering, which is
  // what the "run order" sort reads.
  const indexBase = scanIndexBase;
  scanIndexBase += body.count;
  const before = scanResults.length;
  scanAbort = new AbortController();
  runBtn.disabled = true;
  stopBtn.disabled = false;
  const started = performance.now();
  const showRef = document.getElementById('scan-ref').checked;
  let total = body.count;
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
          scanSummary('running ' + total + ' rollouts…');
        }} else if (ev.type === 'result') {{
          ev.index += indexBase;
          scanResults.push(ev);
          // Append rather than re-render: a full re-sort of N cards on
          // every arrival is O(N^2) table builds. renderScan() applies
          // the chosen sort once the stream ends.
          document.getElementById('scan-results').appendChild(scanCard(ev, showRef));
          renderScanStats();
        }} else if (ev.type === 'progress') {{
          const secs = (performance.now() - started) / 1000;
          scanSummary(
            'step ' + ev.step + ' · ' + (scanResults.length - before) + '/' +
            total + ' finished · ' + secs.toFixed(1) + 's'
          );
        }} else if (ev.type === 'error') {{
          scanSummary('error: ' + ev.error);
        }}
      }}
    }}
    renderScan();
    // Advance the seed so the next click draws fresh factories rather than
    // rebuilding the same ones. "(every kind)" cycles kinds before seeds, so
    // its run only consumes count/kinds seeds.
    const seedInput = document.getElementById('scan-seed');
    seedInput.value = body.seed + (body.kind === ALL_KINDS
      ? Math.ceil(body.count / NUM_LESSON_KINDS)
      : body.count);
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
  scanSummary('');
}}

function syncScanRunLabel() {{
  const c = parseInt(document.getElementById('scan-count').value, 10) || 1;
  document.getElementById('scan-run').textContent = 'Run +' + c;
}}

function bindScan() {{
  document.querySelectorAll('#tabs button').forEach(b =>
    b.addEventListener('click', () => switchTab(b.dataset.tab)));
  document.getElementById('scan-run').addEventListener('click', runScan);
  document.getElementById('scan-stop').addEventListener('click', () => {{
    if (scanAbort) scanAbort.abort();
  }});
  document.getElementById('scan-count').addEventListener('input', syncScanRunLabel);
  document.getElementById('scan-clear-results').addEventListener('click', clearScan);
  document.getElementById('scan-sort').addEventListener('change', renderScan);
  document.getElementById('scan-ref').addEventListener('change', renderScan);
  syncScanRunLabel();
}}

grid = newGrid(SIZE);
renderGrid();
bindHotbar();
bindEditor();
bindHelp();
bindScan();
refreshModelInfo();
computeThroughput();
</script>
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
            "/predict",
            "/apply_prediction",
            "/load_model",
            "/load_lesson",
            "/batch_rollout",
            "/factory_yaml",
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
            elif self.path == "/predict":
                if payload.get("detail") == "action":
                    result = _predict_action(payload["grid"])
                else:
                    result = _predict(payload["grid"])
            elif self.path == "/apply_prediction":
                result = _apply_prediction(
                    payload["grid"], payload["prediction"]
                )
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
        _load_checkpoint(ckpt_path)
        _CHECKPOINT_SOURCE = source
    elif args.checkpoint:
        _load_checkpoint(args.checkpoint)
        _CHECKPOINT_SOURCE = {"kind": "local", "path": args.checkpoint}
    else:
        print("(no checkpoint — model prediction panel disabled)")

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
