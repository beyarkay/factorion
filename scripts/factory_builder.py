"""Interactive factory builder.

Spins up a tiny local HTTP server that serves a drag-and-drop UI for
designing a factory and visualising the flow graph it produces.

The browser POSTs the grid to the server, which runs ``world2graph`` for
visualisation and ``factorion_rs.simulate_throughput`` for throughput,
then returns a rendered graph image.

Usage:
    uv run python scripts/factory_builder.py
    # then open http://localhost:8765
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import tyro  # noqa: E402

os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_DISABLED", "true")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion_rs  # noqa: E402
from factorion import (  # noqa: E402
    Channel,
    Direction,
    Footprint,
    Misc,
    ent_str2b64img,
    entities,
    items,
    new_world,
    plot_flow_network,
    world2graph,
)
from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402


# Order shown in the palette and dropdowns.
PLACEABLE_ENTITIES = [
    "transport_belt",
    "underground_belt",
    "splitter",
    "inserter",
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
    None,
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
DIRECTIONS = ["NONE", "NORTH", "EAST", "SOUTH", "WEST"]
MISC_VALUES = ["NONE", "UNDERGROUND_DOWN", "UNDERGROUND_UP"]


@dataclass
class Args:
    port: int = 8765
    """port for the local HTTP server"""
    size: int = 8
    """default grid size"""
    checkpoint: Optional[str] = None
    """path to a trained SFT/PPO checkpoint (.pt). If set, the UI shows
    the model's predicted next placement and exposes an Apply button."""
    wandb_run: Optional[str] = None
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


def render_graph_png(grid: list[list[dict]]) -> dict:
    """Build the world, run world2graph, and return a base64 PNG plus
    text describing the nodes/edges/throughput."""
    world = build_world(grid)
    G = world2graph(world)
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

    try:
        throughput, num_unreachable = factorion_rs.simulate_throughput(
            world.numpy().astype(np.int64)
        )
        info = f"throughput: {throughput:.4f}  ·  unreachable nodes: {num_unreachable}"
    except Exception as e:
        info = f"throughput failed: {e}"

    # Node names contain literal '\n' (e.g. "transport_belt\n@0,0"), so use
    # repr() to make embedded newlines visible as \n in the edge list panel
    # instead of breaking the line mid-name.
    edges = [[repr(u), repr(v)] for u, v in G.edges]
    return {"png": png_b64, "info": info, "edges": edges}


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

# Reverse lookup: head index -> readable name. The entity head excludes the
# last two catalog entries (source/sink) but its index space starts at 0 and
# aligns with the first N-2 Item values, so entities[idx].name works as-is.
_ENT_NAMES = {ent.value: ent.name for ent in entities.values()}
_ITEM_NAMES = {it.value: it.name for it in items.values()}
_DIR_NAMES = {d.value: d.name for d in Direction}
_MISC_NAMES = {m.value: m.name for m in Misc}


def _load_checkpoint(path: str) -> None:
    """Load the checkpoint and stash it. Clears the per-size agent
    cache so subsequent /predict calls rebuild against the new
    weights — important when this is invoked at runtime by the UI to
    swap models, not just once at startup."""
    global _CHECKPOINT_STATE, _CHECKPOINT_PATH
    state = torch.load(path, map_location="cpu", weights_only=True)
    _CHECKPOINT_STATE = state
    _CHECKPOINT_PATH = path
    _AGENT_CACHE.clear()
    chan1 = state["encoder.0.weight"].shape[0]
    chan2 = state["encoder.2.weight"].shape[0]
    chan3 = state["encoder.4.weight"].shape[0]
    print(
        f"Loaded checkpoint {path} "
        f"(chan1={chan1}, chan2={chan2}, chan3={chan3}, device={_AGENT_DEVICE})"
    )


def _model_info() -> dict:
    """Snapshot of the currently-loaded model for the UI's status line.
    Returns `loaded: False` when nothing is loaded yet, so the UI can
    render "(none loaded)" without special-casing on the JS side."""
    if _CHECKPOINT_STATE is None:
        return {"loaded": False}
    s = _CHECKPOINT_STATE
    return {
        "loaded": True,
        "path": _CHECKPOINT_PATH,
        "chan1": s["encoder.0.weight"].shape[0],
        "chan2": s["encoder.2.weight"].shape[0],
        "chan3": s["encoder.4.weight"].shape[0],
        "device": str(_AGENT_DEVICE),
    }


def _swap_model(kind: str, value: str, project: str, entity: Optional[str]) -> dict:
    """Resolve `value` to a local .pt path based on `kind`, load it,
    and return the new model info. Called by the /load_model POST
    endpoint so the user can switch models without restarting the
    server."""
    value = (value or "").strip()
    if not value:
        raise ValueError("value cannot be empty")
    if kind == "local":
        if not Path(value).exists():
            raise FileNotFoundError(f"checkpoint not found: {value}")
        path = value
    elif kind == "wandb":
        path = _resolve_wandb_checkpoint(value, project, entity)
    else:
        raise ValueError(f"unknown kind: {kind!r} (expected 'local' or 'wandb')")
    _load_checkpoint(path)
    return _model_info()


def _get_agent(size: int) -> AgentCNN:
    if _CHECKPOINT_STATE is None:
        raise RuntimeError("no checkpoint loaded — pass --checkpoint at launch")
    if size in _AGENT_CACHE:
        return _AGENT_CACHE[size]

    chan1 = _CHECKPOINT_STATE["encoder.0.weight"].shape[0]
    chan2 = _CHECKPOINT_STATE["encoder.2.weight"].shape[0]
    chan3 = _CHECKPOINT_STATE["encoder.4.weight"].shape[0]

    env_id = "factorion/FactorioEnv-v0-fb"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=FactorioEnv)
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, size, "fb")])
    try:
        agent = AgentCNN(envs, chan1=chan1, chan2=chan2, chan3=chan3)
    finally:
        envs.close()
    # critic_head's flat dim depends on W*H, so its saved weights are
    # the wrong shape whenever the UI grid size != the training size.
    # Drop those entries before loading; we never call the critic during
    # inference. strict=False would *not* save us here — it ignores
    # missing/unexpected keys but still raises on shape mismatches.
    filtered = {
        k: v for k, v in _CHECKPOINT_STATE.items()
        if not k.startswith("critic_head.")
    }
    missing, unexpected = agent.load_state_dict(filtered, strict=False)
    ignorable = {"critic_head.1.weight", "critic_head.1.bias"}
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


CANDIDATE_TILE_THRESHOLD = 0.05
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
        encoded_BCWH = agent.encoder(obs_CWH)
        tile_logits = agent.tile_logits(encoded_BCWH).reshape(1, -1)
        tile_probs = F.softmax(tile_logits, dim=-1)[0]
        tile_top, tile_rest = _tile_top_p(tile_probs, H)

        # Argmax — used both as the "Apply" target and to condition the
        # side-panel per-head distributions on the same tile the top
        # distribution favours.
        tile_idx = int(tile_probs.argmax().item())
        x, y = tile_idx // H, tile_idx % H

        feats = encoded_BCWH[0, :, x, y].unsqueeze(0)
        ent_top, ent_rest = _top_p_named(F.softmax(agent.ent_head(feats), dim=-1)[0], _ENT_NAMES)
        dir_top, dir_rest = _top_p_named(F.softmax(agent.dir_head(feats), dim=-1)[0], _DIR_NAMES)
        item_top, item_rest = _top_p_named(F.softmax(agent.item_head(feats), dim=-1)[0], _ITEM_NAMES)
        misc_top, misc_rest = _top_p_named(F.softmax(agent.misc_head(feats), dim=-1)[0], _MISC_NAMES)

        # Per-tile argmax for ent/dir/item/misc — one matmul per head
        # against the whole spatial map, reshaped to (W*H, chan3).
        feats_all = encoded_BCWH[0].permute(1, 2, 0).reshape(W * H, -1)
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
    }


# Cache palette icons so the page payload stays small per cell.
def _icon_b64(name: str) -> str:
    try:
        return ent_str2b64img(name)
    except Exception:
        return ""


PALETTE_ICONS = {n: _icon_b64(n) for n in PLACEABLE_ENTITIES + ["empty"]}
ITEM_ICONS = {n: _icon_b64(n) for n in NON_PLACEABLE_ITEMS}


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
    direction_options = "".join(
        f'<option value="{d}">{d}</option>' for d in DIRECTIONS
    )
    misc_options = "".join(
        f'<option value="{m}">{m}</option>' for m in MISC_VALUES
    )

    # Always rendered now — the section exposes a load form so the user
    # can switch models at runtime. When no model is loaded the
    # prediction subsection just shows "(no model loaded)".
    model_panel_html = (
        '<div class="model-panel">'
        '<h3 style="margin-top:0.8em;">'
        '<span class="swatch"></span>Model'
        '</h3>'
        '<div class="model-current help" id="model-current">checking…</div>'
        '<label>source'
        '  <select id="model-kind">'
        '    <option value="local">local path</option>'
        '    <option value="wandb">wandb run id</option>'
        '  </select>'
        '</label>'
        '<label>value'
        '  <input id="model-value" type="text" placeholder="sft_local.pt or run_id">'
        '</label>'
        '<div class="model-buttons">'
        '<button id="model-load">Load model</button>'
        '</div>'
        '<div id="model-load-status" class="help"></div>'
        '<h3 style="margin-top:0.8em;">Prediction</h3>'
        '<div id="model-info" class="help">'
        '(prediction will appear after the next change)</div>'
        '<pre id="model-action"></pre>'
        '<div class="model-buttons">'
        '<button id="model-apply">Apply prediction</button>'
        '<button id="model-refresh">Refresh</button>'
        '</div>'
        '</div>'
    )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Factory builder</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 1em; color: #222; }}
  h1 {{ margin: 0 0 0.4em; }}
  .layout {{ display: grid; grid-template-columns: 1fr 280px; gap: 1em; }}
  .panel {{ border: 1px solid #ccc; border-radius: 6px; padding: 0.6em; background: #fafafa; }}
  .panel h3 {{ margin: 0 0 0.4em; font-size: 0.9em; text-transform: uppercase; color: #555; }}
  .hotbar {{
    display: flex; gap: 0.3em; flex-wrap: wrap;
    user-select: none; -webkit-user-select: none;
  }}
  .hb-slot {{
    position: relative; width: 64px; height: 78px;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; gap: 2px;
    border: 2px solid #ddd; border-radius: 4px; background: white;
    cursor: grab; font-size: 0.7em; padding: 2px;
    user-select: none; -webkit-user-select: none;
  }}
  .hb-slot img {{ width: 32px; height: 32px; pointer-events: none; }}
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
  .grid-wrap {{ display: flex; flex-direction: column; align-items: flex-start; gap: 0.6em; min-width: 0; }}
  .grid-graph-row {{
    display: flex; gap: 1em; align-items: flex-start;
    width: 100%; min-width: 0;
  }}
  .grid-graph-row > #grid-host {{ flex: 0 0 auto; }}
  .graph-view {{ flex: 1 1 0; min-width: 0; }}
  .graph-view h3 {{ margin: 0 0 0.4em; font-size: 0.9em; text-transform: uppercase; color: #555; }}
  .controls {{ display: flex; gap: 0.5em; flex-wrap: wrap; align-items: center; }}
  .controls input[type=number] {{ width: 4em; }}
  table.grid {{ border-collapse: collapse; }}
  table.grid td {{
    width: 56px; height: 56px; border: 1px solid #bbb; padding: 0;
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
  .cell-inner img.ent {{ position: absolute; top: 10%; left: 10%; width: 60%; height: 60%; }}
  .cell-inner img.itm {{ position: absolute; bottom: 4%; right: 4%; width: 30%; height: 30%; }}
  .cell-inner .arrow {{
    position: absolute; bottom: 0; left: 2px;
    font-size: 16px; line-height: 16px; color: #222;
  }}
  .cell-inner .misc {{
    position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
    font-weight: bold; color: white; text-shadow: 0 0 2px black; font-size: 18px;
  }}
  .cell-inner .xy {{ position: absolute; top: 0; left: 1px; font-size: 8px; opacity: 0.5; }}
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
</style></head><body>

<h1>Factory builder</h1>
<p class="help">
  Press <b>1</b>–<b>9</b> or <b>0</b> to pick a hotbar slot, then click a
  tile to place it. Drag a slot onto a tile to do the same. Click a tile
  to select/edit it (ENTITY / DIRECTION / ITEM / MISC / FOOTPRINT in the
  right panel). With a tile selected: <b>r</b> rotates clockwise,
  <b>R</b> counter-clockwise; <b>Delete</b> / <b>Backspace</b> clears it;
  right-click also clears. The graph recomputes automatically on every
  change. <b>Esc</b> deselects the active hotbar slot.
</p>

<div class="layout">

  <div class="grid-wrap">
    <div class="hotbar" id="hotbar">{hotbar_html}</div>
    <div class="controls">
      <label>size <input id="size" type="number" min="2" max="20" value="{default_size}"></label>
      <button id="resize">resize / clear</button>
      <button id="compute">Compute graph</button>
      <button id="export">copy state JSON</button>
    </div>
    <div class="grid-graph-row">
      <div id="grid-host"></div>
      <div class="graph-view">
        <h3>Graph</h3>
        <div class="info" id="info">(graph will appear after Compute graph)</div>
        <img id="out-img" class="out-img" alt="" style="display:none">
        <pre class="edges" id="edges" style="display:none"></pre>
      </div>
    </div>
  </div>

  <div class="panel editor" id="editor">
    <h3>Selected cell</h3>
    <div id="sel-info" class="help">(click a cell to edit)</div>
    <label>entity
      <select id="ed-entity">{item_options}</select>
    </label>
    <label>direction
      <select id="ed-direction">{direction_options}</select>
    </label>
    <label>item (recipe / filter)
      <select id="ed-item">{item_options}</select>
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

<script>
const PALETTE_ICONS = {json.dumps(PALETTE_ICONS)};
const ITEM_ICONS = {json.dumps(ITEM_ICONS)};
const HOTBAR = {json.dumps(HOTBAR)};
const DIR_ARROW = {{ NONE: '', NORTH: '↑', EAST: '→', SOUTH: '↓', WEST: '←' }};
const MISC_GLYPH = {{ NONE: '', UNDERGROUND_DOWN: '⭳', UNDERGROUND_UP: '⭱' }};
const DIR_CYCLE = ['NORTH', 'EAST', 'SOUTH', 'WEST'];
// `modelLoaded` is set by refreshModelInfo() at startup and after each
// successful /load_model call. Prediction calls are gated on it so we
// don't pester the server with /predict when no checkpoint is loaded.
let modelLoaded = false;

let SIZE = {default_size};
let grid = [];           // grid[y][x] = cell dict
let selected = null;     // {{x, y}} or null
let activeHotbar = null; // 0..9 or null
let prediction = null;   // last /predict response (or null)

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

function iconFor(name) {{
  return PALETTE_ICONS[name] || ITEM_ICONS[name] || '';
}}

function renderGrid() {{
  const host = document.getElementById('grid-host');
  // Build (x,y) -> candidate map once per render so the per-cell
  // ghost lookup is O(1).
  const candByXY = {{}};
  if (prediction && prediction.candidates) {{
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
      if (prediction && prediction.x === x && prediction.y === y) td.classList.add('predicted');

      const inner = document.createElement('div');
      inner.className = 'cell-inner';
      let html = `<div class="xy">${{x}},${{y}}</div>`;
      if (c.entity && c.entity !== 'empty')
        html += `<img class="ent" src="${{iconFor(c.entity)}}">`;
      if (c.item && c.item !== 'empty')
        html += `<img class="itm" src="${{iconFor(c.item)}}">`;
      const arrow = DIR_ARROW[c.direction] || '';
      if (arrow) html += `<div class="arrow">${{arrow}}</div>`;
      const m = MISC_GLYPH[c.misc] || '';
      if (m) html += `<div class="misc">${{m}}</div>`;
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
          html += `<img class="ent ghost" style="opacity:${{op}}" src="${{iconFor(cand.entity)}}">`;
        if (cand.item && cand.item !== 'empty')
          html += `<img class="itm ghost" style="opacity:${{op}}" src="${{iconFor(cand.item)}}">`;
        const garrow = DIR_ARROW[cand.direction] || '';
        if (garrow) html += `<div class="arrow ghost" style="opacity:${{op}}">${{garrow}}</div>`;
        const gmisc = MISC_GLYPH[cand.misc] || '';
        if (gmisc) html += `<div class="misc ghost" style="opacity:${{op}}">${{gmisc}}</div>`;
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
        renderGrid(); syncEditor();
      }});
      td.addEventListener('contextmenu', (ev) => {{
        ev.preventDefault();
        grid[y][x] = emptyCell();
        renderGrid();
        if (selected && selected.x === x && selected.y === y) syncEditor();
        scheduleCompute();
      }});
      td.addEventListener('dragover', (ev) => ev.preventDefault());
      td.addEventListener('drop', (ev) => {{
        ev.preventDefault();
        const ent = ev.dataTransfer.getData('text/plain');
        if (!ent) return;
        placeEntity(x, y, ent);
      }});
      tr.appendChild(td);
    }}
    tbl.appendChild(tr);
  }}
  host.replaceChildren(tbl);
}}

function syncEditor() {{
  const info = document.getElementById('sel-info');
  if (!selected) {{ info.textContent = '(click a cell to edit)'; return; }}
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
      ev.dataTransfer.setData('text/plain', el.dataset.entity);
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

let _computeTimer = null;
function scheduleCompute() {{
  clearTimeout(_computeTimer);
  _computeTimer = setTimeout(() => {{
    computeGraph();
    computePrediction();
  }}, 200);
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
      return;
    }}
    prediction = data;
    if (info) {{
      info.textContent =
        'predicted next placement at (' + data.x + ', ' + data.y + ')';
    }}
    if (out) {{
      // Each line: "head:   cand1 (p1), cand2 (p2), ..., rest (R)".
      // The <pre> uses white-space:pre + overflow-x:auto so long top-p
      // lines scroll horizontally instead of wrapping.
      const lines = [
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

function applyPrediction() {{
  if (!prediction) return;
  const {{ x, y, entity, direction, item, misc }} = prediction;
  grid[y][x] = {{
    entity, direction, item, misc, footprint: grid[y][x].footprint,
  }};
  selected = {{ x, y }};
  prediction = null;
  renderGrid(); syncEditor();
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
    document.getElementById('edges').style.display = 'none';
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
    edges.style.display = 'block';
  }} else {{
    edges.style.display = 'none';
  }}
}}

document.getElementById('compute').addEventListener('click', () => {{
  computeGraph();
  computePrediction();
}});
document.getElementById('model-apply').addEventListener('click', applyPrediction);
document.getElementById('model-refresh').addEventListener('click', computePrediction);
document.getElementById('model-load').addEventListener('click', loadModel);

async function refreshModelInfo() {{
  const cur = document.getElementById('model-current');
  try {{
    const resp = await fetch('/model_info');
    const data = await resp.json();
    if (data.loaded) {{
      modelLoaded = true;
      cur.textContent = 'loaded: ' + data.path +
        '  (c1=' + data.chan1 + ' c2=' + data.chan2 + ' c3=' + data.chan3 +
        ' ' + data.device + ')';
    }} else {{
      modelLoaded = false;
      cur.textContent = '(none loaded — paste a path or wandb run id below)';
    }}
  }} catch (e) {{
    cur.textContent = 'model_info failed: ' + e;
  }}
}}

async function loadModel() {{
  const kind = document.getElementById('model-kind').value;
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
      body: JSON.stringify({{ kind, value }}),
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
document.getElementById('resize').addEventListener('click', () => {{
  const n = parseInt(document.getElementById('size').value, 10);
  if (!Number.isFinite(n) || n < 2 || n > 20) return;
  SIZE = n;
  grid = newGrid(SIZE);
  selected = null;
  renderGrid(); syncEditor();
  scheduleCompute();
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
  if (ev.key === 'Escape') {{
    if (activeHotbar !== null) setActiveHotbar(activeHotbar);
    return;
  }}
}});

grid = newGrid(SIZE);
renderGrid();
bindHotbar();
bindEditor();
refreshModelInfo();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    server_version = "FactoryBuilder/0.1"

    def log_message(self, format, *args):  # noqa: A002
        sys.stderr.write("[%s] %s\n" % (self.address_string(), format % args))

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/" or self.path.startswith("/?"):
            body = render_index(self.server.default_size).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/model_info":
            self._send_json(_model_info())
            return
        self.send_error(404)

    def do_POST(self):  # noqa: N802
        if self.path not in ("/graph", "/predict", "/load_model"):
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
            if self.path == "/graph":
                result = render_graph_png(payload["grid"])
            elif self.path == "/predict":
                result = _predict(payload["grid"])
            else:
                result = _swap_model(
                    kind=payload.get("kind", ""),
                    value=payload.get("value", ""),
                    project=self.server.wandb_project,
                    entity=self.server.wandb_entity,
                )
        except Exception as e:
            traceback.print_exc()
            result = {"error": f"{type(e).__name__}: {e}"}
        self._send_json(result)


def _resolve_wandb_checkpoint(
    run_spec: str, project: str, entity: Optional[str]
) -> str:
    """Resolve a W&B run id to a local checkpoint path. Downloads the
    run's first model-type artifact to /tmp/factorion-checkpoints.

    `run_spec` is either a bare id ("abc123") or a full path
    ("user/factorion/abc123"). Sets WANDB_MODE back to online for the
    duration of the call — the module's earlier setdefault to disabled
    is for the local HTTP server, not for fetching."""
    import wandb

    prev_mode = os.environ.pop("WANDB_MODE", None)
    prev_disabled = os.environ.pop("WANDB_DISABLED", None)
    try:
        api = wandb.Api()
        if run_spec.count("/") == 2:
            run = api.run(run_spec)
        else:
            ent = entity or api.default_entity
            run = api.run(f"{ent}/{project}/{run_spec}")
        dest = Path("/tmp/factorion-checkpoints") / run.id
        dest.mkdir(parents=True, exist_ok=True)

        model_arts = [a for a in run.logged_artifacts() if a.type == "model"]
        if not model_arts:
            raise RuntimeError(
                f"run {run.id} has no artifacts of type=model — "
                f"was it trained with --track and the artifact-upload code?"
            )
        # Newest first. Each `download()` returns the local dir holding
        # the artifact's files.
        art = max(model_arts, key=lambda a: a.created_at)
        local_dir = Path(art.download(root=str(dest / art.name.replace(":", "_"))))
        pt_files = sorted(local_dir.glob("*.pt"))
        if not pt_files:
            raise RuntimeError(f"artifact {art.name} contains no .pt file")
        path = str(pt_files[0])
        print(f"Resolved {run_spec} -> {art.name} -> {path}")
        return path
    finally:
        if prev_mode is not None:
            os.environ["WANDB_MODE"] = prev_mode
        if prev_disabled is not None:
            os.environ["WANDB_DISABLED"] = prev_disabled


def main(args: Args) -> None:
    if args.checkpoint and args.wandb_run:
        raise SystemExit("pass either --checkpoint or --wandb-run, not both")
    ckpt_path: Optional[str] = args.checkpoint
    if args.wandb_run:
        ckpt_path = _resolve_wandb_checkpoint(
            args.wandb_run, args.wandb_project, args.wandb_entity,
        )
    if ckpt_path:
        _load_checkpoint(ckpt_path)
    else:
        print("(no checkpoint — model prediction panel disabled)")

    httpd = HTTPServer(("127.0.0.1", args.port), Handler)
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
