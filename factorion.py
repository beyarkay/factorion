"""factorion: environment utilities (datatypes + helpers).

Plain Python module — was previously a marimo notebook, but the notebook
interface was never used outside of factorion.py itself, so the cell
wrappers have been stripped and identifiers are exported directly.
"""

import base64
import functools
import glob
import json
import os
import random
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
import torch
import tqdm
import wandb
from torch.distributions import Categorical

import factorion_rs

# The Rust extension is built out-of-band (`maturin develop`) and is NOT
# rebuilt automatically on import, so a stale wheel silently lacks functions
# added after it was last built. factorion.py and its callers (ppo, sft, and
# the web UI's auto-graph) reach into `factorion_rs` by name at call time, so a
# stale build imports cleanly — `py_items` below still resolves — then crashes
# much later with a cryptic `AttributeError: module 'factorion_rs' has no
# attribute '...'` deep inside a request handler (e.g. `py_build_graph` when the
# web UI builds the flow graph). Fail fast at import with an actionable message.
_REQUIRED_FACTORION_RS = (
    "simulate_throughput",
    "py_build_graph",
    "py_entity_tiles",
    "py_items",
    "py_recipes",
    "py_lesson_kinds",
    "render_factory",
)


def _assert_factorion_rs_current(module) -> None:
    """Raise an actionable ImportError if the installed factorion_rs is missing
    any function this module needs — the tell-tale sign of a stale Rust build."""
    missing = [name for name in _REQUIRED_FACTORION_RS if not hasattr(module, name)]
    if missing:
        raise ImportError(
            "factorion_rs is out of date (missing: "
            + ", ".join(missing)
            + "). Rebuild the Rust extension:\n"
            "  uv run maturin develop --release --manifest-path factorion_rs/Cargo.toml"
        )


_assert_factorion_rs_current(factorion_rs)

wandb.login()


class Html:
    """Minimal HTML wrapper, replaces ``marimo.Html`` for renderers that
    inspected ``.text`` or ``.data`` on the returned object.
    """

    def __init__(self, text):
        self.text = text
        self.data = text

    def __str__(self):
        return self.text

    def _repr_html_(self):
        return self.text


class Channel(Enum):
    # What entity occupies this tile?
    ENTITIES = 0
    # what direction is the entity facing?
    DIRECTION = 1
    # What recipe OR filter is set?
    ITEMS = 2
    # Undergrounds mechanics, see class Misc(Enum)
    MISC = 3
    # 1 if you can build there, 0 if you can't
    FOOTPRINT = 4


class Footprint(Enum):
    UNAVAILABLE = 0
    AVAILABLE = 1


class Misc(Enum):
    NONE = 0
    UNDERGROUND_DOWN = 1
    UNDERGROUND_UP = 2


class Dim(Enum):
    X = 0
    Y = 1


class Direction(Enum):
    """Directions the entity can be facing"""

    NONE = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    WEST = 4


# Unified Item dataclass. Everything in the data model is an Item;
# `is_placeable=True` items can be placed on the grid as entities.
# Width/height/flow are entity properties — non-placeable items
# default to 1×1 and 0 flow.
@dataclass
class Item:
    name: str
    value: int
    is_placeable: bool
    width: int
    height: int
    flow: float


# Items are defined in Rust (factorion_rs/src/types.rs::all_items)
# and exposed to Python via PyO3. To add or change an item, edit
# the Rust enum + all_items() and rebuild the wheel.
#
# The dict additionally contains a synthetic 0 → "empty" entry used
# purely as a tensor-decode sentinel for cells with no item set.
# `Item` itself has no Empty variant in Rust — absence of an item
# is `Option::None`. The Python sentinel is just for `items[v]`
# lookups when reading the world tensor's ENTITIES or ITEMS channel.
items = {
    0: Item(name="empty", value=0, is_placeable=False, width=1, height=1, flow=0.0),
}
for _val, _props in factorion_rs.py_items().items():
    items[_val] = Item(value=_val, **_props)

# `entities` is an alias for `items` for backwards compatibility.
# Post-unification, every grid-placeable entity is also an Item, so
# `entities[v]` returns the same dataclass as `items[v]`. The Rust
# binding hands us the canonical ordering: 1..5 = agent-placeable
# (TB, Inserter, AM1, UB, Splitter), 6..7 = env-spawned (Sink,
# Source — placed last so the policy head can exclude them via
# len(entities)-2), 8..12 = non-placeable items.
entities = items
# Backwards-compat alias for the now-unified Item dataclass.
Entity = Item


@dataclass
class Recipe:
    consumes: dict[str, float]
    produces: dict[str, float]
    crafting_time: float  # canonical wiki seconds per craft


# Recipes are defined in Rust (factorion_rs/src/types.rs::all_recipes)
# and exposed to Python via PyO3. To add a recipe, edit the Rust
# function and rebuild the wheel — Python sees it automatically.
recipes = {
    name: Recipe(
        consumes=dict(data["consumes"]),
        produces=dict(data["produces"]),
        crafting_time=data["crafting_time"],
    )
    for name, data in factorion_rs.py_recipes().items()
}


# LessonKind is owned by Rust (see py_lesson_kinds) and built here, so the kind
# set can't drift — the single-source-of-truth pattern `items`/`recipes` use.
LessonKind = Enum("LessonKind", factorion_rs.py_lesson_kinds())

# Lesson kinds the trainers (PPO/SFT) sample from. The ASSEMBLE_1IN_1OUT and
# ASSEMBLE_2IN_1OUT lessons are commented out below so they're never selected;
# their generators/tests remain intact, so re-enable by un-commenting.
SELECTABLE_LESSON_KINDS = [
    k
    for k in LessonKind
    if k.name
    not in (
        "ASSEMBLE_1IN_1OUT",
        "ASSEMBLE_2IN_1OUT",
    )
]


@dataclass(frozen=True)
class Factory:
    """A complete, valid factory ready to be turned into a training lesson.

    Build one via :func:`build_factory`, then pass it to
    :func:`blank_entities` to produce a (partial, solved, min_required)
    lesson pair. The same ``Factory`` can be blanked many times to
    produce multiple lessons at different difficulties without
    re-running the (expensive) layout search.

    Attributes:
        world_CWH: the complete factory layout, a (C, W, H) tensor.
        total_entities: number of removable entity *units* in the
            factory (multi-tile entities like splitters count as one
            unit). Used as the upper bound for blanking.
        protected_positions: tiles whose entity must never be blanked
            because it carries kind-specific structural information the
            agent cannot reconstruct (recipe channel on the assembler,
            splitter geometry). Source/sink are protected unconditionally
            inside :func:`_remove_entities` and need not appear here.
    """

    world_CWH: torch.Tensor
    total_entities: int
    protected_positions: frozenset = frozenset()


# Map Enum <--> grid deltas
DIR_TO_DELTA = {
    Direction.NORTH: (0, -1),
    Direction.EAST: (1, 0),
    Direction.SOUTH: (0, 1),
    Direction.WEST: (-1, 0),
}
# Reverse map for O(1) (dr, dc) → Direction lookups on the path-building hot
# path. Every delta is unique, so this is exactly the first-match the old
# linear scan over DIR_TO_DELTA.items() returned.
DELTA_TO_DIR = {delta: d for d, delta in DIR_TO_DELTA.items()}


def b64_to_dict(blueprint_string):
    decoded = base64.b64decode(
        blueprint_string.strip()[1:]
    )  # Skip the version byte
    json_data = zlib.decompress(decoded).decode("utf-8")
    return json.loads(json_data)


def dict2b64(dictionary):
    compressed = zlib.compress(json.dumps(dictionary).encode("utf-8"))
    b64_encoded = base64.b64encode(compressed).decode("utf-8")
    blueprint_string = "0" + b64_encoded  # Add version byte
    return blueprint_string


def str2item(s):
    assert s is not None, "input cannot be None"
    return next(
        (v for k, v in items.items() if v.name == s.replace("-", "_")), None
    )


@functools.cache  # pure fn of `s` (entities is import-time constant); see EXPERIMENT_LOG.md
def str2ent(s):
    if s is None:
        print(f"WARN: given string  is None")
        return None
    if s == "source":
        s = "stack_inserter"
    elif s == "sink":
        s = "bulk_inserter"

    for v in entities.values():
        if v.name == s.replace("-", "_"):
            return v
    # TODO I'm almost certianly going to regret hardcoding this
    if s == "electronic_circuit":
        return Entity(
            name="electronic_circuit",
            value=len(entities),
            is_placeable=False,
            width=1,
            height=1,
            flow=0.0,
        )
    print(f"WARN: unknown entity {s}")
    return None


# Hot-path entity/item ids, resolved once at import. `str2ent`/`str2item` do a
# linear scan over the items dict on every call; `_remove_entities` would
# otherwise re-resolve these per cell (965 resets × 121 cells in a 2-iter run).
_EMPTY_ENT_VAL = str2ent("empty").value
_SOURCE_ENT_VAL = str2ent("source").value
_SINK_ENT_VAL = str2ent("sink").value
_EMPTY_ITEM_VAL = str2item("empty").value


def _str2b64img(path, base_path="factorio-icons"):
    try:
        with open(f"{base_path}/{path}.png", "rb") as image_file:
            return "data:image/png;base64," + base64.b64encode(
                image_file.read()
            ).decode("utf-8")
    except:
        return ""


def ent_str2b64img(ent, base_path="factorio-icons"):
    return _str2b64img(str2ent(ent).name)


def item_str2b64img(item, base_path="factorio-icons"):
    return _str2b64img(str2item(item).name)


def new_world(width=8, height=8):
    channels = len(Channel)
    world = np.zeros((width, height, channels), dtype=int)
    world[:, :, Channel.ENTITIES.value] = str2ent("empty").value
    world[:, :, Channel.DIRECTION.value] = Direction.NONE.value
    world[:, :, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
    return world


def add_entity(
    world,
    proto_str,
    x,
    y,
    direction=Direction.NONE,
    recipe="empty",
    misc=Misc.NONE,
):
    proto = str2ent(proto_str)
    EMPTY = str2ent("empty")
    if proto is None:
        proto = EMPTY
    recipe_proto = str2ent(recipe)
    if recipe_proto is None:
        recipe_proto = EMPTY
    assert world[x, y, Channel.ENTITIES.value] == EMPTY.value, (
        f"Can't place {proto_str} at {x},{y} because {entities[world[x, y, Channel.ENTITIES.value]]} is there"
    )
    assert 0 <= x < len(world), f"{x=} is not in [0, {len(world)})"
    assert 0 <= y < len(world[0]), f"{y=} is not in [0, {len(world[0])})"

    world[x, y, Channel.ENTITIES.value] = proto.value
    world[x, y, Channel.DIRECTION.value] = direction.value


def world2html(world_WHC, highlights=None):
    """Render a world tensor as an HTML table.

    highlights: optional {(x, y): css_color_str} to colour specific cells
    (overrides the default unavailable-footprint shading). Useful for
    visualising diffs, model predictions, etc.
    """
    assert len(world_WHC.shape) == 3, (
        f"Expected 3 dimensions got {world_WHC.shape}"
    )
    assert world_WHC.shape[0] == world_WHC.shape[1], (
        f"Expected square got {world_WHC.shape}"
    )
    if type(world_WHC) is not np.ndarray:
        world_WHC = np.array(world_WHC)
    DIRECTION_ARROWS = {
        Direction.NONE.value: "",
        Direction.NORTH.value: "↑",
        Direction.EAST.value: "→",
        Direction.SOUTH.value: "↓",
        Direction.WEST.value: "←",
    }
    html = ["<table style='border-collapse: collapse;'>"]
    W, H, C = world_WHC.shape
    # Pre-compute which cells are secondary tiles of multi-tile entities,
    # mapping (x, y) → anchor entity name for ghost rendering
    ghost_at = {}
    visited_tiles = set()
    for y in range(H):
        for x in range(W):
            if (x, y) in visited_tiles:
                continue
            proto = entities[world_WHC[x, y, Channel.ENTITIES.value]]
            if proto.width == 1 and proto.height == 1:
                continue
            d_val = world_WHC[x, y, Channel.DIRECTION.value]
            tile_list = factorion_rs.py_entity_tiles(x, y, int(d_val), proto.width, proto.height)
            if tile_list is None:
                continue
            anchor = tuple(tile_list[0])
            for tx, ty in tile_list:
                visited_tiles.add((tx, ty))
                if (tx, ty) != anchor:
                    ghost_at[(tx, ty)] = proto.name

    # Belts share borders with adjacent connected belts so a chain like
    # >>> renders as a single outlined strip rather than three boxes.
    BELT_NAMES = {"transport_belt", "underground_belt"}

    def _belt_at(x, y):
        if not (0 <= x < W and 0 <= y < H):
            return False, 0
        p = entities[world_WHC[x, y, Channel.ENTITIES.value]]
        d = int(world_WHC[x, y, Channel.DIRECTION.value])
        return p.name in BELT_NAMES, d

    DIR_INT_TO_DELTA = {
        Direction.NORTH.value: (0, -1),
        Direction.EAST.value: (1, 0),
        Direction.SOUTH.value: (0, 1),
        Direction.WEST.value: (-1, 0),
    }
    DIR_OPPOSITE = {
        Direction.NORTH.value: Direction.SOUTH.value,
        Direction.SOUTH.value: Direction.NORTH.value,
        Direction.EAST.value: Direction.WEST.value,
        Direction.WEST.value: Direction.EAST.value,
    }
    BORDER_SIDES = [
        ("n", Direction.NORTH.value, (0, -1)),
        ("e", Direction.EAST.value, (1, 0)),
        ("s", Direction.SOUTH.value, (0, 1)),
        ("w", Direction.WEST.value, (-1, 0)),
    ]

    for y in range(H):
        html.append("<tr>")
        for x in range(W):
            proto = entities[world_WHC[x, y, Channel.ENTITIES.value]]
            item = items[world_WHC[x, y, Channel.ITEMS.value]]
            direction = world_WHC[x, y, Channel.DIRECTION.value]

            # Secondary tile of a multi-tile entity: render only the
            # ghost overlay, not the cell's own entity icon — otherwise
            # the secondary cell looks like a second copy of the entity.
            if (x, y) in ghost_at:
                entity_icon = ent_str2b64img("empty")
                ghost_icons = [ent_str2b64img(ghost_at[(x, y)])]
            else:
                entity_icon = ent_str2b64img(proto.name)
                ghost_icons = []

            item_icon = item_str2b64img(item.name)
            direction_arrow = DIRECTION_ARROWS.get(direction, "")
            if any(["Channel.MISC" in i for i in map(str, list(Channel))]):
                misc = Misc(world_WHC[x, y, Channel.MISC.value])
            else:
                misc = Misc.NONE
            underground_symbol = (
                "⭳"
                if misc == Misc.UNDERGROUND_DOWN
                else "⭱"
                if misc == Misc.UNDERGROUND_UP
                else ""
            )

            available = (
                world_WHC[x, y, Channel.FOOTPRINT.value]
                == Footprint.AVAILABLE.value
            )
            if highlights and (x, y) in highlights:
                bg_style = f"background: {highlights[(x, y)]};"
            elif not available:
                # Subtle grey cross-hatch (two stacked diagonals) marks
                # UNAVAILABLE tiles. Low opacity so it sits behind any
                # entity icons without competing visually.
                bg_style = (
                    "background:"
                    " repeating-linear-gradient(45deg,"
                    "  rgba(80,80,80,0.1), rgba(80,80,80,0.1) 2px,"
                    "  transparent 2px, transparent 8px),"
                    " repeating-linear-gradient(-45deg,"
                    "  rgba(80,80,80,0.1), rgba(80,80,80,0.1) 2px,"
                    "  transparent 2px, transparent 8px);"
                )
            else:
                bg_style = ""

            ghost_imgs = "\n".join(
                [
                    f"<img src='{ghost_icon}' style=' position: absolute; top: 10%;  left: 10%;  width: 60%; height: 60%; opacity: 20%;'>"
                    for ghost_icon in ghost_icons
                ]
            )

            hide_map = {"n": False, "e": False, "s": False, "w": False}
            if proto.name in BELT_NAMES:
                for side_lbl, side_dir, (sx, sy) in BORDER_SIDES:
                    nb_is_belt, nb_dir = _belt_at(x + sx, y + sy)
                    if not nb_is_belt:
                        continue
                    i_flow_out = direction == side_dir
                    n_flow_in = nb_dir == DIR_OPPOSITE[side_dir]
                    if i_flow_out or n_flow_in:
                        hide_map[side_lbl] = True
            hide_n = hide_map["n"]
            hide_e = hide_map["e"]
            hide_s = hide_map["s"]
            hide_w = hide_map["w"]

            def _border_css(prefix, color):
                return "; ".join(
                    f"border-{side}: 1px solid {'transparent' if hide else color}"
                    for side, hide in (("top", hide_n), ("right", hide_e), ("bottom", hide_s), ("left", hide_w))
                )
            td_border_css = _border_css("td", "black")
            div_border_css = _border_css("div", "grey")

            xy_str = f"{x},{y}"
            cell_content = f"""
           <div style='position: relative; width: 50px; height: 50px; {bg_style}; {div_border_css};'>
    <img src='{entity_icon}' style=' position: absolute; top: 10%;   left: 10%;  width: 60%; height: 60%; '>
    {ghost_imgs}
    <img src='{item_icon}' style=' position: absolute; bottom: 5%;  right: 5%;  width: 20%; height: 20%; '>
    <div style='position: absolute; top: 0; left: 0; font-size: 8px; opacity: 50%'>{xy_str}</div>
    <div style='position: absolute; bottom: 0; left: 0; font-size: 20px;'>{direction_arrow}</div>
    <div style=' position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);  font-size: 20px; font-weight: bold; color: white; '>{underground_symbol}</div>
</div>
            """
            html.append(
                f"<td style='{td_border_css}; padding: 0;'>{cell_content}</td>"
            )
        html.append("</tr>")
    html.append("</table>")
    return Html("".join(html))


# Factorio 2.0 blueprints encode direction as a 16-step enum (0=N, 4=E,
# 8=S, 12=W; diagonals at 2/6/10/14). The model uses Direction
# (NONE=0, N=1, E=2, S=3, W=4). Missing `direction` in a blueprint
# entity means "facing north" — that's how the encoder elides the
# default. Diagonals are not representable in the model and are
# rejected at decode time.
_BP_DIR_TO_MODEL = {
    0: Direction.NORTH,
    4: Direction.EAST,
    8: Direction.SOUTH,
    12: Direction.WEST,
}

# Arrow virtual signals in a constant-combinator marker encode the
# source/sink direction (see factorion-mod/server/blueprint.py).
_ARROW_TO_DIR = {
    "up-arrow": Direction.NORTH,
    "right-arrow": Direction.EAST,
    "down-arrow": Direction.SOUTH,
    "left-arrow": Direction.WEST,
}


def _parse_combinator_marker(e):
    """Decode a source/sink constant-combinator emitted by the
    factorion-mod encoder. The marker carries up to three filters in
    its first section: an item filter (the item that flows), a virtual
    signal (`signal-output` → source, `signal-input` → sink), and an
    arrow virtual signal (`up/right/down/left-arrow` → direction).
    Returns (role, item_name, direction) or None if the combinator
    isn't a recognized marker."""
    sections = (
        e.get("control_behavior", {}).get("sections", {}).get("sections", [])
    )
    if not sections:
        return None
    filters = sections[0].get("filters", [])
    role = None
    item_name = None
    direction = Direction.NORTH
    for f in filters:
        name = f.get("name")
        if name == "signal-output":
            role = "source"
        elif name == "signal-input":
            role = "sink"
        elif name in _ARROW_TO_DIR:
            direction = _ARROW_TO_DIR[name]
        elif f.get("type") != "virtual":
            item_name = name
    if role is None:
        return None
    return role, item_name, direction


def blueprint2world(bp):
    """Decode a Factorio 2.0 blueprint string into a (C, W, H) world
    tensor matching the encoder in factorion-mod/server/blueprint.py.

    Conventions:
      - Position in a Factorio blueprint is the entity's *center*; for
        an N×M entity rotated to occupy w×h tiles, top-left tile is
        floor(center - (w/2, h/2)).
      - Direction is a 16-step enum (0/4/8/12 = N/E/S/W). Diagonal
        values (2/6/10/14) are rejected.
      - Inserter direction in a blueprint points to the drop tile;
        flip by +8 (mod 16) to get the pickup-pointing model
        direction.
      - Source/sink markers are encoded as `constant-combinator`
        entities carrying signal-output/signal-input + arrow + item
        filters; they decode to `stack_inserter` / `bulk_inserter`
        with the corresponding ITEMS / DIRECTION channel values.
    """
    obj = b64_to_dict(bp)
    raw_entities = obj.get("blueprint", {}).get("entities", [])

    # First pass: resolve each blueprint entity to a placement
    # (entity_name, top_left_x, top_left_y, w, h, direction, item_value, misc).
    # Coordinates are in the blueprint's own frame; we translate to a
    # (0,0)-origin grid after the bounds are known.
    placements = []
    for e in raw_entities:
        name = e["name"]
        cx = e["position"]["x"]
        cy = e["position"]["y"]

        if name == "constant-combinator":
            parsed = _parse_combinator_marker(e)
            if parsed is None:
                print(
                    f"WARN: skipping unrecognized constant-combinator at "
                    f"({cx},{cy})"
                )
                continue
            role, item_name, model_dir = parsed
            ent_name = "stack_inserter" if role == "source" else "bulk_inserter"
            tlx = int(cx - 0.5)
            tly = int(cy - 0.5)
            item_val = 0
            if item_name is not None:
                item_meta = str2item(item_name)
                if item_meta is not None:
                    item_val = item_meta.value
            placements.append(
                (ent_name, tlx, tly, 1, 1, model_dir, item_val, Misc.NONE)
            )
            continue

        proto = str2ent(name)
        if proto is None:
            print(f"WARN: skipping unknown entity {name} at ({cx},{cy})")
            continue

        bp_dir = e.get("direction", 0)
        if "inserter" in proto.name:
            # Blueprint direction = drop tile; model direction = pickup.
            bp_dir = (bp_dir + 8) % 16
        if bp_dir not in _BP_DIR_TO_MODEL:
            print(
                f"WARN: skipping {name} with non-cardinal direction "
                f"{bp_dir} at ({cx},{cy})"
            )
            continue
        model_dir = _BP_DIR_TO_MODEL[bp_dir]

        # Multi-tile entities rotate footprint when facing E/W.
        w, h = proto.width, proto.height
        if model_dir in (Direction.EAST, Direction.WEST):
            w, h = h, w
        tlx = int(cx - w / 2)
        tly = int(cy - h / 2)

        item_val = 0
        if "assembling_machine" in proto.name:
            recipe = e.get("recipe")
            if recipe is not None:
                recipe_meta = str2item(recipe)
                if recipe_meta is not None:
                    item_val = recipe_meta.value

        misc = Misc.NONE
        if e.get("type") == "input":
            misc = Misc.UNDERGROUND_DOWN
        elif e.get("type") == "output":
            misc = Misc.UNDERGROUND_UP

        placements.append(
            (proto.name, tlx, tly, w, h, model_dir, item_val, misc)
        )

    if not placements:
        raise ValueError("blueprint contains no recognized entities")

    min_x = min(p[1] for p in placements)
    min_y = min(p[2] for p in placements)
    max_x = max(p[1] + p[3] for p in placements)
    max_y = max(p[2] + p[4] for p in placements)
    W = max_x - min_x
    H = max_y - min_y

    world = new_world(width=W, height=H)  # (W, H, C)
    for name, tlx, tly, w, h, model_dir, item_val, misc in placements:
        ent = str2ent(name)
        ox = tlx - min_x
        oy = tly - min_y
        for dx in range(w):
            for dy in range(h):
                x, y = ox + dx, oy + dy
                world[x, y, Channel.ENTITIES.value] = ent.value
                world[x, y, Channel.DIRECTION.value] = model_dir.value
                if item_val:
                    world[x, y, Channel.ITEMS.value] = item_val
                if misc != Misc.NONE:
                    world[x, y, Channel.MISC.value] = misc.value

    # Return (C, W, H) to match the encoder's input shape.
    return np.transpose(world, (2, 0, 1))


def read_blueprint_file(path):
    """Read a blueprint fixture, stripping full-line ``#`` comments and
    blank lines. Non-comment lines are concatenated so a long b64
    string can be hard-wrapped across multiple lines if convenient."""
    parts = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts.append(stripped)
    return "".join(parts)


def plot_flow_network(G):
    # Lazy imports: matplotlib + networkx are visualization-only (this function
    # and build_graph_nx). Importing them at module scope cost ~0.44s on every
    # `import factorion` — paid by ppo/sft training that never plots. Deferring
    # them here keeps those import-bound entry points (and the benchmarks) fast.
    import matplotlib.pyplot as plt
    import networkx as nx

    # Extract x, y coordinates from node names
    pos = {
        node: (int(x), -int(y))
        for node, (x, y) in ((n, n.split("@")[1].split(",")) for n in G.nodes)
    }
    plt.figure(
        figsize=(
            (len(G.nodes) ** 0.5) * 3,
            (len(G.nodes) ** 0.5) * 3,
        )
    )

    # Draw the graph
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        font_size=12,
        font_weight="bold",
    )

    plt.show()


def plot_loss_history(loss_history):
    # Create a figure
    fig = go.Figure()
    # Add traces for each key in loss_history
    for k in loss_history[-1].keys():
        fig.add_trace(
            go.Scatter(
                x=list(
                    range(len(loss_history))
                ),  # X-axis: range of iterations
                y=[
                    float(v[k]) if k in v else np.nan for v in loss_history
                ],  # Y-axis: values for the current key
                mode="lines",  # Plot as lines
                name=k,  # Legend label
                line=dict(width=0.5),  # Set line width
            )
        )
    # Update layout for better readability
    fig.update_layout(
        title="Loss History",  # Title of the plot
        xaxis_title="Iteration",  # X-axis label
        yaxis_title="Loss",  # Y-axis label
        width=800,  # Set figure width
        height=500,  # Set figure height
    )
    # Show the plot
    fig.show()


def normalise_world(world_T, og_world):
    assert torch.is_tensor(world_T), (
        f"world_T is {type(world_T)}, not a tensor"
    )
    assert torch.is_tensor(og_world), (
        f"og_world is {type(og_world)}, not a tensor"
    )
    assert len(world_T.shape) == 3, (
        f"Expected world_T to have 3 dimensions, but is of shape {world_T.shape}"
    )
    assert len(og_world.shape) == 3, (
        f"Expected og_world to have 3 dimensions, but is of shape {og_world.shape}"
    )
    assert world_T.shape[0] == world_T.shape[1], (
        f"Expected world_T to be square, but is of shape {world_T.shape}"
    )
    assert og_world.shape[0] == og_world.shape[1], (
        f"Expected og_world to be square, but is of shape {og_world.shape}"
    )

    empty_entity_value = str2ent("empty").value

    bulk_inserter_mask = (
        world_T[:, :, Channel.ENTITIES.value] == str2ent("bulk_inserter").value
    )
    world_T[:, :, Channel.ENTITIES.value][bulk_inserter_mask] = (
        empty_entity_value
    )

    stack_inserter_mask = (
        world_T[:, :, Channel.ENTITIES.value]
        == str2ent("stack_inserter").value
    )
    world_T[:, :, Channel.ENTITIES.value][stack_inserter_mask] = (
        empty_entity_value
    )

    green_circ_mask = (
        world_T[:, :, Channel.ENTITIES.value]
        == str2ent("electronic_circuit").value
    )
    world_T[:, :, Channel.ENTITIES.value][green_circ_mask] = empty_entity_value

    # Remove all transport belts without direction
    belt_mask = (
        world_T[:, :, Channel.ENTITIES.value]
        == str2ent("transport_belt").value
    )
    no_direction_mask = (
        world_T[:, :, Channel.DIRECTION.value] == Direction.NONE.value
    )
    world_T[:, :, Channel.ENTITIES.value][belt_mask & no_direction_mask] = (
        empty_entity_value
    )

    # Ensure the model can't just overwrite existing factories with a simpler thing.
    tworld = og_world.clone().detach().to(torch.int64)
    original_had_something = (
        tworld[:, :, Channel.ENTITIES.value] != empty_entity_value
    )
    for ch in list(Channel):
        replacements = tworld[:, :, ch.value][original_had_something]
        world_T[:, :, ch.value][original_had_something] = replacements
    return world_T


def get_min_belts(world_CWH):
    assert world_CWH.shape[1] == world_CWH.shape[2], (
        "Wrong shape: {world_CWH.shape}"
    )
    C, W, H = world_CWH.shape

    stack_inserter_id = str2ent("stack_inserter").value
    bulk_inserter_id = str2ent("bulk_inserter").value
    coords1 = torch.where(
        world_CWH[Channel.ENTITIES.value] == bulk_inserter_id
    )
    assert len(coords1[0]) == len(coords1[1]) == 1, (
        f"Expected 1 bulk inserter, found {coords1} in world {world_CWH}"
    )
    w1, h1 = coords1[0][0], coords1[1][0]

    coords2 = torch.where(
        world_CWH[Channel.ENTITIES.value] == stack_inserter_id
    )
    assert len(coords2[0]) == len(coords2[1]) == 1, (
        f"Expected 1 stack inserter, found {coords2} in world {world_CWH}"
    )
    w2, h2 = coords2[0][0], coords2[1][0]

    # we want an estimate for how many belts are required, so get the
    # coords of the transport belt tile closest to the source/sink
    w1 = torch.clamp(w1, 1, W - 2)
    h1 = torch.clamp(h1, 1, H - 2)
    w2 = torch.clamp(w2, 1, W - 2)
    h2 = torch.clamp(h2, 1, H - 2)

    manhat_dist = torch.abs(w1 - w2) + torch.abs(h1 - h2)
    min_belts = manhat_dist + 1
    return min_belts


def get_new_world(seed, n=6, min_belts=None, source_item=None, sink_item=None):
    stack_inserter_value = str2ent("stack_inserter").value
    bulk_inserter_value = str2ent("bulk_inserter").value
    empty_value = str2ent("empty").value

    if seed is not None:
        np.random.seed(seed)
    assert min_belts != [1], f"min_belts of [1] is sometimes unsatisfiable"
    if min_belts is None:
        min_belts = list(range(0, 64))
    w = new_world(width=n, height=n)
    boundary_tiles = []
    for i in range(n):
        for j in range(n):
            if i in (0, n - 1) and j in (0, n - 1):
                continue
            if i in (0, n - 1) or j in (0, n - 1):
                boundary_tiles.append((i, j))

    # Put a source and a sink on one of the boundaries
    source = boundary_tiles[np.random.choice(len(boundary_tiles))]
    w[source[0], source[1], Channel.ENTITIES.value] = stack_inserter_value
    # TODO not the most efficient, but it'll be okay for now
    limit = 1000
    while limit > 0:
        # Find random location for the sink
        sink = boundary_tiles[np.random.choice(len(boundary_tiles))]
        # Ensure the sink isn't on top of the source
        if source == sink:
            continue
        # Add the sink to the world
        w[sink[0], sink[1], Channel.ENTITIES.value] = bulk_inserter_value
        # Calculate the manhatten distance
        min_belt = get_min_belts(torch.tensor(w).permute(2, 0, 1))
        # If manhatten distance is acceptable and source != sink, we've got
        # our world
        if (source != sink) and (min_belt in min_belts):
            break
        # else, remove the sink from the world and try again
        w[sink[0], sink[1], Channel.ENTITIES.value] = empty_value
    assert limit > 0, "Infinite loop blocked"

    if source_item is not None:
        w[source[0], source[1], Channel.ITEMS.value] = source_item
    if sink_item is not None:
        w[sink[0], sink[1], Channel.ITEMS.value] = sink_item

    # Figure out the direction of the source + sink
    for x, y, is_source in [(*source, True), (*sink, False)]:
        if x == 0:
            w[x, y, Channel.DIRECTION.value] = (
                Direction.EAST if is_source else Direction.WEST
            ).value
        if x == n - 1:
            w[x, y, Channel.DIRECTION.value] = (
                Direction.WEST if is_source else Direction.EAST
            ).value
        if y == 0:
            w[x, y, Channel.DIRECTION.value] = (
                Direction.SOUTH if is_source else Direction.NORTH
            ).value
        if y == n - 1:
            w[x, y, Channel.DIRECTION.value] = (
                Direction.NORTH if is_source else Direction.SOUTH
            ).value

    return torch.tensor(w).to(torch.float)


def sample_world(probabilities):
    assert torch.is_tensor(probabilities), (
        f"probabilities is {type(probabilities)} not torch.Tensor"
    )
    distribution = Categorical(probs=probabilities)
    samples = distribution.sample()
    # make the directions fit the expected values
    d_direction = samples[:, :, Channel.DIRECTION.value]
    mask = d_direction > 0
    d_direction[mask] = d_direction[mask] * 4 - 4
    d_direction[~mask] = -1
    return samples


def eval_model(actor, critic, pars, num_evaluations=1_000, pbar=False):
    torch.manual_seed(42)
    evals = []
    iterator = torch.randint(0, 2**16 - 1, (num_evaluations,)).tolist()
    if pbar:
        iterator = tqdm.tqdm(iterator)
    for seed in iterator:
        original_world = get_new_world(seed, n=4)
        probabilities = actor(original_world)
        normalised_world = normalise_world(
            sample_world(probabilities), original_world
        )
        value = critic(normalised_world.to(torch.float))
        # FIXME(#161): still the old fixed /15.0 normalization. This is the
        # legacy RL-from-scratch eval (no current callers) and its random
        # get_new_world() inputs have no reference factory to normalize by, so
        # there's no per-factory max to use here — left as-is intentionally.
        throughput = torch.tensor(
            factorion_rs.simulate_throughput(
                normalised_world.to(torch.int64).numpy()
            )[0]
            / 15.0,
            dtype=value.dtype,
        )
        num_entities = (
            normalised_world[:, :, Channel.ENTITIES.value]
            != str2ent("empty").value
        ).sum()
        evals.append(
            {
                "seed": seed,
                "original_world": original_world,
                "normalised_world": normalised_world,
                "throughput": throughput,
                "num_entities": num_entities,
            }
        )

    avg_throughput = sum([eval["throughput"] for eval in evals]) / len(evals)
    avg_num_entities = sum([eval["num_entities"] for eval in evals]) / len(
        evals
    )

    return evals, avg_throughput, float(avg_num_entities)


def build_graph_nx(world_WHC):
    """Build the factory connection graph as a networkx ``DiGraph``.

    The Rust engine (``factorion_rs.py_build_graph``) is the single source of
    truth for entity connectivity; this thin wrapper rebuilds its
    ``(node_labels, edges)`` output into a ``networkx`` graph for connectivity
    queries and drawing/layout. Nodes are labelled
    ``f"{entity_name}\\n@{x},{y}"`` and edges follow the engine's
    entity-connection rules.

    Accepts the same ``(W, H, C)`` world — a torch tensor or numpy array — that
    ``factorion_rs.simulate_throughput`` takes.
    """
    import networkx as nx  # lazy: networkx is visualization-only (see top imports)

    arr = world_WHC.numpy() if hasattr(world_WHC, "numpy") else np.asarray(world_WHC)
    nodes, edges = factorion_rs.py_build_graph(arr.astype(np.int64))
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from((nodes[i], nodes[j]) for i, j in edges)
    return G


def _remove_entities(
    world_CWH, num_missing_entities, protected_positions=None
):
    """Remove entities from a completed lesson, respecting multi-tile units.

    protected_positions: optional set of (x, y) the lesson considers
    structurally required. Any entity-group containing one of these tiles
    is excluded from the removable pool, so the agent always sees those
    entities in its input. Source/sink are protected unconditionally via
    the entity-id `skip` set below.

    FOOTPRINT is left as new_world() set it (all-AVAILABLE). A previous
    version marked every empty cell in the completed layout as
    UNAVAILABLE, but that effectively gave away the answer: the
    AVAILABLE cells were exactly the optimal placement set, so the
    agent's task collapsed to "place anything in any AVAILABLE-and-empty
    cell." Specific lessons can override FOOTPRINT to model bounded
    build regions (e.g. hazard-concrete imports), but the *default*
    must not encode the solution.

    Returns min_entities_required (number of entity units removed).
    For multi-tile entities (e.g. splitters), all tiles are removed together
    as a single unit. ``num_missing_entities=inf`` removes every removable
    unit (a full blank down to the protected source/sink).
    """
    protected_positions = protected_positions or set()

    C, W, H = world_CWH.shape
    skip = {_SOURCE_ENT_VAL, _SINK_ENT_VAL, _EMPTY_ENT_VAL}

    # Read the entity/direction channels once as plain int numpy arrays. The
    # world tensor is on the CPU, so reading per cell via torch `.item()` (the
    # old hot path) paid per-op dispatch overhead ~250×/reset; numpy scalar
    # indexing is far cheaper. Values, iteration order and the `random.sample`
    # draw below are unchanged, so the blanked factory is byte-identical.
    ent_chan = world_CWH[Channel.ENTITIES.value].to(torch.int64).numpy()
    dir_chan = world_CWH[Channel.DIRECTION.value].to(torch.int64).numpy()

    # Pass 1: identify which cells are secondary tiles of multi-tile entities.
    # Map secondary → anchor so we skip them in pass 2.
    secondary_tiles = set()
    for x in range(W):
        for y in range(H):
            if (x, y) in secondary_tiles:
                continue
            ent_val = int(ent_chan[x, y])
            if ent_val in skip:
                continue
            ent = entities[ent_val]
            if ent.width == 1 and ent.height == 1:
                continue
            d_val = int(dir_chan[x, y])
            tiles_list = factorion_rs.py_entity_tiles(x, y, d_val, ent.width, ent.height)
            if tiles_list is not None:
                for tx, ty in tiles_list:
                    if (tx, ty) != (x, y):
                        secondary_tiles.add((tx, ty))

    # Pass 2: build entity groups from anchors only
    entity_groups = []
    for x in range(W):
        for y in range(H):
            if (x, y) in secondary_tiles:
                continue
            ent_val = int(ent_chan[x, y])
            if ent_val in skip:
                continue
            ent = entities[ent_val]
            if ent.width == 1 and ent.height == 1:
                group = {(x, y)}
            else:
                d_val = int(dir_chan[x, y])
                tiles_list = factorion_rs.py_entity_tiles(x, y, d_val, ent.width, ent.height)
                group = set(map(tuple, tiles_list)) if tiles_list is not None else {(x, y)}
            if group & protected_positions:
                continue
            entity_groups.append(group)

    num_samples = min(num_missing_entities, len(entity_groups))
    if num_samples == 0:
        return 0

    sampled_groups = random.sample(entity_groups, num_samples)
    for group in sampled_groups:
        for x, y in group:
            world_CWH[Channel.ENTITIES.value, x, y] = _EMPTY_ENT_VAL
            world_CWH[Channel.DIRECTION.value, x, y] = Direction.NONE.value
            world_CWH[Channel.ITEMS.value, x, y] = _EMPTY_ITEM_VAL
            world_CWH[Channel.MISC.value, x, y] = Misc.NONE.value

    return num_samples


def build_factory(
    size: int = 12,
    kind: LessonKind = LessonKind.MOVE_ONE_ITEM,
    *,
    seed: Optional[int] = None,
    random_item: bool = True,
    max_entities: float = float("inf"),
) -> Optional[Factory]:
    """Build a complete, valid factory of the given lesson ``kind``.

    The layout is generated in Rust (``factorion_rs.build_factory``); this thin
    wrapper seeds the RNG and adapts the result into a :class:`Factory`. Returns
    ``None`` when rejection sampling can't satisfy the constraints (tight grid /
    complex kind). Determinism: same ``(size, kind, seed)`` → same factory.
    """
    # Seed the global RNGs so a chained blank_entities(seed=None) stays
    # deterministic in `seed`; with no seed, draw one from the current stream so
    # the result still tracks RNG state.
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        rs_seed = seed
    else:
        rs_seed = random.randrange(2**63)

    result = factorion_rs.build_factory(
        size, kind.value, rs_seed, random_item, max_entities
    )
    if result is None:
        return None
    world_WHC, total_entities, protected = result
    # Rust returns (W, H, C); the Factory carries (C, W, H).
    world_CWH = torch.from_numpy(
        np.ascontiguousarray(np.transpose(world_WHC, (2, 0, 1)))
    )
    return Factory(
        world_CWH=world_CWH,
        total_entities=total_entities,
        protected_positions=frozenset(map(tuple, protected)),
    )


def render_factory(world: "Factory | torch.Tensor | np.ndarray") -> str:
    """Render a factory into the two-character ASCII grid format (the same
    format the textual test fixtures use).

    Accepts a :class:`Factory` or a ``(C, W, H)`` world tensor/array (the layout
    :attr:`Factory.world_CWH` carries). Returns the multi-line grid string,
    where each tile is two characters — an entity char (``b`` belt, ``i``
    inserter, ``a`` assembler, ``Y`` splitter, ``d``/``u`` underground down/up,
    ``S`` source, ``K`` sink) plus a direction marker (``^>v<``), or ``..`` for
    an empty tile. Item/recipe bindings are not shown (they live in the tensor,
    not the grid)."""
    world_CWH = world.world_CWH if isinstance(world, Factory) else world
    world_CWH = np.asarray(world_CWH)
    # Rust expects (W, H, C); the Factory carries (C, W, H).
    world_WHC = np.ascontiguousarray(
        np.transpose(world_CWH, (1, 2, 0)).astype(np.int64)
    )
    return factorion_rs.render_factory(world_WHC)


def blank_entities(
    factory: Factory,
    num_missing_entities: float = float("inf"),
    *,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """Blank up to ``num_missing_entities`` entity units from a factory.

    Returns ``(partial_world_CWH, min_entities_required)``:

    - ``partial_world_CWH``: clone of the factory with some entities removed,
      ready to be fed to the agent as the lesson input.
    - ``min_entities_required``: actual number of entity units removed
      (may be less than ``num_missing_entities`` if the factory has fewer
      removable units, e.g. when many entities are protected).

    The complete (solved) layout stays accessible as ``factory.world_CWH``;
    it isn't returned here to avoid aliasing the factory's tensor into
    caller code.

    Blanking respects multi-tile entities (splitters, assemblers) as
    single removable units, and never removes ``factory.protected_positions``,
    sources, or sinks.

    If ``seed`` is provided, the global RNG is reseeded before blanking
    so the (partial, min_required) pair is deterministic in
    ``(factory, num_missing_entities, seed)``. If ``seed`` is ``None``,
    blanking consumes from the current RNG stream — handy when chained
    directly after a :func:`build_factory` call that already seeded.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    partial = factory.world_CWH.clone()
    min_entities_required = _remove_entities(
        partial,
        num_missing_entities,
        protected_positions=set(factory.protected_positions),
    )
    return partial, min_entities_required


def _bfs_shortest(grid_h, grid_w, start, end, blocked):
    """BFS from start to end, avoiding blocked cells.

    Returns (dist_map, parents_map) where parents_map tracks ALL
    equal-length parents for enumerating all shortest paths.
    Returns (None, None) if no path exists.
    """
    def in_bounds(cell):
        r, c = cell
        return 0 <= r < grid_h and 0 <= c < grid_w

    if not in_bounds(start) or not in_bounds(end):
        return None, None
    if start in blocked or end in blocked:
        return None, None

    deltas = list(DIR_TO_DELTA.values())
    dist = {start: 0}
    parents = defaultdict(list)
    q = deque([start])

    # Hot loop: ~1.5M iterations across a build. Inline the bounds check, cache
    # the current cell's distance (was re-looked-up 4×/expansion), and build each
    # neighbour tuple once. Neighbours are still visited in `deltas` order and
    # appended to `parents` in the same order, so dist/parents — and therefore
    # the enumerated paths and the random.shuffle that consumes them — stay
    # byte-identical to the original.
    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        nd = dist[(r, c)] + 1
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < grid_h and 0 <= nc < grid_w):
                continue
            ncell = (nr, nc)
            if ncell in blocked:
                continue
            cur = dist.get(ncell)
            if cur is None:
                dist[ncell] = nd
                parents[ncell].append((r, c))
                q.append(ncell)
            elif cur == nd:
                parents[ncell].append((r, c))

    if end not in dist:
        return None, None
    return dist, parents

def _path_to_belts(path, end_dir):
    """Convert a list of (x, y) cells into belt placements with directions."""
    belts: List[Tuple[int, int, Direction]] = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        d = DELTA_TO_DIR.get((r2 - r1, c2 - c1))
        if d is not None:
            belts.append((r1, c1, d))
    belts.append((path[-1][0], path[-1][1], end_dir))
    return belts

def find_belt_path(
    grid_h: int,
    grid_w: int,
    start: Tuple[int, int],
    end: Tuple[int, int],
    end_dir: Direction,
    blocked: set,
) -> Optional[List[Tuple[int, int, Direction]]]:
    """Find a single shortest belt path from start to end, avoiding blocked cells.

    Returns a list of (x, y, Direction) tuples for belt placements,
    or None if no path exists. The last belt gets end_dir as its direction.
    """
    dist, parents = _bfs_shortest(grid_h, grid_w, start, end, blocked)
    if dist is None:
        return None

    # Reconstruct single path via parent chain
    path = []
    cell = end
    while cell != start:
        path.append(cell)
        cell = parents[cell][0]  # Take first parent for single path
    path.append(start)
    path.reverse()

    return _path_to_belts(path, end_dir)

def find_belt_paths_with_source_sink_orient(
    entities: torch.Tensor,
    directions: torch.Tensor,
    source_value: int = str2ent("source").value,
    sink_value: int = str2ent("sink").value,
) -> List[List[Tuple[int, int, Direction]]]:
    """Find all shortest belt-placement paths from source to sink.

    Returns a list of paths, each being a list of (row, col, Direction)
    tuples. If no valid path exists, returns [].
    """
    if (
        entities.ndim != 2
        or directions.ndim != 2
        or entities.shape != directions.shape
    ):
        raise ValueError(
            "entities and directions must be 2D tensors of the same shape"
        )
    H, W = entities.shape

    src_pos = (entities == source_value).nonzero(as_tuple=False)
    sink_pos = (entities == sink_value).nonzero(as_tuple=False)
    if len(src_pos) != 1 or len(sink_pos) != 1:
        raise ValueError("must have exactly one source and one sink")
    src = tuple(src_pos[0].tolist())
    sink = tuple(sink_pos[0].tolist())

    src_dir = Direction(directions[src].item())
    sink_dir = Direction(directions[sink].item())
    if src_dir == Direction.NONE or sink_dir == Direction.NONE:
        return []

    dr_s, dc_s = DIR_TO_DELTA[src_dir]
    start = (src[0] + dr_s, src[1] + dc_s)
    dr_k, dc_k = DIR_TO_DELTA[sink_dir]
    end = (sink[0] - dr_k, sink[1] - dc_k)

    if start == src or start == sink or end == src or end == sink:
        return []

    blocked = {src, sink}
    dist, parents = _bfs_shortest(H, W, start, end, blocked)
    if dist is None:
        return []

    # Backtrack all shortest paths
    all_paths: List[List[Tuple[int, int, Direction]]] = []

    def backtrack(cell, rev_path):
        if cell == start:
            path = [start] + list(reversed(rev_path))
            all_paths.append(_path_to_belts(path, sink_dir))
            return
        for p in parents[cell]:
            backtrack(p, rev_path + [cell])

    backtrack(end, [])
    return all_paths

