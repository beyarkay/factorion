"""factorion: environment utilities (datatypes + helpers).

Plain Python module — was previously a marimo notebook, but the notebook
interface was never used outside of factorion.py itself, so the cell
wrappers have been stripped and identifiers are exported directly.
"""

import base64
import glob
import json
import os
import random
import zlib
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import torch
import tqdm
import wandb
from torch.distributions import Categorical

import factorion_rs

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


class LessonKind(Enum):
    MOVE_ONE_ITEM = 0
    SPLITTER_SPLIT = 3
    SPLITTER_MERGE = 4
    ASSEMBLE_1IN_1OUT = 5
    MOVE_VIA_UG_BELT = 6
    ASSEMBLE_2IN_1OUT = 7
    FROM_BLUEPRINT = 8


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


# Directory of curated blueprint strings the FROM_BLUEPRINT lesson
# samples from. Both training (build_factory) and the blueprint-decode
# tests read from here.
LESSON_BLUEPRINT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "lesson_blueprints"
)


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


# Horizontal flip swaps E↔W; vertical flip swaps N↔S. The full
# Direction enum has NONE=0 plus NORTH=1, EAST=2, SOUTH=3, WEST=4 —
# entries omitted from the maps stay put (NONE is unchanged by either
# flip, N/S are unchanged by horizontal flip, E/W by vertical).
_DIR_FLIP_H = {Direction.EAST.value: Direction.WEST.value,
               Direction.WEST.value: Direction.EAST.value}
_DIR_FLIP_V = {Direction.NORTH.value: Direction.SOUTH.value,
               Direction.SOUTH.value: Direction.NORTH.value}


def _flip_world(world_CWH, axis):
    """Mirror a (C, W, H) world tensor along the W axis (axis=1) or H
    axis (axis=2), remapping direction values so flow directions are
    preserved. Works for any tensor with the canonical channel layout.
    """
    assert axis in (1, 2)
    flipped = torch.flip(world_CWH, dims=[axis])
    dir_ch = flipped[Channel.DIRECTION.value].clone()
    remap = _DIR_FLIP_H if axis == 1 else _DIR_FLIP_V
    for old, new in remap.items():
        dir_ch[flipped[Channel.DIRECTION.value] == old] = new
    flipped[Channel.DIRECTION.value] = dir_ch
    return flipped


def _is_gears_factory(world_CWH):
    """True iff every source carries ``iron_plate``, every sink
    carries ``iron_gear_wheel``, and every assembler tile carries
    ``iron_gear_wheel`` in the ITEMS channel. Blueprints that match
    this pattern are eligible for the 1-in-1-out recipe substitution
    augmentation; everything else is left untouched."""
    ent = world_CWH[Channel.ENTITIES.value]
    itm = world_CWH[Channel.ITEMS.value]
    src_id = str2ent("stack_inserter").value
    snk_id = str2ent("bulk_inserter").value
    asm_id = str2ent("assembling_machine_1").value
    plate_id = str2item("iron_plate").value
    gear_id = str2item("iron_gear_wheel").value

    src_mask = ent == src_id
    snk_mask = ent == snk_id
    asm_mask = ent == asm_id

    if not (src_mask.any() and snk_mask.any() and asm_mask.any()):
        return False
    return bool(
        (itm[src_mask] == plate_id).all()
        and (itm[snk_mask] == gear_id).all()
        and (itm[asm_mask] == gear_id).all()
    )


def _substitute_gears_recipe(world_CWH):
    """Replace the ``iron_plate → iron_gear_wheel`` ITEMS triple with
    a randomly chosen 1-in-1-out recipe. Source markers get the new
    input item; sinks and assembler tiles get the new produced item.
    Caller must have already verified eligibility via
    :func:`_is_gears_factory`."""
    one_in_one_out = [
        (name, r) for name, r in recipes.items()
        if len(r.consumes) == 1 and len(r.produces) == 1
    ]
    if not one_in_one_out:
        return world_CWH
    _, recipe = random.choice(one_in_one_out)
    new_input = next(iter(recipe.consumes.keys()))
    new_output = next(iter(recipe.produces.keys()))
    new_input_id = str2item(new_input).value
    new_output_id = str2item(new_output).value

    plate_id = str2item("iron_plate").value
    gear_id = str2item("iron_gear_wheel").value

    out = world_CWH.clone()
    itm = out[Channel.ITEMS.value]
    plate_mask = itm == plate_id
    gear_mask = itm == gear_id
    itm[plate_mask] = new_input_id
    itm[gear_mask] = new_output_id
    return out


def _count_removable_entity_units(world_CWH):
    """Count entity units that would be candidates for blanking by
    :func:`_remove_entities` (sources/sinks/empty excluded; multi-tile
    entities counted once at their anchor tile). The world is iterated
    in (x outer, y inner) order, matching the convention that the
    anchor tile is the first occupied tile encountered in that sweep.
    """
    skip = {
        str2ent("source").value,
        str2ent("sink").value,
        str2ent("empty").value,
    }
    secondary = set()
    count = 0
    _, W, H = world_CWH.shape
    ent_ch = world_CWH[Channel.ENTITIES.value]
    dir_ch = world_CWH[Channel.DIRECTION.value]
    for x in range(W):
        for y in range(H):
            if (x, y) in secondary:
                continue
            ev = int(ent_ch[x, y])
            if ev in skip:
                continue
            ent = entities[ev]
            count += 1
            if ent.width > 1 or ent.height > 1:
                d = int(dir_ch[x, y])
                tiles_list = factorion_rs.py_entity_tiles(
                    x, y, d, ent.width, ent.height
                )
                if tiles_list is not None:
                    for tx, ty in tiles_list:
                        if (tx, ty) != (x, y):
                            secondary.add((tx, ty))
    return count


_OPPOSITE_DIR = {
    Direction.NORTH: Direction.SOUTH,
    Direction.SOUTH: Direction.NORTH,
    Direction.EAST: Direction.WEST,
    Direction.WEST: Direction.EAST,
}


def _extend_belt_chains(world_CWH):
    """Look for the pattern ``empty+ <source|sink> belt`` and, for each
    match, push the marker backward into the empty run while filling
    the vacated tiles with belts in the chain's flow direction.

    Per-marker rules:
      - The scan direction is fixed by the marker: a source pushes
        items along its ``direction``, so the belt sits in that
        direction and empties extend the opposite way. A sink receives
        items in its ``direction``, so the feeder belt sits in the
        opposite direction and empties extend on the sink's output
        side.
      - The adjacent belt must face the marker's flow direction
        (coherence check); mis-aligned chains are left alone since
        extending them would just compound a broken flow.
      - The run of empties behind the marker is counted up to the
        first non-empty cell (or the world edge); call its length
        ``K``. We sample ``J ∼ Uniform{0, …, K}`` and apply the
        extension only when ``J > 0`` — so some matches turn into
        no-ops (giving augmentation variety) and the maximum is never
        guaranteed.

    Multiple markers are processed in shuffled order. Each marker is
    handled at most once; extensions only relocate the marker that
    triggered them and fill cells behind it, so other markers'
    positions stay valid.
    """
    out = world_CWH.clone()
    _, W, H = out.shape

    src_id = str2ent("stack_inserter").value
    snk_id = str2ent("bulk_inserter").value
    tb_id = str2ent("transport_belt").value
    empty_id = str2ent("empty").value
    marker_ids = {src_id, snk_id}

    ent_ch = out[Channel.ENTITIES.value]
    dir_ch = out[Channel.DIRECTION.value]
    item_ch = out[Channel.ITEMS.value]

    markers = [
        (x, y)
        for x in range(W)
        for y in range(H)
        if int(ent_ch[x, y]) in marker_ids
    ]
    random.shuffle(markers)

    for mx, my in markers:
        marker_id_val = int(ent_ch[mx, my])
        if marker_id_val not in marker_ids:
            continue  # another extension already cleared this tile

        marker_dir_val = int(dir_ch[mx, my])
        try:
            marker_dir_enum = Direction(marker_dir_val)
        except ValueError:
            continue
        if marker_dir_enum == Direction.NONE:
            continue

        if marker_id_val == src_id:
            scan_dir = marker_dir_enum
        else:
            scan_dir = _OPPOSITE_DIR[marker_dir_enum]
        dx, dy = DIR_TO_DELTA[scan_dir]

        belt_x, belt_y = mx + dx, my + dy
        if not (0 <= belt_x < W and 0 <= belt_y < H):
            continue
        if int(ent_ch[belt_x, belt_y]) != tb_id:
            continue
        if int(dir_ch[belt_x, belt_y]) != marker_dir_val:
            # Belt isn't aligned with the marker's flow direction; the
            # chain bends or breaks here, so don't extend it.
            continue

        K = 0
        ex, ey = mx - dx, my - dy
        while 0 <= ex < W and 0 <= ey < H:
            if int(ent_ch[ex, ey]) != empty_id:
                break
            K += 1
            ex -= dx
            ey -= dy
        if K == 0:
            continue

        J = random.randint(0, K)
        if J == 0:
            continue

        marker_item = int(item_ch[mx, my])
        new_mx = mx - dx * J
        new_my = my - dy * J

        ent_ch[mx, my] = empty_id
        dir_ch[mx, my] = Direction.NONE.value
        item_ch[mx, my] = 0

        ent_ch[new_mx, new_my] = marker_id_val
        dir_ch[new_mx, new_my] = marker_dir_val
        item_ch[new_mx, new_my] = marker_item

        # Fill the J vacated tiles between the new marker position and
        # the (untouched) existing belt. Cells are
        # (new_mx + j·dx, new_my + j·dy) for j = 1..J, with the j=J
        # cell landing on the old marker tile.
        for j in range(1, J + 1):
            bx = new_mx + dx * j
            by = new_my + dy * j
            ent_ch[bx, by] = tb_id
            dir_ch[bx, by] = marker_dir_val

    return out


def plot_flow_network(G):
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
    skip = {str2ent("source").value, str2ent("sink").value, str2ent("empty").value}

    # Pass 1: identify which cells are secondary tiles of multi-tile entities.
    # Map secondary → anchor so we skip them in pass 2.
    secondary_tiles = set()
    for x in range(W):
        for y in range(H):
            if (x, y) in secondary_tiles:
                continue
            ent_val = world_CWH[Channel.ENTITIES.value, x, y].item()
            if ent_val in skip:
                continue
            ent = entities[ent_val]
            if ent.width == 1 and ent.height == 1:
                continue
            d_val = world_CWH[Channel.DIRECTION.value, x, y].item()
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
            ent_val = world_CWH[Channel.ENTITIES.value, x, y].item()
            if ent_val in skip:
                continue
            ent = entities[ent_val]
            if ent.width == 1 and ent.height == 1:
                group = {(x, y)}
            else:
                d_val = world_CWH[Channel.DIRECTION.value, x, y].item()
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
            world_CWH[Channel.ENTITIES.value, x, y] = str2ent("empty").value
            world_CWH[Channel.DIRECTION.value, x, y] = Direction.NONE.value
            world_CWH[Channel.ITEMS.value, x, y] = str2item("empty").value
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
    """Construct a single complete, valid factory of the given lesson kind.

    Layout search is randomized rejection-sampling; for tight grids or
    complex kinds (e.g. ``SPLITTER_SPLIT`` on ``size=5``) the random
    placements may never satisfy the constraints. In that case this
    function returns ``None`` rather than raising. **Bad inputs**
    (unknown ``kind``, grid too small to physically fit the lesson's
    fixed entities, missing recipe data) still raise — those are
    misconfiguration, not bad luck.

    The recommended retry idiom is to advance the seed and try again::

        factory = None
        attempt_seed = seed
        while factory is None:
            attempt_seed += 1
            factory = build_factory(size, kind, seed=attempt_seed)

    Determinism: same ``(size, kind, seed)`` → same ``Factory``. The
    function seeds the global ``random`` / ``numpy`` / ``torch`` RNGs
    when ``seed`` is provided, and leaves RNG state advanced past the
    layout search on return (so a subsequent :func:`blank_entities`
    call without a ``seed`` continues from the same stream and is
    likewise deterministic).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
    total_entities: int = 0
    protected_positions: frozenset = frozenset()
    world_CWH = torch.tensor(new_world(width=size, height=size)).permute(
        2, 0, 1
    )
    C, W, H = world_CWH.shape
    # No idea why, but there doing kind == LessonKind.MOVE_ONE_ITEM doesn't evaluate to true...
    if kind.value == LessonKind.MOVE_ONE_ITEM.value:
        # Choose a random source/sink
        original_count = max(500, size * size * 4)
        count = original_count
        while count > 0:
            count -= 1
            pos1 = torch.randint(0, H * W, (1,))
            pos2 = torch.randint(0, H * W, (1,))
            if pos1 == pos2:
                # restart the loop until we find non-equal source/sink
                continue

            source_WH = divmod(pos1.item(), W)
            sink_WH = divmod(pos2.item(), W)
            source_dir = random.choice(
                [d for d in Direction if d != Direction.NONE]
            )
            sink_dir = random.choice(
                [d for d in Direction if d != Direction.NONE]
            )

            if random_item:
                item_value = random.choice(
                    [v.value for k, v in items.items() if v.name != "empty"]
                )
            else:
                item_value = str2item("electronic_circuit").value

            world_CWH[Channel.ENTITIES.value, source_WH[0], source_WH[1]] = (
                str2ent("source").value
            )
            world_CWH[Channel.ENTITIES.value, sink_WH[0], sink_WH[1]] = (
                str2ent("sink").value
            )

            world_CWH[Channel.ITEMS.value, source_WH[0], source_WH[1]] = (
                item_value
            )
            world_CWH[Channel.ITEMS.value, sink_WH[0], sink_WH[1]] = item_value

            world_CWH[Channel.DIRECTION.value, source_WH[0], source_WH[1]] = (
                source_dir.value
            )
            world_CWH[Channel.DIRECTION.value, sink_WH[0], sink_WH[1]] = (
                sink_dir.value
            )

            paths = find_belt_paths_with_source_sink_orient(
                entities=world_CWH[Channel.ENTITIES.value],
                directions=world_CWH[Channel.DIRECTION.value],
            )
            # Remove all paths that would require placing too many entities
            paths = list(filter(lambda p: len(p) <= max_entities, paths))

            if len(paths) == 0:
                world_CWH = torch.tensor(
                    new_world(width=size, height=size)
                ).permute(2, 0, 1)
                # Restart the loop until we get a source+sink that can be connected
                continue

            # Choose a valid path that ACTUALLY carries throughput.
            # find_belt_paths_with_source_sink_orient can return paths that are
            # geometrically connected but functionally broken (carry zero items)
            # when the random source/sink placement is degenerate — too close,
            # adjacent, or oriented so the belt can't deliver. Laying one of those
            # blindly produced ~5-13% zero-throughput MOVE_ONE_ITEM factories
            # (size-dependent) — broken training targets no policy can complete.
            # So try paths in random order and keep the first that simulates to
            # positive throughput; if none do, the source/sink is degenerate —
            # resample it.
            random.shuffle(paths)
            chosen_path = None
            for candidate in paths:
                trial = world_CWH.clone()
                for x, y, d in candidate:
                    trial[Channel.ENTITIES.value, x, y] = str2ent(
                        "transport_belt"
                    ).value
                    trial[Channel.DIRECTION.value, x, y] = d.value
                tp, _ = factorion_rs.simulate_throughput(
                    trial.permute(1, 2, 0).to(torch.int64).numpy()
                )
                if tp > 0:
                    chosen_path = candidate
                    world_CWH = trial
                    break

            if chosen_path is None:
                # Every candidate path is functionally broken → degenerate
                # source/sink. Reject this factory and resample.
                world_CWH = torch.tensor(
                    new_world(width=size, height=size)
                ).permute(2, 0, 1)
                continue

            total_entities = len(chosen_path)
            break
        if count == 0:
            return None
    elif kind.value == LessonKind.SPLITTER_SPLIT.value:
        # 1 source → belts → splitter → 2x(belts → sink)
        original_count = max(500, size * size * 10)
        count = original_count

        if random_item:
            item_value = random.choice(
                [v.value for k, v in items.items() if v.name != "empty"]
            )
        else:
            item_value = str2item("electronic_circuit").value

        splitter_ent = str2ent("splitter")

        while count > 0:
            count -= 1
            world_CWH = torch.tensor(new_world(width=size, height=size)).permute(2, 0, 1)
            C, W, H = world_CWH.shape

            dirs = [d for d in Direction if d != Direction.NONE]
            splitter_dir = random.choice(dirs)

            # Pick splitter anchor position; both tiles must fit
            tiles = None
            for _ in range(20):
                sx = random.randint(0, W - 1)
                sy = random.randint(0, H - 1)
                tiles = factorion_rs.py_entity_tiles(sx, sy, splitter_dir.value, splitter_ent.width, splitter_ent.height)
                if tiles is not None and all(0 <= tx < W and 0 <= ty < H for tx, ty in tiles):
                    break
                tiles = None
            if tiles is None:
                continue

            tile_set = set(map(tuple, tiles))
            d_delta = DIR_TO_DELTA[splitter_dir]

            # Compute input/output cells for each splitter tile
            input_cells = []
            output_cells = []
            for tx, ty in tiles:
                inp = (tx - d_delta[0], ty - d_delta[1])
                out = (tx + d_delta[0], ty + d_delta[1])
                input_cells.append(inp)
                output_cells.append(out)

            # All input/output cells must be in bounds and not overlap splitter tiles
            all_io = input_cells + output_cells
            if any(not (0 <= r < W and 0 <= c < H) for r, c in all_io):
                continue
            if any(c in tile_set for c in all_io):
                continue

            # Pick 1 source and 2 sinks at random positions (avoiding splitter tiles and I/O cells)
            reserved = tile_set | set(all_io)
            available = [(x, y) for x in range(W) for y in range(H) if (x, y) not in reserved]
            if len(available) < 3:
                continue

            chosen = random.sample(available, 3)
            source_pos = chosen[0]
            sink1_pos = chosen[1]
            sink2_pos = chosen[2]

            source_dir = random.choice(dirs)
            sink1_dir = random.choice(dirs)
            sink2_dir = random.choice(dirs)

            # Compute connection cells for source and sinks
            ds = DIR_TO_DELTA[source_dir]
            dk1 = DIR_TO_DELTA[sink1_dir]
            dk2 = DIR_TO_DELTA[sink2_dir]

            source_output = (source_pos[0] + ds[0], source_pos[1] + ds[1])
            sink1_input = (sink1_pos[0] - dk1[0], sink1_pos[1] - dk1[1])
            sink2_input = (sink2_pos[0] - dk2[0], sink2_pos[1] - dk2[1])

            # All connection cells must be in bounds
            conn_cells = [source_output, sink1_input, sink2_input]
            if any(not (0 <= r < W and 0 <= c < H) for r, c in conn_cells):
                continue

            # Connection cells must not overlap entity positions or each other
            all_fixed = tile_set | {source_pos, sink1_pos, sink2_pos}
            conn_set = set(conn_cells)
            if len(conn_set) != len(conn_cells):
                continue  # connection cells overlap each other
            if conn_set & all_fixed:
                continue  # connection cells overlap entity positions

            # Path 1: source output → one of the splitter input cells
            # Block entity positions, but NOT the start/end cells of each path.
            # Also block the unused input cell AND the cell behind it to
            # prevent sideloading into the splitter's unused input.
            blocked_base = all_fixed
            unused_input_buffer_0 = (input_cells[0][0] - d_delta[0], input_cells[0][1] - d_delta[1])
            unused_input_buffer_1 = (input_cells[1][0] - d_delta[0], input_cells[1][1] - d_delta[1])

            blocked1 = blocked_base | set(output_cells) | {sink1_input, sink2_input, input_cells[1], unused_input_buffer_1}
            path1 = find_belt_path(W, H, source_output, input_cells[0], splitter_dir, blocked1)
            if path1 is None:
                blocked1 = blocked_base | set(output_cells) | {sink1_input, sink2_input, input_cells[0], unused_input_buffer_0}
                path1 = find_belt_path(W, H, source_output, input_cells[1], splitter_dir, blocked1)
                if path1 is None:
                    continue

            # Determine which input was used vs unused
            path1_end = path1[-1][:2]
            if path1_end == input_cells[0]:
                unused_input = input_cells[1]
                unused_buffer = unused_input_buffer_1
            else:
                unused_input = input_cells[0]
                unused_buffer = unused_input_buffer_0

            path1_cells = {(x, y) for x, y, _ in path1}

            # Block unused input + buffer, plus the cells ahead of each sink
            # (where the sink would output to) to prevent pass-through
            # double-counting where one output path routes through a sink.
            unused_block = {unused_input, unused_buffer}
            sink1_output = (sink1_pos[0] + dk1[0], sink1_pos[1] + dk1[1])
            sink2_output = (sink2_pos[0] + dk2[0], sink2_pos[1] + dk2[1])
            sink_buffers = {sink1_output, sink2_output}

            # Path 2+3: splitter outputs → sink inputs
            # Try both assignments: (out0→sink1, out1→sink2) and (out0→sink2, out1→sink1)
            path2 = None
            path3 = None
            for out_a, out_b, sk_a, sk_a_dir, sk_b, sk_b_dir in [
                (output_cells[0], output_cells[1], sink1_input, sink1_dir, sink2_input, sink2_dir),
                (output_cells[0], output_cells[1], sink2_input, sink2_dir, sink1_input, sink1_dir),
            ]:
                blocked2 = blocked_base | path1_cells | {sk_b, out_b} | set(input_cells) | unused_block | sink_buffers
                p2 = find_belt_path(W, H, out_a, sk_a, sk_a_dir, blocked2)
                if p2 is None:
                    continue
                p2_cells = {(x, y) for x, y, _ in p2}
                blocked3 = blocked_base | path1_cells | p2_cells | {out_a} | set(input_cells) | unused_block | sink_buffers
                p3 = find_belt_path(W, H, out_b, sk_b, sk_b_dir, blocked3)
                if p3 is not None:
                    path2, path3 = p2, p3
                    break
            if path2 is None or path3 is None:
                continue

            path2_cells = {(x, y) for x, y, _ in path2}

            # Enforce max_entities
            total_entities = len(path1) + len(path2) + len(path3) + 1  # +1 for splitter (2 tiles but 1 entity)
            if total_entities > max_entities:
                continue

            # Place source and sinks
            world_CWH[Channel.ENTITIES.value, source_pos[0], source_pos[1]] = str2ent("source").value
            world_CWH[Channel.DIRECTION.value, source_pos[0], source_pos[1]] = source_dir.value
            world_CWH[Channel.ITEMS.value, source_pos[0], source_pos[1]] = item_value

            for sink_pos, sink_dir in [(sink1_pos, sink1_dir), (sink2_pos, sink2_dir)]:
                world_CWH[Channel.ENTITIES.value, sink_pos[0], sink_pos[1]] = str2ent("sink").value
                world_CWH[Channel.DIRECTION.value, sink_pos[0], sink_pos[1]] = sink_dir.value
                world_CWH[Channel.ITEMS.value, sink_pos[0], sink_pos[1]] = item_value

            # Place splitter (all tiles)
            for tx, ty in tiles:
                world_CWH[Channel.ENTITIES.value, tx, ty] = splitter_ent.value
                world_CWH[Channel.DIRECTION.value, tx, ty] = splitter_dir.value

            # Place belt paths
            for x, y, d in path1 + path2 + path3:
                world_CWH[Channel.ENTITIES.value, x, y] = str2ent("transport_belt").value
                world_CWH[Channel.DIRECTION.value, x, y] = d.value

            # Verify throughput > 0
            tp, _ = factorion_rs.simulate_throughput(
                world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
            )
            if tp <= 0:
                continue

            # Splitter is structurally required for the lesson; without
            # it the source(s)/sink(s) layout becomes ambiguous (could
            # be solved by belts alone). Protect it from blanking.
            protected_positions = frozenset(tuple(t) for t in tiles)

            break
        if count == 0:
            return None

    elif kind.value == LessonKind.SPLITTER_MERGE.value:
        # 2x(source → belts) → splitter → belts → 1 sink
        #
        # KNOWN LIMITATION — see issue #75. The splitter's single
        # output belt caps total flow at 15 i/s while sources are
        # infinite. One fully-connected source already saturates the
        # output belt, so the second source's path is decorative —
        # an agent can ignore one source entirely and still hit max
        # throughput. Fine for SFT (which doesn't see throughput),
        # but breaks the lesson's intent under PPO. Fix when wiring
        # up PPO on these lessons (rate-limit sources, OR redesign so
        # both splitter outputs feed the sink).
        original_count = max(500, size * size * 10)
        count = original_count

        if random_item:
            item_value = random.choice(
                [v.value for k, v in items.items() if v.name != "empty"]
            )
        else:
            item_value = str2item("electronic_circuit").value

        splitter_ent = str2ent("splitter")

        while count > 0:
            count -= 1
            world_CWH = torch.tensor(new_world(width=size, height=size)).permute(2, 0, 1)
            C, W, H = world_CWH.shape

            dirs = [d for d in Direction if d != Direction.NONE]
            splitter_dir = random.choice(dirs)

            # Pick splitter anchor; both tiles must fit
            tiles = None
            for _ in range(20):
                sx = random.randint(0, W - 1)
                sy = random.randint(0, H - 1)
                tiles = factorion_rs.py_entity_tiles(sx, sy, splitter_dir.value, splitter_ent.width, splitter_ent.height)
                if tiles is not None and all(0 <= tx < W and 0 <= ty < H for tx, ty in tiles):
                    break
                tiles = None
            if tiles is None:
                continue

            tile_set = set(map(tuple, tiles))
            d_delta = DIR_TO_DELTA[splitter_dir]

            input_cells = [(tx - d_delta[0], ty - d_delta[1]) for tx, ty in tiles]
            output_cells = [(tx + d_delta[0], ty + d_delta[1]) for tx, ty in tiles]

            all_io = input_cells + output_cells
            if any(not (0 <= r < W and 0 <= c < H) for r, c in all_io):
                continue
            if any(c in tile_set for c in all_io):
                continue

            # Pick 2 sources and 1 sink
            reserved = tile_set | set(all_io)
            available = [(x, y) for x in range(W) for y in range(H) if (x, y) not in reserved]
            if len(available) < 3:
                continue

            chosen = random.sample(available, 3)
            source1_pos = chosen[0]
            source2_pos = chosen[1]
            sink_pos = chosen[2]

            source1_dir = random.choice(dirs)
            source2_dir = random.choice(dirs)
            sink_dir = random.choice(dirs)

            ds1 = DIR_TO_DELTA[source1_dir]
            ds2 = DIR_TO_DELTA[source2_dir]
            dk = DIR_TO_DELTA[sink_dir]

            source1_output = (source1_pos[0] + ds1[0], source1_pos[1] + ds1[1])
            source2_output = (source2_pos[0] + ds2[0], source2_pos[1] + ds2[1])
            sink_input = (sink_pos[0] - dk[0], sink_pos[1] - dk[1])

            conn_cells = [source1_output, source2_output, sink_input]
            if any(not (0 <= r < W and 0 <= c < H) for r, c in conn_cells):
                continue

            all_fixed = tile_set | {source1_pos, source2_pos, sink_pos}
            conn_set = set(conn_cells)
            if len(conn_set) != len(conn_cells):
                continue
            if conn_set & all_fixed:
                continue

            # Path 1: source1 output → splitter input 0
            blocked_base = all_fixed
            blocked1 = blocked_base | set(output_cells) | {source2_output, sink_input, input_cells[1]}
            path1 = find_belt_path(W, H, source1_output, input_cells[0], splitter_dir, blocked1)
            if path1 is None:
                blocked1 = blocked_base | set(output_cells) | {source2_output, sink_input, input_cells[0]}
                path1 = find_belt_path(W, H, source1_output, input_cells[1], splitter_dir, blocked1)
                if path1 is None:
                    continue

            path1_cells = {(x, y) for x, y, _ in path1}
            path1_end = path1[-1][:2]
            remaining_input = input_cells[1] if path1_end == input_cells[0] else input_cells[0]

            # Path 2: source2 output → remaining splitter input
            blocked2 = blocked_base | set(output_cells) | path1_cells | {sink_input}
            path2 = find_belt_path(W, H, source2_output, remaining_input, splitter_dir, blocked2)
            if path2 is None:
                continue

            path2_cells = {(x, y) for x, y, _ in path2}

            # Path 3: splitter output → sink input (try both output cells).
            # Block the unused output cell AND the cell ahead of it to
            # prevent sideloading from the output path into the unused output.
            unused_output_buffer_0 = (output_cells[0][0] + d_delta[0], output_cells[0][1] + d_delta[1])
            unused_output_buffer_1 = (output_cells[1][0] + d_delta[0], output_cells[1][1] + d_delta[1])

            blocked3 = blocked_base | path1_cells | path2_cells | set(input_cells) | {output_cells[1], unused_output_buffer_1}
            path3 = find_belt_path(W, H, output_cells[0], sink_input, sink_dir, blocked3)
            if path3 is None:
                blocked3 = blocked_base | path1_cells | path2_cells | set(input_cells) | {output_cells[0], unused_output_buffer_0}
                path3 = find_belt_path(W, H, output_cells[1], sink_input, sink_dir, blocked3)
                if path3 is None:
                    continue

            total_entities = len(path1) + len(path2) + len(path3) + 1
            if total_entities > max_entities:
                continue

            # Place sources and sink
            for src_pos, src_dir in [(source1_pos, source1_dir), (source2_pos, source2_dir)]:
                world_CWH[Channel.ENTITIES.value, src_pos[0], src_pos[1]] = str2ent("source").value
                world_CWH[Channel.DIRECTION.value, src_pos[0], src_pos[1]] = src_dir.value
                world_CWH[Channel.ITEMS.value, src_pos[0], src_pos[1]] = item_value

            world_CWH[Channel.ENTITIES.value, sink_pos[0], sink_pos[1]] = str2ent("sink").value
            world_CWH[Channel.DIRECTION.value, sink_pos[0], sink_pos[1]] = sink_dir.value
            world_CWH[Channel.ITEMS.value, sink_pos[0], sink_pos[1]] = item_value

            # Place splitter
            for tx, ty in tiles:
                world_CWH[Channel.ENTITIES.value, tx, ty] = splitter_ent.value
                world_CWH[Channel.DIRECTION.value, tx, ty] = splitter_dir.value

            # Place belt paths
            for x, y, d in path1 + path2 + path3:
                world_CWH[Channel.ENTITIES.value, x, y] = str2ent("transport_belt").value
                world_CWH[Channel.DIRECTION.value, x, y] = d.value

            # Verify throughput > 0
            tp, _ = factorion_rs.simulate_throughput(
                world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
            )
            if tp <= 0:
                continue

            # Splitter is structurally required for the lesson; without
            # it the source(s)/sink(s) layout becomes ambiguous (could
            # be solved by belts alone). Protect it from blanking.
            protected_positions = frozenset(tuple(t) for t in tiles)

            break
        if count == 0:
            return None

    elif kind.value == LessonKind.ASSEMBLE_1IN_1OUT.value:
        # source → belt → input inserter → 3x3 assembler (with recipe) →
        # output inserter → belt → sink. Recipe is randomly chosen from
        # all 1-input 1-output recipes (copper_cable, iron_gear_wheel
        # at the time of writing).
        one_in_one_out = [
            (name, r)
            for name, r in recipes.items()
            if len(r.consumes) == 1 and len(r.produces) == 1
        ]
        if not one_in_one_out:
            raise Exception("No 1-input 1-output recipes available")

        original_count = max(500, size * size * 12)
        count = original_count
        asm_ent = str2ent("assembling_machine_1")

        # The 12 non-corner perimeter slots around a 3×3 assembler
        # anchored at (ax, ay), expressed as (offset_x, offset_y,
        # input_dir, output_dir). Input inserter direction faces
        # *into* the assembler body; output direction faces *out*.
        perim_slots = [
            # North side (ddy = -1): above the top row
            (0, -1, Direction.SOUTH, Direction.NORTH),
            (1, -1, Direction.SOUTH, Direction.NORTH),
            (2, -1, Direction.SOUTH, Direction.NORTH),
            # South side (ddy = 3): below the bottom row
            (0, 3, Direction.NORTH, Direction.SOUTH),
            (1, 3, Direction.NORTH, Direction.SOUTH),
            (2, 3, Direction.NORTH, Direction.SOUTH),
            # West side (ddx = -1)
            (-1, 0, Direction.EAST, Direction.WEST),
            (-1, 1, Direction.EAST, Direction.WEST),
            (-1, 2, Direction.EAST, Direction.WEST),
            # East side (ddx = 3)
            (3, 0, Direction.WEST, Direction.EAST),
            (3, 1, Direction.WEST, Direction.EAST),
            (3, 2, Direction.WEST, Direction.EAST),
        ]

        while count > 0:
            count -= 1
            world_CWH = torch.tensor(new_world(width=size, height=size)).permute(
                2, 0, 1
            )
            C, W, H = world_CWH.shape

            # 3×3 assembler can't fit
            if W < 3 or H < 3:
                raise Exception(
                    f"Grid {W}×{H} too small for ASSEMBLE_1IN_1OUT (need ≥ 3×3)"
                )

            # Pick recipe + items
            recipe_name, recipe = random.choice(one_in_one_out)
            input_item_name = next(iter(recipe.consumes.keys()))
            output_item_name = next(iter(recipe.produces.keys()))
            recipe_item_value = str2item(recipe_name).value
            input_item_value = str2item(input_item_name).value
            output_item_value = str2item(output_item_name).value

            # Pick assembler anchor
            ax = random.randint(0, W - 3)
            ay = random.randint(0, H - 3)
            asm_tiles = {
                (ax + dx, ay + dy) for dx in range(3) for dy in range(3)
            }

            # Pick distinct input + output perimeter slots
            in_slot, out_slot = random.sample(perim_slots, 2)
            in_dx, in_dy, in_inserter_dir, _ = in_slot
            out_dx, out_dy, _, out_inserter_dir = out_slot

            in_inserter_pos = (ax + in_dx, ay + in_dy)
            out_inserter_pos = (ax + out_dx, ay + out_dy)

            in_dir_delta = DIR_TO_DELTA[in_inserter_dir]
            out_dir_delta = DIR_TO_DELTA[out_inserter_dir]

            # Belt cell that feeds the input inserter (its pickup):
            in_pickup = (
                in_inserter_pos[0] - in_dir_delta[0],
                in_inserter_pos[1] - in_dir_delta[1],
            )
            # Belt cell where the output inserter drops:
            out_drop = (
                out_inserter_pos[0] + out_dir_delta[0],
                out_inserter_pos[1] + out_dir_delta[1],
            )

            # All key cells in bounds
            key_cells = [in_inserter_pos, out_inserter_pos, in_pickup, out_drop]
            if any(not (0 <= c[0] < W and 0 <= c[1] < H) for c in key_cells):
                continue
            # Distinct + outside assembler body
            if len(set(key_cells)) != len(key_cells):
                continue
            if any(c in asm_tiles for c in key_cells):
                continue

            # Compute ALL 12 non-corner perimeter slots and exclude them
            # from the source/sink candidate set. A Source/Sink placed on
            # a perimeter slot is treated as an inserter by
            # AssemblingMachine::connections (Source/Sink are inserter-
            # like), which would feed/drain the assembler at infinite
            # rate — bypassing the input/output inserter bottleneck and
            # breaking the closed-form throughput.
            all_perim = {
                (ax + ddx, ay + ddy)
                for ddx, ddy, _, _ in perim_slots
                if 0 <= ax + ddx < W and 0 <= ay + ddy < H
            }

            # Pick source + sink positions outside everything reserved
            reserved = asm_tiles | set(key_cells) | all_perim
            available = [
                (x, y)
                for x in range(W)
                for y in range(H)
                if (x, y) not in reserved
            ]
            if len(available) < 2:
                continue

            source_pos, sink_pos = random.sample(available, 2)
            dirs = [d for d in Direction if d != Direction.NONE]
            source_dir = random.choice(dirs)
            sink_dir = random.choice(dirs)

            ds = DIR_TO_DELTA[source_dir]
            dk = DIR_TO_DELTA[sink_dir]
            source_output = (source_pos[0] + ds[0], source_pos[1] + ds[1])
            sink_input = (sink_pos[0] - dk[0], sink_pos[1] - dk[1])

            # Source-output / sink-input must be in bounds and not overlap reserved
            if not (0 <= source_output[0] < W and 0 <= source_output[1] < H):
                continue
            if not (0 <= sink_input[0] < W and 0 <= sink_input[1] < H):
                continue
            if source_output in reserved or sink_input in reserved:
                continue
            if source_output == sink_input:
                continue

            all_fixed = reserved | {source_pos, sink_pos}

            # Path 1: source_output → in_pickup. Last belt orients toward
            # the input inserter.
            blocked1 = (
                asm_tiles
                | {in_inserter_pos, out_inserter_pos, source_pos, sink_pos,
                   sink_input, out_drop}
            )
            path1 = find_belt_path(
                W, H, source_output, in_pickup, in_inserter_dir, blocked1
            )
            if path1 is None:
                continue
            path1_cells = {(x, y) for x, y, _ in path1}

            # Path 2: out_drop → sink_input. Last belt orients toward sink.
            blocked2 = (
                asm_tiles
                | {in_inserter_pos, out_inserter_pos, source_pos, sink_pos,
                   in_pickup, source_output}
                | path1_cells
            )
            path2 = find_belt_path(
                W, H, out_drop, sink_input, sink_dir, blocked2
            )
            if path2 is None:
                continue

            # 1 assembler + 2 inserters + belts (assembler counts as 1
            # entity even though it occupies 9 tiles).
            total_entities = len(path1) + len(path2) + 3
            if total_entities > max_entities:
                continue

            # Place source
            world_CWH[Channel.ENTITIES.value, source_pos[0], source_pos[1]] = (
                str2ent("source").value
            )
            world_CWH[Channel.DIRECTION.value, source_pos[0], source_pos[1]] = (
                source_dir.value
            )
            world_CWH[Channel.ITEMS.value, source_pos[0], source_pos[1]] = (
                input_item_value
            )

            # Place sink
            world_CWH[Channel.ENTITIES.value, sink_pos[0], sink_pos[1]] = (
                str2ent("sink").value
            )
            world_CWH[Channel.DIRECTION.value, sink_pos[0], sink_pos[1]] = (
                sink_dir.value
            )
            world_CWH[Channel.ITEMS.value, sink_pos[0], sink_pos[1]] = (
                output_item_value
            )

            # Place assembler (3×3, all tiles tagged with the recipe item)
            for tx, ty in asm_tiles:
                world_CWH[Channel.ENTITIES.value, tx, ty] = asm_ent.value
                world_CWH[Channel.DIRECTION.value, tx, ty] = (
                    Direction.NORTH.value
                )
                world_CWH[Channel.ITEMS.value, tx, ty] = recipe_item_value

            # Place input + output inserters
            world_CWH[
                Channel.ENTITIES.value, in_inserter_pos[0], in_inserter_pos[1]
            ] = str2ent("inserter").value
            world_CWH[
                Channel.DIRECTION.value, in_inserter_pos[0], in_inserter_pos[1]
            ] = in_inserter_dir.value

            world_CWH[
                Channel.ENTITIES.value, out_inserter_pos[0], out_inserter_pos[1]
            ] = str2ent("inserter").value
            world_CWH[
                Channel.DIRECTION.value, out_inserter_pos[0], out_inserter_pos[1]
            ] = out_inserter_dir.value

            # Place belt paths
            for x, y, d in path1 + path2:
                world_CWH[Channel.ENTITIES.value, x, y] = str2ent(
                    "transport_belt"
                ).value
                world_CWH[Channel.DIRECTION.value, x, y] = d.value

            # Verify throughput > 0
            tp, _ = factorion_rs.simulate_throughput(
                world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
            )
            if tp <= 0:
                continue

            # The assembler is a valid blanking target: when removed,
            # the recipe is still inferable from the source (input item)
            # and sink (output item), which are never blanked. Forcing
            # the model to place the assembler is the only way the item
            # head learns to predict non-zero recipes — otherwise every
            # expert action is on a belt/inserter/splitter, all of which
            # have empty ITEMS channels.

            break
        if count == 0:
            return None

    elif kind.value == LessonKind.MOVE_VIA_UG_BELT.value:
        # A 1..4-tile-thick wall spans the grid perpendicular to a chosen
        # flow direction. UG_DOWN sits on the source side flush against
        # the wall; UG_UP sits directly opposite on the sink side. Source
        # and sink are placed randomly on their respective sides, and
        # transport belts route source→UG_DOWN and UG_UP→sink via
        # find_belt_path. FOOTPRINT marks ONLY the wall tiles as
        # UNAVAILABLE — every other tile is freely buildable.
        original_count = max(500, size * size * 8)
        count = original_count

        if random_item:
            item_value = random.choice(
                [v.value for k, v in items.items() if v.name != "empty"]
            )
        else:
            item_value = str2item("electronic_circuit").value

        while count > 0:
            count -= 1
            world_CWH = torch.tensor(new_world(width=size, height=size)).permute(
                2, 0, 1
            )
            C, W, H = world_CWH.shape

            flow_dir = random.choice(
                [d for d in Direction if d != Direction.NONE]
            )
            is_horizontal = flow_dir in (Direction.EAST, Direction.WEST)
            flow_span = W if is_horizontal else H
            perp_span = H if is_horizontal else W

            # Need ≥1 source-side tile + wall (≥1) + ≥1 sink-side tile
            if flow_span < 3:
                raise Exception(
                    f"Grid {W}×{H} too small for MOVE_VIA_UG_BELT (need ≥ 3 along {flow_dir})"
                )

            max_wall = min(4, flow_span - 2)
            wall_thickness = random.randint(1, max_wall)
            wall_lo = random.randint(1, flow_span - 1 - wall_thickness)
            wall_hi = wall_lo + wall_thickness - 1

            forward = flow_dir in (Direction.EAST, Direction.SOUTH)
            if forward:
                ug_down_flow = wall_lo - 1
                ug_up_flow = wall_hi + 1
                src_lo, src_hi = 0, wall_lo - 1
                snk_lo, snk_hi = wall_hi + 1, flow_span - 1
            else:
                ug_down_flow = wall_hi + 1
                ug_up_flow = wall_lo - 1
                src_lo, src_hi = wall_hi + 1, flow_span - 1
                snk_lo, snk_hi = 0, wall_lo - 1

            ug_perp = random.randint(0, perp_span - 1)

            def fp_to_xy(fc, pc, ih=is_horizontal):
                return (fc, pc) if ih else (pc, fc)

            ug_down_pos = fp_to_xy(ug_down_flow, ug_perp)
            ug_up_pos = fp_to_xy(ug_up_flow, ug_perp)

            wall_tiles = set()
            for fc in range(wall_lo, wall_hi + 1):
                for pc in range(perp_span):
                    wall_tiles.add(fp_to_xy(fc, pc))

            source_cells = []
            for fc in range(src_lo, src_hi + 1):
                for pc in range(perp_span):
                    cell = fp_to_xy(fc, pc)
                    if cell != ug_down_pos:
                        source_cells.append(cell)
            sink_cells = []
            for fc in range(snk_lo, snk_hi + 1):
                for pc in range(perp_span):
                    cell = fp_to_xy(fc, pc)
                    if cell != ug_up_pos:
                        sink_cells.append(cell)
            if not source_cells or not sink_cells:
                continue

            source_pos = random.choice(source_cells)
            sink_pos = random.choice(sink_cells)
            dirs = [d for d in Direction if d != Direction.NONE]
            source_dir = random.choice(dirs)
            sink_dir = random.choice(dirs)

            ds = DIR_TO_DELTA[source_dir]
            dk = DIR_TO_DELTA[sink_dir]
            source_drop = (source_pos[0] + ds[0], source_pos[1] + ds[1])
            sink_input = (sink_pos[0] - dk[0], sink_pos[1] - dk[1])

            if not (0 <= source_drop[0] < W and 0 <= source_drop[1] < H):
                continue
            if not (0 <= sink_input[0] < W and 0 <= sink_input[1] < H):
                continue

            # source_drop must be on source side or land directly on UG_DOWN
            if source_drop in wall_tiles:
                continue
            if source_drop in (ug_up_pos, sink_pos):
                continue
            sd_flow = source_drop[0] if is_horizontal else source_drop[1]
            if not (
                src_lo <= sd_flow <= src_hi or source_drop == ug_down_pos
            ):
                continue

            # sink_input must be on sink side or land directly on UG_UP
            if sink_input in wall_tiles:
                continue
            if sink_input in (ug_down_pos, source_pos):
                continue
            si_flow = sink_input[0] if is_horizontal else sink_input[1]
            if not (
                snk_lo <= si_flow <= snk_hi or sink_input == ug_up_pos
            ):
                continue

            flow_delta = DIR_TO_DELTA[flow_dir]

            # Path 1: source_drop → UG_DOWN_input (on source side)
            if source_drop == ug_down_pos:
                path1 = []
            else:
                ug_down_input = (
                    ug_down_pos[0] - flow_delta[0],
                    ug_down_pos[1] - flow_delta[1],
                )
                blocked1 = wall_tiles | {
                    source_pos, sink_pos, ug_down_pos, ug_up_pos, sink_input,
                }
                # Restrict path1 to source side: block all sink-side cells.
                for fc in range(snk_lo, snk_hi + 1):
                    for pc in range(perp_span):
                        blocked1.add(fp_to_xy(fc, pc))
                if source_drop in blocked1 or ug_down_input in blocked1:
                    continue
                path1 = find_belt_path(
                    W, H, source_drop, ug_down_input, flow_dir, blocked1
                )
                if path1 is None:
                    continue

            # Path 2: UG_UP_drop → sink_input (on sink side)
            if sink_input == ug_up_pos:
                path2 = []
            else:
                ug_up_drop = (
                    ug_up_pos[0] + flow_delta[0],
                    ug_up_pos[1] + flow_delta[1],
                )
                path1_cells = {(x, y) for x, y, _ in path1}
                blocked2 = (
                    wall_tiles
                    | {source_pos, sink_pos, ug_down_pos, ug_up_pos, source_drop}
                    | path1_cells
                )
                # Restrict path2 to sink side: block all source-side cells.
                for fc in range(src_lo, src_hi + 1):
                    for pc in range(perp_span):
                        blocked2.add(fp_to_xy(fc, pc))
                if ug_up_drop in blocked2 or sink_input in blocked2:
                    continue
                path2 = find_belt_path(
                    W, H, ug_up_drop, sink_input, sink_dir, blocked2
                )
                if path2 is None:
                    continue

            # Place source/sink
            world_CWH[Channel.ENTITIES.value, source_pos[0], source_pos[1]] = (
                str2ent("source").value
            )
            world_CWH[Channel.DIRECTION.value, source_pos[0], source_pos[1]] = (
                source_dir.value
            )
            world_CWH[Channel.ITEMS.value, source_pos[0], source_pos[1]] = item_value

            world_CWH[Channel.ENTITIES.value, sink_pos[0], sink_pos[1]] = (
                str2ent("sink").value
            )
            world_CWH[Channel.DIRECTION.value, sink_pos[0], sink_pos[1]] = (
                sink_dir.value
            )
            world_CWH[Channel.ITEMS.value, sink_pos[0], sink_pos[1]] = item_value

            # Place UG pair (always facing flow_dir)
            world_CWH[Channel.ENTITIES.value, ug_down_pos[0], ug_down_pos[1]] = (
                str2ent("underground_belt").value
            )
            world_CWH[Channel.DIRECTION.value, ug_down_pos[0], ug_down_pos[1]] = (
                flow_dir.value
            )
            world_CWH[Channel.MISC.value, ug_down_pos[0], ug_down_pos[1]] = (
                Misc.UNDERGROUND_DOWN.value
            )

            world_CWH[Channel.ENTITIES.value, ug_up_pos[0], ug_up_pos[1]] = (
                str2ent("underground_belt").value
            )
            world_CWH[Channel.DIRECTION.value, ug_up_pos[0], ug_up_pos[1]] = (
                flow_dir.value
            )
            world_CWH[Channel.MISC.value, ug_up_pos[0], ug_up_pos[1]] = (
                Misc.UNDERGROUND_UP.value
            )

            # Place belts
            for x, y, d in path1 + path2:
                world_CWH[Channel.ENTITIES.value, x, y] = str2ent(
                    "transport_belt"
                ).value
                world_CWH[Channel.DIRECTION.value, x, y] = d.value

            # Sanity: the solved factory must deliver items.
            tp, _ = factorion_rs.simulate_throughput(
                world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
            )
            if tp <= 0:
                continue

            total_entities = 2 + len(path1) + len(path2)
            if total_entities > max_entities:
                continue

            # Mark only the wall as UNAVAILABLE; every other tile is
            # freely buildable on either side of the wall.
            world_CWH[Channel.FOOTPRINT.value, :, :] = Footprint.AVAILABLE.value
            for wx, wy in wall_tiles:
                world_CWH[Channel.FOOTPRINT.value, wx, wy] = (
                    Footprint.UNAVAILABLE.value
                )

            break
        if count == 0:
            return None

    elif kind.value == LessonKind.ASSEMBLE_2IN_1OUT.value:
        # 2 sources → belts → 2 input inserters → 3×3 assembler (recipe)
        #   → output inserter → belts → 1 sink. Recipe is randomly chosen
        # from all 2-input 1-output recipes in the live recipe table
        # (filtered dynamically — no hardcoded list, so new recipes added
        # in factorion_rs are picked up automatically).
        two_in_one_out = [
            (name, r)
            for name, r in recipes.items()
            if len(r.consumes) == 2 and len(r.produces) == 1
        ]
        if not two_in_one_out:
            raise Exception("No 2-input 1-output recipes available")

        original_count = max(500, size * size * 16)
        count = original_count
        asm_ent = str2ent("assembling_machine_1")

        # 12 non-corner perimeter slots — same set as the 1-in-1-out
        # generator, but here we pick THREE distinct slots: two for
        # input inserters (one per ingredient) and one for the output.
        perim_slots = [
            # North side (ddy = -1)
            (0, -1, Direction.SOUTH, Direction.NORTH),
            (1, -1, Direction.SOUTH, Direction.NORTH),
            (2, -1, Direction.SOUTH, Direction.NORTH),
            # South side (ddy = 3)
            (0, 3, Direction.NORTH, Direction.SOUTH),
            (1, 3, Direction.NORTH, Direction.SOUTH),
            (2, 3, Direction.NORTH, Direction.SOUTH),
            # West side (ddx = -1)
            (-1, 0, Direction.EAST, Direction.WEST),
            (-1, 1, Direction.EAST, Direction.WEST),
            (-1, 2, Direction.EAST, Direction.WEST),
            # East side (ddx = 3)
            (3, 0, Direction.WEST, Direction.EAST),
            (3, 1, Direction.WEST, Direction.EAST),
            (3, 2, Direction.WEST, Direction.EAST),
        ]

        while count > 0:
            count -= 1
            world_CWH = torch.tensor(new_world(width=size, height=size)).permute(
                2, 0, 1
            )
            C, W, H = world_CWH.shape

            if W < 3 or H < 3:
                raise Exception(
                    f"Grid {W}×{H} too small for ASSEMBLE_2IN_1OUT (need ≥ 3×3)"
                )

            # Pick recipe + items
            recipe_name, recipe = random.choice(two_in_one_out)
            input_items = list(recipe.consumes.keys())
            # Randomize which ingredient is "A" vs "B" so the two
            # sources appear in random order across seeds.
            random.shuffle(input_items)
            input_a_name, input_b_name = input_items
            output_item_name = next(iter(recipe.produces.keys()))
            recipe_item_value = str2item(recipe_name).value
            input_a_value = str2item(input_a_name).value
            input_b_value = str2item(input_b_name).value
            output_item_value = str2item(output_item_name).value

            # Pick assembler anchor
            ax = random.randint(0, W - 3)
            ay = random.randint(0, H - 3)
            asm_tiles = {
                (ax + dx, ay + dy) for dx in range(3) for dy in range(3)
            }

            # Pick three distinct perimeter slots: 2 inputs + 1 output
            in_a_slot, in_b_slot, out_slot = random.sample(perim_slots, 3)

            in_a_pos = (ax + in_a_slot[0], ay + in_a_slot[1])
            in_a_dir = in_a_slot[2]
            in_b_pos = (ax + in_b_slot[0], ay + in_b_slot[1])
            in_b_dir = in_b_slot[2]
            out_pos = (ax + out_slot[0], ay + out_slot[1])
            out_dir = out_slot[3]

            in_a_dd = DIR_TO_DELTA[in_a_dir]
            in_b_dd = DIR_TO_DELTA[in_b_dir]
            out_dd = DIR_TO_DELTA[out_dir]

            in_a_pickup = (in_a_pos[0] - in_a_dd[0], in_a_pos[1] - in_a_dd[1])
            in_b_pickup = (in_b_pos[0] - in_b_dd[0], in_b_pos[1] - in_b_dd[1])
            out_drop = (out_pos[0] + out_dd[0], out_pos[1] + out_dd[1])

            key_cells = [
                in_a_pos, in_b_pos, out_pos,
                in_a_pickup, in_b_pickup, out_drop,
            ]
            if any(not (0 <= c[0] < W and 0 <= c[1] < H) for c in key_cells):
                continue
            if len(set(key_cells)) != len(key_cells):
                continue
            if any(c in asm_tiles for c in key_cells):
                continue

            # Exclude all 12 perimeter slots — a Source/Sink placed on
            # one is treated as an inserter by AssemblingMachine::
            # connections and would feed/drain the assembler at infinite
            # rate, breaking the closed-form throughput.
            all_perim = {
                (ax + ddx, ay + ddy)
                for ddx, ddy, _, _ in perim_slots
                if 0 <= ax + ddx < W and 0 <= ay + ddy < H
            }
            reserved = asm_tiles | set(key_cells) | all_perim
            available = [
                (x, y)
                for x in range(W)
                for y in range(H)
                if (x, y) not in reserved
            ]
            if len(available) < 3:
                continue

            src_a_pos, src_b_pos, sink_pos = random.sample(available, 3)
            dirs = [d for d in Direction if d != Direction.NONE]
            src_a_dir = random.choice(dirs)
            src_b_dir = random.choice(dirs)
            sink_dir = random.choice(dirs)

            ds_a = DIR_TO_DELTA[src_a_dir]
            ds_b = DIR_TO_DELTA[src_b_dir]
            dk = DIR_TO_DELTA[sink_dir]
            src_a_out = (src_a_pos[0] + ds_a[0], src_a_pos[1] + ds_a[1])
            src_b_out = (src_b_pos[0] + ds_b[0], src_b_pos[1] + ds_b[1])
            sink_in = (sink_pos[0] - dk[0], sink_pos[1] - dk[1])

            conn = [src_a_out, src_b_out, sink_in]
            if any(not (0 <= c[0] < W and 0 <= c[1] < H) for c in conn):
                continue
            if len(set(conn)) != len(conn):
                continue
            if any(c in reserved for c in conn):
                continue
            if any(c in {src_a_pos, src_b_pos, sink_pos} for c in conn):
                continue

            fixed_cells = (
                asm_tiles
                | {in_a_pos, in_b_pos, out_pos}
                | {src_a_pos, src_b_pos, sink_pos}
            )

            # Path A: source A → input-A pickup
            blocked_a = (
                fixed_cells
                | {in_b_pickup, out_drop, src_b_out, sink_in}
            )
            path_a = find_belt_path(
                W, H, src_a_out, in_a_pickup, in_a_dir, blocked_a
            )
            if path_a is None:
                continue
            path_a_cells = {(x, y) for x, y, _ in path_a}

            # Path B: source B → input-B pickup
            blocked_b = (
                fixed_cells
                | {in_a_pickup, out_drop, src_a_out, sink_in}
                | path_a_cells
            )
            path_b = find_belt_path(
                W, H, src_b_out, in_b_pickup, in_b_dir, blocked_b
            )
            if path_b is None:
                continue
            path_b_cells = {(x, y) for x, y, _ in path_b}

            # Path C: output drop → sink input
            blocked_c = (
                fixed_cells
                | {in_a_pickup, in_b_pickup, src_a_out, src_b_out}
                | path_a_cells
                | path_b_cells
            )
            path_c = find_belt_path(
                W, H, out_drop, sink_in, sink_dir, blocked_c
            )
            if path_c is None:
                continue

            # 1 assembler + 3 inserters + belts (assembler is 1 entity)
            total_entities = len(path_a) + len(path_b) + len(path_c) + 4
            if total_entities > max_entities:
                continue

            # Place sources
            world_CWH[Channel.ENTITIES.value, src_a_pos[0], src_a_pos[1]] = (
                str2ent("source").value
            )
            world_CWH[Channel.DIRECTION.value, src_a_pos[0], src_a_pos[1]] = (
                src_a_dir.value
            )
            world_CWH[Channel.ITEMS.value, src_a_pos[0], src_a_pos[1]] = (
                input_a_value
            )

            world_CWH[Channel.ENTITIES.value, src_b_pos[0], src_b_pos[1]] = (
                str2ent("source").value
            )
            world_CWH[Channel.DIRECTION.value, src_b_pos[0], src_b_pos[1]] = (
                src_b_dir.value
            )
            world_CWH[Channel.ITEMS.value, src_b_pos[0], src_b_pos[1]] = (
                input_b_value
            )

            # Place sink
            world_CWH[Channel.ENTITIES.value, sink_pos[0], sink_pos[1]] = (
                str2ent("sink").value
            )
            world_CWH[Channel.DIRECTION.value, sink_pos[0], sink_pos[1]] = (
                sink_dir.value
            )
            world_CWH[Channel.ITEMS.value, sink_pos[0], sink_pos[1]] = (
                output_item_value
            )

            # Place assembler (3×3, all tiles tagged with the recipe item)
            for tx, ty in asm_tiles:
                world_CWH[Channel.ENTITIES.value, tx, ty] = asm_ent.value
                world_CWH[Channel.DIRECTION.value, tx, ty] = (
                    Direction.NORTH.value
                )
                world_CWH[Channel.ITEMS.value, tx, ty] = recipe_item_value

            # Place 2 input inserters + 1 output inserter
            for ipos, idir in [
                (in_a_pos, in_a_dir),
                (in_b_pos, in_b_dir),
                (out_pos, out_dir),
            ]:
                world_CWH[
                    Channel.ENTITIES.value, ipos[0], ipos[1]
                ] = str2ent("inserter").value
                world_CWH[
                    Channel.DIRECTION.value, ipos[0], ipos[1]
                ] = idir.value

            # Place belt paths
            for x, y, d in path_a + path_b + path_c:
                world_CWH[Channel.ENTITIES.value, x, y] = str2ent(
                    "transport_belt"
                ).value
                world_CWH[Channel.DIRECTION.value, x, y] = d.value

            # Verify throughput > 0
            tp, _ = factorion_rs.simulate_throughput(
                world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
            )
            if tp <= 0:
                continue

            # The assembler is a valid blanking target — same rationale
            # as ASSEMBLE_1IN_1OUT: source/sink items are never blanked
            # so the recipe is inferable, and forcing the model to
            # place the assembler is the only way the item head learns
            # to predict non-zero recipes.

            break
        if count == 0:
            return None

    elif kind.value == LessonKind.FROM_BLUEPRINT.value:
        # Sample a random hand-authored blueprint from
        # `lesson_blueprints/`, decode it, then apply data
        # augmentations:
        #   - optional 1-in-1-out recipe substitution (only for gears
        #     factories: iron-plate → iron-gear-wheel layouts),
        #   - optional horizontal / vertical flip,
        #   - random translation within the size×size grid (if the
        #     decoded footprint is smaller than the world).
        # Blueprints whose footprint exceeds `size` are skipped and a
        # different one is tried. The throughput check rejects
        # blueprints that don't actually produce flow (malformed or
        # incompatible after augmentation).
        bp_paths = sorted(glob.glob(os.path.join(
            LESSON_BLUEPRINT_DIR, "*.txt"
        )))
        if not bp_paths:
            raise Exception(
                f"No blueprints in {LESSON_BLUEPRINT_DIR}; "
                "FROM_BLUEPRINT needs at least one .txt seed."
            )

        original_count = max(500, size * size * 2)
        count = original_count
        while count > 0:
            count -= 1
            bp_path = random.choice(bp_paths)
            try:
                bp_text = read_blueprint_file(bp_path)
                decoded_np = blueprint2world(bp_text)
            except (ValueError, AssertionError, KeyError):
                # Malformed blueprint — try a different sample.
                continue
            decoded = torch.tensor(decoded_np)
            _, w_bp, h_bp = decoded.shape
            if w_bp > size or h_bp > size:
                continue

            # Recipe substitution before flipping — order doesn't
            # matter (the substitution only touches ITEMS values, not
            # positions or directions), but doing it first keeps the
            # mutations grouped by what they affect.
            if _is_gears_factory(decoded):
                decoded = _substitute_gears_recipe(decoded)

            if random.random() < 0.5:
                decoded = _flip_world(decoded, axis=1)  # horizontal
            if random.random() < 0.5:
                decoded = _flip_world(decoded, axis=2)  # vertical

            ox = random.randint(0, size - w_bp)
            oy = random.randint(0, size - h_bp)

            world_CWH = torch.tensor(
                new_world(width=size, height=size)
            ).permute(2, 0, 1)
            world_CWH[:, ox:ox + w_bp, oy:oy + h_bp] = decoded

            # Extend belt chains backward from each source/sink into
            # any empties created by translation (or already present
            # in the source blueprint), filling the gap with belts in
            # the chain's flow direction. Done after translation so the
            # empties available include the world's margins, not just
            # gaps inside the blueprint's own footprint.
            world_CWH = _extend_belt_chains(world_CWH)

            tp, _ = factorion_rs.simulate_throughput(
                world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
            )
            if tp <= 0:
                continue

            total_entities = _count_removable_entity_units(world_CWH)
            if total_entities == 0:
                continue
            if total_entities > max_entities:
                continue

            # Every non-source/sink entity is a valid blanking target.
            # The recipe is still inferable from source/sink ITEM IDs
            # (never blanked), so even the assembler can be removed.
            protected_positions = frozenset()
            break
        if count == 0:
            return None

    else:
        raise ValueError(f"Can't handle {kind}")

    return Factory(
        world_CWH=world_CWH,
        total_entities=total_entities,
        protected_positions=protected_positions,
    )


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

    while q:
        r, c = q.popleft()
        if (r, c) == end:
            break
        for dr, dc in deltas:
            nr, nc = r + dr, c + dc
            if not in_bounds((nr, nc)):
                continue
            if (nr, nc) in blocked:
                continue
            if (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                parents[(nr, nc)].append((r, c))
                q.append((nr, nc))
            elif dist[(nr, nc)] == dist[(r, c)] + 1:
                parents[(nr, nc)].append((r, c))

    if end not in dist:
        return None, None
    return dist, parents

def _path_to_belts(path, end_dir):
    """Convert a list of (x, y) cells into belt placements with directions."""
    belts: List[Tuple[int, int, Direction]] = []
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        dr, dc = (r2 - r1, c2 - c1)
        for d, delta in DIR_TO_DELTA.items():
            if delta == (dr, dc):
                belts.append((r1, c1, d))
                break
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

