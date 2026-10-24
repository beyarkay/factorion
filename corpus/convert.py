"""Human blueprints -> factorion training windows: the HUMAN_BLUEPRINTS lesson.

The corpus stores raw exchange strings so that every conversion decision lives
here and re-runs at load time: teaching the engine a new entity only means
editing `_KNOWN`. A blueprint is rasterised once (`rasterise`), then any number
of windows are cut from it (`window`): a build region whose human layout is
the target, inside a view of the rest of the factory as unbuildable context
to connect to. Entities the engine doesn't model are *masked*: their tiles
become `Footprint.MASKED`, so no loss reaches them, while what the game forbids
(a save's water, lava, space, cliffs) is `Footprint.UNBUILDABLE`.
"""

import base64
import itertools
import json
import math
import random
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

import factorion_rs
from factorion import UNDIRECTED_ENT_IDS, Channel, Direction, Footprint, Misc, items, refresh_power, str2ent, str2item
from denoise import _ASM_RECIPE_ITEM_IDS

# Placement footprint of every base / Space Age / Quality / Elevated Rails
# entity, dumped from the headless game's prototypes (`corpus/README.md`).
_SIZES: dict[str, list] = json.loads((Path(__file__).parent / "entity_sizes.json").read_text())
_LOGISTIC = ("active-provider", "passive-provider", "storage", "requester", "buffer")
# 1.1 names whose 2.0 prototype was renamed.
_SIZES |= {
    "curved-rail": _SIZES["legacy-curved-rail"],
    "filter-inserter": [1, 1, "base"],
    "stack-filter-inserter": [1, 1, "base"],
    **{f"logistic-chest-{k}": [1, 1, "base"] for k in _LOGISTIC},
}

_INSERTERS = ("burner-inserter", "inserter", "fast-inserter", "filter-inserter", "stack-inserter", "stack-filter-inserter", "bulk-inserter")
_TIERS = ("", "fast-", "express-", "turbo-")
# Blueprint name -> the engine entity it decodes to. Higher belt tiers and
# faster inserters keep their geometry at a lower rate, which is the layout
# prior pretraining is after.
_KNOWN: dict[str, int] = {
    **{f"{t}transport-belt": str2ent("transport_belt").value for t in _TIERS},
    **{f"{t}splitter": str2ent("splitter").value for t in _TIERS},
    **{f"{t}underground-belt": str2ent("underground_belt").value for t in _TIERS},
    **{n: str2ent("inserter").value for n in _INSERTERS},
    "long-handed-inserter": str2ent("long_handed_inserter").value,
    **{f"assembling-machine-{i}": str2ent("assembling_machine_1").value for i in (1, 2, 3)},
    **{
        n: str2ent(n.replace("-", "_")).value
        for n in (
            "small-electric-pole", "medium-electric-pole", "big-electric-pole", "substation",
            "wooden-chest", "iron-chest", "steel-chest", "stone-furnace", "steel-furnace", "electric-furnace",
            "solar-panel", "accumulator", "roboport", "radar", "small-lamp", "stone-wall",
        )
    },
    # Robots, and so what a logistic chest requests or provides, are off the
    # map: in the factory it is a steel chest.
    **{n: str2ent("steel_chest").value for k in _LOGISTIC for n in (f"{k}-chest", f"logistic-chest-{k}")},
}
_ASM = str2ent("assembling_machine_1").value
_UG = str2ent("underground_belt").value
# An underground's reach in its own tier (tiles from entrance to exit), and in
# the engine, which models yellow only (`entities.rs::UNDERGROUND_REACH`).
_UG_REACH = {f"{t}underground-belt": r for t, r in zip(_TIERS, (5, 7, 9, 11))}
_ENGINE_UG_REACH = 5
_DIR16 = {0: Direction.NORTH.value, 4: Direction.EAST.value, 8: Direction.SOUTH.value, 12: Direction.WEST.value}
_DRILL_REACH = {"electric-mining-drill": 1, "big-mining-drill": 2}
_GAME_ITEM = {it.name.replace("_", "-"): i for i, it in items.items()}
_DELTA = {Direction.NORTH.value: (0, -1), Direction.EAST.value: (1, 0), Direction.SOUTH.value: (0, 1), Direction.WEST.value: (-1, 0)}


def designs(string: str) -> Iterator[dict]:
    """Every blueprint in an exchange string, books unpacked."""
    stack = [json.loads(zlib.decompress(base64.b64decode(string.strip()[1:])))]
    while stack:
        o = stack.pop()
        if "blueprint_book" in o:
            stack.extend(o["blueprint_book"].get("blueprints") or [])
        elif isinstance(o.get("blueprint"), dict):
            yield o["blueprint"]


# `Raster.inst` of a masked entity's tiles, and of terrain no belt can go on.
_MASKED, _TERRAIN = -1, -2


@dataclass
class Raster:
    """A blueprint's bounding box as engine channels plus `inst`, each tile's
    entity instance (0 = empty, else `_MASKED` / `_TERRAIN`), so a window can
    mask the known entities its edge cuts through."""

    ent: np.ndarray
    dir: np.ndarray
    item: np.ndarray
    misc: np.ndarray
    inst: np.ndarray
    tiles: np.ndarray  # tile count of each instance
    masked: list[tuple[str, int, int, int, int]]  # name, x, y, w, h
    # What a save saw on each belt tile's lanes (item ids, -1 for one the engine lacks).
    lanes: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    # (x, y, facing) of the tile at the front of each mining drill, which it mines onto.
    drills: list[tuple[int, int, int]] = field(default_factory=list)


def rasterise(bp: dict) -> Optional[Raster]:
    """`bp` on its bounding box; None when it holds an entity of unknown size
    (a mod's), which would leave an unknowable hole, or nothing at all. The
    `[x, y, length]` row runs of `bp["unbuildable"]` (a save's terrain) mask
    whatever tiles they leave empty, and its `bp["lanes"]` become `Raster.lanes`."""
    ents = bp.get("entities") or []
    if not ents or any(e["name"] not in _SIZES for e in ents):
        return None
    # 2.0 doubled the direction enum; the major version is the top 16 bits.
    scale = 1 if (bp.get("version", 0) >> 48) >= 2 else 2
    placed = []
    for e in ents:
        name, d = e["name"], e.get("direction", 0) * scale % 16
        w, h, _ = _SIZES[name]
        if d % 4:
            w = h = max(w, h)
        elif d in (4, 12):
            w, h = h, w
        x0 = math.floor(e["position"]["x"] - w / 2 + 0.01)
        y0 = math.floor(e["position"]["y"] - h / 2 + 0.01)
        placed.append((name, d, x0, y0, w, h, e))
    min_x = min(p[2] for p in placed)
    min_y = min(p[3] for p in placed)
    W = max(p[2] + p[4] for p in placed) - min_x
    H = max(p[3] + p[5] for p in placed) - min_y
    z = [np.zeros((W, H), np.int64) for _ in range(5)]
    r = Raster(z[0], z[1], z[2], z[3], z[4], np.zeros(0, np.int64), [])
    ugs = []
    for i, (name, d, x0, y0, w, h, e) in enumerate(placed, 1):
        sl = (slice(x0 - min_x, x0 - min_x + w), slice(y0 - min_y, y0 - min_y + h))
        ent = _KNOWN.get(name)
        item = 0
        if ent == _ASM:
            recipe = str2item(e.get("recipe") or "empty")
            item = recipe.value if recipe else 0
            if item not in _ASM_RECIPE_ITEM_IDS:
                ent = None
        if "inserter" in name:
            d = (d + 8) % 16  # blueprint faces the drop tile, the engine the pickup
        if name in _DRILL_REACH and d in _DIR16:
            dx, dy = _DELTA[_DIR16[d]]
            k = _DRILL_REACH[name]
            r.drills.append((sl[0].start + w // 2 + dx * k, sl[1].start + h // 2 + dy * k, _DIR16[d]))
        if ent is None or d not in _DIR16:
            r.masked.append((name, sl[0].start, sl[1].start, w, h))
            r.inst[sl] = _MASKED
            continue
        r.ent[sl], r.item[sl], r.inst[sl] = ent, item, i
        r.dir[sl] = Direction.NONE.value if ent in UNDIRECTED_ENT_IDS else _DIR16[d]
        if ent == _UG:
            r.misc[sl] = Misc.UNDERGROUND_DOWN.value if e.get("type") == "input" else Misc.UNDERGROUND_UP.value
            ugs.append((sl[0].start, sl[1].start, name))
    _mask_repaired_undergrounds(r, ugs)
    for x, y, n in bp.get("unbuildable", ()):
        if 0 <= y - min_y < H:
            row = r.inst[max(0, x - min_x) : max(0, x - min_x + n), y - min_y]
            row[row == 0] = _TERRAIN
    for x, y, _, names in bp.get("lanes", ()):
        r.lanes.setdefault((x - min_x, y - min_y), set()).update(_GAME_ITEM.get(n, -1) for n in names.split(","))
    r.tiles = np.bincount(r.inst[r.inst > 0], minlength=len(placed) + 1)
    return r if (r.inst > 0).any() else None


def _pair(r: Raster, x: int, y: int, reach: int, same) -> Optional[tuple[int, int]]:
    """The exit an entrance at (x, y) tunnels to: the first underground in
    line that `same` accepts ends the search, pairing iff it's a same-facing
    exit."""
    d = int(r.dir[x, y])
    dx, dy = _DELTA[d]
    for k in range(1, reach + 1):
        tx, ty = x + dx * k, y + dy * k
        if not (0 <= tx < r.ent.shape[0] and 0 <= ty < r.ent.shape[1]):
            return None
        if r.ent[tx, ty] == _UG and same(tx, ty):
            return (tx, ty) if r.misc[tx, ty] == Misc.UNDERGROUND_UP.value and r.dir[tx, ty] == d else None
    return None


def _mask_repaired_undergrounds(r: Raster, ugs: list[tuple[int, int, str]]) -> None:
    """Mask each entrance (and its exit) that, collapsed to the engine's one
    short-reach tier, would pair differently than in the game: a faster tier's
    long tunnel, or tiers weaving under one another."""
    tier = {(x, y): n for x, y, n in ugs}
    entrances = [(x, y, n) for x, y, n in ugs if r.misc[x, y] == Misc.UNDERGROUND_DOWN.value]
    truth = {(x, y): _pair(r, x, y, _UG_REACH[n], lambda tx, ty, n=n: tier.get((tx, ty)) == n) for x, y, n in entrances}
    changed = True
    while changed:
        changed = False
        for x, y, _ in entrances:
            if r.inst[x, y] > 0 and _pair(r, x, y, _ENGINE_UG_REACH, lambda tx, ty: r.inst[tx, ty] > 0) != truth[(x, y)]:
                for tx, ty in filter(None, [(x, y), truth[(x, y)]]):
                    r.ent[tx, ty] = r.misc[tx, ty] = 0
                    r.inst[tx, ty] = _MASKED
                    r.masked.append((tier[(tx, ty)], tx, ty, 1, 1))
                changed = True


def _footprint(inst: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Buildable where kept or empty, unbuildable on terrain, else masked."""
    return np.select(
        [keep | (inst == 0), inst == _TERRAIN], [Footprint.BUILDABLE.value, Footprint.UNBUILDABLE.value], Footprint.MASKED.value
    )


def to_world(r: Raster) -> np.ndarray:
    """The whole of `r` as a (C, W, H) world."""
    world = np.zeros((len(Channel), *r.ent.shape), np.int64)
    for c, a in zip((Channel.ENTITIES, Channel.DIRECTION, Channel.ITEMS, Channel.MISC), (r.ent, r.dir, r.item, r.misc)):
        world[c.value] = a
    world[Channel.FOOTPRINT.value] = _footprint(r.inst, r.inst > 0)
    refresh_power(torch.from_numpy(world))
    return world


# Clockwise quarter turn and east-west mirror of the facing channel.
_TURN = np.array([0, 2, 3, 4, 1])
_MIRROR = np.array([0, 1, 4, 3, 2])


def orient(chans: list[np.ndarray], turns: int, mirror: bool) -> list[np.ndarray]:
    """(W, H) channels in `Channel` order, turned `turns` quarter turns
    clockwise, then mirrored east-west."""
    for _ in range(turns):
        chans = [np.rot90(c) for c in chans]
        chans[Channel.DIRECTION.value] = _TURN[chans[Channel.DIRECTION.value]]
    if mirror:
        chans = [c[::-1] for c in chans]
        chans[Channel.DIRECTION.value] = _MIRROR[chans[Channel.DIRECTION.value]]
    return chans


# A view trains only if known entities cover this much of it, masked ones at
# most this much, and it holds one of these: a field of solar panels or a row
# of lamps says nothing about moving items.
MIN_KNOWN, MAX_MASKED = 0.1, 0.5
_FLOW = [str2ent(n).value for n in ("transport_belt", "underground_belt", "splitter", "inserter", "long_handed_inserter", "assembling_machine_1")]
# Shortest side of a build region, where the view allows.
_MIN_REGION = 8


def cut(
    r: Raster, sl: tuple[slice, slice], min_known: float = MIN_KNOWN, max_masked: float = MAX_MASKED
) -> Optional[np.ndarray]:
    """The tiles of the known entities wholly inside the view `sl` of `r`, or
    None when the view doesn't train."""
    inst = r.inst[sl]
    inner = np.bincount(inst[inst > 0], minlength=len(r.tiles))
    keep = (inst > 0) & (inner == r.tiles)[np.maximum(inst, 0)]
    if keep.mean() < min_known or ((inst != 0) & ~keep).mean() > max_masked or not np.isin(r.ent[sl][keep], _FLOW).any():
        return None
    return keep


@dataclass
class Frame:
    """`window`'s random choices: the view of the blueprint the grid shows
    (x, y, w, h), the build region inside it (bx, by, bw, bh, from the view's
    corner), where the view sits in the grid (ox, oy), then the turn and
    mirror of the whole grid."""

    x: int
    y: int
    w: int
    h: int
    bx: int
    by: int
    bw: int
    bh: int
    ox: int
    oy: int
    turns: int
    mirror: bool


def frame(r: Raster, size: int, rng: random.Random) -> Optional[Frame]:
    """A random `Frame` whose build region holds a belt, inserter or
    assembler; None when `r` has none."""
    W, H = r.ent.shape
    w, h = min(W, size), min(H, size)
    fx, fy = np.nonzero((r.inst > 0) & np.isin(r.ent, _FLOW))
    if not len(fx):
        return None
    k = rng.randrange(len(fx))
    x = min(max(0, fx[k] - rng.randrange(w)), W - w)
    y = min(max(0, fy[k] - rng.randrange(h)), H - h)
    px, py = fx[k] - x, fy[k] - y
    bw, bh = rng.randint(min(_MIN_REGION, w), w), rng.randint(min(_MIN_REGION, h), h)
    bx = rng.randint(max(0, px - bw + 1), min(px, w - bw))
    by = rng.randint(max(0, py - bh + 1), min(py, h - bh))
    ox, oy = rng.randint(0, size - w), rng.randint(0, size - h)
    return Frame(int(x), int(y), w, h, int(bx), int(by), bw, bh, ox, oy, rng.randrange(4), rng.random() < 0.5)


def place(
    r: Raster, f: Frame, size: int, region: bool = False, min_known: float = MIN_KNOWN, max_masked: float = MAX_MASKED
) -> Optional[np.ndarray]:
    """The size × size (C, W, H) world of `f`: the build region is buildable,
    with the human's layout as its target; the rest of the view is the
    unbuildable context it connects to, its known entities kept. Unmodelled
    entities and those the view's edge cuts are masked. Past the blueprint's
    edge is unbuildable, but for a one-tile masked band, so the design's own
    inputs and outputs lead somewhere unknown; past a save `region`'s edge
    is more world, so masked. None when the view doesn't train (`cut`)."""
    sl = (slice(f.x, f.x + f.w), slice(f.y, f.y + f.h))
    keep = cut(r, sl, min_known, max_masked)
    if keep is None:
        return None
    inst = r.inst[sl]
    inside = np.zeros_like(keep)
    inside[f.bx : f.bx + f.bw, f.by : f.by + f.bh] = True
    # An entity straddling the build region's edge is context: never half a machine to build.
    inner = np.bincount(inst[keep & inside], minlength=len(r.tiles))
    build = keep & (inner == r.tiles)[np.maximum(inst, 0)]
    footprint = np.select(
        [build | (inside & (inst == 0)), keep | (inst == 0) | (inst == _TERRAIN)],
        [Footprint.BUILDABLE.value, Footprint.UNBUILDABLE.value],
        Footprint.MASKED.value,
    )
    W, H = r.ent.shape
    gx, gy = np.arange(size)[:, None] - f.ox, np.arange(size)[None, :] - f.oy
    past = ((gx < 0) & (f.x == 0)) | ((gx >= f.w) & (f.x + f.w == W)) | ((gy < 0) & (f.y == 0)) | ((gy >= f.h) & (f.y + f.h == H))
    band = (gx >= -1) & (gx <= f.w) & (gy >= -1) & (gy <= f.h)
    world = np.zeros((len(Channel), size, size), np.int64)
    world[Channel.FOOTPRINT.value] = np.where(past & ~band & (not region), Footprint.UNBUILDABLE.value, Footprint.MASKED.value)
    view = (slice(f.ox, f.ox + f.w), slice(f.oy, f.oy + f.h))
    for c, a in zip((Channel.ENTITIES, Channel.DIRECTION, Channel.ITEMS, Channel.MISC), (r.ent, r.dir, r.item, r.misc)):
        world[c.value][view] = np.where(keep, a[sl], 0)
    world[Channel.FOOTPRINT.value][view] = footprint
    world = np.ascontiguousarray(np.stack(orient(list(world), f.turns, f.mirror)))
    refresh_power(torch.from_numpy(world))
    return world


def window(
    r: Raster, size: int, rng: random.Random, region: bool = False, min_known: float = MIN_KNOWN, max_masked: float = MAX_MASKED
) -> Optional[np.ndarray]:
    """A random training world of `r` (`frame`, `place`); `region` for a save
    region rather than a blueprint."""
    f = frame(r, size, rng)
    return None if f is None else place(r, f, size, region, min_known, max_masked)


_BELT, _SPLITTER = str2ent("transport_belt").value, str2ent("splitter").value
_FLOWING = (_BELT, _UG, _SPLITTER)
_REACH = {str2ent("inserter").value: 1, str2ent("long_handed_inserter").value: 2}
_CHESTS = [str2ent(n).value for n in ("wooden_chest", "iron_chest", "steel_chest")]
_FURNACES = [str2ent(n).value for n in ("stone_furnace", "steel_furnace", "electric_furnace")]
_SOURCE, _SINK = str2ent("source").value, str2ent("sink").value
_ITEM = {it.name: i for i, it in items.items()}
_RECIPES = {_ITEM[k]: v for k, v in factorion_rs.py_recipes().items()}
_SMELTS = [_ITEM[n] for n in ("iron_ore", "copper_ore", "stone", "iron_plate", "coal")]
_SMELTED = [_ITEM[n] for n in ("iron_plate", "copper_plate", "steel_plate", "stone_brick")]
_BELT_ITEMS = sorted(set(items) - {0, _SOURCE, _SINK})
_CH = [c.value for c in (Channel.ENTITIES, Channel.DIRECTION, Channel.ITEMS, Channel.MISC)]
# (x, y, entity, facing, item) of a grid tile a functional window changes.
Edit = tuple[int, int, int, int, int]


def _interior(r: Raster, x: int, y: int, size: int) -> np.ndarray:
    """The size × size world of the human layout in `r`'s (size - 2)² interior
    at (x, y), buildable but where the edge cuts an entity, inside a ring of
    unbuildable tiles."""
    n = size - 2
    sl = (slice(x, x + n), slice(y, y + n))
    inst = r.inst[sl]
    inner = np.bincount(inst[inst > 0], minlength=len(r.tiles))
    whole = (inst > 0) & (inner == r.tiles)[np.maximum(inst, 0)]
    world = np.zeros((len(Channel), size, size), np.int64)
    for c, a in zip(_CH, (r.ent, r.dir, r.item, r.misc)):
        world[c, 1:-1, 1:-1] = np.where(whole, a[sl], 0)
    world[Channel.FOOTPRINT.value] = Footprint.UNBUILDABLE.value
    world[Channel.FOOTPRINT.value, 1:-1, 1:-1] = np.where(whole | (inst == 0), Footprint.BUILDABLE.value, Footprint.UNBUILDABLE.value)
    return world


def paste(r: Raster, x: int, y: int, size: int, edits: list[Edit]) -> np.ndarray:
    """The functional window of `r` at (x, y) that `edits` made (`edit`)."""
    world = _interior(r, x, y, size)
    for gx, gy, ent, d, item in edits:
        world[_CH, gx, gy] = ent, d, item, 0
    refresh_power(torch.from_numpy(world))
    return world


def _machine_items(world: np.ndarray) -> tuple[set[int], set[int]]:
    """(ingredients, products) of the world's assemblers and furnaces."""
    ent, item = world[Channel.ENTITIES.value], world[Channel.ITEMS.value]
    ins, outs = set(), set()
    for v in np.unique(item[ent == _ASM]):
        if (rec := _RECIPES.get(int(v))) is not None:
            ins |= {_ITEM[n] for n in rec["consumes"]}
            outs |= {_ITEM[n] for n in rec["produces"]}
    if np.isin(ent, _FURNACES).any():
        ins |= set(_SMELTS)
        outs |= set(_SMELTED)
    return ins, outs


def deliveries(world: np.ndarray, products: set[int]) -> tuple[float, float]:
    """(items/s reaching the sinks, of which `products`) of a (C, W, H) world."""
    got = factorion_rs.py_sink_deliveries(np.ascontiguousarray(world.transpose(1, 2, 0)))
    names = {items[i].name for i in products}
    return sum(rate for *_, rate in got), sum(rate for _, _, item, rate in got if item in names)


def _isolated(world: np.ndarray, x: int, y: int, d: int) -> bool:
    """Whether the belt a source at (x, y) facing `d` starts is one nothing else
    feeds and no inserter touches, so what it carries can't matter."""
    ent, dirs = world[Channel.ENTITIES.value], world[Channel.DIRECTION.value]
    W, H = ent.shape
    chain, (cx, cy) = [], (x + _DELTA[d][0], y + _DELTA[d][1])
    while 0 <= cx < W and 0 <= cy < H and ent[cx, cy] in (_BELT, _UG) and (cx, cy) not in chain:
        chain.append((cx, cy))
        dx, dy = _DELTA.get(int(dirs[cx, cy]), (0, 0))
        cx, cy = cx + dx, cy + dy
    on = set(chain)
    for ix, iy in np.argwhere(np.isin(ent, list(_REACH))):
        k = _REACH[int(ent[ix, iy])]
        dx, dy = _DELTA.get(int(dirs[ix, iy]), (0, 0))
        if (ix - dx * k, iy - dy * k) in on or (ix + dx * k, iy + dy * k) in on:
            return False
    for i, (cx, cy) in enumerate(chain):
        for d2, (dx, dy) in _DELTA.items():
            u = (cx - dx, cy - dy)
            fed = 0 <= u[0] < W and 0 <= u[1] < H and ent[u] in (*_FLOWING, _SOURCE) and dirs[u] == d2
            if fed and u != (x, y) and (i == 0 or u != chain[i - 1]):
                return False
    return bool(chain)


def edit(r: Raster, x: int, y: int, size: int, rng: random.Random) -> list[Edit]:
    """What turns `r`'s (size - 2)² interior at (x, y) into a working factory
    (`paste`): every flow across the interior's edge, or into an entity the
    engine lacks or the edge cuts, becomes a source or sink on that tile, of
    what the save saw on the belt (`Raster.lanes`), as does a tunnel the edge
    cuts, a belt that starts or ends at nothing, a mining drill's output and
    an inserter's hand over nothing (a wagon). A chest that inserters fill
    and none empty, a store for later, becomes a sink, and one they only
    empty (robots fill it) a source. Where no label settles a marker's item
    it is the one that brings the most of the machines' products to the
    sinks. Medium poles go wherever the human's leave something unpowered
    and there is room."""
    n = size - 2
    W, H = r.ent.shape
    world = _interior(r, x, y, size)
    base = world.copy()
    whole = world[Channel.FOOTPRINT.value, 1:-1, 1:-1].astype(bool) & (r.inst[x : x + n, y : y + n] > 0)

    def inside(t):
        return x <= t[0] < x + n and y <= t[1] < y + n

    def ring(t):
        return x - 1 <= t[0] <= x + n and y - 1 <= t[1] <= y + n and not inside(t)

    def on(t):
        return 0 <= t[0] < W and 0 <= t[1] < H

    def kept(t):
        return inside(t) and whole[t[0] - x, t[1] - y]

    def flowing(t):
        return on(t) and r.ent[t] in _FLOWING and int(r.dir[t]) in _DELTA

    def portable(t):
        """Where flow leaves the modelled factory: the ring, or a masked or cut entity."""
        return ring(t) or (inside(t) and r.inst[t] != 0 and not kept(t) and r.inst[t] != _TERRAIN)

    def fed(t):
        return any(kept(u) and r.ent[u] in _FLOWING and r.dir[u] == d for d, (dx, dy) in _DELTA.items() for u in [(t[0] - dx, t[1] - dy)])

    def label(t):
        """What the save saw on `t`, else on the nearest labelled tile along its belt."""
        if t in r.lanes:
            return r.lanes[t]
        for sign in (1, -1):
            p = t
            for _ in range(40):
                if not flowing(p):
                    break
                dx, dy = _DELTA[int(r.dir[p])]
                p = (p[0] + sign * dx, p[1] + sign * dy)
                if p in r.lanes:
                    return r.lanes[p]
        return set()

    def loose(t):
        """Where an inserter's hand meets what the window lacks: an entity past
        its edge, masked or cut, or nothing at all (a wagon's track)."""
        return (portable(t) and r.inst[t] != 0) or ((inside(t) or ring(t)) and r.inst[t] == 0)

    def away(t, d):
        """A facing for a source on `t` that feeds no belt (it only feeds inserters), `d` if it can."""
        for e in (d, *_DELTA):
            f = (t[0] + _DELTA[e][0], t[1] + _DELTA[e][1])
            if not (kept(f) and r.ent[f] in _FLOWING):
                return e
        return d

    ports: dict[tuple[int, int], tuple[int, int, set[int]]] = {}

    def port(t, kind, d, its):
        ports.setdefault(t, (kind, d, its))

    kept_tiles = [(x + int(i), y + int(j)) for i, j in zip(*np.nonzero(whole))]
    for t in kept_tiles:
        ent, d = int(r.ent[t]), int(r.dir[t])
        if d not in _DELTA:
            continue
        dx, dy = _DELTA[d]
        aft, fore = (t[0] - dx, t[1] - dy), (t[0] + dx, t[1] + dy)
        if ent in _FLOWING:
            if ring(aft) and flowing(aft) and r.dir[aft] == d:
                port(aft, _SOURCE, d, label(aft) or label(t))
            elif portable(aft) and not ring(aft) and not fed(t):
                port(aft, _SOURCE, d, label(t))
            if (ring(fore) and flowing(fore)) or (portable(fore) and not ring(fore)):
                port(fore, _SINK, d, label(t))
        if ent in _REACH:
            k = _REACH[ent]
            pick, drop = (t[0] - dx * k, t[1] - dy * k), (t[0] + dx * k, t[1] + dy * k)
            if loose(pick):
                port(pick, _SOURCE, away(pick, d), r.lanes.get(pick) or label(drop))
            if loose(drop):
                port(drop, _SINK, d, r.lanes.get(drop) or label(pick))
    picks = set()
    for t in kept_tiles:
        if r.ent[t] in _REACH and int(r.dir[t]) in _DELTA:
            dx, dy = _DELTA[int(r.dir[t])]
            picks.add((t[0] - dx * _REACH[int(r.ent[t])], t[1] - dy * _REACH[int(r.ent[t])]))

    def upstream(t):
        """The belt tiles that lead to `t`, straight on before side feeds."""
        line = [t]
        while len(line) < 4 * n:
            p = line[-1]
            fs = [u for d, (dx, dy) in _DELTA.items() for u in [(p[0] - dx, p[1] - dy)] if kept(u) and r.ent[u] in (_BELT, _UG) and r.dir[u] == d]
            fs = [u for u in fs if u not in line]
            if not fs:
                break
            dx, dy = _DELTA[int(r.dir[p])]
            line.append((p[0] - dx, p[1] - dy) if (p[0] - dx, p[1] - dy) in fs else fs[0])
        return set(line)

    for fx, fy, d in r.drills:
        dx, dy = _DELTA[d]
        out = (fx + dx, fy + dy)
        if (inside((fx, fy)) or ring((fx, fy))) and kept(out) and r.ent[out] in (_BELT, _UG):
            port((fx, fy), _SOURCE, d, label(out))
    for t in kept_tiles:
        ent, d = int(r.ent[t]), int(r.dir[t])
        if d not in _DELTA:
            continue
        dx, dy = _DELTA[d]
        if ent == _UG:
            down = r.misc[t] == Misc.UNDERGROUND_DOWN.value
            step = 1 if down else -1  # an entrance looks ahead for its exit, an exit behind
            for k in range(1, _ENGINE_UG_REACH + 1):
                p = (t[0] + step * dx * k, t[1] + step * dy * k)
                if not inside(p):  # its partner, if the region has it at all, is past the edge
                    port(t, _SINK if down else _SOURCE, d, label(t))
                    break
                if r.ent[p] == _UG and r.dir[p] == d and r.misc[p] != r.misc[t]:
                    break
        fore, aft = (t[0] + dx, t[1] + dy), (t[0] - dx, t[1] - dy)
        if ent == _BELT and on(fore) and r.inst[fore] == 0 and not picks & upstream(t):
            port(t, _SINK, d, label(t))
        if ent in (_BELT, _SPLITTER) and not fed(t) and label(t) and on(aft) and r.inst[aft] == 0:
            port(t, _SOURCE, d, label(t))
    for t in kept_tiles:
        if r.ent[t] not in _CHESTS:
            continue
        fills, empties, filled_with, emptied_onto = 0, 0, set(), set()
        for p in [(t[0] + a, t[1] + b) for a in range(-2, 3) for b in range(-2, 3) if (a == 0) != (b == 0)]:
            if kept(p) and r.ent[p] in _REACH and int(r.dir[p]) in _DELTA:
                k = _REACH[int(r.ent[p])]
                dx, dy = _DELTA[int(r.dir[p])]
                if (p[0] + dx * k, p[1] + dy * k) == t:
                    fills += 1
                    filled_with |= r.lanes.get((p[0] - dx * k, p[1] - dy * k), set())
                elif (p[0] - dx * k, p[1] - dy * k) == t:
                    empties += 1
                    emptied_onto |= r.lanes.get((p[0] + dx * k, p[1] + dy * k), set())
        if fills and not empties:
            port(t, _SINK, Direction.NORTH.value, filled_with)
        elif empties and not fills:  # robots fill it
            port(t, _SOURCE, away(t, Direction.NORTH.value), emptied_onto)
    perimeter = [(a, b) for a in range(x - 1, x + n + 1) for b in (y - 1, y + n)] + [(a, b) for a in (x - 1, x + n) for b in range(y, y + n)]
    for t in perimeter:
        if flowing(t):
            d = int(r.dir[t])
            p = (t[0] + _DELTA[d][0], t[1] + _DELTA[d][1])
            if kept(p) and r.ent[p] in (_BELT, _UG) and int(r.dir[p]) not in (d, (d + 1) % 4 + 1):
                port(t, _SOURCE, d, label(t))

    def put(t, kind, d, item):
        world[_CH, t[0] - x + 1, t[1] - y + 1] = kind, d, item, 0

    pending = []
    for t, (kind, d, its) in ports.items():
        known = its - {-1}
        if len(known) == 1:
            put(t, kind, d, next(iter(known)))
        else:
            pending.append((t, kind, d, known))
    ins, outs = _machine_items(world)
    seen = {i for t, its in r.lanes.items() if inside(t) or ring(t) for i in its} - {-1}
    for t, kind, d, known in sorted(pending, key=lambda p: p[1] != _SOURCE):
        sources = set(world[Channel.ITEMS.value][world[Channel.ENTITIES.value] == _SOURCE].tolist())
        options = known or (ins if kind == _SOURCE else outs | ins | sources)
        if not options and kind == _SOURCE and _isolated(world, t[0] - x + 1, t[1] - y + 1, d):
            options = {rng.choice(_BELT_ITEMS)}
        if not options:
            commonest = np.bincount([i for its in r.lanes.values() for i in its if i > 0] or [_ITEM["iron_plate"]]).argmax()
            options = seen or {int(commonest)}

        def value(item):
            put(t, kind, d, item)
            total, crafted = deliveries(world, outs)
            return crafted + 1e-3 * total

        put(t, kind, d, max(sorted(options), key=value) if len(options) > 1 else next(iter(options)))
    powered = factorion_rs.py_place_poles(np.ascontiguousarray(world.transpose(1, 2, 0)))
    if powered is not None:
        world = powered.transpose(2, 0, 1)
    ent, d, item = world[_CH[:3]]
    return [(int(gx), int(gy), int(ent[gx, gy]), int(d[gx, gy]), int(item[gx, gy])) for gx, gy in np.argwhere((world[_CH] != base[_CH]).any(0))]


# Designs windowed at once: consecutive windows come from different designs.
_POOL = 64


def download(repo: str, folder: str = "designs") -> list[Path]:
    """The corpus's `folder` shards (`designs`, `corpus/build.py`, or
    `windows`, `corpus/windows.py`), fetched from the Hugging Face dataset
    `repo` into the local cache."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=f"{folder}/*"))
    return sorted(root.glob(f"{folder}/*.parquet"))


def views(r: Raster, size: int) -> int:
    """The size × size tiles of `r`'s bounding box (no overlap) that train:
    its windows per epoch."""
    W, H = r.ent.shape
    return sum(cut(r, (slice(x, x + size), slice(y, y + size))) is not None for x in range(0, W, size) for y in range(0, H, size))


def held_out(digest: str, every: int = 100) -> bool:
    """Whether a design (by its layout hash) or a save (`corpus.windows`) is
    in the 1/`every` kept out of training, the same on every machine."""
    return int(digest[:8], 16) % every == 0


def _mine(layout_hash: str, worker: int, workers: int) -> bool:
    return int(layout_hash[8:16], 16) % workers == worker


def _strings(files: list[Path], rng: random.Random, keep) -> Iterator[tuple[str, dict]]:
    """The exchange string and metadata of each design of `files` that `keep`
    accepts, in a random order, epoch after epoch."""
    import pyarrow.parquet as pq

    while True:
        for f in rng.sample(files, len(files)):
            pf = pq.ParquetFile(f)
            meta = pf.read(columns=["source", "layout_hash", "modded"]).to_pylist()
            # Only the kept strings: a shard's are hundreds of MB, and every worker reads every shard.
            strings = (s for b in pf.iter_batches(1024, columns=["string"]) for s in b.column(0).to_pylist())
            rows = [(s, m) for s, m in zip(strings, meta) if keep(m)]
            rng.shuffle(rows)
            yield from rows


def stream(
    files: list[Path], size: int, rng: random.Random, worker: int = 0, workers: int = 1, val: bool = False
) -> Iterator[np.ndarray]:
    """Windows of the corpus's unmodded designs (the held-out ones if `val`),
    in epochs: each design gives `views` windows per epoch, interleaved with
    `_POOL` - 1 others. Worker `worker` of `workers` takes its own share of
    the designs, so the workers together see each design once an epoch."""

    def rasters() -> Iterator[tuple[Raster, bool, int]]:
        def keep(m: dict) -> bool:
            return not m["modded"] and held_out(m["layout_hash"]) == val and _mine(m["layout_hash"], worker, workers)

        for string, m in _strings(files, rng, keep):
            r = rasterise(next(designs(string)))
            if r is not None and (n := views(r, size)):
                yield r, m["source"].startswith("save:"), n

    source = rasters()
    pool = list(itertools.islice(source, _POOL))
    # A view `views` counts can fail as a random window: give up on a design after 4x its quota of tries.
    left = [[n, 4 * n] for *_, n in pool]
    while True:
        i = rng.randrange(len(pool))
        r, region, _ = pool[i]
        w = window(r, size, rng, region)
        left[i][1] -= 1
        if w is not None:
            left[i][0] -= 1
            yield w
        if min(left[i]) == 0:
            pool[i] = next(source)
            left[i] = [pool[i][2], 4 * pool[i][2]]


def functional(
    files: list[Path], window_files: list[Path], size: int, rng: random.Random, worker: int = 0, workers: int = 1, val: bool = False
) -> Iterator[np.ndarray]:
    """The functional windows (`corpus.windows`) of `files`' designs, each
    turned and mirrored at random, in epochs: a design's windows in a random
    order, interleaved with `_POOL` - 1 other designs'. `val` takes those of
    the 5% of saves held out, else worker `worker` of `workers` its own share
    of the rest. Only windows whose machines make something: the rest are
    belt routing, which the generated lessons already teach."""
    import pyarrow.parquet as pq

    mine: dict[str, list[tuple[int, int, list[Edit]]]] = {}
    for f in window_files:
        t = pq.read_table(f, columns=["layout_hash", "save", "size", "x", "y", "edits", "crafted"]).to_pylist()
        for w in t:
            if w["size"] == size and w["crafted"] > 0 and held_out(w["save"], 20) == val and (val or _mine(w["layout_hash"], worker, workers)):
                e = w["edits"]
                mine.setdefault(w["layout_hash"], []).append((w["x"], w["y"], [tuple(e[i : i + 5]) for i in range(0, len(e), 5)]))
    if not mine:
        raise ValueError(f"no {size}-wide functional windows to stream")

    def sources() -> Iterator[tuple[Raster, list]]:
        for string, m in _strings(files, rng, lambda m: m["layout_hash"] in mine):
            r = rasterise(next(designs(string)))
            if r is not None:
                yield r, rng.sample(mine[m["layout_hash"]], len(mine[m["layout_hash"]]))

    source = sources()
    pool = list(itertools.islice(source, _POOL))
    while True:
        i = rng.randrange(len(pool))
        r, left = pool[i]
        x, y, edits = left.pop()
        yield np.ascontiguousarray(np.stack(orient(list(paste(r, x, y, size, edits)), rng.randrange(4), rng.random() < 0.5)))
        if not left:
            pool[i] = next(source)
