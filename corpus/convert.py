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
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch

from factorion import UNDIRECTED_ENT_IDS, Channel, Direction, Footprint, Misc, refresh_power, str2ent, str2item
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


def rasterise(bp: dict) -> Optional[Raster]:
    """`bp` on its bounding box; None when it holds an entity of unknown size
    (a mod's), which would leave an unknowable hole, or nothing at all. The
    `[x, y, length]` row runs of `bp["unbuildable"]` (a save's terrain) mask
    whatever tiles they leave empty."""
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


# Designs windowed at once: consecutive windows come from different designs.
_POOL = 64


def download(repo: str) -> list[Path]:
    """The corpus's design shards, fetched from the Hugging Face dataset `repo`
    (`corpus/build.py`) into the local cache."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns="designs/*"))
    return sorted(root.glob("designs/*.parquet"))


def views(r: Raster, size: int) -> int:
    """The size × size tiles of `r`'s bounding box (no overlap) that train:
    its windows per epoch."""
    W, H = r.ent.shape
    return sum(cut(r, (slice(x, x + size), slice(y, y + size))) is not None for x in range(0, W, size) for y in range(0, H, size))


def held_out(layout_hash: str) -> bool:
    """The 1% of designs kept out of training, by layout so the split is the
    same on every machine."""
    return int(layout_hash[:8], 16) % 100 == 0


def stream(
    files: list[Path], size: int, rng: random.Random, worker: int = 0, workers: int = 1, val: bool = False
) -> Iterator[np.ndarray]:
    """Windows of the corpus's unmodded designs (the held-out ones if `val`),
    in epochs: each design gives `views` windows per epoch, interleaved with
    `_POOL` - 1 others. Worker `worker` of `workers` takes its own share of
    the designs, so the workers together see each design once an epoch."""
    import pyarrow.parquet as pq

    def rasters() -> Iterator[tuple[Raster, bool, int]]:
        while True:
            for f in rng.sample(files, len(files)):
                pf = pq.ParquetFile(f)
                meta = pf.read(columns=["source", "layout_hash", "modded"]).to_pylist()
                mine = [
                    not m["modded"] and held_out(m["layout_hash"]) == val and int(m["layout_hash"][8:16], 16) % workers == worker
                    for m in meta
                ]
                # Only this worker's strings: a shard's are hundreds of MB, and every worker reads every shard.
                strings = (s for b in pf.iter_batches(1024, columns=["string"]) for s in b.column(0).to_pylist())
                rows = [(s, m["source"]) for s, m, ok in zip(strings, meta, mine) if ok]
                rng.shuffle(rows)
                for string, src in rows:
                    r = rasterise(next(designs(string)))
                    if r is not None and (n := views(r, size)):
                        yield r, src.startswith("save:"), n

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
