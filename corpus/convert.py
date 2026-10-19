"""Human blueprints -> factorion training windows: the HUMAN_BLUEPRINTS lesson.

The corpus stores raw exchange strings so that every conversion decision lives
here and re-runs at load time: teaching the engine a new entity only means
editing `_KNOWN`. A blueprint is rasterised once (`rasterise`), then any number
of windows are cut from it (`window`). Entities the engine doesn't model are
*masked*: their tiles become unbuildable, so the denoiser sees them as walls
and no loss reaches them.
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

from factorion import POLE_ENT_IDS, Channel, Direction, Footprint, Misc, refresh_power, str2ent, str2item
from denoise import _ASM_RECIPE_ITEM_IDS

# Placement footprint of every base / Space Age / Quality / Elevated Rails
# entity, dumped from the headless game's prototypes (`corpus/README.md`).
_SIZES: dict[str, list] = json.loads((Path(__file__).parent / "entity_sizes.json").read_text())
# 1.1 names whose 2.0 prototype was renamed.
_SIZES |= {
    "curved-rail": _SIZES["legacy-curved-rail"],
    "filter-inserter": [1, 1, "base"],
    "stack-filter-inserter": [1, 1, "base"],
    **{f"logistic-chest-{k}": [1, 1, "base"] for k in ("active-provider", "passive-provider", "storage", "requester", "buffer")},
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
    **{n: str2ent(n.replace("-", "_")).value for n in ("small-electric-pole", "medium-electric-pole", "substation")},
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


@dataclass
class Raster:
    """A blueprint's bounding box as engine channels plus `inst`, each tile's
    entity instance (0 = empty, -1 = masked), so a window can mask the known
    entities its edge cuts through."""

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
            r.inst[sl] = -1
            continue
        r.ent[sl], r.item[sl], r.inst[sl] = ent, item, i
        r.dir[sl] = Direction.NONE.value if ent in POLE_ENT_IDS else _DIR16[d]
        if ent == _UG:
            r.misc[sl] = Misc.UNDERGROUND_DOWN.value if e.get("type") == "input" else Misc.UNDERGROUND_UP.value
            ugs.append((sl[0].start, sl[1].start, name))
    _mask_repaired_undergrounds(r, ugs)
    for x, y, n in bp.get("unbuildable", ()):
        if 0 <= y - min_y < H:
            row = r.inst[max(0, x - min_x) : max(0, x - min_x + n), y - min_y]
            row[row == 0] = -1
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
                    r.inst[tx, ty] = -1
                    r.masked.append((tier[(tx, ty)], tx, ty, 1, 1))
                changed = True


def to_world(r: Raster) -> np.ndarray:
    """The whole of `r` as a (C, W, H) world, masked tiles unbuildable."""
    world = np.zeros((len(Channel), *r.ent.shape), np.int64)
    for c, a in zip((Channel.ENTITIES, Channel.DIRECTION, Channel.ITEMS, Channel.MISC), (r.ent, r.dir, r.item, r.misc)):
        world[c.value] = a
    world[Channel.FOOTPRINT.value] = r.inst >= 0
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


# A window trains only if known entities cover this much of it, and masked
# ones at most this much.
MIN_KNOWN, MAX_MASKED = 0.1, 0.5


def cut(r: Raster, sl: tuple[slice, slice]) -> tuple[np.ndarray, np.ndarray]:
    """(keep, mask) over the window `sl` of `r`: the tiles of known entities
    wholly inside it, and every other occupied tile."""
    inst = r.inst[sl]
    inner = np.bincount(inst[inst > 0], minlength=len(r.tiles))
    keep = (inst > 0) & (inner == r.tiles)[np.maximum(inst, 0)]
    return keep, (inst != 0) & ~keep


def window(
    r: Raster, size: int, rng: random.Random, min_side: int = 6, min_known: float = MIN_KNOWN, max_masked: float = MAX_MASKED
) -> Optional[np.ndarray]:
    """A random w × h window of `r` (min_side ≤ w, h ≤ size, clipped to the
    blueprint), randomly turned and mirrored, placed anywhere in a size × size
    (C, W, H) world. Everything outside the window, every masked tile and
    every entity the window's edge cuts through is unbuildable. None unless
    known entities cover at least `min_known` of the window and masked tiles
    at most `max_masked`."""
    W, H = r.ent.shape
    w, h = min(W, rng.randint(min_side, size)), min(H, rng.randint(min_side, size))
    # Centre the window on a random known tile so sparse blueprints still yield.
    xs, ys = np.nonzero(r.inst > 0)
    k = rng.randrange(len(xs))
    x0 = min(max(0, xs[k] - rng.randrange(w)), W - w)
    y0 = min(max(0, ys[k] - rng.randrange(h)), H - h)
    sl = (slice(x0, x0 + w), slice(y0, y0 + h))
    keep, mask = cut(r, sl)
    if keep.mean() < min_known or mask.mean() > max_masked:
        return None
    chans = [np.where(keep, a[sl], 0) for a in (r.ent, r.dir, r.item, r.misc)]
    chans.append(np.where(mask, Footprint.UNAVAILABLE.value, Footprint.AVAILABLE.value))
    chans = orient(chans, rng.randrange(4), rng.random() < 0.5)
    w, h = chans[0].shape
    world = np.zeros((len(Channel), size, size), np.int64)
    ox, oy = rng.randint(0, size - w), rng.randint(0, size - h)
    for c, a in zip((Channel.ENTITIES, Channel.DIRECTION, Channel.ITEMS, Channel.MISC, Channel.FOOTPRINT), chans):
        world[c.value, ox : ox + w, oy : oy + h] = a
    refresh_power(torch.from_numpy(world))
    return world



# Designs windowed at once: consecutive windows come from different designs.
_POOL = 64


def download(repo: str) -> list[Path]:
    """The corpus's design shards, fetched from the Hugging Face dataset `repo`
    (`corpus/build.py`) into the local cache."""
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(repo, repo_type="dataset", allow_patterns="designs/*"))
    return sorted(root.glob("designs/*.parquet"))


def stream(files: list[Path], size: int, rng: random.Random) -> Iterator[np.ndarray]:
    """Windows of the corpus's unmodded designs, forever. Each design gives
    about one window per quarter grid of its area, interleaved with
    `_POOL` - 1 others."""
    import pyarrow.parquet as pq

    def rasters() -> Iterator[Raster]:
        while True:
            for f in rng.sample(files, len(files)):
                strings = pq.read_table(f, columns=["string"], filters=[("modded", "==", False)]).column("string").to_pylist()
                rng.shuffle(strings)
                for s in strings:
                    r = rasterise(next(designs(s)))
                    if r is not None:
                        yield r

    source = rasters()
    pool = list(itertools.islice(source, _POOL))
    left = [1 + r.inst.size * 4 // size**2 for r in pool]
    while True:
        i = rng.randrange(len(pool))
        w = window(pool[i], size, rng)
        left[i] -= 1
        if left[i] == 0:
            pool[i] = next(source)
            left[i] = 1 + pool[i].inst.size * 4 // size**2
        if w is not None:
            yield w
