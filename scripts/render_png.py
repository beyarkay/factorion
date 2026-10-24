"""Colour-block PNG renderer for factories: a clearer picture than the ASCII
`render_factory`, for looking at what a lesson or a model build actually is.

Belts join where one feeds the next, undergrounds fade into a dotted tunnel,
splitters cross every input to every output, assemblers, sources and sinks
carry their item's Factorio icon, inserters point the way items move
(long-handed ones mark their two-tile reach), small and medium poles are
lightning bolts over the tiles they power (the 2x2 poles show their icons),
chests, furnaces and inert entities are boxes with their own icon, and
unbuildable tiles are hatched. Entities the engine doesn't model (masked out
of a human blueprint, `corpus.convert`) are boxes labelled with their
Factorio name.

Usage:
    uv run python scripts/render_png.py --kind OPPOSITE_SIDES_1IN --seed 3 --out factory.png
    # or, from Python: save(world_CWH, "factory.png", title, crop=(x0, x1, y0, y1))
"""

import functools
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import tyro  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Patch, Polygon, Rectangle  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from denoise import part_map  # noqa: E402
from factorion import DIR_TO_DELTA, Channel, Direction, Footprint, LessonKind, Misc, build_factory, items, refresh_power, str2ent  # noqa: E402

ICONS = Path(__file__).resolve().parent.parent / "factorio-icons"
ID = {n: str2ent(n).value for n in (
    "transport_belt", "underground_belt", "splitter", "inserter", "long_handed_inserter",
    "assembling_machine_1", "small_electric_pole", "medium_electric_pole", "substation", "big_electric_pole", "source", "sink",
)}
CHESTS = ("wooden_chest", "iron_chest", "steel_chest")
FURNACES = ("stone_furnace", "steel_furnace", "electric_furnace")
INERT = ("solar_panel", "accumulator", "roboport", "radar", "small_lamp", "stone_wall")
BELTISH = {ID["transport_belt"], ID["underground_belt"], ID["splitter"]}
C = dict(
    unbuildable="#4a4a4a", masked="#d4d4d4", empty="#fbfbf8", powered="#d3e9fb", grid="#dcdcdc", belt="#e0b03a",
    belt_dark="#6b4e0c", splitter="#55595e", asm="#c9c3b6", asm_edge="#5d5a52", inserter="#f2c200",
    inserter_edge="#7a6200", long="#e07b00", pole="#1f6fd1", source="#1e9e4a", sink="#7b3fa0",
    chest="#c7a26b", furnace="#e3a07a", inert="#b4c0cc", unmodelled="#c98b8b", unmodelled_edge="#7a2e2e",
)
BOXED = {str2ent(n).value: C[kind] for kind, names in (("chest", CHESTS), ("furnace", FURNACES), ("inert", INERT)) for n in names}
BELT_W = 0.8
BOLT = np.array([(0.58, 0.05), (0.25, 0.55), (0.47, 0.55), (0.38, 0.95), (0.75, 0.4), (0.53, 0.4), (0.68, 0.05)])


@functools.cache
def _icon(item_id: int):
    path = ICONS / f"{items[int(item_id)].name}.png"
    return mpimg.imread(path) if path.exists() else None


def draw(world_CWH, ax, title="", crop=None, masked=()):
    """Draw a (C, W, H) world onto `ax`. `crop` = (x0, x1, y0, y1), exclusive
    ends; everything is drawn and the axes clip it, so an entity straddling the
    crop shows its visible part. `masked` = (name, x, y, w, h) boxes of the
    unmodelled entities."""
    w = torch.as_tensor(world_CWH).to("cpu", torch.int64, copy=True)
    refresh_power(w)
    w = w.numpy()
    ent, dirs, itm, misc = w[Channel.ENTITIES.value], w[Channel.DIRECTION.value], w[Channel.ITEMS.value], w[Channel.MISC.value]
    fp, pw = w[Channel.FOOTPRINT.value], w[Channel.POWER.value]
    W, H = ent.shape
    x0, x1, y0, y1 = crop or (0, W, 0, H)
    inb = lambda x, y: 0 <= x < W and 0 <= y < H  # noqa: E731
    delta = lambda d: DIR_TO_DELTA.get(Direction(int(d)), (0, 0))  # noqa: E731
    down = lambda x, y: ent[x, y] == ID["underground_belt"] and misc[x, y] == Misc.UNDERGROUND_DOWN.value  # noqa: E731

    def icon(item_id, cx, cy, size):
        img = _icon(item_id)
        if img is not None:
            ax.imshow(img, extent=(cx - size / 2, cx + size / 2, cy + size / 2, cy - size / 2), zorder=6)

    # Points per tile, to size the text drawn on tiles.
    tile = 72 / ax.figure.dpi * min(ax.bbox.width / (x1 - x0), ax.bbox.height / (y1 - y0))
    for x in range(x0, x1):
        for y in range(y0, y1):
            f = Footprint(int(fp[x, y]))
            colour = {Footprint.UNBUILDABLE: C["unbuildable"], Footprint.MASKED: C["masked"]}.get(f, C["powered"] if pw[x, y] else C["empty"])
            ax.add_patch(Rectangle((x, y), 1, 1, facecolor=colour, edgecolor=C["grid"], lw=0.4,
                                   hatch="///" if f == Footprint.UNBUILDABLE else None, zorder=0))
            if f == Footprint.MASKED:
                ax.text(x + 0.5, y + 0.5, "?", ha="center", va="center", fontsize=0.35 * tile, color="#8a8a8a", zorder=0.5)

    def seg(x, y, dx, dy, fade=False):
        # Half-tile strip from the tile centre towards (dx, dy); `fade` thins
        # it to nothing at the edge, where the belt goes underground.
        n = 8 if fade else 1
        for i in range(n):
            a = 1 - (i + 0.5) / n if fade else 1.0
            lo, hi = 0.5 * i / n, 0.5 * (i + 1) / n
            if dx:
                left = x + 0.5 + (dx * lo if dx > 0 else dx * hi)
                ax.add_patch(Rectangle((left, y + 0.5 - BELT_W / 2), hi - lo, BELT_W, facecolor=C["belt"], alpha=a, lw=0, zorder=2))
            else:
                top = y + 0.5 + (dy * lo if dy > 0 else dy * hi)
                ax.add_patch(Rectangle((x + 0.5 - BELT_W / 2, top), BELT_W, hi - lo, facecolor=C["belt"], alpha=a, lw=0, zorder=2))

    def chevron(x, y, d):
        dx, dy = delta(d)
        ax.add_patch(FancyArrow(x + 0.5 - dx * 0.2, y + 0.5 - dy * 0.2, dx * 0.22, dy * 0.22, width=0.03,
                                head_width=0.4, head_length=0.22, color=C["belt_dark"], lw=0, zorder=3))

    def feeds(x, y, tx, ty):
        # Does the belt-ish tile/source at (x, y) push items into (tx, ty)?
        if not inb(x, y) or not (ent[x, y] in BELTISH or ent[x, y] == ID["source"]) or down(x, y):
            return False
        dx, dy = delta(dirs[x, y])
        return (x + dx, y + dy) == (tx, ty)

    parts = part_map(w)
    for x in range(W):
        for y in range(H):
            e = ent[x, y]
            dx, dy = delta(dirs[x, y])
            fed = [(fx, fy) for fx, fy in DIR_TO_DELTA.values() if feeds(x + fx, y + fy, x, y)]
            if e in (ID["transport_belt"], ID["sink"]):
                for fx, fy in fed:
                    seg(x, y, fx, fy)
            if e == ID["transport_belt"]:
                if not fed:
                    seg(x, y, -dx, -dy)
                seg(x, y, dx, dy)
                ax.add_patch(Rectangle((x + 0.5 - BELT_W / 2, y + 0.5 - BELT_W / 2), BELT_W, BELT_W, facecolor=C["belt"], lw=0, zorder=2))
                chevron(x, y, dirs[x, y])
            elif e == ID["underground_belt"]:
                if down(x, y):
                    seg(x, y, -dx, -dy)
                    seg(x, y, dx, dy, fade=True)
                    # Paired as the engine pairs it: the first underground belt
                    # within 5 tiles, if it is an exit facing the same way.
                    for k in range(1, 6):
                        tx, ty = x + dx * k, y + dy * k
                        if not inb(tx, ty):
                            break
                        if ent[tx, ty] == ID["underground_belt"]:
                            if misc[tx, ty] == Misc.UNDERGROUND_UP.value and dirs[tx, ty] == dirs[x, y]:
                                ax.plot([x + 0.5 + dx * 0.5, tx + 0.5 - dx * 0.5], [y + 0.5 + dy * 0.5, ty + 0.5 - dy * 0.5],
                                        ls=(0, (1.5, 2.5)), color=C["belt"], lw=2.2, alpha=0.6, zorder=1)
                            break
                else:
                    seg(x, y, -dx, -dy, fade=True)
                    seg(x, y, dx, dy)
                chevron(x, y, dirs[x, y])
            elif e == ID["splitter"] and parts[x, y] == 0:
                # A box across both lanes, every input crossing to every output.
                ox, oy = (0, 1) if dx else (1, 0)
                tiles = [(x, y), (x + ox, y + oy)]
                for tx, ty in tiles:
                    seg(tx, ty, -dx, -dy)
                    seg(tx, ty, dx, dy)
                ax.add_patch(Rectangle((x + 0.5 - (0.28 if dx else 0.45), y + 0.5 - (0.45 if dx else 0.28)),
                                       0.56 if dx else 1.9, 1.9 if dx else 0.56,
                                       facecolor=C["splitter"], edgecolor="black", lw=0.8, zorder=4))
                ends = lambda s: [(tx + 0.5 + s * dx * 0.26, ty + 0.5 + s * dy * 0.26) for tx, ty in tiles]  # noqa: E731
                for ax_, ay_ in ends(-1):
                    for bx_, by_ in ends(1):
                        ax.add_line(Line2D([ax_, bx_], [ay_, by_], color=C["belt"], lw=1.8, zorder=5))
            elif e in (ID["inserter"], ID["long_handed_inserter"]):
                long_ = e == ID["long_handed_inserter"]
                if long_:
                    # Dotted reach: picks up two tiles behind, drops two ahead.
                    ax.plot([x + 0.5 - 2 * dx, x + 0.5 + 2 * dx], [y + 0.5 - 2 * dy, y + 0.5 + 2 * dy],
                            ls=(0, (1, 2)), color=C["long"], lw=1.4, zorder=7)
                    for s in (-2, 2):
                        ax.add_patch(Circle((x + 0.5 + s * dx, y + 0.5 + s * dy), 0.13,
                                            facecolor="none", edgecolor=C["long"], lw=1.4, zorder=7))
                width, length = 0.32, 0.92
                ax.add_patch(Rectangle((x + 0.5 - (length if dx else width) / 2, y + 0.5 - (width if dx else length) / 2),
                                       length if dx else width, width if dx else length,
                                       facecolor=C["long"] if long_ else C["inserter"], edgecolor=C["inserter_edge"], lw=0.9, zorder=6))
                ax.add_patch(FancyArrow(x + 0.5 - dx * 0.3, y + 0.5 - dy * 0.3, dx * 0.4, dy * 0.4, width=0.05,
                                        head_width=0.24, head_length=0.18, color="black", lw=0, zorder=7))
            elif e in (ID["small_electric_pole"], ID["medium_electric_pole"]):
                ax.add_patch(Polygon(BOLT + (x, y), closed=True, facecolor=C["pole"], edgecolor="#0b2f63", lw=0.6, zorder=6))
            if e in (ID["source"], ID["sink"]):
                ax.add_patch(Rectangle((x + 0.04, y + 0.04), 0.92, 0.92, facecolor=C["source"] if e == ID["source"] else C["sink"],
                                       lw=0, zorder=5))
                icon(itm[x, y], x + 0.5, y + 0.5, 0.72)
                if dirs[x, y]:
                    ax.add_patch(FancyArrow(x + 0.5 + dx * 0.36, y + 0.5 + dy * 0.36, dx * 0.02, dy * 0.02, width=0.01,
                                            head_width=0.3, head_length=0.12, color="white", lw=0, zorder=7))

    for x, y in zip(*np.nonzero(ent == ID["assembling_machine_1"])):
        if parts[x, y] == 0:
            ax.add_patch(Rectangle((x + 0.06, y + 0.06), 2.88, 2.88, facecolor=C["asm"], edgecolor=C["asm_edge"], lw=1.5, zorder=4))
            icon(itm[x, y], x + 1.5, y + 1.5, 1.9)
    # The two 2x2 poles look alike as bolts, so they show their own icons.
    for x, y in zip(*np.nonzero(np.isin(ent, [ID["substation"], ID["big_electric_pole"]]))):
        if parts[x, y] == 0:
            icon(ent[x, y], x + 1, y + 1, 1.8)
    for x, y in zip(*np.nonzero(np.isin(ent, list(BOXED)))):
        if parts[x, y] == 0:
            side = items[int(ent[x, y])].width
            ax.add_patch(Rectangle((x + 0.06, y + 0.06), side - 0.12, side - 0.12, facecolor=BOXED[int(ent[x, y])],
                                   edgecolor=C["asm_edge"], lw=1.2, zorder=4))
            icon(ent[x, y], x + side / 2, y + side / 2, 0.75 * side)

    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    # Each label is as big as its box allows, one word a line.
    for name, mx, my, mw, mh in masked:
        ax.add_patch(Rectangle((mx + 0.06, my + 0.06), mw - 0.12, mh - 0.12, facecolor=C["unmodelled"],
                               edgecolor=C["unmodelled_edge"], lw=0.8, zorder=2))
        words = name.split("-")
        along, across = (mh, mw) if mh > mw else (mw, mh)
        size = min(0.85 * tile * along / (0.6 * max(map(len, words))), 0.85 * tile * across / (1.2 * len(words)), 14)
        ax.text(mx + mw / 2, my + mh / 2, "\n".join(words), ha="center", va="center", fontsize=size,
                linespacing=1.0, rotation=90 if mh > mw else 0, color=C["unmodelled_edge"], clip_on=True, zorder=7)

    ax.set_title(title, fontsize=8)


LEGEND = [
    Patch(facecolor=C["belt"], label="belt (chevron = flow)"),
    Patch(facecolor=C["belt"], alpha=0.35, label="underground (fades; dotted = tunnel)"),
    Patch(facecolor=C["splitter"], edgecolor="black", label="splitter (inputs cross to outputs)"),
    Patch(facecolor=C["asm"], edgecolor=C["asm_edge"], label="assembler (recipe icon)"),
    Patch(facecolor=C["inserter"], edgecolor=C["inserter_edge"], label="inserter (arrow = item flow)"),
    Patch(facecolor=C["long"], edgecolor=C["inserter_edge"], label="long-handed (dotted = 2-tile reach)"),
    Patch(facecolor=C["pole"], label="electric pole (bolt, or its icon if 2x2)"),
    Patch(facecolor=C["chest"], edgecolor=C["asm_edge"], label="chest (icon)"),
    Patch(facecolor=C["furnace"], edgecolor=C["asm_edge"], label="furnace (smelts what it is fed)"),
    Patch(facecolor=C["inert"], edgecolor=C["asm_edge"], label="solar, accumulator, roboport, radar, lamp, stone wall"),
    Patch(facecolor=C["powered"], label="powered tile"),
    Patch(facecolor=C["source"], label="source (item icon)"),
    Patch(facecolor=C["sink"], label="sink (item icon)"),
    Patch(facecolor=C["unbuildable"], hatch="///", label="unbuildable (immutable)"),
    Patch(facecolor=C["masked"], label="masked ? (no loss: unknown entity, or past the window)"),
    Patch(facecolor=C["unmodelled"], edgecolor=C["unmodelled_edge"], label="unmodelled entity (its tiles are masked)"),
]


def save(world_CWH, path, title="", crop=None, masked=()):
    x0, x1, y0, y1 = crop or (0, world_CWH.shape[1], 0, world_CWH.shape[2])
    # Big human blueprints shrink their tiles to stay a viewable image.
    inch = min(0.4, 60 / max(x1 - x0, y1 - y0))
    fig, ax = plt.subplots(figsize=(inch * (x1 - x0) + 2.8, inch * (y1 - y0) + 0.5))
    draw(world_CWH, ax, title, crop, masked)
    ax.legend(handles=LEGEND, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=6, frameon=False)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main(kind: str, seed: int = 0, size: int = 32, out: str = "factory.png") -> None:
    """Render the `kind` lesson built from `seed` on a `size` grid to `out`."""
    factory = build_factory(size=size, kind=LessonKind[kind], seed=seed)
    if factory is None:
        raise SystemExit(f"{kind} did not build at size {size} from seed {seed}")
    save(factory.world_CWH, out, f"{kind} seed {seed}")


if __name__ == "__main__":
    tyro.cli(main)
