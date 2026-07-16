"""Green-fields PoC: Factorio blueprint -> (K, 50, 50) tile tensor for a masked encoder.

Standalone. Not wired into the training code, not robust — just to see if the
idea works. Each tile is a token; the transformer would embed each field plane
and sum. Everything here is losslessly tile-local (per the agreed schema):
entity, direction, floor, ug/loader type, splitter priority, recipe, footprint
(anchor + body cells), and small fixed-slot module/filter sets. Wires, trains,
schedules, quality, colour, etc. are intentionally dropped.

Run:  python3 masked_encoder/vectoriser.py
"""
import os
import json
import math
import base64
import zlib
import collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BP = os.path.join(ROOT, "data", "blueprints")
W = 50                       # window side
MOD_SLOTS, FILT_SLOTS = 4, 5

WHITELIST = sorted(l.strip() for l in open(os.path.join(ROOT, "data", "vanilla_whitelist.txt")) if l.strip())
RENAME = json.load(open(os.path.join(ROOT, "data", "normalise_map_1_1_to_2_0.json")))

# nominal (w,h); everything not listed is 1x1
FOOTPRINT = {
    "assembling-machine-1": (3, 3), "assembling-machine-2": (3, 3), "assembling-machine-3": (3, 3),
    "chemical-plant": (3, 3), "oil-refinery": (5, 5), "electric-furnace": (3, 3),
    "steel-furnace": (2, 2), "stone-furnace": (2, 2), "centrifuge": (3, 3), "lab": (3, 3),
    "beacon": (3, 3), "radar": (3, 3), "solar-panel": (3, 3), "accumulator": (2, 2),
    "nuclear-reactor": (5, 5), "heat-exchanger": (3, 2), "steam-turbine": (3, 5), "steam-engine": (3, 5),
    "boiler": (3, 2), "rocket-silo": (9, 9), "roboport": (4, 4), "storage-tank": (3, 3),
    "pumpjack": (3, 3), "pump": (1, 2), "offshore-pump": (1, 2),
    "big-electric-pole": (2, 2), "substation": (2, 2), "train-stop": (2, 2),
    "gun-turret": (2, 2), "laser-turret": (2, 2), "flamethrower-turret": (2, 3), "artillery-turret": (3, 3),
    "burner-mining-drill": (2, 2), "electric-mining-drill": (3, 3), "burner-generator": (2, 2),
    "straight-rail": (2, 2), "legacy-straight-rail": (2, 2), "curved-rail": (4, 4),
    "legacy-curved-rail": (4, 4), "curved-rail-a": (4, 4), "curved-rail-b": (4, 4), "half-diagonal-rail": (2, 4),
    "arithmetic-combinator": (1, 2), "decider-combinator": (1, 2), "selector-combinator": (1, 2),
    "power-switch": (2, 2), "loader": (1, 2), "fast-loader": (1, 2), "express-loader": (1, 2),
}
HORIZ = {4, 12}   # E/W in the 16-direction (2.0) convention; inputs normalised via dir_scale()

# rails don't have rectangular footprints and legitimately overlap each other
# (crossings). Collapse all track pieces to ONE occupancy token, painted only
# onto free tiles (real entities take precedence), dropping direction/type.
RAIL_TILES = {"straight-rail", "curved-rail", "legacy-straight-rail", "legacy-curved-rail",
              "curved-rail-a", "curved-rail-b", "half-diagonal-rail"}
RAIL_TOKEN = "factorio-rail"


def snap(v):
    """Round half-up consistently (Python's round() uses banker's rounding, which
    breaks grid alignment for blueprints centred on a tile boundary)."""
    return math.floor(v + 0.5)


def footprint(name, direction):
    w, h = FOOTPRINT.get(name, (1, 1))
    if w != h and direction in HORIZ:
        w, h = h, w
    return w, h


def dir_scale(ents):
    """1.x blueprints encode cardinals in 8 directions (E=2, S=4, W=6); 2.0 uses
    16 (E=4, S=8, W=12). Normalise 8-dir designs to 16-dir so footprint rotation
    and the stored direction token are unambiguous — otherwise 8-dir's 4 (south)
    collides with 16-dir's 4 (east) and south-facing non-square entities get
    their footprint wrongly rotated. Heuristic: any value >7 ⇒ already 16-dir."""
    return 1 if any(e.get("direction", 0) > 7 for e in ents) else 2


# ---- field planes (all int16, 0 = none/empty) ----
PLANES = ["role", "entity", "direction", "floor", "type", "in_prio", "out_prio",
          "recipe", "body_dx", "body_dy"] + [f"mod{i}" for i in range(MOD_SLOTS)] + [f"filt{i}" for i in range(FILT_SLOTS)]
ROLE = {"empty": 0, "floor": 1, "anchor": 2, "body": 3, "mask": 4, "rail": 5}
TYPE = {None: 0, "input": 1, "output": 2}
PRIO = {None: 0, "left": 1, "right": 2}


def load_vocabs():
    p = os.path.join(HERE, "vocabs.json")
    if os.path.exists(p):
        return json.load(open(p))
    print("building vocabs from corpus (one-time)...")
    rec, item, floor = collections.Counter(), collections.Counter(), collections.Counter()
    lines = open(os.path.join(BP, "factorio_school.individual.txt")).read().splitlines()
    for s in lines[::max(1, len(lines) // 12000)][:12000]:
        bp = decode(s)
        if bp is None:
            continue
        ents = bp.get("entities") or []
        ns = {RENAME.get(e["name"], e["name"]) for e in ents}
        if not ns <= set(WHITELIST):
            continue
        for t in bp.get("tiles") or []:
            floor[t.get("name")] += 1
        for e in ents:
            if "recipe" in e:
                rec[e["recipe"]] += 1
            for k in module_items(e):
                item[k] += 1
            for k in filter_items(e):
                item[k] += 1
    ent_names = [n for n in WHITELIST if n not in RAIL_TILES] + [RAIL_TOKEN]
    v = {
        "entity": {n: i + 1 for i, n in enumerate(ent_names)},
        "recipe": {n: i + 1 for i, (n, _) in enumerate(rec.most_common())},
        "item": {n: i + 1 for i, (n, _) in enumerate(item.most_common())},
        "floor": {n: i + 1 for i, (n, _) in enumerate(floor.most_common())},
    }
    json.dump(v, open(p, "w"))
    print(f"  entity={len(v['entity'])} recipe={len(v['recipe'])} item={len(v['item'])} floor={len(v['floor'])}")
    return v


def decode(s):
    try:
        return json.loads(zlib.decompress(base64.b64decode(s[1:])))["blueprint"]
    except Exception:
        return None


def module_items(e):
    it = e.get("items")
    if isinstance(it, dict):
        return list(it)
    if isinstance(it, list):
        out = []
        for x in it:
            idd = x.get("id")
            out.append(idd.get("name") if isinstance(idd, dict) else idd)
        return out
    return []


def filter_items(e):
    out = [f.get("name") if isinstance(f, dict) else f for f in (e.get("filters") or [])]
    if isinstance(e.get("filter"), str):
        out.append(e["filter"])
    return [x for x in out if x]


def vectorise(bp, vocab, crop=(0, 0)):
    """Return (K,50,50) int16 tensor or None if empty/non-vanilla. Large designs
    are cropped to a 50x50 window whose top-left is `crop` (tile coords)."""
    ents = bp.get("entities") or []
    ns = {RENAME.get(e["name"], e["name"]) for e in ents}
    if not ents or not ns <= set(WHITELIST):
        return None
    # place onto an integer tile grid
    scale = dir_scale(ents)
    placed = []
    for e in ents:
        name = RENAME.get(e["name"], e["name"])
        d = e.get("direction", 0) * scale
        w, h = footprint(name, d)
        left = snap(e["position"]["x"] - w / 2)
        top = snap(e["position"]["y"] - h / 2)
        placed.append((name, d, e, left, top, w, h))
    minx = min(p[3] for p in placed)
    miny = min(p[4] for p in placed)
    T = {p: np.zeros((W, W), np.int16) for p in PLANES}
    ev, rv, iv, fv = vocab["entity"], vocab["recipe"], vocab["item"], vocab["floor"]
    ox, oy = crop
    # floors
    for t in bp.get("tiles") or []:
        x = snap(t["position"]["x"]) - minx - ox
        y = snap(t["position"]["y"]) - miny - oy
        if 0 <= x < W and 0 <= y < W and t.get("name") in fv:
            T["floor"][y, x] = fv[t["name"]]
            if T["role"][y, x] == 0:
                T["role"][y, x] = ROLE["floor"]
    # pass 1: real (non-rail) entities fill their footprints
    for name, d, e, left, top, w, h in placed:
        if name in RAIL_TILES:
            continue
        ax, ay = left - minx - ox, top - miny - oy
        mods = [iv[m] for m in module_items(e) if m in iv][:MOD_SLOTS]
        filts = [iv[f] for f in filter_items(e) if f in iv][:FILT_SLOTS]
        for dx in range(w):
            for dy in range(h):
                x, y = ax + dx, ay + dy
                if not (0 <= x < W and 0 <= y < W):
                    continue
                is_anchor = (dx == 0 and dy == 0)
                T["role"][y, x] = ROLE["anchor"] if is_anchor else ROLE["body"]
                T["entity"][y, x] = ev[name]
                T["direction"][y, x] = d + 1
                T["body_dx"][y, x] = dx
                T["body_dy"][y, x] = dy
                if is_anchor:
                    T["type"][y, x] = TYPE.get(e.get("type"), 0)
                    T["in_prio"][y, x] = PRIO.get(e.get("input_priority"), 0)
                    T["out_prio"][y, x] = PRIO.get(e.get("output_priority"), 0)
                    if "recipe" in e and e["recipe"] in rv:
                        T["recipe"][y, x] = rv[e["recipe"]]
                    for i, m in enumerate(mods):
                        T[f"mod{i}"][y, x] = m
                    for i, f in enumerate(filts):
                        T[f"filt{i}"][y, x] = f
    # pass 2: rail blob — one token, only onto tiles no real entity claimed
    rid = ev[RAIL_TOKEN]
    for name, d, e, left, top, w, h in placed:
        if name not in RAIL_TILES:
            continue
        ax, ay = left - minx - ox, top - miny - oy
        for dx in range(w):
            for dy in range(h):
                x, y = ax + dx, ay + dy
                if 0 <= x < W and 0 <= y < W and T["role"][y, x] in (ROLE["empty"], ROLE["floor"]):
                    T["role"][y, x] = ROLE["rail"]
                    T["entity"][y, x] = rid
    return np.stack([T[p] for p in PLANES])


def devectorise(tensor, vocab):
    """Reconstruct the entity set from a tensor (for round-trip checking)."""
    inv_e = {i: n for n, i in vocab["entity"].items()}
    inv_r = {i: n for n, i in vocab["recipe"].items()}
    inv_i = {i: n for n, i in vocab["item"].items()}
    P = {p: tensor[i] for i, p in enumerate(PLANES)}
    out = []
    ys, xs = np.where(P["role"] == ROLE["anchor"])
    for y, x in zip(ys, xs):
        name = inv_e[int(P["entity"][y, x])]
        rec = inv_r.get(int(P["recipe"][y, x]))
        mods = sorted(inv_i[int(P[f"mod{i}"][y, x])] for i in range(MOD_SLOTS) if P[f"mod{i}"][y, x])
        filts = sorted(inv_i[int(P[f"filt{i}"][y, x])] for i in range(FILT_SLOTS) if P[f"filt{i}"][y, x])
        out.append((name, int(P["direction"][y, x]) - 1, int(x), int(y), rec, tuple(mods), tuple(filts)))
    return out


def original_tuples(bp, vocab):
    """The same tuple form, straight from the blueprint, for comparison."""
    ns = {RENAME.get(e["name"], e["name"]) for e in bp["entities"]}
    scale = dir_scale(bp["entities"])
    placed = []
    for e in bp["entities"]:
        name = RENAME.get(e["name"], e["name"])
        d = e.get("direction", 0) * scale
        w, h = footprint(name, d)
        placed.append((name, d, e, snap(e["position"]["x"] - w / 2), snap(e["position"]["y"] - h / 2)))
    minx = min(p[3] for p in placed)
    miny = min(p[4] for p in placed)
    iv, rv = vocab["item"], vocab["recipe"]
    out = []
    for name, d, e, left, top in placed:
        rec = e.get("recipe") if e.get("recipe") in rv else None
        mods = sorted({m for m in module_items(e) if m in iv})
        filts = sorted({f for f in filter_items(e) if f in iv})
        out.append((name, d, left - minx, top - miny, rec, tuple(mods), tuple(filts)))
    return out


RENDER = {"transport-belt": "b", "fast-transport-belt": "b", "express-transport-belt": "b",
          "underground-belt": "u", "inserter": "i", "fast-inserter": "i", "bulk-inserter": "i",
          "long-handed-inserter": "l", "pipe": "=", "small-electric-pole": "+", "medium-electric-pole": "+",
          "assembling-machine-1": "A", "assembling-machine-2": "A", "assembling-machine-3": "A",
          "straight-rail": "r", "legacy-straight-rail": "r", "stone-wall": "#"}


def render(tensor, vocab):
    inv_e = {i: n for n, i in vocab["entity"].items()}
    P = {p: tensor[i] for i, p in enumerate(PLANES)}
    rows = []
    for y in range(W):
        row = ""
        for x in range(W):
            r = P["role"][y, x]
            if r in (ROLE["anchor"], ROLE["body"]):
                row += RENDER.get(inv_e[int(P["entity"][y, x])], "o")
            elif r == ROLE["rail"]:
                row += "r"
            elif r == ROLE["floor"]:
                row += "."
            else:
                row += " "
        rows.append(row.rstrip())
    return "\n".join(r for r in rows if r)


if __name__ == "__main__":
    vocab = load_vocabs()
    print(f"planes={len(PLANES)}: {PLANES}")
    lines = open(os.path.join(BP, "factorio_school.individual.txt")).read().splitlines()
    n_ok = n_try = n_fit = 0
    ent_match = ent_total = 0
    shown = False
    for s in lines[::37][:4000]:
        bp = decode(s)
        if bp is None:
            continue
        t = vectorise(bp, vocab)
        if t is None:
            continue
        n_try += 1
        orig_all = set(original_tuples(bp, vocab))
        # only score designs that fit the window (cropping loses tiles by design)
        maxx = max((p[2] for p in orig_all), default=0)
        maxy = max((p[3] for p in orig_all), default=0)
        if maxx >= W or maxy >= W:
            continue
        n_fit += 1
        # rails are collapsed to an occupancy blob by design -> score non-rail fidelity
        orig = {p for p in orig_all if p[0] not in RAIL_TILES}
        got = {p for p in devectorise(t, vocab) if p[0] != RAIL_TOKEN}
        ent_total += len(orig)
        ent_match += len(orig & got)
        if orig == got:
            n_ok += 1
        if not shown and 20 <= len(bp["entities"]) <= 120:
            print("\n--- sample design render (tensor -> ASCII) ---")
            print(render(t, vocab))
            print(f"(entities={len(bp['entities'])}, exact round-trip={'YES' if orig==got else 'NO'})")
            shown = True
    print(f"\nfitting designs scored: {n_fit}")
    print(f"exact round-trip (all fields, all entities): {n_ok}/{n_fit} = {100*n_ok/max(1,n_fit):.1f}%")
    print(f"per-entity recovery: {ent_match}/{ent_total} = {100*ent_match/max(1,ent_total):.2f}%")
