"""How much of the corpus the HUMAN_BLUEPRINTS lesson can use under the
current converter. Re-run it after teaching the engine an entity.

    uv run python -m corpus.stats corpus_out/designs/*.parquet [--size 32]

`windows` counts the size × size tiles of every design's bounding box (no
overlap, no augmentation) that pass `window`'s filter: a floor on the
distinct training windows, which overlap, turn and mirror freely.
"""

import argparse
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq

from corpus.convert import MAX_MASKED, MIN_KNOWN, cut, designs, rasterise
from factorion import items


def _design(size: int, row: dict) -> Counter:
    bp = next(designs(row["string"]))
    n = len(bp.get("entities") or [])
    src = row["source"]
    c = Counter({"designs": 1, "entities": n, f"source/{src}/designs": 1})
    r = rasterise(bp)
    if r is None:
        c["rejected_designs"] += 1
        c["rejected_entities"] += n
        return c
    c["kept_entities"] += int((r.tiles > 0).sum())
    c["masked_entities"] += len(r.masked)
    c["kept_tiles"] += int((r.inst > 0).sum())
    c["masked_tiles"] += int((r.inst < 0).sum())
    for name, *_ in r.masked:
        c[f"masked/{name}"] += 1
    for e, tiles in Counter(r.ent[r.inst > 0].tolist()).items():
        c[f"kept/{items[e].name}"] += tiles // (items[e].width * items[e].height)
    for recipe, tiles in Counter(r.item[r.inst > 0].tolist()).items():
        if recipe:
            c[f"recipe/{items[recipe].name}"] += tiles // 9
    W, H = r.ent.shape
    for x in range(0, W, size):
        for y in range(0, H, size):
            keep, mask = cut(r, (slice(x, x + size), slice(y, y + size)))
            if keep.mean() >= MIN_KNOWN and mask.mean() <= MAX_MASKED:
                c["windows"] += 1
                c[f"source/{src}/windows"] += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--size", type=int, default=32)
    args = ap.parse_args()
    total: Counter = Counter()
    with ProcessPoolExecutor() as ex:
        for f in args.files:
            rows = pq.read_table(f, columns=["string", "source"], filters=[("modded", "==", False)]).to_pylist()
            for c in ex.map(partial(_design, args.size), rows, chunksize=64):
                total.update(c)
    print(json.dumps(dict(sorted(total.items())), indent=1))


if __name__ == "__main__":
    main()
