"""Lane-labelled save regions -> the corpus's functional windows.

    uv run python -m corpus.windows raw/*.jsonl.gz --out corpus_out [--repo user/name]

Writes `windows/<input>.parquet` for each input file not yet written, so an
interrupted run resumes: a row per functional window (`convert.edit`) of a
region that delivers anything, holding the design's `layout_hash` (its row in
`designs/`, which this leaves untouched), a hash of the save it came from, the
interior's corner, the edits that make it work and what it then delivers.
`convert.functional` pastes the edits back on at load time.
"""

import argparse
import gzip
import hashlib
import json
import os
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from corpus.build import layout_hash
from corpus.convert import _ASM, _FURNACES, Raster, _machine_items, cut, deliveries, designs, edit, paste, rasterise

SIZE = 32
# Interior corners a stride apart, each nudged up to this far to cut the fewest machines.
_STRIDE, _NUDGE = 15, 3
SCHEMA = pa.schema(
    [
        ("layout_hash", pa.string()),
        ("save", pa.string()),
        ("size", pa.int16()),
        ("x", pa.int16()),
        ("y", pa.int16()),
        ("edits", pa.list_(pa.int16())),  # (x, y, entity, facing, item) runs
        ("delivered", pa.float32()),  # items/s reaching the sinks
        ("crafted", pa.float32()),  # of which machines made
    ]
)


def _cut_machines(r: Raster, x: int, y: int, n: int) -> int:
    inst = r.inst[x : x + n, y : y + n]
    inner = np.bincount(inst[inst > 0], minlength=len(r.tiles))
    cut_ids = np.nonzero((inner > 0) & (inner < r.tiles))[0]
    return int(np.isin(r.ent[x : x + n, y : y + n][np.isin(inst, cut_ids)], [_ASM, *_FURNACES]).sum())


def corners(r: Raster, size: int) -> list[tuple[int, int]]:
    """Where `r`'s functional windows' interiors start: a stride apart, nudged
    so the edge cuts the fewest machines, ring inside `r`, and training (`cut`)."""
    n = size - 2
    W, H = r.ent.shape
    out = set()
    for x0 in range(1, W - n, _STRIDE):
        for y0 in range(1, H - n, _STRIDE):
            near = [
                (x, y)
                for x in range(max(1, x0 - _NUDGE), min(W - n - 1, x0 + _NUDGE) + 1)
                for y in range(max(1, y0 - _NUDGE), min(H - n - 1, y0 + _NUDGE) + 1)
            ]
            x, y = min(near, key=lambda p: (_cut_machines(r, p[0], p[1], n), abs(p[0] - x0) + abs(p[1] - y0)))
            if cut(r, (slice(x, x + n), slice(y, y + n))) is not None:
                out.add((x, y))
    return sorted(out)


def windows(rec: dict, size: int = SIZE) -> list[dict]:
    """The rows of a scraped save region's functional windows that deliver anything."""
    bp = next(designs(rec["string"]))
    for key in ("unbuildable", "lanes"):
        bp[key] = rec.get(key) or []
    r = rasterise(bp)
    if r is None or not r.lanes:
        return []
    layout = layout_hash(bp)
    save = hashlib.blake2b(rec["id"].split("#")[0].encode(), digest_size=16).hexdigest()
    rows = []
    for x, y in corners(r, size):
        edits = edit(r, x, y, size, random.Random(f"{layout}@{x},{y}"))
        w = paste(r, x, y, size, edits)
        delivered, crafted = deliveries(w, _machine_items(w)[1])
        if delivered > 0:
            rows.append(
                {"layout_hash": layout, "save": save, "size": size, "x": x, "y": y,
                 "edits": [v for e in edits for v in e], "delivered": delivered, "crafted": crafted}
            )
    return rows


def _file(args: tuple[Path, Path]) -> int:
    src, dst = args
    with gzip.open(src, "rt") as f:
        rows = [row for line in f for row in windows(json.loads(line))]
    tmp = dst.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows, SCHEMA), tmp)
    os.replace(tmp, dst)
    return len(rows)


def build(raw: list[Path], out: Path, procs: int | None = None) -> int:
    """Write `out/windows/` for the inputs it lacks; the number of windows added."""
    (out / "windows").mkdir(parents=True, exist_ok=True)
    todo = [(p, out / "windows" / f"{p.name.split('.')[0]}.parquet") for p in raw]
    todo = [t for t in todo if not t[1].exists()]
    with Pool(procs) as pool:
        return sum(pool.imap_unordered(_file, todo))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--procs", type=int)
    ap.add_argument("--repo", help="Hugging Face dataset repo to upload `windows/` to")
    args = ap.parse_args()
    print(build(args.raw, args.out, args.procs), "windows added")
    if args.repo:
        from huggingface_hub import HfApi

        HfApi().upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=args.out / "windows", path_in_repo="windows")


if __name__ == "__main__":
    main()
