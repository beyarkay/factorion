"""Scraped records -> the Hugging Face blueprint corpus.

    uv run python -m corpus.build raw/*.jsonl.gz --out corpus_out [--repo user/name]

Writes `raw/` (the scraped records verbatim, books and all) and `designs/`
(every blueprint unpacked from them, deduplicated by layout, one exchange
string per row plus enough metadata to filter without decoding). Nothing is
filtered for factorion here: `corpus.convert` decides that at load time.
"""

import argparse
import base64
import gzip
import hashlib
import json
import shutil
import zlib
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from corpus.convert import _SIZES, designs

_SHARD_ROWS = 20_000


def _encode(bp: dict) -> str:
    return "0" + base64.b64encode(zlib.compress(json.dumps({"blueprint": bp}, separators=(",", ":")).encode())).decode()


def _rows(rec: dict) -> list[dict]:
    """One row per entity-bearing blueprint in a scraped record."""
    try:
        bps = list(designs(rec["string"]))
    except Exception:
        return []
    for key in ("unbuildable", "lanes"):  # a save region's terrain and belt contents (`corpus.extract_saves`)
        if rec.get(key):
            bps[0][key] = rec[key]
    out = []
    for bp in bps:
        try:
            if bp.get("entities"):
                out.append(_row(rec, bp))
        except (KeyError, TypeError, ValueError):
            pass  # a malformed blueprint
    return out


def _row(rec: dict, bp: dict) -> dict:
    ents = bp["entities"]
    xs = [e["position"]["x"] for e in ents]
    ys = [e["position"]["y"] for e in ents]
    x0, y0 = min(xs), min(ys)
    layout = sorted(
        repr((e["name"], e["position"]["x"] - x0, e["position"]["y"] - y0, e.get("direction", 0), e.get("recipe"), e.get("type")))
        for e in ents
    )
    counts = Counter(e["name"] for e in ents)
    v = bp.get("version", 0)
    return {
        "string": _encode(bp),
        "source": rec["source"],
        "source_id": str(rec.get("id", "")),
        "url": rec.get("url") or "",
        "title": rec.get("title") or "",
        "label": bp.get("label") or "",
        "version": f"{v >> 48}.{(v >> 32) & 0xFFFF}.{(v >> 16) & 0xFFFF}",
        "n_entities": len(ents),
        "width": int(max(xs) - x0) + 1,
        "height": int(max(ys) - y0) + 1,
        "entity_counts": json.dumps(dict(counts.most_common())),
        "modded": any(n not in _SIZES for n in counts),
        "space_age": v >> 48 >= 2 and any(_SIZES.get(n, [0, 0, ""])[2] == "space-age" for n in counts),
        "layout_hash": hashlib.blake2b(json.dumps(layout).encode(), digest_size=16).hexdigest(),
    }


def _records(paths: list[Path]):
    for p in paths:
        with gzip.open(p, "rt") as f:
            try:
                for line in f:
                    yield json.loads(line)
            except (EOFError, json.JSONDecodeError):
                print(f"{p} is truncated: stopped at the last whole record")  # a scraper still writing it


def build(raw: list[Path], out: Path) -> Counter:
    """Write `out/raw`, `out/designs` and the dataset card from the scraped
    records in `raw`."""
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "designs").mkdir(exist_ok=True)
    shutil.copy(Path(__file__).parent / "dataset_card.md", out / "README.md")
    for p in raw:
        shutil.copy(p, out / "raw" / p.name)
    seen: set[str] = set()
    stats: Counter = Counter()
    shard: list[dict] = []
    n_shards = 0

    def flush() -> None:
        nonlocal shard, n_shards
        pq.write_table(pa.Table.from_pylist(shard), out / "designs" / f"part-{n_shards:05d}.parquet")
        shard, n_shards = [], n_shards + 1

    with Pool() as pool:
        for rows in pool.imap(_rows, _records(raw), chunksize=16):
            stats["records"] += 1
            for row in rows:
                stats["designs"] += 1
                if row["layout_hash"] in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(row["layout_hash"])
                stats[f"kept/{row['source']}"] += 1
                shard.append(row)
                if len(shard) == _SHARD_ROWS:
                    flush()
    if shard:
        flush()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--repo", help="Hugging Face dataset repo to upload to")
    args = ap.parse_args()
    print(json.dumps(build(args.raw, args.out), indent=1))
    if args.repo:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=args.out)


if __name__ == "__main__":
    main()
