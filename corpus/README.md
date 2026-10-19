# The human-blueprint corpus (the HUMAN_BLUEPRINTS lesson)

Pretraining data from the factories people have actually built: blueprints
scraped from the community sites, plus the bases in real save files, windowed
into factorion training examples. It widens SFT's prior beyond the procedural
lessons; it has no sources or sinks, so it is never evaluated.

```
scraped records ──corpus.build──▶ HF dataset ──corpus.download──▶ SFT stream
(raw/*.jsonl.gz)                 (designs/*.parquet)              (corpus.convert, at load time)
saves ──corpus.extract_saves──▶ raw/saves.jsonl.gz
```

The dataset stores raw exchange strings, one design per row, and every
decision about factorion happens in `convert.py` when SFT loads it. Teaching
the engine a new entity is an edit to `_KNOWN`, never a re-download. Windows
cost about 2.5 ms per worker, a third of a procedural lesson, so nothing is
cached.

## Conversion (`convert.py`)

- `rasterise` lays a blueprint on its bounding box. Higher belt tiers become
  yellow, every inserter variant becomes `inserter`, and assembling machines
  1–3 become `assembling_machine_1` when the engine knows their recipe.
  Every logistic chest becomes a steel chest. Poles, wooden, iron and steel
  chests, furnaces (which carry no recipe), solar panels, accumulators,
  roboports, radars, lamps and stone walls stay as they are.
- Everything else is **masked**: its footprint becomes `Footprint.UNKNOWN`,
  so no loss reaches it, and the denoiser can tell a tile a human hid from
  one the game forbids (`UNAVAILABLE`: a save's water, lava, space, cliffs). That covers the entity, an
  assembler whose recipe the engine doesn't know, and an underground pair that
  would pair differently once collapsed to yellow's 5-tile reach.
- A design holding an entity whose size is unknown (a mod's) is dropped.
  `entity_sizes.json` lists every vanilla, Space Age, Quality and Elevated
  Rails entity's footprint, dumped from the game:
  `/sc local o={} for n,p in pairs(prototypes.entity) do o[n]={p.tile_width,p.tile_height,prototypes.get_history(p.type,n).created} end helpers.write_file("sizes.json",helpers.table_to_json(o))`
  (non-buildable prototypes such as explosions and corpses are filtered out).
- `window` cuts a random w × h window (6 ≤ w, h ≤ grid size) centred on a
  known entity, masks whatever its edge cuts through, turns and mirrors it, and
  places it anywhere in the grid with everything outside it unknown. A
  window trains only if at least 10% of it is known entities and at most 50%
  is masked.

## Commands

```bash
# scraped records + extracted saves -> raw/ + designs/ (deduplicated by layout)
uv run python -m corpus.build raw/*.jsonl.gz --out corpus_out [--repo beyarkay/factorio-blueprints]
# every region of every save with buildings -> scraped-style records
uv run python -m corpus.extract_saves saves/*.zip --index saves/index.jsonl --out raw/saves.jsonl.gz
# how much of it the current converter can use
uv run python -m corpus.stats corpus_out/designs/*.parquet
# SFT with a quarter of examples from the corpus
uv run python sft.py --corpus-frac 0.25
```

The scrapers that wrote `raw/` (blueprint sites, forums, Reddit via Arctic
Shift, pastebins, and the speedrun.com / forum save downloader feeding
`extract_saves`) live with the data, in the dataset's `scripts/`.
