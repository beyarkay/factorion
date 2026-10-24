# The human-blueprint corpus (the HUMAN_BLUEPRINTS lesson)

Pretraining data from the factories people have actually built: blueprints
scraped from the community sites, plus the bases in real save files, windowed
into factorion training examples. It widens SFT's prior beyond the procedural
lessons. Plain windows have no sources or sinks, so they are never evaluated;
the functional windows of save regions have both, and SFT scores held-out
ones as `val/human/`.

```
scraped records ──corpus.build──▶ HF dataset ──corpus.download──▶ SFT stream
(raw/*.jsonl.gz)                 (designs/*.parquet)              (corpus.convert, at load time)
saves ──corpus.extract_saves──▶ raw/saves.jsonl.gz
save regions ──corpus.windows──▶ HF dataset (windows/*.parquet) ──▶ SFT stream (corpus.convert.functional)
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
- Everything else is **masked**: its footprint becomes `Footprint.MASKED`,
  so no loss reaches it. What the game forbids (a save's water, lava, space,
  cliffs) is `UNBUILDABLE`. That covers the entity, an
  assembler whose recipe the engine doesn't know, and an underground pair that
  would pair differently once collapsed to yellow's 5-tile reach.
- A design holding an entity whose size is unknown (a mod's) is dropped.
  `entity_sizes.json` lists every vanilla, Space Age, Quality and Elevated
  Rails entity's footprint, dumped from the game:
  `/sc local o={} for n,p in pairs(prototypes.entity) do o[n]={p.tile_width,p.tile_height,prototypes.get_history(p.type,n).created} end helpers.write_file("sizes.json",helpers.table_to_json(o))`
  (non-buildable prototypes such as explosions and corpses are filtered out).
- `window` shows a view of the design as big as the grid (smaller only where
  the design is), and picks a random build region in it (8 tiles a side up
  to the whole view) around a belt, inserter or assembler:
  - the build region is `BUILDABLE`; the human's layout there is the target;
  - the rest of the view is `UNBUILDABLE` context that keeps its known
    entities, so the build has to connect to the factory around it (an
    entity straddling the region's edge is context);
  - unmodelled entities, and entities the view's edge cuts, are `MASKED`;
  - past a blueprint's edge is `UNBUILDABLE`, but for a one-tile `MASKED`
    band so its own inputs and outputs lead somewhere unknown; past a save
    region's edge is more world, so `MASKED`.

  The whole grid is then turned and mirrored. A view trains only if at least
  10% of it is known entities, at most 50% is masked, and it holds a belt,
  inserter or assembler.
- `stream` runs in epochs: each design gives `views` windows an epoch (the
  grid-sized tiles of it that would train, about 650k over the corpus at
  32 × 32), each DataLoader worker takes its own share of the designs, and
  the 1% `held_out` by layout hash never train. SFT scores those as
  `val/corpus/...` and, when `corpus_frac` is 1 (thput then means nothing
  yet), selects checkpoints on `val/corpus/type_acc`.

## Functional windows (`windows.py`)

A save records what each belt lane carried (`lanes`), which says where a
working factory's sources and sinks go. `convert.edit` turns a save region's
30 × 30 interior, all of it buildable, into a working 32 × 32 factory:

- every flow across the interior's edge, or into an entity the engine lacks
  or the edge cuts, becomes a source or sink on that tile, of the item the
  save saw on the belt (or along it), as do a tunnel the edge cuts, a
  labelled belt fed by nothing, a belt that ends at nothing and nothing
  picks from upstream, and a mining drill's output;
- a chest inserters fill and none empty (a store for later) becomes a sink;
- a marker no label settles gets the item that brings the most of the
  machines' products to the sinks (a belt nothing touches, any item);
- medium poles power whatever the human's poles leave unpowered, where there
  is room.

About 94% of windows then deliver something, and two thirds of those with a
machine get its products out. Each takes ~30 ms (a few dozen simulations to
choose items), too slow for SFT to make on the fly, so `windows.py`
precomputes them: interiors 15 tiles apart, each nudged up to 3 to cut the
fewest machines, keeping those that deliver, as their edits over the human
layout in `windows/` beside `designs/`, which it leaves untouched.
`convert.functional` pastes them back on at load time, turned and mirrored
at random. The 5% of saves `held_out` by save never train; SFT generates
`eval_seeds_per_kind` of their windows from their markers, scored against
the human's own build, as `val/human/thput@N`.

## Commands

```bash
# scraped records + extracted saves -> raw/ + designs/ (deduplicated by layout)
uv run python -m corpus.build raw/*.jsonl.gz --out corpus_out [--repo beyarkay/factorio-blueprints]
# every region of every save with buildings -> scraped-style records
uv run python -m corpus.extract_saves saves/*.zip --index saves/index.jsonl --out raw/saves.jsonl.gz
# how much of it the current converter can use
uv run python -m corpus.stats corpus_out/designs/*.parquet
# lane-labelled save regions -> windows/, resuming where it stopped
uv run python -m corpus.windows raw/saves_lanes/*.jsonl.gz --out corpus_out [--repo beyarkay/factorio-blueprints]
# SFT with a quarter of examples from the corpus
uv run python sft.py --corpus-frac 0.25
# ... or from its functional windows
uv run python sft.py --windows-frac 0.25
```

The scrapers that wrote `raw/` (blueprint sites, forums, Reddit via Arctic
Shift, pastebins, and the speedrun.com / forum save downloader feeding
`extract_saves`) live with the data, in the dataset's `scripts/`.
