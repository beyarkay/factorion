---
configs:
- config_name: designs
  default: true
  data_files: designs/*.parquet
- config_name: raw
  data_files: raw/*.jsonl.gz
- config_name: windows
  data_files: windows/*.parquet
---

# Factorio blueprints

Human-made Factorio layouts: community blueprints scraped from the public
sharing sites and forums, plus the bases in real save files (speedrun.com
runs and forums.factorio.com attachments), blueprinted region by region in
the headless game. Built for pretraining
[factorion](https://github.com/beyarkay/factorion)'s factory denoiser; see
its `corpus/` for the code that builds and reads it.

## `designs` (default)

One row per unique design (blueprint books unpacked; deduplicated by exact
layout, ignoring position).

| column | |
|---|---|
| `string` | the design as a Factorio exchange string (`"0"` + base64(zlib(json))), importable in game. Save regions also carry `unbuildable`: `[x, y, length]` row runs of terrain no belt can go on (water, lava, space, cliffs), in the blueprint's frame, and `lanes`: `[x, y, line, "item,item"]`, what each belt lane (line 1 = left) held when the game was saved |
| `source`, `source_id`, `url`, `title` | where it came from (`save:<site>` for save regions; `source_id` is `<save>#<surface>#<region x>,<region y>` for those, 256-tile regions) |
| `label` | the design's own label |
| `version` | game version that wrote it (`major.minor.patch`) |
| `n_entities`, `width`, `height` | entity count and the extent of the entity positions |
| `entity_counts` | JSON `{entity name: count}`, to filter without decoding |
| `modded` | holds an entity outside base, Space Age, Quality and Elevated Rails |
| `space_age` | holds a Space Age entity (2.0+ only) |
| `layout_hash` | the deduplication key |

## `windows`

Working factories cut from the save regions: a 30 × 30 interior of a
region's human layout inside a ring of sources and sinks of what its belts
carried, one row per window that delivers anything. Paste `edits` onto the
design's layout to rebuild it (factorion's `corpus.convert.paste`).

| column | |
|---|---|
| `layout_hash` | the design (`designs`) it is cut from |
| `save` | a hash of the save the design is a region of |
| `size`, `x`, `y` | the window's side, and its interior's corner in the design's bounding box |
| `edits` | `(x, y, entity, facing, item)` runs, in the window's own frame, of each tile that becomes a source, sink or pole (factorion's ids) |
| `delivered`, `crafted` | items/s reaching the sinks, and of those, what machines in the window made |

## `raw`

The scraped records verbatim, one JSON object per line, books intact:
`source`, `id`, `string`, and `url`, `title`, `author`, `tags`, `created`
where the site gives them.

## Caveats

The blueprints are their authors' work, collected from public pages; this
dataset is for research. Save regions are cut on a 256-tile grid, so an
entity straddling a seam appears in both neighbours.
