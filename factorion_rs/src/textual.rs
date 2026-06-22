//! Textual factory format for unit tests.
//!
//! A factory is described as a YAML header, a `---` separator, and an ASCII
//! grid. The header carries the per-entity item bindings and the expected
//! throughput; the grid carries the geometry. Example:
//!
//! ```text
//! items:
//! - { x: 0, y: 0, item: copper_plate }   # what the source emits
//! - { x: 3, y: 0, item: copper_plate }   # what the sink counts
//! throughput:
//! - { item: copper_plate, per_second: 15 }
//! ---
//! S> b> b> K> ..
//! ```
//!
//! ## Grid encoding
//!
//! Every tile is exactly **two characters**, and tiles are separated by one
//! filler character — so the parser walks each row as *gobble 2, skip 1,
//! gobble 2, skip 1, …*. Multi-tile entities overwrite the filler with their
//! own body, which is why we never split on whitespace.
//!
//! For a tile's two characters:
//! - a **direction** is any of `^ > v <` appearing in either character
//!   (North/East/South/West); absence means [`Direction::None`].
//! - an **entity** is any registry character (see [`ENTITY_CHARS`]) appearing
//!   in either character. Entity characters never overlap the direction
//!   characters, so order does not matter (`b^` and `^b` are the same belt).
//! - `..` is an empty tile; `  ` (two spaces) is the blank interior of a
//!   multi-tile entity and is only legal inside one.
//!
//! ## Multi-tile entities
//!
//! An entity's footprint comes from [`Item::size`], so a new entity is one row
//! in [`ENTITY_CHARS`] and the rest is automatic. The body characters are
//! drawn across the whole footprint, blank (`  `) tiles allowed only in the
//! interior:
//!
//! ```text
//!   assembler (3×3, square)     splitter (2×1)
//!   aaaaaaaa                    east:  Y>     south: YvvvY
//!   aa    aa                           Y>
//!   aaaaaaaa
//! ```
//!
//! The first body tile in reading order is the anchor (top-left), matching how
//! `build_graph` re-derives multi-tile entities.

use serde::Deserialize;

use crate::entities::entity_tiles;
use crate::graph::build_graph;
use crate::throughput::calc_throughput;
use crate::types::{Channel, Direction, Item, Misc};
use crate::world::World;

/// Float tolerance for throughput comparisons.
const TOL: f64 = 1e-9;

/// Registry of grid characters → (entity, underground state).
///
/// This is the single place to extend the format with a new entity. The
/// footprint is *not* listed here — it is read from [`Item::size`] so the two
/// can never disagree.
const ENTITY_CHARS: &[(char, Item, Misc)] = &[
    ('b', Item::TransportBelt, Misc::None),
    ('i', Item::Inserter, Misc::None),
    ('a', Item::AssemblingMachine1, Misc::None),
    ('Y', Item::Splitter, Misc::None),
    ('d', Item::UndergroundBelt, Misc::UndergroundDown),
    ('u', Item::UndergroundBelt, Misc::UndergroundUp),
    ('S', Item::Source, Misc::None),
    ('K', Item::Sink, Misc::None),
];

/// Direction markers. Deliberately disjoint from every [`ENTITY_CHARS`] entry.
const DIR_CHARS: &[(char, Direction)] = &[
    ('^', Direction::North),
    ('>', Direction::East),
    ('v', Direction::South),
    ('<', Direction::West),
];

fn entity_for_char(c: char) -> Option<(Item, Misc)> {
    ENTITY_CHARS
        .iter()
        .find(|(ch, _, _)| *ch == c)
        .map(|(_, item, misc)| (*item, *misc))
}

/// The (first) grid character for an entity — used for error messages.
fn char_for_item(item: Item) -> char {
    ENTITY_CHARS
        .iter()
        .find(|(_, i, _)| *i == item)
        .map(|(c, _, _)| *c)
        .unwrap_or('?')
}

fn dir_for_char(c: char) -> Option<Direction> {
    DIR_CHARS.iter().find(|(ch, _)| *ch == c).map(|(_, d)| *d)
}

/// Grid character for rendering an entity, honouring its underground state
/// (so `UndergroundBelt` + `UndergroundUp` renders as `u`, not `d`).
fn render_char(item: Item, misc: Misc) -> char {
    ENTITY_CHARS
        .iter()
        .find(|(_, i, m)| *i == item && *m == misc)
        .or_else(|| ENTITY_CHARS.iter().find(|(_, i, _)| *i == item))
        .map(|(c, _, _)| *c)
        .unwrap_or('?')
}

/// Direction character for rendering; [`Direction::None`] renders as `.`.
fn char_for_dir(dir: Direction) -> char {
    DIR_CHARS
        .iter()
        .find(|(_, d)| *d == dir)
        .map(|(c, _)| *c)
        .unwrap_or('.')
}

/// A classified grid tile, before multi-tile entities are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Cell {
    /// `..` — nothing here.
    Empty,
    /// `  ` — blank interior of a multi-tile entity; illegal on its own.
    Interior,
    /// A placed entity (single- or multi-tile; resolved during placement).
    Entity {
        item: Item,
        misc: Misc,
        dir: Direction,
    },
}

/// Classify a single two-character tile.
fn classify(c0: char, c1: char) -> Result<Cell, String> {
    if c0 == '.' && c1 == '.' {
        return Ok(Cell::Empty);
    }
    if c0 == ' ' && c1 == ' ' {
        return Ok(Cell::Interior);
    }

    let dir = match (dir_for_char(c0), dir_for_char(c1)) {
        (Some(_), Some(_)) => {
            return Err(format!("two direction markers in tile '{c0}{c1}'"));
        }
        (Some(d), None) | (None, Some(d)) => d,
        (None, None) => Direction::None,
    };

    let entity = match (entity_for_char(c0), entity_for_char(c1)) {
        (Some(a), Some(b)) if a != b => {
            return Err(format!(
                "two different entity characters in tile '{c0}{c1}'"
            ));
        }
        (Some(e), _) | (_, Some(e)) => e,
        (None, None) => {
            return Err(format!("tile '{c0}{c1}' has no entity character"));
        }
    };

    Ok(Cell::Entity {
        item: entity.0,
        misc: entity.1,
        dir,
    })
}

/// Tokenize the grid body into a rectangular `[y][x]` matrix of [`Cell`]s.
///
/// Blank lines are ignored (so triple-quoted literals can breathe). Leading and
/// trailing whitespace on a line is stripped — the leftmost tile is never a
/// blank interior (those are always enclosed by body characters), so indenting
/// a grid for readability is safe. Short rows are right-padded with empty tiles.
fn tokenize(body: &str) -> Result<Vec<Vec<Cell>>, String> {
    let lines: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    if lines.is_empty() {
        return Err("grid is empty".to_string());
    }

    let mut rows: Vec<Vec<Cell>> = Vec::with_capacity(lines.len());
    let mut width = 0usize;
    for (y, line) in lines.iter().enumerate() {
        let chars: Vec<char> = line.chars().collect();
        let len = chars.len();
        // A row of N tiles is `2*N + (N-1)` = `3*N - 1` characters wide.
        if !(len + 1).is_multiple_of(3) {
            return Err(format!(
                "row {y} is {len} chars wide; each tile is 2 chars + 1 filler, so width must be 3*cols-1"
            ));
        }
        let cols = (len + 1) / 3;
        let mut row = Vec::with_capacity(cols);
        for x in 0..cols {
            let c0 = chars[3 * x];
            let c1 = chars[3 * x + 1];
            row.push(classify(c0, c1).map_err(|e| format!("row {y}, col {x}: {e}"))?);
        }
        width = width.max(cols);
        rows.push(row);
    }

    for row in &mut rows {
        row.resize(width, Cell::Empty);
    }
    Ok(rows)
}

/// Write an entity's tile into the world (entity / direction / misc).
/// The items channel is left at 0; item bindings are applied from the header.
fn write_tile(world: &mut World, x: usize, y: usize, item: Item, dir: Direction, misc: Misc) {
    world.set(x, y, Channel::Entities, item as i64);
    world.set(x, y, Channel::Direction, dir as i64);
    world.set(x, y, Channel::Misc, misc as i64);
}

/// Place a multi-tile entity anchored at `(ax, ay)`, validating that the drawn
/// footprint matches `Item::size()`. Returns the occupied tiles (anchor first).
fn place_multi(
    world: &mut World,
    rows: &[Vec<Cell>],
    claimed: &mut [Vec<bool>],
    (ax, ay): (usize, usize),
    (item, misc, dir): (Item, Misc, Direction),
) -> Result<Vec<(usize, usize)>, String> {
    let (w, h) = item.size();
    let square = w == h;
    if !square && dir == Direction::None {
        return Err(format!(
            "multi-tile entity {item:?} at ({ax},{ay}) needs a direction marker"
        ));
    }
    let tiles = entity_tiles(ax, ay, dir, w, h)
        .ok_or_else(|| format!("cannot compute footprint for {item:?} at ({ax},{ay})"))?;

    // Bounding box, to tell perimeter tiles (must show the body char) from
    // interior tiles (may be left blank).
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for p in &tiles {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }

    let grid_h = rows.len() as i64;
    let grid_w = rows[0].len() as i64;
    let mut placed = Vec::with_capacity(tiles.len());
    for p in &tiles {
        if p.x < 0 || p.y < 0 || p.x >= grid_w || p.y >= grid_h {
            return Err(format!(
                "{item:?} at ({ax},{ay}) extends out of bounds to ({},{})",
                p.x, p.y
            ));
        }
        let (tx, ty) = (p.x as usize, p.y as usize);
        if claimed[ty][tx] {
            return Err(format!(
                "{item:?} at ({ax},{ay}) overlaps an entity already placed at ({tx},{ty})"
            ));
        }
        let on_perimeter = p.x == min_x || p.x == max_x || p.y == min_y || p.y == max_y;
        match rows[ty][tx] {
            // A body tile is fine anywhere. For square entities (rotation-
            // independent) we don't insist the drawn directions agree.
            Cell::Entity {
                item: cell_item,
                dir: cell_dir,
                ..
            } if cell_item == item && (square || cell_dir == dir) => {}
            // A blank interior tile is fine, but only in the interior.
            Cell::Interior if !on_perimeter => {}
            _ => {
                return Err(format!(
                    "{item:?} at ({ax},{ay}): tile ({tx},{ty}) breaks the footprint \
                     (expected body '{}'{})",
                    char_for_item(item),
                    if on_perimeter {
                        ""
                    } else {
                        " or blank interior"
                    },
                ));
            }
        }
        write_tile(world, tx, ty, item, dir, misc);
        claimed[ty][tx] = true;
        placed.push((tx, ty));
    }
    Ok(placed)
}

/// The grid half of a parsed factory: the world plus, for each placed entity,
/// its occupied tiles (`tiles[0]` is the anchor). The tile lists let header
/// item bindings reach every footprint tile of a multi-tile entity.
struct ParsedGrid {
    world: World,
    placements: Vec<Vec<(usize, usize)>>,
}

/// Parse the grid body into a [`ParsedGrid`].
#[allow(clippy::needless_range_loop)]
fn parse_grid(body: &str) -> Result<ParsedGrid, String> {
    let rows = tokenize(body)?;
    let height = rows.len();
    let width = rows[0].len();

    let mut world = World::empty(width, height);
    let mut claimed = vec![vec![false; width]; height];
    let mut placements = Vec::new();

    for y in 0..height {
        for x in 0..width {
            if claimed[y][x] {
                continue;
            }
            match rows[y][x] {
                Cell::Empty => {}
                Cell::Interior => {
                    return Err(format!(
                        "blank tile at ({x},{y}) is not inside a multi-tile entity"
                    ));
                }
                Cell::Entity { item, misc, dir } => {
                    let (w, h) = item.size();
                    if w == 1 && h == 1 {
                        write_tile(&mut world, x, y, item, dir, misc);
                        claimed[y][x] = true;
                        placements.push(vec![(x, y)]);
                    } else {
                        let tiles = place_multi(
                            &mut world,
                            &rows,
                            &mut claimed,
                            (x, y),
                            (item, misc, dir),
                        )?;
                        placements.push(tiles);
                    }
                }
            }
        }
    }
    Ok(ParsedGrid { world, placements })
}

// ── YAML header ──────────────────────────────────────────────────────────────

/// The YAML front-matter. All fields optional; unknown keys are rejected so a
/// typo'd key is an error rather than a silent no-op.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Header {
    /// Per-coordinate item bindings. For a source this is what it emits, for a
    /// sink what it counts, for an assembler the recipe (product). The `(x, y)`
    /// may be any tile of a multi-tile entity — it resolves to the whole
    /// footprint.
    #[serde(default)]
    items: Vec<ItemBinding>,
    /// Expected per-sink deliveries, asserted by [`assert_throughput`].
    #[serde(default)]
    throughput: Vec<ThroughputEntry>,
    /// Reserved: an expected-graph description. Parsed but not yet asserted.
    #[serde(default)]
    graph: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ItemBinding {
    x: usize,
    y: usize,
    item: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ThroughputEntry {
    item: String,
    per_second: f64,
}

/// A parsed factory fixture: the world plus the header's expected assertions.
#[derive(Debug)]
pub(crate) struct FactorySpec {
    pub world: World,
    pub expected_throughput: Vec<DeliverySpec>,
    /// Raw `graph:` block from the header, if any. Reserved for a future
    /// edge-set assertion; not interpreted yet.
    pub expected_graph: Option<String>,
}

/// One expected sink delivery: an item and the rate at which it should arrive.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DeliverySpec {
    pub item: Option<Item>,
    pub per_second: f64,
}

/// Split the document into (header YAML, grid body) on the first line that is
/// exactly `---`. With no separator, the whole input is the grid.
fn split_envelope(text: &str) -> (&str, &str) {
    let mut offset = 0;
    for line in text.split_inclusive('\n') {
        if line.trim_end_matches('\n').trim() == "---" {
            return (&text[..offset], &text[offset + line.len()..]);
        }
        offset += line.len();
    }
    ("", text)
}

/// Route header item bindings to the world. Each binding lands on every tile of
/// the entity that occupies its coordinate (so an assembler recipe reaches the
/// anchor `build_graph` reads, whichever tile you name).
fn apply_items(parsed: &mut ParsedGrid, bindings: &[ItemBinding]) -> Result<(), String> {
    for b in bindings {
        let item =
            Item::from_name(&b.item).ok_or_else(|| format!("items: unknown item '{}'", b.item))?;
        let tiles = parsed
            .placements
            .iter()
            .find(|tiles| tiles.contains(&(b.x, b.y)))
            .cloned()
            .ok_or_else(|| format!("items: no entity at ({},{})", b.x, b.y))?;
        for (tx, ty) in tiles {
            parsed.world.set(tx, ty, Channel::Items, item as i64);
        }
    }
    Ok(())
}

/// Parse a full textual factory (YAML header + `---` + grid) into a
/// [`FactorySpec`].
pub(crate) fn parse(text: &str) -> Result<FactorySpec, String> {
    let (header_src, body) = split_envelope(text);
    let header: Header = if header_src.trim().is_empty() {
        Header::default()
    } else {
        serde_yaml::from_str(header_src).map_err(|e| format!("header YAML: {e}"))?
    };

    let mut parsed = parse_grid(body)?;
    apply_items(&mut parsed, &header.items)?;

    let expected_throughput = header
        .throughput
        .iter()
        .map(|t| {
            let item = Item::from_name(&t.item)
                .ok_or_else(|| format!("throughput: unknown item '{}'", t.item))?;
            Ok(DeliverySpec {
                item: Some(item),
                per_second: t.per_second,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(FactorySpec {
        world: parsed.world,
        expected_throughput,
        expected_graph: header.graph,
    })
}

// ── Assertions & rendering ─────────────────────────────────────────────────

/// Build the graph, compute throughput, and assert the per-sink deliveries
/// match the header's `throughput:` as multisets (order-independent, within a
/// float tolerance). Panics with the rendered factory on mismatch.
pub(crate) fn assert_throughput(spec: &FactorySpec) {
    let graph = build_graph(&spec.world);
    let (deliveries, _unreachable) = calc_throughput(&graph);
    let got: Vec<(Option<Item>, f64)> = deliveries.iter().map(|d| (d.item, d.achieved)).collect();
    let want: Vec<(Option<Item>, f64)> = spec
        .expected_throughput
        .iter()
        .map(|d| (d.item, d.per_second))
        .collect();

    let mut remaining = got.clone();
    let mut unmatched_want = Vec::new();
    for &(witem, wrate) in &want {
        match remaining
            .iter()
            .position(|&(gitem, grate)| gitem == witem && (grate - wrate).abs() <= TOL)
        {
            Some(pos) => {
                remaining.remove(pos);
            }
            None => unmatched_want.push((witem, wrate)),
        }
    }

    assert!(
        unmatched_want.is_empty() && remaining.is_empty(),
        "throughput mismatch\n  expected: {want:?}\n  got:      {got:?}\n  \
         unmatched expected: {unmatched_want:?}\n  unmatched got: {remaining:?}\n\
         factory:\n{}",
        render(&spec.world)
    );
}

/// Render a [`World`] back into the grid format (geometry only — item bindings
/// live in the header, not the grid). The inverse of [`parse_grid`] for the
/// shapes the format supports, so `parse_grid(render(w))` round-trips geometry.
#[allow(clippy::needless_range_loop)]
pub(crate) fn render(world: &World) -> String {
    let w = world.width();
    let h = world.height();
    let cols = if w == 0 { 0 } else { 3 * w - 1 };

    // Start every tile as empty (`..`); fillers default to spaces.
    let mut buf = vec![vec![' '; cols]; h];
    for y in 0..h {
        for x in 0..w {
            buf[y][3 * x] = '.';
            buf[y][3 * x + 1] = '.';
        }
    }

    let mut done = vec![vec![false; w]; h];
    for y in 0..h {
        for x in 0..w {
            if done[y][x] {
                continue;
            }
            let item = match world.entity_at(x, y) {
                Some(i) => i,
                None => continue,
            };
            let (ew, eh) = item.size();
            if ew == 1 && eh == 1 {
                buf[y][3 * x] = render_char(item, world.misc_at(x, y));
                buf[y][3 * x + 1] = char_for_dir(world.direction_at(x, y));
                done[y][x] = true;
            } else {
                render_multi(&mut buf, &mut done, world, (x, y), item);
            }
        }
    }

    buf.iter()
        .map(|row| row.iter().collect::<String>().trim_end().to_string())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render one multi-tile entity into the character buffer: a bordered box for
/// square entities (blank interior) and a bracket/stack for linear ones.
fn render_multi(
    buf: &mut [Vec<char>],
    done: &mut [Vec<bool>],
    world: &World,
    (ax, ay): (usize, usize),
    item: Item,
) {
    let (ew, eh) = item.size();
    let dir = world.direction_at(ax, ay);
    let tiles = match entity_tiles(ax, ay, dir, ew, eh) {
        Some(t) => t,
        None => {
            // Degenerate; fall back to a single tile so render never panics.
            buf[ay][3 * ax] = render_char(item, world.misc_at(ax, ay));
            buf[ay][3 * ax + 1] = char_for_dir(dir);
            done[ay][ax] = true;
            return;
        }
    };

    let (mut min_x, mut min_y, mut max_x, mut max_y) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for p in &tiles {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    let multi_row = max_y > min_y;
    let multi_col = max_x > min_x;
    let ech = render_char(item, world.misc_at(ax, ay));
    let dch = char_for_dir(dir);

    for p in &tiles {
        let (tx, ty) = (p.x as usize, p.y as usize);
        done[ty][tx] = true;
        let on_perimeter = p.x == min_x || p.x == max_x || p.y == min_y || p.y == max_y;
        let (c0, c1) = if !on_perimeter {
            (' ', ' ') // blank interior
        } else if multi_row && multi_col {
            (ech, ech) // square box border
        } else if multi_col {
            // horizontal bracket: caps on the ends, direction fill between
            if p.x == min_x {
                (ech, dch)
            } else if p.x == max_x {
                (dch, ech)
            } else {
                (dch, dch)
            }
        } else {
            (ech, dch) // vertical stack
        };
        buf[ty][3 * tx] = c0;
        buf[ty][3 * tx + 1] = c1;
    }

    // Fill the filler between horizontally-adjacent tiles of this entity when
    // both touching characters are non-blank and equal (gives `aaaaaaaa` and
    // `YvvvY` their solid look).
    for p in &tiles {
        let (tx, ty) = (p.x as usize, p.y as usize);
        if tiles.iter().any(|q| q.x == p.x + 1 && q.y == p.y) {
            let right = buf[ty][3 * tx + 1];
            let next_left = buf[ty][3 * (tx + 1)];
            if right != ' ' && right == next_left {
                buf[ty][3 * tx + 2] = right;
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Convenience: parse a grid body and return just the world.
    fn grid(body: &str) -> World {
        parse_grid(body).unwrap().world
    }

    #[test]
    fn test_classify_basic() {
        assert_eq!(classify('.', '.'), Ok(Cell::Empty));
        assert_eq!(classify(' ', ' '), Ok(Cell::Interior));
        assert_eq!(
            classify('b', '>'),
            Ok(Cell::Entity {
                item: Item::TransportBelt,
                misc: Misc::None,
                dir: Direction::East
            })
        );
        // Order does not matter — direction can be either character.
        assert_eq!(classify('>', 'b'), classify('b', '>'));
        // Underground state rides on the entity character.
        assert_eq!(
            classify('d', '>'),
            Ok(Cell::Entity {
                item: Item::UndergroundBelt,
                misc: Misc::UndergroundDown,
                dir: Direction::East
            })
        );
        assert_eq!(
            classify('u', '<'),
            Ok(Cell::Entity {
                item: Item::UndergroundBelt,
                misc: Misc::UndergroundUp,
                dir: Direction::West
            })
        );
        // No-direction entity (e.g. an assembler body tile).
        assert_eq!(
            classify('a', 'a'),
            Ok(Cell::Entity {
                item: Item::AssemblingMachine1,
                misc: Misc::None,
                dir: Direction::None
            })
        );
    }

    #[test]
    fn test_classify_errors() {
        assert!(classify('>', '<').is_err()); // two directions
        assert!(classify('x', 'y').is_err()); // no entity
        assert!(classify('S', 'b').is_err()); // two different entities
    }

    #[test]
    fn test_belt_chain_grid() {
        // The make_belt_chain_world fixture from graph.rs / throughput.rs.
        let w = grid("S> b> b> K> ..");
        assert_eq!(w.width(), 5);
        assert_eq!(w.height(), 1);
        assert_eq!(w.entity_at(0, 0), Some(Item::Source));
        assert_eq!(w.entity_at(1, 0), Some(Item::TransportBelt));
        assert_eq!(w.entity_at(2, 0), Some(Item::TransportBelt));
        assert_eq!(w.entity_at(3, 0), Some(Item::Sink));
        assert_eq!(w.entity_at(4, 0), None);
        for x in 0..4 {
            assert_eq!(w.direction_at(x, 0), Direction::East);
        }
    }

    #[test]
    fn test_all_four_directions() {
        let w = grid(
            "
            b^ b> .. ..
            .. .. bv b<
            ",
        );
        assert_eq!(w.direction_at(0, 0), Direction::North);
        assert_eq!(w.direction_at(1, 0), Direction::East);
        assert_eq!(w.direction_at(2, 1), Direction::South);
        assert_eq!(w.direction_at(3, 1), Direction::West);
    }

    #[test]
    fn test_underground_misc() {
        let w = grid("d> b> u>");
        assert_eq!(w.entity_at(0, 0), Some(Item::UndergroundBelt));
        assert_eq!(w.misc_at(0, 0), Misc::UndergroundDown);
        assert_eq!(w.entity_at(2, 0), Some(Item::UndergroundBelt));
        assert_eq!(w.misc_at(2, 0), Misc::UndergroundUp);
    }

    #[test]
    fn test_ragged_rows_are_padded() {
        // Second row is shorter; it should pad with empty tiles, not error.
        let w = grid(
            "
            S> b> K>
            b>
            ",
        );
        assert_eq!(w.width(), 3);
        assert_eq!(w.height(), 2);
        assert_eq!(w.entity_at(0, 1), Some(Item::TransportBelt));
        assert_eq!(w.entity_at(1, 1), None);
    }

    #[test]
    fn test_grid_errors() {
        // Width not of the form 3*cols-1.
        assert!(parse_grid("b>b>").is_err());
        // Stray blank tile outside any multi-tile entity.
        assert!(parse_grid("b>    b>").is_err());
    }

    #[test]
    fn test_splitter_east_west_stacked() {
        // East/West splitters span vertically (two rows).
        for (body, want) in [("Y>\nY>", Direction::East), ("Y<\nY<", Direction::West)] {
            let w = grid(body);
            assert_eq!(w.entity_at(0, 0), Some(Item::Splitter), "{body}");
            assert_eq!(w.entity_at(0, 1), Some(Item::Splitter), "{body}");
            assert_eq!(w.direction_at(0, 0), want, "{body}");
            assert_eq!(w.direction_at(0, 1), want, "{body}");
        }
    }

    #[test]
    fn test_splitter_north_south_bracket() {
        // North/South splitters span horizontally — the YvvvY bracket look.
        for (body, want) in [("YvvvY", Direction::South), ("Y^^^Y", Direction::North)] {
            let w = grid(body);
            assert_eq!(w.width(), 2, "{body}");
            assert_eq!(w.entity_at(0, 0), Some(Item::Splitter), "{body}");
            assert_eq!(w.entity_at(1, 0), Some(Item::Splitter), "{body}");
            assert_eq!(w.direction_at(0, 0), want, "{body}");
            assert_eq!(w.direction_at(1, 0), want, "{body}");
        }
    }

    #[test]
    fn test_assembler_box() {
        let parsed = parse_grid(
            "
            .. .. .. .. ..
            .. aaaaaaaa ..
            .. aa    aa ..
            .. aaaaaaaa ..
            .. .. .. .. ..
            ",
        )
        .unwrap();
        let w = &parsed.world;
        // 3×3 block of assembler tiles at cols 1..3, rows 1..3.
        for y in 1..=3 {
            for x in 1..=3 {
                assert_eq!(
                    w.entity_at(x, y),
                    Some(Item::AssemblingMachine1),
                    "({x},{y})"
                );
            }
        }
        // Surroundings stay empty.
        assert_eq!(w.entity_at(0, 0), None);
        assert_eq!(w.entity_at(4, 4), None);
        // Exactly one placement, covering all 9 tiles, anchored top-left.
        assert_eq!(parsed.placements.len(), 1);
        assert_eq!(parsed.placements[0].len(), 9);
        assert_eq!(parsed.placements[0][0], (1, 1));
    }

    #[test]
    fn test_multitile_errors() {
        // Splitter without a direction marker.
        assert!(parse_grid("Y.").is_err());
        // Assembler that runs off the edge of the grid.
        assert!(parse_grid("aa").is_err());
        // Assembler with a hole punched in its perimeter.
        assert!(parse_grid(
            "
            aaaaaaaa
            aa    ..
            aaaaaaaa
            "
        )
        .is_err());
    }

    #[test]
    fn test_full_envelope_belt_chain() {
        let spec = parse(
            "
items:
- { x: 0, y: 0, item: copper_plate }
- { x: 3, y: 0, item: copper_plate }
throughput:
- { item: copper_plate, per_second: 15 }
---
S> b> b> K> ..
",
        )
        .unwrap();
        // Geometry.
        assert_eq!(spec.world.entity_at(0, 0), Some(Item::Source));
        assert_eq!(spec.world.entity_at(3, 0), Some(Item::Sink));
        // Item bindings landed on source and sink.
        assert_eq!(spec.world.item_at(0, 0), Some(Item::CopperPlate));
        assert_eq!(spec.world.item_at(3, 0), Some(Item::CopperPlate));
        // Expected throughput parsed into typed form.
        assert_eq!(
            spec.expected_throughput,
            vec![DeliverySpec {
                item: Some(Item::CopperPlate),
                per_second: 15.0,
            }]
        );
    }

    #[test]
    fn test_item_binding_fills_assembler_footprint() {
        // Bind via a NON-anchor tile (2,2); it must still reach the anchor (1,1)
        // and every other footprint tile.
        let spec = parse(
            "
items:
- { x: 2, y: 2, item: iron_gear_wheel }
---
.. .. .. .. ..
.. aaaaaaaa ..
.. aa    aa ..
.. aaaaaaaa ..
.. .. .. .. ..
",
        )
        .unwrap();
        for y in 1..=3 {
            for x in 1..=3 {
                assert_eq!(
                    spec.world.item_at(x, y),
                    Some(Item::IronGearWheel),
                    "({x},{y})"
                );
            }
        }
    }

    #[test]
    fn test_envelope_without_header() {
        // No `---` → whole input is the grid, header empty.
        let spec = parse("S> b> K>").unwrap();
        assert_eq!(spec.world.width(), 3);
        assert!(spec.expected_throughput.is_empty());
        assert!(spec.expected_graph.is_none());
    }

    #[test]
    fn test_graph_field_captured() {
        let spec = parse(
            "
graph: |
  S@0,0 -> b@1,0
---
S> b> K>
",
        )
        .unwrap();
        assert_eq!(spec.expected_graph.as_deref(), Some("S@0,0 -> b@1,0\n"));
    }

    #[test]
    fn test_header_errors() {
        // Unknown item name.
        assert!(parse("items:\n- { x: 0, y: 0, item: nope }\n---\nS>").is_err());
        // Item binding on an empty tile.
        assert!(parse("items:\n- { x: 1, y: 0, item: copper_plate }\n---\nS> ..").is_err());
        // Unknown header key (deny_unknown_fields).
        assert!(parse("bogus: 1\n---\nS>").is_err());
    }

    #[test]
    fn test_render_roundtrip_geometry() {
        // Exercises every shape: single tiles, both undergrounds, a vertical
        // splitter, and a boxed assembler.
        let w1 = grid(
            "
            S> b> d> .. u> b> K>
            .. .. .. .. .. .. ..
            Y> .. aaaaaaaa .. ..
            Y> .. aa    aa .. ..
            .. .. aaaaaaaa .. ..
            ",
        );
        let text = render(&w1);
        let w2 = grid(&text);

        assert_eq!(w1.width(), w2.width(), "rendered:\n{text}");
        assert_eq!(w1.height(), w2.height(), "rendered:\n{text}");
        for y in 0..w1.height() {
            for x in 0..w1.width() {
                assert_eq!(
                    w1.entity_at(x, y),
                    w2.entity_at(x, y),
                    "entity ({x},{y})\n{text}"
                );
                assert_eq!(
                    w1.direction_at(x, y),
                    w2.direction_at(x, y),
                    "dir ({x},{y})\n{text}"
                );
                assert_eq!(w1.misc_at(x, y), w2.misc_at(x, y), "misc ({x},{y})\n{text}");
            }
        }
    }

    #[test]
    fn test_assert_throughput_belt_chain() {
        // End-to-end: parse → build_graph → calc_throughput → assert. A belt
        // tops out at 15 items/s, so an infinite source delivers 15 to the sink.
        let spec = parse(
            "
items:
- { x: 0, y: 0, item: copper_cable }
- { x: 3, y: 0, item: copper_cable }
throughput:
- { item: copper_cable, per_second: 15 }
---
S> b> b> K> ..
",
        )
        .unwrap();
        assert_throughput(&spec);
    }

    #[test]
    #[should_panic(expected = "throughput mismatch")]
    fn test_assert_throughput_detects_wrong_rate() {
        let spec = parse(
            "
items:
- { x: 0, y: 0, item: copper_cable }
- { x: 3, y: 0, item: copper_cable }
throughput:
- { item: copper_cable, per_second: 999 }
---
S> b> b> K> ..
",
        )
        .unwrap();
        assert_throughput(&spec);
    }

    // ── Ports of existing throughput.rs tests, to show the format end-to-end ──

    #[test]
    fn test_port_inserter_limited() {
        // throughput.rs::test_source_inserter_belt_sink — an inserter caps the
        // line at its 0.86 i/s rate.
        let spec = parse(
            "
items:
- { x: 0, y: 0, item: copper_cable }
- { x: 3, y: 0, item: copper_cable }
throughput:
- { item: copper_cable, per_second: 0.86 }
---
S> i> b> K>
",
        )
        .unwrap();
        assert_throughput(&spec);
    }

    #[test]
    fn test_port_splitter_split() {
        // throughput.rs::test_splitter_split — one input fans out through a
        // splitter to two sinks, 7.5 i/s each.
        let spec = parse(
            "
items:
- { x: 0, y: 0, item: copper_cable }
- { x: 4, y: 0, item: copper_cable }
- { x: 4, y: 1, item: copper_cable }
throughput:
- { item: copper_cable, per_second: 7.5 }
- { item: copper_cable, per_second: 7.5 }
---
S> b> Y> b> K> ..
.. .. Y> b> K> ..
",
        )
        .unwrap();
        assert_throughput(&spec);
    }

    #[test]
    fn test_port_disconnected_scores_zero() {
        // throughput.rs::test_disconnected_entities — no path source→sink, so
        // the sink's delivery is 0.
        let spec = parse(
            "
items:
- { x: 0, y: 0, item: copper_cable }
- { x: 4, y: 4, item: copper_cable }
throughput:
- { item: copper_cable, per_second: 0 }
---
S> .. .. .. ..
.. .. .. .. ..
.. .. b> .. ..
.. .. .. .. ..
.. .. .. .. K>
",
        )
        .unwrap();
        assert_throughput(&spec);
    }
}
