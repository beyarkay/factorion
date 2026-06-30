//! Render a [`World`] back into the two-character ASCII grid format used by
//! the textual factory fixtures (and exposed to Python as
//! `factorion_rs.render_factory`).
//!
//! Every tile is exactly **two characters**, tiles separated by one filler
//! character. For a single-tile entity the two characters are its registry
//! character (see [`ENTITY_CHARS`]) and a direction marker (`^ > v <`, or `.`
//! for [`Direction::None`]); an empty tile is `..`. Multi-tile entities
//! (assembler, splitter) draw their body across the whole footprint with a
//! blank (`  `) interior. This is the inverse of the textual parser, so
//! `parse_grid(render(w))` round-trips a world's geometry.
//!
//! The character tables and the low-level `render_char`/`char_for_dir`/`bbox`/
//! `on_perimeter` helpers are shared with the (test-only) textual parser in
//! [`crate::textual`].

use crate::entities::entity_tiles;
use crate::types::{Direction, Item, Misc, Pos};
use crate::world::World;

/// Registry of entity characters. The single source of truth for how each
/// placeable [`Item`] is drawn (and parsed). Underground belts carry their
/// [`Misc`] state so the down/up entrances render distinctly.
pub(crate) const ENTITY_CHARS: &[(char, Item, Misc)] = &[
    ('b', Item::TransportBelt, Misc::None),
    ('i', Item::Inserter, Misc::None),
    ('l', Item::LongHandedInserter, Misc::None),
    ('a', Item::AssemblingMachine1, Misc::None),
    ('Y', Item::Splitter, Misc::None),
    ('d', Item::UndergroundBelt, Misc::UndergroundDown),
    ('u', Item::UndergroundBelt, Misc::UndergroundUp),
    ('S', Item::Source, Misc::None),
    ('K', Item::Sink, Misc::None),
];

/// Direction markers. Deliberately disjoint from every [`ENTITY_CHARS`] entry.
pub(crate) const DIR_CHARS: &[(char, Direction)] = &[
    ('^', Direction::North),
    ('>', Direction::East),
    ('v', Direction::South),
    ('<', Direction::West),
];

/// Grid character for rendering an entity, honouring its underground state
/// (so `UndergroundBelt` + `UndergroundUp` renders as `u`, not `d`).
pub(crate) fn render_char(item: Item, misc: Misc) -> char {
    ENTITY_CHARS
        .iter()
        .find(|(_, i, m)| *i == item && *m == misc)
        .or_else(|| ENTITY_CHARS.iter().find(|(_, i, _)| *i == item))
        .map(|(c, _, _)| *c)
        .unwrap_or('?')
}

/// Direction character for rendering; [`Direction::None`] renders as `.`.
pub(crate) fn char_for_dir(dir: Direction) -> char {
    DIR_CHARS
        .iter()
        .find(|(_, d)| *d == dir)
        .map(|(c, _)| *c)
        .unwrap_or('.')
}

/// Axis-aligned bounding box `(min_x, min_y, max_x, max_y)` of a tile set.
/// `tiles` is always non-empty (an entity occupies at least its anchor).
pub(crate) fn bbox(tiles: &[Pos]) -> (i64, i64, i64, i64) {
    let (mut min_x, mut min_y, mut max_x, mut max_y) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for p in tiles {
        min_x = min_x.min(p.x);
        min_y = min_y.min(p.y);
        max_x = max_x.max(p.x);
        max_y = max_y.max(p.y);
    }
    (min_x, min_y, max_x, max_y)
}

/// Whether `p` lies on the perimeter of the bounding box (vs. its interior).
pub(crate) fn on_perimeter(p: Pos, (min_x, min_y, max_x, max_y): (i64, i64, i64, i64)) -> bool {
    p.x == min_x || p.x == max_x || p.y == min_y || p.y == max_y
}

/// Render a [`World`] back into the grid format (geometry only — item bindings
/// live in the header, not the grid). The inverse of `parse_grid` for the
/// shapes the format supports, so `parse_grid(render(w))` round-trips geometry.
pub(crate) fn render(world: &World) -> String {
    let w = world.width();
    let h = world.height();
    let cols = if w == 0 { 0 } else { 3 * w - 1 };

    // Start every tile as empty (`..`); fillers default to spaces.
    let mut buf = vec![vec![' '; cols]; h];
    for row in buf.iter_mut() {
        for x in 0..w {
            row[3 * x] = '.';
            row[3 * x + 1] = '.';
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

    let bb = bbox(&tiles);
    let (min_x, min_y, max_x, max_y) = bb;
    let multi_row = max_y > min_y;
    let multi_col = max_x > min_x;
    let ech = render_char(item, world.misc_at(ax, ay));
    let dch = char_for_dir(dir);

    for p in &tiles {
        let (tx, ty) = (p.x as usize, p.y as usize);
        done[ty][tx] = true;
        let (c0, c1) = if !on_perimeter(*p, bb) {
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

    /// Every placeable item has an entity character, and no entry maps to a
    /// non-placeable item. Adding a placeable entity without wiring it into the
    /// renderer would otherwise silently produce an unparseable `?`.
    #[test]
    fn test_entity_registry_matches_placeable_items() {
        for &item in crate::types::all_items() {
            if item.is_placeable() {
                assert!(
                    ENTITY_CHARS.iter().any(|(_, i, _)| *i == item),
                    "placeable {item:?} has no ENTITY_CHARS entry"
                );
            }
        }
        for (ch, item, _) in ENTITY_CHARS {
            assert!(
                item.is_placeable(),
                "ENTITY_CHARS entry '{ch}' maps to non-placeable {item:?}"
            );
        }
    }

    #[test]
    fn test_render_single_tile_chain() {
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        assert_eq!(render(&w), "S> b> K>");
    }

    #[test]
    fn test_render_assembler_box() {
        // A 3x3 assembler renders as a bordered box with a blank interior.
        let mut w = World::empty(3, 3);
        w.place(
            0,
            0,
            Item::AssemblingMachine1,
            Direction::North,
            Some(Item::ElectronicCircuit),
        );
        assert_eq!(render(&w), "aaaaaaaa\naa    aa\naaaaaaaa");
    }

    /// The on-disk grid a fixture author wrote, canonicalized the same way the
    /// parser consumes it: trim each line, drop blank lines and `#` comments.
    /// `render` emits exactly this shape (no leading indent, trailing trimmed),
    /// so a faithful round-trip is byte-for-byte equality against it.
    fn canonical_grid(grid_src: &str) -> String {
        grid_src
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Full round-trip over every fixture on disk: parse each YAML factory into
    /// a `World` (disk → factory), render it back to a grid string (factory →
    /// string), and assert the string is byte-for-byte the grid that was on
    /// disk. Geometry survives the trip unchanged for every shape the format
    /// supports — this is the render⇄parse inverse, exercised against the whole
    /// fixture corpus rather than a handful of hand-built worlds.
    #[test]
    fn test_render_roundtrips_every_fixture_on_disk() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/factories");
        let read = std::fs::read_dir(dir);
        assert!(
            read.is_ok(),
            "cannot read factories dir {dir}: {:?}",
            read.err()
        );
        let mut files: Vec<std::path::PathBuf> = read
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
            .collect();
        files.sort();
        assert!(!files.is_empty(), "no .yaml factory files found in {dir}");

        let mut checked = 0usize;
        for path in &files {
            let display = path.display();
            let text = std::fs::read_to_string(path);
            assert!(text.is_ok(), "cannot read {display}: {:?}", text.err());
            // The (test-only) textual parser is the inverse under test.
            let specs = crate::textual::parse_many(&text.unwrap());
            assert!(specs.is_ok(), "cannot parse {display}: {:?}", specs.err());
            for (i, spec) in specs.unwrap().iter().enumerate() {
                let want = canonical_grid(&spec.grid_src);
                let got = render(&spec.world);
                assert_eq!(
                    got,
                    want,
                    "\n===== render round-trip changed the grid in {display} (factory #{}) =====\
                     \n--- on disk ---\n{want}\n--- render(parse(disk)) ---\n{got}\n",
                    i + 1
                );
                checked += 1;
            }
        }
        assert!(checked > 0, "no factory documents were checked");
    }
}
