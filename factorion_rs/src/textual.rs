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
//! Adding a new entity is a one-line change to [`ENTITY_CHARS`]; its footprint
//! comes from [`Item::size`], so the multi-tile machinery picks it up for free.

use crate::types::{Channel, Direction, Item, Misc};
use crate::world::World;

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

fn dir_for_char(c: char) -> Option<Direction> {
    DIR_CHARS.iter().find(|(ch, _)| *ch == c).map(|(_, d)| *d)
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

/// Write a single-tile entity into the world (entity / direction / misc).
/// The items channel is left at 0; item bindings are applied from the header.
fn place_single(world: &mut World, x: usize, y: usize, item: Item, dir: Direction, misc: Misc) {
    world.set(x, y, Channel::Entities, item as i64);
    world.set(x, y, Channel::Direction, dir as i64);
    world.set(x, y, Channel::Misc, misc as i64);
}

/// Parse the grid body into a [`World`]. Single-tile entities only — multi-tile
/// resolution is layered on next.
fn parse_grid(body: &str) -> Result<World, String> {
    let rows = tokenize(body)?;
    let height = rows.len();
    let width = rows[0].len();

    let mut world = World::empty(width, height);
    for (y, row) in rows.iter().enumerate() {
        for (x, cell) in row.iter().enumerate() {
            match *cell {
                Cell::Empty => {}
                Cell::Interior => {
                    return Err(format!(
                        "blank tile at ({x},{y}) is not inside a multi-tile entity"
                    ));
                }
                Cell::Entity { item, misc, dir } => {
                    let (w, h) = item.size();
                    if w == 1 && h == 1 {
                        place_single(&mut world, x, y, item, dir, misc);
                    } else {
                        return Err(format!(
                            "multi-tile entity {item:?} at ({x},{y}) is not yet supported"
                        ));
                    }
                }
            }
        }
    }
    Ok(world)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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
        let w = parse_grid("S> b> b> K> ..").unwrap();
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
        let w = parse_grid(
            "
            b^ b> .. ..
            .. .. bv b<
            ",
        )
        .unwrap();
        assert_eq!(w.direction_at(0, 0), Direction::North);
        assert_eq!(w.direction_at(1, 0), Direction::East);
        assert_eq!(w.direction_at(2, 1), Direction::South);
        assert_eq!(w.direction_at(3, 1), Direction::West);
    }

    #[test]
    fn test_underground_misc() {
        let w = parse_grid("d> b> u>").unwrap();
        assert_eq!(w.entity_at(0, 0), Some(Item::UndergroundBelt));
        assert_eq!(w.misc_at(0, 0), Misc::UndergroundDown);
        assert_eq!(w.entity_at(2, 0), Some(Item::UndergroundBelt));
        assert_eq!(w.misc_at(2, 0), Misc::UndergroundUp);
    }

    #[test]
    fn test_ragged_rows_are_padded() {
        // Second row is shorter; it should pad with empty tiles, not error.
        let w = parse_grid(
            "
            S> b> K>
            b>
            ",
        )
        .unwrap();
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
}
