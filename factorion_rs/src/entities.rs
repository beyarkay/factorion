use std::collections::HashMap;

use crate::types::{get_recipe, Direction, Item, Misc, NodeId, Pos};
use crate::world::World;

/// An edge in the factory graph: (source_node, destination_node).
pub type Edge = (NodeId, NodeId);

/// How far an underground-belt entrance reaches for its paired exit
/// (delta 1..UNDERGROUND_REACH — up to 5 tiles apart, like a yellow belt).
const UNDERGROUND_REACH: i64 = 6;

/// Trait abstracting over factory entity types.
///
/// Each entity type implements this trait to define:
/// - How it connects to neighboring entities (graph edges)
/// - How it transforms input flow into output flow
/// - Its maximum throughput rate
pub trait FactoryEntity {
    /// Which entity kind this is. Used to look up flow_rate from the
    /// single source of truth in Item::flow_rate().
    fn kind(&self) -> Item;

    /// Return the edges this entity contributes to the graph.
    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge>;

    /// Given accumulated input flow rates, compute output flow rates.
    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64>;

    /// Maximum items/second this entity can transfer.
    /// Default delegates to Item::flow_rate().
    fn flow_rate(&self) -> f64 {
        self.kind().flow_rate()
    }
}

/// Stack-allocated entity dispatch. Wraps concrete entity structs.
/// Only constructible from placeable Items.
pub enum EntityEnum {
    TransportBelt(TransportBelt),
    Inserter(Inserter),
    AssemblingMachine(AssemblingMachine),
    UndergroundBelt(UndergroundBelt),
    Sink(Sink),
    Source(Source),
    Splitter(Splitter),
}

impl EntityEnum {
    /// Build an EntityEnum from a placeable Item. Returns `None` for
    /// non-placeable items (caller-side data error — non-placeable
    /// items should never appear in the entities channel).
    pub fn new(kind: Item, item: Option<Item>, misc: Misc) -> Option<Self> {
        Some(match kind {
            Item::TransportBelt => Self::TransportBelt(TransportBelt),
            Item::Inserter => Self::Inserter(Inserter),
            Item::AssemblingMachine1 => {
                Self::AssemblingMachine(AssemblingMachine { recipe_item: item })
            }
            Item::UndergroundBelt => Self::UndergroundBelt(UndergroundBelt { misc }),
            Item::Sink => Self::Sink(Sink),
            Item::Source => Self::Source(Source { item }),
            Item::Splitter => Self::Splitter(Splitter),
            _ => return None,
        })
    }
}

impl FactoryEntity for EntityEnum {
    fn kind(&self) -> Item {
        match self {
            Self::TransportBelt(e) => e.kind(),
            Self::Inserter(e) => e.kind(),
            Self::AssemblingMachine(e) => e.kind(),
            Self::UndergroundBelt(e) => e.kind(),
            Self::Sink(e) => e.kind(),
            Self::Source(e) => e.kind(),
            Self::Splitter(e) => e.kind(),
        }
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        match self {
            Self::TransportBelt(e) => e.connections(pos, dir, world),
            Self::Inserter(e) => e.connections(pos, dir, world),
            Self::AssemblingMachine(e) => e.connections(pos, dir, world),
            Self::UndergroundBelt(e) => e.connections(pos, dir, world),
            Self::Sink(e) => e.connections(pos, dir, world),
            Self::Source(e) => e.connections(pos, dir, world),
            Self::Splitter(e) => e.connections(pos, dir, world),
        }
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        match self {
            Self::TransportBelt(e) => e.transform_flow(input),
            Self::Inserter(e) => e.transform_flow(input),
            Self::AssemblingMachine(e) => e.transform_flow(input),
            Self::UndergroundBelt(e) => e.transform_flow(input),
            Self::Sink(e) => e.transform_flow(input),
            Self::Source(e) => e.transform_flow(input),
            Self::Splitter(e) => e.transform_flow(input),
        }
    }
}

// ── Transport Belt ──────────────────────────────────────────────────────────

pub struct TransportBelt;

impl FactoryEntity for TransportBelt {
    fn kind(&self) -> Item {
        Item::TransportBelt
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        belt_connections(Item::TransportBelt, pos, dir, world)
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        input
            .iter()
            .map(|(&item, &rate)| (item, rate.min(self.flow_rate())))
            .collect()
    }
}

// ── Inserter ────────────────────────────────────────────────────────────────

pub struct Inserter;

impl FactoryEntity for Inserter {
    fn kind(&self) -> Item {
        Item::Inserter
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        inserter_connections(Item::Inserter, pos, dir, world)
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        input
            .iter()
            .map(|(&item, &rate)| (item, rate.min(self.flow_rate())))
            .collect()
    }
}

// ── Assembling Machine ──────────────────────────────────────────────────────

pub struct AssemblingMachine {
    recipe_item: Option<Item>,
}

impl FactoryEntity for AssemblingMachine {
    fn kind(&self) -> Item {
        Item::AssemblingMachine1
    }

    fn connections(&self, pos: (usize, usize), _dir: Direction, world: &World) -> Vec<Edge> {
        let mut edges = Vec::new();
        let (x, y) = pos;
        let self_id = NodeId::new(Item::AssemblingMachine1, x, y);

        // Assembling machines are 3x3. The anchor is the top-left corner.
        // Search the perimeter (ring around the 3x3) for inserters.
        for ddx in -1i64..=3 {
            let nx = x as i64 + ddx;
            if nx < 0 || nx >= world.width() as i64 {
                continue;
            }
            for ddy in -1i64..=3 {
                let ny = y as i64 + ddy;
                if ny < 0 || ny >= world.height() as i64 {
                    continue;
                }
                // Skip tiles inside the 3x3 assembler body
                if (0..3).contains(&ddx) && (0..3).contains(&ddy) {
                    continue;
                }
                // Skip corners: only the orthogonal perimeter slots can hold an
                // inserter that reaches into the assembler body.
                if (ddx == -1 || ddx == 3) && (ddy == -1 || ddy == 3) {
                    continue;
                }

                let nx_u = nx as usize;
                let ny_u = ny as usize;
                let other_entity = match world.entity_at(nx_u, ny_u) {
                    Some(e) => e,
                    None => continue,
                };

                // Only inserter-like entities can interact with assembling
                // machines. The Source and Sink markers count too: they insert
                // into / pull from the assembler exactly like a plain inserter.
                if !matches!(other_entity, Item::Inserter | Item::Source | Item::Sink) {
                    continue;
                }

                let other_dir = world.direction_at(nx_u, ny_u);
                let other_id = NodeId::new(other_entity, nx_u, ny_u);

                // Determine edge direction between assembler and the adjacent inserter-like entity.
                // When the entity's facing direction points away from the assembler body
                // (e.g. entity is above assembler and faces north), items flow OUT of the
                // assembler: assembler → entity. Otherwise, items flow IN: entity → assembler.
                let assembler_outputs_to_entity = (other_dir == Direction::North && ddy < 0)
                    || (other_dir == Direction::South && ddy > 0)
                    || (other_dir == Direction::West && ddx < 0)
                    || (other_dir == Direction::East && ddx > 0);

                if assembler_outputs_to_entity {
                    // Assembler → entity (entity takes from assembler)
                    edges.push((self_id.clone(), other_id));
                } else {
                    // Entity → Assembler (entity feeds into assembler)
                    edges.push((other_id, self_id.clone()));
                }
            }
        }

        edges
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        let recipe_item = match self.recipe_item {
            Some(i) => i,
            None => return HashMap::new(),
        };

        let recipe = match get_recipe(recipe_item) {
            Some(r) => r,
            None => return HashMap::new(),
        };

        // Find the minimum ratio of available input to required input
        let mut min_ratio: f64 = 1.0;
        for &(item, required) in recipe.consumes.iter() {
            let available = input.get(&item).copied().unwrap_or(0.0);
            let ratio = available / required;
            min_ratio = min_ratio.min(ratio);
        }

        // Produce outputs scaled by the minimum ratio
        recipe
            .produces
            .iter()
            .map(|&(item, rate)| (item, rate * min_ratio))
            .collect()
    }
}

// ── Underground Belt ────────────────────────────────────────────────────────

pub struct UndergroundBelt {
    misc: Misc,
}

impl FactoryEntity for UndergroundBelt {
    fn kind(&self) -> Item {
        Item::UndergroundBelt
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        let mut edges = Vec::new();
        let (x, y) = pos;
        let self_id = NodeId::new(Item::UndergroundBelt, x, y);
        if dir == Direction::None {
            return edges;
        }
        let (dx, dy) = dir.delta();

        match self.misc {
            // Exit: items come *out* and flow onto the cell directly ahead,
            // exactly like a belt's output — a belt or an underground
            // *entrance*, unless it faces back at us. Nothing flows *into* an
            // exit (its input is the tunnel, not the adjacent cell).
            Misc::UndergroundUp => {
                let (ax, ay) = (x as i64 + dx, y as i64 + dy);
                if world.in_bounds(ax, ay) {
                    let (au, av) = (ax as usize, ay as usize);
                    if let Some(dst) = world.entity_at(au, av) {
                        let droppable =
                            matches!(dst, Item::TransportBelt | Item::Source | Item::Sink)
                                || (dst == Item::UndergroundBelt
                                    && world.misc_at(au, av) == Misc::UndergroundDown);
                        if droppable && world.direction_at(au, av) != dir.opposite() {
                            edges.push((self_id, NodeId::new(dst, au, av)));
                        }
                    }
                }
            }
            // Entrance: its output goes underground to the paired exit — the
            // first same-direction exit within reach. The tunnel passes beneath
            // non-underground entities; the first underground belt it meets ends
            // the search, and pairs only if that belt is a matching exit.
            Misc::UndergroundDown => {
                for delta in 1..UNDERGROUND_REACH {
                    let (tx, ty) = (x as i64 + dx * delta, y as i64 + dy * delta);
                    if !world.in_bounds(tx, ty) {
                        break;
                    }
                    let (tu, tv) = (tx as usize, ty as usize);
                    if let Some(e) = world.entity_at(tu, tv) {
                        if e == Item::UndergroundBelt {
                            if world.misc_at(tu, tv) == Misc::UndergroundUp
                                && world.direction_at(tu, tv) == dir
                            {
                                edges.push((self_id, NodeId::new(e, tu, tv)));
                            }
                            break;
                        }
                    }
                }
            }
            _ => {}
        }

        edges
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        input
            .iter()
            .map(|(&item, &rate)| (item, rate.min(self.flow_rate())))
            .collect()
    }
}

// ── Source (stack_inserter) ─────────────────────────────────────────────────
//
// A source (stack_inserter) uses the same inserter connection logic: picks up
// from behind, drops onto belts/assemblers ahead. Its special behavior is that
// it has infinite output.

pub struct Source {
    item: Option<Item>,
}

impl FactoryEntity for Source {
    fn kind(&self) -> Item {
        Item::Source
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        // A source connects exactly like a belt (never to another source/sink).
        belt_connections(Item::Source, pos, dir, world)
    }

    fn transform_flow(&self, _input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        // A Source with no item set produces nothing. With one set,
        // it produces an infinite flow of that item.
        let mut output = HashMap::new();
        if let Some(item) = self.item {
            output.insert(item, f64::INFINITY);
        }
        output
    }
}

// ── Sink (bulk_inserter) ────────────────────────────────────────────────────
//
// Same as Source: bulk_inserter contains "inserter", so it uses inserter
// connection logic. Its special behavior is infinite throughput at the output.

pub struct Sink;

impl FactoryEntity for Sink {
    fn kind(&self) -> Item {
        Item::Sink
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        // A sink connects exactly like a belt (never to another source/sink).
        belt_connections(Item::Sink, pos, dir, world)
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        // Sinks pass through everything (infinite capacity)
        input.clone()
    }
}

/// Belt-style connections, shared by transport belts and the source/sink
/// markers — a source/sink connects exactly like a belt, except it never links
/// to another source/sink.
///
/// Pulls from a same-direction belt-like (belt/source/sink) or underground exit
/// behind; drops onto a belt-like or underground *entrance* ahead (and
/// side-loads onto a perpendicular exit). Nothing flows inline into an exit.
fn belt_connections(
    self_kind: Item,
    pos: (usize, usize),
    dir: Direction,
    world: &World,
) -> Vec<Edge> {
    let mut edges = Vec::new();
    let (x, y) = pos;
    let (dx, dy) = dir.delta();
    let self_id = NodeId::new(self_kind, x, y);
    let self_is_ss = matches!(self_kind, Item::Source | Item::Sink);
    let belt_like = |it: Item| matches!(it, Item::TransportBelt | Item::Source | Item::Sink);

    // Pull from the cell behind if it's a same-direction belt-like or
    // underground exit (an entrance sends its items underground instead).
    let (bx, by) = (x as i64 - dx, y as i64 - dy);
    if world.in_bounds(bx, by) {
        let (bx, by) = (bx as usize, by as usize);
        if let Some(src) = world.entity_at(bx, by) {
            let beltish = (belt_like(src)
                || (src == Item::UndergroundBelt
                    && world.misc_at(bx, by) != Misc::UndergroundDown))
                && world.direction_at(bx, by) == dir;
            let blocked = self_is_ss && matches!(src, Item::Source | Item::Sink);
            if beltish && !blocked {
                edges.push((NodeId::new(src, bx, by), self_id.clone()));
            }
        }
    }

    // Drop onto the cell ahead: a belt-like or an underground entrance (never
    // inline into an exit, though a perpendicular exit can be side-loaded).
    let (fx, fy) = (x as i64 + dx, y as i64 + dy);
    if world.in_bounds(fx, fy) {
        let (fx, fy) = (fx as usize, fy as usize);
        if let Some(dst) = world.entity_at(fx, fy) {
            let dst_dir = world.direction_at(fx, fy);
            let dst_misc = world.misc_at(fx, fy);
            let droppable = belt_like(dst)
                || (dst == Item::UndergroundBelt && dst_misc == Misc::UndergroundDown);
            let opposing = droppable && dst_dir == dir.opposite();
            let exit_sideload = dst == Item::UndergroundBelt
                && dst_misc == Misc::UndergroundUp
                && dst_dir != dir
                && dst_dir != dir.opposite();
            let blocked = self_is_ss && matches!(dst, Item::Source | Item::Sink);
            if ((droppable && !opposing) || exit_sideload) && !blocked {
                edges.push((self_id, NodeId::new(dst, fx, fy)));
            }
        }
    }

    edges
}

/// Inserter connection logic: pick up from the belt-like / underground /
/// assembler behind, and drop onto the belt-like / underground / assembler
/// ahead. Source/sink count as belt-like entities here (they connect like belts).
fn inserter_connections(
    self_kind: Item,
    pos: (usize, usize),
    dir: Direction,
    world: &World,
) -> Vec<Edge> {
    let mut edges = Vec::new();
    let (x, y) = pos;
    let (dx, dy) = dir.delta();
    let self_id = NodeId::new(self_kind, x, y);

    // Pick up from behind (opposite of facing direction). A real inserter can
    // only pick up from an entity that carries or produces items: a source,
    // belt, underground belt, or assembler. Notably NOT another inserter (see
    // #122) or a sink. (Splitters are excluded too, mirroring the drop side —
    // not quite true to Factorio, may revisit.)
    //
    // The source/sink markers share this function but keep the permissive
    // pickup: a sink legitimately receives from the inserter/belt/assembler
    // behind it — an inserter can't *drop* onto a sink (see the drop filter
    // below), so delivery is modelled as the sink pulling — and a source's
    // input is ignored anyway (its output is infinite).
    let src_x = x as i64 - dx;
    let src_y = y as i64 - dy;
    if world.in_bounds(src_x, src_y) {
        let sx = src_x as usize;
        let sy = src_y as usize;
        if let Some(src_entity) = world.entity_at(sx, sy) {
            let src_is_pickable = matches!(
                src_entity,
                Item::Source
                    | Item::Sink
                    | Item::TransportBelt
                    | Item::UndergroundBelt
                    | Item::AssemblingMachine1
            );
            if src_is_pickable {
                edges.push((NodeId::new(src_entity, sx, sy), self_id.clone()));
            }
        }
    }

    // Drop onto the cell ahead (in facing direction)
    let dst_x = x as i64 + dx;
    let dst_y = y as i64 + dy;
    if world.in_bounds(dst_x, dst_y) {
        let dx_u = dst_x as usize;
        let dy_u = dst_y as usize;
        if let Some(dst_entity) = world.entity_at(dx_u, dy_u) {
            // Can only insert into belts or assembling machines
            let dst_is_insertable = matches!(
                dst_entity,
                Item::TransportBelt
                    | Item::UndergroundBelt
                    | Item::AssemblingMachine1
                    | Item::Source
                    | Item::Sink
            );
            if dst_is_insertable {
                edges.push((self_id, NodeId::new(dst_entity, dx_u, dy_u)));
            }
        }
    }

    edges
}

/// Compute all tiles occupied by an entity given its anchor, direction, and size.
///
/// `width` is the entity's extent perpendicular to flow direction.
/// `height` is the entity's extent along the flow direction.
/// For a 1x1 entity, returns just the anchor. For a 2x1 entity facing east,
/// returns [(x,y), (x,y+1)] since the width extends along +Y (perpendicular to east).
///
/// This works for any entity size: 1x1, 3x3 (assembler), 2x2 (furnace), etc.
pub fn entity_tiles(
    x: usize,
    y: usize,
    dir: Direction,
    width: usize,
    height: usize,
) -> Option<Vec<Pos>> {
    if width == 1 && height == 1 {
        return Some(vec![Pos::new(x as i64, y as i64)]);
    }
    // For square entities, direction doesn't affect the footprint.
    let effective_dir = if width == height {
        Direction::East
    } else {
        dir
    };
    let mut tiles = Vec::with_capacity(width * height);
    for w in 0..width {
        for h in 0..height {
            let (dx, dy) = match effective_dir {
                Direction::East => (h as i64, w as i64),
                Direction::West => (-(h as i64), w as i64),
                Direction::North => (w as i64, -(h as i64)),
                Direction::South => (w as i64, h as i64),
                Direction::None => return None,
            };
            tiles.push(Pos::new(x as i64 + dx, y as i64 + dy));
        }
    }
    Some(tiles)
}

// ── Splitter ───────────────────────────────────────────────────────────────
//
// A splitter is 2 tiles wide (perpendicular to flow) and 1 tile deep.
// It has up to 2 inputs (behind both tiles) and up to 2 outputs (ahead of both tiles).
// Flow splitting (dividing output among successors) is handled in calc_throughput,
// not here — transform_flow just caps at the flow rate.

pub struct Splitter;

impl FactoryEntity for Splitter {
    fn kind(&self) -> Item {
        Item::Splitter
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        let mut edges = Vec::new();
        let (x, y) = pos;
        let (dx, dy) = dir.delta();
        let self_id = NodeId::new(Item::Splitter, x, y);

        let tiles = match entity_tiles(x, y, dir, 2, 1) {
            Some(t) => t,
            None => return edges,
        };
        let tile_set: std::collections::HashSet<Pos> = tiles.iter().copied().collect();

        for &tile in &tiles {
            // Input: cell behind this tile (opposite of facing direction)
            let in_pos = Pos::new(tile.x - dx, tile.y - dy);
            if let Some((ix, iy)) = in_pos.to_usize() {
                if world.in_bounds(in_pos.x, in_pos.y) {
                    if let Some(src_entity) = world.entity_at(ix, iy) {
                        let src_dir = world.direction_at(ix, iy);
                        // Only accept belt-like entities pointing into the splitter
                        let src_is_belt =
                            matches!(src_entity, Item::TransportBelt | Item::UndergroundBelt)
                                && src_dir == dir;
                        // Also accept sources/sinks (they use inserter-style connections)
                        let src_is_source_sink =
                            matches!(src_entity, Item::Source | Item::Sink) && src_dir == dir;
                        if (src_is_belt || src_is_source_sink) && !tile_set.contains(&in_pos) {
                            edges.push((NodeId::new(src_entity, ix, iy), self_id.clone()));
                        }
                    }
                }
            }

            // Output: cell ahead of this tile (in facing direction)
            let out_pos = Pos::new(tile.x + dx, tile.y + dy);
            if let Some((ox, oy)) = out_pos.to_usize() {
                if world.in_bounds(out_pos.x, out_pos.y) {
                    if let Some(dst_entity) = world.entity_at(ox, oy) {
                        let dst_dir = world.direction_at(ox, oy);
                        // Only connect to belt-like entities or sinks, not opposing
                        let dst_is_belt =
                            matches!(dst_entity, Item::TransportBelt | Item::UndergroundBelt);
                        let dst_is_sink = matches!(dst_entity, Item::Source | Item::Sink);
                        let dst_not_opposing = dst_dir != dir.opposite();
                        if (dst_is_belt || dst_is_sink)
                            && dst_not_opposing
                            && !tile_set.contains(&out_pos)
                        {
                            edges.push((self_id.clone(), NodeId::new(dst_entity, ox, oy)));
                        }
                    }
                }
            }
        }

        edges
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        input
            .iter()
            .map(|(&item, &rate)| (item, rate.min(self.flow_rate())))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_tiles_exhaustive() {
        let dirs = [
            Direction::North,
            Direction::East,
            Direction::South,
            Direction::West,
        ];

        for x in 0..10 {
            for y in 0..10 {
                for width in 1..=5 {
                    for height in 1..=5 {
                        for &dir in &dirs {
                            let tiles = entity_tiles(x, y, dir, width, height);
                            let tiles = tiles.unwrap();

                            // Correct number of tiles
                            assert_eq!(
                                tiles.len(),
                                width * height,
                                "x={x} y={y} w={width} h={height} dir={dir:?}"
                            );

                            // First tile is always the anchor
                            assert_eq!(
                                tiles[0],
                                Pos::new(x as i64, y as i64),
                                "anchor mismatch at x={x} y={y} w={width} h={height} dir={dir:?}"
                            );

                            // No duplicate tiles
                            let unique: std::collections::HashSet<Pos> =
                                tiles.iter().copied().collect();
                            assert_eq!(
                                unique.len(),
                                tiles.len(),
                                "duplicates at x={x} y={y} w={width} h={height} dir={dir:?}"
                            );

                            // All tiles are contiguous (each adjacent to at least one other)
                            for tile in &tiles[1..] {
                                let has_neighbor = tiles.iter().any(|other| {
                                    *other != *tile
                                        && (tile.x - other.x).abs() + (tile.y - other.y).abs() == 1
                                });
                                assert!(
                                    has_neighbor,
                                    "isolated tile {tile:?} at x={x} y={y} w={width} h={height} dir={dir:?}"
                                );
                            }
                        }

                        // Square entities: all directions produce the same tiles
                        if width == height {
                            let baseline =
                                entity_tiles(x, y, Direction::East, width, height).unwrap();
                            for &dir in &dirs {
                                let tiles = entity_tiles(x, y, dir, width, height).unwrap();
                                assert_eq!(
                                    tiles, baseline,
                                    "square {width}x{height} at ({x},{y}): {dir:?} differs from East"
                                );
                            }
                            // Direction::None also works for square
                            let none_tiles =
                                entity_tiles(x, y, Direction::None, width, height).unwrap();
                            assert_eq!(
                                none_tiles, baseline,
                                "square {width}x{height} at ({x},{y}): None differs from East"
                            );
                        }

                        // Non-square: Direction::None returns None
                        if width != height {
                            assert!(
                                entity_tiles(x, y, Direction::None, width, height).is_none(),
                                "non-square {width}x{height} with None should return None"
                            );
                        }
                    }
                }

                // 1x1 always returns the anchor regardless of direction
                for &dir in &[Direction::None, Direction::East, Direction::North] {
                    let tiles = entity_tiles(x, y, dir, 1, 1).unwrap();
                    assert_eq!(tiles, vec![Pos::new(x as i64, y as i64)]);
                }
            }
        }
    }

    /// Helper to build a small world with a few entities for testing connections.
    fn make_belt_chain_world() -> World {
        // 5x1 world: Source(east) -> Belt(east) -> Belt(east) -> Sink(east)
        let mut w = World::empty(5, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        w
    }

    #[test]
    fn test_transport_belt_connections_chain() {
        let w = make_belt_chain_world();

        // Belt at (1,0): pulls from the source behind (a belt-like now) and
        // feeds the belt ahead.
        let belt = TransportBelt;
        let edges = belt.connections((1, 0), Direction::East, &w);

        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(
            NodeId::new(Item::Source, 0, 0),
            NodeId::new(Item::TransportBelt, 1, 0),
        )));
        assert!(edges.contains(&(
            NodeId::new(Item::TransportBelt, 1, 0),
            NodeId::new(Item::TransportBelt, 2, 0),
        )));
    }

    #[test]
    fn test_transport_belt_chain_second_belt() {
        let w = make_belt_chain_world();

        // Belt at (2,0): pulls from the belt behind and feeds the sink ahead
        // (a sink is a belt-like now).
        let belt = TransportBelt;
        let edges = belt.connections((2, 0), Direction::East, &w);

        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(
            NodeId::new(Item::TransportBelt, 1, 0),
            NodeId::new(Item::TransportBelt, 2, 0),
        )));
        assert!(edges.contains(&(
            NodeId::new(Item::TransportBelt, 2, 0),
            NodeId::new(Item::Sink, 3, 0),
        )));
    }

    #[test]
    fn test_transport_belt_no_opposing_connection() {
        // Two belts facing each other should not connect
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 0, Item::TransportBelt, Direction::West, None);

        let belt = TransportBelt;
        let edges = belt.connections((0, 0), Direction::East, &w);
        // The belt ahead is facing the opposite direction → no connection
        assert!(edges.is_empty());
    }

    #[test]
    fn test_inserter_connections() {
        // Inserter at (1,0) facing east, source at (0,0), belt at (2,0)
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Source, Direction::East, None);
        w.place(1, 0, Item::Inserter, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);

        let inserter = Inserter;
        let edges = inserter.connections((1, 0), Direction::East, &w);

        assert_eq!(edges.len(), 2);
        // Source → Inserter
        assert!(edges.contains(&(
            NodeId::new(Item::Source, 0, 0),
            NodeId::new(Item::Inserter, 1, 0),
        )));
        // Inserter → Belt
        assert!(edges.contains(&(
            NodeId::new(Item::Inserter, 1, 0),
            NodeId::new(Item::TransportBelt, 2, 0),
        )));
    }

    #[test]
    fn test_inserter_wont_drop_on_empty() {
        // Inserter facing east, source behind, empty ahead
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Source, Direction::None, None);
        w.place(1, 0, Item::Inserter, Direction::East, None);

        let inserter = Inserter;
        let edges = inserter.connections((1, 0), Direction::East, &w);

        // Only source→inserter, no inserter→empty
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_inserter_wont_pick_up_from_inserter() {
        // Regression for #122. The pickup side accepted *any* non-empty
        // entity behind the inserter, so an inserter facing North at (6,3)
        // picked up from the inserter directly behind it at (6,4) — an
        // inserter→inserter edge that can't exist physically. An inserter may
        // only pick up from a belt, underground belt, assembler, or source.
        //
        // North = (0,-1): the pickup cell (behind) is (6, 3-(-1)) = (6,4);
        // the drop cell (ahead) is (6, 3+(-1)) = (6,2), which is empty.
        let mut w = World::empty(8, 8);
        w.place(6, 3, Item::Inserter, Direction::North, None);
        w.place(6, 4, Item::Inserter, Direction::West, None);

        let inserter = Inserter;
        let edges = inserter.connections((6, 3), Direction::North, &w);

        let bad = (
            NodeId::new(Item::Inserter, 6, 4),
            NodeId::new(Item::Inserter, 6, 3),
        );
        assert!(
            !edges.contains(&bad),
            "inserter (6,3) must not pick up from the inserter behind it at \
             (6,4); got edges: {edges:?}"
        );
        // Nothing insertable ahead either, so there should be no edges at all.
        assert!(edges.is_empty(), "expected no edges, got: {edges:?}");
    }

    #[test]
    fn test_assembler_connections() {
        // 6x6 world with assembler at (1,1) and inserters on perimeter
        let mut w = World::empty(6, 6);
        w.place(
            1,
            1,
            Item::AssemblingMachine1,
            Direction::None,
            Some(Item::ElectronicCircuit),
        );

        // Inserter at (0,1) facing east → inserting into assembler
        w.place(0, 1, Item::Inserter, Direction::East, None);

        // Inserter at (4,2) facing east → taking from assembler (facing away)
        w.place(4, 2, Item::Inserter, Direction::East, None);

        let asm = AssemblingMachine {
            recipe_item: Some(Item::ElectronicCircuit),
        };
        let edges = asm.connections((1, 1), Direction::None, &w);

        assert_eq!(edges.len(), 2);
        // Inserter at (0,1) facing east, ddx=-1 → the inserter is to the left.
        // other_d == East, ddx < 0 → matches condition for "assembler → inserter" (output).
        // Wait, let me re-check: ddx = 0 - 1 = -1, ddy = 1 - 1 = 0.
        // Condition: other_d == EAST and ddx > 0 → ddx = -1, not > 0 → NOT inserting into.
        // other_d == WEST and ddx < 0 → East != West → not this.
        // Actually let me re-derive from the assembler→inserter rule: when
        //   (other_d == North and ddy < 0)
        //     or (other_d == South and ddy > 0)
        //     or (other_d == West and ddx < 0)
        //     or (other_d == East and ddx > 0)
        // the edge is assembler → inserter (the assembler outputs to it);
        // otherwise it is inserter → assembler.
        // So when inserter direction matches the offset direction from assembler → assembler outputs to inserter.
        // For inserter at (0,1): ddx = -1, ddy = 0, dir = East. East and ddx > 0? No. So it's other → self = inserter → assembler.
        // For inserter at (4,2): ddx = 3, ddy = 1, dir = East. East and ddx > 0? Yes. So it's self → other = assembler → inserter.
    }

    #[test]
    fn test_source_transform_flow() {
        let source = Source {
            item: Some(Item::CopperCable),
        };
        let output = source.transform_flow(&HashMap::new());
        assert_eq!(output[&Item::CopperCable], f64::INFINITY);
    }

    #[test]
    fn test_assembler_transform_flow() {
        let asm = AssemblingMachine {
            recipe_item: Some(Item::ElectronicCircuit),
        };

        // Full input matches recipe exactly: 3 copper cable + 1 iron plate → 1 EC
        let input = HashMap::from([(Item::CopperCable, 3.0), (Item::IronPlate, 1.0)]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 1.0).abs() < 1e-9);

        // Half copper cable available: ratio = min(1.5/3, 1/1) = 0.5 → 0.5 EC
        let input = HashMap::from([(Item::CopperCable, 1.5), (Item::IronPlate, 1.0)]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 0.5).abs() < 1e-9);

        // Missing ingredient → 0 output
        let input = HashMap::from([(Item::CopperCable, 3.0)]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_belt_transform_flow() {
        let belt = TransportBelt;
        let input = HashMap::from([(Item::CopperCable, 20.0)]);
        let output = belt.transform_flow(&input);
        // Capped at flow rate of 15.0
        assert!((output[&Item::CopperCable] - 15.0).abs() < 1e-9);

        let input = HashMap::from([(Item::CopperCable, 5.0)]);
        let output = belt.transform_flow(&input);
        assert!((output[&Item::CopperCable] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_underground_belt_down_connections() {
        // Underground down at (1,0) facing east, underground up at (3,0) facing east
        let mut w = World::empty(5, 1);
        w.place_underground(1, 0, Direction::East, Misc::UndergroundDown);
        w.place_underground(3, 0, Direction::East, Misc::UndergroundUp);

        let ub = UndergroundBelt {
            misc: Misc::UndergroundDown,
        };
        let edges = ub.connections((1, 0), Direction::East, &w);

        // Should find the underground belt at (3,0)
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, NodeId::new(Item::UndergroundBelt, 1, 0));
        assert_eq!(edges[0].1, NodeId::new(Item::UndergroundBelt, 3, 0));
    }

    #[test]
    fn test_underground_belt_up_connections() {
        // Underground exit at (3,0) facing east, belt at (4,0) facing east.
        let mut w = World::empty(5, 1);
        w.place_underground(3, 0, Direction::East, Misc::UndergroundUp);
        w.place(4, 0, Item::TransportBelt, Direction::East, None);

        let ub = UndergroundBelt {
            misc: Misc::UndergroundUp,
        };
        let edges = ub.connections((3, 0), Direction::East, &w);

        // An exit outputs like a belt: it feeds the cell directly ahead.
        assert_eq!(
            edges,
            vec![(
                NodeId::new(Item::UndergroundBelt, 3, 0),
                NodeId::new(Item::TransportBelt, 4, 0),
            )]
        );
    }

    #[test]
    fn test_splitter_connections_east() {
        // East-facing splitter at (2,0)/(2,1), belts feeding in and out
        let mut w = World::empty(5, 3);
        // Input belt behind splitter tile (2,0) → at (1,0)
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        // Splitter at (2,0) and (2,1)
        w.place_splitter(2, 0, Direction::East, None);
        // Output belts ahead of splitter tiles
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 1, Item::TransportBelt, Direction::East, None);

        let splitter = Splitter;
        let edges = splitter.connections((2, 0), Direction::East, &w);

        // Should have: belt(1,0)->splitter, splitter->belt(3,0), splitter->belt(3,1)
        let self_id = NodeId::new(Item::Splitter, 2, 0);
        assert!(edges.contains(&(NodeId::new(Item::TransportBelt, 1, 0), self_id.clone())));
        assert!(edges.contains(&(self_id.clone(), NodeId::new(Item::TransportBelt, 3, 0))));
        assert!(edges.contains(&(self_id.clone(), NodeId::new(Item::TransportBelt, 3, 1))));
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_splitter_connections_north() {
        // North-facing splitter at (0,2)/(1,2), belts feeding in and out
        let mut w = World::empty(3, 5);
        // Input belt behind splitter tile (0,2) → at (0,3) (south of anchor, since facing north)
        w.place(0, 3, Item::TransportBelt, Direction::North, None);
        // Splitter at (0,2) and (1,2)
        w.place_splitter(0, 2, Direction::North, None);
        // Output belts ahead
        w.place(0, 1, Item::TransportBelt, Direction::North, None);
        w.place(1, 1, Item::TransportBelt, Direction::North, None);

        let splitter = Splitter;
        let edges = splitter.connections((0, 2), Direction::North, &w);

        let self_id = NodeId::new(Item::Splitter, 0, 2);
        assert!(edges.contains(&(NodeId::new(Item::TransportBelt, 0, 3), self_id.clone())));
        assert!(edges.contains(&(self_id.clone(), NodeId::new(Item::TransportBelt, 0, 1))));
        assert!(edges.contains(&(self_id.clone(), NodeId::new(Item::TransportBelt, 1, 1))));
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_splitter_transform_flow() {
        let splitter = Splitter;
        // 20 i/s passes through (under 30 cap)
        let input = HashMap::from([(Item::CopperCable, 20.0)]);
        let output = splitter.transform_flow(&input);
        assert!((output[&Item::CopperCable] - 20.0).abs() < 1e-9);

        // 40 i/s capped at 30 (2 lanes × 15)
        let input = HashMap::from([(Item::CopperCable, 40.0)]);
        let output = splitter.transform_flow(&input);
        assert!((output[&Item::CopperCable] - 30.0).abs() < 1e-9);
    }

    #[test]
    fn test_entity_enum_new_returns_none_for_non_placeable() {
        assert!(EntityEnum::new(Item::CopperCable, None, Misc::None).is_none());
        assert!(EntityEnum::new(Item::IronGearWheel, None, Misc::None).is_none());
        assert!(EntityEnum::new(Item::TransportBelt, None, Misc::None).is_some());
    }
}
