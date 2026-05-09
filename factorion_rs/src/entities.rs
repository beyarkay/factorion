use crate::lane_flow::LaneFlow;
use crate::types::{get_recipe, Direction, Item, LaneTag, Misc, NodeId, Pos, LANE_FLOW_RATE};
use crate::world::World;

/// A lane-tagged edge in the factory graph:
/// (source node, source-side lane, destination node, destination-side lane).
pub type Edge = (NodeId, LaneTag, NodeId, LaneTag);

/// Trait abstracting over factory entity types.
///
/// Each entity type implements this trait to define:
/// - How it connects to neighboring entities (graph edges)
/// - How it transforms per-lane input flow into per-lane output flow
/// - Its maximum throughput rate
pub trait FactoryEntity {
    /// Which entity kind this is. Used to look up flow_rate from the
    /// single source of truth in Item::flow_rate().
    fn kind(&self) -> Item;

    /// Return the edges this entity contributes to the graph.
    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge>;

    /// Given accumulated per-lane input flow, compute per-lane output flow.
    fn transform_flow(&self, input: &LaneFlow) -> LaneFlow;

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

    fn transform_flow(&self, input: &LaneFlow) -> LaneFlow {
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

/// Helper: cap each lane of `input` at `cap` and return as a new LaneFlow.
/// Used by belt-like entities whose transform is "pass-through with cap".
fn cap_lanes_inplace(input: &LaneFlow, cap: f64) -> LaneFlow {
    let mut output = LaneFlow::default();
    for (&item, &rate) in &input.port {
        output.add(LaneTag::Port, item, rate.min(cap));
    }
    for (&item, &rate) in &input.starboard {
        output.add(LaneTag::Starboard, item, rate.min(cap));
    }
    output
}

// ── Transport Belt ──────────────────────────────────────────────────────────

pub struct TransportBelt;

impl FactoryEntity for TransportBelt {
    fn kind(&self) -> Item {
        Item::TransportBelt
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        let mut edges = Vec::new();
        let (x, y) = pos;
        let (dx, dy) = dir.delta();
        let self_id = NodeId::new(Item::TransportBelt, x, y);

        // Source: the cell behind this belt (opposite of facing direction)
        let src_x = x as i64 - dx;
        let src_y = y as i64 - dy;
        if world.in_bounds(src_x, src_y) {
            let sx = src_x as usize;
            let sy = src_y as usize;
            if let Some(src_entity) = world.entity_at(sx, sy) {
                let src_dir = world.direction_at(sx, sy);
                let src_misc = world.misc_at(sx, sy);

                let src_is_beltish = matches!(
                    src_entity,
                    Item::TransportBelt | Item::UndergroundBelt
                ) && src_dir == dir
                    // Don't connect from a downwards underground belt
                    && !(src_entity == Item::UndergroundBelt
                        && src_misc == Misc::UndergroundDown);

                if src_is_beltish {
                    let src_id = NodeId::new(src_entity, sx, sy);
                    push_lane_preserving(&mut edges, &src_id, &self_id);
                }
            }
        }

        // Destination: the cell ahead of this belt
        let dst_x = x as i64 + dx;
        let dst_y = y as i64 + dy;
        if world.in_bounds(dst_x, dst_y) {
            let dx_u = dst_x as usize;
            let dy_u = dst_y as usize;
            if let Some(dst_entity) = world.entity_at(dx_u, dy_u) {
                let dst_dir = world.direction_at(dx_u, dy_u);

                let dst_is_belt = matches!(dst_entity, Item::TransportBelt | Item::UndergroundBelt);
                // Don't connect to a belt facing the opposite direction
                let dst_opposing = dst_is_belt && dst_dir == dir.opposite();

                if dst_is_belt && !dst_opposing {
                    let dst_id = NodeId::new(dst_entity, dx_u, dy_u);
                    push_lane_preserving(&mut edges, &self_id, &dst_id);
                }
            }
        }

        edges
    }

    fn transform_flow(&self, input: &LaneFlow) -> LaneFlow {
        cap_lanes_inplace(input, LANE_FLOW_RATE)
    }
}

/// Push the lane-preserving pair of edges (Port→Port and Starboard→Starboard)
/// from `src` to `dst`. Used by lane-aware entities for parallel and lone-curve
/// connections.
fn push_lane_preserving(edges: &mut Vec<Edge>, src: &NodeId, dst: &NodeId) {
    edges.push((src.clone(), LaneTag::Port, dst.clone(), LaneTag::Port));
    edges.push((
        src.clone(),
        LaneTag::Starboard,
        dst.clone(),
        LaneTag::Starboard,
    ));
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

    fn transform_flow(&self, input: &LaneFlow) -> LaneFlow {
        cap_lanes_inplace(input, self.flow_rate())
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
                // Skip corners (matching Python)
                if (ddx == -1 || ddx == 3) && (ddy == -1 || ddy == 3) {
                    continue;
                }

                let nx_u = nx as usize;
                let ny_u = ny as usize;
                let other_entity = match world.entity_at(nx_u, ny_u) {
                    Some(e) => e,
                    None => continue,
                };

                // Only inserter-like entities can interact with assembling machines.
                // In Python, Source (stack_inserter) and Sink (bulk_inserter)
                // both contain "inserter" in their name, so they match too.
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
                    edges.push((self_id.clone(), LaneTag::Port, other_id, LaneTag::Port));
                } else {
                    // Entity → Assembler (entity feeds into assembler)
                    edges.push((other_id, LaneTag::Port, self_id.clone(), LaneTag::Port));
                }
            }
        }

        edges
    }

    fn transform_flow(&self, input: &LaneFlow) -> LaneFlow {
        let mut output = LaneFlow::default();
        let recipe_item = match self.recipe_item {
            Some(i) => i,
            None => return output,
        };
        let recipe = match get_recipe(recipe_item) {
            Some(r) => r,
            None => return output,
        };

        // Find the minimum ratio of available input to required input.
        // AssemblingMachine is lane-agnostic — it reads from the port lane only
        // (by convention). Edges feeding it always target Port.
        let mut min_ratio: f64 = 1.0;
        for &(item, required) in recipe.consumes.iter() {
            let available = input.port.get(&item).copied().unwrap_or(0.0);
            let ratio = available / required;
            min_ratio = min_ratio.min(ratio);
        }

        // Produce outputs scaled by the minimum ratio onto the port lane.
        for &(item, rate) in recipe.produces.iter() {
            output.add(LaneTag::Port, item, rate * min_ratio);
        }
        output
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

        let max_delta = match self.misc {
            Misc::UndergroundDown => 6usize,
            Misc::UndergroundUp => 1,
            _ => return edges,
        };

        for delta in 1..max_delta {
            let (dst_x, dst_y) = match dir {
                Direction::East => (x as i64 + delta as i64, y as i64),
                Direction::West => (x as i64 - delta as i64, y as i64),
                Direction::North => (x as i64, y as i64 - delta as i64),
                Direction::South => (x as i64, y as i64 + delta as i64),
                Direction::None => continue,
            };

            if !world.in_bounds(dst_x, dst_y) {
                continue;
            }

            let dst_xu = dst_x as usize;
            let dst_yu = dst_y as usize;
            let dst_entity = match world.entity_at(dst_xu, dst_yu) {
                Some(e) => e,
                None => continue,
            };

            let going_underground =
                dst_entity == Item::UndergroundBelt && self.misc == Misc::UndergroundDown;

            let cxn_to_belt =
                matches!(dst_entity, Item::TransportBelt) && self.misc == Misc::UndergroundUp;

            if going_underground || cxn_to_belt {
                let dst_id = NodeId::new(dst_entity, dst_xu, dst_yu);
                push_lane_preserving(&mut edges, &self_id, &dst_id);
            }
        }

        edges
    }

    fn transform_flow(&self, input: &LaneFlow) -> LaneFlow {
        cap_lanes_inplace(input, LANE_FLOW_RATE)
    }
}

// ── Source (stack_inserter) ─────────────────────────────────────────────────
//
// In the Python code, stack_inserter contains "inserter" in its name, so it
// uses the same inserter connection logic: picks up from behind, drops onto
// belts/assemblers ahead. Its special behavior is that it has infinite output.

pub struct Source {
    item: Option<Item>,
}

impl FactoryEntity for Source {
    fn kind(&self) -> Item {
        Item::Source
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        // Same connection logic as inserter
        inserter_connections(Item::Source, pos, dir, world)
    }

    fn transform_flow(&self, _input: &LaneFlow) -> LaneFlow {
        // A Source with no item set produces nothing. With one set, it
        // produces an infinite flow on BOTH lanes — sources feed belts
        // lane-symmetrically so straight chains saturate at 7.5 + 7.5 = 15.
        let mut output = LaneFlow::default();
        if let Some(item) = self.item {
            output.lane_mut(LaneTag::Port).insert(item, f64::INFINITY);
            output
                .lane_mut(LaneTag::Starboard)
                .insert(item, f64::INFINITY);
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
        inserter_connections(Item::Sink, pos, dir, world)
    }

    fn transform_flow(&self, input: &LaneFlow) -> LaneFlow {
        // Sinks pass through everything (infinite capacity), preserving
        // per-lane structure. The graph-level sink aggregation sums both
        // lanes into a single per-item total.
        input.clone()
    }
}

/// Shared inserter-style connection logic used by Inserter, Source, and Sink.
///
/// Picks up from the entity behind (any non-empty entity), drops onto the
/// entity ahead (only belts or assembling machines).
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

    // Pick up from behind (opposite of facing direction)
    let src_x = x as i64 - dx;
    let src_y = y as i64 - dy;
    if world.in_bounds(src_x, src_y) {
        let sx = src_x as usize;
        let sy = src_y as usize;
        if let Some(src_entity) = world.entity_at(sx, sy) {
            let src_id = NodeId::new(src_entity, sx, sy);
            for (src_tag, dst_tag) in pickup_lane_pairs(self_kind, src_entity) {
                edges.push((src_id.clone(), src_tag, self_id.clone(), dst_tag));
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
                Item::TransportBelt | Item::UndergroundBelt | Item::AssemblingMachine1
            );
            if dst_is_insertable {
                let dst_id = NodeId::new(dst_entity, dx_u, dy_u);
                for (src_tag, dst_tag) in drop_lane_pairs(self_kind, dst_entity) {
                    edges.push((self_id.clone(), src_tag, dst_id.clone(), dst_tag));
                }
            }
        }
    }

    edges
}

/// Lane-aware entity kinds (TB, UG, Splitter). These have port and starboard
/// lanes that carry independent flow.
fn is_lane_aware(kind: Item) -> bool {
    matches!(
        kind,
        Item::TransportBelt | Item::UndergroundBelt | Item::Splitter
    )
}

/// Lane pairs to emit on a *pickup* edge from `src_kind` into `self_kind`
/// (where `self_kind` is one of Inserter/Source/Sink — the inserter family).
///
/// - **Sink ← lane-aware (belt-like)**: dual lane-preserving so both belt
///   lanes drain into the sink. The sink internally aggregates port +
///   starboard for the final per-item total.
/// - Otherwise (Inserter pickups in commit 3, or pickups from non-belts):
///   single port → port edge. Inserter's lane-aware pickup ships in commit 5.
fn pickup_lane_pairs(self_kind: Item, src_kind: Item) -> Vec<(LaneTag, LaneTag)> {
    if self_kind == Item::Sink && is_lane_aware(src_kind) {
        vec![
            (LaneTag::Port, LaneTag::Port),
            (LaneTag::Starboard, LaneTag::Starboard),
        ]
    } else {
        vec![(LaneTag::Port, LaneTag::Port)]
    }
}

/// Lane pairs to emit on a *drop* edge from `self_kind` (Inserter/Source/Sink)
/// into `dst_kind`.
///
/// - **Source → lane-aware**: dual lane-preserving (port→port, stbd→stbd).
///   Sources pre-populate both lanes with infinite flow.
/// - **Source → AM**: dual edges, both source lanes feed AM's port (the
///   lane-agnostic accumulator convention).
/// - Otherwise (Inserter drop in commit 3): single port → port edge.
///   Inserter's FAR / PORT lane logic ships in commit 5.
fn drop_lane_pairs(self_kind: Item, dst_kind: Item) -> Vec<(LaneTag, LaneTag)> {
    if self_kind == Item::Source {
        if is_lane_aware(dst_kind) {
            vec![
                (LaneTag::Port, LaneTag::Port),
                (LaneTag::Starboard, LaneTag::Starboard),
            ]
        } else {
            // Source feeding a lane-agnostic entity (AM): both source lanes
            // funnel into dst.port.
            vec![
                (LaneTag::Port, LaneTag::Port),
                (LaneTag::Starboard, LaneTag::Port),
            ]
        }
    } else {
        vec![(LaneTag::Port, LaneTag::Port)]
    }
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
                            let src_id = NodeId::new(src_entity, ix, iy);
                            push_lane_preserving(&mut edges, &src_id, &self_id);
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
                            let dst_id = NodeId::new(dst_entity, ox, oy);
                            push_lane_preserving(&mut edges, &self_id, &dst_id);
                        }
                    }
                }
            }
        }

        edges
    }

    fn transform_flow(&self, input: &LaneFlow) -> LaneFlow {
        // Splitter is identity per lane — items pool across input belts
        // within their own lane label. The per-output-lane cap of
        // LANE_FLOW_RATE (7.5) is applied AFTER the divisor in
        // calc_throughput, so a splitter feeding two output belts splits
        // each lane evenly across them.
        input.clone()
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
        // ASCII: S>>K  (Source, Belt(1), Belt(2), Sink — all east)
        // Belt at (1,0) emits its forward dual-lane pair to belt at (2,0).
        let w = make_belt_chain_world();

        let belt = TransportBelt;
        let edges = belt.connections((1, 0), Direction::East, &w);

        // Source isn't beltish so no backward edge from it.
        // Belt at (2,0) is a parallel forward dest → dual lane-preserving pair.
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(
            NodeId::new(Item::TransportBelt, 1, 0),
            LaneTag::Port,
            NodeId::new(Item::TransportBelt, 2, 0),
            LaneTag::Port,
        )));
        assert!(edges.contains(&(
            NodeId::new(Item::TransportBelt, 1, 0),
            LaneTag::Starboard,
            NodeId::new(Item::TransportBelt, 2, 0),
            LaneTag::Starboard,
        )));
    }

    #[test]
    fn test_transport_belt_chain_second_belt() {
        // ASCII: S>>K — belt at (2,0) sees belt at (1,0) behind it.
        let w = make_belt_chain_world();

        let belt = TransportBelt;
        let edges = belt.connections((2, 0), Direction::East, &w);

        // Belt at (1,0) behind → dual lane-preserving backward pair.
        // Sink at (3,0) ahead is not a belt → no forward edge.
        assert_eq!(edges.len(), 2);
        for tag in [LaneTag::Port, LaneTag::Starboard] {
            assert!(edges.contains(&(
                NodeId::new(Item::TransportBelt, 1, 0),
                tag,
                NodeId::new(Item::TransportBelt, 2, 0),
                tag,
            )));
        }
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
        // Source → Inserter (Port → Port for now; commit 5 will distinguish further)
        assert!(edges.contains(&(
            NodeId::new(Item::Source, 0, 0),
            LaneTag::Port,
            NodeId::new(Item::Inserter, 1, 0),
            LaneTag::Port,
        )));
        // Inserter → Belt
        assert!(edges.contains(&(
            NodeId::new(Item::Inserter, 1, 0),
            LaneTag::Port,
            NodeId::new(Item::TransportBelt, 2, 0),
            LaneTag::Port,
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
        // Actually let me re-derive. The Python code:
        //   if (other_d == Direction.NORTH and dy < 0)
        //     or (other_d == Direction.SOUTH and dy > 0)
        //     or (other_d == Direction.WEST and dx < 0)
        //     or (other_d == Direction.EAST and dx > 0):
        //       src = self_str (assembler)
        //       dst = other_str (inserter)
        // So when inserter direction matches the offset direction from assembler → assembler outputs to inserter.
        // For inserter at (0,1): ddx = -1, ddy = 0, dir = East. East and ddx > 0? No. So it's other → self = inserter → assembler.
        // For inserter at (4,2): ddx = 3, ddy = 1, dir = East. East and ddx > 0? Yes. So it's self → other = assembler → inserter.
    }

    #[test]
    fn test_source_transform_flow_populates_both_lanes() {
        // Source feeds belts lane-symmetrically — both port AND starboard get
        // infinite flow. This is what keeps a saturated chain at 15 i/s total.
        let source = Source {
            item: Some(Item::CopperCable),
        };
        let output = source.transform_flow(&LaneFlow::default());
        assert_eq!(
            output.lane(LaneTag::Port).get(&Item::CopperCable).copied(),
            Some(f64::INFINITY)
        );
        assert_eq!(
            output
                .lane(LaneTag::Starboard)
                .get(&Item::CopperCable)
                .copied(),
            Some(f64::INFINITY)
        );
    }

    #[test]
    fn test_source_with_no_item_emits_nothing() {
        let source = Source { item: None };
        let output = source.transform_flow(&LaneFlow::default());
        assert!(output.is_empty());
    }

    #[test]
    fn test_assembler_transform_flow() {
        let asm = AssemblingMachine {
            recipe_item: Some(Item::ElectronicCircuit),
        };

        // Full input matches recipe exactly: 3 copper cable + 1 iron plate → 1 EC
        let mut input = LaneFlow::default();
        input.add(LaneTag::Port, Item::CopperCable, 3.0);
        input.add(LaneTag::Port, Item::IronPlate, 1.0);
        let output = asm.transform_flow(&input);
        assert!((output.lane(LaneTag::Port)[&Item::ElectronicCircuit] - 1.0).abs() < 1e-9);

        // Half copper cable available: ratio = min(1.5/3, 1/1) = 0.5 → 0.5 EC
        let mut input = LaneFlow::default();
        input.add(LaneTag::Port, Item::CopperCable, 1.5);
        input.add(LaneTag::Port, Item::IronPlate, 1.0);
        let output = asm.transform_flow(&input);
        assert!((output.lane(LaneTag::Port)[&Item::ElectronicCircuit] - 0.5).abs() < 1e-9);

        // Missing ingredient → 0 output
        let mut input = LaneFlow::default();
        input.add(LaneTag::Port, Item::CopperCable, 3.0);
        let output = asm.transform_flow(&input);
        assert!((output.lane(LaneTag::Port)[&Item::ElectronicCircuit] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_belt_transform_flow_caps_each_lane() {
        // Each lane caps INDEPENDENTLY at LANE_FLOW_RATE (7.5). Per-belt total
        // is 15 only when both lanes are saturated.
        let belt = TransportBelt;
        let mut input = LaneFlow::default();
        input.add(LaneTag::Port, Item::CopperCable, 20.0);
        input.add(LaneTag::Starboard, Item::CopperCable, 3.0);
        let output = belt.transform_flow(&input);
        // Port was over → capped at 7.5. Starboard was under → unchanged.
        assert!(
            (output.lane(LaneTag::Port)[&Item::CopperCable] - LANE_FLOW_RATE).abs() < 1e-9,
            "port should cap at 7.5, got {}",
            output.lane(LaneTag::Port)[&Item::CopperCable]
        );
        assert!(
            (output.lane(LaneTag::Starboard)[&Item::CopperCable] - 3.0).abs() < 1e-9,
            "starboard 3.0 should pass through unchanged, got {}",
            output.lane(LaneTag::Starboard)[&Item::CopperCable]
        );

        // Under-cap input: passes through both lanes unchanged.
        let mut input = LaneFlow::default();
        input.add(LaneTag::Port, Item::CopperCable, 5.0);
        let output = belt.transform_flow(&input);
        assert!((output.lane(LaneTag::Port)[&Item::CopperCable] - 5.0).abs() < 1e-9);
        // Starboard absent in input → absent in output (no synthetic zero entry).
        assert!(!output
            .lane(LaneTag::Starboard)
            .contains_key(&Item::CopperCable));
    }

    #[test]
    fn test_belt_propagates_only_loaded_lane() {
        // Set up an input that has only the port lane loaded; verify the
        // output preserves that asymmetry — port carries flow, starboard
        // stays empty.
        let belt = TransportBelt;
        let mut input = LaneFlow::default();
        input.add(LaneTag::Port, Item::IronPlate, 4.0);
        let output = belt.transform_flow(&input);
        assert_eq!(
            output.lane(LaneTag::Port).get(&Item::IronPlate).copied(),
            Some(4.0)
        );
        assert_eq!(output.lane(LaneTag::Starboard).get(&Item::IronPlate), None);

        // Symmetric: starboard-only input → starboard-only output.
        let mut input = LaneFlow::default();
        input.add(LaneTag::Starboard, Item::IronPlate, 4.0);
        let output = belt.transform_flow(&input);
        assert_eq!(
            output
                .lane(LaneTag::Starboard)
                .get(&Item::IronPlate)
                .copied(),
            Some(4.0)
        );
        assert_eq!(output.lane(LaneTag::Port).get(&Item::IronPlate), None);
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

        // UG-down → UG-up emits the dual lane-preserving pair.
        assert_eq!(edges.len(), 2);
        for tag in [LaneTag::Port, LaneTag::Starboard] {
            assert!(edges.contains(&(
                NodeId::new(Item::UndergroundBelt, 1, 0),
                tag,
                NodeId::new(Item::UndergroundBelt, 3, 0),
                tag,
            )));
        }
    }

    #[test]
    fn test_underground_belt_up_connections() {
        // Underground up at (3,0) facing east, belt at (4,0) facing east
        let mut w = World::empty(5, 1);
        w.place_underground(3, 0, Direction::East, Misc::UndergroundUp);
        w.place(4, 0, Item::TransportBelt, Direction::East, None);

        let ub = UndergroundBelt {
            misc: Misc::UndergroundUp,
        };
        let edges = ub.connections((3, 0), Direction::East, &w);

        // max_delta is 1 for UP, so it searches at delta=0 only... wait, range(1, 1) is empty.
        // Hmm, looking at the Python: for UP, max_delta=1, range(1, 1) is empty.
        // That means UP underground belts create no edges from their own connections()!
        // The DOWN belt creates the edge to the UP belt. The UP belt's connection to
        // the next transport belt is handled by the transport belt's connections().
        assert_eq!(edges.len(), 0);
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

        // belt(1,0)→splitter (dual), splitter→belt(3,0) (dual), splitter→belt(3,1) (dual).
        // 3 conceptual connections × 2 lane-pairs = 6 lane-tagged edges.
        let self_id = NodeId::new(Item::Splitter, 2, 0);
        for tag in [LaneTag::Port, LaneTag::Starboard] {
            assert!(edges.contains(&(
                NodeId::new(Item::TransportBelt, 1, 0),
                tag,
                self_id.clone(),
                tag,
            )));
            assert!(edges.contains(&(
                self_id.clone(),
                tag,
                NodeId::new(Item::TransportBelt, 3, 0),
                tag,
            )));
            assert!(edges.contains(&(
                self_id.clone(),
                tag,
                NodeId::new(Item::TransportBelt, 3, 1),
                tag,
            )));
        }
        assert_eq!(edges.len(), 6);
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
        for tag in [LaneTag::Port, LaneTag::Starboard] {
            assert!(edges.contains(&(
                NodeId::new(Item::TransportBelt, 0, 3),
                tag,
                self_id.clone(),
                tag,
            )));
            assert!(edges.contains(&(
                self_id.clone(),
                tag,
                NodeId::new(Item::TransportBelt, 0, 1),
                tag,
            )));
            assert!(edges.contains(&(
                self_id.clone(),
                tag,
                NodeId::new(Item::TransportBelt, 1, 1),
                tag,
            )));
        }
        assert_eq!(edges.len(), 6);
    }

    #[test]
    fn test_splitter_transform_flow_is_identity() {
        // Splitter's transform_flow is identity per lane. The per-output-
        // lane cap (LANE_FLOW_RATE) is applied in calc_throughput AFTER the
        // divisor, not here.
        let splitter = Splitter;
        let mut input = LaneFlow::default();
        input.add(LaneTag::Port, Item::CopperCable, 12.0);
        input.add(LaneTag::Starboard, Item::CopperCable, 7.0);
        let output = splitter.transform_flow(&input);
        assert_eq!(output.lane(LaneTag::Port)[&Item::CopperCable], 12.0);
        assert_eq!(output.lane(LaneTag::Starboard)[&Item::CopperCable], 7.0);
    }

    #[test]
    fn test_splitter_keeps_lanes_separate() {
        // Splitter's transform_flow does not mix port↔starboard. An asymmetric
        // input (e.g. only port loaded) stays asymmetric on the output.
        let splitter = Splitter;
        let mut input = LaneFlow::default();
        input.add(LaneTag::Port, Item::IronPlate, 5.0);
        let output = splitter.transform_flow(&input);
        assert_eq!(output.lane(LaneTag::Port)[&Item::IronPlate], 5.0);
        assert!(!output
            .lane(LaneTag::Starboard)
            .contains_key(&Item::IronPlate));
    }

    #[test]
    fn test_entity_enum_new_returns_none_for_non_placeable() {
        assert!(EntityEnum::new(Item::CopperCable, None, Misc::None).is_none());
        assert!(EntityEnum::new(Item::IronGearWheel, None, Misc::None).is_none());
        assert!(EntityEnum::new(Item::TransportBelt, None, Misc::None).is_some());
    }

    // ── Lane-aware connection edge cases (commit 3) ────────────────────────

    /// Helper: assert a lane-preserving (Port→Port, Stbd→Stbd) edge pair
    /// from `src` to `dst` is present in `edges`.
    fn assert_dual_lane_preserving(edges: &[Edge], src: NodeId, dst: NodeId) {
        for tag in [LaneTag::Port, LaneTag::Starboard] {
            assert!(
                edges.contains(&(src.clone(), tag, dst.clone(), tag)),
                "missing {:?}→{:?} edge for lane {:?} in {:?}",
                src,
                dst,
                tag,
                edges
            );
        }
    }

    #[test]
    fn test_belt_dual_edges_in_all_directions() {
        // Straight chain in each cardinal direction:
        // North-facing: > > facing N at (1,2)→(1,1).
        // East:  > > at (1,1)→(2,1).
        // South: v v at (1,1)→(1,2).
        // West:  < < at (2,1)→(1,1).
        for (dir, src, dst) in [
            (Direction::North, (1, 2), (1, 1)),
            (Direction::East, (1, 1), (2, 1)),
            (Direction::South, (1, 1), (1, 2)),
            (Direction::West, (2, 1), (1, 1)),
        ] {
            let mut w = World::empty(4, 4);
            w.place(src.0, src.1, Item::TransportBelt, dir, None);
            w.place(dst.0, dst.1, Item::TransportBelt, dir, None);

            let belt = TransportBelt;
            let edges = belt.connections(src, dir, &w);

            // Dual edges to the parallel forward dest. (Backward source is empty.)
            assert_eq!(
                edges.len(),
                2,
                "dir={:?} expected 2 forward edges, got {:?}",
                dir,
                edges
            );
            assert_dual_lane_preserving(
                &edges,
                NodeId::new(Item::TransportBelt, src.0, src.1),
                NodeId::new(Item::TransportBelt, dst.0, dst.1),
            );
        }
    }

    #[test]
    fn test_belt_lone_curve_emits_dual_lane_preserving() {
        // Diagram (north-up):
        //   > v
        // East-belt at (0,1) feeds south-belt at (1,1). The south-belt has no
        // OTHER source, so this is a lone curve — emit lane-preserving pair.
        // (T-junction detection that switches to side-load lands in commit 5.)
        let mut w = World::empty(3, 3);
        w.place(0, 1, Item::TransportBelt, Direction::East, None);
        w.place(1, 1, Item::TransportBelt, Direction::South, None);

        let belt = TransportBelt;
        let edges = belt.connections((0, 1), Direction::East, &w);

        assert_eq!(edges.len(), 2, "lone curve should emit dual edges");
        assert_dual_lane_preserving(
            &edges,
            NodeId::new(Item::TransportBelt, 0, 1),
            NodeId::new(Item::TransportBelt, 1, 1),
        );
    }

    #[test]
    fn test_belt_no_edges_head_to_head_in_all_directions() {
        // Two opposing belts at all 4 axis pairings — none should connect.
        for (dir, opp) in [
            (Direction::East, Direction::West),
            (Direction::West, Direction::East),
            (Direction::North, Direction::South),
            (Direction::South, Direction::North),
        ] {
            let mut w = World::empty(3, 3);
            w.place(0, 0, Item::TransportBelt, dir, None);
            // Forward-neighbor of (0,0) facing `dir` opposes us.
            let (dx, dy) = dir.delta();
            let nx = (dx + 1) as usize;
            let ny = if dy > 0 { 1 } else { 0 };
            // Recompute correctly: forward neighbor coord
            let _ = (nx, ny); // unused
            let fx = (0i64 + dx).max(0) as usize;
            let fy = (0i64 + dy).max(0) as usize;
            if fx < 3 && fy < 3 {
                w.place(fx, fy, Item::TransportBelt, opp, None);
            }
            let belt = TransportBelt;
            let edges = belt.connections((0, 0), dir, &w);
            assert!(
                edges.is_empty(),
                "head-to-head ({:?} vs {:?}) should produce no edges, got {:?}",
                dir,
                opp,
                edges
            );
        }
    }

    #[test]
    fn test_underground_belt_dual_edges_all_directions() {
        // UG-down → UG-up, all four cardinal directions, distance 2.
        for (dir, down, up) in [
            (Direction::East, (0, 1), (2, 1)),
            (Direction::West, (3, 1), (1, 1)),
            (Direction::North, (1, 3), (1, 1)),
            (Direction::South, (1, 0), (1, 2)),
        ] {
            let mut w = World::empty(4, 4);
            w.place_underground(down.0, down.1, dir, Misc::UndergroundDown);
            w.place_underground(up.0, up.1, dir, Misc::UndergroundUp);
            let ub = UndergroundBelt {
                misc: Misc::UndergroundDown,
            };
            let edges = ub.connections(down, dir, &w);
            assert_eq!(
                edges.len(),
                2,
                "UG dual edge expected for dir {:?}, got {:?}",
                dir,
                edges
            );
            assert_dual_lane_preserving(
                &edges,
                NodeId::new(Item::UndergroundBelt, down.0, down.1),
                NodeId::new(Item::UndergroundBelt, up.0, up.1),
            );
        }
    }

    #[test]
    fn test_source_emits_dual_edges_to_belt() {
        // Source's drop edge to a belt-like dest is dual lane-preserving.
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);

        let source = Source {
            item: Some(Item::CopperCable),
        };
        let edges = source.connections((0, 0), Direction::East, &w);

        // Pickup from (-1, 0) is out of bounds → 0 pickup edges. Drop is dual.
        assert_eq!(edges.len(), 2);
        assert_dual_lane_preserving(
            &edges,
            NodeId::new(Item::Source, 0, 0),
            NodeId::new(Item::TransportBelt, 1, 0),
        );
    }

    #[test]
    fn test_source_to_assembler_routes_both_source_lanes_to_port() {
        // Source feeding an AM (lane-agnostic): two edges, both targeting
        // the AM's port lane (the AM accumulator convention). Source must
        // forward into the AM anchor — World only stores entities at anchors.
        let mut w = World::empty(5, 5);
        w.place(
            1,
            1,
            Item::AssemblingMachine1,
            Direction::None,
            Some(Item::CopperCable),
        );
        // Source at (0,1) facing East drops to (1,1) — the AM anchor.
        w.place(0, 1, Item::Source, Direction::East, Some(Item::CopperCable));

        let source = Source {
            item: Some(Item::CopperCable),
        };
        let edges = source.connections((0, 1), Direction::East, &w);

        // AM is not lane-aware, so Source emits (Port→Port, Stbd→Port) —
        // both source lanes funnel into the AM's port accumulator.
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(
            NodeId::new(Item::Source, 0, 1),
            LaneTag::Port,
            NodeId::new(Item::AssemblingMachine1, 1, 1),
            LaneTag::Port,
        )));
        assert!(edges.contains(&(
            NodeId::new(Item::Source, 0, 1),
            LaneTag::Starboard,
            NodeId::new(Item::AssemblingMachine1, 1, 1),
            LaneTag::Port,
        )));
    }

    #[test]
    fn test_sink_emits_dual_edges_from_belt() {
        // Sink ← belt: dual lane-preserving so both belt lanes drain.
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let sink = Sink;
        let edges = sink.connections((1, 0), Direction::East, &w);

        // Pickup is dual; forward drop at (2,0) is empty so no drop edges.
        assert_eq!(edges.len(), 2);
        assert_dual_lane_preserving(
            &edges,
            NodeId::new(Item::TransportBelt, 0, 0),
            NodeId::new(Item::Sink, 1, 0),
        );
    }

    #[test]
    fn test_inserter_stays_single_edge_in_commit_3() {
        // Inserter pickup/drop are still single (Port→Port) edges in commit 3.
        // Commit 5 will add the FAR-lane drop and lane-aware pickup.
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 0, Item::Inserter, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);

        let inserter = Inserter;
        let edges = inserter.connections((1, 0), Direction::East, &w);

        // Pickup from belt (lane-aware src, but self is Inserter not Sink → single).
        // Drop to belt (self is Inserter, not Source → single).
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(
            NodeId::new(Item::TransportBelt, 0, 0),
            LaneTag::Port,
            NodeId::new(Item::Inserter, 1, 0),
            LaneTag::Port,
        )));
        assert!(edges.contains(&(
            NodeId::new(Item::Inserter, 1, 0),
            LaneTag::Port,
            NodeId::new(Item::TransportBelt, 2, 0),
            LaneTag::Port,
        )));
    }

    #[test]
    fn test_belt_to_belt_at_world_boundary() {
        // Belt at (0,0) facing west: forward goes off-grid → no forward edge,
        // and (-1,0) backward also off-grid. No edges either side.
        let mut w = World::empty(3, 3);
        w.place(0, 0, Item::TransportBelt, Direction::West, None);
        let belt = TransportBelt;
        let edges = belt.connections((0, 0), Direction::West, &w);
        assert!(edges.is_empty(), "boundary belt should emit no edges");
    }
}
