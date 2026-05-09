use std::collections::HashMap;

use crate::types::{get_recipe, Direction, Item, LaneSide, Misc, NodeId, Pos, LANE_FLOW_RATE};
use crate::world::World;

/// A directed edge in the factory graph, identified by source and
/// destination NodeIds. Lane / port information lives in the NodeIds —
/// edges themselves are untagged.
pub type Edge = (NodeId, NodeId);

/// Trait abstracting over factory entity types.
///
/// Each entity type implements this trait to define:
/// - How it connects to neighbouring entities (graph edges)
/// - How it transforms input flow into output flow per node
/// - Its maximum throughput rate
/// - Whether it exposes per-lane port-nodes
pub trait FactoryEntity {
    /// Which entity kind this is. Used to look up flow_rate from the
    /// single source of truth in Item::flow_rate().
    fn kind(&self) -> Item;

    /// Return the edges this entity contributes to the graph.
    ///
    /// Lane-aware entities (TB / UG / Splitter) emit edges that target
    /// specific port-nodes (`NodeId::port` / `NodeId::starboard`).
    /// Lane-agnostic entities use `NodeId::single`.
    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge>;

    /// Given accumulated input flow, compute output flow.
    ///
    /// Each port-node has its own flow accumulator, so this is a flat
    /// HashMap rather than a per-lane struct.
    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64>;

    /// Maximum items/second this entity can transfer per port-node.
    /// Default delegates to `Item::flow_rate()`.
    fn flow_rate(&self) -> f64 {
        self.kind().flow_rate()
    }

    /// Whether this entity contributes per-lane port-nodes to the graph.
    ///
    /// `true` for lane-aware entities (TransportBelt, UndergroundBelt,
    /// Splitter): each anchor tile gets one Port and one Starboard node.
    /// `false` for lane-agnostic entities (Inserter, Source, Sink,
    /// AssemblingMachine): a single node per anchor.
    ///
    /// Deliberately has NO default — adding a new entity must explicitly
    /// declare its lane-awareness, which forces the author to think
    /// about how it interacts with the per-lane edge rules.
    fn is_lane_aware(&self) -> bool;
}

/// Stack-allocated entity dispatch. Wraps concrete entity structs.
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
    /// non-placeable items.
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

    fn is_lane_aware(&self) -> bool {
        match self {
            Self::TransportBelt(e) => e.is_lane_aware(),
            Self::Inserter(e) => e.is_lane_aware(),
            Self::AssemblingMachine(e) => e.is_lane_aware(),
            Self::UndergroundBelt(e) => e.is_lane_aware(),
            Self::Sink(e) => e.is_lane_aware(),
            Self::Source(e) => e.is_lane_aware(),
            Self::Splitter(e) => e.is_lane_aware(),
        }
    }
}

// ── Helper: emit lane-preserving edges between two lane-aware entities ─────

/// Push the pair of lane-preserving edges (src.port → dst.port and
/// src.starboard → dst.starboard) for parallel forward / backward
/// connections and lone-curve forwards.
fn push_lane_preserving(
    edges: &mut Vec<Edge>,
    src_kind: Item,
    src_pos: (usize, usize),
    dst_kind: Item,
    dst_pos: (usize, usize),
) {
    edges.push((
        NodeId::port(src_kind, src_pos.0, src_pos.1),
        NodeId::port(dst_kind, dst_pos.0, dst_pos.1),
    ));
    edges.push((
        NodeId::starboard(src_kind, src_pos.0, src_pos.1),
        NodeId::starboard(dst_kind, dst_pos.0, dst_pos.1),
    ));
}

/// Lookup-by-Item shim for code that has an entity kind but not an
/// instance (graph builder, connection emitters). Constructs an enum
/// value with throwaway item/misc and reads its `is_lane_aware()`. The
/// trait method on each impl is the canonical source of truth — this
/// shim is a convenience wrapper.
///
/// Returns `false` for non-placeable items (no entity, hence not
/// lane-aware).
pub(crate) fn is_lane_aware(kind: Item) -> bool {
    EntityEnum::new(kind, None, Misc::None)
        .map(|e| e.is_lane_aware())
        .unwrap_or(false)
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

        // Source: the cell behind this belt (opposite of facing direction).
        // Same-direction beltish predecessors emit lane-preserving edges
        // INTO us (this matches the python world2graph behavior of
        // emitting edges from both endpoints).
        let src_x = x as i64 - dx;
        let src_y = y as i64 - dy;
        if world.in_bounds(src_x, src_y) {
            let sx = src_x as usize;
            let sy = src_y as usize;
            if let Some(src_entity) = world.entity_at(sx, sy) {
                let src_dir = world.direction_at(sx, sy);
                let src_misc = world.misc_at(sx, sy);

                let src_is_beltish =
                    matches!(src_entity, Item::TransportBelt | Item::UndergroundBelt)
                        && src_dir == dir
                        && !(src_entity == Item::UndergroundBelt
                            && src_misc == Misc::UndergroundDown);

                if src_is_beltish {
                    push_lane_preserving(
                        &mut edges,
                        src_entity,
                        (sx, sy),
                        Item::TransportBelt,
                        (x, y),
                    );
                }
            }
        }

        // Destination: the cell ahead of this belt.
        let dst_x = x as i64 + dx;
        let dst_y = y as i64 + dy;
        if world.in_bounds(dst_x, dst_y) {
            let dx_u = dst_x as usize;
            let dy_u = dst_y as usize;
            if let Some(dst_entity) = world.entity_at(dx_u, dy_u) {
                let dst_dir = world.direction_at(dx_u, dy_u);
                let dst_is_belt = matches!(dst_entity, Item::TransportBelt | Item::UndergroundBelt);
                let dst_opposing = dst_is_belt && dst_dir == dir.opposite();

                if dst_is_belt && !dst_opposing {
                    if dir == dst_dir {
                        // Parallel forward: lane-preserving.
                        push_lane_preserving(
                            &mut edges,
                            Item::TransportBelt,
                            (x, y),
                            dst_entity,
                            (dx_u, dy_u),
                        );
                    } else if dest_has_other_source((dx_u, dy_u), dst_dir, (x, y), world) {
                        // Perpendicular forward into a JUNCTION (the dst
                        // already has another source). Side-load: route
                        // BOTH our lanes onto whichever side of dst we
                        // physically sit on.
                        let our_side =
                            match dst_dir.side_of(x as i64 - dx_u as i64, y as i64 - dy_u as i64) {
                                Some(s) => s,
                                // Geometry shouldn't fail for a perpendicular
                                // forward neighbour — fall back to lane-
                                // preserving rather than guessing.
                                None => {
                                    push_lane_preserving(
                                        &mut edges,
                                        Item::TransportBelt,
                                        (x, y),
                                        dst_entity,
                                        (dx_u, dy_u),
                                    );
                                    return edges;
                                }
                            };
                        let dst_lane = NodeId::lane(dst_entity, dx_u, dy_u, our_side);
                        edges.push((NodeId::port(Item::TransportBelt, x, y), dst_lane.clone()));
                        edges.push((NodeId::starboard(Item::TransportBelt, x, y), dst_lane));
                    } else {
                        // Perpendicular forward, lone curve (no other
                        // source on dst): lane-preserving.
                        push_lane_preserving(
                            &mut edges,
                            Item::TransportBelt,
                            (x, y),
                            dst_entity,
                            (dx_u, dy_u),
                        );
                    }
                }
            }
        }

        edges
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        // Each port-node caps independently at LANE_FLOW_RATE (7.5 i/s).
        input
            .iter()
            .map(|(&item, &rate)| (item, rate.min(LANE_FLOW_RATE)))
            .collect()
    }

    fn is_lane_aware(&self) -> bool {
        true
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

    fn is_lane_aware(&self) -> bool {
        false
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
        let self_id = NodeId::single(Item::AssemblingMachine1, x, y);

        // Search the perimeter of the 3x3 footprint for inserter-likes.
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
                if (0..3).contains(&ddx) && (0..3).contains(&ddy) {
                    continue; // inside the body
                }
                if (ddx == -1 || ddx == 3) && (ddy == -1 || ddy == 3) {
                    continue; // skip corners (matches Python)
                }

                let nx_u = nx as usize;
                let ny_u = ny as usize;
                let other_entity = match world.entity_at(nx_u, ny_u) {
                    Some(e) => e,
                    None => continue,
                };

                if !matches!(other_entity, Item::Inserter | Item::Source | Item::Sink) {
                    continue;
                }

                let other_dir = world.direction_at(nx_u, ny_u);
                let other_id = NodeId::single(other_entity, nx_u, ny_u);

                let assembler_outputs_to_entity = (other_dir == Direction::North && ddy < 0)
                    || (other_dir == Direction::South && ddy > 0)
                    || (other_dir == Direction::West && ddx < 0)
                    || (other_dir == Direction::East && ddx > 0);

                if assembler_outputs_to_entity {
                    edges.push((self_id.clone(), other_id));
                } else {
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

        let mut min_ratio: f64 = 1.0;
        for &(item, required) in recipe.consumes.iter() {
            let available = input.get(&item).copied().unwrap_or(0.0);
            let ratio = available / required;
            min_ratio = min_ratio.min(ratio);
        }
        recipe
            .produces
            .iter()
            .map(|&(item, rate)| (item, rate * min_ratio))
            .collect()
    }

    fn is_lane_aware(&self) -> bool {
        false
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
                push_lane_preserving(
                    &mut edges,
                    Item::UndergroundBelt,
                    (x, y),
                    dst_entity,
                    (dst_xu, dst_yu),
                );
            }
        }

        edges
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        input
            .iter()
            .map(|(&item, &rate)| (item, rate.min(LANE_FLOW_RATE)))
            .collect()
    }

    fn is_lane_aware(&self) -> bool {
        true
    }
}

// ── Source (stack_inserter) ─────────────────────────────────────────────────

pub struct Source {
    item: Option<Item>,
}

impl FactoryEntity for Source {
    fn kind(&self) -> Item {
        Item::Source
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        inserter_connections(Item::Source, pos, dir, world)
    }

    fn transform_flow(&self, _input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        let mut output = HashMap::new();
        if let Some(item) = self.item {
            output.insert(item, f64::INFINITY);
        }
        output
    }

    fn is_lane_aware(&self) -> bool {
        false
    }
}

// ── Sink (bulk_inserter) ────────────────────────────────────────────────────

pub struct Sink;

impl FactoryEntity for Sink {
    fn kind(&self) -> Item {
        Item::Sink
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        inserter_connections(Item::Sink, pos, dir, world)
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        // Sinks pass everything through (infinite capacity).
        input.clone()
    }

    fn is_lane_aware(&self) -> bool {
        false
    }
}

/// Shared connection logic for the inserter family (Inserter, Source,
/// Sink). The caller supplies its own `Item` kind via `self_kind`.
///
/// - Pickup edges go from the source entity behind into `self`.
///   * Source is lane-aware (TB/UG/Splitter) → emit one edge from each
///     of its two port-nodes into our single node.
///   * Source is lane-agnostic → single edge.
/// - Drop edges go from `self` to the entity ahead.
///   * Destination is lane-aware → emit edges into BOTH dst port-nodes
///     (Source feeds both lanes equally; Inserter / Sink in commit 4.5
///     follow the same lane-symmetric pattern, with FAR / PORT-lane
///     specialisation landing in commit 5).
///   * Destination is AssemblingMachine → single edge.
fn inserter_connections(
    self_kind: Item,
    pos: (usize, usize),
    dir: Direction,
    world: &World,
) -> Vec<Edge> {
    let mut edges = Vec::new();
    let (x, y) = pos;
    let (dx, dy) = dir.delta();
    let self_id = NodeId::single(self_kind, x, y);

    // Pickup from behind.
    let src_x = x as i64 - dx;
    let src_y = y as i64 - dy;
    if world.in_bounds(src_x, src_y) {
        let sx = src_x as usize;
        let sy = src_y as usize;
        if let Some(src_entity) = world.entity_at(sx, sy) {
            if is_lane_aware(src_entity) {
                edges.push((NodeId::port(src_entity, sx, sy), self_id.clone()));
                edges.push((NodeId::starboard(src_entity, sx, sy), self_id.clone()));
            } else {
                edges.push((NodeId::single(src_entity, sx, sy), self_id.clone()));
            }
        }
    }

    // Drop ahead.
    //
    // Inserter / Source / Sink all share the same scan but emit
    // different edge shapes:
    // - **Source** has unbounded supply, so it dual-emits to BOTH
    //   destination port-nodes — every lane saturates independently.
    //   **Sink** mirrors its accumulated input on output, so dual-emit
    //   preserves the dual-lane total when a sink ever has a downstream
    //   belt.
    // - **Inserter** has a finite per-step budget (0.86 i/s) that lands
    //   on exactly ONE lane of the destination. Which lane is chosen by
    //   geometry: perpendicular drops land on the FAR side; in-line
    //   drops land on the PORT side. See `inserter_drop_lane`.
    let dst_x = x as i64 + dx;
    let dst_y = y as i64 + dy;
    if world.in_bounds(dst_x, dst_y) {
        let dx_u = dst_x as usize;
        let dy_u = dst_y as usize;
        if let Some(dst_entity) = world.entity_at(dx_u, dy_u) {
            let dst_dir = world.direction_at(dx_u, dy_u);
            let dst_is_insertable = matches!(
                dst_entity,
                Item::TransportBelt | Item::UndergroundBelt | Item::AssemblingMachine1
            );
            if dst_is_insertable {
                if is_lane_aware(dst_entity) {
                    if self_kind == Item::Inserter {
                        // Pick the right lane by geometry.
                        let drop_side = inserter_drop_lane((x, y), dir, (dx_u, dy_u), dst_dir);
                        edges.push((
                            self_id.clone(),
                            NodeId::lane(dst_entity, dx_u, dy_u, drop_side),
                        ));
                    } else {
                        // Source / Sink: dual emit to both port-nodes.
                        edges.push((self_id.clone(), NodeId::port(dst_entity, dx_u, dy_u)));
                        edges.push((self_id.clone(), NodeId::starboard(dst_entity, dx_u, dy_u)));
                    }
                } else {
                    edges.push((self_id.clone(), NodeId::single(dst_entity, dx_u, dy_u)));
                }
            }
        }
    }

    edges
}

/// True if the destination belt at `dst_pos` (facing `dst_dir`) has
/// any belt-like source feeding it OTHER than the belt at `excluding`.
///
/// Sources we consider:
/// 1. **Parallel-back**: a TB or UG-up at `dst - dst_dir.delta()` facing
///    the same direction as dst. UG-down does NOT count — it diverts
///    flow into its tunnel rather than pushing forward.
/// 2. **Underground tunnel**: a UG-down within 1..=5 cells back facing
///    the same direction as dst (its forward exit lands on dst).
/// 3. **Perpendicular from another side**: a TB at the port-side or
///    starboard-side neighbour of dst whose forward delta lands on dst.
///    UG-down on the side cell counts too if its tunnel exits onto dst,
///    but for simplicity we only treat plain TBs as side-source
///    candidates here — that matches the side-loading cases the spec
///    enumerates.
///
/// `excluding` is skipped during all three checks so we don't count
/// ourselves as our own "other source".
fn dest_has_other_source(
    dst_pos: (usize, usize),
    dst_dir: Direction,
    excluding: (usize, usize),
    world: &World,
) -> bool {
    let (dx, dy) = dst_dir.delta();

    // 1. Parallel-back source.
    let bx = dst_pos.0 as i64 - dx;
    let by = dst_pos.1 as i64 - dy;
    if world.in_bounds(bx, by) {
        let b = (bx as usize, by as usize);
        if b != excluding {
            if let Some(e) = world.entity_at(b.0, b.1) {
                let is_back_source = match e {
                    Item::TransportBelt => world.direction_at(b.0, b.1) == dst_dir,
                    Item::UndergroundBelt => {
                        world.direction_at(b.0, b.1) == dst_dir
                            && world.misc_at(b.0, b.1) == Misc::UndergroundUp
                    }
                    _ => false,
                };
                if is_back_source {
                    return true;
                }
            }
        }
    }

    // 2. Underground tunnel: UG-down 1..=5 cells back facing dst_dir.
    for k in 1..=5i64 {
        let px = dst_pos.0 as i64 - k * dx;
        let py = dst_pos.1 as i64 - k * dy;
        if !world.in_bounds(px, py) {
            break;
        }
        let p = (px as usize, py as usize);
        if p == excluding {
            continue;
        }
        if let Some(e) = world.entity_at(p.0, p.1) {
            if e == Item::UndergroundBelt
                && world.direction_at(p.0, p.1) == dst_dir
                && world.misc_at(p.0, p.1) == Misc::UndergroundDown
            {
                return true;
            }
        }
    }

    // 3. Perpendicular sources on the port and starboard side cells.
    for offset in [
        dst_dir.port_neighbor_offset(),
        dst_dir.starboard_neighbor_offset(),
    ] {
        let nx = dst_pos.0 as i64 + offset.0;
        let ny = dst_pos.1 as i64 + offset.1;
        if !world.in_bounds(nx, ny) {
            continue;
        }
        let n = (nx as usize, ny as usize);
        if n == excluding {
            continue;
        }
        if let Some(e) = world.entity_at(n.0, n.1) {
            if !matches!(e, Item::TransportBelt | Item::UndergroundBelt) {
                continue;
            }
            let ndir = world.direction_at(n.0, n.1);
            if !ndir.is_perpendicular(dst_dir) {
                continue;
            }
            // Does this side belt's forward step land on dst?
            let (ndx, ndy) = ndir.delta();
            if n.0 as i64 + ndx == dst_pos.0 as i64 && n.1 as i64 + ndy == dst_pos.1 as i64 {
                // For UG-down this would be its tunnel entry, not a
                // forward feed — skip.
                if e == Item::UndergroundBelt && world.misc_at(n.0, n.1) == Misc::UndergroundDown {
                    continue;
                }
                return true;
            }
        }
    }

    false
}

/// Choose the destination belt lane for an inserter drop:
/// - **Perpendicular** drop (inserter direction ⊥ belt direction):
///   the inserter sits on one side of the belt and reaches OVER the
///   near lane to deliver onto the FAR lane.
/// - **In-line** drop (parallel or anti-parallel): the inserter is in
///   front of or behind the belt; items land on the PORT lane.
fn inserter_drop_lane(
    self_pos: (usize, usize),
    self_dir: Direction,
    dst_pos: (usize, usize),
    dst_dir: Direction,
) -> LaneSide {
    if !self_dir.is_perpendicular(dst_dir) {
        return LaneSide::Port;
    }
    let dx = self_pos.0 as i64 - dst_pos.0 as i64;
    let dy = self_pos.1 as i64 - dst_pos.1 as i64;
    match dst_dir.side_of(dx, dy) {
        Some(side) => side.opposite(),
        // Geometry didn't classify the inserter as a side neighbour —
        // shouldn't happen for valid placements, but PORT is the safe
        // default (matches the in-line rule).
        None => LaneSide::Port,
    }
}

/// Compute all tiles occupied by an entity given its anchor, direction, and size.
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

pub struct Splitter;

impl FactoryEntity for Splitter {
    fn kind(&self) -> Item {
        Item::Splitter
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        let mut edges = Vec::new();
        let (x, y) = pos;
        let (dx, dy) = dir.delta();

        let tiles = match entity_tiles(x, y, dir, 2, 1) {
            Some(t) => t,
            None => return edges,
        };
        let tile_set: std::collections::HashSet<Pos> = tiles.iter().copied().collect();

        for &tile in &tiles {
            // Input cell behind this tile.
            let in_pos = Pos::new(tile.x - dx, tile.y - dy);
            if let Some((ix, iy)) = in_pos.to_usize() {
                if world.in_bounds(in_pos.x, in_pos.y) {
                    if let Some(src_entity) = world.entity_at(ix, iy) {
                        let src_dir = world.direction_at(ix, iy);
                        let src_is_belt =
                            matches!(src_entity, Item::TransportBelt | Item::UndergroundBelt)
                                && src_dir == dir;
                        let src_is_source_sink =
                            matches!(src_entity, Item::Source | Item::Sink) && src_dir == dir;

                        if src_is_belt && !tile_set.contains(&in_pos) {
                            push_lane_preserving(
                                &mut edges,
                                src_entity,
                                (ix, iy),
                                Item::Splitter,
                                (x, y),
                            );
                        } else if src_is_source_sink && !tile_set.contains(&in_pos) {
                            // Source / Sink (lane-agnostic) feeds both
                            // splitter port-nodes equally.
                            let src_id = NodeId::single(src_entity, ix, iy);
                            edges.push((src_id.clone(), NodeId::port(Item::Splitter, x, y)));
                            edges.push((src_id, NodeId::starboard(Item::Splitter, x, y)));
                        }
                    }
                }
            }

            // Output cell ahead of this tile.
            let out_pos = Pos::new(tile.x + dx, tile.y + dy);
            if let Some((ox, oy)) = out_pos.to_usize() {
                if world.in_bounds(out_pos.x, out_pos.y) {
                    if let Some(dst_entity) = world.entity_at(ox, oy) {
                        let dst_dir = world.direction_at(ox, oy);
                        let dst_is_belt =
                            matches!(dst_entity, Item::TransportBelt | Item::UndergroundBelt);
                        let dst_is_sink = matches!(dst_entity, Item::Source | Item::Sink);
                        let dst_not_opposing = dst_dir != dir.opposite();
                        if dst_is_belt && dst_not_opposing && !tile_set.contains(&out_pos) {
                            push_lane_preserving(
                                &mut edges,
                                Item::Splitter,
                                (x, y),
                                dst_entity,
                                (ox, oy),
                            );
                        } else if dst_is_sink && dst_not_opposing && !tile_set.contains(&out_pos) {
                            let dst_id = NodeId::single(dst_entity, ox, oy);
                            edges.push((NodeId::port(Item::Splitter, x, y), dst_id.clone()));
                            edges.push((NodeId::starboard(Item::Splitter, x, y), dst_id));
                        }
                    }
                }
            }
        }

        edges
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        // Identity per port-node. Per-output-lane cap and successor
        // divisor are applied in throughput.rs. Each splitter port-node
        // pools all its incoming flow on that lane label.
        input.clone()
    }

    fn is_lane_aware(&self) -> bool {
        true
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::types::PortRole;

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

                            assert_eq!(
                                tiles.len(),
                                width * height,
                                "x={x} y={y} w={width} h={height} dir={dir:?}"
                            );
                            assert_eq!(
                                tiles[0],
                                Pos::new(x as i64, y as i64),
                                "anchor mismatch at x={x} y={y} w={width} h={height} dir={dir:?}"
                            );
                            let unique: std::collections::HashSet<Pos> =
                                tiles.iter().copied().collect();
                            assert_eq!(
                                unique.len(),
                                tiles.len(),
                                "duplicates at x={x} y={y} w={width} h={height} dir={dir:?}"
                            );
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
                            let none_tiles =
                                entity_tiles(x, y, Direction::None, width, height).unwrap();
                            assert_eq!(
                                none_tiles, baseline,
                                "square {width}x{height} at ({x},{y}): None differs from East"
                            );
                        }

                        if width != height {
                            assert!(
                                entity_tiles(x, y, Direction::None, width, height).is_none(),
                                "non-square {width}x{height} with None should return None"
                            );
                        }
                    }
                }

                for &dir in &[Direction::None, Direction::East, Direction::North] {
                    let tiles = entity_tiles(x, y, dir, 1, 1).unwrap();
                    assert_eq!(tiles, vec![Pos::new(x as i64, y as i64)]);
                }
            }
        }
    }

    fn make_belt_chain_world() -> World {
        let mut w = World::empty(5, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        w
    }

    /// Helper: assert a lane-preserving (port→port, stbd→stbd) edge pair
    /// from `src` to `dst` is present in `edges`.
    fn assert_dual_lane_preserving(
        edges: &[Edge],
        src_kind: Item,
        src_pos: (usize, usize),
        dst_kind: Item,
        dst_pos: (usize, usize),
    ) {
        for port in [PortRole::Port, PortRole::Starboard] {
            let src = NodeId {
                entity_kind: src_kind,
                x: src_pos.0,
                y: src_pos.1,
                port,
            };
            let dst = NodeId {
                entity_kind: dst_kind,
                x: dst_pos.0,
                y: dst_pos.1,
                port,
            };
            assert!(
                edges.contains(&(src.clone(), dst.clone())),
                "missing lane-preserving edge {:?} → {:?} (port {:?})",
                src,
                dst,
                port
            );
        }
    }

    #[test]
    fn test_transport_belt_connections_chain() {
        // ASCII: S>>K  (Source, Belt(1), Belt(2), Sink — all east)
        let w = make_belt_chain_world();
        let belt = TransportBelt;
        let edges = belt.connections((1, 0), Direction::East, &w);

        // Source isn't beltish → no backward edge from it.
        // Belt at (2,0) is parallel forward → 2 lane-preserving edges.
        assert_eq!(edges.len(), 2);
        assert_dual_lane_preserving(
            &edges,
            Item::TransportBelt,
            (1, 0),
            Item::TransportBelt,
            (2, 0),
        );
    }

    #[test]
    fn test_transport_belt_chain_second_belt() {
        let w = make_belt_chain_world();
        let belt = TransportBelt;
        let edges = belt.connections((2, 0), Direction::East, &w);
        assert_eq!(edges.len(), 2);
        assert_dual_lane_preserving(
            &edges,
            Item::TransportBelt,
            (1, 0),
            Item::TransportBelt,
            (2, 0),
        );
    }

    #[test]
    fn test_transport_belt_no_opposing_connection() {
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 0, Item::TransportBelt, Direction::West, None);

        let belt = TransportBelt;
        let edges = belt.connections((0, 0), Direction::East, &w);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_belt_dual_edges_in_all_directions() {
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

            assert_eq!(
                edges.len(),
                2,
                "dir={:?} expected 2 forward edges, got {:?}",
                dir,
                edges
            );
            assert_dual_lane_preserving(&edges, Item::TransportBelt, src, Item::TransportBelt, dst);
        }
    }

    #[test]
    fn test_belt_lone_curve_emits_dual_lane_preserving() {
        // Diagram: > v  — east belt feeds south belt at (1,1).
        let mut w = World::empty(3, 3);
        w.place(0, 1, Item::TransportBelt, Direction::East, None);
        w.place(1, 1, Item::TransportBelt, Direction::South, None);

        let belt = TransportBelt;
        let edges = belt.connections((0, 1), Direction::East, &w);

        assert_eq!(edges.len(), 2, "lone curve should emit dual edges");
        assert_dual_lane_preserving(
            &edges,
            Item::TransportBelt,
            (0, 1),
            Item::TransportBelt,
            (1, 1),
        );
    }

    #[test]
    fn test_underground_belt_dual_edges_all_directions() {
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
            assert_eq!(edges.len(), 2);
            assert_dual_lane_preserving(
                &edges,
                Item::UndergroundBelt,
                down,
                Item::UndergroundBelt,
                up,
            );
        }
    }

    #[test]
    fn test_underground_belt_up_emits_no_edges() {
        let mut w = World::empty(5, 1);
        w.place_underground(3, 0, Direction::East, Misc::UndergroundUp);
        w.place(4, 0, Item::TransportBelt, Direction::East, None);
        let ub = UndergroundBelt {
            misc: Misc::UndergroundUp,
        };
        let edges = ub.connections((3, 0), Direction::East, &w);
        assert!(edges.is_empty());
    }

    #[test]
    fn test_inserter_pickup_dual_drop_single() {
        // Inserter has a finite 0.86 i/s budget and drops on a single
        // lane. Pickup combines both source lanes (the inserter caps
        // afterwards, so dual pickup doesn't duplicate the budget).
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 0, Item::Inserter, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);

        let inserter = Inserter;
        let edges = inserter.connections((1, 0), Direction::East, &w);

        // Pickup: 2 edges (belt.port → ins, belt.stbd → ins).
        // Drop:   1 edge  (ins → dst belt.port).
        assert_eq!(edges.len(), 3);
        let ins = NodeId::single(Item::Inserter, 1, 0);
        assert!(edges.contains(&(NodeId::port(Item::TransportBelt, 0, 0), ins.clone())));
        assert!(edges.contains(&(NodeId::starboard(Item::TransportBelt, 0, 0), ins.clone())));
        assert!(edges.contains(&(ins, NodeId::port(Item::TransportBelt, 2, 0))));
        assert!(!edges
            .iter()
            .any(|e| e.1 == NodeId::starboard(Item::TransportBelt, 2, 0)));
    }

    #[test]
    fn test_inserter_with_lane_agnostic_source_emits_single_edge() {
        // Source behind, empty ahead.
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Source, Direction::None, None);
        w.place(1, 0, Item::Inserter, Direction::East, None);

        let inserter = Inserter;
        let edges = inserter.connections((1, 0), Direction::East, &w);
        // Source is lane-agnostic → single pickup edge.
        assert_eq!(edges.len(), 1);
        assert!(edges.contains(&(
            NodeId::single(Item::Source, 0, 0),
            NodeId::single(Item::Inserter, 1, 0),
        )));
    }

    #[test]
    fn test_source_emits_two_edges_to_belt() {
        // Source feeds both lanes of a destination belt.
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);

        let source = Source {
            item: Some(Item::CopperCable),
        };
        let edges = source.connections((0, 0), Direction::East, &w);

        assert_eq!(edges.len(), 2);
        let src = NodeId::single(Item::Source, 0, 0);
        assert!(edges.contains(&(src.clone(), NodeId::port(Item::TransportBelt, 1, 0))));
        assert!(edges.contains(&(src, NodeId::starboard(Item::TransportBelt, 1, 0))));
    }

    #[test]
    fn test_sink_emits_two_pickup_edges_from_belt() {
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let sink = Sink;
        let edges = sink.connections((1, 0), Direction::East, &w);

        assert_eq!(edges.len(), 2);
        let sink_id = NodeId::single(Item::Sink, 1, 0);
        assert!(edges.contains(&(NodeId::port(Item::TransportBelt, 0, 0), sink_id.clone())));
        assert!(edges.contains(&(NodeId::starboard(Item::TransportBelt, 0, 0), sink_id)));
    }

    #[test]
    fn test_assembler_connections() {
        let mut w = World::empty(6, 6);
        w.place(
            1,
            1,
            Item::AssemblingMachine1,
            Direction::None,
            Some(Item::ElectronicCircuit),
        );
        w.place(0, 1, Item::Inserter, Direction::East, None);
        w.place(4, 2, Item::Inserter, Direction::East, None);

        let asm = AssemblingMachine {
            recipe_item: Some(Item::ElectronicCircuit),
        };
        let edges = asm.connections((1, 1), Direction::None, &w);
        // 2 inserters around AM → 2 edges, each between single nodes.
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_belt_transform_flow_caps_at_lane_rate() {
        let belt = TransportBelt;
        let input = HashMap::from([(Item::CopperCable, 20.0)]);
        let output = belt.transform_flow(&input);
        assert!((output[&Item::CopperCable] - LANE_FLOW_RATE).abs() < 1e-9);

        let input = HashMap::from([(Item::CopperCable, 5.0)]);
        let output = belt.transform_flow(&input);
        assert!((output[&Item::CopperCable] - 5.0).abs() < 1e-9);
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
    fn test_source_with_no_item_emits_nothing() {
        let source = Source { item: None };
        let output = source.transform_flow(&HashMap::new());
        assert!(output.is_empty());
    }

    #[test]
    fn test_assembler_transform_flow() {
        let asm = AssemblingMachine {
            recipe_item: Some(Item::ElectronicCircuit),
        };
        let input = HashMap::from([(Item::CopperCable, 3.0), (Item::IronPlate, 1.0)]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 1.0).abs() < 1e-9);

        let input = HashMap::from([(Item::CopperCable, 1.5), (Item::IronPlate, 1.0)]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 0.5).abs() < 1e-9);

        let input = HashMap::from([(Item::CopperCable, 3.0)]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_splitter_connections_east() {
        let mut w = World::empty(5, 3);
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place_splitter(2, 0, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 1, Item::TransportBelt, Direction::East, None);

        let splitter = Splitter;
        let edges = splitter.connections((2, 0), Direction::East, &w);

        // 1 input belt + 2 output belts = 3 conceptual connections × 2 lane-pairs = 6 edges.
        assert_eq!(edges.len(), 6);
        // Belt(1,0) → Splitter(2,0) lane-preserving.
        assert_dual_lane_preserving(&edges, Item::TransportBelt, (1, 0), Item::Splitter, (2, 0));
        // Splitter → Belt(3,0).
        assert_dual_lane_preserving(&edges, Item::Splitter, (2, 0), Item::TransportBelt, (3, 0));
        // Splitter → Belt(3,1).
        assert_dual_lane_preserving(&edges, Item::Splitter, (2, 0), Item::TransportBelt, (3, 1));
    }

    #[test]
    fn test_splitter_transform_flow_is_identity() {
        let splitter = Splitter;
        let input = HashMap::from([(Item::CopperCable, 12.0)]);
        let output = splitter.transform_flow(&input);
        assert_eq!(output[&Item::CopperCable], 12.0);
    }

    #[test]
    fn test_is_lane_aware_per_entity() {
        // Lane-aware: TB, UG, Splitter.
        assert!(TransportBelt.is_lane_aware());
        assert!(UndergroundBelt {
            misc: Misc::UndergroundDown
        }
        .is_lane_aware());
        assert!(Splitter.is_lane_aware());

        // Lane-agnostic: Inserter, Source, Sink, AssemblingMachine.
        assert!(!Inserter.is_lane_aware());
        assert!(!Source { item: None }.is_lane_aware());
        assert!(!Sink.is_lane_aware());
        assert!(!AssemblingMachine { recipe_item: None }.is_lane_aware());
    }

    #[test]
    fn test_is_lane_aware_shim_matches_trait() {
        // The free `is_lane_aware(kind)` shim must agree with the trait
        // method. If a future entity is added but the shim is forgotten,
        // this test will fail.
        for kind in [
            Item::TransportBelt,
            Item::UndergroundBelt,
            Item::Splitter,
            Item::Inserter,
            Item::Source,
            Item::Sink,
            Item::AssemblingMachine1,
        ] {
            let entity = EntityEnum::new(kind, None, Misc::None).expect("placeable items only");
            assert_eq!(
                super::is_lane_aware(kind),
                entity.is_lane_aware(),
                "shim disagrees with trait for {:?}",
                kind
            );
        }

        // Non-placeable items: shim returns false (no entity → not lane-aware).
        assert!(!super::is_lane_aware(Item::CopperPlate));
        assert!(!super::is_lane_aware(Item::IronGearWheel));
    }

    #[test]
    fn test_entity_enum_new_returns_none_for_non_placeable() {
        assert!(EntityEnum::new(Item::CopperCable, None, Misc::None).is_none());
        assert!(EntityEnum::new(Item::IronGearWheel, None, Misc::None).is_none());
        assert!(EntityEnum::new(Item::TransportBelt, None, Misc::None).is_some());
    }

    #[test]
    fn test_inserter_perpendicular_drops_on_far_lane() {
        // Inserter on the NORTH side of an east-facing belt drops on
        // the FAR lane = starboard.
        // Layout (north up):
        //   I       <- (1,0) inserter facing south
        //   >       <- (1,1) belt facing east
        let mut w = World::empty(3, 3);
        w.place(1, 0, Item::Inserter, Direction::South, None);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);

        let inserter = Inserter;
        let edges = inserter.connections((1, 0), Direction::South, &w);

        // Pickup behind: out of bounds → 0 edges. Drop ahead is the belt;
        // perpendicular geometry → FAR lane = starboard of east-facing belt.
        let ins = NodeId::single(Item::Inserter, 1, 0);
        let stbd = NodeId::starboard(Item::TransportBelt, 1, 1);
        let port = NodeId::port(Item::TransportBelt, 1, 1);
        assert!(edges.contains(&(ins.clone(), stbd)));
        assert!(!edges.iter().any(|e| e.1 == port));
    }

    #[test]
    fn test_inserter_perpendicular_other_side_drops_on_other_far() {
        // Symmetric case — inserter on the SOUTH side of the same belt
        // drops on FAR = port (the opposite of stbd-side neighbour).
        let mut w = World::empty(3, 3);
        w.place(1, 2, Item::Inserter, Direction::North, None);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);

        let inserter = Inserter;
        let edges = inserter.connections((1, 2), Direction::North, &w);

        let ins = NodeId::single(Item::Inserter, 1, 2);
        let port = NodeId::port(Item::TransportBelt, 1, 1);
        let stbd = NodeId::starboard(Item::TransportBelt, 1, 1);
        assert!(edges.contains(&(ins.clone(), port)));
        assert!(!edges.iter().any(|e| e.1 == stbd));
    }

    #[test]
    fn test_inserter_in_line_drops_on_port_lane() {
        // I>>>  — inserter facing east, belt facing east. In-line drop
        // → PORT lane.
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Inserter, Direction::East, None);
        w.place(1, 0, Item::TransportBelt, Direction::East, None);

        let inserter = Inserter;
        let edges = inserter.connections((0, 0), Direction::East, &w);

        let ins = NodeId::single(Item::Inserter, 0, 0);
        let port = NodeId::port(Item::TransportBelt, 1, 0);
        let stbd = NodeId::starboard(Item::TransportBelt, 1, 0);
        assert!(edges.contains(&(ins.clone(), port)));
        assert!(!edges.iter().any(|e| e.1 == stbd));
    }

    #[test]
    fn test_inserter_anti_parallel_drops_on_port_lane() {
        // Inserter facing east at (0,0), belt facing west at (1,0).
        // Anti-parallel: still in-line → PORT lane.
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Inserter, Direction::East, None);
        w.place(1, 0, Item::TransportBelt, Direction::West, None);

        let inserter = Inserter;
        let edges = inserter.connections((0, 0), Direction::East, &w);

        let ins = NodeId::single(Item::Inserter, 0, 0);
        let port = NodeId::port(Item::TransportBelt, 1, 0);
        let stbd = NodeId::starboard(Item::TransportBelt, 1, 0);
        assert!(edges.contains(&(ins.clone(), port)));
        assert!(!edges.iter().any(|e| e.1 == stbd));
    }

    #[test]
    fn test_belt_t_junction_emits_side_load() {
        // ASCII (north-up):
        //   >>v<<
        // (0,0)E, (1,0)E, (2,0)S — junction tile, (3,0)W, (4,0)W.
        // The east-going belt (1,0) and the west-going belt (3,0) both
        // forward-feed (2,0). Each side-loads onto its physical side of
        // the junction.
        let mut w = World::empty(5, 2);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::South, None);
        w.place(3, 0, Item::TransportBelt, Direction::West, None);
        w.place(4, 0, Item::TransportBelt, Direction::West, None);

        let belt = TransportBelt;
        // Belt (1,0) east → (2,0) south. Belt is on the WEST side of dst.
        // South-facing dst: south.starboard = west → our_side = stbd.
        // Both our lanes route to dst.stbd.
        let edges = belt.connections((1, 0), Direction::East, &w);
        let dst_stbd = NodeId::starboard(Item::TransportBelt, 2, 0);
        let dst_port = NodeId::port(Item::TransportBelt, 2, 0);
        // The (1,0)→(2,0) forward emit is the perpendicular case: side-load
        // both lanes onto dst.stbd.
        assert!(edges.contains(&(NodeId::port(Item::TransportBelt, 1, 0), dst_stbd.clone())));
        assert!(edges.contains(&(NodeId::starboard(Item::TransportBelt, 1, 0), dst_stbd)));
        // No edges from (1,0) onto dst.port — that side of the junction
        // belongs to the (3,0) west belt.
        assert!(!edges
            .iter()
            .any(|e| e.0 == NodeId::port(Item::TransportBelt, 1, 0) && e.1 == dst_port));

        // Symmetric check on belt (3,0) west → (2,0) south. Belt is on
        // EAST side of dst. South.port = east → our_side = port.
        let edges = belt.connections((3, 0), Direction::West, &w);
        let dst_port = NodeId::port(Item::TransportBelt, 2, 0);
        let dst_stbd = NodeId::starboard(Item::TransportBelt, 2, 0);
        assert!(edges.contains(&(NodeId::port(Item::TransportBelt, 3, 0), dst_port.clone())));
        assert!(edges.contains(&(NodeId::starboard(Item::TransportBelt, 3, 0), dst_port)));
        assert!(!edges
            .iter()
            .any(|e| e.0 == NodeId::port(Item::TransportBelt, 3, 0) && e.1 == dst_stbd));
    }

    #[test]
    fn test_belt_parallel_plus_perpendicular_junction() {
        // ASCII:
        //   v       — (2,0) S, parallel-back source for (2,1)
        //   >v      — (1,1) E forward-feeds (2,1) S
        //   v       — (2,2) S
        //   v       — etc.
        // (2,1) is the junction. It has BOTH a parallel-back (2,0) and a
        // perpendicular (1,1). The east belt (1,1) should detect the
        // junction (because (2,0) is the other source) and side-load.
        let mut w = World::empty(4, 4);
        w.place(2, 0, Item::TransportBelt, Direction::South, None);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);
        w.place(2, 1, Item::TransportBelt, Direction::South, None);
        w.place(2, 2, Item::TransportBelt, Direction::South, None);

        let belt = TransportBelt;
        let edges = belt.connections((1, 1), Direction::East, &w);

        // (1,1) is on the WEST side of (2,1). South.stbd = west → our_side = stbd.
        // Both (1,1) lanes side-load onto (2,1).stbd.
        let dst_stbd = NodeId::starboard(Item::TransportBelt, 2, 1);
        assert!(edges.contains(&(NodeId::port(Item::TransportBelt, 1, 1), dst_stbd.clone())));
        assert!(edges.contains(&(NodeId::starboard(Item::TransportBelt, 1, 1), dst_stbd)));
        // The (1,1)→(2,1) edge does NOT go to dst.port (that's reserved
        // for any source on the other side).
        let dst_port = NodeId::port(Item::TransportBelt, 2, 1);
        assert!(!edges
            .iter()
            .any(|e| e.0 == NodeId::port(Item::TransportBelt, 1, 1) && e.1 == dst_port));
    }

    #[test]
    fn test_belt_lone_curve_stays_lane_preserving() {
        // > v with no other source on (1,1) → still lane-preserving.
        let mut w = World::empty(3, 3);
        w.place(0, 1, Item::TransportBelt, Direction::East, None);
        w.place(1, 1, Item::TransportBelt, Direction::South, None);

        let belt = TransportBelt;
        let edges = belt.connections((0, 1), Direction::East, &w);
        // Lone curve: emit dual lane-preserving edges.
        assert_dual_lane_preserving(
            &edges,
            Item::TransportBelt,
            (0, 1),
            Item::TransportBelt,
            (1, 1),
        );
        // Nothing onto dst.<side> — that's the side-load shape.
        let dst_port = NodeId::port(Item::TransportBelt, 1, 1);
        let dst_stbd = NodeId::starboard(Item::TransportBelt, 1, 1);
        assert!(edges.contains(&(NodeId::port(Item::TransportBelt, 0, 1), dst_port)));
        assert!(edges.contains(&(NodeId::starboard(Item::TransportBelt, 0, 1), dst_stbd)));
    }

    #[test]
    fn test_belt_underground_tunnel_creates_other_source() {
        // UG-down at (0,0)E feeds tunnel exit at (3,0)E (UG-up there).
        // Place a perpendicular belt at (3,1)N forward-feeding (3,0).
        // The UG-up at (3,-)? Actually let me redo — we want (3,0) to
        // be a junction with multiple sources: a UG tunnel coming in and
        // a perpendicular belt.
        //
        // Layout:
        //   d . . u v   (col 0..4, row 0)
        //         B     (col 3, row 1) east belt forward to (3,0)? No,
        // perpendicular feeders need to be a side neighbour of (3,0).
        //
        // Simpler: junction at (3,0) with a UG tunnel exit (UG-up at
        // (3,0)) and a perpendicular belt at (3,1)N feeding it.
        //
        // Actually the perpendicular check operates on the BELT calling
        // connections. Let me model directly: the test belt is the
        // perpendicular one (3,1)N. Its forward = (3,0). Junction check
        // for (3,1)→(3,0): does (3,0) have any other source? Yes: the
        // UG-up at (3,0)?... no, (3,0) IS the dst. Its parallel-back is
        // (3,1) (excluding=us), so no. UG tunnel: search (3, k) for
        // k>0 — but (3,0) is east-facing? Not yet specified.
        //
        // Let me set up a proper case: UG-down at (0,0) east, UG-up at
        // (3,0) east. Perpendicular belt (3,1) north feeds (3,0).
        //
        // For belt (3,1)→(3,0): is there another source on (3,0)?
        // Parallel-back: (2,0) — empty, no. UG tunnel: search west
        // (since dst_dir=east → back direction is west) for UG-down.
        // Found at (0,0) → other source! → side-load.
        let mut w = World::empty(5, 3);
        w.place_underground(0, 0, Direction::East, Misc::UndergroundDown);
        w.place_underground(3, 0, Direction::East, Misc::UndergroundUp);
        w.place(3, 1, Item::TransportBelt, Direction::North, None);

        let belt = TransportBelt;
        let edges = belt.connections((3, 1), Direction::North, &w);

        // (3,1) → (3,0) is perpendicular forward. UG-down at (0,0) is in
        // the back-search range → junction → side-load.
        // (3,1) is on the SOUTH side of east-facing (3,0).
        // East.stbd = south → our_side = stbd.
        let dst_stbd = NodeId::starboard(Item::UndergroundBelt, 3, 0);
        assert!(edges.contains(&(NodeId::port(Item::TransportBelt, 3, 1), dst_stbd.clone())));
        assert!(edges.contains(&(NodeId::starboard(Item::TransportBelt, 3, 1), dst_stbd)));
    }

    #[test]
    fn test_inserter_drop_lane_helper() {
        // Inserter on north (port) side of east-facing belt → far lane = stbd.
        // Inserter at (1,0) facing south, dst belt at (1,1) facing east.
        assert_eq!(
            inserter_drop_lane((1, 0), Direction::South, (1, 1), Direction::East),
            LaneSide::Starboard
        );
        // Inserter on south (stbd) side of east-facing belt → far lane = port.
        assert_eq!(
            inserter_drop_lane((1, 2), Direction::North, (1, 1), Direction::East),
            LaneSide::Port
        );
        // In-line (parallel) drop: I>>>  → port.
        assert_eq!(
            inserter_drop_lane((0, 0), Direction::East, (1, 0), Direction::East),
            LaneSide::Port
        );
    }
}
