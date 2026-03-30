use std::collections::HashMap;

use crate::types::{get_recipe, Direction, EntityKind, Item, Misc, NodeId, Pos};
use crate::world::World;

/// An edge in the factory graph: (source_node, destination_node).
pub type Edge = (NodeId, NodeId);

/// Trait abstracting over factory entity types.
///
/// Each entity type implements this trait to define:
/// - How it connects to neighboring entities (graph edges)
/// - How it transforms input flow into output flow
/// - Its maximum throughput rate
pub trait FactoryEntity {
    /// Which entity kind this is. Used to look up flow_rate from the
    /// single source of truth in EntityKind::flow_rate().
    fn kind(&self) -> EntityKind;

    /// Return the edges this entity contributes to the graph.
    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge>;

    /// Given accumulated input flow rates, compute output flow rates.
    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64>;

    /// Maximum items/second this entity can transfer.
    /// Default delegates to EntityKind::flow_rate().
    fn flow_rate(&self) -> f64 {
        self.kind().flow_rate()
    }
}

/// Stack-allocated entity dispatch. Wraps concrete entity structs.
pub enum EntityEnum {
    TransportBelt(TransportBelt),
    Inserter(Inserter),
    AssemblingMachine(AssemblingMachine),
    UndergroundBelt(UndergroundBelt),
    Sink(Sink),
    Source(Source),
    Empty(EmptyEntity),
}

impl EntityEnum {
    pub fn new(kind: EntityKind, item: Item, misc: Misc) -> Self {
        match kind {
            EntityKind::TransportBelt => Self::TransportBelt(TransportBelt),
            EntityKind::Inserter => Self::Inserter(Inserter),
            EntityKind::AssemblingMachine1 => {
                Self::AssemblingMachine(AssemblingMachine { recipe_item: item })
            }
            EntityKind::UndergroundBelt => Self::UndergroundBelt(UndergroundBelt { misc }),
            EntityKind::Sink => Self::Sink(Sink),
            EntityKind::Source => Self::Source(Source { item }),
            EntityKind::Empty => Self::Empty(EmptyEntity),
        }
    }
}

impl FactoryEntity for EntityEnum {
    fn kind(&self) -> EntityKind {
        match self {
            Self::TransportBelt(e) => e.kind(),
            Self::Inserter(e) => e.kind(),
            Self::AssemblingMachine(e) => e.kind(),
            Self::UndergroundBelt(e) => e.kind(),
            Self::Sink(e) => e.kind(),
            Self::Source(e) => e.kind(),
            Self::Empty(e) => e.kind(),
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
            Self::Empty(e) => e.connections(pos, dir, world),
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
            Self::Empty(e) => e.transform_flow(input),
        }
    }
}

// ── Transport Belt ──────────────────────────────────────────────────────────

pub struct TransportBelt;

impl FactoryEntity for TransportBelt {
    fn kind(&self) -> EntityKind {
        EntityKind::TransportBelt
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        let mut edges = Vec::new();
        let (x, y) = pos;
        let (dx, dy) = dir.delta();
        let self_id = NodeId::new(EntityKind::TransportBelt, x, y);

        // Source: the cell behind this belt (opposite of facing direction)
        let src_x = x as i64 - dx;
        let src_y = y as i64 - dy;
        if world.in_bounds(src_x, src_y) {
            let sx = src_x as usize;
            let sy = src_y as usize;
            let src_entity = world.entity_at(sx, sy);
            let src_dir = world.direction_at(sx, sy);
            let src_misc = world.misc_at(sx, sy);

            let src_is_beltish = matches!(
                src_entity,
                EntityKind::TransportBelt | EntityKind::UndergroundBelt
            ) && src_dir == dir
                // Don't connect from a downwards underground belt
                && !(src_entity == EntityKind::UndergroundBelt
                    && src_misc == Misc::UndergroundDown);

            if src_is_beltish {
                edges.push((NodeId::new(src_entity, sx, sy), self_id.clone()));
            }
        }

        // Destination: the cell ahead of this belt
        let dst_x = x as i64 + dx;
        let dst_y = y as i64 + dy;
        if world.in_bounds(dst_x, dst_y) {
            let dx_u = dst_x as usize;
            let dy_u = dst_y as usize;
            let dst_entity = world.entity_at(dx_u, dy_u);
            let dst_dir = world.direction_at(dx_u, dy_u);

            let dst_is_belt = matches!(
                dst_entity,
                EntityKind::TransportBelt | EntityKind::UndergroundBelt
            );
            // Don't connect to a belt facing the opposite direction
            let dst_opposing = dst_is_belt && dst_dir == dir.opposite();

            if dst_is_belt && !dst_opposing {
                edges.push((self_id, NodeId::new(dst_entity, dx_u, dy_u)));
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

// ── Inserter ────────────────────────────────────────────────────────────────

pub struct Inserter;

impl FactoryEntity for Inserter {
    fn kind(&self) -> EntityKind {
        EntityKind::Inserter
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        inserter_connections(EntityKind::Inserter, pos, dir, world)
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
    recipe_item: Item,
}

impl FactoryEntity for AssemblingMachine {
    fn kind(&self) -> EntityKind {
        EntityKind::AssemblingMachine1
    }

    fn connections(&self, pos: (usize, usize), _dir: Direction, world: &World) -> Vec<Edge> {
        let mut edges = Vec::new();
        let (x, y) = pos;
        let self_id = NodeId::new(EntityKind::AssemblingMachine1, x, y);

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
                let other_entity = world.entity_at(nx_u, ny_u);

                // Only inserter-like entities can interact with assembling machines.
                // In Python, Source (stack_inserter) and Sink (bulk_inserter)
                // both contain "inserter" in their name, so they match too.
                if !matches!(
                    other_entity,
                    EntityKind::Inserter | EntityKind::Source | EntityKind::Sink
                ) {
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
        if self.recipe_item == Item::Empty {
            return HashMap::new();
        }

        let recipe = match get_recipe(self.recipe_item) {
            Some(r) => r,
            None => return HashMap::new(),
        };

        // Find the minimum ratio of available input to required input
        let mut min_ratio: f64 = 1.0;
        for (item, &required) in &recipe.consumes {
            let available = input.get(item).copied().unwrap_or(0.0);
            let ratio = available / required;
            min_ratio = min_ratio.min(ratio);
        }

        // Produce outputs scaled by the minimum ratio
        recipe
            .produces
            .iter()
            .map(|(&item, &rate)| (item, rate * min_ratio))
            .collect()
    }
}

// ── Underground Belt ────────────────────────────────────────────────────────

pub struct UndergroundBelt {
    misc: Misc,
}

impl FactoryEntity for UndergroundBelt {
    fn kind(&self) -> EntityKind {
        EntityKind::UndergroundBelt
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        let mut edges = Vec::new();
        let (x, y) = pos;
        let self_id = NodeId::new(EntityKind::UndergroundBelt, x, y);

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
            let dst_entity = world.entity_at(dst_xu, dst_yu);

            let going_underground =
                dst_entity == EntityKind::UndergroundBelt && self.misc == Misc::UndergroundDown;

            let cxn_to_belt =
                matches!(dst_entity, EntityKind::TransportBelt) && self.misc == Misc::UndergroundUp;

            if going_underground || cxn_to_belt {
                edges.push((self_id.clone(), NodeId::new(dst_entity, dst_xu, dst_yu)));
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

// ── Source (stack_inserter) ─────────────────────────────────────────────────
//
// In the Python code, stack_inserter contains "inserter" in its name, so it
// uses the same inserter connection logic: picks up from behind, drops onto
// belts/assemblers ahead. Its special behavior is that it has infinite output.

pub struct Source {
    item: Item,
}

impl FactoryEntity for Source {
    fn kind(&self) -> EntityKind {
        EntityKind::Source
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        // Same connection logic as inserter
        inserter_connections(EntityKind::Source, pos, dir, world)
    }

    fn transform_flow(&self, _input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        // Sources produce infinite items
        let mut output = HashMap::new();
        output.insert(self.item, f64::INFINITY);
        output
    }
}

// ── Sink (bulk_inserter) ────────────────────────────────────────────────────
//
// Same as Source: bulk_inserter contains "inserter", so it uses inserter
// connection logic. Its special behavior is infinite throughput at the output.

pub struct Sink;

impl FactoryEntity for Sink {
    fn kind(&self) -> EntityKind {
        EntityKind::Sink
    }

    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        inserter_connections(EntityKind::Sink, pos, dir, world)
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        // Sinks pass through everything (infinite capacity)
        input.clone()
    }
}

/// Shared inserter-style connection logic used by Inserter, Source, and Sink.
///
/// Picks up from the entity behind (any non-empty entity), drops onto the
/// entity ahead (only belts or assembling machines).
fn inserter_connections(
    self_kind: EntityKind,
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
        let src_entity = world.entity_at(sx, sy);
        if src_entity != EntityKind::Empty {
            edges.push((NodeId::new(src_entity, sx, sy), self_id.clone()));
        }
    }

    // Drop onto the cell ahead (in facing direction)
    let dst_x = x as i64 + dx;
    let dst_y = y as i64 + dy;
    if world.in_bounds(dst_x, dst_y) {
        let dx_u = dst_x as usize;
        let dy_u = dst_y as usize;
        let dst_entity = world.entity_at(dx_u, dy_u);
        // Can only insert into belts or assembling machines
        let dst_is_insertable = matches!(
            dst_entity,
            EntityKind::TransportBelt
                | EntityKind::UndergroundBelt
                | EntityKind::AssemblingMachine1
        );
        if dst_is_insertable {
            edges.push((self_id, NodeId::new(dst_entity, dx_u, dy_u)));
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

// ── Empty ───────────────────────────────────────────────────────────────────

pub struct EmptyEntity;

impl FactoryEntity for EmptyEntity {
    fn kind(&self) -> EntityKind {
        EntityKind::Empty
    }

    fn connections(&self, _pos: (usize, usize), _dir: Direction, _world: &World) -> Vec<Edge> {
        Vec::new()
    }

    fn transform_flow(&self, _input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        HashMap::new()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Helper to build a small world with a few entities for testing connections.
    fn make_belt_chain_world() -> World {
        // 5x1 world: Source(east) -> Belt(east) -> Belt(east) -> Sink(east)
        let mut w = World::empty(5, 1);
        w.place(0, 0, EntityKind::Source, Direction::East, Item::CopperCable);
        w.place(
            1,
            0,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );
        w.place(
            2,
            0,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );
        w.place(3, 0, EntityKind::Sink, Direction::East, Item::CopperCable);
        w
    }

    #[test]
    fn test_transport_belt_connections_chain() {
        let w = make_belt_chain_world();

        // Belt at (1,0) should connect from source behind and to belt ahead
        let belt = TransportBelt;
        let edges = belt.connections((1, 0), Direction::East, &w);

        // Source is behind but it's not a belt, so belt doesn't create that edge
        // Belt at (2,0) is ahead and is a belt with same direction → edge created
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, NodeId::new(EntityKind::TransportBelt, 1, 0));
        assert_eq!(edges[0].1, NodeId::new(EntityKind::TransportBelt, 2, 0));
    }

    #[test]
    fn test_transport_belt_chain_second_belt() {
        let w = make_belt_chain_world();

        // Belt at (2,0) should connect from belt behind
        let belt = TransportBelt;
        let edges = belt.connections((2, 0), Direction::East, &w);

        // Belt at (1,0) behind → edge from (1,0) to (2,0)
        // Sink at (3,0) ahead is not a belt → no forward edge
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].0, NodeId::new(EntityKind::TransportBelt, 1, 0));
        assert_eq!(edges[0].1, NodeId::new(EntityKind::TransportBelt, 2, 0));
    }

    #[test]
    fn test_transport_belt_no_opposing_connection() {
        // Two belts facing each other should not connect
        let mut w = World::empty(3, 1);
        w.place(
            0,
            0,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );
        w.place(
            1,
            0,
            EntityKind::TransportBelt,
            Direction::West,
            Item::Empty,
        );

        let belt = TransportBelt;
        let edges = belt.connections((0, 0), Direction::East, &w);
        // The belt ahead is facing the opposite direction → no connection
        assert!(edges.is_empty());
    }

    #[test]
    fn test_inserter_connections() {
        // Inserter at (1,0) facing east, source at (0,0), belt at (2,0)
        let mut w = World::empty(3, 1);
        w.place(0, 0, EntityKind::Source, Direction::East, Item::Empty);
        w.place(1, 0, EntityKind::Inserter, Direction::East, Item::Empty);
        w.place(
            2,
            0,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );

        let inserter = Inserter;
        let edges = inserter.connections((1, 0), Direction::East, &w);

        assert_eq!(edges.len(), 2);
        // Source → Inserter
        assert!(edges.contains(&(
            NodeId::new(EntityKind::Source, 0, 0),
            NodeId::new(EntityKind::Inserter, 1, 0),
        )));
        // Inserter → Belt
        assert!(edges.contains(&(
            NodeId::new(EntityKind::Inserter, 1, 0),
            NodeId::new(EntityKind::TransportBelt, 2, 0),
        )));
    }

    #[test]
    fn test_inserter_wont_drop_on_empty() {
        // Inserter facing east, source behind, empty ahead
        let mut w = World::empty(3, 1);
        w.place(0, 0, EntityKind::Source, Direction::None, Item::Empty);
        w.place(1, 0, EntityKind::Inserter, Direction::East, Item::Empty);

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
            EntityKind::AssemblingMachine1,
            Direction::None,
            Item::ElectronicCircuit,
        );

        // Inserter at (0,1) facing east → inserting into assembler
        w.place(0, 1, EntityKind::Inserter, Direction::East, Item::Empty);

        // Inserter at (4,2) facing east → taking from assembler (facing away)
        w.place(4, 2, EntityKind::Inserter, Direction::East, Item::Empty);

        let asm = AssemblingMachine {
            recipe_item: Item::ElectronicCircuit,
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
    fn test_source_transform_flow() {
        let source = Source {
            item: Item::CopperCable,
        };
        let output = source.transform_flow(&HashMap::new());
        assert_eq!(output[&Item::CopperCable], f64::INFINITY);
    }

    #[test]
    fn test_assembler_transform_flow() {
        let asm = AssemblingMachine {
            recipe_item: Item::ElectronicCircuit,
        };

        // Full input: 6 copper cable + 2 iron plate → 2 electronic circuits
        let input = HashMap::from([(Item::CopperCable, 6.0), (Item::IronPlate, 2.0)]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 2.0).abs() < 1e-9);

        // Half input: 3 copper cable + 2 iron plate → ratio = min(3/6, 2/2) = 0.5
        let input = HashMap::from([(Item::CopperCable, 3.0), (Item::IronPlate, 2.0)]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 1.0).abs() < 1e-9);

        // Missing ingredient
        let input = HashMap::from([(Item::CopperCable, 6.0)]);
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
        assert_eq!(edges[0].0, NodeId::new(EntityKind::UndergroundBelt, 1, 0));
        assert_eq!(edges[0].1, NodeId::new(EntityKind::UndergroundBelt, 3, 0));
    }

    #[test]
    fn test_underground_belt_up_connections() {
        // Underground up at (3,0) facing east, belt at (4,0) facing east
        let mut w = World::empty(5, 1);
        w.place_underground(3, 0, Direction::East, Misc::UndergroundUp);
        w.place(
            4,
            0,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );

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
    fn test_empty_entity() {
        let e = EmptyEntity;
        let w = World::empty(3, 3);
        assert!(e.connections((0, 0), Direction::None, &w).is_empty());
        assert!(e.transform_flow(&HashMap::new()).is_empty());
        assert_eq!(e.flow_rate(), 0.0);
    }
}
