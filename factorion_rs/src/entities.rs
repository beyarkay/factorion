use std::collections::HashMap;

use crate::types::{Direction, EntityKind, Item, Misc, NodeId, get_recipe};
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
    /// Return the edges this entity contributes to the graph.
    ///
    /// `pos` is the entity's (x, y) position. `dir` is the entity's facing direction.
    /// `world` provides read access to the grid for inspecting neighbors.
    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge>;

    /// Given accumulated input flow rates, compute output flow rates.
    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64>;

    /// Maximum items/second this entity can transfer.
    fn flow_rate(&self) -> f64;
}

/// Dispatch: create the appropriate trait object for an entity kind.
pub fn make_entity(kind: EntityKind, item: Item, misc: Misc) -> Box<dyn FactoryEntity> {
    match kind {
        EntityKind::TransportBelt => Box::new(TransportBelt),
        EntityKind::Inserter => Box::new(Inserter),
        EntityKind::AssemblingMachine1 => Box::new(AssemblingMachine { recipe_item: item }),
        EntityKind::UndergroundBelt => Box::new(UndergroundBelt { misc }),
        EntityKind::Sink => Box::new(Sink),
        EntityKind::Source => Box::new(Source { item }),
        EntityKind::Empty => Box::new(EmptyEntity),
    }
}

// ── Transport Belt ──────────────────────────────────────────────────────────

struct TransportBelt;

impl FactoryEntity for TransportBelt {
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
                edges.push((
                    NodeId::new(src_entity, sx, sy),
                    self_id.clone(),
                ));
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
            let opposite = Direction::South as i64 - Direction::North as i64;
            let dst_opposing = dst_is_belt
                && (dst_dir as i64 - dir as i64).abs() == opposite;

            if dst_is_belt && !dst_opposing {
                edges.push((
                    self_id,
                    NodeId::new(dst_entity, dx_u, dy_u),
                ));
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

    fn flow_rate(&self) -> f64 {
        15.0
    }
}

// ── Inserter ────────────────────────────────────────────────────────────────

struct Inserter;

impl FactoryEntity for Inserter {
    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        inserter_connections(EntityKind::Inserter, pos, dir, world)
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        input
            .iter()
            .map(|(&item, &rate)| (item, rate.min(self.flow_rate())))
            .collect()
    }

    fn flow_rate(&self) -> f64 {
        0.86
    }
}

// ── Assembling Machine ──────────────────────────────────────────────────────

struct AssemblingMachine {
    recipe_item: Item,
}

impl FactoryEntity for AssemblingMachine {
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

                // Only inserters can interact with assembling machines
                if other_entity != EntityKind::Inserter {
                    continue;
                }

                let other_dir = world.direction_at(nx_u, ny_u);
                let other_id = NodeId::new(EntityKind::Inserter, nx_u, ny_u);

                // Determine if the inserter is inserting INTO or OUT OF the assembler.
                // The inserter's direction points from pickup to dropoff.
                // If the inserter faces toward the assembler interior → inserting into assembler.
                // If the inserter faces away from the assembler interior → taking from assembler.
                let is_inserting_into_assembler =
                    (other_dir == Direction::North && ddy < 0)
                        || (other_dir == Direction::South && ddy > 0)
                        || (other_dir == Direction::West && ddx < 0)
                        || (other_dir == Direction::East && ddx > 0);

                // Wait — the Python code uses a different convention. Let me re-check:
                // Python: Direction is "self -> other" where self=assembler, other=inserter.
                // If other_d == NORTH and dy < 0, that means the inserter is above the assembler
                // and facing north (away from assembler), so direction is assembler -> inserter.
                // Actually re-reading the Python more carefully:
                //   if (other_d == Direction.NORTH and dy < 0)
                //   -> src = self_str (assembler), dst = other_str (inserter)
                // So when the inserter is above (dy < 0) and faces north, items flow FROM assembler TO inserter.
                // Otherwise, items flow FROM inserter TO assembler.
                if is_inserting_into_assembler {
                    // Assembler → Inserter (inserter takes from assembler)
                    edges.push((self_id.clone(), other_id));
                } else {
                    // Inserter → Assembler (inserter puts into assembler)
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

    fn flow_rate(&self) -> f64 {
        0.5
    }
}

// ── Underground Belt ────────────────────────────────────────────────────────

struct UndergroundBelt {
    misc: Misc,
}

impl FactoryEntity for UndergroundBelt {
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
            let (_dx, _dy) = dir.delta();
            let (_src_x, _src_y, dst_x, dst_y) = match dir {
                Direction::East => (x as i64 - 1, y as i64, x as i64 + delta as i64, y as i64),
                Direction::West => (x as i64 + 1, y as i64, x as i64 - delta as i64, y as i64),
                Direction::North => (x as i64, y as i64 + 1, x as i64, y as i64 - delta as i64),
                Direction::South => (x as i64, y as i64 - 1, x as i64, y as i64 + delta as i64),
                Direction::None => continue,
            };

            if !world.in_bounds(dst_x, dst_y) {
                continue;
            }

            let dst_xu = dst_x as usize;
            let dst_yu = dst_y as usize;
            let dst_entity = world.entity_at(dst_xu, dst_yu);

            let going_underground = dst_entity == EntityKind::UndergroundBelt
                && self.misc == Misc::UndergroundDown;

            let cxn_to_belt = matches!(dst_entity, EntityKind::TransportBelt)
                && self.misc == Misc::UndergroundUp;

            if going_underground || cxn_to_belt {
                edges.push((
                    self_id.clone(),
                    NodeId::new(dst_entity, dst_xu, dst_yu),
                ));
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

    fn flow_rate(&self) -> f64 {
        15.0
    }
}

// ── Source (stack_inserter) ─────────────────────────────────────────────────
//
// In the Python code, stack_inserter contains "inserter" in its name, so it
// uses the same inserter connection logic: picks up from behind, drops onto
// belts/assemblers ahead. Its special behavior is that it has infinite output.

struct Source {
    item: Item,
}

impl FactoryEntity for Source {
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

    fn flow_rate(&self) -> f64 {
        f64::INFINITY
    }
}

// ── Sink (bulk_inserter) ────────────────────────────────────────────────────
//
// Same as Source: bulk_inserter contains "inserter", so it uses inserter
// connection logic. Its special behavior is infinite throughput at the output.

struct Sink;

impl FactoryEntity for Sink {
    fn connections(&self, pos: (usize, usize), dir: Direction, world: &World) -> Vec<Edge> {
        inserter_connections(EntityKind::Sink, pos, dir, world)
    }

    fn transform_flow(&self, input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        // Sinks pass through everything (infinite capacity)
        input.clone()
    }

    fn flow_rate(&self) -> f64 {
        f64::INFINITY
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

// ── Empty ───────────────────────────────────────────────────────────────────

struct EmptyEntity;

impl FactoryEntity for EmptyEntity {
    fn connections(&self, _pos: (usize, usize), _dir: Direction, _world: &World) -> Vec<Edge> {
        Vec::new()
    }

    fn transform_flow(&self, _input: &HashMap<Item, f64>) -> HashMap<Item, f64> {
        HashMap::new()
    }

    fn flow_rate(&self) -> f64 {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Channel;

    /// Helper to build a small world with a few entities for testing connections.
    fn make_belt_chain_world() -> World {
        // 5x1 world: Source(east) -> Belt(east) -> Belt(east) -> Sink(east)
        let mut w = World::empty(5, 1);
        // Source at (0,0)
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(0, 0, Channel::Items, Item::CopperCable as i64);
        // Belt at (1,0)
        w.set(1, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);
        // Belt at (2,0)
        w.set(2, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(2, 0, Channel::Direction, Direction::East as i64);
        // Sink at (3,0)
        w.set(3, 0, Channel::Entities, EntityKind::Sink as i64);
        w.set(3, 0, Channel::Direction, Direction::East as i64);
        w.set(3, 0, Channel::Items, Item::CopperCable as i64);
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
        w.set(0, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(1, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(1, 0, Channel::Direction, Direction::West as i64);

        let belt = TransportBelt;
        let edges = belt.connections((0, 0), Direction::East, &w);
        // The belt ahead is facing the opposite direction → no connection
        assert!(edges.is_empty());
    }

    #[test]
    fn test_inserter_connections() {
        // Inserter at (1,0) facing east, source at (0,0), belt at (2,0)
        let mut w = World::empty(3, 1);
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(1, 0, Channel::Entities, EntityKind::Inserter as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);
        w.set(2, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(2, 0, Channel::Direction, Direction::East as i64);

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
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(1, 0, Channel::Entities, EntityKind::Inserter as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);

        let inserter = Inserter;
        let edges = inserter.connections((1, 0), Direction::East, &w);

        // Only source→inserter, no inserter→empty
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_assembler_connections() {
        // 6x6 world with assembler at (1,1) and inserters on perimeter
        let mut w = World::empty(6, 6);
        w.set(1, 1, Channel::Entities, EntityKind::AssemblingMachine1 as i64);
        w.set(1, 1, Channel::Items, Item::ElectronicCircuit as i64);

        // Inserter at (0,1) facing east → inserting into assembler
        w.set(0, 1, Channel::Entities, EntityKind::Inserter as i64);
        w.set(0, 1, Channel::Direction, Direction::East as i64);

        // Inserter at (4,2) facing east → taking from assembler (facing away)
        w.set(4, 2, Channel::Entities, EntityKind::Inserter as i64);
        w.set(4, 2, Channel::Direction, Direction::East as i64);

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
        let input = HashMap::from([
            (Item::CopperCable, 6.0),
            (Item::IronPlate, 2.0),
        ]);
        let output = asm.transform_flow(&input);
        assert!((output[&Item::ElectronicCircuit] - 2.0).abs() < 1e-9);

        // Half input: 3 copper cable + 2 iron plate → ratio = min(3/6, 2/2) = 0.5
        let input = HashMap::from([
            (Item::CopperCable, 3.0),
            (Item::IronPlate, 2.0),
        ]);
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
        w.set(1, 0, Channel::Entities, EntityKind::UndergroundBelt as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);
        w.set(1, 0, Channel::Misc, Misc::UndergroundDown as i64);

        w.set(3, 0, Channel::Entities, EntityKind::UndergroundBelt as i64);
        w.set(3, 0, Channel::Direction, Direction::East as i64);
        w.set(3, 0, Channel::Misc, Misc::UndergroundUp as i64);

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
        w.set(3, 0, Channel::Entities, EntityKind::UndergroundBelt as i64);
        w.set(3, 0, Channel::Direction, Direction::East as i64);
        w.set(3, 0, Channel::Misc, Misc::UndergroundUp as i64);

        w.set(4, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(4, 0, Channel::Direction, Direction::East as i64);

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
