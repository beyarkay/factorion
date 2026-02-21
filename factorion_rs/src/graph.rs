use std::collections::HashMap;

use crate::entities::make_entity;
use crate::types::{EntityKind, Item, NodeId};
use crate::world::World;

/// A node in the factory graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    #[allow(dead_code)]
    pub id: NodeId,
    pub entity_kind: EntityKind,
    pub item: Item,
    /// Recipe item for assembling machines (determines what they craft).
    pub recipe_item: Item,
    /// Accumulated input flow rates per item type.
    pub input: HashMap<Item, f64>,
    /// Computed output flow rates per item type.
    pub output: HashMap<Item, f64>,
}

/// A directed graph representing a factory's entity connections.
#[derive(Debug)]
pub struct FactoryGraph {
    /// Nodes indexed by their position in this vec.
    pub nodes: Vec<GraphNode>,
    /// Map from NodeId to index in `nodes`.
    #[allow(dead_code)]
    pub node_index: HashMap<NodeId, usize>,
    /// Adjacency list: edges[i] = list of node indices that node i has edges TO.
    pub successors: Vec<Vec<usize>>,
    /// Reverse adjacency: predecessors[i] = list of node indices that have edges TO node i.
    pub predecessors: Vec<Vec<usize>>,
}

impl FactoryGraph {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    #[cfg(test)]
    pub fn get_node(&self, idx: usize) -> &GraphNode {
        &self.nodes[idx]
    }

    #[cfg(test)]
    pub fn get_node_mut(&mut self, idx: usize) -> &mut GraphNode {
        &mut self.nodes[idx]
    }

    #[cfg(test)]
    pub fn get_index(&self, id: &NodeId) -> Option<usize> {
        self.node_index.get(id).copied()
    }
}

/// Build a directed graph from a World, mirroring the Python `world2graph` function.
pub fn build_graph(world: &World) -> FactoryGraph {
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut node_index: HashMap<NodeId, usize> = HashMap::new();
    let mut edge_list: Vec<(NodeId, NodeId)> = Vec::new();

    // First pass: create nodes for all non-empty cells
    for x in 0..world.width() {
        for y in 0..world.height() {
            let entity_kind = world.entity_at(x, y);
            if entity_kind == EntityKind::Empty {
                continue;
            }

            let item = world.item_at(x, y);
            let direction = world.direction_at(x, y);
            let misc = world.misc_at(x, y);

            let node_id = NodeId::new(entity_kind, x, y);

            // Set initial output for sources (stack_inserters produce infinite items)
            let output = if entity_kind == EntityKind::Source {
                let mut m = HashMap::new();
                m.insert(item, f64::INFINITY);
                m
            } else {
                HashMap::new()
            };

            let idx = nodes.len();
            nodes.push(GraphNode {
                id: node_id.clone(),
                entity_kind,
                item,
                recipe_item: if entity_kind == EntityKind::AssemblingMachine1 {
                    item
                } else {
                    Item::Empty
                },
                input: HashMap::new(),
                output,
            });
            node_index.insert(node_id, idx);

            // Get connections from the entity trait impl
            let entity = make_entity(entity_kind, item, misc);
            let edges = entity.connections((x, y), direction, world);
            edge_list.extend(edges);
        }
    }

    // Build adjacency lists from collected edges
    let n = nodes.len();
    let mut successors = vec![Vec::new(); n];
    let mut predecessors = vec![Vec::new(); n];

    for (src_id, dst_id) in &edge_list {
        if let (Some(&src_idx), Some(&dst_idx)) = (node_index.get(src_id), node_index.get(dst_id)) {
            if !successors[src_idx].contains(&dst_idx) {
                successors[src_idx].push(dst_idx);
            }
            if !predecessors[dst_idx].contains(&src_idx) {
                predecessors[dst_idx].push(src_idx);
            }
        }
    }

    FactoryGraph {
        nodes,
        node_index,
        successors,
        predecessors,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Channel, Direction, Misc};

    #[test]
    fn test_empty_world_graph() {
        let w = World::empty(5, 5);
        let g = build_graph(&w);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_single_belt_no_edges() {
        let mut w = World::empty(3, 3);
        w.set(1, 1, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(1, 1, Channel::Direction, Direction::East as i64);

        let g = build_graph(&w);
        assert_eq!(g.node_count(), 1);
        assert!(g.successors[0].is_empty());
        assert!(g.predecessors[0].is_empty());
    }

    #[test]
    fn test_belt_chain_graph() {
        // Source → Belt → Belt → Sink
        let mut w = World::empty(5, 1);
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(0, 0, Channel::Items, Item::CopperCable as i64);

        w.set(1, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);

        w.set(2, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(2, 0, Channel::Direction, Direction::East as i64);

        w.set(3, 0, Channel::Entities, EntityKind::Sink as i64);
        w.set(3, 0, Channel::Direction, Direction::East as i64);
        w.set(3, 0, Channel::Items, Item::CopperCable as i64);

        let g = build_graph(&w);
        assert_eq!(g.node_count(), 4);

        // Belt(1,0) → Belt(2,0) edge should exist
        let belt1 = g
            .get_index(&NodeId::new(EntityKind::TransportBelt, 1, 0))
            .unwrap();
        let belt2 = g
            .get_index(&NodeId::new(EntityKind::TransportBelt, 2, 0))
            .unwrap();
        assert!(g.successors[belt1].contains(&belt2));
        assert!(g.predecessors[belt2].contains(&belt1));

        // Source uses inserter-style connections: drops onto belt at (1,0)
        let source = g.get_index(&NodeId::new(EntityKind::Source, 0, 0)).unwrap();
        assert!(g.successors[source].contains(&belt1));

        // Sink picks up from belt at (2,0)
        let sink = g.get_index(&NodeId::new(EntityKind::Sink, 3, 0)).unwrap();
        assert!(g.predecessors[sink].contains(&belt2));
    }

    #[test]
    fn test_inserter_chain_graph() {
        // Source → Inserter → Belt → Inserter → Sink
        let mut w = World::empty(5, 1);
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(0, 0, Channel::Items, Item::CopperCable as i64);

        w.set(1, 0, Channel::Entities, EntityKind::Inserter as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);

        w.set(2, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(2, 0, Channel::Direction, Direction::East as i64);

        w.set(3, 0, Channel::Entities, EntityKind::Inserter as i64);
        w.set(3, 0, Channel::Direction, Direction::East as i64);

        w.set(4, 0, Channel::Entities, EntityKind::Sink as i64);
        w.set(4, 0, Channel::Direction, Direction::East as i64);
        w.set(4, 0, Channel::Items, Item::CopperCable as i64);

        let g = build_graph(&w);
        assert_eq!(g.node_count(), 5);

        // Source → Inserter(1,0)
        let source = g.get_index(&NodeId::new(EntityKind::Source, 0, 0)).unwrap();
        let ins1 = g
            .get_index(&NodeId::new(EntityKind::Inserter, 1, 0))
            .unwrap();
        assert!(g.successors[source].contains(&ins1) || g.predecessors[ins1].contains(&source));

        // Inserter(1,0) → Belt(2,0)
        let belt = g
            .get_index(&NodeId::new(EntityKind::TransportBelt, 2, 0))
            .unwrap();
        assert!(g.successors[ins1].contains(&belt));

        // Belt(2,0) → Inserter(3,0)
        let ins2 = g
            .get_index(&NodeId::new(EntityKind::Inserter, 3, 0))
            .unwrap();
        assert!(g.predecessors[ins2].contains(&belt));
    }

    #[test]
    fn test_underground_belt_graph() {
        // Belt → Underground(down) ... Underground(up) → Belt
        let mut w = World::empty(6, 1);
        w.set(0, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);

        w.set(1, 0, Channel::Entities, EntityKind::UndergroundBelt as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);
        w.set(1, 0, Channel::Misc, Misc::UndergroundDown as i64);

        w.set(4, 0, Channel::Entities, EntityKind::UndergroundBelt as i64);
        w.set(4, 0, Channel::Direction, Direction::East as i64);
        w.set(4, 0, Channel::Misc, Misc::UndergroundUp as i64);

        w.set(5, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(5, 0, Channel::Direction, Direction::East as i64);

        let g = build_graph(&w);
        assert_eq!(g.node_count(), 4);

        // Underground(down,1,0) → Underground(up,4,0)
        let ug_down = g
            .get_index(&NodeId::new(EntityKind::UndergroundBelt, 1, 0))
            .unwrap();
        let ug_up = g
            .get_index(&NodeId::new(EntityKind::UndergroundBelt, 4, 0))
            .unwrap();
        assert!(g.successors[ug_down].contains(&ug_up));

        // Belt(0,0) should NOT connect to underground_down (belt doesn't connect to
        // underground_down because its src_is_beltish check excludes underground_down)
        // Actually wait: Belt at (0,0) facing east, underground at (1,0) is ahead.
        // The belt's connections check dst: is it a belt? UndergroundBelt is belt-ish → dst_is_belt = true.
        // Not opposing direction → should connect.
        let belt0 = g
            .get_index(&NodeId::new(EntityKind::TransportBelt, 0, 0))
            .unwrap();
        assert!(g.successors[belt0].contains(&ug_down));
    }
}
