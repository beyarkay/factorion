use std::collections::HashMap;

use crate::entities::{entity_tiles, EntityEnum, FactoryEntity};
use crate::types::{EntityKind, Item, Misc, NodeId};
use crate::world::World;

/// A node in the factory graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    #[allow(dead_code)]
    pub id: NodeId,
    pub entity_kind: EntityKind,
    pub item: Item,
    pub misc: Misc,
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
    #[allow(dead_code)]
    pub fn get_node(&self, idx: usize) -> &GraphNode {
        &self.nodes[idx]
    }

    #[cfg(test)]
    #[allow(dead_code)]
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

    // For multi-tile entities: maps secondary tile (x,y) → anchor (x,y).
    // Used to (a) skip secondary tiles during node creation, and
    // (b) remap edge endpoints so all edges point to the anchor node.
    let mut anchor_of: HashMap<(usize, usize), (usize, usize)> = HashMap::new();

    // First pass: create one node per entity (anchor tile only for multi-tile)
    for x in 0..world.width() {
        for y in 0..world.height() {
            let entity_kind = world.entity_at(x, y);
            if entity_kind == EntityKind::Empty {
                continue;
            }

            if anchor_of.contains_key(&(x, y)) {
                continue;
            }

            let item = world.item_at(x, y);
            let direction = world.direction_at(x, y);
            let misc = world.misc_at(x, y);

            // For multi-tile entities, register secondary tiles → anchor.
            // Square entities (e.g. 3x3 assembler) use East as default
            // direction for tile computation since their footprint is
            // rotation-independent.
            let (ew, eh) = entity_kind.size();
            if ew > 1 || eh > 1 {
                if let Some(tiles) = entity_tiles(x, y, direction, ew, eh) {
                    for tile in &tiles[1..] {
                        if let Some((tx, ty)) = tile.to_usize() {
                            anchor_of.insert((tx, ty), (x, y));
                        }
                    }
                }
            }

            let node_id = NodeId::new(entity_kind, x, y);

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
                misc,
                recipe_item: if entity_kind == EntityKind::AssemblingMachine1 {
                    item
                } else {
                    Item::Empty
                },
                input: HashMap::new(),
                output,
            });
            node_index.insert(node_id, idx);

            let entity = EntityEnum::new(entity_kind, item, misc);
            let edges = entity.connections((x, y), direction, world);
            edge_list.extend(edges);
        }
    }

    // Build adjacency lists, remapping any edge endpoint that targets a
    // secondary tile of a multi-tile entity to the anchor node instead.
    let n = nodes.len();
    let mut successors = vec![Vec::new(); n];
    let mut predecessors = vec![Vec::new(); n];

    for (src_id, dst_id) in &edge_list {
        let src_remapped = remap_to_anchor(src_id, &anchor_of);
        let dst_remapped = remap_to_anchor(dst_id, &anchor_of);
        if let (Some(&src_idx), Some(&dst_idx)) =
            (node_index.get(&src_remapped), node_index.get(&dst_remapped))
        {
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

/// If (id.x, id.y) is a secondary tile of a multi-tile entity, return a new
/// NodeId pointing to the anchor. Otherwise return the original.
fn remap_to_anchor(id: &NodeId, anchor_of: &HashMap<(usize, usize), (usize, usize)>) -> NodeId {
    if let Some(&(ax, ay)) = anchor_of.get(&(id.x, id.y)) {
        NodeId::new(id.entity_kind, ax, ay)
    } else {
        id.clone()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::types::{Direction, Misc};

    #[test]
    fn test_empty_world_graph() {
        let w = World::empty(5, 5);
        let g = build_graph(&w);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_single_belt_no_edges() {
        let mut w = World::empty(3, 3);
        w.place(
            1,
            1,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );

        let g = build_graph(&w);
        assert_eq!(g.node_count(), 1);
        assert!(g.successors[0].is_empty());
        assert!(g.predecessors[0].is_empty());
    }

    #[test]
    fn test_belt_chain_graph() {
        // Source → Belt → Belt → Sink
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
        w.place(0, 0, EntityKind::Source, Direction::East, Item::CopperCable);
        w.place(1, 0, EntityKind::Inserter, Direction::East, Item::Empty);
        w.place(
            2,
            0,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );
        w.place(3, 0, EntityKind::Inserter, Direction::East, Item::Empty);
        w.place(4, 0, EntityKind::Sink, Direction::East, Item::CopperCable);

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
        w.place(
            0,
            0,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );
        w.place_underground(1, 0, Direction::East, Misc::UndergroundDown);
        w.place_underground(4, 0, Direction::East, Misc::UndergroundUp);
        w.place(
            5,
            0,
            EntityKind::TransportBelt,
            Direction::East,
            Item::Empty,
        );

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
