use std::collections::HashMap;

use crate::entities::{entity_tiles, EntityEnum, FactoryEntity};
use crate::types::{Item, Misc, NodeId, PortRole};
use crate::world::World;

/// A node in the factory graph.
///
/// One node corresponds to one logical I/O port of an entity (see
/// `NodeId` / `PortRole`). Lane-aware entities (TB, UG, Splitter)
/// contribute two nodes per anchor — one port-side and one starboard-
/// side. Lane-agnostic entities contribute a single node.
///
/// Each node holds its OWN flow accumulator — `input` and `output` are
/// flat `HashMap<Item, f64>` because the lane axis is encoded in the
/// node's identity rather than in per-edge tags.
#[derive(Debug, Clone)]
pub struct GraphNode {
    #[allow(dead_code)]
    pub id: NodeId,
    pub entity_kind: Item,
    pub item: Option<Item>,
    pub misc: Misc,
    /// Recipe for assembling machines (determines what they craft).
    /// `None` for non-assemblers and assemblers with no recipe set.
    pub recipe_item: Option<Item>,
    pub input: HashMap<Item, f64>,
    pub output: HashMap<Item, f64>,
}

/// A directed graph representing a factory's entity connections.
///
/// Edges are plain `(src_index, dst_index)` — there are no per-edge
/// lane tags. The lane that an edge represents is implicit in the
/// node identities at either end.
#[derive(Debug)]
pub struct FactoryGraph {
    /// Nodes indexed by their position in this vec.
    pub nodes: Vec<GraphNode>,
    /// Map from NodeId to index in `nodes`.
    #[allow(dead_code)]
    pub node_index: HashMap<NodeId, usize>,
    /// Adjacency list: successors[i] = node indices that node i has edges TO.
    pub successors: Vec<Vec<usize>>,
    /// Reverse adjacency: predecessors[i] = node indices with edges TO node i.
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

/// Lane-aware entity kinds: each contributes TWO graph nodes per anchor
/// tile (one port, one starboard).
fn is_lane_aware(kind: Item) -> bool {
    matches!(
        kind,
        Item::TransportBelt | Item::UndergroundBelt | Item::Splitter
    )
}

/// Port roles to instantiate as graph nodes for an entity. Lane-aware
/// entities expand into two nodes; lane-agnostic into one.
fn port_roles_for(kind: Item) -> &'static [PortRole] {
    if is_lane_aware(kind) {
        &[PortRole::Port, PortRole::Starboard]
    } else {
        &[PortRole::Single]
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

    // First pass: create one or two nodes per entity (anchor tile only
    // for multi-tile entities). Lane-aware entities get a port and
    // starboard node; lane-agnostic entities get a single node.
    for x in 0..world.width() {
        for y in 0..world.height() {
            // Empty cell, or a stray non-placeable item in the entities
            // channel (data error). Either way, no graph node.
            let entity_kind = match world.entity_at(x, y) {
                Some(k) if k.is_placeable() => k,
                _ => continue,
            };

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

            // Allocate one or two nodes for this entity.
            for &port in port_roles_for(entity_kind) {
                let node_id = NodeId {
                    entity_kind,
                    x,
                    y,
                    port,
                };

                // Source pre-populates its output with infinite flow on
                // ITS configured item. Lane-aware sources don't exist —
                // Source is always single. Each port-node starts empty
                // otherwise.
                let output = if entity_kind == Item::Source {
                    let mut m = HashMap::new();
                    if let Some(i) = item {
                        m.insert(i, f64::INFINITY);
                    }
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
                    recipe_item: if entity_kind == Item::AssemblingMachine1 {
                        item
                    } else {
                        None
                    },
                    input: HashMap::new(),
                    output,
                });
                node_index.insert(node_id, idx);
            }

            // entity_kind is guaranteed placeable by the loop guard, so
            // EntityEnum::new always returns Some here. We still match
            // defensively rather than unwrap.
            if let Some(entity) = EntityEnum::new(entity_kind, item, misc) {
                let edges = entity.connections((x, y), direction, world);
                edge_list.extend(edges);
            }
        }
    }

    // Build adjacency lists, remapping any edge endpoint that targets a
    // secondary tile of a multi-tile entity to the anchor instead, while
    // preserving the port role.
    let n = nodes.len();
    let mut successors: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut predecessors: Vec<Vec<usize>> = vec![Vec::new(); n];

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

/// If (id.x, id.y) is a secondary tile of a multi-tile entity, return a
/// new NodeId pointing to the anchor (preserving entity_kind and port).
/// Otherwise return the original.
fn remap_to_anchor(id: &NodeId, anchor_of: &HashMap<(usize, usize), (usize, usize)>) -> NodeId {
    if let Some(&(ax, ay)) = anchor_of.get(&(id.x, id.y)) {
        NodeId {
            entity_kind: id.entity_kind,
            x: ax,
            y: ay,
            port: id.port,
        }
    } else {
        id.clone()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::types::{Direction, Misc};

    /// Helper: assert there's at least one edge from src→dst in `g`.
    fn has_edge_to(g: &FactoryGraph, src: usize, dst: usize) -> bool {
        g.successors[src].contains(&dst)
    }

    #[test]
    fn test_empty_world_graph() {
        let w = World::empty(5, 5);
        let g = build_graph(&w);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_single_belt_creates_two_nodes() {
        // Lane-aware entities allocate one port-node + one starboard-node.
        let mut w = World::empty(3, 3);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);

        let g = build_graph(&w);
        assert_eq!(g.node_count(), 2);
        let p = g.get_index(&NodeId::port(Item::TransportBelt, 1, 1));
        let s = g.get_index(&NodeId::starboard(Item::TransportBelt, 1, 1));
        assert!(p.is_some());
        assert!(s.is_some());
        assert_ne!(p, s);
        // Belt has no neighbours → no edges from either node.
        for idx in [p.unwrap(), s.unwrap()] {
            assert!(g.successors[idx].is_empty());
            assert!(g.predecessors[idx].is_empty());
        }
    }

    #[test]
    fn test_single_inserter_creates_one_node() {
        // Lane-agnostic entities allocate a single node.
        let mut w = World::empty(3, 3);
        w.place(1, 1, Item::Inserter, Direction::East, None);
        let g = build_graph(&w);
        assert_eq!(g.node_count(), 1);
        assert!(g.get_index(&NodeId::single(Item::Inserter, 1, 1)).is_some());
        assert!(g.get_index(&NodeId::port(Item::Inserter, 1, 1)).is_none());
    }

    #[test]
    fn test_belt_chain_lane_preserving() {
        // Source → Belt → Belt → Sink. Belts emit two lane-preserving
        // edges per parallel forward connection.
        let mut w = World::empty(5, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        // 1 source + 2×2 belt + 1 sink = 6 nodes.
        assert_eq!(g.node_count(), 6);

        let belt1_p = g
            .get_index(&NodeId::port(Item::TransportBelt, 1, 0))
            .unwrap();
        let belt1_s = g
            .get_index(&NodeId::starboard(Item::TransportBelt, 1, 0))
            .unwrap();
        let belt2_p = g
            .get_index(&NodeId::port(Item::TransportBelt, 2, 0))
            .unwrap();
        let belt2_s = g
            .get_index(&NodeId::starboard(Item::TransportBelt, 2, 0))
            .unwrap();

        // Lane-preserving belt-to-belt: port→port, stbd→stbd.
        assert!(has_edge_to(&g, belt1_p, belt2_p));
        assert!(has_edge_to(&g, belt1_s, belt2_s));
        // No cross-lane wiring for parallel belts.
        assert!(!has_edge_to(&g, belt1_p, belt2_s));
        assert!(!has_edge_to(&g, belt1_s, belt2_p));

        // Source feeds both lanes of belt1.
        let source = g.get_index(&NodeId::single(Item::Source, 0, 0)).unwrap();
        assert!(has_edge_to(&g, source, belt1_p));
        assert!(has_edge_to(&g, source, belt1_s));

        // Sink drains both lanes of belt2.
        let sink = g.get_index(&NodeId::single(Item::Sink, 3, 0)).unwrap();
        assert!(has_edge_to(&g, belt2_p, sink));
        assert!(has_edge_to(&g, belt2_s, sink));
    }

    #[test]
    fn test_underground_belt_lane_preserving() {
        // Belt → UG-down ... UG-up → Belt — lanes preserved through tunnel.
        let mut w = World::empty(6, 1);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place_underground(1, 0, Direction::East, Misc::UndergroundDown);
        w.place_underground(4, 0, Direction::East, Misc::UndergroundUp);
        w.place(5, 0, Item::TransportBelt, Direction::East, None);

        let g = build_graph(&w);
        assert_eq!(g.node_count(), 8); // 4 entities × 2 lanes

        let down_p = g
            .get_index(&NodeId::port(Item::UndergroundBelt, 1, 0))
            .unwrap();
        let down_s = g
            .get_index(&NodeId::starboard(Item::UndergroundBelt, 1, 0))
            .unwrap();
        let up_p = g
            .get_index(&NodeId::port(Item::UndergroundBelt, 4, 0))
            .unwrap();
        let up_s = g
            .get_index(&NodeId::starboard(Item::UndergroundBelt, 4, 0))
            .unwrap();
        assert!(has_edge_to(&g, down_p, up_p));
        assert!(has_edge_to(&g, down_s, up_s));
        assert!(!has_edge_to(&g, down_p, up_s));
    }
}
