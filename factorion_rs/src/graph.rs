use std::collections::HashMap;

use crate::entities::{entity_tiles, is_curved_belt, EntityEnum, FactoryEntity};
use crate::types::{Direction, Item, Lane, Misc, NodeId};
use crate::world::World;
use strum::IntoEnumIterator;

/// A node in the factory graph.
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub entity_kind: Item,
    /// The item carried/produced/configured at this tile, if any. `None`
    /// means the items channel was 0 (no item set).
    pub item: Option<Item>,
    pub misc: Misc,
    /// Recipe for assembling machines (determines what they craft).
    /// `None` for non-assemblers and assemblers with no recipe set.
    pub recipe_item: Option<Item>,
    /// Anchor tile of the entity unit this node belongs to. Nodes are no
    /// longer 1:1 with entities (a belt tile owns two lane nodes, a splitter
    /// four across its two tiles); grouping by anchor recovers the entity —
    /// e.g. unreachability is counted per entity, not per lane node.
    pub anchor: (usize, usize),
    /// The tile's facing. The solver needs it to resolve inserter pickup
    /// lane priority (near lane when the belt runs perpendicular to the
    /// inserter, left lane otherwise).
    pub direction: Direction,
    /// Whether this tile is a CURVED transport belt (lone side feed).
    /// Inserters picking from a curve prefer its LEFT lane regardless of
    /// relative orientation.
    pub curved: bool,
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

/// Build a directed graph from a World, with edges per the engine's
/// entity-connection rules. This is the single source of truth for factory
/// graph construction.
///
/// Nodes are not 1:1 with entities: lane-aware entities (belts, underground
/// belts, and EACH tile of a splitter) get one node per lane; everything
/// else gets a single node at its anchor tile (multi-tile assemblers still
/// collapse their secondary tiles onto the anchor).
pub fn build_graph(world: &World) -> FactoryGraph {
    let mut nodes: Vec<GraphNode> = Vec::new();
    let mut node_index: HashMap<NodeId, usize> = HashMap::new();
    let mut edge_list: Vec<(NodeId, NodeId)> = Vec::new();

    // For lane-less multi-tile entities (assembler): maps secondary tile
    // (x,y) → anchor (x,y). Used to (a) skip secondary tiles during node
    // creation, and (b) remap edge endpoints so all edges point to the
    // anchor node.
    let mut anchor_of: HashMap<(usize, usize), (usize, usize)> = HashMap::new();

    // For lane-aware multi-tile entities (splitter): secondary tile → anchor.
    // These tiles DO get their own lane nodes (a splitter is two belts side
    // by side); the map only records unit membership and suppresses a second
    // `connections()` call — the anchor visit emits edges for both tiles.
    let mut lane_tile_anchor: HashMap<(usize, usize), (usize, usize)> = HashMap::new();

    // First pass: create nodes and collect edges. The scan order (x, then y,
    // ascending) visits every multi-tile anchor before its secondary tiles,
    // so the anchor registers them ahead of their own visit.
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
            let secondary_lane_tile = lane_tile_anchor.contains_key(&(x, y));

            let item = world.item_at(x, y);
            let direction = world.direction_at(x, y);
            let misc = world.misc_at(x, y);

            // For multi-tile entities, register secondary tiles → anchor.
            // Square entities (e.g. 3x3 assembler) use East as default
            // direction for tile computation since their footprint is
            // rotation-independent.
            let (ew, eh) = entity_kind.size();
            if !secondary_lane_tile && (ew > 1 || eh > 1) {
                if let Some(tiles) = entity_tiles(x, y, direction, ew, eh) {
                    for tile in &tiles[1..] {
                        if let Some((tx, ty)) = tile.to_usize() {
                            if entity_kind.is_lane_aware() {
                                lane_tile_anchor.insert((tx, ty), (x, y));
                            } else {
                                anchor_of.insert((tx, ty), (x, y));
                            }
                        }
                    }
                }
            }

            let anchor = lane_tile_anchor.get(&(x, y)).copied().unwrap_or((x, y));
            let curved = entity_kind == Item::TransportBelt && is_curved_belt(world, (x, y));

            let node_ids: Vec<NodeId> = if entity_kind.is_lane_aware() {
                Lane::iter()
                    .map(|lane| NodeId::new(entity_kind, x, y, Some(lane)))
                    .collect()
            } else {
                vec![NodeId::new(entity_kind, x, y, None)]
            };
            for node_id in node_ids {
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
                    anchor,
                    direction,
                    curved,
                    input: HashMap::new(),
                    output,
                });
                node_index.insert(node_id, idx);
            }

            // entity_kind is guaranteed placeable by the loop guard, so
            // EntityEnum::new always returns Some here. We still match
            // defensively rather than unwrap. Secondary lane tiles skip
            // this: their unit's anchor already emitted edges for them.
            if !secondary_lane_tile {
                if let Some(entity) = EntityEnum::new(entity_kind, item, misc) {
                    let edges = entity.connections((x, y), direction, world);
                    edge_list.extend(edges);
                }
            }
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

/// If (id.x, id.y) is a secondary tile of a lane-less multi-tile entity
/// (assembler), return a new NodeId pointing to the anchor. Otherwise return
/// the original. Splitter tiles are NOT in this map — each keeps its own
/// per-tile lane nodes.
fn remap_to_anchor(id: &NodeId, anchor_of: &HashMap<(usize, usize), (usize, usize)>) -> NodeId {
    if let Some(&(ax, ay)) = anchor_of.get(&(id.x, id.y)) {
        NodeId {
            entity_kind: id.entity_kind,
            x: ax,
            y: ay,
            lane: id.lane,
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

    /// 1x1 entity configs exercised by the exhaustive connectivity checks.
    /// (Multi-tile assembler/splitter handled separately.)
    struct ConnCfg {
        label: &'static str,
        item: Item,
        misc: Misc,
    }

    const CONN_CFGS: &[ConnCfg] = &[
        ConnCfg {
            label: "belt",
            item: Item::TransportBelt,
            misc: Misc::None,
        },
        ConnCfg {
            label: "inserter",
            item: Item::Inserter,
            misc: Misc::None,
        },
        ConnCfg {
            label: "ug_up",
            item: Item::UndergroundBelt,
            misc: Misc::UndergroundUp,
        },
        ConnCfg {
            label: "ug_down",
            item: Item::UndergroundBelt,
            misc: Misc::UndergroundDown,
        },
        ConnCfg {
            label: "source",
            item: Item::Source,
            misc: Misc::None,
        },
        ConnCfg {
            label: "sink",
            item: Item::Sink,
            misc: Misc::None,
        },
    ];

    const CONN_DIRS: &[(Direction, &str)] = &[
        (Direction::North, "N"),
        (Direction::East, "E"),
        (Direction::South, "S"),
        (Direction::West, "W"),
    ];

    // The four orthogonal adjacencies: where B sits relative to A.
    const CONN_DELTAS: &[(i64, i64)] = &[(-1, 0), (0, -1), (1, 0), (0, 1)];

    fn place_conn_cfg(w: &mut World, x: usize, y: usize, c: &ConnCfg, dir: Direction) {
        if c.item == Item::UndergroundBelt {
            w.place_underground(x, y, dir, c.misc);
        } else {
            w.place(x, y, c.item, dir, None);
        }
    }

    /// What the engine connects between two adjacent entities.
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum Conn {
        None,
        AToB,
        BToA,
        Both,
    }

    /// Build a 2-entity world (A at centre, B at centre+delta) and return what
    /// the engine connects via directed graph edges.
    fn conn_between(
        a: &ConnCfg,
        a_dir: Direction,
        b: &ConnCfg,
        b_dir: Direction,
        delta: (i64, i64),
    ) -> Conn {
        let (cx, cy) = (3usize, 3usize);
        let bx = (cx as i64 + delta.0) as usize;
        let by = (cy as i64 + delta.1) as usize;
        let mut w = World::empty(7, 7);
        place_conn_cfg(&mut w, cx, cy, a, a_dir);
        place_conn_cfg(&mut w, bx, by, b, b_dir);
        let g = build_graph(&w);
        // Connectivity is judged at the ENTITY level: any edge between any of
        // A's nodes and any of B's (belt-ish entities own one node per lane).
        let nodes_at = |x: usize, y: usize| -> Vec<usize> {
            (0..g.node_count())
                .filter(|&i| g.nodes[i].id.x == x && g.nodes[i].id.y == y)
                .collect()
        };
        let ia = nodes_at(cx, cy);
        let ib = nodes_at(bx, by);
        let any_edge = |from: &[usize], to: &[usize]| {
            from.iter()
                .any(|&i| g.successors[i].iter().any(|j| to.contains(j)))
        };
        let a2b = any_edge(&ia, &ib);
        let b2a = any_edge(&ib, &ia);
        match (a2b, b2a) {
            (false, false) => Conn::None,
            (true, false) => Conn::AToB,
            (false, true) => Conn::BToA,
            (true, true) => Conn::Both,
        }
    }

    /// Rotate a facing 90° clockwise: the vector (dx,dy) -> (-dy,dx).
    /// North(0,-1)->East(1,0)->South(0,1)->West(-1,0)->North.
    fn rot90_dir(d: Direction) -> Direction {
        match d {
            Direction::North => Direction::East,
            Direction::East => Direction::South,
            Direction::South => Direction::West,
            Direction::West => Direction::North,
            Direction::None => Direction::None,
        }
    }

    /// Rotate an adjacency delta 90° clockwise the same way: (dx,dy)->(-dy,dx).
    fn rot90_delta(d: (i64, i64)) -> (i64, i64) {
        (-d.1, d.0)
    }

    /// Rotating a whole 2-entity configuration by 90/180/270° must not change
    /// what connects to what. A sits at the centre of a square world, so a
    /// world rotation is exactly: rotate both facings and the adjacency delta
    /// together. Proving this lets the full connectivity table drop the
    /// rotation axis (4× fewer cases) without losing coverage.
    #[test]
    fn connectivity_is_rotation_invariant() {
        for a in CONN_CFGS {
            for &(a_dir, _) in CONN_DIRS {
                for b in CONN_CFGS {
                    for &(b_dir, _) in CONN_DIRS {
                        for &delta in CONN_DELTAS {
                            let base = conn_between(a, a_dir, b, b_dir, delta);
                            let (mut ad, mut bd, mut d) = (a_dir, b_dir, delta);
                            for turn in 1..=3 {
                                ad = rot90_dir(ad);
                                bd = rot90_dir(bd);
                                d = rot90_delta(d);
                                let rotated = conn_between(a, ad, b, bd, d);
                                assert_eq!(
                                    base, rotated,
                                    "rotating {}/{:?} + {}/{:?} (B@{:?}) by {}*90deg \
                                     changed connectivity ({:?} -> {:?})",
                                    a.label, a_dir, b.label, b_dir, delta, turn, base, rotated
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_empty_world_graph() {
        let w = World::empty(5, 5);
        let g = build_graph(&w);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_single_belt_no_edges() {
        let mut w = World::empty(3, 3);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);

        let g = build_graph(&w);
        // One node per lane.
        assert_eq!(g.node_count(), 2);
        assert!(g.successors.iter().all(|s| s.is_empty()));
        assert!(g.predecessors.iter().all(|p| p.is_empty()));
    }

    #[test]
    fn test_belt_chain_graph() {
        // Source → Belt → Belt → Sink
        let mut w = World::empty(5, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        // Source(1) + 2 belts (2 lane nodes each) + Sink(1).
        assert_eq!(g.node_count(), 6);

        let source = g.get_index(&NodeId::new(Item::Source, 0, 0, None)).unwrap();
        let sink = g.get_index(&NodeId::new(Item::Sink, 3, 0, None)).unwrap();
        for lane in Lane::iter() {
            // Belt(1,0) → Belt(2,0) is lane-preserving.
            let belt1 = g
                .get_index(&NodeId::new(Item::TransportBelt, 1, 0, Some(lane)))
                .unwrap();
            let belt2 = g
                .get_index(&NodeId::new(Item::TransportBelt, 2, 0, Some(lane)))
                .unwrap();
            assert!(g.successors[belt1].contains(&belt2));
            assert!(g.predecessors[belt2].contains(&belt1));

            // The source fills both lanes of the belt at (1,0)…
            assert!(g.successors[source].contains(&belt1));
            // …and the sink drains both lanes of the belt at (2,0).
            assert!(g.predecessors[sink].contains(&belt2));
        }
    }

    #[test]
    fn test_inserter_chain_graph() {
        // Source → Inserter → Belt → Inserter → Sink
        let mut w = World::empty(5, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::Inserter, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Inserter, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        // Source + 2 inserters + sink (single) + belt (2 lane nodes).
        assert_eq!(g.node_count(), 6);

        // Source → Inserter(1,0)
        let source = g.get_index(&NodeId::new(Item::Source, 0, 0, None)).unwrap();
        let ins1 = g
            .get_index(&NodeId::new(Item::Inserter, 1, 0, None))
            .unwrap();
        assert!(g.successors[source].contains(&ins1) || g.predecessors[ins1].contains(&source));

        let ins2 = g
            .get_index(&NodeId::new(Item::Inserter, 3, 0, None))
            .unwrap();
        // Inserter(1,0) → Belt(2,0): parallel belt → drops on the RIGHT
        // lane only.
        let belt_r = g
            .get_index(&NodeId::new(
                Item::TransportBelt,
                2,
                0,
                Some(crate::types::Lane::Right),
            ))
            .unwrap();
        assert!(g.successors[ins1].contains(&belt_r));
        for lane in Lane::iter() {
            // Belt(2,0) → Inserter(3,0): picks from both lanes.
            let belt = g
                .get_index(&NodeId::new(Item::TransportBelt, 2, 0, Some(lane)))
                .unwrap();
            assert!(g.predecessors[ins2].contains(&belt));
        }
    }

    #[test]
    fn test_underground_belt_graph() {
        // Belt → Underground(down) ... Underground(up) → Belt
        let mut w = World::empty(6, 1);
        w.place(0, 0, Item::TransportBelt, Direction::East, None);
        w.place_underground(1, 0, Direction::East, Misc::UndergroundDown);
        w.place_underground(4, 0, Direction::East, Misc::UndergroundUp);
        w.place(5, 0, Item::TransportBelt, Direction::East, None);

        let g = build_graph(&w);
        // 4 belt-ish entities × 2 lane nodes.
        assert_eq!(g.node_count(), 8);

        for lane in Lane::iter() {
            // Underground(down,1,0) → Underground(up,4,0): lanes persist
            // through the tunnel.
            let ug_down = g
                .get_index(&NodeId::new(Item::UndergroundBelt, 1, 0, Some(lane)))
                .unwrap();
            let ug_up = g
                .get_index(&NodeId::new(Item::UndergroundBelt, 4, 0, Some(lane)))
                .unwrap();
            assert!(g.successors[ug_down].contains(&ug_up));

            let belt0 = g
                .get_index(&NodeId::new(Item::TransportBelt, 0, 0, Some(lane)))
                .unwrap();
            assert!(g.successors[belt0].contains(&ug_down));
        }
    }

    #[test]
    fn test_splitter_four_lane_nodes() {
        // A splitter is two belts side by side, each with two lanes: four
        // nodes, all sharing the anchor for entity-unit grouping.
        let mut w = World::empty(4, 3);
        w.place_splitter(1, 0, Direction::East, None);

        let g = build_graph(&w);
        assert_eq!(g.node_count(), 4);
        for tile_y in [0, 1] {
            for lane in Lane::iter() {
                let idx = g
                    .get_index(&NodeId::new(Item::Splitter, 1, tile_y, Some(lane)))
                    .unwrap();
                assert_eq!(g.nodes[idx].anchor, (1, 0));
            }
        }
    }
}
