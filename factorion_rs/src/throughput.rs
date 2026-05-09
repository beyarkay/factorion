use std::collections::{HashMap, HashSet, VecDeque};

use crate::entities::{EntityEnum, FactoryEntity};
use crate::graph::FactoryGraph;
use crate::types::{Item, LANE_FLOW_RATE};

/// Calculate the throughput of a factory graph.
///
/// Returns `(output_per_item, num_unreachable)` matching the Python
/// `calc_throughput`. The flow is computed per port-node — lane info is
/// implicit in node identities, so the solver itself is simpler than
/// the per-edge-tagged version it replaced.
pub fn calc_throughput(graph: &FactoryGraph) -> (HashMap<Item, f64>, usize) {
    if graph.node_count() == 0 {
        return (HashMap::new(), 0);
    }

    if has_cycle(graph) {
        return (HashMap::new(), 0);
    }

    let sources: Vec<usize> = (0..graph.node_count())
        .filter(|&i| graph.nodes[i].entity_kind == Item::Source)
        .collect();
    let sinks: Vec<usize> = (0..graph.node_count())
        .filter(|&i| graph.nodes[i].entity_kind == Item::Sink)
        .collect();

    if sources.is_empty() || sinks.is_empty() {
        return (HashMap::new(), count_entities(graph));
    }

    let reachable_from_sources = reachable_from(&sources, graph, false);

    // Kahn's algorithm: in-degree counts reachable predecessors.
    let mut in_degree: Vec<usize> = graph
        .predecessors
        .iter()
        .map(|preds| {
            preds
                .iter()
                .filter(|&&p| reachable_from_sources.contains(&p))
                .count()
        })
        .collect();

    let mut queue: VecDeque<usize> = VecDeque::new();
    let mut in_queue: HashSet<usize> = HashSet::new();

    for &s in &sources {
        queue.push_back(s);
        in_queue.insert(s);
    }

    let mut node_inputs: Vec<HashMap<Item, f64>> =
        graph.nodes.iter().map(|n| n.input.clone()).collect();
    let mut node_outputs: Vec<HashMap<Item, f64>> =
        graph.nodes.iter().map(|n| n.output.clone()).collect();

    let mut already_processed: HashSet<usize> = HashSet::new();

    while let Some(node_idx) = queue.pop_front() {
        in_queue.remove(&node_idx);

        if already_processed.contains(&node_idx) {
            continue;
        }

        let entity_kind = graph.nodes[node_idx].entity_kind;

        if node_outputs[node_idx].is_empty() {
            // Accumulate input from all predecessors' outputs.
            let mut accumulated_input: HashMap<Item, f64> = HashMap::new();
            for &pred in &graph.predecessors[node_idx] {
                for (&item, &flow_rate) in &node_outputs[pred] {
                    *accumulated_input.entry(item).or_insert(0.0) += flow_rate;
                }
            }
            node_inputs[node_idx] = accumulated_input.clone();

            let item = if entity_kind == Item::AssemblingMachine1 {
                graph.nodes[node_idx].recipe_item
            } else {
                graph.nodes[node_idx].item
            };
            let misc = graph.nodes[node_idx].misc;
            if let Some(entity) = EntityEnum::new(entity_kind, item, misc) {
                node_outputs[node_idx] = entity.transform_flow(&accumulated_input);
            }

            // Splitter post-processing: divide output evenly across
            // successors (each output port-node gets an equal share),
            // then cap each output at LANE_FLOW_RATE — the per-belt-lane
            // bound. With node-per-port the divisor sees only same-lane
            // successors of THIS port-node, so no per-lane filter is
            // needed.
            if entity_kind == Item::Splitter {
                let num_successors = graph.successors[node_idx].len();
                if num_successors > 1 {
                    for rate in node_outputs[node_idx].values_mut() {
                        *rate /= num_successors as f64;
                    }
                }
                for rate in node_outputs[node_idx].values_mut() {
                    *rate = rate.min(LANE_FLOW_RATE);
                }
            }
        }

        already_processed.insert(node_idx);

        for &succ in &graph.successors[node_idx] {
            if already_processed.contains(&succ) {
                continue;
            }
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 && !in_queue.contains(&succ) {
                queue.push_back(succ);
                in_queue.insert(succ);
            }
        }
    }

    // Sum sink outputs across all sink nodes (including both lane-port
    // nodes if the sink is fed by a multi-lane upstream — sinks have a
    // single node and aggregate naturally).
    let mut total_output: HashMap<Item, f64> = HashMap::new();
    for &sink_idx in &sinks {
        for (&item, &rate) in &node_outputs[sink_idx] {
            *total_output.entry(item).or_insert(0.0) += rate;
        }
    }

    let can_reach_sink = reachable_from(&sinks, graph, true);
    let reachable_from_source = reachable_from(&sources, graph, false);
    let on_path: HashSet<usize> = can_reach_sink
        .intersection(&reachable_from_source)
        .copied()
        .collect();

    // Unreachable counts ENTITIES (one per anchor tile), not port-nodes.
    // A lane-aware entity contributes two port-nodes; if even one of
    // them is on the source→sink path, the entity is "doing something"
    // and should not count as unreachable. This matches the per-entity
    // reachability count Python's calc_throughput returns.
    let mut entity_to_nodes: HashMap<(Item, usize, usize), Vec<usize>> = HashMap::new();
    for (i, node) in graph.nodes.iter().enumerate() {
        let key = (node.entity_kind, node.id.x, node.id.y);
        entity_to_nodes.entry(key).or_default().push(i);
    }
    let unreachable = entity_to_nodes
        .values()
        .filter(|nodes| nodes.iter().all(|&n| !on_path.contains(&n)))
        .count();

    (total_output, unreachable)
}

/// Count distinct entities (anchor tiles) in the graph. Lane-aware
/// entities contribute one count regardless of how many port-nodes
/// they span — matches the per-entity reachability count Python's
/// `calc_throughput` returns.
fn count_entities(graph: &FactoryGraph) -> usize {
    let mut entities: HashSet<(Item, usize, usize)> = HashSet::new();
    for node in &graph.nodes {
        entities.insert((node.entity_kind, node.id.x, node.id.y));
    }
    entities.len()
}

/// Check if the graph has any cycles using DFS.
fn has_cycle(graph: &FactoryGraph) -> bool {
    let n = graph.node_count();
    let mut visited = vec![0u8; n];

    for start in 0..n {
        if visited[start] != 0 {
            continue;
        }
        let mut stack = vec![(start, 0usize)];
        visited[start] = 1;

        while let Some((node, succ_idx)) = stack.last_mut() {
            if *succ_idx < graph.successors[*node].len() {
                let next = graph.successors[*node][*succ_idx];
                *succ_idx += 1;
                if visited[next] == 1 {
                    return true;
                }
                if visited[next] == 0 {
                    visited[next] = 1;
                    stack.push((next, 0));
                }
            } else {
                visited[*node] = 2;
                stack.pop();
            }
        }
    }
    false
}

fn reachable_from(starts: &[usize], graph: &FactoryGraph, reverse: bool) -> HashSet<usize> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    for &s in starts {
        visited.insert(s);
        queue.push_back(s);
    }
    while let Some(node) = queue.pop_front() {
        let neighbors = if reverse {
            &graph.predecessors[node]
        } else {
            &graph.successors[node]
        };
        for &next in neighbors {
            if visited.insert(next) {
                queue.push_back(next);
            }
        }
    }
    visited
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::graph::{build_graph, GraphNode};
    use crate::types::{Direction, Misc, NodeId, PortRole};
    use crate::world::World;

    #[test]
    fn test_empty_world_throughput() {
        let w = World::empty(5, 5);
        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        assert!(output.is_empty());
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_source_belt_belt_sink() {
        // Source feeds both belt lanes; each lane caps at 7.5; sink aggregates 15.
        let mut w = World::empty(4, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        let throughput = output[&Item::CopperCable];
        assert!(
            (throughput - 15.0).abs() < 1e-9,
            "Expected 15.0, got {}",
            throughput
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_source_inserter_belt_sink() {
        let mut w = World::empty(4, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::Inserter, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        let throughput = output[&Item::CopperCable];
        assert!(
            (throughput - 0.86).abs() < 1e-9,
            "Expected 0.86, got {}",
            throughput
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_disconnected_entities() {
        let mut w = World::empty(5, 5);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(4, 4, Item::Sink, Direction::East, Some(Item::CopperCable));
        w.place(2, 2, Item::TransportBelt, Direction::East, None);

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        assert!(output.is_empty() || output.values().all(|&v| v == 0.0));
        // Three ENTITIES, all disconnected → unreachable = 3 (the belt's
        // two port-nodes are both off-path so the entity counts once).
        assert_eq!(unreachable, 3);
    }

    #[test]
    fn test_has_cycle_detection() {
        let mut g = FactoryGraph {
            nodes: vec![
                GraphNode {
                    id: NodeId::single(Item::TransportBelt, 0, 0),
                    entity_kind: Item::TransportBelt,
                    item: None,
                    misc: Misc::None,
                    recipe_item: None,
                    input: HashMap::new(),
                    output: HashMap::new(),
                },
                GraphNode {
                    id: NodeId::single(Item::TransportBelt, 1, 0),
                    entity_kind: Item::TransportBelt,
                    item: None,
                    misc: Misc::None,
                    recipe_item: None,
                    input: HashMap::new(),
                    output: HashMap::new(),
                },
            ],
            node_index: HashMap::from([
                (NodeId::single(Item::TransportBelt, 0, 0), 0),
                (NodeId::single(Item::TransportBelt, 1, 0), 1),
            ]),
            successors: vec![vec![1], vec![0]],
            predecessors: vec![vec![1], vec![0]],
        };

        assert!(has_cycle(&g));

        g.successors = vec![vec![1], vec![]];
        g.predecessors = vec![vec![], vec![0]];
        assert!(!has_cycle(&g));
    }

    #[test]
    fn test_splitter_passthrough() {
        let mut w = World::empty(6, 2);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place_splitter(2, 0, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        let throughput = output[&Item::CopperCable];
        assert!(
            (throughput - 15.0).abs() < 1e-9,
            "Passthrough splitter should give 15.0, got {}",
            throughput
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_splitter_split() {
        let mut w = World::empty(6, 2);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place_splitter(2, 0, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 1, Item::TransportBelt, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        w.place(4, 1, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        let throughput = output[&Item::CopperCable];
        assert!(
            (throughput - 15.0).abs() < 1e-9,
            "Split should give 15.0 total, got {}",
            throughput
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_splitter_merge() {
        let mut w = World::empty(6, 2);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(0, 1, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);
        w.place_splitter(2, 0, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        let throughput = output[&Item::CopperCable];
        assert!(
            (throughput - 15.0).abs() < 1e-9,
            "Merge should give 15.0, got {}",
            throughput
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_splitter_full_throughput() {
        let mut w = World::empty(6, 2);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(0, 1, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);
        w.place_splitter(2, 0, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 1, Item::TransportBelt, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        w.place(4, 1, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        let throughput = output[&Item::CopperCable];
        assert!(
            (throughput - 30.0).abs() < 1e-9,
            "Full splitter should give 30.0, got {}",
            throughput
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_source_only_no_sink() {
        let mut w = World::empty(3, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, _) = calc_throughput(&g);
        assert!(output.is_empty());
    }

    #[test]
    fn test_belt_chain_saturates_at_15_via_both_lanes() {
        let mut w = World::empty(5, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, _) = calc_throughput(&g);
        assert!((output[&Item::CopperCable] - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_underground_belt_preserves_15_throughput() {
        let mut w = World::empty(7, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::IronPlate));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place_underground(2, 0, Direction::East, Misc::UndergroundDown);
        w.place_underground(4, 0, Direction::East, Misc::UndergroundUp);
        w.place(5, 0, Item::TransportBelt, Direction::East, None);
        w.place(6, 0, Item::Sink, Direction::East, Some(Item::IronPlate));

        let g = build_graph(&w);
        let (output, _) = calc_throughput(&g);
        assert!((output[&Item::IronPlate] - 15.0).abs() < 1e-9);
    }

    #[test]
    fn test_lone_curve_east_to_south_keeps_15() {
        let mut w = World::empty(5, 5);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::South, None);
        w.place(3, 1, Item::TransportBelt, Direction::South, None);
        w.place(3, 2, Item::TransportBelt, Direction::South, None);
        w.place(3, 3, Item::Sink, Direction::South, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, _) = calc_throughput(&g);
        assert!(
            (output[&Item::CopperCable] - 15.0).abs() < 1e-6,
            "lone E→S curve expected 15.0, got {}",
            output[&Item::CopperCable]
        );
    }

    #[test]
    fn test_lone_curve_south_to_east_keeps_15() {
        let mut w = World::empty(5, 5);
        w.place(
            0,
            0,
            Item::Source,
            Direction::South,
            Some(Item::CopperCable),
        );
        w.place(0, 1, Item::TransportBelt, Direction::South, None);
        w.place(0, 2, Item::TransportBelt, Direction::East, None);
        w.place(1, 2, Item::TransportBelt, Direction::East, None);
        w.place(2, 2, Item::TransportBelt, Direction::East, None);
        w.place(3, 2, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, _) = calc_throughput(&g);
        assert!((output[&Item::CopperCable] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_node_per_port_doubles_belt_node_count() {
        // A trivial belt-chain factory: 1 source + 3 belts + 1 sink. With
        // node-per-port: 1 + 6 + 1 = 8 nodes (each belt contributes two).
        let mut w = World::empty(5, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        let g = build_graph(&w);
        assert_eq!(g.node_count(), 8);
        // Sanity-check NodeId.port enum tag for one of the belts.
        let port_node = g
            .get_index(&NodeId::port(Item::TransportBelt, 2, 0))
            .unwrap();
        assert_eq!(g.nodes[port_node].id.port, PortRole::Port);
    }
}
