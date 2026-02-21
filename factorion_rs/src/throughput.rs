use std::collections::{HashMap, HashSet, VecDeque};

use crate::entities::make_entity;
use crate::graph::FactoryGraph;
use crate::types::{EntityKind, Item, Misc};

/// Calculate the throughput of a factory graph.
///
/// Returns (output_per_item, num_unreachable) matching the Python `calc_throughput`.
///
/// Algorithm:
/// 1. Detect cycles → return 0 if any exist
/// 2. BFS from sources (stack_inserters), propagating flow rates
/// 3. At each node, compute output from input using entity trait
/// 4. Collect output at sinks (bulk_inserters)
/// 5. Count unreachable nodes
pub fn calc_throughput(graph: &FactoryGraph) -> (HashMap<Item, f64>, usize) {
    if graph.node_count() == 0 {
        return (HashMap::new(), 0);
    }

    // 1. Cycle detection
    if has_cycle(graph) {
        return (HashMap::from([(Item::Empty, 0.0)]), 0);
    }

    // Identify sources and sinks
    let sources: Vec<usize> = (0..graph.node_count())
        .filter(|&i| graph.nodes[i].entity_kind == EntityKind::Source)
        .collect();
    let sinks: Vec<usize> = (0..graph.node_count())
        .filter(|&i| graph.nodes[i].entity_kind == EntityKind::Sink)
        .collect();

    if sources.is_empty() || sinks.is_empty() {
        return (HashMap::new(), 0);
    }

    // Find nodes reachable from sources (for determining processing order)
    let reachable_from_sources = reachable_from(&sources, graph, false);

    // 2. Topological BFS from sources
    // We use a queue and process nodes when all their "true" predecessors
    // (those reachable from a source) have been processed.
    let mut already_processed: HashSet<usize> = HashSet::new();
    let mut queue: VecDeque<usize> = VecDeque::new();

    // Initialize: start with sources
    for &s in &sources {
        queue.push_back(s);
    }

    let max_iterations = graph.node_count() * graph.node_count();
    let mut iteration = 0;

    // Clone the graph's nodes into a mutable working copy
    let mut node_inputs: Vec<HashMap<Item, f64>> = graph
        .nodes
        .iter()
        .map(|n| n.input.clone())
        .collect();
    let mut node_outputs: Vec<HashMap<Item, f64>> = graph
        .nodes
        .iter()
        .map(|n| n.output.clone())
        .collect();

    while let Some(node_idx) = queue.pop_back() {
        iteration += 1;
        if iteration > max_iterations {
            break;
        }

        // Check if all true predecessors have been processed
        let true_preds: Vec<usize> = graph.predecessors[node_idx]
            .iter()
            .filter(|&&p| reachable_from_sources.contains(&p))
            .copied()
            .collect();

        if true_preds.iter().any(|p| !already_processed.contains(p)) {
            // Not ready yet, push to front of queue (will try again later)
            if !queue.is_empty() {
                queue.push_front(node_idx);
            }
            continue;
        }

        if already_processed.contains(&node_idx) {
            continue;
        }

        let entity_kind = graph.nodes[node_idx].entity_kind;

        // If this node's output is already set (e.g., source), skip input computation
        if node_outputs[node_idx].is_empty() {
            // Accumulate input from all predecessors' outputs
            let mut accumulated_input: HashMap<Item, f64> = HashMap::new();
            for &pred in &graph.predecessors[node_idx] {
                for (&item, &flow_rate) in &node_outputs[pred] {
                    *accumulated_input.entry(item).or_insert(0.0) += flow_rate;
                }
            }
            node_inputs[node_idx] = accumulated_input.clone();

            // Compute output using entity's transform_flow
            let entity = make_entity(
                entity_kind,
                graph.nodes[node_idx].item,
                Misc::None, // misc not needed for transform_flow
            );

            if entity_kind == EntityKind::AssemblingMachine1 {
                // For assemblers, use the specialized transform that checks recipes
                let asm_entity = make_entity(
                    entity_kind,
                    graph.nodes[node_idx].recipe_item,
                    Misc::None,
                );
                node_outputs[node_idx] = asm_entity.transform_flow(&accumulated_input);
            } else {
                node_outputs[node_idx] = entity.transform_flow(&accumulated_input);
            }
        }

        // Add unprocessed successors to the queue
        for &succ in &graph.successors[node_idx] {
            if !already_processed.contains(&succ) && !queue.contains(&succ) {
                queue.push_back(succ);
            }
        }

        already_processed.insert(node_idx);
    }

    // 3. Collect output at sinks
    let mut total_output: HashMap<Item, f64> = HashMap::new();
    for &sink_idx in &sinks {
        for (&item, &rate) in &node_outputs[sink_idx] {
            *total_output.entry(item).or_insert(0.0) += rate;
        }
    }

    // 4. Count unreachable nodes
    let can_reach_sink = reachable_from(&sinks, graph, true); // reverse reachability
    let reachable_from_source = reachable_from(&sources, graph, false);
    let on_path: HashSet<usize> = can_reach_sink
        .intersection(&reachable_from_source)
        .copied()
        .collect();
    // Include sources and sinks themselves
    let mut on_path_with_endpoints = on_path;
    for &s in &sources {
        on_path_with_endpoints.insert(s);
    }
    for &s in &sinks {
        on_path_with_endpoints.insert(s);
    }

    let all_nodes: HashSet<usize> = (0..graph.node_count()).collect();
    let unreachable = all_nodes.difference(&on_path_with_endpoints).count();

    (total_output, unreachable)
}

/// Check if the graph has any cycles using DFS.
fn has_cycle(graph: &FactoryGraph) -> bool {
    let n = graph.node_count();
    let mut visited = vec![0u8; n]; // 0=unvisited, 1=in-stack, 2=done

    for start in 0..n {
        if visited[start] != 0 {
            continue;
        }
        let mut stack = vec![(start, 0usize)]; // (node, next_successor_index)
        visited[start] = 1;

        while let Some((node, succ_idx)) = stack.last_mut() {
            if *succ_idx < graph.successors[*node].len() {
                let next = graph.successors[*node][*succ_idx];
                *succ_idx += 1;
                if visited[next] == 1 {
                    return true; // Back edge = cycle
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

/// Find all nodes reachable from `starts`.
/// If `reverse` is true, traverse predecessors instead of successors.
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
mod tests {
    use super::*;
    use crate::graph::build_graph;
    use crate::types::{Channel, Direction, NodeId};
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
        // Source → Belt → Belt → Sink
        // Source and Sink act as inserters (drop onto / pickup from belts).
        // Throughput limited by belt rate (15.0).
        let mut w = World::empty(4, 1);

        // Source at (0,0) facing east, drops onto belt at (1,0)
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(0, 0, Channel::Items, Item::CopperCable as i64);

        // Belt at (1,0)
        w.set(1, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);

        // Belt at (2,0)
        w.set(2, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(2, 0, Channel::Direction, Direction::East as i64);

        // Sink at (3,0) facing east, picks up from belt at (2,0)
        w.set(3, 0, Channel::Entities, EntityKind::Sink as i64);
        w.set(3, 0, Channel::Direction, Direction::East as i64);
        w.set(3, 0, Channel::Items, Item::CopperCable as i64);

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);

        // Belt limits throughput to 15.0
        assert!(
            output.contains_key(&Item::CopperCable),
            "Expected CopperCable in output, got {:?}",
            output
        );
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
        // Source → Inserter → Belt → Sink
        // Throughput limited by inserter rate (0.86).
        let mut w = World::empty(4, 1);

        // Source at (0,0) facing east
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(0, 0, Channel::Items, Item::CopperCable as i64);

        // Inserter at (1,0) facing east (picks up from source, drops on belt)
        w.set(1, 0, Channel::Entities, EntityKind::Inserter as i64);
        w.set(1, 0, Channel::Direction, Direction::East as i64);

        // Belt at (2,0) facing east
        w.set(2, 0, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(2, 0, Channel::Direction, Direction::East as i64);

        // Sink at (3,0) facing east (picks up from belt)
        w.set(3, 0, Channel::Entities, EntityKind::Sink as i64);
        w.set(3, 0, Channel::Direction, Direction::East as i64);
        w.set(3, 0, Channel::Items, Item::CopperCable as i64);

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);

        assert!(
            output.contains_key(&Item::CopperCable),
            "Expected CopperCable in output, got {:?}",
            output
        );
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
        // Source and sink exist but aren't connected. A lone belt sits in the middle.
        let mut w = World::empty(5, 5);

        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(0, 0, Channel::Items, Item::CopperCable as i64);

        w.set(4, 4, Channel::Entities, EntityKind::Sink as i64);
        w.set(4, 4, Channel::Direction, Direction::East as i64);
        w.set(4, 4, Channel::Items, Item::CopperCable as i64);

        // Disconnected belt
        w.set(2, 2, Channel::Entities, EntityKind::TransportBelt as i64);
        w.set(2, 2, Channel::Direction, Direction::East as i64);

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);

        // No path from source to sink → empty output
        assert!(output.is_empty() || output.values().all(|&v| v == 0.0));
        // All 3 entities should be unreachable (none are on a source→sink path)
        // Actually: source and sink are endpoints but not on a path through each other
        // The belt, source, and sink are all disconnected from each other
        assert!(unreachable >= 1); // at least the belt is unreachable
    }

    #[test]
    fn test_has_cycle_detection() {
        // Build a graph with a cycle manually
        let mut g = FactoryGraph {
            nodes: vec![
                crate::graph::GraphNode {
                    id: NodeId::new(EntityKind::TransportBelt, 0, 0),
                    entity_kind: EntityKind::TransportBelt,
                    item: Item::Empty,
                    recipe_item: Item::Empty,
                    input: HashMap::new(),
                    output: HashMap::new(),
                },
                crate::graph::GraphNode {
                    id: NodeId::new(EntityKind::TransportBelt, 1, 0),
                    entity_kind: EntityKind::TransportBelt,
                    item: Item::Empty,
                    recipe_item: Item::Empty,
                    input: HashMap::new(),
                    output: HashMap::new(),
                },
            ],
            node_index: HashMap::from([
                (NodeId::new(EntityKind::TransportBelt, 0, 0), 0),
                (NodeId::new(EntityKind::TransportBelt, 1, 0), 1),
            ]),
            successors: vec![vec![1], vec![0]], // 0→1→0 cycle
            predecessors: vec![vec![1], vec![0]],
        };

        assert!(has_cycle(&g));

        // Make it acyclic
        g.successors = vec![vec![1], vec![]];
        g.predecessors = vec![vec![], vec![0]];
        assert!(!has_cycle(&g));
    }

    #[test]
    fn test_source_only_no_sink() {
        let mut w = World::empty(3, 1);
        w.set(0, 0, Channel::Entities, EntityKind::Source as i64);
        w.set(0, 0, Channel::Direction, Direction::East as i64);
        w.set(0, 0, Channel::Items, Item::CopperCable as i64);

        let g = build_graph(&w);
        let (output, _) = calc_throughput(&g);
        assert!(output.is_empty());
    }
}
