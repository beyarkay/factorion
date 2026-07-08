use std::collections::{HashMap, HashSet, VecDeque};

use crate::entities::{EntityEnum, FactoryEntity};
use crate::graph::FactoryGraph;
use crate::types::{Item, Lane};

/// Exponent for the factory score's power mean over per-sink deliveries.
///
/// `score = ((1/N) · Σ achievedᵢ^p)^(1/p)` with `p = FACTORY_SCORE_P`. At
/// `p = 0.5` the mean is concave, so it rewards spreading flow across all
/// sinks (for `p < 1`, the power mean is maximised — for a fixed total — by
/// an even split) and drags the score down for every under-served sink,
/// *without* zeroing it on partial coverage the way the geometric mean
/// (`p → 0`) would. An unused sink contributes `0` to the sum: a real but
/// recoverable penalty, not a hand-tuned constant. The mean is homogeneous
/// of degree one, so the score keeps items/s units and lives in `[0, ∞)`.
pub const FACTORY_SCORE_P: f64 = 0.5;

/// What a single sink received: the item it was configured to accept and
/// the achieved throughput (items/s) of that item arriving at it. One
/// `SinkDelivery` per (sink, expected-item) demand — today one per sink,
/// since a sink carries a single item, but the score generalises to
/// multi-item sinks by emitting one delivery per accepted item.
#[derive(Debug, Clone, PartialEq)]
pub struct SinkDelivery {
    /// Anchor tile of the sink entity this delivery belongs to, so callers
    /// (e.g. the Factorio parity harness) can match per-sink rates by grid
    /// position rather than relying on iteration order.
    pub anchor: (usize, usize),
    pub item: Option<Item>,
    pub achieved: f64,
}

/// Generalised (power / Hölder) mean of non-negative `values` with exponent
/// `p`: `((1/N) · Σ vᵢ^p)^(1/p)`. `p = 1` is the arithmetic mean, `p → 0`
/// the geometric mean, `p = 0.5` our soft unused-sink penalty. Empty input
/// → `0.0`. `p` must be non-zero (the `p → 0` limit is the geometric mean,
/// not computed here).
pub fn power_mean(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum_of_powers: f64 = values.iter().map(|&v| v.powf(p)).sum();
    (sum_of_powers / values.len() as f64).powf(1.0 / p)
}

/// Aggregate per-sink deliveries into a single factory throughput score: the
/// power mean (exponent [`FACTORY_SCORE_P`]) of the achieved throughputs.
/// This penalises unused sinks — wiring one source straight to one of two
/// sinks scores below feeding both — without a per-sink penalty constant. A
/// non-finite result (e.g. an unbounded source-adjacent-to-sink layout)
/// collapses to `0.0`, matching the engine's other degenerate cases.
pub fn factory_score(deliveries: &[SinkDelivery]) -> f64 {
    let achieved: Vec<f64> = deliveries.iter().map(|d| d.achieved).collect();
    let score = power_mean(&achieved, FACTORY_SCORE_P);
    if score.is_finite() {
        score
    } else {
        0.0
    }
}

/// Calculate the per-sink deliveries of a factory graph.
///
/// Returns `(deliveries, num_unreachable)`, one [`SinkDelivery`] per sink:
/// the item it was configured to accept and how much of that item reached
/// it. Collapse to a scalar score with [`factory_score`].
///
/// Algorithm:
/// 1. Detect cycles → return no deliveries if any exist
/// 2. BFS from sources (stack_inserters), propagating flow rates
/// 3. At each node, compute output from input using entity trait
/// 4. Collect each sink's delivery of its configured item
/// 5. Count unreachable nodes
pub fn calc_throughput(graph: &FactoryGraph) -> (Vec<SinkDelivery>, usize) {
    if graph.node_count() == 0 {
        return (Vec::new(), 0);
    }

    // 1. Cycle detection — no deliveries (downstream treats as 0 throughput).
    if has_cycle(graph) {
        return (Vec::new(), 0);
    }

    let sources: Vec<usize> = (0..graph.node_count())
        .filter(|&i| graph.nodes[i].entity_kind == Item::Source)
        .collect();
    let sinks: Vec<usize> = (0..graph.node_count())
        .filter(|&i| graph.nodes[i].entity_kind == Item::Sink)
        .collect();

    if sources.is_empty() || sinks.is_empty() {
        // No sources or no sinks → no throughput, every entity is unreachable
        return (
            Vec::new(),
            count_unreachable_entities(graph, &HashSet::new()),
        );
    }

    // Find nodes reachable from sources (for determining processing order)
    let reachable_from_sources = reachable_from(&sources, graph, false);

    // 2. Kahn's algorithm: topological BFS from sources.
    // Count "true" in-degree (only from predecessors reachable from a source).
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

        // If this node's output is already set (e.g., source), skip input computation
        if node_outputs[node_idx].is_empty() {
            // Accumulate input from all predecessors' outputs. Inserters
            // fill their capacity lane-by-lane in priority order instead of
            // summing everything.
            let accumulated_input: HashMap<Item, f64> =
                if matches!(entity_kind, Item::Inserter | Item::LongHandedInserter) {
                    pickup_input(graph, node_idx, &node_outputs)
                } else {
                    let mut acc: HashMap<Item, f64> = HashMap::new();
                    for &pred in &graph.predecessors[node_idx] {
                        for (&item, &flow_rate) in &node_outputs[pred] {
                            *acc.entry(item).or_insert(0.0) += flow_rate;
                        }
                    }
                    acc
                };
            node_inputs[node_idx] = accumulated_input.clone();

            // Compute output using stack-allocated enum dispatch (no heap allocation)
            let item = if entity_kind == Item::AssemblingMachine1 {
                graph.nodes[node_idx].recipe_item
            } else {
                graph.nodes[node_idx].item
            };
            let misc = graph.nodes[node_idx].misc;
            // entity_kind comes from a graph node, which only contains
            // placeable items, so new() always returns Some here.
            if let Some(entity) = EntityEnum::new(entity_kind, item, misc) {
                node_outputs[node_idx] = entity.transform_flow(&accumulated_input);
            }

            // A node hands its contents to its successors, so fanning out to
            // k successors splits the flow k ways (each branch gets 1/k).
            // Without this, every successor reads the node's *full* output and
            // the same items are counted once per branch — the #87 fan-out
            // double-count. Splitters are just the most visible case; a belt
            // feeding two inserters splits exactly the same way. Sources are
            // intentionally excluded: their output is pre-set to an infinite
            // supply (so `is_empty()` skips this block) and feeds every branch
            // fully, dividing INFINITY changes nothing downstream anyway.
            let num_successors = graph.successors[node_idx].len();
            if num_successors > 1 {
                for rate in node_outputs[node_idx].values_mut() {
                    *rate /= num_successors as f64;
                }
            }
        }

        already_processed.insert(node_idx);

        // Decrement in-degree of successors; enqueue when all predecessors processed
        for &succ in &graph.successors[node_idx] {
            if already_processed.contains(&succ) {
                continue;
            }
            // Saturating sub in case of edges from unreachable predecessors
            in_degree[succ] = in_degree[succ].saturating_sub(1);
            if in_degree[succ] == 0 && !in_queue.contains(&succ) {
                queue.push_back(succ);
                in_queue.insert(succ);
            }
        }
    }

    // 3. Collect each sink's delivery. A sink only scores the item it is
    //    configured to receive — its tile's ITEMS channel, surfaced as
    //    `node.item`. Counting *any* item that reaches a sink would let a
    //    policy reward-hack assemble lessons (route raw input straight to
    //    the sink, never build the assembler) and split lessons (wire one
    //    source to one of two sinks, ignore the rest). We record the
    //    achieved rate of the expected item per sink and let `factory_score`
    //    penalise the under-served ones — summing across sinks here would
    //    hide an unused sink behind a fully-fed one.
    let mut deliveries: Vec<SinkDelivery> = Vec::with_capacity(sinks.len());
    for &sink_idx in &sinks {
        let expected = graph.nodes[sink_idx].item;
        let achieved = match expected {
            Some(item) => node_outputs[sink_idx].get(&item).copied().unwrap_or(0.0),
            None => 0.0,
        };
        deliveries.push(SinkDelivery {
            anchor: graph.nodes[sink_idx].anchor,
            item: expected,
            achieved,
        });
    }

    // 4. Count unreachable ENTITIES (not nodes)
    // on_path = can_reach_sink ∩ reachable_from_source.
    // Note: reachable_from includes the start nodes themselves, so sources are in
    // reachable_from_source and sinks are in can_reach_sink. If there's a path from
    // source to sink, both will be in the intersection. If not, they're unreachable.
    let can_reach_sink = reachable_from(&sinks, graph, true); // reverse reachability
    let reachable_from_source = reachable_from(&sources, graph, false);
    let on_path: HashSet<usize> = can_reach_sink
        .intersection(&reachable_from_source)
        .copied()
        .collect();

    (deliveries, count_unreachable_entities(graph, &on_path))
}

/// Count entity units with NO node on a source→sink path. Since dual lanes,
/// nodes are not 1:1 with entities — a belt tile owns two lane nodes and a
/// splitter four — and plenty of legitimate layouts leave a lane forever
/// empty (an inserter only ever fills one lane of the belt it drops on), so
/// unreachability is an entity-level judgment: an entity is an orphan iff
/// none of its nodes lie on a path. Grouping is by the entity's anchor tile.
fn count_unreachable_entities(graph: &FactoryGraph, on_path: &HashSet<usize>) -> usize {
    let mut unit_on_path: HashMap<(usize, usize), bool> = HashMap::new();
    for (idx, node) in graph.nodes.iter().enumerate() {
        let hit = unit_on_path.entry(node.anchor).or_insert(false);
        *hit |= on_path.contains(&idx);
    }
    unit_on_path.values().filter(|&&hit| !hit).count()
}

/// An inserter's accumulated input: greedily fill its capacity (0.86 i/s)
/// from its predecessors in lane-priority order, per the wiki pickup rules.
/// When picking from a belt perpendicular to the inserter, the NEAR lane is
/// preferred (the lane on the inserter's side — `Lane::on_side` of the
/// belt's facing and the belt→inserter direction, which is the inserter's
/// own facing); from a parallel/anti-parallel belt or a curve, the belt's
/// LEFT lane. The other lane only tops up whatever capacity remains — this
/// is the steady-state reading of "takes from the far lane if the near lane
/// is empty", and it decides which ITEM wins the inserter when the two
/// lanes carry different items. Items within one predecessor drain in id
/// order for determinism (HashMap iteration order is not).
fn pickup_input(
    graph: &FactoryGraph,
    node_idx: usize,
    node_outputs: &[HashMap<Item, f64>],
) -> HashMap<Item, f64> {
    let ins_dir = graph.nodes[node_idx].direction;
    let mut preds = graph.predecessors[node_idx].clone();
    preds.sort_by_key(|&p| {
        let pn = &graph.nodes[p];
        match pn.id.lane {
            Some(lane) => {
                let preferred = if pn.curved {
                    Lane::Left
                } else {
                    Lane::on_side(pn.direction, ins_dir).unwrap_or(Lane::Left)
                };
                usize::from(lane != preferred)
            }
            // Lane-less predecessors (assembler/source) can't co-occur with
            // lane nodes — an inserter picks from a single tile — but order
            // them last for form.
            None => 2,
        }
    });

    let mut remaining = graph.nodes[node_idx].entity_kind.flow_rate();
    let mut input: HashMap<Item, f64> = HashMap::new();
    for p in preds {
        if remaining <= 0.0 {
            break;
        }
        let mut items: Vec<(Item, f64)> = node_outputs[p].iter().map(|(&i, &r)| (i, r)).collect();
        items.sort_by_key(|&(i, _)| i as i64);
        for (item, rate) in items {
            if remaining <= 0.0 {
                break;
            }
            let take = rate.min(remaining);
            if take > 0.0 {
                *input.entry(item).or_insert(0.0) += take;
                remaining -= take;
            }
        }
    }
    input
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::graph::build_graph;
    use crate::types::{Direction, Misc, NodeId};
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

        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);

        // Single sink, belt-limited to 15.0.
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].item, Some(Item::CopperCable));
        assert!(
            (output[0].achieved - 15.0).abs() < 1e-9,
            "Expected 15.0, got {}",
            output[0].achieved
        );
        assert!((factory_score(&output) - 15.0).abs() < 1e-9);
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_source_inserter_belt_sink() {
        // Source → Inserter → Belt → Sink
        // Throughput limited by inserter rate (0.86).
        let mut w = World::empty(4, 1);

        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::Inserter, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);

        // Single sink, inserter-limited to 0.86.
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].item, Some(Item::CopperCable));
        assert!(
            (output[0].achieved - 0.86).abs() < 1e-9,
            "Expected 0.86, got {}",
            output[0].achieved
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_disconnected_entities() {
        // Source and sink exist but aren't connected. A lone belt sits in the middle.
        let mut w = World::empty(5, 5);

        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(4, 4, Item::Sink, Direction::East, Some(Item::CopperCable));
        w.place(2, 2, Item::TransportBelt, Direction::East, None);

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);

        // No path from source to sink → the sink's delivery is 0, so the
        // factory scores 0. All 3 entities are off any source→sink path.
        assert!(output.iter().all(|d| d.achieved == 0.0), "{:?}", output);
        assert_eq!(factory_score(&output), 0.0);
        assert_eq!(unreachable, 3); // all 3 entities are disconnected
    }

    #[test]
    fn test_has_cycle_detection() {
        // Build a graph with a cycle manually
        let mut g = FactoryGraph {
            nodes: vec![
                crate::graph::GraphNode {
                    id: NodeId::new(Item::TransportBelt, 0, 0, None),
                    entity_kind: Item::TransportBelt,
                    item: None,
                    misc: Misc::None,
                    recipe_item: None,
                    anchor: (0, 0),
                    direction: Direction::East,
                    curved: false,
                    input: HashMap::new(),
                    output: HashMap::new(),
                },
                crate::graph::GraphNode {
                    id: NodeId::new(Item::TransportBelt, 1, 0, None),
                    entity_kind: Item::TransportBelt,
                    item: None,
                    misc: Misc::None,
                    recipe_item: None,
                    anchor: (1, 0),
                    direction: Direction::East,
                    curved: false,
                    input: HashMap::new(),
                    output: HashMap::new(),
                },
            ],
            node_index: HashMap::from([
                (NodeId::new(Item::TransportBelt, 0, 0, None), 0),
                (NodeId::new(Item::TransportBelt, 1, 0, None), 1),
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
    fn test_splitter_passthrough() {
        // Source → Belt → Splitter → Belt → Sink (1 input, 1 output = no splitting)
        let mut w = World::empty(6, 2);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place_splitter(2, 0, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        let score = factory_score(&output);
        assert!(
            (score - 15.0).abs() < 1e-9,
            "Passthrough splitter should give 15.0, got {}",
            score
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_splitter_split() {
        // Source → Belt → Splitter → Belt → Sink1
        //                          → Belt → Sink2
        // Each sink should get 7.5
        let mut w = World::empty(6, 2);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place_splitter(2, 0, Direction::East, None);
        // Two output belts, one per splitter tile
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 1, Item::TransportBelt, Direction::East, None);
        // Two sinks
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        w.place(4, 1, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        // Two sinks, evenly split at 7.5 each. The power-mean score of two
        // equal deliveries is that per-sink value (7.5), not the 15.0 sum.
        assert_eq!(output.len(), 2);
        assert!(
            output.iter().all(|d| (d.achieved - 7.5).abs() < 1e-9),
            "each sink should get 7.5, got {:?}",
            output
        );
        assert!((factory_score(&output) - 7.5).abs() < 1e-9);
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_splitter_merge() {
        // Source1 → Belt → Splitter → Belt → Sink
        // Source2 → Belt ↗
        let mut w = World::empty(6, 2);
        // Two sources feeding into both input lanes
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(0, 1, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);
        w.place_splitter(2, 0, Direction::East, None);
        w.place(3, 0, Item::TransportBelt, Direction::East, None);
        w.place(4, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);
        // Two inputs (15+15=30), splitter cap 30, 1 output belt caps at 15.
        // Single sink → score is just its delivery (15.0).
        let score = factory_score(&output);
        assert!(
            (score - 15.0).abs() < 1e-9,
            "Merge should give 15.0, got {}",
            score
        );
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_splitter_full_throughput() {
        // 2 inputs + 2 outputs = full 30 i/s throughput (15 per output)
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
        // 30 in, splitter passes 30, each of two output sinks gets 15. The
        // power-mean score of two full sinks is 15.0 (per-sink), not 30.0.
        assert_eq!(output.len(), 2);
        assert!(
            output.iter().all(|d| (d.achieved - 15.0).abs() < 1e-9),
            "each sink should get 15.0, got {:?}",
            output
        );
        assert!((factory_score(&output) - 15.0).abs() < 1e-9);
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
    fn test_assemble_bypass_scores_zero() {
        // Reward-hack regression: a sink only counts the item it is
        // configured to receive. Here the sink expects CopperCable (the
        // crafted output), but the policy "cheats" by laying a straight
        // belt that runs the raw CopperPlate input from source to sink at
        // belt speed, never building the assembler. The raw input is not
        // the expected output, so this must score 0 — not 15.
        let mut w = World::empty(4, 1);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperPlate));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);

        // The sink expects CopperCable; only raw CopperPlate is routed to it,
        // which is not its configured item, so it delivers 0 → factory scores 0.
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].item, Some(Item::CopperCable));
        assert_eq!(
            output[0].achieved, 0.0,
            "Bypass should yield 0 CopperCable, got {:?}",
            output
        );
        assert_eq!(factory_score(&output), 0.0);
        assert_eq!(unreachable, 0);
    }

    #[test]
    fn test_assemble_genuine_craft_scores_recipe_rate() {
        // The complement to the bypass test: a real craft scores. The
        // assembler (recipe CopperPlate → 2× CopperCable) is fed raw
        // CopperPlate via an inserter and hands CopperCable to the sink via
        // an output inserter + belt. The sink expects CopperCable, so the
        // crafted output is counted. Throughput is inserter-limited (0.86).
        //
        //   Source(CopperPlate) → Inserter → Assembler(→CopperCable)
        //                                  → Inserter → Belt → Sink(CopperCable)
        let mut w = World::empty(9, 5);
        w.place(0, 2, Item::Source, Direction::East, Some(Item::CopperPlate));
        w.place(1, 2, Item::Inserter, Direction::East, None);
        // 3x3 assembler, anchor (2,1), recipe keyed by its output item.
        w.place_multi_tile(
            2,
            1,
            Item::AssemblingMachine1,
            Direction::East,
            Some(Item::CopperCable),
            3,
            3,
        );
        w.place(5, 2, Item::Inserter, Direction::East, None);
        w.place(6, 2, Item::TransportBelt, Direction::East, None);
        w.place(7, 2, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, unreachable) = calc_throughput(&g);

        // Only the crafted CopperCable is delivered, inserter-limited to 0.86.
        assert_eq!(output.len(), 1);
        assert_eq!(output[0].item, Some(Item::CopperCable));
        assert!(
            (output[0].achieved - 0.86).abs() < 1e-9,
            "Expected inserter-limited 0.86 CopperCable, got {:?}",
            output
        );
        assert_eq!(unreachable, 0);
    }

    /// Two-item belt for the pickup-priority tests: iron sideloaded onto the
    /// LEFT lane (from the north), copper onto the RIGHT lane (from the
    /// south) of an east-running belt at (1,1)→(2,1).
    fn make_two_lane_belt_world(w_extra: impl FnOnce(&mut World)) -> World {
        let mut w = World::empty(5, 4);
        w.place(1, 0, Item::Source, Direction::South, Some(Item::IronPlate));
        w.place(
            1,
            2,
            Item::Source,
            Direction::North,
            Some(Item::CopperPlate),
        );
        w.place(1, 1, Item::TransportBelt, Direction::East, None);
        w.place(2, 1, Item::TransportBelt, Direction::East, None);
        w_extra(&mut w);
        w
    }

    #[test]
    fn test_pickup_prefers_left_lane_from_parallel_belt() {
        // In-line inserter at (3,1): the belt runs parallel, so it prefers
        // the LEFT lane (iron) and its 0.86 capacity is spent entirely on
        // iron — no copper gets through.
        for (sink_item, want) in [(Item::IronPlate, 0.86), (Item::CopperPlate, 0.0)] {
            let w = make_two_lane_belt_world(|w| {
                w.place(3, 1, Item::Inserter, Direction::East, None);
                w.place(4, 1, Item::Sink, Direction::East, Some(sink_item));
            });
            let g = build_graph(&w);
            let (output, _) = calc_throughput(&g);
            assert_eq!(output.len(), 1);
            assert!(
                (output[0].achieved - want).abs() < 1e-9,
                "sink {sink_item:?}: expected {want}, got {:?}",
                output
            );
        }
    }

    #[test]
    fn test_pickup_prefers_near_lane_from_perpendicular_belt() {
        // Perpendicular inserter at (2,2), south of the belt: the near lane
        // is the belt's south side — its RIGHT lane, carrying copper.
        for (sink_item, want) in [(Item::CopperPlate, 0.86), (Item::IronPlate, 0.0)] {
            let w = make_two_lane_belt_world(|w| {
                w.place(2, 2, Item::Inserter, Direction::South, None);
                w.place(2, 3, Item::Sink, Direction::South, Some(sink_item));
            });
            let g = build_graph(&w);
            let (output, _) = calc_throughput(&g);
            assert_eq!(output.len(), 1);
            assert!(
                (output[0].achieved - want).abs() < 1e-9,
                "sink {sink_item:?}: expected {want}, got {:?}",
                output
            );
        }
    }

    #[test]
    fn test_pickup_falls_back_to_other_lane() {
        // Only the non-preferred lane carries items (iron on the LEFT lane,
        // perpendicular pickup from the south prefers the empty RIGHT lane):
        // the inserter tops up from the far lane and still moves 0.86.
        let mut w = World::empty(5, 4);
        w.place(1, 0, Item::Source, Direction::South, Some(Item::IronPlate));
        // A bare belt stub behind (1,1) keeps it straight (two feeders → the
        // side source sideloads onto the LEFT lane instead of curving).
        w.place(0, 1, Item::TransportBelt, Direction::East, None);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);
        w.place(2, 1, Item::TransportBelt, Direction::East, None);
        w.place(2, 2, Item::Inserter, Direction::South, None);
        w.place(2, 3, Item::Sink, Direction::South, Some(Item::IronPlate));

        let g = build_graph(&w);
        let (output, _) = calc_throughput(&g);
        assert_eq!(output.len(), 1);
        assert!(
            (output[0].achieved - 0.86).abs() < 1e-9,
            "expected fallback pickup of 0.86, got {:?}",
            output
        );
    }

    #[test]
    fn test_power_mean_basic() {
        // Empty → 0; a single value and a set of equal values → that value.
        assert_eq!(power_mean(&[], 0.5), 0.0);
        assert!((power_mean(&[7.5], 0.5) - 7.5).abs() < 1e-9);
        assert!((power_mean(&[7.5, 7.5], 0.5) - 7.5).abs() < 1e-9);
        assert!((power_mean(&[15.0, 15.0], 0.5) - 15.0).abs() < 1e-9);
        // p=0.5: (15, 0) → ((√15 + 0)/2)² = 3.75 (penalised, not zero).
        assert!((power_mean(&[15.0, 0.0], 0.5) - 3.75).abs() < 1e-9);
        // p=1 is the plain arithmetic mean — no imbalance penalty.
        assert!((power_mean(&[15.0, 0.0], 1.0) - 7.5).abs() < 1e-9);
        // Homogeneous of degree one: scaling all inputs scales the mean.
        assert!((power_mean(&[30.0, 0.0], 0.5) - 7.5).abs() < 1e-9);
    }

    #[test]
    fn test_unused_sink_penalised() {
        // Bypass and balanced deliver the SAME raw total (15), but the power
        // mean rewards the even split (7.5) over dumping it all in one sink
        // (3.75), and feeding both sinks fully (15) beats both.
        let bypass = [
            SinkDelivery {
                anchor: (0, 0),
                item: Some(Item::CopperCable),
                achieved: 15.0,
            },
            SinkDelivery {
                anchor: (0, 0),
                item: Some(Item::CopperCable),
                achieved: 0.0,
            },
        ];
        let balanced = [
            SinkDelivery {
                anchor: (0, 0),
                item: Some(Item::CopperCable),
                achieved: 7.5,
            },
            SinkDelivery {
                anchor: (0, 0),
                item: Some(Item::CopperCable),
                achieved: 7.5,
            },
        ];
        let full = [
            SinkDelivery {
                anchor: (0, 0),
                item: Some(Item::CopperCable),
                achieved: 15.0,
            },
            SinkDelivery {
                anchor: (0, 0),
                item: Some(Item::CopperCable),
                achieved: 15.0,
            },
        ];
        assert!((factory_score(&bypass) - 3.75).abs() < 1e-9);
        assert!((factory_score(&balanced) - 7.5).abs() < 1e-9);
        assert!((factory_score(&full) - 15.0).abs() < 1e-9);
        assert!(factory_score(&bypass) < factory_score(&balanced));
        assert!(factory_score(&balanced) < factory_score(&full));
    }

    #[test]
    fn test_split_one_of_two_sinks_penalised() {
        // Two sinks both want CopperCable, but the source is wired straight to
        // one of them (ignoring the other). End-to-end, the engine reports the
        // unfed sink's 0 delivery, so the power-mean score is the penalised
        // 3.75 — where summing across sinks would have hidden it as a full 15.
        let mut w = World::empty(5, 2);
        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::TransportBelt, Direction::East, None);
        w.place(2, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        // Second sink, never fed.
        w.place(4, 1, Item::Sink, Direction::East, Some(Item::CopperCable));

        let g = build_graph(&w);
        let (output, _) = calc_throughput(&g);
        assert_eq!(output.len(), 2);
        assert!(output.iter().any(|d| (d.achieved - 15.0).abs() < 1e-9));
        assert!(output.iter().any(|d| d.achieved == 0.0));
        let score = factory_score(&output);
        assert!(
            (score - 3.75).abs() < 1e-9,
            "one-of-two-sinks bypass should score 3.75, got {}",
            score
        );
    }
}
