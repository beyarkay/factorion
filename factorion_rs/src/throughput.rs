use std::collections::{HashMap, HashSet, VecDeque};

use crate::entities::{EntityEnum, FactoryEntity};
use crate::graph::FactoryGraph;
use crate::types::{all_items, get_recipe, Item, Lane};

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

/// Sweeps a loop gets to settle. Each pass around a balancer's loop shrinks
/// the gap to its steady state geometrically, so this only binds on a loop
/// that returns nearly all of its flow, which is then scored with its loops
/// cut.
const MAX_SWEEPS: usize = 1000;

/// Change in a rate (items/s) below which a loop counts as settled.
const SETTLED: f64 = 1e-12;

/// What each lane node was offered by each feeder in a sweep, by the rank it
/// fills in: `(feeder, lane) → (rank, items/s)`.
type Fed = HashMap<(usize, usize), (usize, f64)>;

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
/// 1. Cut the edges that close a loop, leaving a DAG
/// 2. BFS from sources (stack_inserters), propagating flow rates, each cut
///    edge carrying what its source passed on in the previous sweep
/// 3. At each node, compute output from input using entity trait
/// 4. Repeat 2-3 until the flow on the cut edges settles
/// 5. Collect each sink's delivery of its configured item
/// 6. Count unreachable nodes
pub fn calc_throughput(graph: &FactoryGraph) -> (Vec<SinkDelivery>, usize) {
    if graph.node_count() == 0 {
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

    let (back, on_loop) = loops(graph, &sources);

    // 2. Kahn's algorithm: topological BFS from sources.
    // Count "true" in-degree (only from predecessors reachable from a source).
    let in_degree: Vec<usize> = graph
        .predecessors
        .iter()
        .enumerate()
        .map(|(n, preds)| {
            preds
                .iter()
                .filter(|&&p| reachable_from_sources.contains(&p) && !back.contains(&(p, n)))
                .count()
        })
        .collect();

    // Inserters on a tile pick up before the flow moves on, so a node's
    // other successors wait for its inserters and get what they leave.
    let is_picker = |n: usize| {
        matches!(
            graph.nodes[n].entity_kind,
            Item::Inserter | Item::LongHandedInserter
        )
    };
    let n_pickers: Vec<usize> = graph
        .successors
        .iter()
        .map(|succs| succs.iter().filter(|&&s| is_picker(s)).count())
        .collect();
    let n_others: Vec<usize> = graph
        .successors
        .iter()
        .zip(&n_pickers)
        .map(|(succs, &k)| succs.len() - k)
        .collect();
    let mut pending_pickers = vec![0usize; graph.node_count()];
    for &p in &reachable_from_sources {
        for &s in &graph.successors[p] {
            if !is_picker(s) && !back.contains(&(p, s)) {
                pending_pickers[s] += n_pickers[p];
            }
        }
    }
    let pending_pickers = pending_pickers;

    let mut node_outputs: Vec<HashMap<Item, f64>> =
        graph.nodes.iter().map(|n| n.output.clone()).collect();
    let mut taken: Vec<HashMap<Item, f64>> = vec![HashMap::new(); graph.node_count()];
    let demand = downstream_demand(graph);
    let mut loop_tails: Vec<usize> = back.iter().map(|&(u, _)| u).collect();
    loop_tails.sort_unstable();
    loop_tails.dedup();
    let mut accepts = vec![f64::INFINITY; graph.node_count()];
    let mut outlets: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    let mut cut: Option<Vec<HashMap<Item, f64>>> = None;

    // A sweep reads a cut edge's source before computing it, so it sees last
    // sweep's flow there; the first sweep, with nothing carried yet, is the
    // pass over the graph with its loops cut. Every sweep moves only what
    // sources put in, so the state the sweeps settle in can't deliver more
    // than the sources emit; one that never settles falls back to the first.
    for sweep in 0..MAX_SWEEPS {
        let settled: Vec<(HashMap<Item, f64>, HashMap<Item, f64>)> = loop_tails
            .iter()
            .map(|&u| (node_outputs[u].clone(), taken[u].clone()))
            .collect();
        // A loop tail keeps last sweep's takes until it is computed again, so
        // what it passes on along a cut edge is net of its inserters.
        let mut carried: Vec<HashMap<Item, f64>> = vec![HashMap::new(); graph.node_count()];
        for &u in &loop_tails {
            carried[u] = std::mem::take(&mut taken[u]);
        }
        taken = carried.clone();
        let mut in_degree = in_degree.clone();
        let mut pending_pickers = pending_pickers.clone();
        let mut room = demand.clone();
        let mut order: Vec<usize> = Vec::new();
        let mut fed: Option<Fed> = (!back.is_empty()).then(HashMap::new);

        let mut queue: VecDeque<usize> = VecDeque::new();
        let mut in_queue: HashSet<usize> = HashSet::new();

        for &s in &sources {
            queue.push_back(s);
            in_queue.insert(s);
        }

        let mut already_processed: HashSet<usize> = HashSet::new();

        // Waiting on inserters can deadlock when an inserter is also fed through
        // the node waiting on it (a source filling an assembler that an inserter
        // picks from as well as the source); release the stalled node then
        // rather than leave it, and everything downstream, without flow.
        while let Some(node_idx) = queue.pop_front().or_else(|| {
            (0..graph.node_count()).find(|&n| {
                in_degree[n] == 0
                    && reachable_from_sources.contains(&n)
                    && !already_processed.contains(&n)
            })
        }) {
            in_queue.remove(&node_idx);

            if already_processed.contains(&node_idx) {
                continue;
            }

            let entity_kind = graph.nodes[node_idx].entity_kind;
            for (item, rate) in &carried[node_idx] {
                *taken[node_idx].entry(*item).or_insert(0.0) -= rate;
            }

            // If this node's output is already set (e.g., source), skip input computation
            if graph.nodes[node_idx].output.is_empty() {
                // Accumulate input from all predecessors' outputs. Inserters
                // divide a fixed capacity over their lanes instead of summing
                // everything.
                let accumulated_input: HashMap<Item, f64> =
                    if matches!(entity_kind, Item::Inserter | Item::LongHandedInserter) {
                        pickup_input(
                            graph,
                            node_idx,
                            entity_kind.flow_rate().min(accepts[node_idx]),
                            &node_outputs,
                            &n_pickers,
                            &mut taken,
                            &demand,
                            &mut room,
                        )
                    } else if entity_kind.is_lane_aware() {
                        lane_input(
                            graph,
                            node_idx,
                            &|pred, left| {
                                let leftover = (&node_outputs[pred], &taken[pred]);
                                share(&outlets, n_others[pred], leftover, pred, node_idx, left)
                            },
                            &node_outputs,
                            &taken,
                            fed.as_mut(),
                        )
                    } else {
                        let mut acc: HashMap<Item, f64> = HashMap::new();
                        for &pred in &graph.predecessors[node_idx] {
                            // What the inserters leave is shared, not copied, so
                            // no item is counted once per branch (#87).
                            for (&item, &flow_rate) in &node_outputs[pred] {
                                let left =
                                    flow_rate - taken[pred].get(&item).copied().unwrap_or(0.0);
                                let leftover = (&node_outputs[pred], &taken[pred]);
                                *acc.entry(item).or_insert(0.0) += share(
                                    &outlets,
                                    n_others[pred],
                                    leftover,
                                    pred,
                                    node_idx,
                                    left.max(0.0),
                                );
                            }
                        }
                        acc
                    };

                // Compute output using stack-allocated enum dispatch (no heap allocation)
                let item = if entity_kind.crafts() {
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
            }

            already_processed.insert(node_idx);
            order.push(node_idx);

            if is_picker(node_idx) {
                for &pred in &graph.predecessors[node_idx] {
                    if !reachable_from_sources.contains(&pred) {
                        continue;
                    }
                    for &succ in &graph.successors[pred] {
                        if !is_picker(succ) && !back.contains(&(pred, succ)) {
                            pending_pickers[succ] = pending_pickers[succ].saturating_sub(1);
                            if pending_pickers[succ] == 0
                                && in_degree[succ] == 0
                                && !already_processed.contains(&succ)
                                && !in_queue.contains(&succ)
                            {
                                queue.push_back(succ);
                                in_queue.insert(succ);
                            }
                        }
                    }
                }
            }

            // Decrement in-degree of successors; enqueue when all predecessors processed
            for &succ in &graph.successors[node_idx] {
                if already_processed.contains(&succ) || back.contains(&(node_idx, succ)) {
                    continue;
                }
                // Saturating sub in case of edges from unreachable predecessors
                in_degree[succ] = in_degree[succ].saturating_sub(1);
                if in_degree[succ] == 0 && pending_pickers[succ] == 0 && !in_queue.contains(&succ) {
                    queue.push_back(succ);
                    in_queue.insert(succ);
                }
            }
        }

        let Some(fed) = fed else {
            break;
        };
        let cut = cut.get_or_insert_with(|| node_outputs.clone());
        let backed_up = backpressure(graph, &order, &on_loop, &fed, &taken, &mut accepts);
        let unsettled =
            loop_tails.iter().zip(&settled).any(|(&u, (out, took))| {
                differs(&node_outputs[u], out) || differs(&taken[u], took)
            }) || backed_up.len() != outlets.len()
                || backed_up.iter().any(|(n, outs)| {
                    outlets.get(n).is_none_or(|old| {
                        old.iter()
                            .zip(outs)
                            .any(|(a, b)| a.0 != b.0 || (a.1 - b.1).abs() > SETTLED)
                    })
                });
        outlets = backed_up;
        if !unsettled {
            break;
        }
        if sweep + 1 == MAX_SWEEPS {
            node_outputs = cut.clone();
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
    // on_path = can_reach_sink ∩ reachable_from_sources.
    // Note: reachable_from includes the start nodes themselves, so sources are in
    // reachable_from_sources and sinks are in can_reach_sink. If there's a path from
    // source to sink, both will be in the intersection. If not, they're unreachable.
    let can_reach_sink = reachable_from(&sinks, graph, true); // reverse reachability
    let on_path: HashSet<usize> = can_reach_sink
        .intersection(&reachable_from_sources)
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

/// A lane node's input. Flow arriving straight on fills the lane first,
/// then sideloads from its flanks, then inserter drops: each only fills the
/// gaps the ones before it leave, so a sideload onto a saturated lane gets
/// nothing.
fn lane_input(
    graph: &FactoryGraph,
    node_idx: usize,
    share: &dyn Fn(usize, f64) -> f64,
    node_outputs: &[HashMap<Item, f64>],
    taken: &[HashMap<Item, f64>],
    mut fed: Option<&mut Fed>,
) -> HashMap<Item, f64> {
    let node = &graph.nodes[node_idx];
    let priority = |p: usize| {
        let pred = &graph.nodes[p];
        if matches!(pred.entity_kind, Item::Inserter | Item::LongHandedInserter) {
            2
        } else if pred.direction != node.direction {
            // Judged by travel, not adjacency: a splitter feeds each tile
            // beside it from its far tile too.
            1
        } else {
            0
        }
    };
    let cap = node.entity_kind.flow_rate() / 2.0;
    let mut room = cap;
    let mut input: HashMap<Item, f64> = HashMap::new();
    for rank in 0..3 {
        let mut offered: HashMap<Item, f64> = HashMap::new();
        let mut by_pred: Vec<(usize, f64)> = Vec::new();
        for &pred in graph.predecessors[node_idx]
            .iter()
            .filter(|&&p| priority(p) == rank)
        {
            // What the inserters leave is shared, not copied, so no item is
            // counted once per branch (#87). A source's infinite rate is
            // clamped to what the lane could ever take.
            let mut from_pred = 0.0;
            for (&item, &flow_rate) in &node_outputs[pred] {
                let left = flow_rate - taken[pred].get(&item).copied().unwrap_or(0.0);
                let rate = share(pred, left.max(0.0)).min(cap);
                *offered.entry(item).or_insert(0.0) += rate;
                from_pred += rate;
            }
            if fed.is_some() {
                by_pred.push((pred, from_pred));
            }
        }
        let total: f64 = offered.values().sum();
        let scale = if total > room { room / total } else { 1.0 };
        if let Some(fed) = fed.as_mut() {
            for (pred, rate) in by_pred {
                fed.insert((pred, node_idx), (rank, rate));
            }
        }
        for (item, rate) in offered {
            *input.entry(item).or_insert(0.0) += rate * scale;
        }
        room -= total.min(room);
    }
    input
}

/// How much of each item a node can pass on in steady state, set by what
/// lies downstream: a belt that leads nowhere fills up and stops, and an
/// assembler only accepts ingredients for the crafts its output can drain.
/// `rest` covers every item without an entry in `items`.
#[derive(Debug, Clone, Default)]
struct Demand {
    rest: f64,
    items: HashMap<Item, f64>,
}

impl Demand {
    fn get(&self, item: Item) -> f64 {
        self.items.get(&item).copied().unwrap_or(self.rest)
    }

    fn consume(&mut self, item: Item, rate: f64) {
        *self.items.get_mut(&item).unwrap_or(&mut self.rest) -= rate;
    }
}

/// Every node's [`Demand`], from the sinks back. A node on (or feeding) a
/// loop waits on itself, so those take the least fixed point instead: from
/// nothing, re-evaluated until they settle.
fn downstream_demand(graph: &FactoryGraph) -> Vec<Demand> {
    let n = graph.node_count();
    let mut demand = vec![Demand::default(); n];
    let mut pending: Vec<usize> = graph.successors.iter().map(Vec::len).collect();
    let mut ready: Vec<usize> = (0..n).filter(|&i| pending[i] == 0).collect();
    while let Some(node) = ready.pop() {
        demand[node] = node_demand(graph, node, &demand);
        for &p in &graph.predecessors[node] {
            pending[p] -= 1;
            if pending[p] == 0 {
                ready.push(p);
            }
        }
    }
    let looped: Vec<usize> = (0..n).filter(|&i| pending[i] > 0).rev().collect();
    for _ in 0..MAX_SWEEPS {
        let mut unsettled = false;
        for &node in &looped {
            let new = node_demand(graph, node, &demand);
            let old = &demand[node];
            unsettled |= (new.rest - old.rest).abs() > SETTLED
                || new.items.len() != old.items.len()
                || new
                    .items
                    .iter()
                    .any(|(&item, &rate)| (rate - old.get(item)).abs() > SETTLED);
            demand[node] = new;
        }
        if !unsettled {
            break;
        }
    }
    demand
}

/// One node's [`Demand`] given its successors'.
fn node_demand(graph: &FactoryGraph, node: usize, demand: &[Demand]) -> Demand {
    let succs = &graph.successors[node];
    let total = |item: Item| succs.iter().map(|&s| demand[s].get(item)).sum::<f64>();
    let kind = graph.nodes[node].entity_kind;
    match kind {
        Item::Sink => Demand {
            rest: f64::INFINITY,
            items: HashMap::new(),
        },
        kind if kind.crafts() => match graph.nodes[node].recipe_item.and_then(get_recipe) {
            Some(recipe) => {
                let crafts = recipe
                    .produces
                    .iter()
                    .map(|&(item, rate)| total(item) / rate)
                    .fold(kind.flow_rate() / recipe.crafting_time, f64::min);
                // A burner takes any one fuel for all of its energy.
                let fuel = kind.fuel_per_craft(&recipe).map(|kj| {
                    all_items()
                        .into_iter()
                        .filter_map(move |item| Some((item, crafts * kj / item.fuel_value()?)))
                });
                Demand {
                    rest: 0.0,
                    items: recipe
                        .consumes
                        .iter()
                        .map(|&(item, rate)| (item, rate * crafts))
                        .chain(fuel.into_iter().flatten())
                        .collect(),
                }
            }
            None => Demand::default(),
        },
        _ => {
            let cap = if kind.is_lane_aware() {
                kind.flow_rate() / 2.0
            } else {
                kind.flow_rate()
            };
            Demand {
                rest: succs.iter().map(|&s| demand[s].rest).sum::<f64>().min(cap),
                items: succs
                    .iter()
                    .flat_map(|&s| demand[s].items.keys())
                    .map(|&item| (item, total(item).min(cap)))
                    .collect(),
            }
        }
    }
}

/// Split `budget` max-min fairly over `(cap, weight)` claims: each gets
/// `λ·weight` clamped to its cap, with `λ` as large as the budget allows.
fn water_fill(claims: &[(f64, f64)], mut budget: f64) -> Vec<f64> {
    let mut order: Vec<usize> = (0..claims.len()).filter(|&i| claims[i].1 > 0.0).collect();
    order.sort_by(|&a, &b| (claims[a].0 / claims[a].1).total_cmp(&(claims[b].0 / claims[b].1)));
    let mut weight: f64 = order.iter().map(|&i| claims[i].1).sum();
    let mut out = vec![0.0; claims.len()];
    for i in order {
        let (cap, w) = claims[i];
        out[i] = cap.min(budget * w / weight).max(0.0);
        budget -= out[i];
        weight -= w;
    }
    out
}

/// An inserter's accumulated input, recorded in `taken` so the flow it
/// leaves moves on, and in `room` so the next inserter into the same
/// destination only tops up what is still missing. The inserters picking
/// from one lane share it evenly, and a hand only picks what its destination
/// will take. Into a crafting machine it fills in recipe (and fuel)
/// proportion from both lanes at once, since the machine refuses an
/// ingredient it already has enough of. Anywhere else it fills from the near lane first — the lane on
/// the inserter's side, or the belt's left lane when the inserter is in line
/// with it — and reaches the far lane only for what the near one cannot
/// give, splitting each lane's share across its items in proportion to
/// their rates.
#[allow(clippy::too_many_arguments)]
fn pickup_input(
    graph: &FactoryGraph,
    node_idx: usize,
    mut hand: f64,
    node_outputs: &[HashMap<Item, f64>],
    n_pickers: &[usize],
    taken: &mut [HashMap<Item, f64>],
    demand: &[Demand],
    room: &mut [Demand],
) -> HashMap<Item, f64> {
    let mut input: HashMap<Item, f64> = HashMap::new();
    let Some(&dest) = graph.successors[node_idx].first() else {
        return input;
    };
    let dir = graph.nodes[node_idx].direction;
    let mut preds = graph.predecessors[node_idx].clone();
    preds.sort_by_key(|&p| {
        let pred = &graph.nodes[p];
        // A curve's near lane is its left one from either free side (measured
        // in Factorio).
        let near = if pred.curved {
            Lane::Left
        } else {
            Lane::on_side(pred.direction, dir).unwrap_or(Lane::Left)
        };
        pred.id.lane.is_some_and(|lane| lane != near)
    });
    let into_assembler = graph.nodes[dest].entity_kind.crafts();
    let groups: Vec<&[usize]> = if into_assembler {
        vec![&preds]
    } else {
        preds.chunks(1).collect()
    };
    let available = |p: usize, item: Item| {
        node_outputs[p].get(&item).copied().unwrap_or(0.0) / n_pickers[p] as f64
    };

    for group in groups {
        let mut items: Vec<Item> = group
            .iter()
            .flat_map(|&p| node_outputs[p].keys().copied())
            .collect();
        items.sort_by_key(|&item| item as usize);
        items.dedup();
        let claims: Vec<(f64, f64)> = items
            .iter()
            .map(|&item| {
                let supply: f64 = group.iter().map(|&p| available(p, item)).sum();
                // A source's rate is infinite, but it carries a single item,
                // so any weight gives it the whole hand.
                let weight = if into_assembler {
                    demand[dest].get(item)
                } else if supply.is_finite() {
                    supply
                } else {
                    1.0
                };
                (supply.min(room[dest].get(item)), weight)
            })
            .collect();
        let takes = water_fill(&claims, hand);
        for (&item, &take) in items.iter().zip(&takes) {
            if take <= 0.0 {
                continue;
            }
            let mut left = take;
            for &p in group {
                let from_p = left.min(available(p, item));
                if from_p > 0.0 {
                    *taken[p].entry(item).or_insert(0.0) += from_p;
                    left -= from_p;
                }
            }
            *input.entry(item).or_insert(0.0) += take;
            room[dest].consume(item, take);
            hand -= take;
        }
    }
    input
}

/// The edges a DFS from `starts` finds closing a loop (cutting them leaves
/// what is reachable acyclic), and which nodes lie on a loop.
fn loops(graph: &FactoryGraph, starts: &[usize]) -> (HashSet<(usize, usize)>, Vec<bool>) {
    let mut back = HashSet::new();
    let mut visited = vec![0u8; graph.node_count()]; // 0=unvisited, 1=in-stack, 2=done
    let mut finished = Vec::new();

    for &start in starts {
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
                    back.insert((*node, next));
                }
                if visited[next] == 0 {
                    visited[next] = 1;
                    stack.push((next, 0));
                }
            } else {
                visited[*node] = 2;
                finished.push(*node);
                stack.pop();
            }
        }
    }

    // Kosaraju: in reverse finishing order, what reaches a node backwards
    // and is not yet claimed is its strongly connected component.
    let mut on_loop = vec![false; graph.node_count()];
    if !back.is_empty() {
        let mut claimed = vec![false; graph.node_count()];
        for &root in finished.iter().rev() {
            if claimed[root] {
                continue;
            }
            claimed[root] = true;
            let mut component = vec![root];
            let mut i = 0;
            while i < component.len() {
                for &p in &graph.predecessors[component[i]] {
                    if visited[p] == 2 && !claimed[p] {
                        claimed[p] = true;
                        component.push(p);
                    }
                }
                i += 1;
            }
            if component.len() > 1 || back.contains(&(root, root)) {
                for n in component {
                    on_loop[n] = true;
                }
            }
        }
    }
    (back, on_loop)
}

/// What `pred` passes `node` of an item it has `left` of once its inserters
/// have taken theirs: an even share, except that a splitter touching a loop
/// sends an output that has backed up only what it takes and the rest to the
/// others, as the game's splitters do.
fn share(
    outlets: &HashMap<usize, Vec<(usize, f64)>>,
    n_others: usize,
    (output, taken): (&HashMap<Item, f64>, &HashMap<Item, f64>),
    pred: usize,
    node: usize,
    left: f64,
) -> f64 {
    let Some(outs) = outlets.get(&pred) else {
        return left / n_others as f64;
    };
    let total: f64 = output
        .iter()
        .map(|(item, rate)| (rate - taken.get(item).copied().unwrap_or(0.0)).max(0.0))
        .sum();
    let claims: Vec<(f64, f64)> = outs.iter().map(|&(_, room)| (room, 1.0)).collect();
    let sent = water_fill(&claims, total);
    match outs.iter().position(|&(q, _)| q == node) {
        Some(i) if total > 0.0 => left * sent[i] / total,
        _ => 0.0,
    }
}

/// Walk a sweep back from the sinks to find what each lane node can pass
/// on: what its outputs take from it once the flow they put ahead of it is
/// in, plus what its inserters take. A splitter touching a loop then sends
/// each output only what it takes and the rest to the other, and an inserter
/// on or into a loop moves only what its drop takes, so a backed-up loop
/// fills and stops as in the game. Off the loops nothing backs up, as in an
/// acyclic factory.
fn backpressure(
    graph: &FactoryGraph,
    order: &[usize],
    on_loop: &[bool],
    fed: &Fed,
    taken: &[HashMap<Item, f64>],
    accepts: &mut [f64],
) -> HashMap<usize, Vec<(usize, f64)>> {
    let outputs = |n: usize| {
        graph.successors[n].iter().copied().filter(|&q| {
            !matches!(
                graph.nodes[q].entity_kind,
                Item::Inserter | Item::LongHandedInserter
            )
        })
    };
    // An output takes what is left once the flow it puts ahead of `n` is in,
    // shared fairly with whatever arrives level with `n`.
    let takes = |n: usize, q: usize, accepts: &[f64]| {
        let kind = graph.nodes[q].entity_kind;
        let Some(&(rank, _)) = fed.get(&(n, q)).filter(|_| kind.is_lane_aware()) else {
            return f64::INFINITY;
        };
        let offers: Vec<(usize, f64)> = graph.predecessors[q]
            .iter()
            .filter(|&&p| p != n)
            .filter_map(|p| fed.get(&(*p, q)).copied())
            .collect();
        let mut room = accepts[q].min(kind.flow_rate() / 2.0);
        for r in 0..rank {
            room -= offers
                .iter()
                .filter(|o| o.0 == r)
                .map(|o| o.1)
                .sum::<f64>()
                .min(room);
        }
        let mut level: Vec<(f64, f64)> = offers
            .iter()
            .filter(|o| o.0 == rank)
            .map(|o| (o.1, 1.0))
            .collect();
        level.push((f64::INFINITY, 1.0));
        water_fill(&level, room)[level.len() - 1]
    };
    // Both tiles of a splitter split as one, so a splitter touching a loop
    // overflows from either tile.
    let looped: HashSet<(usize, usize)> = (0..graph.node_count())
        .filter(|&n| on_loop[n])
        .map(|n| graph.nodes[n].anchor)
        .collect();
    let mut outlets = HashMap::new();
    for &n in order.iter().rev() {
        let kind = graph.nodes[n].entity_kind;
        let cap = if kind.is_lane_aware() {
            kind.flow_rate() / 2.0
        } else if matches!(kind, Item::Inserter | Item::LongHandedInserter)
            && (on_loop[n] || outputs(n).any(|q| on_loop[q]))
        {
            kind.flow_rate()
        } else {
            continue;
        };
        let picked: f64 = taken[n].values().sum();
        let onward: f64 = outputs(n).map(|q| takes(n, q, accepts)).sum();
        accepts[n] = (onward + picked).min(cap);
        if looped.contains(&graph.nodes[n].anchor) && outputs(n).count() > 1 {
            outlets.insert(n, outputs(n).map(|q| (q, takes(n, q, accepts))).collect());
        }
    }
    outlets
}

/// Whether two per-item rates differ by more than [`SETTLED`].
fn differs(a: &HashMap<Item, f64>, b: &HashMap<Item, f64>) -> bool {
    let rate = |m: &HashMap<Item, f64>, item| m.get(item).copied().unwrap_or(0.0);
    a.keys()
        .chain(b.keys())
        .any(|item| (rate(a, item) - rate(b, item)).abs() > SETTLED)
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
    use crate::rng::Rng;
    use crate::types::{Channel, Direction, Misc, NodeId};
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
        let mut w = World::empty(4, 2);

        w.place(0, 0, Item::Source, Direction::East, Some(Item::CopperCable));
        w.place(1, 0, Item::Inserter, Direction::East, None);
        w.place(2, 0, Item::TransportBelt, Direction::East, None);
        w.place(3, 0, Item::Sink, Direction::East, Some(Item::CopperCable));
        w.place(1, 1, Item::MediumElectricPole, Direction::None, None);

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
                    direction: Direction::North,
                    curved: false,
                    recipe_item: None,
                    anchor: (0, 0),
                    output: HashMap::new(),
                },
                crate::graph::GraphNode {
                    id: NodeId::new(Item::TransportBelt, 1, 0, None),
                    entity_kind: Item::TransportBelt,
                    item: None,
                    misc: Misc::None,
                    direction: Direction::North,
                    curved: false,
                    recipe_item: None,
                    anchor: (1, 0),
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

        assert_eq!(loops(&g, &[0]).0, HashSet::from([(1, 0)]));

        // Make it acyclic
        g.successors = vec![vec![1], vec![]];
        g.predecessors = vec![vec![], vec![0]];
        assert!(loops(&g, &[0]).0.is_empty());
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
        w.place(3, 4, Item::MediumElectricPole, Direction::None, None);

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

    /// Two-item belt for the pickup tests: iron sideloaded onto the LEFT
    /// lane (from the north), copper onto the RIGHT lane (from the south)
    /// of an east-running belt at (1,1)→(2,1).
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
        w.place(4, 3, Item::MediumElectricPole, Direction::None, None);
        w_extra(&mut w);
        w
    }

    #[test]
    fn test_pickup_prefers_near_lane() {
        // Neither lane runs dry, so each hand takes only its near lane: the
        // belt's left (iron) in-line at (3,1), the lane on its own side
        // (copper) perpendicular at (2,2).
        for (ins, sink, dir, near) in [
            ((3, 1), (4, 1), Direction::East, Item::IronPlate),
            ((2, 2), (2, 3), Direction::South, Item::CopperPlate),
        ] {
            for sink_item in [Item::IronPlate, Item::CopperPlate] {
                let w = make_two_lane_belt_world(|w| {
                    w.place(ins.0, ins.1, Item::Inserter, dir, None);
                    w.place(sink.0, sink.1, Item::Sink, dir, Some(sink_item));
                });
                let g = build_graph(&w);
                let (output, _) = calc_throughput(&g);
                assert_eq!(output.len(), 1);
                let expected = if sink_item == near { 0.86 } else { 0.0 };
                assert!(
                    (output[0].achieved - expected).abs() < 1e-9,
                    "inserter {ins:?} sink {sink_item:?}: expected {expected}, got {:?}",
                    output
                );
            }
        }
    }

    #[test]
    fn test_pickup_falls_back_to_other_lane() {
        // Only the far lane carries items, so the hand reaches across for
        // the whole 0.86.
        let mut w = World::empty(5, 4);
        w.place(1, 0, Item::Source, Direction::South, Some(Item::IronPlate));
        // A bare belt stub behind (1,1) keeps it straight (two feeders → the
        // side source sideloads onto the LEFT lane instead of curving).
        w.place(0, 1, Item::TransportBelt, Direction::East, None);
        w.place(1, 1, Item::TransportBelt, Direction::East, None);
        w.place(2, 1, Item::TransportBelt, Direction::East, None);
        w.place(2, 2, Item::Inserter, Direction::South, None);
        w.place(2, 3, Item::Sink, Direction::South, Some(Item::IronPlate));
        w.place(4, 3, Item::MediumElectricPole, Direction::None, None);

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

    /// Most a factory could deliver: the max flow from its sources to its
    /// sinks with every node passing at most its own rate.
    fn max_flow(g: &FactoryGraph) -> f64 {
        let n = g.node_count();
        let (source, sink) = (2 * n, 2 * n + 1);
        let mut adj = vec![Vec::new(); 2 * n + 2];
        let mut cap: HashMap<(usize, usize), f64> = HashMap::new();
        let mut edge = |a: usize, b: usize, c: f64| {
            adj[a].push(b);
            adj[b].push(a);
            *cap.entry((a, b)).or_insert(0.0) += c;
            cap.entry((b, a)).or_insert(0.0);
        };
        for (i, node) in g.nodes.iter().enumerate() {
            let kind = node.entity_kind;
            match kind {
                Item::Source => edge(source, 2 * i + 1, f64::INFINITY),
                Item::Sink => edge(2 * i, sink, f64::INFINITY),
                _ if kind.is_lane_aware() => edge(2 * i, 2 * i + 1, kind.flow_rate() / 2.0),
                _ => edge(2 * i, 2 * i + 1, kind.flow_rate()),
            }
            for &j in &g.successors[i] {
                edge(2 * i + 1, 2 * j, f64::INFINITY);
            }
        }
        let mut flow = 0.0;
        loop {
            let mut prev = vec![usize::MAX; 2 * n + 2];
            let mut queue = VecDeque::from([source]);
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if prev[v] == usize::MAX && cap[&(u, v)] > 1e-12 {
                        prev[v] = u;
                        queue.push_back(v);
                    }
                }
            }
            if prev[sink] == usize::MAX {
                return flow;
            }
            let mut path = vec![sink];
            while path[path.len() - 1] != source {
                path.push(prev[path[path.len() - 1]]);
            }
            let push = path
                .windows(2)
                .map(|w| cap[&(w[1], w[0])])
                .fold(f64::INFINITY, f64::min);
            if push.is_infinite() {
                return push;
            }
            for w in path.windows(2) {
                *cap.get_mut(&(w[1], w[0])).unwrap() -= push;
                *cap.get_mut(&(w[0], w[1])).unwrap() += push;
            }
            flow += push;
        }
    }

    #[test]
    fn test_loops_never_beat_max_flow() {
        // RL maximises throughput, so no loop may deliver more than the
        // belts could carry from the sources at all. Scramble the balancers
        // (loops everywhere) and hold every build to its max flow.
        let specs =
            crate::textual::parse_many(include_str!("../tests/factories/balancers.yaml")).unwrap();
        let mut rng = Rng::seeded(0);
        let mut looped = 0;
        for _ in 0..2000 {
            let mut w = specs[rng.choice_index(specs.len())].world.clone();
            for _ in 0..rng.randint(1, 6) {
                let (x, y) = loop {
                    let x = rng.choice_index(w.width());
                    let y = rng.choice_index(w.height());
                    if w.entity_at(x, y).is_some() {
                        break (x, y);
                    }
                };
                match rng.randrange(5) {
                    0..=2 => w.set(x, y, Channel::Direction, rng.randint(1, 4)),
                    3 => w.set(x, y, Channel::Misc, rng.randint(0, 2)),
                    _ => w.set(x, y, Channel::Entities, 0),
                }
            }
            let g = build_graph(&w);
            let sources: Vec<usize> = (0..g.node_count())
                .filter(|&i| g.nodes[i].entity_kind == Item::Source)
                .collect();
            looped += usize::from(!loops(&g, &sources).0.is_empty());
            let delivered: f64 = calc_throughput(&g).0.iter().map(|d| d.achieved).sum();
            assert!(
                delivered <= max_flow(&g) + 1e-9,
                "{delivered} > {}",
                max_flow(&g)
            );
        }
        assert!(looped > 500, "only {looped} scrambled builds kept a loop");
    }
}
