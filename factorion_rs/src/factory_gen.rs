//! A byte-for-byte port of Python's `build_factory` (factorion.py).
//!
//! After the single-RNG refactor, the Python generator draws every bit of
//! layout randomness from CPython's `random` module. This module pairs the
//! faithful [`crate::pyrandom`] generator with a line-for-line port of the
//! placement logic so that, for the same `(size, kind, seed)`, the Rust
//! factory is identical to Python's — the same entities, directions, items
//! and footprint in every tile. The parity is fuzz-tested over thousands of
//! seeds in `tests/test_build_factory_parity.py`.
//!
//! The port mirrors Python's control flow exactly, including its quirks
//! (e.g. the rejection-sampling `count` that returns `None` even when the
//! *last* attempt succeeded). Where Python iterates a list built by a
//! comprehension and then draws `random.choice`/`shuffle`/`sample` over it,
//! the Rust must build the *same list in the same order* so the draws index
//! the same elements.

use crate::graph::build_graph;
use crate::pyrandom::PyRandom;
use crate::throughput::{calc_throughput, factory_score};
use crate::types::{all_items, Channel, Direction, Item};
use crate::world::World;
use std::collections::{HashMap, HashSet, VecDeque};

/// The lesson kinds, matching `factorion.py::LessonKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LessonKind {
    MoveOneItem = 0,
    SplitterSplit = 3,
    SplitterMerge = 4,
    Assemble1In1Out = 5,
    MoveViaUgBelt = 6,
    Assemble2In1Out = 7,
    FromBlueprint = 8,
    MoveOneItemChaos = 9,
}

impl LessonKind {
    pub fn from_i64(v: i64) -> Option<Self> {
        match v {
            0 => Some(LessonKind::MoveOneItem),
            3 => Some(LessonKind::SplitterSplit),
            4 => Some(LessonKind::SplitterMerge),
            5 => Some(LessonKind::Assemble1In1Out),
            6 => Some(LessonKind::MoveViaUgBelt),
            7 => Some(LessonKind::Assemble2In1Out),
            8 => Some(LessonKind::FromBlueprint),
            9 => Some(LessonKind::MoveOneItemChaos),
            _ => None,
        }
    }
}

/// The non-NONE directions in `Direction` enum order — the exact list
/// `[d for d in Direction if d != Direction.NONE]` that every
/// `random.choice(dirs)` indexes into.
const DIRS: [Direction; 4] = [
    Direction::North,
    Direction::East,
    Direction::South,
    Direction::West,
];

/// Neighbour deltas in `list(DIR_TO_DELTA.values())` order: N, E, S, W.
/// BFS expands and records parents in this order, which fixes the order of
/// the enumerated paths (and therefore the `random.shuffle` that consumes
/// them).
const BFS_DELTAS: [(i64, i64); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

/// Inverse of `BFS_DELTAS`: a one-step (dx, dy) → belt direction, matching
/// Python's `DELTA_TO_DIR`. Returns `None` for non-unit deltas.
fn delta_to_dir(dx: i64, dy: i64) -> Option<Direction> {
    match (dx, dy) {
        (0, -1) => Some(Direction::North),
        (1, 0) => Some(Direction::East),
        (0, 1) => Some(Direction::South),
        (-1, 0) => Some(Direction::West),
        _ => None,
    }
}

/// The `random.choice` pool for a factory's transported item:
/// `[v.value for k, v in items.items() if v.name != "empty"]`, i.e. every
/// `Item` in `all_items()` order (the synthetic "empty" sentinel is value 0
/// and excluded). Returned as raw channel values.
fn item_pool() -> Vec<i64> {
    all_items().iter().map(|&i| i as i64).collect()
}

/// Throughput score of a fully-placed world — the Rust-native equivalent of
/// `factorion_rs.simulate_throughput(world)[0]` that Python calls.
fn world_throughput(world: &World) -> f64 {
    let graph = build_graph(world);
    let (deliveries, _) = calc_throughput(&graph);
    factory_score(&deliveries)
}

type Cell = (i64, i64);
type Belt = (i64, i64, Direction);
/// BFS output: distance to each reachable cell, and every equal-distance
/// predecessor of each (so all shortest paths can be enumerated).
type BfsResult = (HashMap<Cell, usize>, HashMap<Cell, Vec<Cell>>);

/// BFS from `start` to `end` avoiding `blocked`, returning `(dist, parents)`
/// where `parents[cell]` holds *all* equal-distance predecessors (so all
/// shortest paths can be enumerated). `None` if `end` is unreachable.
/// A faithful port of `factorion.py::_bfs_shortest`.
fn bfs_shortest(size: i64, start: Cell, end: Cell, blocked: &HashSet<Cell>) -> Option<BfsResult> {
    let in_bounds = |c: Cell| 0 <= c.0 && c.0 < size && 0 <= c.1 && c.1 < size;
    if !in_bounds(start) || !in_bounds(end) {
        return None;
    }
    if blocked.contains(&start) || blocked.contains(&end) {
        return None;
    }

    let mut dist: HashMap<Cell, usize> = HashMap::new();
    let mut parents: HashMap<Cell, Vec<Cell>> = HashMap::new();
    let mut q: VecDeque<Cell> = VecDeque::new();
    dist.insert(start, 0);
    q.push_back(start);

    while let Some((r, c)) = q.pop_front() {
        if (r, c) == end {
            break;
        }
        let nd = dist[&(r, c)] + 1;
        for (dr, dc) in BFS_DELTAS {
            let nr = r + dr;
            let nc = c + dc;
            if !(0 <= nr && nr < size && 0 <= nc && nc < size) {
                continue;
            }
            let ncell = (nr, nc);
            if blocked.contains(&ncell) {
                continue;
            }
            match dist.get(&ncell) {
                None => {
                    dist.insert(ncell, nd);
                    parents.entry(ncell).or_default().push((r, c));
                    q.push_back(ncell);
                }
                Some(&cur) if cur == nd => {
                    parents.entry(ncell).or_default().push((r, c));
                }
                _ => {}
            }
        }
    }

    if !dist.contains_key(&end) {
        return None;
    }
    Some((dist, parents))
}

/// Convert a cell run into belt placements, last belt taking `end_dir`.
/// Port of `factorion.py::_path_to_belts`.
fn path_to_belts(path: &[Cell], end_dir: Direction) -> Vec<Belt> {
    let mut belts: Vec<Belt> = Vec::new();
    for w in path.windows(2) {
        let (r1, c1) = w[0];
        let (r2, c2) = w[1];
        if let Some(d) = delta_to_dir(r2 - r1, c2 - c1) {
            belts.push((r1, c1, d));
        }
    }
    if let Some(&(lx, ly)) = path.last() {
        belts.push((lx, ly, end_dir));
    }
    belts
}

/// A single shortest belt path from `start` to `end`, or `None`.
/// Port of `factorion.py::find_belt_path`.
fn find_belt_path(
    size: i64,
    start: Cell,
    end: Cell,
    end_dir: Direction,
    blocked: &HashSet<Cell>,
) -> Option<Vec<Belt>> {
    let (_dist, parents) = bfs_shortest(size, start, end, blocked)?;
    let mut path: Vec<Cell> = Vec::new();
    let mut cell = end;
    while cell != start {
        path.push(cell);
        cell = parents[&cell][0];
    }
    path.push(start);
    path.reverse();
    Some(path_to_belts(&path, end_dir))
}

/// The `(x, y)` cells of a belt run (dropping directions).
fn belt_cells(belts: &[Belt]) -> Vec<Cell> {
    belts.iter().map(|&(x, y, _)| (x, y)).collect()
}

/// The `(x, y)` cells of a belt run as a set.
#[allow(dead_code)] // consumed by the splitter/assembler lesson ports
fn belt_cell_set(belts: &[Belt]) -> HashSet<Cell> {
    belts.iter().map(|&(x, y, _)| (x, y)).collect()
}

/// All shortest belt paths from a source's output cell to a sink's input
/// cell, given their positions and facings. Port of
/// `factorion.py::find_belt_paths_with_source_sink_orient`.
fn find_belt_paths_with_source_sink_orient(
    size: i64,
    src: Cell,
    src_dir: Direction,
    sink: Cell,
    sink_dir: Direction,
) -> Vec<Vec<Belt>> {
    if src_dir == Direction::None || sink_dir == Direction::None {
        return vec![];
    }
    let (dr_s, dc_s) = src_dir.delta();
    let start = (src.0 + dr_s, src.1 + dc_s);
    let (dr_k, dc_k) = sink_dir.delta();
    let end = (sink.0 - dr_k, sink.1 - dc_k);

    if start == src || start == sink || end == src || end == sink {
        return vec![];
    }

    let mut blocked: HashSet<Cell> = HashSet::new();
    blocked.insert(src);
    blocked.insert(sink);
    let (_dist, parents) = match bfs_shortest(size, start, end, &blocked) {
        Some(v) => v,
        None => return vec![],
    };

    let mut all_paths: Vec<Vec<Belt>> = Vec::new();
    // Recursive backtrack over `parents`, in the same order Python's nested
    // function visits them.
    fn backtrack(
        cell: Cell,
        start: Cell,
        sink_dir: Direction,
        parents: &HashMap<Cell, Vec<Cell>>,
        rev_path: &mut Vec<Cell>,
        all_paths: &mut Vec<Vec<Belt>>,
    ) {
        if cell == start {
            let mut path = vec![start];
            path.extend(rev_path.iter().rev().copied());
            all_paths.push(path_to_belts(&path, sink_dir));
            return;
        }
        if let Some(ps) = parents.get(&cell) {
            for &p in ps {
                rev_path.push(cell);
                backtrack(p, start, sink_dir, parents, rev_path, all_paths);
                rev_path.pop();
            }
        }
    }
    let mut rev_path: Vec<Cell> = Vec::new();
    backtrack(
        end,
        start,
        sink_dir,
        &parents,
        &mut rev_path,
        &mut all_paths,
    );
    all_paths
}

/// The result of a successful `build_factory`: a complete world plus the
/// blanking bookkeeping Python's `Factory` carries.
pub struct BuiltFactory {
    pub world: World,
    pub total_entities: usize,
    pub protected_positions: Vec<(usize, usize)>,
}

/// Lay a run of belts into `world`.
fn place_belts(world: &mut World, belts: &[Belt]) {
    for &(x, y, d) in belts {
        world.set(
            x as usize,
            y as usize,
            Channel::Entities,
            Item::TransportBelt as i64,
        );
        world.set(x as usize, y as usize, Channel::Direction, d as i64);
    }
}

/// Port of `build_factory(size, kind, seed=..., random_item, max_entities)`.
/// Always seeds from `seed` (the `seed is not None` path). Returns `None`
/// when rejection sampling is exhausted, matching Python.
pub fn build_factory(
    size: usize,
    kind: LessonKind,
    seed: u64,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let mut rng = PyRandom::seeded(seed);
    match kind {
        LessonKind::MoveOneItem => build_move_one_item(size, &mut rng, random_item, max_entities),
        LessonKind::MoveOneItemChaos => {
            build_move_one_item_chaos(size, &mut rng, random_item, max_entities)
        }
        // Remaining kinds are ported in subsequent commits.
        _ => None,
    }
}

/// Place a source or sink marker (entity + facing + carried item).
fn place_marker(world: &mut World, pos: Cell, ent: Item, dir: Direction, item_value: i64) {
    let (x, y) = (pos.0 as usize, pos.1 as usize);
    world.set(x, y, Channel::Entities, ent as i64);
    world.set(x, y, Channel::Direction, dir as i64);
    world.set(x, y, Channel::Items, item_value);
}

fn build_move_one_item(
    size: usize,
    rng: &mut PyRandom,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let pool = item_pool();
    let mut count = (500).max(size * size * 4);
    while count > 0 {
        count -= 1;
        let pos1 = rng.randrange((size * size) as u64) as i64;
        let pos2 = rng.randrange((size * size) as u64) as i64;
        if pos1 == pos2 {
            continue;
        }
        // divmod(pos, W) with W = size
        let source_wh = (pos1 / s, pos1 % s);
        let sink_wh = (pos2 / s, pos2 % s);
        let source_dir = DIRS[rng.choice_index(4)];
        let sink_dir = DIRS[rng.choice_index(4)];
        let item_value = if random_item {
            pool[rng.choice_index(pool.len())]
        } else {
            Item::ElectronicCircuit as i64
        };

        let mut world = World::empty(size, size);
        let (sx, sy) = (source_wh.0 as usize, source_wh.1 as usize);
        let (kx, ky) = (sink_wh.0 as usize, sink_wh.1 as usize);
        world.set(sx, sy, Channel::Entities, Item::Source as i64);
        world.set(kx, ky, Channel::Entities, Item::Sink as i64);
        world.set(sx, sy, Channel::Items, item_value);
        world.set(kx, ky, Channel::Items, item_value);
        world.set(sx, sy, Channel::Direction, source_dir as i64);
        world.set(kx, ky, Channel::Direction, sink_dir as i64);

        let mut paths =
            find_belt_paths_with_source_sink_orient(s, source_wh, source_dir, sink_wh, sink_dir);
        paths.retain(|p| (p.len() as f64) <= max_entities);

        if paths.is_empty() {
            continue;
        }

        rng.shuffle(&mut paths);
        let mut chosen: Option<Vec<Belt>> = None;
        for candidate in &paths {
            let mut trial = world.clone();
            place_belts(&mut trial, candidate);
            if world_throughput(&trial) > 0.0 {
                chosen = Some(candidate.clone());
                world = trial;
                break;
            }
        }

        let chosen = match chosen {
            Some(c) => c,
            None => continue,
        };

        let total_entities = chosen.len();
        return finish(world, total_entities, vec![], count);
    }
    None
}

fn build_move_one_item_chaos(
    size: usize,
    rng: &mut PyRandom,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let pool = item_pool();
    let mut count = (500).max(size * size * 4);
    while count > 0 {
        count -= 1;
        let pos1 = rng.randrange((size * size) as u64) as i64;
        let pos2 = rng.randrange((size * size) as u64) as i64;
        if pos1 == pos2 {
            continue;
        }
        let source_wh = (pos1 / s, pos1 % s);
        let sink_wh = (pos2 / s, pos2 % s);
        let source_dir = DIRS[rng.choice_index(4)];
        let sink_dir = DIRS[rng.choice_index(4)];
        let item_value = if random_item {
            pool[rng.choice_index(pool.len())]
        } else {
            Item::ElectronicCircuit as i64
        };

        let ds = source_dir.delta();
        let dk = sink_dir.delta();
        let start = (source_wh.0 + ds.0, source_wh.1 + ds.1);
        let end = (sink_wh.0 - dk.0, sink_wh.1 - dk.1);

        let fixed: HashSet<Cell> = [source_wh, sink_wh].into_iter().collect();
        if !(0 <= start.0 && start.0 < s && 0 <= start.1 && start.1 < s) {
            continue;
        }
        if !(0 <= end.0 && end.0 < s && 0 <= end.1 && end.1 < s) {
            continue;
        }
        if fixed.contains(&start) || fixed.contains(&end) || start == end {
            continue;
        }

        // mid candidates in `for x in range(W) for y in range(H)` order.
        let mut reserved = fixed.clone();
        reserved.insert(start);
        reserved.insert(end);
        let mut mid_candidates: Vec<Cell> = Vec::new();
        for x in 0..s {
            for y in 0..s {
                if !reserved.contains(&(x, y)) {
                    mid_candidates.push((x, y));
                }
            }
        }
        if mid_candidates.is_empty() {
            continue;
        }
        let mid = mid_candidates[rng.choice_index(mid_candidates.len())];

        // Segment A: source output → intermediate (protected later).
        let mut blocked_a = fixed.clone();
        blocked_a.insert(end);
        let belts_a = match find_belt_path(s, start, mid, source_dir, &blocked_a) {
            Some(b) => b,
            None => continue,
        };
        let cells_a = belt_cells(&belts_a);

        // Segment B: intermediate → sink input (removable).
        let mut blocked_b = fixed.clone();
        for &c in &cells_a {
            if c != mid {
                blocked_b.insert(c);
            }
        }
        let belts_b = match find_belt_path(s, mid, end, sink_dir, &blocked_b) {
            Some(b) => b,
            None => continue,
        };
        let cells_b = belt_cells(&belts_b);

        // Stitch (mid shared) and recompute directions over the whole chain.
        let mut full_cells = cells_a.clone();
        full_cells.extend_from_slice(&cells_b[1..]);
        let full_belts = path_to_belts(&full_cells, sink_dir);
        if (full_belts.len() as f64) > max_entities {
            continue;
        }

        let mut world = World::empty(size, size);
        place_marker(&mut world, source_wh, Item::Source, source_dir, item_value);
        place_marker(&mut world, sink_wh, Item::Sink, sink_dir, item_value);
        place_belts(&mut world, &full_belts);

        if world_throughput(&world) <= 0.0 {
            continue;
        }

        let total_entities = full_belts.len();
        // Protect the source→intermediate stub.
        let protected: Vec<(usize, usize)> = cells_a
            .iter()
            .map(|&(x, y)| (x as usize, y as usize))
            .collect();
        return finish(world, total_entities, protected, count);
    }
    None
}

/// Mirror Python's post-loop `if count == 0: return None`: a factory found
/// on the iteration that drove `count` to 0 is still discarded.
fn finish(
    world: World,
    total_entities: usize,
    protected_positions: Vec<(usize, usize)>,
    count: usize,
) -> Option<BuiltFactory> {
    if count == 0 {
        return None;
    }
    Some(BuiltFactory {
        world,
        total_entities,
        protected_positions,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn test_move_one_item_smoke() {
        // A handful of seeds should all produce positive-throughput factories.
        let mut built = 0;
        for seed in 0..50u64 {
            if let Some(f) = build_factory(10, LessonKind::MoveOneItem, seed, true, f64::INFINITY) {
                assert!(world_throughput(&f.world) > 0.0, "seed={seed}");
                assert!(f.total_entities >= 1);
                built += 1;
            }
        }
        assert!(built > 40, "most seeds should build, got {built}");
    }

    #[test]
    fn test_delta_to_dir_roundtrip() {
        for d in DIRS {
            let (dx, dy) = d.delta();
            assert_eq!(delta_to_dir(dx, dy), Some(d));
        }
        assert_eq!(delta_to_dir(2, 0), None);
    }
}
