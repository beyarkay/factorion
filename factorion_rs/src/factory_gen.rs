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

use crate::blueprints::lesson_blueprints;
use crate::entities::entity_tiles;
use crate::graph::build_graph;
use crate::pyrandom::PyRandom;
use crate::throughput::{calc_throughput, factory_score};
use crate::types::{all_items, all_recipes, Channel, Direction, Item, Misc, Recipe, NUM_CHANNELS};
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

/// `[(name, r) for name, r in recipes.items() if len(consumes)==1 and len(produces)==1]`
/// in `all_recipes()` order — the `random.choice` pool for 1-in-1-out lessons.
fn one_in_one_out() -> Vec<(Item, Recipe)> {
    all_recipes()
        .into_iter()
        .filter(|(_, r)| r.consumes.len() == 1 && r.produces.len() == 1)
        .collect()
}

/// `[(name, r) for ... if len(consumes)==2 and len(produces)==1]`.
fn two_in_one_out() -> Vec<(Item, Recipe)> {
    all_recipes()
        .into_iter()
        .filter(|(_, r)| r.consumes.len() == 2 && r.produces.len() == 1)
        .collect()
}

/// The 12 non-corner perimeter slots around a 3×3 assembler anchored at
/// `(ax, ay)`, as `(offset_x, offset_y, input_inserter_dir, output_inserter_dir)`.
/// Same order as Python's `perim_slots` so `random.sample` indexes match.
const PERIM_SLOTS: [(i64, i64, Direction, Direction); 12] = [
    // North side (ddy = -1)
    (0, -1, Direction::South, Direction::North),
    (1, -1, Direction::South, Direction::North),
    (2, -1, Direction::South, Direction::North),
    // South side (ddy = 3)
    (0, 3, Direction::North, Direction::South),
    (1, 3, Direction::North, Direction::South),
    (2, 3, Direction::North, Direction::South),
    // West side (ddx = -1)
    (-1, 0, Direction::East, Direction::West),
    (-1, 1, Direction::East, Direction::West),
    (-1, 2, Direction::East, Direction::West),
    // East side (ddx = 3)
    (3, 0, Direction::West, Direction::East),
    (3, 1, Direction::West, Direction::East),
    (3, 2, Direction::West, Direction::East),
];

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
        LessonKind::SplitterSplit => {
            build_splitter_split(size, &mut rng, random_item, max_entities)
        }
        LessonKind::SplitterMerge => {
            build_splitter_merge(size, &mut rng, random_item, max_entities)
        }
        LessonKind::Assemble1In1Out => build_assemble_1in1out(size, &mut rng, max_entities),
        LessonKind::MoveViaUgBelt => {
            build_move_via_ug_belt(size, &mut rng, random_item, max_entities)
        }
        LessonKind::Assemble2In1Out => build_assemble_2in1out(size, &mut rng, max_entities),
        LessonKind::FromBlueprint => build_from_blueprint(size, &mut rng, max_entities),
    }
}

// ── FROM_BLUEPRINT helpers ───────────────────────────────────────────────────

/// A decoded blueprint as a mutable `(W, H, C)` buffer, used for the
/// pre-translation augmentations (recipe substitution, flips).
#[derive(Clone)]
struct Bp {
    w: usize,
    h: usize,
    data: Vec<i64>,
}

impl Bp {
    fn idx(&self, x: usize, y: usize, c: usize) -> usize {
        (x * self.h + y) * (NUM_CHANNELS) + c
    }
    fn get(&self, x: usize, y: usize, c: Channel) -> i64 {
        self.data[self.idx(x, y, c.index())]
    }
    fn set(&mut self, x: usize, y: usize, c: Channel, v: i64) {
        let i = self.idx(x, y, c.index());
        self.data[i] = v;
    }
}

/// `factorion.py::_is_gears_factory`: every source carries iron_plate, every
/// sink iron_gear_wheel, every assembler tile iron_gear_wheel (and at least
/// one of each exists).
fn is_gears_factory(bp: &Bp) -> bool {
    let plate = Item::IronPlate as i64;
    let gear = Item::IronGearWheel as i64;
    let (mut any_src, mut any_snk, mut any_asm) = (false, false, false);
    let mut ok = true;
    for x in 0..bp.w {
        for y in 0..bp.h {
            let ent = bp.get(x, y, Channel::Entities);
            let itm = bp.get(x, y, Channel::Items);
            if ent == Item::Source as i64 {
                any_src = true;
                ok &= itm == plate;
            } else if ent == Item::Sink as i64 {
                any_snk = true;
                ok &= itm == gear;
            } else if ent == Item::AssemblingMachine1 as i64 {
                any_asm = true;
                ok &= itm == gear;
            }
        }
    }
    any_src && any_snk && any_asm && ok
}

/// `factorion.py::_substitute_gears_recipe`: swap the iron_plate → gear ITEMS
/// triple for a randomly chosen 1-in-1-out recipe.
fn substitute_gears_recipe(bp: &Bp, rng: &mut PyRandom) -> Bp {
    let recipes = one_in_one_out();
    if recipes.is_empty() {
        return bp.clone();
    }
    let (recipe_key, recipe) = recipes[rng.choice_index(recipes.len())].clone();
    let new_input_id = recipe.consumes.first().0 as i64;
    let new_output_id = recipe_key as i64; // produces == key for 1-in-1-out
    let plate = Item::IronPlate as i64;
    let gear = Item::IronGearWheel as i64;
    let mut out = bp.clone();
    for x in 0..out.w {
        for y in 0..out.h {
            let itm = out.get(x, y, Channel::Items);
            if itm == plate {
                out.set(x, y, Channel::Items, new_input_id);
            } else if itm == gear {
                out.set(x, y, Channel::Items, new_output_id);
            }
        }
    }
    out
}

/// `factorion.py::_flip_world`: mirror along the W axis (`horizontal`) or H
/// axis, remapping flow directions so flow is preserved.
fn flip_world(bp: &Bp, horizontal: bool) -> Bp {
    let mut out = Bp {
        w: bp.w,
        h: bp.h,
        data: vec![0; bp.data.len()],
    };
    for x in 0..bp.w {
        for y in 0..bp.h {
            let (sx, sy) = if horizontal {
                (bp.w - 1 - x, y)
            } else {
                (x, bp.h - 1 - y)
            };
            for c in 0..NUM_CHANNELS {
                let v = bp.data[bp.idx(sx, sy, c)];
                let i = out.idx(x, y, c);
                out.data[i] = v;
            }
            // Remap direction for the flow that was mirrored.
            let d = out.get(x, y, Channel::Direction);
            let nd = if horizontal {
                match Direction::from_i64(d) {
                    Direction::East => Direction::West as i64,
                    Direction::West => Direction::East as i64,
                    _ => d,
                }
            } else {
                match Direction::from_i64(d) {
                    Direction::North => Direction::South as i64,
                    Direction::South => Direction::North as i64,
                    _ => d,
                }
            };
            out.set(x, y, Channel::Direction, nd);
        }
    }
    out
}

/// `factorion.py::_extend_belt_chains`: push each source/sink marker backward
/// into the empty run behind it, filling the vacated tiles with belts. The
/// marker order is shuffled and each extension draws `randint(0, K)`.
fn extend_belt_chains(world: &mut World, rng: &mut PyRandom) {
    let w = world.width() as i64;
    let h = world.height() as i64;
    let src_id = Item::Source as i64;
    let snk_id = Item::Sink as i64;
    let tb_id = Item::TransportBelt as i64;

    let mut markers: Vec<Cell> = Vec::new();
    for x in 0..w {
        for y in 0..h {
            let e = world.get(x as usize, y as usize, Channel::Entities);
            if e == src_id || e == snk_id {
                markers.push((x, y));
            }
        }
    }
    rng.shuffle(&mut markers);

    for (mx, my) in markers {
        let marker_id_val = world.get(mx as usize, my as usize, Channel::Entities);
        if marker_id_val != src_id && marker_id_val != snk_id {
            continue; // cleared by a prior extension
        }
        let marker_dir =
            Direction::from_i64(world.get(mx as usize, my as usize, Channel::Direction));
        if marker_dir == Direction::None {
            continue;
        }
        let marker_dir_val = marker_dir as i64;
        let scan_dir = if marker_id_val == src_id {
            marker_dir
        } else {
            marker_dir.opposite()
        };
        let (dx, dy) = scan_dir.delta();

        let (belt_x, belt_y) = (mx + dx, my + dy);
        if !(0 <= belt_x && belt_x < w && 0 <= belt_y && belt_y < h) {
            continue;
        }
        if world.get(belt_x as usize, belt_y as usize, Channel::Entities) != tb_id {
            continue;
        }
        if world.get(belt_x as usize, belt_y as usize, Channel::Direction) != marker_dir_val {
            continue;
        }

        let mut k = 0i64;
        let (mut ex, mut ey) = (mx - dx, my - dy);
        while 0 <= ex && ex < w && 0 <= ey && ey < h {
            if world.get(ex as usize, ey as usize, Channel::Entities) != 0 {
                break;
            }
            k += 1;
            ex -= dx;
            ey -= dy;
        }
        if k == 0 {
            continue;
        }

        let j = rng.randint(0, k);
        if j == 0 {
            continue;
        }

        let marker_item = world.get(mx as usize, my as usize, Channel::Items);
        let new_mx = mx - dx * j;
        let new_my = my - dy * j;

        world.set(mx as usize, my as usize, Channel::Entities, 0);
        world.set(
            mx as usize,
            my as usize,
            Channel::Direction,
            Direction::None as i64,
        );
        world.set(mx as usize, my as usize, Channel::Items, 0);

        world.set(
            new_mx as usize,
            new_my as usize,
            Channel::Entities,
            marker_id_val,
        );
        world.set(
            new_mx as usize,
            new_my as usize,
            Channel::Direction,
            marker_dir_val,
        );
        world.set(
            new_mx as usize,
            new_my as usize,
            Channel::Items,
            marker_item,
        );

        for jj in 1..=j {
            let bx = new_mx + dx * jj;
            let by = new_my + dy * jj;
            world.set(bx as usize, by as usize, Channel::Entities, tb_id);
            world.set(bx as usize, by as usize, Channel::Direction, marker_dir_val);
        }
    }
}

/// `factorion.py::_count_removable_entity_units`: count non-source/sink/empty
/// entity units (multi-tile counted once at the anchor), x-outer/y-inner.
fn count_removable_entity_units(world: &World) -> usize {
    let w = world.width();
    let h = world.height();
    let mut secondary: HashSet<Cell> = HashSet::new();
    let mut count = 0usize;
    for x in 0..w {
        for y in 0..h {
            if secondary.contains(&(x as i64, y as i64)) {
                continue;
            }
            let ev = world.get(x, y, Channel::Entities);
            if ev == 0 || ev == Item::Source as i64 || ev == Item::Sink as i64 {
                continue;
            }
            let ent = match Item::from_i64(ev) {
                Some(e) => e,
                None => continue,
            };
            count += 1;
            let (ew, eh) = ent.size();
            if ew > 1 || eh > 1 {
                let d = world.direction_at(x, y);
                if let Some(tiles) = entity_tiles(x, y, d, ew, eh) {
                    for t in tiles {
                        if (t.x, t.y) != (x as i64, y as i64) {
                            secondary.insert((t.x, t.y));
                        }
                    }
                }
            }
        }
    }
    count
}

#[allow(clippy::too_many_lines)]
fn build_from_blueprint(
    size: usize,
    rng: &mut PyRandom,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let blueprints = lesson_blueprints();
    // Python raises if there are no blueprints; the embedded table is non-empty.
    let mut count = (500).max(size * size * 2);

    while count > 0 {
        count -= 1;
        let bp_src = &blueprints[rng.choice_index(blueprints.len())];
        let w_bp = bp_src.w;
        let h_bp = bp_src.h;
        if w_bp > size || h_bp > size {
            continue;
        }

        let mut decoded = Bp {
            w: w_bp,
            h: h_bp,
            data: bp_src.data.to_vec(),
        };

        if is_gears_factory(&decoded) {
            decoded = substitute_gears_recipe(&decoded, rng);
        }
        if rng.random() < 0.5 {
            decoded = flip_world(&decoded, true); // horizontal (W axis)
        }
        if rng.random() < 0.5 {
            decoded = flip_world(&decoded, false); // vertical (H axis)
        }

        let ox = rng.randint(0, (size - w_bp) as i64) as usize;
        let oy = rng.randint(0, (size - h_bp) as i64) as usize;

        let mut world = World::empty(size, size);
        for x in 0..w_bp {
            for y in 0..h_bp {
                for c in 0..NUM_CHANNELS {
                    let v = decoded.data[decoded.idx(x, y, c)];
                    world.set(ox + x, oy + y, channel_of(c), v);
                }
            }
        }

        extend_belt_chains(&mut world, rng);

        if world_throughput(&world) <= 0.0 {
            continue;
        }

        let total_entities = count_removable_entity_units(&world);
        if total_entities == 0 {
            continue;
        }
        if (total_entities as f64) > max_entities {
            continue;
        }

        return finish(world, total_entities, vec![], count);
    }
    None
}

/// Channel for a raw channel index (the inverse of `Channel::index`).
fn channel_of(c: usize) -> Channel {
    match c {
        0 => Channel::Entities,
        1 => Channel::Direction,
        2 => Channel::Items,
        3 => Channel::Misc,
        _ => Channel::Footprint,
    }
}

/// `(x, y)` is within a `size x size` grid.
fn in_grid(c: Cell, size: i64) -> bool {
    0 <= c.0 && c.0 < size && 0 <= c.1 && c.1 < size
}

/// Splitter tiles for `(sx, sy, dir)`, as `Cell`s, or `None` for an invalid
/// direction. Mirrors `factorion_rs.py_entity_tiles(sx, sy, dir, 2, 1)`.
fn splitter_tiles(sx: i64, sy: i64, dir: Direction) -> Option<Vec<Cell>> {
    entity_tiles(sx as usize, sy as usize, dir, 2, 1)
        .map(|tiles| tiles.into_iter().map(|p| (p.x, p.y)).collect())
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

#[allow(clippy::too_many_lines)]
fn build_splitter_split(
    size: usize,
    rng: &mut PyRandom,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let pool = item_pool();
    let mut count = (500).max(size * size * 10);

    // NOTE: unlike MOVE_ONE_ITEM, the item is drawn ONCE before the loop.
    let item_value = if random_item {
        pool[rng.choice_index(pool.len())]
    } else {
        Item::ElectronicCircuit as i64
    };

    while count > 0 {
        count -= 1;
        let splitter_dir = DIRS[rng.choice_index(4)];

        // Pick splitter anchor; both tiles must fit (up to 20 tries).
        let mut tiles: Option<Vec<Cell>> = None;
        for _ in 0..20 {
            let sx = rng.randint(0, s - 1);
            let sy = rng.randint(0, s - 1);
            let t = splitter_tiles(sx, sy, splitter_dir);
            if let Some(ref tt) = t {
                if tt.iter().all(|&c| in_grid(c, s)) {
                    tiles = t;
                    break;
                }
            }
        }
        let tiles = match tiles {
            Some(t) => t,
            None => continue,
        };
        let tile_set: HashSet<Cell> = tiles.iter().copied().collect();
        let dd = splitter_dir.delta();

        let input_cells: Vec<Cell> = tiles
            .iter()
            .map(|&(tx, ty)| (tx - dd.0, ty - dd.1))
            .collect();
        let output_cells: Vec<Cell> = tiles
            .iter()
            .map(|&(tx, ty)| (tx + dd.0, ty + dd.1))
            .collect();
        let all_io: Vec<Cell> = input_cells
            .iter()
            .chain(output_cells.iter())
            .copied()
            .collect();
        if all_io.iter().any(|&c| !in_grid(c, s)) {
            continue;
        }
        if all_io.iter().any(|c| tile_set.contains(c)) {
            continue;
        }

        let reserved: HashSet<Cell> = tile_set
            .iter()
            .copied()
            .chain(all_io.iter().copied())
            .collect();
        let mut available: Vec<Cell> = Vec::new();
        for x in 0..s {
            for y in 0..s {
                if !reserved.contains(&(x, y)) {
                    available.push((x, y));
                }
            }
        }
        if available.len() < 3 {
            continue;
        }

        let chosen = rng.sample(&available, 3);
        let source_pos = chosen[0];
        let sink1_pos = chosen[1];
        let sink2_pos = chosen[2];

        let source_dir = DIRS[rng.choice_index(4)];
        let sink1_dir = DIRS[rng.choice_index(4)];
        let sink2_dir = DIRS[rng.choice_index(4)];

        let ds = source_dir.delta();
        let dk1 = sink1_dir.delta();
        let dk2 = sink2_dir.delta();
        let source_output = (source_pos.0 + ds.0, source_pos.1 + ds.1);
        let sink1_input = (sink1_pos.0 - dk1.0, sink1_pos.1 - dk1.1);
        let sink2_input = (sink2_pos.0 - dk2.0, sink2_pos.1 - dk2.1);

        let conn_cells = [source_output, sink1_input, sink2_input];
        if conn_cells.iter().any(|&c| !in_grid(c, s)) {
            continue;
        }
        let all_fixed: HashSet<Cell> = tile_set
            .iter()
            .copied()
            .chain([source_pos, sink1_pos, sink2_pos])
            .collect();
        let conn_set: HashSet<Cell> = conn_cells.iter().copied().collect();
        if conn_set.len() != conn_cells.len() {
            continue;
        }
        if conn_set.iter().any(|c| all_fixed.contains(c)) {
            continue;
        }

        // Path 1: source output → one of the splitter inputs.
        let blocked_base = &all_fixed;
        let unused_input_buffer_0 = (input_cells[0].0 - dd.0, input_cells[0].1 - dd.1);
        let unused_input_buffer_1 = (input_cells[1].0 - dd.0, input_cells[1].1 - dd.1);

        let mut blocked1: HashSet<Cell> = blocked_base.clone();
        blocked1.extend(output_cells.iter().copied());
        blocked1.extend([
            sink1_input,
            sink2_input,
            input_cells[1],
            unused_input_buffer_1,
        ]);
        let mut path1 = find_belt_path(s, source_output, input_cells[0], splitter_dir, &blocked1);
        if path1.is_none() {
            let mut b1: HashSet<Cell> = blocked_base.clone();
            b1.extend(output_cells.iter().copied());
            b1.extend([
                sink1_input,
                sink2_input,
                input_cells[0],
                unused_input_buffer_0,
            ]);
            path1 = find_belt_path(s, source_output, input_cells[1], splitter_dir, &b1);
            if path1.is_none() {
                continue;
            }
        }
        let path1 = path1?;
        let path1_end = (path1[path1.len() - 1].0, path1[path1.len() - 1].1);
        let (unused_input, unused_buffer) = if path1_end == input_cells[0] {
            (input_cells[1], unused_input_buffer_1)
        } else {
            (input_cells[0], unused_input_buffer_0)
        };
        let path1_cells = belt_cell_set(&path1);

        let unused_block = [unused_input, unused_buffer];
        let sink1_output = (sink1_pos.0 + dk1.0, sink1_pos.1 + dk1.1);
        let sink2_output = (sink2_pos.0 + dk2.0, sink2_pos.1 + dk2.1);
        let sink_buffers = [sink1_output, sink2_output];

        // Path 2 + 3: try both sink assignments.
        let mut found: Option<(Vec<Belt>, Vec<Belt>)> = None;
        for (out_a, out_b, sk_a, sk_a_dir, sk_b, sk_b_dir) in [
            (
                output_cells[0],
                output_cells[1],
                sink1_input,
                sink1_dir,
                sink2_input,
                sink2_dir,
            ),
            (
                output_cells[0],
                output_cells[1],
                sink2_input,
                sink2_dir,
                sink1_input,
                sink1_dir,
            ),
        ] {
            let mut blocked2: HashSet<Cell> = blocked_base.clone();
            blocked2.extend(path1_cells.iter().copied());
            blocked2.extend([sk_b, out_b]);
            blocked2.extend(input_cells.iter().copied());
            blocked2.extend(unused_block);
            blocked2.extend(sink_buffers);
            let p2 = match find_belt_path(s, out_a, sk_a, sk_a_dir, &blocked2) {
                Some(p) => p,
                None => continue,
            };
            let p2_cells = belt_cell_set(&p2);
            let mut blocked3: HashSet<Cell> = blocked_base.clone();
            blocked3.extend(path1_cells.iter().copied());
            blocked3.extend(p2_cells.iter().copied());
            blocked3.insert(out_a);
            blocked3.extend(input_cells.iter().copied());
            blocked3.extend(unused_block);
            blocked3.extend(sink_buffers);
            if let Some(p3) = find_belt_path(s, out_b, sk_b, sk_b_dir, &blocked3) {
                found = Some((p2, p3));
                break;
            }
        }
        let (path2, path3) = match found {
            Some(v) => v,
            None => continue,
        };

        let total_entities = path1.len() + path2.len() + path3.len() + 1;
        if (total_entities as f64) > max_entities {
            continue;
        }

        let mut world = World::empty(size, size);
        place_marker(&mut world, source_pos, Item::Source, source_dir, item_value);
        place_marker(&mut world, sink1_pos, Item::Sink, sink1_dir, item_value);
        place_marker(&mut world, sink2_pos, Item::Sink, sink2_dir, item_value);
        for &(tx, ty) in &tiles {
            world.set(
                tx as usize,
                ty as usize,
                Channel::Entities,
                Item::Splitter as i64,
            );
            world.set(
                tx as usize,
                ty as usize,
                Channel::Direction,
                splitter_dir as i64,
            );
        }
        place_belts(&mut world, &path1);
        place_belts(&mut world, &path2);
        place_belts(&mut world, &path3);

        if world_throughput(&world) <= 0.0 {
            continue;
        }

        let protected: Vec<(usize, usize)> = tiles
            .iter()
            .map(|&(x, y)| (x as usize, y as usize))
            .collect();
        return finish(world, total_entities, protected, count);
    }
    None
}

#[allow(clippy::too_many_lines)]
fn build_splitter_merge(
    size: usize,
    rng: &mut PyRandom,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let pool = item_pool();
    let mut count = (500).max(size * size * 10);

    let item_value = if random_item {
        pool[rng.choice_index(pool.len())]
    } else {
        Item::ElectronicCircuit as i64
    };

    while count > 0 {
        count -= 1;
        let splitter_dir = DIRS[rng.choice_index(4)];

        let mut tiles: Option<Vec<Cell>> = None;
        for _ in 0..20 {
            let sx = rng.randint(0, s - 1);
            let sy = rng.randint(0, s - 1);
            let t = splitter_tiles(sx, sy, splitter_dir);
            if let Some(ref tt) = t {
                if tt.iter().all(|&c| in_grid(c, s)) {
                    tiles = t;
                    break;
                }
            }
        }
        let tiles = match tiles {
            Some(t) => t,
            None => continue,
        };
        let tile_set: HashSet<Cell> = tiles.iter().copied().collect();
        let dd = splitter_dir.delta();

        let input_cells: Vec<Cell> = tiles
            .iter()
            .map(|&(tx, ty)| (tx - dd.0, ty - dd.1))
            .collect();
        let output_cells: Vec<Cell> = tiles
            .iter()
            .map(|&(tx, ty)| (tx + dd.0, ty + dd.1))
            .collect();
        let all_io: Vec<Cell> = input_cells
            .iter()
            .chain(output_cells.iter())
            .copied()
            .collect();
        if all_io.iter().any(|&c| !in_grid(c, s)) {
            continue;
        }
        if all_io.iter().any(|c| tile_set.contains(c)) {
            continue;
        }

        let reserved: HashSet<Cell> = tile_set
            .iter()
            .copied()
            .chain(all_io.iter().copied())
            .collect();
        let mut available: Vec<Cell> = Vec::new();
        for x in 0..s {
            for y in 0..s {
                if !reserved.contains(&(x, y)) {
                    available.push((x, y));
                }
            }
        }
        if available.len() < 3 {
            continue;
        }

        let chosen = rng.sample(&available, 3);
        let source1_pos = chosen[0];
        let source2_pos = chosen[1];
        let sink_pos = chosen[2];

        let source1_dir = DIRS[rng.choice_index(4)];
        let source2_dir = DIRS[rng.choice_index(4)];
        let sink_dir = DIRS[rng.choice_index(4)];

        let ds1 = source1_dir.delta();
        let ds2 = source2_dir.delta();
        let dk = sink_dir.delta();
        let source1_output = (source1_pos.0 + ds1.0, source1_pos.1 + ds1.1);
        let source2_output = (source2_pos.0 + ds2.0, source2_pos.1 + ds2.1);
        let sink_input = (sink_pos.0 - dk.0, sink_pos.1 - dk.1);

        let conn_cells = [source1_output, source2_output, sink_input];
        if conn_cells.iter().any(|&c| !in_grid(c, s)) {
            continue;
        }
        let all_fixed: HashSet<Cell> = tile_set
            .iter()
            .copied()
            .chain([source1_pos, source2_pos, sink_pos])
            .collect();
        let conn_set: HashSet<Cell> = conn_cells.iter().copied().collect();
        if conn_set.len() != conn_cells.len() {
            continue;
        }
        if conn_set.iter().any(|c| all_fixed.contains(c)) {
            continue;
        }

        let blocked_base = &all_fixed;

        // Path 1: source1 output → splitter input 0 (fallback input 1).
        let mut blocked1: HashSet<Cell> = blocked_base.clone();
        blocked1.extend(output_cells.iter().copied());
        blocked1.extend([source2_output, sink_input, input_cells[1]]);
        let mut path1 = find_belt_path(s, source1_output, input_cells[0], splitter_dir, &blocked1);
        if path1.is_none() {
            let mut b1: HashSet<Cell> = blocked_base.clone();
            b1.extend(output_cells.iter().copied());
            b1.extend([source2_output, sink_input, input_cells[0]]);
            path1 = find_belt_path(s, source1_output, input_cells[1], splitter_dir, &b1);
            if path1.is_none() {
                continue;
            }
        }
        let path1 = path1?;
        let path1_cells = belt_cell_set(&path1);
        let path1_end = (path1[path1.len() - 1].0, path1[path1.len() - 1].1);
        let remaining_input = if path1_end == input_cells[0] {
            input_cells[1]
        } else {
            input_cells[0]
        };

        // Path 2: source2 output → remaining splitter input.
        let mut blocked2: HashSet<Cell> = blocked_base.clone();
        blocked2.extend(output_cells.iter().copied());
        blocked2.extend(path1_cells.iter().copied());
        blocked2.insert(sink_input);
        let path2 =
            match find_belt_path(s, source2_output, remaining_input, splitter_dir, &blocked2) {
                Some(p) => p,
                None => continue,
            };
        let path2_cells = belt_cell_set(&path2);

        // Path 3: splitter output → sink input (try output 0, fallback 1).
        let unused_output_buffer_0 = (output_cells[0].0 + dd.0, output_cells[0].1 + dd.1);
        let unused_output_buffer_1 = (output_cells[1].0 + dd.0, output_cells[1].1 + dd.1);

        let mut blocked3: HashSet<Cell> = blocked_base.clone();
        blocked3.extend(path1_cells.iter().copied());
        blocked3.extend(path2_cells.iter().copied());
        blocked3.extend(input_cells.iter().copied());
        blocked3.extend([output_cells[1], unused_output_buffer_1]);
        let mut path3 = find_belt_path(s, output_cells[0], sink_input, sink_dir, &blocked3);
        if path3.is_none() {
            let mut b3: HashSet<Cell> = blocked_base.clone();
            b3.extend(path1_cells.iter().copied());
            b3.extend(path2_cells.iter().copied());
            b3.extend(input_cells.iter().copied());
            b3.extend([output_cells[0], unused_output_buffer_0]);
            path3 = find_belt_path(s, output_cells[1], sink_input, sink_dir, &b3);
            if path3.is_none() {
                continue;
            }
        }
        let path3 = path3?;

        let total_entities = path1.len() + path2.len() + path3.len() + 1;
        if (total_entities as f64) > max_entities {
            continue;
        }

        let mut world = World::empty(size, size);
        place_marker(
            &mut world,
            source1_pos,
            Item::Source,
            source1_dir,
            item_value,
        );
        place_marker(
            &mut world,
            source2_pos,
            Item::Source,
            source2_dir,
            item_value,
        );
        place_marker(&mut world, sink_pos, Item::Sink, sink_dir, item_value);
        for &(tx, ty) in &tiles {
            world.set(
                tx as usize,
                ty as usize,
                Channel::Entities,
                Item::Splitter as i64,
            );
            world.set(
                tx as usize,
                ty as usize,
                Channel::Direction,
                splitter_dir as i64,
            );
        }
        place_belts(&mut world, &path1);
        place_belts(&mut world, &path2);
        place_belts(&mut world, &path3);

        if world_throughput(&world) <= 0.0 {
            continue;
        }

        let protected: Vec<(usize, usize)> = tiles
            .iter()
            .map(|&(x, y)| (x as usize, y as usize))
            .collect();
        return finish(world, total_entities, protected, count);
    }
    None
}

/// Place a 3×3 assembler anchored at `(ax, ay)`, every tile facing North and
/// tagged with the recipe item — matching Python's assembler placement.
fn place_assembler(world: &mut World, ax: i64, ay: i64, recipe_item_value: i64) {
    for dx in 0..3 {
        for dy in 0..3 {
            let (x, y) = ((ax + dx) as usize, (ay + dy) as usize);
            world.set(x, y, Channel::Entities, Item::AssemblingMachine1 as i64);
            world.set(x, y, Channel::Direction, Direction::North as i64);
            world.set(x, y, Channel::Items, recipe_item_value);
        }
    }
}

/// Place an inserter at `pos` facing `dir`.
fn place_inserter(world: &mut World, pos: Cell, dir: Direction) {
    let (x, y) = (pos.0 as usize, pos.1 as usize);
    world.set(x, y, Channel::Entities, Item::Inserter as i64);
    world.set(x, y, Channel::Direction, dir as i64);
}

#[allow(clippy::too_many_lines)]
fn build_assemble_1in1out(
    size: usize,
    rng: &mut PyRandom,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let recipes = one_in_one_out();
    // Python raises if the list is empty; the recipe table guarantees it isn't.
    let mut count = (500).max(size * size * 12);

    while count > 0 {
        count -= 1;

        // Pick recipe + items.
        let (recipe_key, recipe) = recipes[rng.choice_index(recipes.len())].clone();
        let input_item_value = recipe.consumes.first().0 as i64;
        let output_item_value = recipe.produces.first().0 as i64;
        let recipe_item_value = recipe_key as i64;

        let ax = rng.randint(0, s - 3);
        let ay = rng.randint(0, s - 3);
        let asm_tiles: HashSet<Cell> = (0..3)
            .flat_map(|dx| (0..3).map(move |dy| (ax + dx, ay + dy)))
            .collect();

        // Distinct input + output perimeter slots.
        let chosen = rng.sample(&PERIM_SLOTS, 2);
        let in_slot = chosen[0];
        let out_slot = chosen[1];
        let in_inserter_dir = in_slot.2;
        let out_inserter_dir = out_slot.3;

        let in_inserter_pos = (ax + in_slot.0, ay + in_slot.1);
        let out_inserter_pos = (ax + out_slot.0, ay + out_slot.1);
        let in_dd = in_inserter_dir.delta();
        let out_dd = out_inserter_dir.delta();
        let in_pickup = (in_inserter_pos.0 - in_dd.0, in_inserter_pos.1 - in_dd.1);
        let out_drop = (out_inserter_pos.0 + out_dd.0, out_inserter_pos.1 + out_dd.1);

        let key_cells = [in_inserter_pos, out_inserter_pos, in_pickup, out_drop];
        if key_cells.iter().any(|&c| !in_grid(c, s)) {
            continue;
        }
        if key_cells.iter().collect::<HashSet<_>>().len() != key_cells.len() {
            continue;
        }
        if key_cells.iter().any(|c| asm_tiles.contains(c)) {
            continue;
        }

        let all_perim: HashSet<Cell> = PERIM_SLOTS
            .iter()
            .map(|&(ddx, ddy, _, _)| (ax + ddx, ay + ddy))
            .filter(|&c| in_grid(c, s))
            .collect();
        let reserved: HashSet<Cell> = asm_tiles
            .iter()
            .copied()
            .chain(key_cells.iter().copied())
            .chain(all_perim.iter().copied())
            .collect();
        let mut available: Vec<Cell> = Vec::new();
        for x in 0..s {
            for y in 0..s {
                if !reserved.contains(&(x, y)) {
                    available.push((x, y));
                }
            }
        }
        if available.len() < 2 {
            continue;
        }

        let chosen = rng.sample(&available, 2);
        let source_pos = chosen[0];
        let sink_pos = chosen[1];
        let source_dir = DIRS[rng.choice_index(4)];
        let sink_dir = DIRS[rng.choice_index(4)];

        let ds = source_dir.delta();
        let dk = sink_dir.delta();
        let source_output = (source_pos.0 + ds.0, source_pos.1 + ds.1);
        let sink_input = (sink_pos.0 - dk.0, sink_pos.1 - dk.1);

        if !in_grid(source_output, s) || !in_grid(sink_input, s) {
            continue;
        }
        if reserved.contains(&source_output) || reserved.contains(&sink_input) {
            continue;
        }
        if source_output == sink_input {
            continue;
        }

        // Path 1: source_output → in_pickup.
        let mut blocked1: HashSet<Cell> = asm_tiles.clone();
        blocked1.extend([
            in_inserter_pos,
            out_inserter_pos,
            source_pos,
            sink_pos,
            sink_input,
            out_drop,
        ]);
        let path1 = match find_belt_path(s, source_output, in_pickup, in_inserter_dir, &blocked1) {
            Some(p) => p,
            None => continue,
        };
        let path1_cells = belt_cell_set(&path1);

        // Path 2: out_drop → sink_input.
        let mut blocked2: HashSet<Cell> = asm_tiles.clone();
        blocked2.extend([
            in_inserter_pos,
            out_inserter_pos,
            source_pos,
            sink_pos,
            in_pickup,
            source_output,
        ]);
        blocked2.extend(path1_cells.iter().copied());
        let path2 = match find_belt_path(s, out_drop, sink_input, sink_dir, &blocked2) {
            Some(p) => p,
            None => continue,
        };

        let total_entities = path1.len() + path2.len() + 3;
        if (total_entities as f64) > max_entities {
            continue;
        }

        let mut world = World::empty(size, size);
        place_marker(
            &mut world,
            source_pos,
            Item::Source,
            source_dir,
            input_item_value,
        );
        place_marker(
            &mut world,
            sink_pos,
            Item::Sink,
            sink_dir,
            output_item_value,
        );
        place_assembler(&mut world, ax, ay, recipe_item_value);
        place_inserter(&mut world, in_inserter_pos, in_inserter_dir);
        place_inserter(&mut world, out_inserter_pos, out_inserter_dir);
        place_belts(&mut world, &path1);
        place_belts(&mut world, &path2);

        if world_throughput(&world) <= 0.0 {
            continue;
        }

        return finish(world, total_entities, vec![], count);
    }
    None
}

#[allow(clippy::too_many_lines)]
fn build_move_via_ug_belt(
    size: usize,
    rng: &mut PyRandom,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let pool = item_pool();
    let mut count = (500).max(size * size * 8);

    let item_value = if random_item {
        pool[rng.choice_index(pool.len())]
    } else {
        Item::ElectronicCircuit as i64
    };

    while count > 0 {
        count -= 1;
        let flow_dir = DIRS[rng.choice_index(4)];
        let is_horizontal = matches!(flow_dir, Direction::East | Direction::West);
        let flow_span = s; // W if horizontal else H — square grid
        let perp_span = s;

        // `fp_to_xy`: flow/perp coords → (x, y).
        let fp_to_xy = |fc: i64, pc: i64| -> Cell {
            if is_horizontal {
                (fc, pc)
            } else {
                (pc, fc)
            }
        };

        let max_wall = 4.min(flow_span - 2);
        let wall_thickness = rng.randint(1, max_wall);
        let wall_lo = rng.randint(1, flow_span - 1 - wall_thickness);
        let wall_hi = wall_lo + wall_thickness - 1;

        let forward = matches!(flow_dir, Direction::East | Direction::South);
        let (ug_down_flow, ug_up_flow, src_lo, src_hi, snk_lo, snk_hi) = if forward {
            (
                wall_lo - 1,
                wall_hi + 1,
                0,
                wall_lo - 1,
                wall_hi + 1,
                flow_span - 1,
            )
        } else {
            (
                wall_hi + 1,
                wall_lo - 1,
                wall_hi + 1,
                flow_span - 1,
                0,
                wall_lo - 1,
            )
        };

        let ug_perp = rng.randint(0, perp_span - 1);
        let ug_down_pos = fp_to_xy(ug_down_flow, ug_perp);
        let ug_up_pos = fp_to_xy(ug_up_flow, ug_perp);

        let mut wall_tiles: HashSet<Cell> = HashSet::new();
        for fc in wall_lo..=wall_hi {
            for pc in 0..perp_span {
                wall_tiles.insert(fp_to_xy(fc, pc));
            }
        }

        let mut source_cells: Vec<Cell> = Vec::new();
        for fc in src_lo..=src_hi {
            for pc in 0..perp_span {
                let cell = fp_to_xy(fc, pc);
                if cell != ug_down_pos {
                    source_cells.push(cell);
                }
            }
        }
        let mut sink_cells: Vec<Cell> = Vec::new();
        for fc in snk_lo..=snk_hi {
            for pc in 0..perp_span {
                let cell = fp_to_xy(fc, pc);
                if cell != ug_up_pos {
                    sink_cells.push(cell);
                }
            }
        }
        if source_cells.is_empty() || sink_cells.is_empty() {
            continue;
        }

        let source_pos = source_cells[rng.choice_index(source_cells.len())];
        let sink_pos = sink_cells[rng.choice_index(sink_cells.len())];
        let source_dir = DIRS[rng.choice_index(4)];
        let sink_dir = DIRS[rng.choice_index(4)];

        let ds = source_dir.delta();
        let dk = sink_dir.delta();
        let source_drop = (source_pos.0 + ds.0, source_pos.1 + ds.1);
        let sink_input = (sink_pos.0 - dk.0, sink_pos.1 - dk.1);

        if !in_grid(source_drop, s) || !in_grid(sink_input, s) {
            continue;
        }

        if wall_tiles.contains(&source_drop) {
            continue;
        }
        if source_drop == ug_up_pos || source_drop == sink_pos {
            continue;
        }
        let sd_flow = if is_horizontal {
            source_drop.0
        } else {
            source_drop.1
        };
        if !((src_lo..=src_hi).contains(&sd_flow) || source_drop == ug_down_pos) {
            continue;
        }

        if wall_tiles.contains(&sink_input) {
            continue;
        }
        if sink_input == ug_down_pos || sink_input == source_pos {
            continue;
        }
        let si_flow = if is_horizontal {
            sink_input.0
        } else {
            sink_input.1
        };
        if !((snk_lo..=snk_hi).contains(&si_flow) || sink_input == ug_up_pos) {
            continue;
        }

        let flow_delta = flow_dir.delta();

        // Path 1: source_drop → UG_DOWN input (on the source side).
        let path1: Vec<Belt> = if source_drop == ug_down_pos {
            vec![]
        } else {
            let ug_down_input = (ug_down_pos.0 - flow_delta.0, ug_down_pos.1 - flow_delta.1);
            let mut blocked1: HashSet<Cell> = wall_tiles.clone();
            blocked1.extend([source_pos, sink_pos, ug_down_pos, ug_up_pos, sink_input]);
            for fc in snk_lo..=snk_hi {
                for pc in 0..perp_span {
                    blocked1.insert(fp_to_xy(fc, pc));
                }
            }
            if blocked1.contains(&source_drop) || blocked1.contains(&ug_down_input) {
                continue;
            }
            match find_belt_path(s, source_drop, ug_down_input, flow_dir, &blocked1) {
                Some(p) => p,
                None => continue,
            }
        };

        // Path 2: UG_UP drop → sink_input (on the sink side).
        let path2: Vec<Belt> = if sink_input == ug_up_pos {
            vec![]
        } else {
            let ug_up_drop = (ug_up_pos.0 + flow_delta.0, ug_up_pos.1 + flow_delta.1);
            let path1_cells = belt_cell_set(&path1);
            let mut blocked2: HashSet<Cell> = wall_tiles.clone();
            blocked2.extend([source_pos, sink_pos, ug_down_pos, ug_up_pos, source_drop]);
            blocked2.extend(path1_cells.iter().copied());
            for fc in src_lo..=src_hi {
                for pc in 0..perp_span {
                    blocked2.insert(fp_to_xy(fc, pc));
                }
            }
            if blocked2.contains(&ug_up_drop) || blocked2.contains(&sink_input) {
                continue;
            }
            match find_belt_path(s, ug_up_drop, sink_input, sink_dir, &blocked2) {
                Some(p) => p,
                None => continue,
            }
        };

        let mut world = World::empty(size, size);
        place_marker(&mut world, source_pos, Item::Source, source_dir, item_value);
        place_marker(&mut world, sink_pos, Item::Sink, sink_dir, item_value);
        world.place_underground(
            ug_down_pos.0 as usize,
            ug_down_pos.1 as usize,
            flow_dir,
            Misc::UndergroundDown,
        );
        world.place_underground(
            ug_up_pos.0 as usize,
            ug_up_pos.1 as usize,
            flow_dir,
            Misc::UndergroundUp,
        );
        place_belts(&mut world, &path1);
        place_belts(&mut world, &path2);

        if world_throughput(&world) <= 0.0 {
            continue;
        }

        let total_entities = 2 + path1.len() + path2.len();
        if (total_entities as f64) > max_entities {
            continue;
        }

        // Mark only the wall as UNAVAILABLE; every other tile is buildable.
        for &(wx, wy) in &wall_tiles {
            world.set(wx as usize, wy as usize, Channel::Footprint, 0);
        }

        return finish(world, total_entities, vec![], count);
    }
    None
}

#[allow(clippy::too_many_lines)]
fn build_assemble_2in1out(
    size: usize,
    rng: &mut PyRandom,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let recipes = two_in_one_out();
    let mut count = (500).max(size * size * 16);

    while count > 0 {
        count -= 1;

        let (recipe_key, recipe) = recipes[rng.choice_index(recipes.len())].clone();
        // Randomize which ingredient is "A" vs "B".
        let mut input_items: Vec<Item> = recipe.consumes.iter().map(|&(i, _)| i).collect();
        rng.shuffle(&mut input_items);
        let input_a_value = input_items[0] as i64;
        let input_b_value = input_items[1] as i64;
        let output_item_value = recipe.produces.first().0 as i64;
        let recipe_item_value = recipe_key as i64;

        let ax = rng.randint(0, s - 3);
        let ay = rng.randint(0, s - 3);
        let asm_tiles: HashSet<Cell> = (0..3)
            .flat_map(|dx| (0..3).map(move |dy| (ax + dx, ay + dy)))
            .collect();

        // Three distinct perimeter slots: 2 inputs + 1 output.
        let chosen = rng.sample(&PERIM_SLOTS, 3);
        let in_a_slot = chosen[0];
        let in_b_slot = chosen[1];
        let out_slot = chosen[2];
        let in_a_pos = (ax + in_a_slot.0, ay + in_a_slot.1);
        let in_a_dir = in_a_slot.2;
        let in_b_pos = (ax + in_b_slot.0, ay + in_b_slot.1);
        let in_b_dir = in_b_slot.2;
        let out_pos = (ax + out_slot.0, ay + out_slot.1);
        let out_dir = out_slot.3;

        let in_a_dd = in_a_dir.delta();
        let in_b_dd = in_b_dir.delta();
        let out_dd = out_dir.delta();
        let in_a_pickup = (in_a_pos.0 - in_a_dd.0, in_a_pos.1 - in_a_dd.1);
        let in_b_pickup = (in_b_pos.0 - in_b_dd.0, in_b_pos.1 - in_b_dd.1);
        let out_drop = (out_pos.0 + out_dd.0, out_pos.1 + out_dd.1);

        let key_cells = [
            in_a_pos,
            in_b_pos,
            out_pos,
            in_a_pickup,
            in_b_pickup,
            out_drop,
        ];
        if key_cells.iter().any(|&c| !in_grid(c, s)) {
            continue;
        }
        if key_cells.iter().collect::<HashSet<_>>().len() != key_cells.len() {
            continue;
        }
        if key_cells.iter().any(|c| asm_tiles.contains(c)) {
            continue;
        }

        let all_perim: HashSet<Cell> = PERIM_SLOTS
            .iter()
            .map(|&(ddx, ddy, _, _)| (ax + ddx, ay + ddy))
            .filter(|&c| in_grid(c, s))
            .collect();
        let reserved: HashSet<Cell> = asm_tiles
            .iter()
            .copied()
            .chain(key_cells.iter().copied())
            .chain(all_perim.iter().copied())
            .collect();
        let mut available: Vec<Cell> = Vec::new();
        for x in 0..s {
            for y in 0..s {
                if !reserved.contains(&(x, y)) {
                    available.push((x, y));
                }
            }
        }
        if available.len() < 3 {
            continue;
        }

        let chosen = rng.sample(&available, 3);
        let src_a_pos = chosen[0];
        let src_b_pos = chosen[1];
        let sink_pos = chosen[2];
        let src_a_dir = DIRS[rng.choice_index(4)];
        let src_b_dir = DIRS[rng.choice_index(4)];
        let sink_dir = DIRS[rng.choice_index(4)];

        let ds_a = src_a_dir.delta();
        let ds_b = src_b_dir.delta();
        let dk = sink_dir.delta();
        let src_a_out = (src_a_pos.0 + ds_a.0, src_a_pos.1 + ds_a.1);
        let src_b_out = (src_b_pos.0 + ds_b.0, src_b_pos.1 + ds_b.1);
        let sink_in = (sink_pos.0 - dk.0, sink_pos.1 - dk.1);

        let conn = [src_a_out, src_b_out, sink_in];
        if conn.iter().any(|&c| !in_grid(c, s)) {
            continue;
        }
        if conn.iter().collect::<HashSet<_>>().len() != conn.len() {
            continue;
        }
        if conn.iter().any(|c| reserved.contains(c)) {
            continue;
        }
        let endpoints: HashSet<Cell> = [src_a_pos, src_b_pos, sink_pos].into_iter().collect();
        if conn.iter().any(|c| endpoints.contains(c)) {
            continue;
        }

        let fixed_cells: HashSet<Cell> = asm_tiles
            .iter()
            .copied()
            .chain([in_a_pos, in_b_pos, out_pos, src_a_pos, src_b_pos, sink_pos])
            .collect();

        // Path A: source A → input-A pickup.
        let mut blocked_a = fixed_cells.clone();
        blocked_a.extend([in_b_pickup, out_drop, src_b_out, sink_in]);
        let path_a = match find_belt_path(s, src_a_out, in_a_pickup, in_a_dir, &blocked_a) {
            Some(p) => p,
            None => continue,
        };
        let path_a_cells = belt_cell_set(&path_a);

        // Path B: source B → input-B pickup.
        let mut blocked_b = fixed_cells.clone();
        blocked_b.extend([in_a_pickup, out_drop, src_a_out, sink_in]);
        blocked_b.extend(path_a_cells.iter().copied());
        let path_b = match find_belt_path(s, src_b_out, in_b_pickup, in_b_dir, &blocked_b) {
            Some(p) => p,
            None => continue,
        };
        let path_b_cells = belt_cell_set(&path_b);

        // Path C: output drop → sink input.
        let mut blocked_c = fixed_cells.clone();
        blocked_c.extend([in_a_pickup, in_b_pickup, src_a_out, src_b_out]);
        blocked_c.extend(path_a_cells.iter().copied());
        blocked_c.extend(path_b_cells.iter().copied());
        let path_c = match find_belt_path(s, out_drop, sink_in, sink_dir, &blocked_c) {
            Some(p) => p,
            None => continue,
        };

        let total_entities = path_a.len() + path_b.len() + path_c.len() + 4;
        if (total_entities as f64) > max_entities {
            continue;
        }

        let mut world = World::empty(size, size);
        place_marker(
            &mut world,
            src_a_pos,
            Item::Source,
            src_a_dir,
            input_a_value,
        );
        place_marker(
            &mut world,
            src_b_pos,
            Item::Source,
            src_b_dir,
            input_b_value,
        );
        place_marker(
            &mut world,
            sink_pos,
            Item::Sink,
            sink_dir,
            output_item_value,
        );
        place_assembler(&mut world, ax, ay, recipe_item_value);
        place_inserter(&mut world, in_a_pos, in_a_dir);
        place_inserter(&mut world, in_b_pos, in_b_dir);
        place_inserter(&mut world, out_pos, out_dir);
        place_belts(&mut world, &path_a);
        place_belts(&mut world, &path_b);
        place_belts(&mut world, &path_c);

        if world_throughput(&world) <= 0.0 {
            continue;
        }

        return finish(world, total_entities, vec![], count);
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
