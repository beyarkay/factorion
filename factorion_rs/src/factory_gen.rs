//! Procedural generation of complete, valid training factories.
//!
//! [`build_factory`] builds a known-correct factory of a given [`LessonKind`]
//! — laying sources, sinks, transport/underground belts, splitters and
//! assemblers onto a grid so items flow from every source to every sink — by
//! randomized rejection sampling: it places a candidate layout, checks it
//! carries positive throughput (via the [`crate::throughput`] engine), and on
//! failure retries until one succeeds or the per-kind attempt budget is
//! exhausted, returning `None` in that case.
//!
//! All layout randomness is drawn from [`crate::rng`], a fast deterministic
//! generator seeded once per call, so the same `(size, kind, seed)` always
//! produces the same factory. Each [`LessonKind`] is a different entity/layout
//! pattern; the result is a [`BuiltFactory`] (the world tensor plus the
//! blanking bookkeeping a training lesson needs).

use crate::graph::build_graph;
use crate::rng::Rng;
use crate::throughput::{calc_throughput, factory_score};
use crate::types::{all_items, all_recipes, Channel, Direction, Item, Misc, Recipe};
use crate::world::World;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// The lesson kinds. This is the single source of truth for the kind set:
/// Python's `LessonKind` enum is built from [`all_lesson_kinds`] via the PyO3
/// `py_lesson_kinds` binding, so the two can't drift. The integer values are
/// non-contiguous for historical reasons (removed kinds left gaps).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LessonKind {
    MoveOneItem = 0,
    SplitterSplit = 3,
    SplitterMerge = 4,
    #[deprecated]
    Assemble1In1Out = 5,
    MoveViaUgBelt = 6,
    #[deprecated]
    Assemble2In1Out = 7,
    Memorise1IngredientRecipes = 8,
    MoveOneItemChaos = 9,
    CrossUnderBelt = 10,
    Memorise2IngredientRecipes = 11,
    Memorise3IngredientRecipes = 12,
    Memorise4IngredientRecipes = 13,
    Memorise5IngredientRecipes = 14,
    Factory1Ingredient = 15,
}

impl LessonKind {
    pub fn from_i64(v: i64) -> Option<Self> {
        match v {
            0 => Some(LessonKind::MoveOneItem),
            3 => Some(LessonKind::SplitterSplit),
            4 => Some(LessonKind::SplitterMerge),
            #[allow(deprecated)]
            5 => Some(LessonKind::Assemble1In1Out),
            6 => Some(LessonKind::MoveViaUgBelt),
            #[allow(deprecated)]
            7 => Some(LessonKind::Assemble2In1Out),
            8 => Some(LessonKind::Memorise1IngredientRecipes),
            9 => Some(LessonKind::MoveOneItemChaos),
            10 => Some(LessonKind::CrossUnderBelt),
            11 => Some(LessonKind::Memorise2IngredientRecipes),
            12 => Some(LessonKind::Memorise3IngredientRecipes),
            13 => Some(LessonKind::Memorise4IngredientRecipes),
            14 => Some(LessonKind::Memorise5IngredientRecipes),
            15 => Some(LessonKind::Factory1Ingredient),
            _ => None,
        }
    }

    /// The canonical SCREAMING_SNAKE_CASE name — the identifier the Python
    /// `LessonKind` enum member takes.
    pub fn name(self) -> &'static str {
        match self {
            LessonKind::MoveOneItem => "MOVE_ONE_ITEM",
            LessonKind::SplitterSplit => "SPLITTER_SPLIT",
            LessonKind::SplitterMerge => "SPLITTER_MERGE",
            #[allow(deprecated)]
            LessonKind::Assemble1In1Out => "ASSEMBLE_1IN_1OUT",
            LessonKind::MoveViaUgBelt => "MOVE_VIA_UG_BELT",
            #[allow(deprecated)]
            LessonKind::Assemble2In1Out => "ASSEMBLE_2IN_1OUT",
            LessonKind::Memorise1IngredientRecipes => "MEMORISE_1_INGREDIENT_RECIPES",
            LessonKind::Memorise2IngredientRecipes => "MEMORISE_2_INGREDIENT_RECIPES",
            LessonKind::Memorise3IngredientRecipes => "MEMORISE_3_INGREDIENT_RECIPES",
            LessonKind::Memorise4IngredientRecipes => "MEMORISE_4_INGREDIENT_RECIPES",
            LessonKind::Memorise5IngredientRecipes => "MEMORISE_5_INGREDIENT_RECIPES",
            LessonKind::MoveOneItemChaos => "MOVE_ONE_ITEM_CHAOS",
            LessonKind::CrossUnderBelt => "CROSS_UNDER_BELT",
            LessonKind::Factory1Ingredient => "FACTORY_1_INGREDIENT",
        }
    }
}

/// Every [`LessonKind`], in the order the Python enum lists them (which fixes
/// `list(LessonKind)` iteration — the lesson sampler relies on it).
pub fn all_lesson_kinds() -> &'static [LessonKind] {
    &[
        LessonKind::MoveOneItem,
        LessonKind::SplitterSplit,
        LessonKind::SplitterMerge,
        // LessonKind::Assemble1In1Out,
        LessonKind::MoveViaUgBelt,
        // LessonKind::Assemble2In1Out,
        LessonKind::Memorise1IngredientRecipes,
        LessonKind::Memorise2IngredientRecipes,
        LessonKind::Memorise3IngredientRecipes,
        LessonKind::Memorise4IngredientRecipes,
        LessonKind::Memorise5IngredientRecipes,
        LessonKind::MoveOneItemChaos,
        LessonKind::CrossUnderBelt,
        LessonKind::Factory1Ingredient,
    ]
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
/// the enumerated paths (and therefore the shuffle that consumes them).
const BFS_DELTAS: [(i64, i64); 4] = [(0, -1), (1, 0), (0, 1), (-1, 0)];

/// Inverse of `BFS_DELTAS`: a one-step (dx, dy) → belt direction. Returns
/// `None` for non-unit deltas.
fn delta_to_dir(dx: i64, dy: i64) -> Option<Direction> {
    match (dx, dy) {
        (0, -1) => Some(Direction::North),
        (1, 0) => Some(Direction::East),
        (0, 1) => Some(Direction::South),
        (-1, 0) => Some(Direction::West),
        _ => None,
    }
}

/// The pool a factory's transported item is chosen from: every `Item` in
/// `all_items()` order, as raw channel values.
fn item_pool() -> Vec<i64> {
    all_items().iter().map(|&i| i as i64).collect()
}

/// The 12 non-corner perimeter slots around a 3×3 assembler anchored at
/// `(ax, ay)`, as `(offset_x, offset_y, input_inserter_dir, output_inserter_dir)`.
/// The slot order is load-bearing: the inserter/source/sink sampling indexes
/// into it, so reordering changes which factory a seed produces.
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

/// Throughput score of a fully-placed world: build its flow graph and return
/// the achieved-throughput score across its sinks.
fn world_throughput(world: &World) -> f64 {
    let graph = build_graph(world);
    let (deliveries, _) = calc_throughput(&graph);
    factory_score(&deliveries)
}

type Cell = (i64, i64);

/// Whether a route may tunnel under blocked cells, and how it may begin.
enum Underground {
    /// Belt-only: a plain shortest belt search; tunnels are never used.
    Off,
    /// Tunnels may clear blocked cells. `Some(dir)` also lets the route open with
    /// a tunnel entrance on `start` itself (source → UG_DOWN facing `dir`).
    On(Option<Direction>),
}

/// Up to `shortest_n` distinct minimal-cost routes from `start` to `end`
/// (`shortest_n < 0` returns every valid one), each a `Vec<UgPlacement>` laying
/// exactly one entity per cell from `start` through `end` (which faces `end_dir`).
///
/// A weighted Dijkstra over `(x, y, arrival_dir)` states: a belt step costs 4 and
/// a `span`-tile tunnel costs `5 * (span + 1)` (an integral 1.25×/tile penalty),
/// so tunnels only ever clear blocked cells, with minimal span. Tunnels run
/// straight; 180° reversals and self-overlapping routes are rejected. See
/// [`Underground`] for the tunnel options; `Underground::Off` reduces this to a
/// plain uniform-cost belt search.
fn find_belt_paths(
    start: Cell,
    end: Cell,
    end_dir: Direction,
    size: i64,
    blocked: &HashSet<Cell>,
    underground: Underground,
    shortest_n: i64,
) -> Vec<Vec<UgPlacement>> {
    let in_bounds = |c: Cell| 0 <= c.0 && c.0 < size && 0 <= c.1 && c.1 < size;
    if !in_bounds(start) || !in_bounds(end) || blocked.contains(&start) || blocked.contains(&end) {
        return vec![];
    }

    // Whether tunnels are allowed, and the direction (if any) the route opens a
    // tunnel toward on `start` itself.
    let (allow_underground, start_adir) = match underground {
        Underground::Off => (false, 0),
        Underground::On(start_dir) => (true, start_dir.map_or(0, |d| d as i64)),
    };

    // State = (x, y, arrival_dir_value); arrival_dir 0 == "free" (the start).
    type State = (i64, i64, i64);
    let start_state: State = (start.0, start.1, start_adir);

    let mut dist: HashMap<State, u64> = HashMap::new();
    // Every min-cost incoming edge of a state: the predecessor state plus the
    // placement(s) emitted on that edge, so all shortest routes can enumerate.
    let mut preds: HashMap<State, Vec<(State, Vec<UgPlacement>)>> = HashMap::new();
    dist.insert(start_state, 0);
    let mut counter: u64 = 0;
    let mut pq: BinaryHeap<Reverse<(u64, u64, State)>> = BinaryHeap::new();
    pq.push(Reverse((0, counter, start_state)));
    counter += 1;

    // Record a min-cost edge into `ns`: reset predecessors on a strict
    // improvement, accumulate on a tie (equal-cost edges enumerate alternatives).
    let mut relax = |dist: &mut HashMap<State, u64>,
                     preds: &mut HashMap<State, Vec<(State, Vec<UgPlacement>)>>,
                     pq: &mut BinaryHeap<Reverse<(u64, u64, State)>>,
                     ns: State,
                     nd: u64,
                     from: State,
                     emit: Vec<UgPlacement>| {
        let cur = *dist.get(&ns).unwrap_or(&u64::MAX);
        if nd < cur {
            dist.insert(ns, nd);
            preds.insert(ns, vec![(from, emit)]);
            pq.push(Reverse((nd, counter, ns)));
            counter += 1;
        } else if nd == cur {
            preds.entry(ns).or_default().push((from, emit));
        }
    };

    // Cheapest arrival(s) onto `end`: the state to backtrack from and the
    // entity/entities covering `end` itself (a belt facing `end_dir`, or a
    // UG_DOWN/UG_UP tunnel pair surfacing onto `end`).
    let mut goal_cost: u64 = u64::MAX;
    let mut goal_arrivals: Vec<(State, Vec<UgPlacement>)> = Vec::new();

    while let Some(Reverse((d_so_far, _, state))) = pq.pop() {
        if d_so_far > *dist.get(&state).unwrap_or(&u64::MAX) {
            continue;
        }
        // All equal-cost arrivals pop before any strictly costlier state, so once
        // we pop past `goal_cost` every minimal route has been recorded.
        if d_so_far > goal_cost {
            break;
        }
        let (cx, cy, adir) = state;
        if (cx, cy) == end {
            // Reached on the surface: `end` becomes a belt facing `end_dir`.
            if d_so_far < goal_cost {
                goal_cost = d_so_far;
                goal_arrivals.clear();
            }
            goal_arrivals.push((state, vec![(end.0, end.1, end_dir, Misc::None)]));
            continue;
        }

        let back: Option<(i64, i64)> = if adir != 0 {
            let (dx, dy) = Direction::from_i64(adir).delta();
            Some((-dx, -dy))
        } else {
            None
        };

        // Normal surface belt step (no 180° reversal): this cell becomes a belt.
        for d in DIRS {
            let (dx, dy) = d.delta();
            if back == Some((dx, dy)) {
                continue;
            }
            let n = (cx + dx, cy + dy);
            if !in_bounds(n) || blocked.contains(&n) {
                continue;
            }
            let ns: State = (n.0, n.1, d as i64);
            let nd = d_so_far + 4;
            relax(
                &mut dist,
                &mut preds,
                &mut pq,
                ns,
                nd,
                state,
                vec![(cx, cy, d, Misc::None)],
            );
        }

        // Underground tunnel, continuing straight in the arrival direction.
        if allow_underground && adir != 0 {
            let d = Direction::from_i64(adir);
            let (dx, dy) = d.delta();
            for span in 2..=UNDERGROUND_MAX_OFFSET {
                let exit_cell = (cx + dx * span, cy + dy * span);
                if !in_bounds(exit_cell) || blocked.contains(&exit_cell) {
                    continue;
                }
                // Surface straight onto `end` (UG_UP → sink), no trailing belt.
                // Charged like a normal tunnel, so used only when the obstruction
                // reaches the sink.
                if exit_cell == end && d == end_dir {
                    let cost = d_so_far + 5 * (span as u64 + 1);
                    if cost < goal_cost {
                        goal_cost = cost;
                        goal_arrivals.clear();
                    }
                    if cost == goal_cost {
                        goal_arrivals.push((
                            state,
                            vec![
                                (cx, cy, d, Misc::UndergroundDown),
                                (end.0, end.1, d, Misc::UndergroundUp),
                            ],
                        ));
                    }
                    continue;
                }
                let surface = (cx + dx * (span + 1), cy + dy * (span + 1));
                if !in_bounds(surface) || blocked.contains(&surface) {
                    continue;
                }
                let ns: State = (surface.0, surface.1, d as i64);
                let nd = d_so_far + 5 * (span as u64 + 1);
                relax(
                    &mut dist,
                    &mut preds,
                    &mut pq,
                    ns,
                    nd,
                    state,
                    vec![
                        (cx, cy, d, Misc::UndergroundDown),
                        (exit_cell.0, exit_cell.1, d, Misc::UndergroundUp),
                    ],
                );
            }
        }
    }

    if goal_arrivals.is_empty() {
        return vec![];
    }

    // Walk every predecessor chain from each goal arrival back to `start_state`,
    // emitting start→end routes and dropping any that reuse a tile, until
    // `shortest_n` valid routes are collected (`< 0` == unbounded).
    let limit = if shortest_n < 0 {
        usize::MAX
    } else {
        shortest_n as usize
    };
    fn walk(
        state: State,
        start_state: State,
        preds: &HashMap<State, Vec<(State, Vec<UgPlacement>)>>,
        acc: &mut Vec<UgPlacement>,
        tail: &[UgPlacement],
        results: &mut Vec<Vec<UgPlacement>>,
        limit: usize,
    ) {
        if results.len() >= limit {
            return;
        }
        if state == start_state {
            let mut path: Vec<UgPlacement> = acc.iter().rev().copied().collect();
            path.extend_from_slice(tail);
            // One entity per tile: reject any self-overlapping (reorienting) detour.
            let mut seen: HashSet<Cell> = HashSet::new();
            if path.iter().all(|&(x, y, _, _)| seen.insert((x, y))) {
                results.push(path);
            }
            return;
        }
        let Some(ps) = preds.get(&state) else {
            return;
        };
        for (pstate, emit) in ps {
            if results.len() >= limit {
                return;
            }
            for &p in emit.iter().rev() {
                acc.push(p);
            }
            walk(*pstate, start_state, preds, acc, tail, results, limit);
            for _ in emit {
                acc.pop();
            }
        }
    }

    let mut results: Vec<Vec<UgPlacement>> = Vec::new();
    for (gstate, tail) in &goal_arrivals {
        if results.len() >= limit {
            break;
        }
        let mut acc: Vec<UgPlacement> = Vec::new();
        walk(
            *gstate,
            start_state,
            &preds,
            &mut acc,
            tail,
            &mut results,
            limit,
        );
    }
    results
}

/// Convert a cell run into belt placements, last belt taking `end_dir`.
fn path_to_belts(path: &[Cell], end_dir: Direction) -> Vec<UgPlacement> {
    let mut belts: Vec<UgPlacement> = Vec::new();
    for w in path.windows(2) {
        let (r1, c1) = w[0];
        let (r2, c2) = w[1];
        if let Some(d) = delta_to_dir(r2 - r1, c2 - c1) {
            belts.push((r1, c1, d, Misc::None));
        }
    }
    if let Some(&(lx, ly)) = path.last() {
        belts.push((lx, ly, end_dir, Misc::None));
    }
    belts
}

/// The single shortest route from `start` to `end`, or `None` — a thin wrapper
/// over [`find_belt_paths`] for the common "just one path" case.
fn find_belt_path(
    start: Cell,
    end: Cell,
    end_dir: Direction,
    size: i64,
    blocked: &HashSet<Cell>,
    underground: Underground,
) -> Option<Vec<UgPlacement>> {
    find_belt_paths(start, end, end_dir, size, blocked, underground, 1)
        .into_iter()
        .next()
}

/// The `(x, y)` cells of a belt run (dropping directions).
fn belt_cells(belts: &[UgPlacement]) -> Vec<Cell> {
    belts.iter().map(|&(x, y, _, _)| (x, y)).collect()
}

/// The `(x, y)` cells of a belt run as a set.
fn belt_cell_set(belts: &[UgPlacement]) -> HashSet<Cell> {
    belts.iter().map(|&(x, y, _, _)| (x, y)).collect()
}

/// The result of a successful `build_factory`: a complete world plus the
/// bookkeeping a training lesson needs to blank it — the removable-entity
/// count and the positions that must never be blanked.
pub struct BuiltFactory {
    pub world: World,
    pub total_entities: usize,
    pub protected_positions: Vec<(usize, usize)>,
}

/// Lay a route into `world`: plain belts, and underground tunnel ends for any
/// `Misc`-tagged placement.
fn place_belts(world: &mut World, belts: &[UgPlacement]) {
    for &(x, y, d, m) in belts {
        if m == Misc::None {
            world.set(
                x as usize,
                y as usize,
                Channel::Entities,
                Item::TransportBelt as i64,
            );
            world.set(x as usize, y as usize, Channel::Direction, d as i64);
        } else {
            world.place_underground(x as usize, y as usize, d, m);
        }
    }
}

/// Build a complete, valid factory of the given lesson `kind` on a
/// `size × size` grid, seeding the RNG from `seed`. `random_item` picks a
/// random transported item (vs a fixed default); `max_entities` caps the
/// entity count. Returns `None` when rejection sampling is exhausted.
pub fn build_factory(
    size: usize,
    kind: LessonKind,
    seed: u64,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let mut rng = Rng::seeded(seed);
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
        // LessonKind::Assemble1In1Out => build_assemble_1in1out(size, &mut rng, max_entities),
        LessonKind::MoveViaUgBelt => {
            build_move_via_ug_belt(size, &mut rng, random_item, max_entities)
        }
        // LessonKind::Assemble2In1Out => build_assemble_2in1out(size, &mut rng, max_entities),
        LessonKind::Memorise1IngredientRecipes => {
            build_memorise_recipes(size, &mut rng, max_entities, 1)
        }
        LessonKind::Memorise2IngredientRecipes => {
            build_memorise_recipes(size, &mut rng, max_entities, 2)
        }
        LessonKind::Memorise3IngredientRecipes => {
            build_memorise_recipes(size, &mut rng, max_entities, 3)
        }
        LessonKind::Memorise4IngredientRecipes => {
            build_memorise_recipes(size, &mut rng, max_entities, 4)
        }
        LessonKind::Memorise5IngredientRecipes => {
            build_memorise_recipes(size, &mut rng, max_entities, 5)
        }
        LessonKind::CrossUnderBelt => {
            build_cross_under_belt(size, &mut rng, random_item, max_entities)
        }
        LessonKind::Factory1Ingredient => build_factory_1_ingredient(size, &mut rng, max_entities),
        _ => None,
    }
}

/// `(x, y)` is within a `size x size` grid.
fn in_grid(c: Cell, size: i64) -> bool {
    0 <= c.0 && c.0 < size && 0 <= c.1 && c.1 < size
}

/// The free cells (those not in `reserved`) that a lesson samples its
/// source/sink positions from, in x-outer/y-inner order. The order is
/// load-bearing — sampling indexes into it.
fn available_cells(s: i64, reserved: &HashSet<Cell>) -> Vec<Cell> {
    let mut available = Vec::new();
    for x in 0..s {
        for y in 0..s {
            if !reserved.contains(&(x, y)) {
                available.push((x, y));
            }
        }
    }
    available
}

/// Rotate a direction 90° clockwise (North→East→South→West→North). `None`
/// stays `None`.
fn rotate_dir_cw(d: Direction) -> Direction {
    match d {
        Direction::North => Direction::East,
        Direction::East => Direction::South,
        Direction::South => Direction::West,
        Direction::West => Direction::North,
        Direction::None => Direction::None,
    }
}

/// Rotate a square world 90° clockwise: cell `(x, y)` → `(s-1-y, x)`, every
/// entity's facing turned with it. Multi-tile entities (splitters) stay valid
/// because each occupied tile is remapped independently — for any facing the
/// downstream anchor is still the first tile visited in scan order. Empty
/// cells keep their `World::empty` defaults.
fn rotate_world_cw(world: &World) -> World {
    let s = world.width();
    debug_assert_eq!(s, world.height(), "rotation assumes a square grid");
    let mut out = World::empty(s, s);
    for x in 0..s {
        for y in 0..s {
            if world.entity_at(x, y).is_none() {
                continue;
            }
            let (nx, ny) = (s - 1 - y, x);
            out.set(
                nx,
                ny,
                Channel::Entities,
                world.get(x, y, Channel::Entities),
            );
            let ndir = rotate_dir_cw(world.direction_at(x, y));
            out.set(nx, ny, Channel::Direction, ndir as i64);
            out.set(nx, ny, Channel::Items, world.get(x, y, Channel::Items));
            out.set(nx, ny, Channel::Misc, world.get(x, y, Channel::Misc));
        }
    }
    out
}

/// Place a splitter's tiles (entity + facing) into the world.
fn place_splitter(world: &mut World, tiles: &[Cell], dir: Direction) {
    for &(tx, ty) in tiles {
        world.set(
            tx as usize,
            ty as usize,
            Channel::Entities,
            Item::Splitter as i64,
        );
        world.set(tx as usize, ty as usize, Channel::Direction, dir as i64);
    }
}

/// The in-bounds tiles of all 12 assembler perimeter slots for anchor
/// `(ax, ay)` — the cells a source/sink must avoid so it isn't treated as an
/// inserter feeding the assembler.
fn all_perim_set(ax: i64, ay: i64, s: i64) -> HashSet<Cell> {
    PERIM_SLOTS
        .iter()
        .map(|&(ddx, ddy, _, _)| (ax + ddx, ay + ddy))
        .filter(|&c| in_grid(c, s))
        .collect()
}

/// Place a source or sink marker (entity + facing + carried item).
fn place_marker(world: &mut World, pos: Cell, ent: Item, dir: Direction, item_value: i64) {
    let (x, y) = (pos.0 as usize, pos.1 as usize);
    world.set(x, y, Channel::Entities, ent as i64);
    world.set(x, y, Channel::Direction, dir as i64);
    world.set(x, y, Channel::Items, item_value);
}

/// Place a 3×3 assembler anchored at `(ax, ay)`, every tile facing North and
/// tagged with the recipe item.
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

/// The source/sink draw shared by MOVE_ONE_ITEM and its CHAOS variant: two
/// distinct random tiles (`pos1 == pos2` is rejected as `None`), each with a
/// random facing, plus the transported item.
struct SourceSink {
    source_wh: Cell,
    sink_wh: Cell,
    source_dir: Direction,
    sink_dir: Direction,
    item_value: i64,
}

fn draw_source_sink(
    rng: &mut Rng,
    size: usize,
    pool: &[i64],
    random_item: bool,
) -> Option<SourceSink> {
    let s = size as i64;
    let pos1 = rng.randrange((size * size) as u64) as i64;
    let pos2 = rng.randrange((size * size) as u64) as i64;
    if pos1 == pos2 {
        return None;
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
    Some(SourceSink {
        source_wh,
        sink_wh,
        source_dir,
        sink_dir,
        item_value,
    })
}

fn build_move_one_item(
    size: usize,
    rng: &mut Rng,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let pool = item_pool();
    let mut count = (500).max(size * size * 4);
    while count > 0 {
        count -= 1;
        let SourceSink {
            source_wh,
            sink_wh,
            source_dir,
            sink_dir,
            item_value,
        } = match draw_source_sink(rng, size, &pool, random_item) {
            Some(ss) => ss,
            None => continue,
        };

        let mut world = World::empty(size, size);
        place_marker(&mut world, source_wh, Item::Source, source_dir, item_value);
        place_marker(&mut world, sink_wh, Item::Sink, sink_dir, item_value);

        // Route from the source's output cell to the sink's input cell, keeping
        // the markers themselves clear (all shortest belt paths; no tunnels).
        let (dr_s, dc_s) = source_dir.delta();
        let start = (source_wh.0 + dr_s, source_wh.1 + dc_s);
        let (dr_k, dc_k) = sink_dir.delta();
        let end = (sink_wh.0 - dr_k, sink_wh.1 - dc_k);
        let mut blocked: HashSet<Cell> = HashSet::new();
        blocked.insert(source_wh);
        blocked.insert(sink_wh);
        let mut paths = find_belt_paths(start, end, sink_dir, s, &blocked, Underground::Off, -1);
        paths.retain(|p| (p.len() as f64) <= max_entities);

        if paths.is_empty() {
            continue;
        }

        rng.shuffle(&mut paths);
        let mut chosen: Option<Vec<UgPlacement>> = None;
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
    rng: &mut Rng,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    let pool = item_pool();
    let mut count = (500).max(size * size * 4);
    while count > 0 {
        count -= 1;
        let SourceSink {
            source_wh,
            sink_wh,
            source_dir,
            sink_dir,
            item_value,
        } = match draw_source_sink(rng, size, &pool, random_item) {
            Some(ss) => ss,
            None => continue,
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
        let belts_a = match find_belt_path(start, mid, source_dir, s, &blocked_a, Underground::Off)
        {
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
        let belts_b = match find_belt_path(mid, end, sink_dir, s, &blocked_b, Underground::Off) {
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

/// Rebuild the entity placements of a run from just its ordered cells (the
/// inverse of [`placement_cells`]). Adjacent cells become a belt facing the
/// step; two colinear cells that are >1 apart become an underground pair
/// (DOWN at the first, UP at the second) tunnelling between them. The final
/// belt takes `end_dir`. Because a run is rebuilt purely from cell adjacency,
/// reversing the cell list yields a valid reversed run — corners re-orient
/// correctly and tunnels flip (DOWN↔UP, direction 180°) for free.
fn path_to_placements(cells: &[Cell], end_dir: Direction) -> Option<Vec<UgPlacement>> {
    let mut out: Vec<UgPlacement> = Vec::new();
    let mut i = 0;
    while i < cells.len() {
        let (x, y) = cells[i];
        match cells.get(i + 1) {
            Some(&(nx, ny)) if (nx - x).abs() + (ny - y).abs() == 1 => {
                out.push((x, y, delta_to_dir(nx - x, ny - y)?, Misc::None));
                i += 1;
            }
            Some(&(nx, ny)) => {
                // Colinear span-apart cells: a straight tunnel between them.
                let d = delta_to_dir((nx - x).signum(), (ny - y).signum())?;
                out.push((x, y, d, Misc::UndergroundDown));
                out.push((nx, ny, d, Misc::UndergroundUp));
                i += 2;
            }
            None => {
                out.push((x, y, end_dir, Misc::None));
                i += 1;
            }
        }
    }
    Some(out)
}

/// A candidate wiring: `(total placements, stem, arm→prong1, arm→prong2)`.
type Wiring = (usize, Vec<Cell>, Vec<Cell>, Vec<Cell>);

/// A wired splitter Y in canonical (pre-rotation) orientation: a north-facing
/// splitter with a base marker below it and two prong markers above, plus the
/// runs connecting them (belts, and undergrounds where an arm tunnels under an
/// obstruction). Runs are stored as cell sequences in split flow order (base →
/// splitter, splitter → prong), so SPLITTER_SPLIT places them as-is (source →
/// splitter → 2 sinks) and SPLITTER_MERGE reverses each sequence — rebuilt from
/// the reversed cells by [`path_to_placements`] so corners re-orient and
/// tunnels flip cleanly — into 2 sources → splitter → sink.
struct SplitterYPlan {
    item_value: i64,
    tiles: Vec<Cell>,
    base_pos: Cell,
    base_dir_split: Direction,
    prong1_pos: Cell,
    prong1_dir_split: Direction,
    prong2_pos: Cell,
    prong2_dir_split: Direction,
    stem: Vec<Cell>,
    arm1: Vec<Cell>,
    arm2: Vec<Cell>,
    total_entities: usize,
}

/// Draw one prong (a sink, in split terms) in the north region, facing *away*
/// from the splitter — so as a fed sink its input faces back toward the
/// splitter, and after the merge flip (a feeding source) it faces toward it.
/// The splitter sits at row `sy`, columns `sx..=sx+1`, strictly south of the
/// prong, so north is always an away-facing; add the horizontal away-facing
/// when the prong is off to one side. Returns `(pos, split-facing, input cell)`.
fn draw_prong(rng: &mut Rng, s: i64, sy: i64, sx: i64) -> Option<(Cell, Direction, Cell)> {
    let px = rng.randint(0, s - 1);
    let py = rng.randint(0, sy - 2);
    let mut away = vec![Direction::North];
    if px < sx {
        away.push(Direction::West); // splitter is east → face west
    } else if px > sx + 1 {
        away.push(Direction::East); // splitter is west → face east
    }
    let dir = away[rng.choice_index(away.len())];
    let d = dir.delta();
    let input = (px - d.0, py - d.1);
    if !in_grid(input, s) {
        return None;
    }
    Some(((px, py), dir, input))
}

/// Wire two splitter outputs to two prong inputs in a given order, the second
/// path routed around the first. Returns the two runs mapped to (prong1, prong2)
/// regardless of the order they were pathed in.
#[allow(clippy::too_many_arguments)]
fn wire_arms(
    s: i64,
    first_out: Cell,
    first_in: Cell,
    first_dir: Direction,
    first_is_prong1: bool,
    second_out: Cell,
    second_in: Cell,
    second_dir: Direction,
    base_blocked: &HashSet<Cell>,
) -> Option<(Vec<Cell>, Vec<Cell>)> {
    let mut b1 = base_blocked.clone();
    b1.extend([second_out, second_in]);
    let p1 = belt_cells(&find_belt_path(
        first_out,
        first_in,
        first_dir,
        s,
        &b1,
        Underground::On(None),
    )?);
    let mut b2 = base_blocked.clone();
    b2.extend(p1.iter().copied());
    b2.insert(first_out);
    let p2 = belt_cells(&find_belt_path(
        second_out,
        second_in,
        second_dir,
        s,
        &b2,
        Underground::On(None),
    )?);
    if first_is_prong1 {
        Some((p1, p2))
    } else {
        Some((p2, p1))
    }
}

/// Build a canonical (pre-rotation) splitter Y: a north-facing splitter with
/// the source ("base") just below it — a short straight stem — and two sinks
/// ("prongs") forward of it, facing outward. Among the choices of which
/// splitter input the stem uses and which output feeds which prong (in which
/// order), keep the wiring with the fewest placements, so no arm wraps the long
/// way around. Arms may tunnel under an obstruction via undergrounds. Returns
/// the plan, or `None` if this draw can't be wired.
fn canonical_split(
    size: usize,
    rng: &mut Rng,
    item_value: i64,
    max_entities: f64,
) -> Option<SplitterYPlan> {
    let s = size as i64;
    // Need ≥2 rows above the splitter for prongs and ≥2 below for the base.
    if s < 5 {
        return None;
    }
    let dd = Direction::North.delta(); // (0, -1)

    // Splitter: a horizontal 2-tile pair facing north.
    let sy = rng.randint(2, s - 3);
    let sx = rng.randint(0, s - 2);
    let tiles = vec![(sx, sy), (sx + 1, sy)];
    let tile_set: HashSet<Cell> = tiles.iter().copied().collect();
    let inputs: Vec<Cell> = tiles.iter().map(|&(x, y)| (x - dd.0, y - dd.1)).collect();
    let outputs: Vec<Cell> = tiles.iter().map(|&(x, y)| (x + dd.0, y + dd.1)).collect();

    // Base source under one of the splitter tiles, facing inward (north).
    let bx = sx + rng.randint(0, 1);
    let by = rng.randint(sy + 2, s - 1);
    let base_pos = (bx, by);
    let base_dir = Direction::North;
    let base_output = (base_pos.0, base_pos.1 - 1);

    // Two prong sinks forward of the base, facing away from the splitter.
    let (prong1_pos, prong1_dir, prong1_in) = draw_prong(rng, s, sy, sx)?;
    let (prong2_pos, prong2_dir, prong2_in) = draw_prong(rng, s, sy, sx)?;

    // Distinctness / bounds of markers and their connection cells.
    let markers = [base_pos, prong1_pos, prong2_pos];
    let all_fixed: HashSet<Cell> = tile_set.iter().copied().chain(markers).collect();
    if all_fixed.len() != tile_set.len() + 3 {
        return None;
    }
    let conns = [base_output, prong1_in, prong2_in];
    if conns.iter().any(|&c| !in_grid(c, s)) {
        return None;
    }
    let conn_set: HashSet<Cell> = conns.iter().copied().collect();
    if conn_set.len() != conns.len() || conn_set.iter().any(|c| all_fixed.contains(c)) {
        return None;
    }

    // Cells on the far side of each prong (the side it faces) are blocked so no
    // belt sneaks up behind a sink.
    let prong_backs: Vec<Cell> = [(prong1_pos, prong1_dir), (prong2_pos, prong2_dir)]
        .iter()
        .map(|&((px, py), d)| (px + d.delta().0, py + d.delta().1))
        .collect();

    // Search wiring combinations, keeping the one with the fewest placements.
    let mut best: Option<Wiring> = None;
    for stem_i in 0..2usize {
        let stem_in = inputs[stem_i];
        let other_in = inputs[1 - stem_i];
        let mut blocked_stem = all_fixed.clone();
        blocked_stem.extend(outputs.iter().copied());
        blocked_stem.extend([other_in, prong1_in, prong2_in]);
        blocked_stem.extend(prong_backs.iter().copied());
        let stem = match find_belt_path(
            base_output,
            stem_in,
            Direction::North,
            s,
            &blocked_stem,
            Underground::On(None),
        ) {
            Some(b) => belt_cells(&b),
            None => continue,
        };

        let mut base_blocked = all_fixed.clone();
        base_blocked.extend(stem.iter().copied());
        base_blocked.extend(inputs.iter().copied());
        base_blocked.extend(prong_backs.iter().copied());

        // Both output→prong assignments, both pathing orders.
        for &(o1, o2) in &[(0usize, 1usize), (1usize, 0usize)] {
            for &prong1_first in &[true, false] {
                let arms = if prong1_first {
                    wire_arms(
                        s,
                        outputs[o1],
                        prong1_in,
                        prong1_dir,
                        true,
                        outputs[o2],
                        prong2_in,
                        prong2_dir,
                        &base_blocked,
                    )
                } else {
                    wire_arms(
                        s,
                        outputs[o2],
                        prong2_in,
                        prong2_dir,
                        false,
                        outputs[o1],
                        prong1_in,
                        prong1_dir,
                        &base_blocked,
                    )
                };
                if let Some((arm1, arm2)) = arms {
                    let total = stem.len() + arm1.len() + arm2.len();
                    if best.as_ref().is_none_or(|(bt, ..)| total < *bt) {
                        best = Some((total, stem.clone(), arm1, arm2));
                    }
                }
            }
        }
    }

    let (total_belts, stem, arm1, arm2) = best?;
    let total_entities = total_belts + 1; // + splitter
    if (total_entities as f64) > max_entities {
        return None;
    }

    Some(SplitterYPlan {
        item_value,
        tiles,
        base_pos,
        base_dir_split: base_dir,
        prong1_pos,
        prong1_dir_split: prong1_dir,
        prong2_pos,
        prong2_dir_split: prong2_dir,
        stem,
        arm1,
        arm2,
        total_entities,
    })
}

/// Lay one belt run: for a split, the cells as-is ending in `split_end`; for a
/// merge, the reversed cells ending in `merge_end` (facings recomputed from the
/// reversed sequence, so corners are correct).
fn place_run(
    world: &mut World,
    cells: &[Cell],
    split_end: Direction,
    merge_end: Direction,
    merge: bool,
) -> Option<()> {
    let run = if merge {
        let rev: Vec<Cell> = cells.iter().rev().copied().collect();
        path_to_placements(&rev, merge_end)?
    } else {
        path_to_placements(cells, split_end)?
    };
    place_belts(world, &run);
    Some(())
}

/// Realise a [`SplitterYPlan`] as either SPLITTER_SPLIT (`merge = false`) or
/// SPLITTER_MERGE (`merge = true`, flow reversed).
fn place_plan(plan: &SplitterYPlan, size: usize, merge: bool) -> Option<World> {
    let mut world = World::empty(size, size);
    let splitter_dir = if merge {
        Direction::South
    } else {
        Direction::North
    };
    let (base_ent, prong_ent) = if merge {
        (Item::Sink, Item::Source)
    } else {
        (Item::Source, Item::Sink)
    };
    let flip = |d: Direction| if merge { d.opposite() } else { d };

    place_marker(
        &mut world,
        plan.base_pos,
        base_ent,
        flip(plan.base_dir_split),
        plan.item_value,
    );
    place_marker(
        &mut world,
        plan.prong1_pos,
        prong_ent,
        flip(plan.prong1_dir_split),
        plan.item_value,
    );
    place_marker(
        &mut world,
        plan.prong2_pos,
        prong_ent,
        flip(plan.prong2_dir_split),
        plan.item_value,
    );
    place_splitter(&mut world, &plan.tiles, splitter_dir);

    // Merge reverses every run; each then ends by feeding a south-facing
    // splitter (arms) or the south-facing base sink (stem).
    place_run(
        &mut world,
        &plan.stem,
        Direction::North,
        Direction::South,
        merge,
    )?;
    place_run(
        &mut world,
        &plan.arm1,
        plan.prong1_dir_split,
        Direction::South,
        merge,
    )?;
    place_run(
        &mut world,
        &plan.arm2,
        plan.prong2_dir_split,
        Direction::South,
        merge,
    )?;
    Some(world)
}

/// Build a splitter Y lesson. Both SPLITTER_SPLIT and SPLITTER_MERGE are the
/// same Y — a base wired through a splitter to two prongs — so both come from
/// one canonical (base-toward-south-edge, facing north) layout: SPLITTER_MERGE
/// reverses the flow, and a random number of 90° rotations then orients the
/// whole factory. The splitter is left removable (no protected positions) so
/// the policy learns to place it.
fn build_splitter_y(
    size: usize,
    rng: &mut Rng,
    random_item: bool,
    max_entities: f64,
    merge: bool,
) -> Option<BuiltFactory> {
    let pool = item_pool();
    let mut count = (500).max(size * size * 10);

    // NOTE: the item is drawn ONCE before the loop (as SPLITTER_SPLIT always has).
    let item_value = if random_item {
        pool[rng.choice_index(pool.len())]
    } else {
        Item::ElectronicCircuit as i64
    };

    while count > 0 {
        count -= 1;
        let plan = match canonical_split(size, rng, item_value, max_entities) {
            Some(p) => p,
            None => continue,
        };
        let mut world = match place_plan(&plan, size, merge) {
            Some(w) => w,
            None => continue,
        };
        for _ in 0..rng.choice_index(4) {
            world = rotate_world_cw(&world);
        }
        // Reject dead builds and any orphan tiles (a tunnel could otherwise
        // strand a belt off every source→sink path).
        let (deliveries, unreachable) = calc_throughput(&build_graph(&world));
        if factory_score(&deliveries) <= 0.0 || unreachable != 0 {
            continue;
        }
        return finish(world, plan.total_entities, Vec::new(), count);
    }
    None
}

fn build_splitter_split(
    size: usize,
    rng: &mut Rng,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    build_splitter_y(size, rng, random_item, max_entities, false)
}

fn build_splitter_merge(
    size: usize,
    rng: &mut Rng,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    build_splitter_y(size, rng, random_item, max_entities, true)
}

#[allow(unused)]
fn build_assemble_1in1out(size: usize, rng: &mut Rng, max_entities: f64) -> Option<BuiltFactory> {
    let s = size as i64;
    // The choice pool: 1-in-1-out recipes in all_recipes() order. The recipe
    // table guarantees it is non-empty, so the choice below can't underflow.
    let recipes: Vec<(Item, Recipe)> = all_recipes()
        .into_iter()
        .filter(|(_, r)| r.consumes.len() == 1 && r.produces.len() == 1)
        .collect();
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

        let all_perim = all_perim_set(ax, ay, s);
        let reserved: HashSet<Cell> = asm_tiles
            .iter()
            .copied()
            .chain(key_cells.iter().copied())
            .chain(all_perim.iter().copied())
            .collect();
        let available = available_cells(s, &reserved);
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
        let path1 = match find_belt_path(
            source_output,
            in_pickup,
            in_inserter_dir,
            s,
            &blocked1,
            Underground::Off,
        ) {
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
        let path2 = match find_belt_path(
            out_drop,
            sink_input,
            sink_dir,
            s,
            &blocked2,
            Underground::Off,
        ) {
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

fn build_move_via_ug_belt(
    size: usize,
    rng: &mut Rng,
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
        let path1: Vec<UgPlacement> = if source_drop == ug_down_pos {
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
            match find_belt_path(
                source_drop,
                ug_down_input,
                flow_dir,
                s,
                &blocked1,
                Underground::Off,
            ) {
                Some(p) => p,
                None => continue,
            }
        };

        // Path 2: UG_UP drop → sink_input (on the sink side).
        let path2: Vec<UgPlacement> = if sink_input == ug_up_pos {
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
            match find_belt_path(
                ug_up_drop,
                sink_input,
                sink_dir,
                s,
                &blocked2,
                Underground::Off,
            ) {
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

#[allow(unused)]
fn build_assemble_2in1out(size: usize, rng: &mut Rng, max_entities: f64) -> Option<BuiltFactory> {
    let s = size as i64;
    // The `random.choice` pool: 2-in-1-out recipes in all_recipes() order.
    let recipes: Vec<(Item, Recipe)> = all_recipes()
        .into_iter()
        .filter(|(_, r)| r.consumes.len() == 2 && r.produces.len() == 1)
        .collect();
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

        let all_perim = all_perim_set(ax, ay, s);
        let reserved: HashSet<Cell> = asm_tiles
            .iter()
            .copied()
            .chain(key_cells.iter().copied())
            .chain(all_perim.iter().copied())
            .collect();
        let available = available_cells(s, &reserved);
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
        let path_a = match find_belt_path(
            src_a_out,
            in_a_pickup,
            in_a_dir,
            s,
            &blocked_a,
            Underground::Off,
        ) {
            Some(p) => p,
            None => continue,
        };
        let path_a_cells = belt_cell_set(&path_a);

        // Path B: source B → input-B pickup.
        let mut blocked_b = fixed_cells.clone();
        blocked_b.extend([in_a_pickup, out_drop, src_a_out, sink_in]);
        blocked_b.extend(path_a_cells.iter().copied());
        let path_b = match find_belt_path(
            src_b_out,
            in_b_pickup,
            in_b_dir,
            s,
            &blocked_b,
            Underground::Off,
        ) {
            Some(p) => p,
            None => continue,
        };
        let path_b_cells = belt_cell_set(&path_b);

        // Path C: output drop → sink input.
        let mut blocked_c = fixed_cells.clone();
        blocked_c.extend([in_a_pickup, in_b_pickup, src_a_out, src_b_out]);
        blocked_c.extend(path_a_cells.iter().copied());
        blocked_c.extend(path_b_cells.iter().copied());
        let path_c =
            match find_belt_path(out_drop, sink_in, sink_dir, s, &blocked_c, Underground::Off) {
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

/// Build a MEMORISE_`N`_INGREDIENT_RECIPES factory: a single assembler fed and
/// drained by the most compact possible arms — every ingredient and product
/// travels `source → belt → inserter → assembler → inserter → belt → sink` with
/// **exactly one** belt between a source/sink and its inserter. The recipe is
/// drawn at random from the [`all_recipes`] entries that consume exactly
/// `n_ingredients` items, so the number of input arms equals `n_ingredients`
/// (one source per input) and there is one output arm per product. Each arm's
/// inserter sits on a randomly-chosen, non-corner assembler perimeter slot, and
/// the source/sink hangs off a randomly-chosen free neighbour of that arm's
/// belt. The assembler anchor itself is random.
///
/// Splitting the memorise lesson by ingredient count lets each lesson drill a
/// fixed input-arm count in isolation. The lesson teaches the policy to
/// *memorise* which items a recipe consumes and produces (and the assembler's
/// recipe tag), stripped of any long-belt routing — the routing is fixed at one
/// tile, so only the recipe identity and the immediate inserter/belt geometry
/// vary.
fn build_memorise_recipes(
    size: usize,
    rng: &mut Rng,
    max_entities: f64,
    n_ingredients: usize,
) -> Option<BuiltFactory> {
    let s = size as i64;
    // Only recipes with exactly `n_ingredients` inputs are eligible — the
    // lesson drills memorising recipe identity at a fixed input-arm count. If no
    // recipe matches (an ingredient count the table never uses) there is nothing
    // to build, so reject immediately.
    let recipes: Vec<(Item, Recipe)> = all_recipes()
        .into_iter()
        .filter(|(_, r)| r.consumes.len() == n_ingredients)
        .collect();
    if recipes.is_empty() {
        return None;
    }
    let mut count = (500).max(size * size * 16);

    while count > 0 {
        count -= 1;

        let (recipe_key, recipe) = recipes[rng.choice_index(recipes.len())].clone();
        let recipe_item_value = recipe_key as i64;
        // One arm per ingredient (inputs) and one per product (outputs).
        // Shuffle the ingredients so a recipe's items aren't always bound to
        // the same perimeter slots across seeds.
        let mut input_items: Vec<Item> = recipe.consumes.iter().map(|&(i, _)| i).collect();
        rng.shuffle(&mut input_items);
        let output_items: Vec<Item> = recipe.produces.iter().map(|&(i, _)| i).collect();
        let n_in = input_items.len();
        let n_arms = n_in + output_items.len();

        // Need one distinct perimeter slot per arm. With at most 5 ingredients
        // + 1 product this is always <= 12 = PERIM_SLOTS.len(), but guard
        // anyway so a future many-input recipe rejects cleanly.
        if n_arms > PERIM_SLOTS.len() {
            continue;
        }

        let ax = rng.randint(0, s - 3);
        let ay = rng.randint(0, s - 3);
        let asm_tiles: HashSet<Cell> = (0..3)
            .flat_map(|dx| (0..3).map(move |dy| (ax + dx, ay + dy)))
            .collect();
        // Source/sink markers must never sit on the assembler perimeter: a
        // source/sink there would be read as an inserter feeding/draining the
        // assembler directly, bypassing this arm's belt+inserter.
        let perim = all_perim_set(ax, ay, s);

        let slots = rng.sample(&PERIM_SLOTS, n_arms);

        // Plan every arm, reserving cells as we go so arms can't overlap. Any
        // arm that can't be placed rejects the whole candidate.
        let mut occupied: HashSet<Cell> = asm_tiles.clone();
        let mut inserters: Vec<(Cell, Direction)> = Vec::with_capacity(n_arms);
        let mut belts: Vec<UgPlacement> = Vec::with_capacity(n_arms);
        // (position, direction, carried-item, is_source)
        let mut markers: Vec<(Cell, Direction, i64, bool)> = Vec::with_capacity(n_arms);

        let mut ok = true;
        for (idx, &(off_x, off_y, in_dir, out_dir)) in slots.iter().enumerate() {
            let is_input = idx < n_in;
            let inserter_pos = (ax + off_x, ay + off_y);
            // Inserter faces INTO the assembler for inputs, OUT for outputs.
            let inserter_dir = if is_input { in_dir } else { out_dir };
            if !in_grid(inserter_pos, s) || occupied.contains(&inserter_pos) {
                ok = false;
                break;
            }

            // The single belt sits on the inserter's pickup cell (inputs) or
            // drop cell (outputs) — one step away along the inserter's facing.
            let dd = inserter_dir.delta();
            let belt_pos = if is_input {
                (inserter_pos.0 - dd.0, inserter_pos.1 - dd.1)
            } else {
                (inserter_pos.0 + dd.0, inserter_pos.1 + dd.1)
            };
            if !in_grid(belt_pos, s) || occupied.contains(&belt_pos) {
                ok = false;
                break;
            }

            // The source/sink hangs off any free neighbour of the belt other
            // than the inserter. Enumerate in BFS_DELTAS order for determinism.
            let mut candidates: Vec<Cell> = Vec::new();
            for &(ndx, ndy) in &BFS_DELTAS {
                let c = (belt_pos.0 + ndx, belt_pos.1 + ndy);
                if c == inserter_pos || !in_grid(c, s) {
                    continue;
                }
                if occupied.contains(&c) || perim.contains(&c) {
                    continue;
                }
                candidates.push(c);
            }
            if candidates.is_empty() {
                ok = false;
                break;
            }
            let marker_pos = candidates[rng.choice_index(candidates.len())];

            let (belt_dir, marker_dir, item_value, is_source) = if is_input {
                // Belt carries toward the inserter; source faces the belt so it
                // drops its item onto it.
                let src_dir =
                    match delta_to_dir(belt_pos.0 - marker_pos.0, belt_pos.1 - marker_pos.1) {
                        Some(d) => d,
                        None => {
                            ok = false;
                            break;
                        }
                    };
                (inserter_dir, src_dir, input_items[idx] as i64, true)
            } else {
                // Belt carries toward the sink; the sink faces away from the
                // belt so it pulls the item off it.
                let snk_dir =
                    match delta_to_dir(marker_pos.0 - belt_pos.0, marker_pos.1 - belt_pos.1) {
                        Some(d) => d,
                        None => {
                            ok = false;
                            break;
                        }
                    };
                (snk_dir, snk_dir, output_items[idx - n_in] as i64, false)
            };

            inserters.push((inserter_pos, inserter_dir));
            belts.push((belt_pos.0, belt_pos.1, belt_dir, Misc::None));
            markers.push((marker_pos, marker_dir, item_value, is_source));
            occupied.insert(inserter_pos);
            occupied.insert(belt_pos);
            occupied.insert(marker_pos);
        }
        if !ok {
            continue;
        }

        // Removable units: belts + inserters + the assembler (sources/sinks
        // are env-spawned and never blanked, so they don't count).
        let total_entities = inserters.len() + belts.len() + 1;
        if (total_entities as f64) > max_entities {
            continue;
        }

        let mut world = World::empty(size, size);
        place_assembler(&mut world, ax, ay, recipe_item_value);
        for &(pos, dir) in &inserters {
            place_inserter(&mut world, pos, dir);
        }
        place_belts(&mut world, &belts);
        for &(pos, dir, item_value, is_source) in &markers {
            let ent = if is_source { Item::Source } else { Item::Sink };
            place_marker(&mut world, pos, ent, dir, item_value);
        }

        if world_throughput(&world) <= 0.0 {
            continue;
        }

        return finish(world, total_entities, vec![], count);
    }
    None
}

/// Whether any underground tunnel in these routes is severed by a crossing:
/// the engine pairs an entrance with the FIRST underground tile it meets
/// ahead — whatever that tile's facing — so any other UG tile strictly inside
/// a tunnel's span breaks the pair and orphans the downstream side. Relies on
/// [`find_belt_paths`] emitting each tunnel's Down placement immediately
/// followed by its Up.
fn tunnels_crossed(paths: &[&[UgPlacement]]) -> bool {
    let ug_tiles: HashSet<Cell> = paths
        .iter()
        .flat_map(|p| p.iter())
        .filter(|&&(.., m)| m != Misc::None)
        .map(|&(x, y, ..)| (x, y))
        .collect();
    for path in paths {
        for pair in path.windows(2) {
            let (x, y, d, m) = pair[0];
            if m != Misc::UndergroundDown {
                continue;
            }
            let (ex, ey, ..) = pair[1];
            let (dx, dy) = d.delta();
            let (mut cx, mut cy) = (x + dx, y + dy);
            while (cx, cy) != (ex, ey) {
                if ug_tiles.contains(&(cx, cy)) {
                    return true;
                }
                cx += dx;
                cy += dy;
            }
        }
    }
    false
}

/// Build a FACTORY_1_INGREDIENT factory: a row of assemblers all crafting the
/// same 1-in-1-out recipe, fed from a shared input belt lane along one side
/// and drained onto a shared output lane along the other — the classic
/// lined-up production-row layout. Unlike the MEMORISE lessons (whose
/// source/sink hug the assembler), the source and sink sit at semi-arbitrary
/// free cells and are wired to the lane ends by the UG-aware belt router, so
/// routing distance varies per seed.
///
/// Throughput deliberately varies: the assembler count (1..row capacity), the
/// per-assembler input/output inserter counts (1-3 each) and the recipe's
/// crafting time combine so a given factory may be input-inserter limited,
/// recipe-speed limited, or output-inserter limited — giving the critic both
/// good and bad layouts to rank. Dead or orphan-tile candidates are prevented
/// constructively (the sink's faced cell stays empty so the sink never feeds
/// a belt and closes a cycle; crossed tunnels are rejected before placement);
/// the final throughput/orphan gate remains as a safety net.
///
/// Built in a canonical orientation — a horizontal assembler row, input lane
/// north, output lane south — then the whole world is rotated by a random
/// number of 90° turns. Each lane's flow direction is picked by trying both
/// and keeping the shorter source/sink route (ties random), the way a player
/// would feed a lane from whichever end is nearer.
fn build_factory_1_ingredient(
    size: usize,
    rng: &mut Rng,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    // Canonical footprint is 7 rows: input lane, input inserters, 3 assembler
    // rows, output inserters, output lane.
    if s < 7 {
        return None;
    }
    let recipes: Vec<(Item, Recipe)> = all_recipes()
        .into_iter()
        .filter(|(_, r)| r.consumes.len() == 1 && r.produces.len() == 1)
        .collect();
    if recipes.is_empty() {
        return None;
    }
    let mut count = (500).max(size * size * 16);

    while count > 0 {
        count -= 1;

        let (recipe_key, recipe) = recipes[rng.choice_index(recipes.len())].clone();
        let input_item_value = recipe.consumes.first().0 as i64;
        let output_item_value = recipe.produces.first().0 as i64;
        let recipe_item_value = recipe_key as i64;

        // Assembler row: `n_asm` 3×3 machines separated by `gap` empty columns.
        let gap = rng.randint(0, 1);
        let n_max = (s + gap) / (3 + gap); // 3n + gap(n-1) <= s
        let n_asm = rng.randint(1, n_max);
        let row_w = 3 * n_asm + gap * (n_asm - 1);
        let ax0 = rng.randint(0, s - row_w);
        let ay = rng.randint(2, s - 5);
        let in_lane_y = ay - 2;
        let out_lane_y = ay + 4;

        // Per assembler: 1-3 input inserters on the north side (facing south,
        // into the machine) and 1-3 output inserters on the south side (facing
        // south, away from it). Their pickup/drop cells fix the lane extents.
        let mut asm_tiles: HashSet<Cell> = HashSet::new();
        let mut inserters: Vec<(Cell, Direction)> = Vec::new();
        let mut pickup_xs: Vec<i64> = Vec::new();
        let mut drop_xs: Vec<i64> = Vec::new();
        let mut anchors: Vec<i64> = Vec::new();
        for i in 0..n_asm {
            let ax = ax0 + i * (3 + gap);
            anchors.push(ax);
            asm_tiles.extend((0..3).flat_map(|dx| (0..3).map(move |dy| (ax + dx, ay + dy))));
            let cols = [ax, ax + 1, ax + 2];
            let k_in = rng.randint(1, 3) as usize;
            for &x in &rng.sample(&cols, k_in) {
                inserters.push(((x, ay - 1), Direction::South));
                pickup_xs.push(x);
            }
            let k_out = rng.randint(1, 3) as usize;
            for &x in &rng.sample(&cols, k_out) {
                inserters.push(((x, ay + 3), Direction::South));
                drop_xs.push(x);
            }
        }

        // The lanes span exactly their pickup/drop cells — no belt may extend
        // past the last pickup or the exit, or it would strand items (orphan
        // tiles). The input lane's head is fed by the source route; the output
        // lane's exit cell starts the sink route.
        // Both sides have >= 1 inserter per assembler, so min/max exist.
        let span = |xs: &[i64]| Some((*xs.iter().min()?, *xs.iter().max()?));
        let Some((in_lo, in_hi)) = span(&pickup_xs) else {
            continue;
        };
        let Some((out_lo, out_hi)) = span(&drop_xs) else {
            continue;
        };

        // Cells no route or marker may take: machines, inserters, both lanes.
        let mut reserved: HashSet<Cell> = asm_tiles.clone();
        reserved.extend(inserters.iter().map(|&(c, _)| c));
        reserved.extend((in_lo..=in_hi).map(|x| (x, in_lane_y)));
        reserved.extend((out_lo..=out_hi).map(|x| (x, out_lane_y)));

        // Source/sink markers keep off every assembler perimeter (a marker
        // there reads as feeding/draining the machine directly).
        let mut marker_excluded = reserved.clone();
        for &ax in &anchors {
            marker_excluded.extend(all_perim_set(ax, ay, s));
        }
        let available = available_cells(s, &marker_excluded);
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
        let source_out = (source_pos.0 + ds.0, source_pos.1 + ds.1);
        let sink_in = (sink_pos.0 - dk.0, sink_pos.1 - dk.1);
        let conns = [source_out, sink_in];
        if conns
            .iter()
            .any(|&c| !in_grid(c, s) || reserved.contains(&c))
        {
            continue;
        }
        if source_out == sink_in
            || source_out == sink_pos
            || sink_in == source_pos
            || source_pos == sink_pos
        {
            continue;
        }
        // The cell the sink FACES must stay empty: sinks connect like belts,
        // so a sink pointing into a belt would FEED it — output items leaking
        // back toward the input lane close a cycle, which the throughput
        // engine scores as a dead factory.
        let sink_face = (sink_pos.0 + dk.0, sink_pos.1 + dk.1);
        if in_grid(sink_face, s)
            && (reserved.contains(&sink_face) || sink_face == source_pos || sink_face == source_out)
        {
            continue;
        }

        // Fixed obstacles common to both route searches; lane cells are added
        // per direction below (a route-owned head/exit cell must stay open).
        let mut fixed: HashSet<Cell> = asm_tiles.clone();
        fixed.extend(inserters.iter().map(|&(c, _)| c));
        fixed.extend([source_pos, sink_pos, sink_face]);

        // Route 1: source drop → input lane head, arriving in the lane's flow
        // direction (the source feeds the start head-on, so the route may open
        // with a tunnel entrance). Both lane directions are tried and the
        // shorter route wins (ties broken by the shuffle) — a player feeds the
        // lane from whichever end is nearer the source instead of snaking the
        // belt around the row.
        let mut dir_choices = [Direction::East, Direction::West];
        rng.shuffle(&mut dir_choices);
        // (in_dir, head, overshoot, route)
        let mut best1: Option<(Direction, Cell, Cell, Vec<UgPlacement>)> = None;
        for &d in &dir_choices {
            let dd = d.delta();
            let (head, tail) = if d == Direction::East {
                ((in_lo, in_lane_y), (in_hi, in_lane_y))
            } else {
                ((in_hi, in_lane_y), (in_lo, in_lane_y))
            };
            // The lane's tail belt points one past the lane (the overshoot
            // cell): an entity there would siphon the input items off the
            // lane, so it must stay empty for this direction to be usable.
            let overshoot = (tail.0 + dd.0, tail.1 + dd.1);
            if [source_pos, sink_pos, source_out, sink_in, sink_face].contains(&overshoot) {
                continue;
            }
            let mut blocked = fixed.clone();
            blocked.extend(
                (in_lo..=in_hi)
                    .map(|x| (x, in_lane_y))
                    .filter(|&c| c != head),
            );
            blocked.extend((out_lo..=out_hi).map(|x| (x, out_lane_y)));
            blocked.extend([overshoot, sink_in]);
            if let Some(p) = find_belt_path(
                source_out,
                head,
                d,
                s,
                &blocked,
                Underground::On(Some(source_dir)),
            ) {
                if best1.as_ref().is_none_or(|(.., bp)| p.len() < bp.len()) {
                    best1 = Some((d, head, overshoot, p));
                }
            }
        }
        let Some((in_dir, in_head, overshoot, path1)) = best1 else {
            continue;
        };
        let path1_cells = belt_cell_set(&path1);

        // Route 2: output lane exit → sink input, again keeping the shorter of
        // the two lane directions. Items arrive on the exit cell travelling
        // the lane direction (from the lane belt behind it, or dropped by the
        // inserter above it), which also rules out a 180° reversal.
        rng.shuffle(&mut dir_choices);
        // (out_dir, exit, route)
        let mut best2: Option<(Direction, Cell, Vec<UgPlacement>)> = None;
        for &d in &dir_choices {
            let exit = if d == Direction::East {
                (out_hi, out_lane_y)
            } else {
                (out_lo, out_lane_y)
            };
            let mut blocked = fixed.clone();
            blocked.extend((in_lo..=in_hi).map(|x| (x, in_lane_y)));
            blocked.extend(
                (out_lo..=out_hi)
                    .map(|x| (x, out_lane_y))
                    .filter(|&c| c != exit),
            );
            blocked.insert(overshoot);
            blocked.extend(path1_cells.iter().copied());
            if let Some(p) = find_belt_path(
                exit,
                sink_in,
                sink_dir,
                s,
                &blocked,
                Underground::On(Some(d)),
            ) {
                if best2.as_ref().is_none_or(|(.., bp)| p.len() < bp.len()) {
                    best2 = Some((d, exit, p));
                }
            }
        }
        let Some((out_dir, out_exit, path2)) = best2 else {
            continue;
        };

        // The two routes are pathed independently, so one's tunnel may pass
        // under the other's underground tile — which severs the tunnel (the
        // engine pairs an entrance with the FIRST UG tile ahead, whatever its
        // facing) and orphans everything downstream. Reject such crossings.
        if tunnels_crossed(&[&path1, &path2]) {
            continue;
        }

        // Lane belts, minus the two route-owned cells (head/exit).
        let mut lane_belts: Vec<UgPlacement> = Vec::new();
        for x in in_lo..=in_hi {
            if (x, in_lane_y) != in_head {
                lane_belts.push((x, in_lane_y, in_dir, Misc::None));
            }
        }
        for x in out_lo..=out_hi {
            if (x, out_lane_y) != out_exit {
                lane_belts.push((x, out_lane_y, out_dir, Misc::None));
            }
        }

        // Every assembler counts as ONE removable unit (matching
        // blank_entities), as do inserters, lane belts and route placements.
        let total_entities =
            lane_belts.len() + path1.len() + path2.len() + inserters.len() + n_asm as usize;
        if (total_entities as f64) > max_entities {
            continue;
        }

        let mut world = World::empty(size, size);
        for &ax in &anchors {
            place_assembler(&mut world, ax, ay, recipe_item_value);
        }
        for &(pos, dir) in &inserters {
            place_inserter(&mut world, pos, dir);
        }
        place_belts(&mut world, &lane_belts);
        place_belts(&mut world, &path1);
        place_belts(&mut world, &path2);
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

        for _ in 0..rng.choice_index(4) {
            world = rotate_world_cw(&world);
        }

        // Reject dead builds and orphan tiles; anything slower-but-flowing is
        // kept on purpose (the lesson wants a throughput spread).
        let (deliveries, unreachable) = calc_throughput(&build_graph(&world));
        if factory_score(&deliveries) <= 0.0 || unreachable != 0 {
            continue;
        }

        return finish(world, total_entities, vec![], count);
    }
    None
}

// ── CROSS_UNDER_BELT: an obstruction belt-line cut + a crossing that tunnels
// under it ──────────────────────────────────────────────────────────────────

/// Max tiles spanned by an underground tunnel (engine reach is 6, `entities.rs`).
const UNDERGROUND_MAX_OFFSET: i64 = 5;

/// A belt/underground placement `(x, y, facing, misc)`: `Misc::None` for a plain
/// belt, `UndergroundDown`/`UndergroundUp` for the tunnel ends.
type UgPlacement = (i64, i64, Direction, Misc);

/// 4-connected components of the free cells (in-grid, not in `obstruction`),
/// returned largest-first (stable on ties).
fn free_components(obstruction: &HashSet<Cell>, s: i64) -> Vec<Vec<Cell>> {
    let mut seen: HashSet<Cell> = HashSet::new();
    let mut comps: Vec<Vec<Cell>> = Vec::new();
    for x in 0..s {
        for y in 0..s {
            let cell = (x, y);
            if obstruction.contains(&cell) || seen.contains(&cell) {
                continue;
            }
            let mut stack = vec![cell];
            seen.insert(cell);
            let mut comp = vec![cell];
            while let Some((cx, cy)) = stack.pop() {
                for (dx, dy) in BFS_DELTAS {
                    let nb = (cx + dx, cy + dy);
                    if in_grid(nb, s) && !obstruction.contains(&nb) && !seen.contains(&nb) {
                        seen.insert(nb);
                        stack.push(nb);
                        comp.push(nb);
                    }
                }
            }
            comps.push(comp);
        }
    }
    comps.sort_by_key(|c| Reverse(c.len()));
    comps
}

/// A source/sink pair on the two opposite edges of one axis, with a random
/// flow direction. `span_x` spans the left/right edges (flows E or W);
/// otherwise the top/bottom edges (flows N or S).
fn edge_endpoints(rng: &mut Rng, span_x: bool, s: i64) -> (Cell, Cell, Direction) {
    let (a, b, fwd, bwd) = if span_x {
        let ay = rng.randint(0, s - 1);
        let by = rng.randint(0, s - 1);
        ((0, ay), (s - 1, by), Direction::East, Direction::West)
    } else {
        let ax = rng.randint(0, s - 1);
        let bx = rng.randint(0, s - 1);
        ((ax, 0), (bx, s - 1), Direction::South, Direction::North)
    };
    if rng.choice_index(2) == 0 {
        (a, b, fwd)
    } else {
        (b, a, bwd)
    }
}

/// Pick a random crossing endpoint inside `comp` (avoiding cells `near` the
/// obstruction) plus a facing whose belt cell — the source's drop, or the cell
/// feeding the sink — is also free in the same component. Returns
/// `(tile, facing, belt_cell)` or `None`.
fn pick_endpoint(
    rng: &mut Rng,
    comp: &[Cell],
    near: &HashSet<Cell>,
    is_source: bool,
) -> Option<(Cell, Direction, Cell)> {
    let comp_set: HashSet<Cell> = comp.iter().copied().collect();
    let mut cands: Vec<Cell> = comp.iter().copied().filter(|c| !near.contains(c)).collect();
    rng.shuffle(&mut cands);
    let mut dirs: Vec<Direction> = DIRS.to_vec();
    for &cell in &cands {
        rng.shuffle(&mut dirs);
        for &d in &dirs {
            let (dx, dy) = d.delta();
            let nb = if is_source {
                (cell.0 + dx, cell.1 + dy)
            } else {
                (cell.0 - dx, cell.1 - dy)
            };
            if comp_set.contains(&nb) {
                return Some((cell, d, nb));
            }
        }
    }
    None
}

/// CROSS_UNDER_BELT: an obstruction belt line (source → belts → sink) runs
/// between two opposite edges, forming a winding CUT that separates the grid into
/// two halves. A second source/sink pair sits one per side; the crossing that
/// joins them is routed with [`find_belt_paths`] (underground enabled), so the
/// only way across is to tunnel UNDER the cut. The two lines carry distinct
/// items, so gating on
/// `tp >= belt_flow` (both delivered) and no orphan tiles rejects a crossing that
/// fails to connect.
fn build_cross_under_belt(
    size: usize,
    rng: &mut Rng,
    random_item: bool,
    max_entities: f64,
) -> Option<BuiltFactory> {
    let s = size as i64;
    if s < 5 {
        return None;
    }
    let belt_flow = Item::TransportBelt.flow_rate();
    let pool = item_pool();
    let (cross_item, obs_item) = if random_item {
        let two = rng.sample(&pool, 2);
        (two[0], two[1])
    } else {
        (Item::ElectronicCircuit as i64, Item::CopperCable as i64)
    };

    let mut count = (500).max(size * size * 16);
    while count > 0 {
        count -= 1;

        // The obstruction runs between two opposite edges, forming a cut.
        let obs_span_x = rng.choice_index(2) == 0;
        let (obs_source, obs_sink, obs_dir) = edge_endpoints(rng, obs_span_x, s);
        let od = obs_dir.delta();
        let obs_start = (obs_source.0 + od.0, obs_source.1 + od.1);
        let obs_end = (obs_sink.0 - od.0, obs_sink.1 - od.1);

        // Route the obstruction (plain belts) between its two edges.
        let mut obs_blocked: HashSet<Cell> = HashSet::new();
        obs_blocked.insert(obs_source);
        obs_blocked.insert(obs_sink);
        let obs_path = match find_belt_path(
            obs_start,
            obs_end,
            obs_dir,
            s,
            &obs_blocked,
            Underground::Off,
        ) {
            Some(p) => p,
            None => continue,
        };
        let mut obstruction_tiles: HashSet<Cell> = belt_cell_set(&obs_path);
        obstruction_tiles.insert(obs_source);
        obstruction_tiles.insert(obs_sink);

        // The cut splits the remaining cells into two sides; the crossing's
        // source and sink go one on each side, so the only route between them
        // is under the cut.
        let comps = free_components(&obstruction_tiles, s);
        if comps.len() < 2 {
            continue;
        }
        let (mut comp_a, mut comp_b) = (comps[0].clone(), comps[1].clone());
        if rng.choice_index(2) == 0 {
            std::mem::swap(&mut comp_a, &mut comp_b);
        }

        // Keep at least one clear tile between each endpoint and the
        // obstruction (the source/sink have room; the belts/UG may hug it).
        let mut near: HashSet<Cell> = HashSet::new();
        for &(ox, oy) in &obstruction_tiles {
            near.insert((ox, oy));
            for (dx, dy) in BFS_DELTAS {
                near.insert((ox + dx, oy + dy));
            }
        }

        let (cross_source, source_dir, cross_start) = match pick_endpoint(rng, &comp_a, &near, true)
        {
            Some(v) => v,
            None => continue,
        };
        let (cross_sink, sink_dir, cross_end) = match pick_endpoint(rng, &comp_b, &near, false) {
            Some(v) => v,
            None => continue,
        };

        // Route the crossing UNDER the obstruction, allowing tunnels. The
        // endpoints are in different components, so the path must cross the
        // cut.
        let mut cross_blocked = obstruction_tiles.clone();
        cross_blocked.insert(cross_source);
        cross_blocked.insert(cross_sink);
        let cross_path = match find_belt_path(
            cross_start,
            cross_end,
            sink_dir,
            s,
            &cross_blocked,
            Underground::On(Some(source_dir)),
        ) {
            Some(p) => p,
            None => continue,
        };
        if !cross_path.iter().any(|&(_, _, _, m)| m != Misc::None) {
            continue;
        }

        // Build the world.
        let mut world = World::empty(size, size);
        place_marker(&mut world, obs_source, Item::Source, obs_dir, obs_item);
        place_marker(&mut world, obs_sink, Item::Sink, obs_dir, obs_item);
        place_belts(&mut world, &obs_path);
        place_marker(
            &mut world,
            cross_source,
            Item::Source,
            source_dir,
            cross_item,
        );
        place_marker(&mut world, cross_sink, Item::Sink, sink_dir, cross_item);
        place_belts(&mut world, &cross_path);

        // Both lines must deliver at full belt speed, and no entity may be an
        // orphan (off every source→sink path).
        let (deliveries, unreachable) = calc_throughput(&build_graph(&world));
        if factory_score(&deliveries) < belt_flow - 1e-6 || unreachable != 0 {
            continue;
        }

        let total_entities = cross_path.len();
        if (total_entities as f64) > max_entities {
            continue;
        }

        // Protect the obstruction cut so blank_entities never removes it.
        let mut protected_positions: Vec<(usize, usize)> = obstruction_tiles
            .iter()
            .map(|&(x, y)| (x as usize, y as usize))
            .collect();
        protected_positions.sort_unstable();
        return finish(world, total_entities, protected_positions, count);
    }
    None
}

/// Wrap a finished factory, but honor the rejection-sampling budget: a
/// factory found on the very attempt that drove `count` to 0 is discarded
/// (returns `None`), so an exhausted budget always means "no factory".
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
#[allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]
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
    fn test_memorise_recipes_smoke() {
        // Each per-ingredient-count memorise lesson should produce
        // positive-throughput factories with one belt per arm (belts ==
        // inserters), exactly one assembler unit, one sink, and exactly
        // `n_ingredients` sources (one input arm per ingredient).
        let kinds = [
            (LessonKind::Memorise1IngredientRecipes, 1),
            (LessonKind::Memorise2IngredientRecipes, 2),
            (LessonKind::Memorise3IngredientRecipes, 3),
            (LessonKind::Memorise4IngredientRecipes, 4),
            (LessonKind::Memorise5IngredientRecipes, 5),
        ];
        for (kind, n_ingredients) in kinds {
            let mut built = 0;
            for seed in 0..50u64 {
                if let Some(f) = build_factory(11, kind, seed, true, f64::INFINITY) {
                    assert!(world_throughput(&f.world) > 0.0, "{kind:?} seed={seed}");

                    let mut n_belt = 0;
                    let mut n_inserter = 0;
                    let mut n_assembler = 0;
                    let mut n_source = 0;
                    let mut n_sink = 0;
                    for x in 0..f.world.width() {
                        for y in 0..f.world.height() {
                            match f.world.entity_at(x, y) {
                                Some(Item::TransportBelt) => n_belt += 1,
                                Some(Item::Inserter) => n_inserter += 1,
                                Some(Item::AssemblingMachine1) => n_assembler += 1,
                                Some(Item::Source) => n_source += 1,
                                Some(Item::Sink) => n_sink += 1,
                                _ => {}
                            }
                        }
                    }
                    // One belt per arm == one belt per inserter.
                    assert_eq!(
                        n_belt, n_inserter,
                        "{kind:?} seed={seed}: belts != inserters"
                    );
                    // The assembler is 3x3 → 9 tiles.
                    assert_eq!(n_assembler, 9, "{kind:?} seed={seed}: assembler not 3x3");
                    // One source per ingredient — the lesson's defining property.
                    assert_eq!(
                        n_source, n_ingredients,
                        "{kind:?} seed={seed}: source count != ingredient count"
                    );
                    assert_eq!(n_sink, 1, "{kind:?} seed={seed}: expected exactly one sink");
                    // Inserters = sources + sinks (one inserter per arm).
                    assert_eq!(
                        n_inserter,
                        n_source + n_sink,
                        "{kind:?} seed={seed}: inserter count != arm count"
                    );
                    built += 1;
                }
            }
            assert!(built > 40, "{kind:?}: most seeds should build, got {built}");
        }
    }

    #[test]
    fn test_factory_1_ingredient_smoke() {
        // Positive throughput, exactly one source and one sink, whole 3×3
        // assemblers, and per-assembler inserter counts within 1-3 per side
        // (2-6 total per machine). Counts vary across seeds.
        let mut built = 0;
        let mut asm_counts: HashSet<usize> = HashSet::new();
        for seed in 0..50u64 {
            let Some(f) = build_factory(
                11,
                LessonKind::Factory1Ingredient,
                seed,
                true,
                f64::INFINITY,
            ) else {
                continue;
            };
            built += 1;
            let (tp, unreachable) = tp_unreachable(&f.world);
            assert!(tp > 0.0, "seed={seed}");
            assert_eq!(unreachable, 0, "seed={seed} has orphan tiles");
            assert_eq!(count_entity(&f.world, Item::Source), 1, "seed={seed}");
            assert_eq!(count_entity(&f.world, Item::Sink), 1, "seed={seed}");
            let asm_tiles = count_entity(&f.world, Item::AssemblingMachine1);
            assert!(
                asm_tiles > 0 && asm_tiles.is_multiple_of(9),
                "seed={seed}: assembler tiles {asm_tiles} not whole 3x3 machines"
            );
            let n_asm = asm_tiles / 9;
            asm_counts.insert(n_asm);
            let n_inserter = count_entity(&f.world, Item::Inserter);
            assert!(
                (2 * n_asm..=6 * n_asm).contains(&n_inserter),
                "seed={seed}: {n_inserter} inserters for {n_asm} assemblers"
            );
        }
        assert!(built > 40, "most seeds should build, got {built}");
        assert!(asm_counts.len() > 1, "assembler count never varied");
    }

    #[test]
    fn test_delta_to_dir_roundtrip() {
        for d in DIRS {
            let (dx, dy) = d.delta();
            assert_eq!(delta_to_dir(dx, dy), Some(d));
        }
        assert_eq!(delta_to_dir(2, 0), None);
    }

    fn count_entity(world: &World, item: Item) -> usize {
        let mut n = 0;
        for x in 0..world.width() {
            for y in 0..world.height() {
                if world.get(x, y, Channel::Entities) == item as i64 {
                    n += 1;
                }
            }
        }
        n
    }

    fn wall(col: i64, size: i64) -> HashSet<Cell> {
        (0..size).map(|y| (col, y)).collect()
    }

    /// Build a CROSS_UNDER_BELT factory, panicking on the rare reject.
    fn build_cub(size: usize, seed: u64) -> BuiltFactory {
        build_factory(size, LessonKind::CrossUnderBelt, seed, true, f64::INFINITY)
            .unwrap_or_else(|| panic!("seed={seed} failed to build"))
    }

    /// Throughput score and orphan-tile count of a placed world.
    fn tp_unreachable(world: &World) -> (f64, usize) {
        let (deliveries, unreachable) = calc_throughput(&build_graph(world));
        (factory_score(&deliveries), unreachable)
    }

    /// The obstruction (its source/sink + belt cut) is `protected_positions`;
    /// the crossing's source/sink are everything else.
    struct Layout {
        ug_downs: Vec<Cell>,
        ug_ups: Vec<Cell>,
        flow_dir: Option<Direction>,
        obstruction: HashSet<Cell>,
        obs_source: Cell,
        obs_sink: Cell,
        cross_source: Cell,
        cross_sink: Cell,
    }

    fn layout(f: &BuiltFactory) -> Layout {
        let w = &f.world;
        let obstruction: HashSet<Cell> = f
            .protected_positions
            .iter()
            .map(|&(x, y)| (x as i64, y as i64))
            .collect();
        let (mut sources, mut sinks) = (Vec::new(), Vec::new());
        let (mut ug_downs, mut ug_ups) = (Vec::new(), Vec::new());
        for x in 0..w.width() {
            for y in 0..w.height() {
                let cell = (x as i64, y as i64);
                match w.get(x, y, Channel::Entities) {
                    e if e == Item::Source as i64 => sources.push(cell),
                    e if e == Item::Sink as i64 => sinks.push(cell),
                    e if e == Item::UndergroundBelt as i64 => {
                        match Misc::from_i64(w.get(x, y, Channel::Misc)) {
                            Misc::UndergroundDown => ug_downs.push(cell),
                            Misc::UndergroundUp => ug_ups.push(cell),
                            _ => {}
                        }
                    }
                    _ => {}
                }
            }
        }
        let flow_dir = ug_downs
            .first()
            .map(|&(x, y)| Direction::from_i64(w.get(x as usize, y as usize, Channel::Direction)));
        let find = |v: &[Cell], inside: bool| {
            *v.iter()
                .find(|c| obstruction.contains(c) == inside)
                .expect("expected one obstruction and one crossing endpoint")
        };
        Layout {
            obs_source: find(&sources, true),
            cross_source: find(&sources, false),
            obs_sink: find(&sinks, true),
            cross_sink: find(&sinks, false),
            ug_downs,
            ug_ups,
            flow_dir,
            obstruction,
        }
    }

    #[test]
    fn test_cross_under_belt_smoke() {
        let belt_flow = Item::TransportBelt.flow_rate();
        let mut built = 0;
        for seed in 0..60u64 {
            let f = match build_factory(12, LessonKind::CrossUnderBelt, seed, true, f64::INFINITY) {
                Some(f) => f,
                None => continue,
            };
            built += 1;
            let (tp, unreachable) = tp_unreachable(&f.world);
            // Both lines deliver at full belt speed; nothing is orphaned.
            assert!(tp >= belt_flow - 1e-6, "seed={seed} tp={tp}");
            assert_eq!(unreachable, 0, "seed={seed} has orphan tiles");
            // One source/sink pair for the obstruction, one for the crossing.
            assert_eq!(count_entity(&f.world, Item::Source), 2, "seed={seed}");
            assert_eq!(count_entity(&f.world, Item::Sink), 2, "seed={seed}");
            // The crossing tunnels under the cut, so a UG pair is present.
            let ug = count_entity(&f.world, Item::UndergroundBelt);
            assert!(ug >= 2 && ug.is_multiple_of(2), "seed={seed} ug={ug}");
        }
        assert!(built > 50, "most seeds should build, got {built}");
    }

    #[test]
    fn test_cross_under_belt_two_distinct_items() {
        // The obstruction and crossing carry different items (independent flows).
        for seed in 0..30u64 {
            let f = match build_factory(12, LessonKind::CrossUnderBelt, seed, true, f64::INFINITY) {
                Some(f) => f,
                None => continue,
            };
            let mut src_items = Vec::new();
            for x in 0..f.world.width() {
                for y in 0..f.world.height() {
                    if f.world.get(x, y, Channel::Entities) == Item::Source as i64 {
                        src_items.push(f.world.get(x, y, Channel::Items));
                    }
                }
            }
            assert_eq!(src_items.len(), 2);
            assert_ne!(src_items[0], src_items[1], "seed={seed}");
        }
    }

    #[test]
    fn test_ug_router_open_space_stays_on_surface() {
        // Nothing blocked → a tunnel is never cheaper than walking.
        let path = find_belt_path(
            (1, 4),
            (7, 4),
            Direction::East,
            9,
            &HashSet::new(),
            Underground::On(None),
        )
        .unwrap();
        assert!(path.iter().all(|&(_, _, _, m)| m == Misc::None));
    }

    #[test]
    fn test_shortest_n_caps_and_enumerates() {
        // Open grid, (0,0) → (2,2): the 6 monotone Manhattan routes are the only
        // shortest ones. `shortest_n` caps how many come back; -1 returns all.
        let free = HashSet::new();
        let all = find_belt_paths(
            (0, 0),
            (2, 2),
            Direction::East,
            5,
            &free,
            Underground::Off,
            -1,
        );
        assert_eq!(all.len(), 6);
        for p in &all {
            // 5 cells (0,0)..=(2,2), all plain belts, no tile reused.
            assert_eq!(p.len(), 5);
            assert!(p.iter().all(|&(_, _, _, m)| m == Misc::None));
            let cells: HashSet<Cell> = p.iter().map(|&(x, y, _, _)| (x, y)).collect();
            assert_eq!(cells.len(), 5);
            assert_eq!(
                p.last().map(|&(x, y, d, _)| (x, y, d)),
                Some((2, 2, Direction::East))
            );
        }
        // A positive cap returns exactly that many; 1 matches `find_belt_path`.
        assert_eq!(
            find_belt_paths(
                (0, 0),
                (2, 2),
                Direction::East,
                5,
                &free,
                Underground::Off,
                2
            )
            .len(),
            2
        );
        assert_eq!(
            find_belt_paths(
                (0, 0),
                (2, 2),
                Direction::East,
                5,
                &free,
                Underground::Off,
                1
            )
            .len(),
            1
        );
        // Unreachable target → no routes, whatever the cap.
        let blocked = wall(1, 5);
        assert!(find_belt_paths(
            (0, 0),
            (2, 2),
            Direction::East,
            5,
            &blocked,
            Underground::Off,
            -1
        )
        .is_empty());
    }

    #[test]
    fn test_ug_router_tunnels_under_a_wall() {
        let w = wall(4, 9);
        // Plain BFS can't cross a full-height wall.
        assert!(find_belt_path((1, 4), (7, 4), Direction::East, 9, &w, Underground::Off).is_none());
        // The UG-aware router tunnels under it with one entrance/exit pair.
        let path = find_belt_path(
            (1, 4),
            (7, 4),
            Direction::East,
            9,
            &w,
            Underground::On(None),
        )
        .unwrap();
        let downs: Vec<_> = path
            .iter()
            .filter(|&&(_, _, _, m)| m == Misc::UndergroundDown)
            .collect();
        let ups: Vec<_> = path
            .iter()
            .filter(|&&(_, _, _, m)| m == Misc::UndergroundUp)
            .collect();
        assert_eq!(downs.len(), 1);
        assert_eq!(ups.len(), 1);
    }

    #[test]
    fn test_ug_router_minimal_span_hugs_the_wall() {
        let w = wall(4, 9);
        let path = find_belt_path(
            (1, 4),
            (8, 4),
            Direction::East,
            9,
            &w,
            Underground::On(None),
        )
        .unwrap();
        let down = path
            .iter()
            .find(|&&(_, _, _, m)| m == Misc::UndergroundDown)
            .unwrap();
        let up = path
            .iter()
            .find(|&&(_, _, _, m)| m == Misc::UndergroundUp)
            .unwrap();
        // Entrance just before the wall, exit just after; both face the flow.
        assert_eq!((down.0, down.1, down.2), (3, 4, Direction::East));
        assert_eq!((up.0, up.1, up.2), (5, 4, Direction::East));
    }

    #[test]
    fn test_ug_router_start_can_be_entrance() {
        // start_dir given (source feeds `start` head-on) + a wall right after
        // start → the entrance sits on `start` itself (source → UG_DOWN).
        let w = wall(2, 9);
        let path = find_belt_path(
            (1, 4),
            (5, 4),
            Direction::East,
            9,
            &w,
            Underground::On(Some(Direction::East)),
        )
        .unwrap();
        assert_eq!(path[0], (1, 4, Direction::East, Misc::UndergroundDown));
    }

    #[test]
    fn test_ug_router_surfaces_into_sink() {
        // Wall right before `end` → the exit lands on `end` (UG_UP → sink).
        let w = wall(6, 9);
        let path = find_belt_path(
            (1, 4),
            (7, 4),
            Direction::East,
            9,
            &w,
            Underground::On(None),
        )
        .unwrap();
        assert_eq!(
            *path.last().unwrap(),
            (7, 4, Direction::East, Misc::UndergroundUp)
        );
    }

    #[test]
    fn test_ug_router_never_reuses_a_tile() {
        // A wall with a gap at y==0 (a go-around exists) across several start
        // facings: whatever the router returns, every tile is distinct.
        let w: HashSet<Cell> = (1..9).map(|y| (4, y)).collect();
        for sd in [None, Some(Direction::North), Some(Direction::East)] {
            if let Some(path) =
                find_belt_path((1, 4), (7, 4), Direction::East, 9, &w, Underground::On(sd))
            {
                let mut seen = HashSet::new();
                for &(x, y, _, _) in &path {
                    assert!(seen.insert((x, y)), "tile reused with start_dir={sd:?}");
                }
            }
        }
    }

    #[test]
    fn test_cross_deterministic() {
        // Same seed → byte-identical world.
        for seed in 0..5u64 {
            let (a, b) = (build_cub(12, seed), build_cub(12, seed));
            for x in 0..a.world.width() {
                for y in 0..a.world.height() {
                    for ch in [
                        Channel::Entities,
                        Channel::Direction,
                        Channel::Items,
                        Channel::Misc,
                    ] {
                        assert_eq!(a.world.get(x, y, ch), b.world.get(x, y, ch), "seed={seed}");
                    }
                }
            }
        }
    }

    #[test]
    fn test_cross_obstruction_is_connected_cut_between_opposite_edges() {
        for seed in 0..40u64 {
            let info = layout(&build_cub(8, seed));
            assert!(info.obstruction.contains(&info.obs_source));
            assert!(info.obstruction.contains(&info.obs_sink));
            // Spans one axis edge-to-edge.
            let span_x = {
                let xs: HashSet<i64> = [info.obs_source.0, info.obs_sink.0].into();
                xs == HashSet::from([0, 7])
            };
            let span_y = {
                let ys: HashSet<i64> = [info.obs_source.1, info.obs_sink.1].into();
                ys == HashSet::from([0, 7])
            };
            assert!(span_x || span_y, "seed={seed}: not edge-to-edge");
            let coord = |c: &Cell| if span_x { c.0 } else { c.1 };
            assert_eq!(info.obstruction.iter().map(coord).min(), Some(0));
            assert_eq!(info.obstruction.iter().map(coord).max(), Some(7));
            // 4-connected.
            let start = *info.obstruction.iter().next().unwrap();
            let mut seen = HashSet::from([start]);
            let mut stack = vec![start];
            while let Some((x, y)) = stack.pop() {
                for (dx, dy) in BFS_DELTAS {
                    let n = (x + dx, y + dy);
                    if info.obstruction.contains(&n) && seen.insert(n) {
                        stack.push(n);
                    }
                }
            }
            assert_eq!(seen, info.obstruction, "seed={seed}: cut not 4-connected");
        }
    }

    #[test]
    fn test_cross_obstruction_endpoints_are_random_and_winding() {
        let mut endpoints: HashSet<Cell> = HashSet::new();
        let mut winding = 0;
        for seed in 0..60u64 {
            let info = layout(&build_cub(10, seed));
            endpoints.insert(info.obs_source);
            endpoints.insert(info.obs_sink);
            let xs: HashSet<i64> = info.obstruction.iter().map(|c| c.0).collect();
            let ys: HashSet<i64> = info.obstruction.iter().map(|c| c.1).collect();
            if xs.len() > 1 && ys.len() > 1 {
                winding += 1;
            }
        }
        assert!(endpoints.len() > 6, "obstruction endpoints look fixed");
        assert!(winding > 0, "no winding obstruction across 60 seeds");
    }

    #[test]
    fn test_cross_no_source_adjacent_to_sink() {
        for seed in 0..40u64 {
            let info = layout(&build_cub(8, seed));
            for s in [info.obs_source, info.cross_source] {
                for k in [info.obs_sink, info.cross_sink] {
                    assert_ne!((s.0 - k.0).abs() + (s.1 - k.1).abs(), 1, "seed={seed}");
                }
            }
        }
    }

    #[test]
    fn test_cross_underground_straddles_obstruction() {
        for seed in 0..40u64 {
            let info = layout(&build_cub(8, seed));
            let (ddx, ddy) = info.flow_dir.unwrap().delta();
            for &(dx, dy) in &info.ug_downs {
                let mut between = Vec::new();
                for step in 1..6 {
                    let cell = (dx + ddx * step, dy + ddy * step);
                    if info.ug_ups.contains(&cell) {
                        assert!(
                            between.iter().any(|c| info.obstruction.contains(c)),
                            "seed={seed}: tunnel skips the obstruction"
                        );
                        break;
                    }
                    between.push(cell);
                }
            }
        }
    }

    #[test]
    fn test_cross_endpoints_clear_of_obstruction() {
        for seed in 0..40u64 {
            let info = layout(&build_cub(8, seed));
            for c in [info.cross_source, info.cross_sink] {
                let gap = info
                    .obstruction
                    .iter()
                    .map(|o| (c.0 - o.0).abs() + (c.1 - o.1).abs())
                    .min()
                    .unwrap();
                assert!(
                    gap >= 2,
                    "seed={seed}: endpoint {c:?} hugs the cut (gap {gap})"
                );
            }
        }
    }

    #[test]
    fn test_cross_endpoints_can_be_interior() {
        let mut interior = 0;
        for seed in 0..60u64 {
            let info = layout(&build_cub(12, seed));
            let on_edge = |c: &Cell| c.0 == 0 || c.0 == 11 || c.1 == 0 || c.1 == 11;
            interior += [info.cross_source, info.cross_sink]
                .iter()
                .filter(|c| !on_edge(c))
                .count();
        }
        assert!(interior > 0, "crossing endpoints are always on the edge");
    }

    #[test]
    fn test_cross_all_flow_directions_appear() {
        let mut seen: HashSet<Direction> = HashSet::new();
        for seed in 0..200u64 {
            if let Some(d) = layout(&build_cub(8, seed)).flow_dir {
                seen.insert(d);
            }
            if seen.len() == 4 {
                break;
            }
        }
        assert_eq!(seen.len(), 4, "only saw {seen:?}");
    }

    #[test]
    fn test_cross_delivers_with_obstruction_removed() {
        // The crossing is self-sufficient: clear the protected obstruction and
        // it still delivers.
        for seed in 0..30u64 {
            let f = build_cub(8, seed);
            let mut world = f.world.clone();
            for &(x, y) in &f.protected_positions {
                // Empty == channel value 0 (no `Item::Empty` variant exists).
                world.set(x, y, Channel::Entities, 0);
                world.set(x, y, Channel::Direction, Direction::None as i64);
                world.set(x, y, Channel::Items, 0);
                world.set(x, y, Channel::Misc, Misc::None as i64);
            }
            assert!(tp_unreachable(&world).0 > 0.0, "seed={seed}");
        }
    }

    #[test]
    fn test_no_orphan_tiles_every_lesson() {
        // No lesson's solved factory may contain orphan tiles (unreachable == 0).
        for &kind in all_lesson_kinds() {
            let mut checked = 0;
            for seed in 0..20u64 {
                let Some(f) = build_factory(12, kind, seed, true, f64::INFINITY) else {
                    continue;
                };
                checked += 1;
                let (_, unreachable) = tp_unreachable(&f.world);
                assert_eq!(
                    unreachable,
                    0,
                    "{} seed={seed}: {unreachable} orphans",
                    kind.name()
                );
            }
            assert!(checked > 0, "no {} factories built", kind.name());
        }
    }

    #[test]
    fn test_splitter_lessons_leave_splitter_removable() {
        // The splitter must be removable (empty protected_positions) so the
        // policy can learn to place it; the built factory still contains it.
        for &kind in &[LessonKind::SplitterSplit, LessonKind::SplitterMerge] {
            let mut checked = 0;
            for seed in 0..20u64 {
                let Some(f) = build_factory(12, kind, seed, true, f64::INFINITY) else {
                    continue;
                };
                checked += 1;
                assert!(
                    f.protected_positions.is_empty(),
                    "{} seed={seed}: splitter must be removable, got protected {:?}",
                    kind.name(),
                    f.protected_positions
                );
                let mut splitter_tiles = 0;
                for x in 0..f.world.width() {
                    for y in 0..f.world.height() {
                        if f.world.entity_at(x, y) == Some(Item::Splitter) {
                            splitter_tiles += 1;
                        }
                    }
                }
                assert_eq!(
                    splitter_tiles,
                    2,
                    "{} seed={seed}: expected splitter",
                    kind.name()
                );
            }
            assert!(checked > 0, "no {} factories built", kind.name());
        }
    }

    #[test]
    fn test_splitter_y_roles() {
        // The Y wires one base to two prongs through the splitter:
        // SPLITTER_SPLIT is 1 source → 2 sinks, SPLITTER_MERGE is 2 sources →
        // 1 sink. Both build with positive throughput across seeds.
        for &(kind, want_sources, want_sinks) in &[
            (LessonKind::SplitterSplit, 1, 2),
            (LessonKind::SplitterMerge, 2, 1),
        ] {
            let mut checked = 0;
            for seed in 0..20u64 {
                let Some(f) = build_factory(12, kind, seed, true, f64::INFINITY) else {
                    continue;
                };
                checked += 1;
                assert!(
                    world_throughput(&f.world) > 0.0,
                    "{} seed={seed}",
                    kind.name()
                );
                let (mut sources, mut sinks, mut splitters) = (0, 0, 0);
                for x in 0..f.world.width() {
                    for y in 0..f.world.height() {
                        match f.world.entity_at(x, y) {
                            Some(Item::Source) => sources += 1,
                            Some(Item::Sink) => sinks += 1,
                            Some(Item::Splitter) => splitters += 1,
                            _ => {}
                        }
                    }
                }
                assert_eq!(
                    sources,
                    want_sources,
                    "{} seed={seed}: sources",
                    kind.name()
                );
                assert_eq!(sinks, want_sinks, "{} seed={seed}: sinks", kind.name());
                assert_eq!(splitters, 2, "{} seed={seed}: splitter tiles", kind.name());
            }
            assert!(checked > 0, "no {} factories built", kind.name());
        }
    }

    #[test]
    fn test_splitter_marker_facings() {
        // Sources feed the splitter → they face toward it; sinks are fed by it
        // → they face away. That's the sign of facing · (splitterCentroid −
        // markerPos), which rotation preserves. (Centroid doubled to stay in
        // integers: the two splitter tiles sum to 2·centroid.)
        for &kind in &[LessonKind::SplitterSplit, LessonKind::SplitterMerge] {
            for seed in 0..30u64 {
                let Some(f) = build_factory(12, kind, seed, true, f64::INFINITY) else {
                    continue;
                };
                let (mut sx2, mut sy2, mut n) = (0i64, 0i64, 0i64);
                for x in 0..f.world.width() {
                    for y in 0..f.world.height() {
                        if f.world.entity_at(x, y) == Some(Item::Splitter) {
                            sx2 += x as i64;
                            sy2 += y as i64;
                            n += 1;
                        }
                    }
                }
                assert_eq!(n, 2, "{} seed={seed}", kind.name());
                for x in 0..f.world.width() {
                    for y in 0..f.world.height() {
                        let ent = f.world.entity_at(x, y);
                        let (is_src, is_snk) = (ent == Some(Item::Source), ent == Some(Item::Sink));
                        if !is_src && !is_snk {
                            continue;
                        }
                        let d = f.world.direction_at(x, y).delta();
                        let (vx, vy) = (sx2 - 2 * x as i64, sy2 - 2 * y as i64);
                        let dot = d.0 * vx + d.1 * vy;
                        if is_src {
                            assert!(
                                dot > 0,
                                "{} seed={seed}: source at ({x},{y}) not facing toward splitter",
                                kind.name()
                            );
                        } else {
                            assert!(
                                dot < 0,
                                "{} seed={seed}: sink at ({x},{y}) not facing away from splitter",
                                kind.name()
                            );
                        }
                    }
                }
            }
        }
    }
}
