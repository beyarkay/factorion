// Forbid type-system escape hatches that crash at runtime
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::unreachable)]
// Forbid unwrap/expect in non-test code (tests use #[allow] where needed)
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
// Without the `pyo3-bindings` feature this crate has no non-test entry points:
// every root (`simulate_throughput`, `build_graph`, `entity_tiles`, the `py_*`
// functions, …) is gated behind that feature, so a non-test `--no-default-features`
// build leaves dead-code analysis with nothing to anchor on and flags the entire
// throughput engine. That configuration is only used to run `cargo test
// --no-default-features` (the default extension-module build can't link libpython
// under `cargo test`), so suppress dead-code there. The shipping default build
// still gets full dead-code checks.
#![cfg_attr(not(feature = "pyo3-bindings"), allow(dead_code))]

// `nonempty::nonempty![...]` expands to a path through `::alloc::vec`, which
// requires `alloc` to be visible at the crate root.
extern crate alloc;

mod entities;
mod factory_gen;
mod graph;
mod render;
mod rng;
#[cfg(test)]
mod textual;
mod throughput;
mod types;
mod world;

#[cfg(feature = "pyo3-bindings")]
use numpy::{IntoPyArray, PyReadonlyArray3};
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3-bindings")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3-bindings")]
use entities::entity_tiles;
#[cfg(feature = "pyo3-bindings")]
use factory_gen::{all_lesson_kinds, build_factory as rs_build_factory, LessonKind};
#[cfg(feature = "pyo3-bindings")]
use graph::build_graph;
#[cfg(feature = "pyo3-bindings")]
use render::render as render_world;
#[cfg(feature = "pyo3-bindings")]
use std::collections::HashMap;
#[cfg(feature = "pyo3-bindings")]
use throughput::{calc_throughput, factory_score};
#[cfg(feature = "pyo3-bindings")]
use types::{all_items, all_recipes, Direction, Item, Recipe};
#[cfg(feature = "pyo3-bindings")]
use world::World;

/// Calculate the throughput score of a factory represented as a 3D tensor.
///
/// Input: numpy array of shape (W, H, C) with dtype i64, where channels are:
///   0: entity ID, 1: direction, 2: item/recipe, 3: misc (underground state), 4: footprint
///
/// Returns: (score, num_unreachable).
/// The score is the power mean (see [`factory_score`]) of each sink's
/// achieved throughput of its configured item, so unused / under-served
/// sinks drag it down rather than being hidden by a fully-fed sink.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn simulate_throughput(world: PyReadonlyArray3<i64>) -> PyResult<(f64, usize)> {
    let world = World::from_numpy(&world);
    let graph = build_graph(&world);
    let (deliveries, num_unreachable) = calc_throughput(&graph);
    Ok((factory_score(&deliveries), num_unreachable))
}

/// Python-facing shape of one sink's delivery: the sink's anchor tile, the
/// item it is configured to accept (`None` when unset), and the achieved
/// items/s of that item reaching it.
#[cfg(feature = "pyo3-bindings")]
type PySinkDelivery = (usize, usize, Option<String>, f64);

/// Per-sink achieved throughput of a factory represented as a 3D tensor.
///
/// Input: numpy array of shape (W, H, C), dtype i64 — the same world tensor
/// `simulate_throughput` takes.
///
/// Returns one `(x, y, item_name, achieved_items_per_sec)` tuple per sink,
/// keyed by the sink's anchor tile. This is the un-aggregated form of
/// `simulate_throughput`'s score (which collapses these through
/// [`factory_score`]) — the Factorio parity harness compares each sink's
/// rate against the real game individually so a mismatch points at a sink,
/// not just at the factory.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn py_sink_deliveries(world: PyReadonlyArray3<i64>) -> PyResult<Vec<PySinkDelivery>> {
    let world = World::from_numpy(&world);
    let graph = build_graph(&world);
    let (deliveries, _) = calc_throughput(&graph);
    Ok(deliveries
        .into_iter()
        .map(|d| {
            (
                d.anchor.0,
                d.anchor.1,
                d.item.map(|i| i.name().to_string()),
                d.achieved,
            )
        })
        .collect())
}

/// Python-facing shape of a built factory graph: the node labels and the
/// `(src_index, dst_index)` edges indexing into them.
#[cfg(feature = "pyo3-bindings")]
type PyGraphData = (Vec<String>, Vec<(usize, usize)>);

/// Build the factory connection graph and return it as plain Python data.
///
/// Input: numpy array of shape (W, H, C), dtype i64 — the same world tensor
/// `simulate_throughput` takes.
///
/// Returns `(nodes, edges)` where:
///   * `nodes` is a list of node labels in the engine's canonical
///     `"{entity_char}@{x},{y}"` format (grid-registry chars: `b`, `i`, `Y`,
///     `d`/`u`, `S`, `K`, …), plus a `:L`/`:R` suffix for the two lane nodes
///     of a belt-ish tile (lane-less entities collapse to their anchor tile;
///     splitter tiles keep per-tile identity), and
///   * `edges` is a list of `(src_index, dst_index)` pairs indexing into
///     `nodes`.
///
/// This is enough to rebuild a `networkx.DiGraph` for connectivity checks and
/// visualization layout, and is the single Python entry point for factory
/// graph construction.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn py_build_graph(world: PyReadonlyArray3<i64>) -> PyResult<PyGraphData> {
    let world = World::from_numpy(&world);
    let graph = build_graph(&world);
    let nodes: Vec<String> = graph.nodes.iter().map(|node| node.label()).collect();
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for (src, succ) in graph.successors.iter().enumerate() {
        for &dst in succ {
            edges.push((src, dst));
        }
    }
    Ok((nodes, edges))
}

/// Compute all tiles occupied by a multi-tile entity.
///
/// Args:
///   x, y: anchor position
///   direction: facing direction (1=N, 2=E, 3=S, 4=W)
///   width: extent perpendicular to flow
///   height: extent along flow
///
/// Returns list of (x, y) tuples, or None if direction is 0 (None) on a non-square multi-tile entity.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn py_entity_tiles(
    x: usize,
    y: usize,
    direction: i64,
    width: usize,
    height: usize,
) -> PyResult<Option<Vec<(i64, i64)>>> {
    let dir = Direction::from_i64(direction);
    Ok(entity_tiles(x, y, dir, width, height)
        .map(|tiles| tiles.into_iter().map(|p| (p.x, p.y)).collect()))
}

/// Return every Item as a dict keyed by integer value, with full
/// per-item properties.
///
/// Shape: `{int_value: {"name": str, "is_placeable": bool,
///                      "width": int, "height": int, "flow": float}}`.
///
/// Single source of truth for item identity and entity properties —
/// Python's `items` dict (and the older `entities` dict) are built from
/// this at module load.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn py_items(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let outer = PyDict::new(py);
    for item in all_items() {
        let entry = PyDict::new(py);
        let (w, h) = item.size();
        entry.set_item("name", item.name())?;
        entry.set_item("is_placeable", item.is_placeable())?;
        entry.set_item("width", w)?;
        entry.set_item("height", h)?;
        entry.set_item("flow", item.flow_rate())?;
        outer.set_item(item as i64, entry)?;
    }
    Ok(outer.into())
}

/// Return all crafting recipes as a Python dict.
///
/// Shape: `{item_name: {"consumes": {item_name: rate, ...},
///                      "produces": {item_name: rate, ...},
///                      "crafting_time": seconds_per_craft,
///                      "produced_by": [machine_name, ...],
///                      "total_raw": {item_name: amount, ...},
///                      "total_raw_time": cumulative_seconds}}`.
///
/// `total_raw` is derived (each ingredient reduced through the recipe set to
/// items with no recipe); `total_raw_time` is the summed craft time of the
/// whole tree — see [`types::TotalRaw`].
///
/// This is the single source of truth for recipe data — Python builds its
/// `recipes` dict from this at module load.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn py_recipes(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let outer = PyDict::new(py);
    let recipes = all_recipes();
    // Built once and shared across every recipe's total_raw expansion.
    let index: HashMap<Item, Recipe> = recipes.iter().cloned().collect();
    for (item, recipe) in &recipes {
        let entry = PyDict::new(py);
        let consumes = PyDict::new(py);
        for &(i, rate) in recipe.consumes.iter() {
            consumes.set_item(i.name(), rate)?;
        }
        let produces = PyDict::new(py);
        for &(i, rate) in recipe.produces.iter() {
            produces.set_item(i.name(), rate)?;
        }
        let total = recipe.total_raw(&index);
        let total_raw = PyDict::new(py);
        for &(i, amount) in total.items.iter() {
            total_raw.set_item(i.name(), amount)?;
        }
        let produced_by: Vec<&str> = recipe.produced_by.iter().map(|m| m.name()).collect();
        entry.set_item("consumes", consumes)?;
        entry.set_item("produces", produces)?;
        entry.set_item("crafting_time", recipe.crafting_time)?;
        entry.set_item("produced_by", produced_by)?;
        entry.set_item("total_raw", total_raw)?;
        entry.set_item("total_raw_time", total.time)?;
        outer.set_item(item.name(), entry)?;
    }
    Ok(outer.into())
}

/// Python-facing shape of a built factory: the world tensor, the count of
/// removable entity units, and the protected (never-blanked) positions.
#[cfg(feature = "pyo3-bindings")]
type PyFactory = (Py<numpy::PyArray3<i64>>, usize, Vec<(usize, usize)>);

/// Build a complete, valid factory of the given lesson `kind` (an integer
/// `LessonKind` value) on a `size × size` grid, seeded from `seed`.
///
/// Returns `(world, total_entities, protected_positions)` where `world` is
/// a `(W, H, C)` int64 array (the same shape `simulate_throughput` takes),
/// or `None` when rejection sampling is exhausted. Raises `ValueError` for an
/// unknown lesson kind.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
#[pyo3(signature = (size, kind, seed, random_item=true, max_entities=f64::INFINITY))]
fn build_factory(
    py: Python<'_>,
    size: usize,
    kind: i64,
    seed: i64,
    random_item: bool,
    max_entities: f64,
) -> PyResult<Option<PyFactory>> {
    let lesson = LessonKind::from_i64(kind)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown kind {kind}")))?;
    let built = match rs_build_factory(size, lesson, seed.unsigned_abs(), random_item, max_entities)
    {
        Some(b) => b,
        None => return Ok(None),
    };
    let total = built.total_entities;
    let protected = built.protected_positions.clone();
    let (data, w, h, c) = built.world.into_whc();
    let arr = numpy::ndarray::Array3::from_shape_vec((w, h, c), data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(Some((arr.into_pyarray(py).unbind(), total, protected)))
}

/// Render a factory world tensor into the two-character ASCII grid format.
///
/// Input: numpy array of shape (W, H, C), dtype i64 — the same world tensor
/// `simulate_throughput` takes. Returns the multi-line grid string (geometry
/// only; item/recipe bindings live in the tensor, not the grid), where each
/// tile is two characters — an entity char (`b`=belt, `i`=inserter, `a`=
/// assembler, `Y`=splitter, `d`/`u`=underground down/up, `S`=source, `K`=sink)
/// plus a direction marker (`^>v<`), or `..` for an empty tile. Multi-tile
/// entities draw their body across the footprint with a blank interior.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn render_factory(world: PyReadonlyArray3<i64>) -> PyResult<String> {
    let world = World::from_numpy(&world);
    Ok(render_world(&world))
}

/// Return the lesson kinds as an ordered `{NAME: int_value}` dict — the single
/// source of truth Python builds its `LessonKind` enum from. Insertion order
/// matches `all_lesson_kinds()` so `list(LessonKind)` iterates the same way.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn py_lesson_kinds(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    for &kind in all_lesson_kinds() {
        d.set_item(kind.name(), kind as i64)?;
    }
    Ok(d.into())
}

#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn factorion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_throughput, m)?)?;
    m.add_function(wrap_pyfunction!(py_sink_deliveries, m)?)?;
    m.add_function(wrap_pyfunction!(py_build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(py_entity_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(py_items, m)?)?;
    m.add_function(wrap_pyfunction!(py_recipes, m)?)?;
    m.add_function(wrap_pyfunction!(py_lesson_kinds, m)?)?;
    m.add_function(wrap_pyfunction!(build_factory, m)?)?;
    m.add_function(wrap_pyfunction!(render_factory, m)?)?;
    Ok(())
}
