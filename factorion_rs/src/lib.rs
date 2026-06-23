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
mod graph;
#[cfg(test)]
mod textual;
mod throughput;
mod types;
mod world;

#[cfg(feature = "pyo3-bindings")]
use numpy::PyReadonlyArray3;
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;
#[cfg(feature = "pyo3-bindings")]
use pyo3::types::PyDict;

#[cfg(feature = "pyo3-bindings")]
use entities::entity_tiles;
#[cfg(feature = "pyo3-bindings")]
use graph::build_graph;
#[cfg(feature = "pyo3-bindings")]
use throughput::{calc_throughput, factory_score};
#[cfg(feature = "pyo3-bindings")]
use types::{all_items, all_recipes, Direction};
#[cfg(feature = "pyo3-bindings")]
use world::World;

/// Calculate the throughput score of a factory represented as a 3D tensor.
///
/// Input: numpy array of shape (W, H, C) with dtype i64, where channels are:
///   0: entity ID, 1: direction, 2: item/recipe, 3: misc (underground state), 4: footprint
///
/// Returns: (score, num_unreachable) matching funge_throughput's signature.
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
///   * `nodes` is a list of node labels in the engine's
///     `"{entity_name}\n@{x},{y}"` format (one per placeable entity, anchor
///     tile only for multi-tile entities), and
///   * `edges` is a list of `(src_index, dst_index)` pairs indexing into
///     `nodes`.
///
/// This is enough to rebuild a `networkx.DiGraph` for connectivity checks and
/// visualization layout, and is the single Python entry point for factory
/// graph construction (replacing the former Python `world2graph`).
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn py_build_graph(world: PyReadonlyArray3<i64>) -> PyResult<PyGraphData> {
    let world = World::from_numpy(&world);
    let graph = build_graph(&world);
    let nodes: Vec<String> = graph.nodes.iter().map(|node| node.id.label()).collect();
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
    for &item in all_items() {
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
///                      "crafting_time": seconds_per_craft}}`.
///
/// This is the single source of truth for recipe data — Python builds its
/// `recipes` dict from this at module load.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn py_recipes(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let outer = PyDict::new(py);
    for (item, recipe) in all_recipes() {
        let entry = PyDict::new(py);
        let consumes = PyDict::new(py);
        for &(i, rate) in recipe.consumes.iter() {
            consumes.set_item(i.name(), rate)?;
        }
        let produces = PyDict::new(py);
        for &(i, rate) in recipe.produces.iter() {
            produces.set_item(i.name(), rate)?;
        }
        entry.set_item("consumes", consumes)?;
        entry.set_item("produces", produces)?;
        entry.set_item("crafting_time", recipe.crafting_time)?;
        outer.set_item(item.name(), entry)?;
    }
    Ok(outer.into())
}

#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn factorion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_throughput, m)?)?;
    m.add_function(wrap_pyfunction!(py_build_graph, m)?)?;
    m.add_function(wrap_pyfunction!(py_entity_tiles, m)?)?;
    m.add_function(wrap_pyfunction!(py_items, m)?)?;
    m.add_function(wrap_pyfunction!(py_recipes, m)?)?;
    Ok(())
}
