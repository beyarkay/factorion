// Forbid type-system escape hatches that crash at runtime
#![deny(clippy::panic)]
#![deny(clippy::todo)]
#![deny(clippy::unimplemented)]
#![deny(clippy::unreachable)]
// Forbid unwrap/expect in non-test code (tests use #[allow] where needed)
#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]

mod entities;
mod graph;
mod throughput;
mod types;
mod world;

#[cfg(feature = "pyo3-bindings")]
use numpy::PyReadonlyArray3;
#[cfg(feature = "pyo3-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "pyo3-bindings")]
use entities::entity_tiles;
#[cfg(feature = "pyo3-bindings")]
use graph::build_graph;
#[cfg(feature = "pyo3-bindings")]
use throughput::calc_throughput;
#[cfg(feature = "pyo3-bindings")]
use types::Direction;
#[cfg(feature = "pyo3-bindings")]
use world::World;

/// Calculate the throughput of a factory represented as a 3D tensor.
///
/// Input: numpy array of shape (W, H, C) with dtype i64, where channels are:
///   0: entity ID, 1: direction, 2: item/recipe, 3: misc (underground state), 4: footprint
///
/// Returns: (throughput, num_unreachable) matching funge_throughput's signature.
#[cfg(feature = "pyo3-bindings")]
#[pyfunction]
fn simulate_throughput(world: PyReadonlyArray3<i64>) -> PyResult<(f64, usize)> {
    let world = World::from_numpy(&world);
    let graph = build_graph(&world);
    let (throughput_map, num_unreachable) = calc_throughput(&graph);

    if throughput_map.is_empty() {
        return Ok((0.0, num_unreachable));
    }

    // Return the maximum item throughput for deterministic results regardless of
    // HashMap iteration order. Python returns an arbitrary single item's throughput
    // via list(values())[0]; using max is deterministic and equivalent for the
    // typical single-item case.
    let throughput: f64 = throughput_map
        .values()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if throughput.is_infinite() {
        return Ok((0.0, num_unreachable));
    }

    Ok((throughput, num_unreachable))
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

#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn factorion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_throughput, m)?)?;
    m.add_function(wrap_pyfunction!(py_entity_tiles, m)?)?;
    Ok(())
}
