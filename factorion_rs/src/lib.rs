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
use graph::build_graph;
#[cfg(feature = "pyo3-bindings")]
use throughput::calc_throughput;
#[cfg(feature = "pyo3-bindings")]
use world::World;

/// Calculate the throughput of a factory represented as a 3D tensor.
///
/// Input: numpy array of shape (W, H, 4) with dtype i64, where channels are:
///   0: entity ID, 1: direction, 2: item/recipe, 3: misc (underground state)
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

#[cfg(feature = "pyo3-bindings")]
#[pymodule]
fn factorion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_throughput, m)?)?;
    Ok(())
}
