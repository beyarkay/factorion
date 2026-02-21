mod entities;
mod graph;
mod throughput;
mod types;
mod world;

use numpy::PyReadonlyArray3;
use pyo3::prelude::*;

use graph::build_graph;
use throughput::calc_throughput;
use world::World;

/// Calculate the throughput of a factory represented as a 3D tensor.
///
/// Input: numpy array of shape (W, H, 4) with dtype i64, where channels are:
///   0: entity ID, 1: direction, 2: item/recipe, 3: misc (underground state)
///
/// Returns: (throughput, num_unreachable) matching funge_throughput's signature.
#[pyfunction]
fn simulate_throughput(world: PyReadonlyArray3<i64>) -> PyResult<(f64, usize)> {
    let world = World::from_numpy(&world);
    let graph = build_graph(&world);
    let (throughput_map, num_unreachable) = calc_throughput(&graph);

    if throughput_map.is_empty() {
        return Ok((0.0, num_unreachable));
    }

    // Sum all items' throughput for deterministic results regardless of HashMap order.
    // For single-item factories this is equivalent to taking the only value.
    let throughput: f64 = throughput_map.values().copied().sum();
    if throughput.is_infinite() {
        return Ok((0.0, num_unreachable));
    }

    Ok((throughput, num_unreachable))
}

#[pymodule]
fn factorion_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(simulate_throughput, m)?)?;
    Ok(())
}
