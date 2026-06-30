"""Benchmark: PyO3 ``factorion_rs.build_factory`` vs pure-Python
``factorion.build_factory``, per lesson.

The Rust port exists to speed up RL rollouts by building factories faster.
This script measures the per-call wall time of the PyO3 binding and the
original Python implementation over ~1000 seeds per lesson, reporting
mean/min/max/std (microseconds) and the speedup.

The companion native-Rust baseline (no Python boundary) lives in
``factorion_rs/src/factory_gen.rs::bench_native_build_factory``; run it with::

    cargo test --release -p factorion_rs bench_native_build_factory \\
        -- --ignored --nocapture

Run this one with::

    WANDB_MODE=disabled uv run python tests/bench_build_factory.py

The lesson list and grid sizes are kept in sync with ``BENCH_LESSONS`` in the
Rust benchmark so all three measurements (Python, PyO3, native) use identical
workloads.
"""

import os
import statistics
import sys
import time

# Allow running directly (`python tests/bench_build_factory.py`): put the repo
# root on the path so `factorion` resolves.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import factorion_rs  # noqa: E402
from factorion import LessonKind, build_factory  # noqa: E402

# (kind, grid size) — mirror of Rust's BENCH_LESSONS.
BENCH_LESSONS = [
    (LessonKind.MOVE_ONE_ITEM, 12),
    (LessonKind.MOVE_ONE_ITEM_CHAOS, 12),
    (LessonKind.SPLITTER_SPLIT, 12),
    (LessonKind.SPLITTER_MERGE, 12),
    (LessonKind.ASSEMBLE_1IN_1OUT, 12),
    (LessonKind.MOVE_VIA_UG_BELT, 12),
    (LessonKind.ASSEMBLE_2IN_1OUT, 12),
]

N = 1000


def _stats_us(samples_s):
    us = [s * 1e6 for s in samples_s]
    return (
        statistics.mean(us),
        min(us),
        max(us),
        statistics.pstdev(us),
    )


def _time_pyo3(kind, size):
    samples = []
    for seed in range(N):
        t0 = time.perf_counter()
        factorion_rs.build_factory(size, kind.value, seed, True, float("inf"))
        samples.append(time.perf_counter() - t0)
    return samples


def _time_python(kind, size):
    samples = []
    for seed in range(N):
        t0 = time.perf_counter()
        build_factory(size=size, kind=kind, seed=seed)
        samples.append(time.perf_counter() - t0)
    return samples


def main():
    header = (
        f"{'lesson':<22} {'impl':<7} "
        f"{'mean':>9} {'min':>9} {'max':>9} {'std':>9}"
    )
    print(f"\nbuild_factory benchmark ({N} seeds/lesson, microseconds per call)")
    print(header)
    print("-" * len(header))
    for kind, size in BENCH_LESSONS:
        py = _stats_us(_time_python(kind, size))
        rs = _stats_us(_time_pyo3(kind, size))
        speedup = py[0] / rs[0] if rs[0] else float("nan")
        print(
            f"{kind.name:<22} {'python':<7} "
            f"{py[0]:>9.1f} {py[1]:>9.1f} {py[2]:>9.1f} {py[3]:>9.1f}"
        )
        print(
            f"{'':<22} {'pyo3':<7} "
            f"{rs[0]:>9.1f} {rs[1]:>9.1f} {rs[2]:>9.1f} {rs[3]:>9.1f}"
            f"   ({speedup:.1f}x faster)"
        )


if __name__ == "__main__":
    main()
