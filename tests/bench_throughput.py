"""Benchmarks comparing Python and Rust throughput calculation.

Run with:
    python tests/bench_throughput.py

All calls go through Python (the Rust version is called via its Python bindings),
mirroring real-world usage in FactorioEnv.step().

Covers:
  - Tiny/small/medium/large random worlds
  - Generated lesson worlds
  - Handcrafted worlds (zigzag, assembler, underground, mixed)
  - Dense vs sparse random worlds
  - Belt-only vs mixed-entity random worlds
"""

import random
import sys
import time

import torch

# Allow running directly: python tests/bench_throughput.py
sys.path.insert(0, "tests")
from helpers import (
    Direction,
    LessonKind,
    Misc,
    generate_lesson,
    make_world,
    py_throughput_safe,
    rs_throughput,
    set_assembler,
    set_entity,
)


# ── World generators ─────────────────────────────────────────────────────────

ENTITY_VALUES = [0, 1, 2, 4, 5, 6]  # skip assembler (3x3)
DIRECTION_VALUES = [0, 1, 2, 3, 4]
ITEM_VALUES = [0, 1, 2, 3, 4]
MISC_VALUES = [0, 1, 2]


def random_world(size, rng, density=0.3):
    """Random world with mixed entities."""
    world = torch.zeros((size, size, 4), dtype=torch.int64)
    for x in range(size):
        for y in range(size):
            if rng.random() < density:
                world[x, y, 0] = rng.choice(ENTITY_VALUES)
                world[x, y, 1] = rng.choice(DIRECTION_VALUES)
                world[x, y, 2] = rng.choice(ITEM_VALUES)
                world[x, y, 3] = rng.choice(MISC_VALUES)
    return world


def random_belt_world(size, rng, density=0.4):
    """Random world with only belts, sources, and sinks."""
    world = torch.zeros((size, size, 4), dtype=torch.int64)
    # Place a source
    sx, sy = rng.randint(0, size - 1), rng.randint(0, size - 1)
    world[sx, sy, 0] = 6  # stack_inserter/source
    world[sx, sy, 1] = rng.choice([1, 2, 3, 4])
    world[sx, sy, 2] = rng.choice([1, 2, 3])
    # Place a sink
    for _ in range(20):
        bx, by = rng.randint(0, size - 1), rng.randint(0, size - 1)
        if (bx, by) != (sx, sy):
            break
    world[bx, by, 0] = 5  # bulk_inserter/sink
    world[bx, by, 1] = rng.choice([1, 2, 3, 4])
    world[bx, by, 2] = world[sx, sy, 2]  # same item
    # Fill belts
    for x in range(size):
        for y in range(size):
            if (x, y) in ((sx, sy), (bx, by)):
                continue
            if rng.random() < density:
                world[x, y, 0] = 1  # transport_belt
                world[x, y, 1] = rng.choice([1, 2, 3, 4])
    return world


def lesson_world(size, seed):
    """World from generate_lesson."""
    result = generate_lesson(
        size=size,
        kind=LessonKind.MOVE_ONE_ITEM,
        num_missing_entities=0,
        seed=seed,
    )
    world_CWH = result[0]
    return world_CWH.permute(1, 2, 0).to(torch.int64)


def handcrafted_straight_belt(length):
    """Simple source -> N belts -> sink."""
    world = make_world(length + 2)
    set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
    for x in range(1, length + 1):
        set_entity(world, x, 0, "transport_belt", Direction.EAST)
    set_entity(world, length + 1, 0, "bulk_inserter", Direction.EAST, "iron_plate")
    return world


def handcrafted_zigzag(rows):
    """Zigzag serpentine belt pattern with `rows` rows, width=8."""
    width = 8
    world = make_world(width, rows + 2)
    set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_cable")
    x, y = 1, 0
    direction = Direction.EAST
    while y < rows:
        if direction == Direction.EAST:
            while x < width - 1:
                set_entity(world, x, y, "transport_belt", Direction.EAST)
                x += 1
            set_entity(world, x, y, "transport_belt", Direction.SOUTH)
            y += 1
            if y > rows:
                break
            direction = Direction.WEST
        else:
            while x > 0:
                set_entity(world, x, y, "transport_belt", Direction.WEST)
                x -= 1
            set_entity(world, x, y, "transport_belt", Direction.SOUTH)
            y += 1
            if y > rows:
                break
            direction = Direction.EAST
    set_entity(world, x, y, "bulk_inserter", Direction.SOUTH, "copper_cable")
    return world


def handcrafted_underground_chain(num_pairs):
    """Chain of underground belt pairs."""
    length = num_pairs * 6 + 2
    world = make_world(length)
    set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "copper_plate")
    pos = 1
    for _ in range(num_pairs):
        set_entity(
            world, pos, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_DOWN.value,
        )
        pos += 4
        set_entity(
            world, pos, 0, "underground_belt", Direction.EAST,
            misc=Misc.UNDERGROUND_UP.value,
        )
        pos += 1
        if pos < length - 1:
            set_entity(world, pos, 0, "transport_belt", Direction.EAST)
            pos += 1
    set_entity(
        world, min(pos, length - 1), 0, "bulk_inserter", Direction.EAST, "copper_plate"
    )
    return world


def handcrafted_assembler_line():
    """Source -> inserter -> assembler(copper_cable) -> inserter -> belt chain -> sink."""
    world = make_world(12, 5)
    set_entity(world, 0, 2, "stack_inserter", Direction.EAST, "copper_plate")
    set_entity(world, 1, 2, "transport_belt", Direction.EAST)
    set_entity(world, 2, 2, "transport_belt", Direction.EAST)
    set_entity(world, 3, 2, "inserter", Direction.EAST)
    set_assembler(world, 4, 1, "copper_cable")
    set_entity(world, 7, 2, "inserter", Direction.EAST)
    set_entity(world, 8, 2, "transport_belt", Direction.EAST)
    set_entity(world, 9, 2, "transport_belt", Direction.EAST)
    set_entity(world, 10, 2, "transport_belt", Direction.EAST)
    set_entity(world, 11, 2, "bulk_inserter", Direction.EAST, "copper_cable")
    return world


def handcrafted_electronic_circuit():
    """Two-input electronic circuit factory."""
    world = make_world(11, 9)
    set_entity(world, 0, 4, "stack_inserter", Direction.EAST, "copper_cable")
    set_entity(world, 1, 4, "transport_belt", Direction.EAST)
    set_entity(world, 2, 4, "inserter", Direction.EAST)
    set_entity(world, 4, 0, "stack_inserter", Direction.SOUTH, "iron_plate")
    set_entity(world, 4, 1, "transport_belt", Direction.SOUTH)
    set_entity(world, 4, 2, "inserter", Direction.SOUTH)
    set_assembler(world, 3, 3, "electronic_circuit")
    set_entity(world, 6, 4, "inserter", Direction.EAST)
    set_entity(world, 7, 4, "transport_belt", Direction.EAST)
    set_entity(world, 8, 4, "transport_belt", Direction.EAST)
    set_entity(world, 9, 4, "transport_belt", Direction.EAST)
    set_entity(
        world, 10, 4, "bulk_inserter", Direction.EAST, "electronic_circuit"
    )
    return world


def handcrafted_parallel_paths(n_paths):
    """N independent parallel source->belt chain->sink paths."""
    world = make_world(8, n_paths * 2)
    for i in range(n_paths):
        row = i * 2
        set_entity(world, 0, row, "stack_inserter", Direction.EAST, "iron_plate")
        for x in range(1, 7):
            set_entity(world, x, row, "transport_belt", Direction.EAST)
        set_entity(world, 7, row, "bulk_inserter", Direction.EAST, "iron_plate")
    return world


def handcrafted_inserter_chain(n_inserters):
    """Source -> (inserter -> belt) * n -> sink. Each inserter bottlenecks."""
    length = 2 + n_inserters * 2
    world = make_world(length)
    set_entity(world, 0, 0, "stack_inserter", Direction.EAST, "iron_plate")
    pos = 1
    for _ in range(n_inserters):
        set_entity(world, pos, 0, "inserter", Direction.EAST)
        pos += 1
        set_entity(world, pos, 0, "transport_belt", Direction.EAST)
        pos += 1
    set_entity(world, pos, 0, "bulk_inserter", Direction.EAST, "iron_plate")
    return world


# ── Benchmark runner ─────────────────────────────────────────────────────────


def bench(name, worlds, n_repeats=3):
    """Benchmark Python vs Rust on a list of worlds."""
    # Warmup
    for w in worlds[:2]:
        try:
            py_throughput_safe(w)
        except Exception:
            pass
        rs_throughput(w)

    # Time Python
    py_times = []
    py_errors = 0
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for w in worlds:
            try:
                py_throughput_safe(w)
            except Exception:
                py_errors += 1
        py_times.append(time.perf_counter() - t0)

    # Time Rust
    rs_times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        for w in worlds:
            rs_throughput(w)
        rs_times.append(time.perf_counter() - t0)

    py_avg = sum(py_times) / len(py_times)
    rs_avg = sum(rs_times) / len(rs_times)
    speedup = py_avg / rs_avg if rs_avg > 0 else float("inf")

    return {
        "name": name,
        "n_worlds": len(worlds),
        "n_repeats": n_repeats,
        "py_total_s": py_avg,
        "rs_total_s": rs_avg,
        "py_per_world_ms": (py_avg / len(worlds)) * 1000,
        "rs_per_world_ms": (rs_avg / len(worlds)) * 1000,
        "speedup": speedup,
        "py_errors": py_errors // n_repeats,
    }


def print_results(results):
    """Pretty-print benchmark results."""
    print(f"\n{'=' * 90}")
    print(f"{'Benchmark':<40} {'Worlds':>6} {'Python':>10} {'Rust':>10} {'Speedup':>10}")
    print(f"{'':40} {'':>6} {'(ms/world)':>10} {'(ms/world)':>10} {'':>10}")
    print(f"{'-' * 90}")

    for r in results:
        print(
            f"{r['name']:<40} {r['n_worlds']:>6} "
            f"{r['py_per_world_ms']:>9.3f}{'*' if r['py_errors'] else ' '} "
            f"{r['rs_per_world_ms']:>9.3f}  "
            f"{r['speedup']:>9.1f}x"
        )

    print(f"{'=' * 90}")
    print("* = some Python calls raised exceptions (counted as 0ms)")
    print()

    py_total = sum(r["py_total_s"] for r in results)
    rs_total = sum(r["rs_total_s"] for r in results)
    total_worlds = sum(r["n_worlds"] for r in results)
    print(f"Total worlds benchmarked: {total_worlds}")
    print(f"Total Python time: {py_total:.3f}s")
    print(f"Total Rust time:   {rs_total:.3f}s")
    print(f"Overall speedup:   {py_total / rs_total:.1f}x")


def main():
    results = []
    rng = random.Random(42)

    print("Generating random worlds...")
    for size, label in [(3, "tiny 3x3"), (5, "small 5x5"), (8, "medium 8x8"), (15, "large 15x15"), (25, "xlarge 25x25")]:
        worlds = [random_world(size, random.Random(rng.randint(0, 99999))) for _ in range(50)]
        results.append(bench(f"Random mixed ({label})", worlds))

    print("Generating belt-only worlds...")
    for size, label in [(5, "5x5"), (10, "10x10"), (20, "20x20")]:
        worlds = [random_belt_world(size, random.Random(rng.randint(0, 99999))) for _ in range(50)]
        results.append(bench(f"Random belts ({label})", worlds))

    print("Generating dense/sparse worlds...")
    worlds_sparse = [random_world(10, random.Random(rng.randint(0, 99999)), density=0.1) for _ in range(50)]
    worlds_dense = [random_world(10, random.Random(rng.randint(0, 99999)), density=0.7) for _ in range(50)]
    results.append(bench("Sparse 10x10 (10% density)", worlds_sparse))
    results.append(bench("Dense 10x10 (70% density)", worlds_dense))

    print("Generating lesson worlds...")
    lesson_worlds = []
    for seed in range(50):
        try:
            lesson_worlds.append(lesson_world(5, seed))
        except (ValueError, RuntimeError):
            pass
    if lesson_worlds:
        results.append(bench("Generated lessons (5x5)", lesson_worlds))

    print("Generating handcrafted worlds...")
    for length in [5, 10, 20, 50]:
        worlds = [handcrafted_straight_belt(length)]
        results.append(bench(f"Straight belt chain (len={length})", worlds, n_repeats=10))

    for rows in [2, 4, 8]:
        worlds = [handcrafted_zigzag(rows)]
        results.append(bench(f"Zigzag serpentine ({rows} rows)", worlds, n_repeats=10))

    for pairs in [1, 3, 5]:
        worlds = [handcrafted_underground_chain(pairs)]
        results.append(bench(f"Underground chain ({pairs} pairs)", worlds, n_repeats=10))

    worlds = [handcrafted_assembler_line()]
    results.append(bench("Copper cable factory", worlds, n_repeats=10))
    worlds = [handcrafted_electronic_circuit()]
    results.append(bench("Electronic circuit factory", worlds, n_repeats=10))

    for n in [2, 5, 10]:
        worlds = [handcrafted_parallel_paths(n)]
        results.append(bench(f"Parallel paths (n={n})", worlds, n_repeats=10))

    for n in [1, 3, 5]:
        worlds = [handcrafted_inserter_chain(n)]
        results.append(bench(f"Inserter chain (n={n})", worlds, n_repeats=10))

    for size in [5, 15, 30]:
        worlds = [make_world(size)]
        results.append(bench(f"Empty world ({size}x{size})", worlds, n_repeats=10))

    print_results(results)


if __name__ == "__main__":
    main()
