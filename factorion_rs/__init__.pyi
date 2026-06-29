"""Type stubs for the `factorion_rs` PyO3 extension module.

The compiled extension is built from `factorion_rs/src/lib.rs` (see the
`#[pymodule] fn factorion_rs` there). These stubs let type checkers resolve
`import factorion_rs` against the actual exported functions instead of the
crate source directory.
"""

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray

class ItemProps(TypedDict):
    name: str
    is_placeable: bool
    width: int
    height: int
    flow: float

class RecipeData(TypedDict):
    consumes: dict[str, float]
    produces: dict[str, float]
    crafting_time: float

def simulate_throughput(world: NDArray[np.int64]) -> tuple[float, int]:
    """(throughput, num_unreachable) for a (W, H, C) int64 world array."""
    ...

def py_build_graph(
    world: NDArray[np.int64],
) -> tuple[list[str], list[tuple[int, int]]]:
    """Factory connection graph for a (W, H, C) int64 world array.

    Returns (node_labels, edges): node labels in the engine's
    ``"{entity_name}\\n@{x},{y}"`` format, and ``(src_index, dst_index)`` edges
    indexing into them.
    """
    ...

def py_entity_tiles(
    x: int, y: int, direction: int, width: int, height: int
) -> list[tuple[int, int]] | None:
    """Tiles occupied by a multi-tile entity, or None for an invalid direction."""
    ...

def py_items() -> dict[int, ItemProps]:
    """Every Item keyed by int value -> {name, is_placeable, width, height, flow}."""
    ...

def py_recipes() -> dict[str, RecipeData]:
    """Every recipe keyed by item name -> {consumes, produces, crafting_time}."""
    ...

def build_factory(
    size: int,
    kind: int,
    seed: int,
    random_item: bool = True,
    max_entities: float = float("inf"),
) -> tuple[NDArray[np.int64], int, list[tuple[int, int]]] | None:
    """Rust port of ``factorion.build_factory``.

    Builds a complete, valid factory of the given lesson ``kind`` (an int
    ``LessonKind`` value), drawing from the same CPython MT19937 stream as
    Python so the result is byte-identical for the same ``(size, kind,
    seed)``. Returns ``(world, total_entities, protected_positions)`` where
    ``world`` is a ``(W, H, C)`` int64 array, or ``None`` when rejection
    sampling is exhausted. Raises ``ValueError`` for an unknown kind.
    """
    ...
