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
