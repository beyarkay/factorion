"""Type stubs for the `factorion_rs` PyO3 extension. Signatures only — the
docs live with the implementations (`src/lib.rs`) and callers (`factorion.py`).
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
    produced_by: list[str]
    total_raw: dict[str, float]
    total_raw_time: float

def simulate_throughput(world: NDArray[np.int64]) -> tuple[float, int]: ...
def py_sink_deliveries(
    world: NDArray[np.int64],
) -> list[tuple[int, int, str | None, float]]: ...
def py_build_graph(
    world: NDArray[np.int64],
) -> tuple[list[str], list[tuple[int, int]]]: ...
def py_entity_tiles(
    x: int, y: int, direction: int, width: int, height: int
) -> list[tuple[int, int]] | None: ...
def py_items() -> dict[int, ItemProps]: ...
def py_recipes() -> dict[str, RecipeData]: ...
def py_lesson_kinds() -> dict[str, int]: ...
def build_factory(
    size: int,
    kind: int,
    seed: int,
    random_item: bool = True,
    max_entities: float = float("inf"),
) -> tuple[NDArray[np.int64], int, list[tuple[int, int]]] | None: ...
def render_factory(world: NDArray[np.int64]) -> str: ...
