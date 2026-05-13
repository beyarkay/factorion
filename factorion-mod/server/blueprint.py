"""Tensor → Factorio blueprint string.

We reuse `factorion.dict2b64` for the final compression step; the only work
here is walking the (channels, W, H) world tensor and emitting the
entities-array dict that Factorio's `LuaItemStack.import_stack` consumes.

Conventions matter:
- Position of an N×M entity is at its *center* (so a 1×1 transport belt at
  tile (3, 5) has position {x: 3.5, y: 5.5}).
- Direction in Factorio 2.0 blueprints is a 16-step enum: 0=N, 4=E, 8=S,
  12=W. The model uses factorion.Direction (1..4); convert with `dir*4-4`.
- Inserters' blueprint direction points to their *drop tile*, not pickup.
  The training-time decoder flips them by +8; encoding must flip back.
"""

from __future__ import annotations

from typing import List

import numpy as np

from factorion import Channel, Misc, dict2b64, entities, items

_DIR_MODEL_TO_BP = {0: None, 1: 0, 2: 4, 3: 8, 4: 12}

# The mod's source / sink markers live as stack_inserter / bulk_inserter in
# the obs tensor (they're env-spawned, never placed by the model). For the
# emitted blueprint, render them as `constant-combinator` entities carrying
# two filter signals in their first section:
#   - the item that flows (e.g. "electronic-circuit"), as an item filter
#   - the virtual signal `signal-output` (for source) or `signal-input`
#     (for sink), which round-trip-identifies the marker on re-import.
# This gives a single, paste-able blueprint that visibly distinguishes
# source from sink and tells the player what item the marker is for.
_SOURCE_MARKER_NAME = "stack_inserter"
_SINK_MARKER_NAME = "bulk_inserter"
_COMBINATOR_NAME = "constant-combinator"
_SIGNAL_SOURCE = "signal-output"   # virtual signal name for sources
_SIGNAL_SINK = "signal-input"      # virtual signal name for sinks

# Model Direction enum (1=N, 2=E, 3=S, 4=W) → arrow virtual signal.
# Note these are NOT prefixed with "signal-" despite the related
# signal-input / signal-output / signal-anything prototypes that ARE
# prefixed. The four arrow signals just live as bare "up-arrow", etc.
_DIR_SIGNAL = {
    1: "up-arrow",
    2: "right-arrow",
    3: "down-arrow",
    4: "left-arrow",
}


def _combinator_marker(cx: float, cy: float, item_name: str | None,
                       signal_name: str, direction_model: int) -> dict:
    """Build a constant-combinator dict carrying three filters in its
    first section: the flowing item, a source/sink virtual signal, and
    an arrow virtual signal for the marker's direction."""
    filters = []
    if item_name and item_name != "empty":
        filters.append({
            "index": 2,
            "name": _hyphenate(item_name),
            "quality": "normal",
            "comparator": "=",
            "count": 1,
        })
    filters.append({
        "index": 3,
        "type": "virtual",
        "name": signal_name,
        "quality": "normal",
        "comparator": "=",
        "count": 1,
    })
    arrow = _DIR_SIGNAL.get(direction_model)
    if arrow is not None:
        filters.append({
            "index": 4,
            "type": "virtual",
            "name": arrow,
            "quality": "normal",
            "comparator": "=",
            "count": 1,
        })
    return {
        "name": _COMBINATOR_NAME,
        "position": {"x": cx, "y": cy},
        "direction": 4,  # face East — combinators don't direct flow visually
        "control_behavior": {
            "sections": {
                "sections": [
                    {"index": 1, "filters": filters},
                ],
            },
        },
    }


def _hyphenate(name: str) -> str:
    return name.replace("_", "-")


def _make_entity(
    name: str,
    cx: float,
    cy: float,
    direction: int | None,
    *,
    recipe: str | None = None,
    type_: str | None = None,
) -> dict:
    e: dict = {"name": name, "position": {"x": cx, "y": cy}}
    if direction is not None:
        # Inserter pickup→drop flip must happen *before* eliding the
        # default direction=0; otherwise a model-NORTH inserter (whose
        # drop side is SOUTH = bp 8) would be emitted with no
        # `direction` field and round-trip as SOUTH.
        if "inserter" in name:
            direction = (direction + 8) % 16
        if direction != 0:
            e["direction"] = direction
    if recipe is not None:
        e["recipe"] = recipe
    if type_ is not None:
        e["type"] = type_
    return e


def world_tensor_to_blueprint_dict(
    world_CWH: np.ndarray,
    *,
    label: str | None = None,
    description: str | None = None,
) -> dict:
    """Convert a (channels, W, H) numpy tensor to a Factorio blueprint dict.

    Skips:
      - empty cells (entity id 0)
      - source/sink (entity ids 6/7) — these are env-spawned markers, not
        things the player wants in their blueprint.
      - non-anchor tiles of multi-tile entities (we use the ITEMS channel
        being non-zero, plus MISC for undergrounds, to disambiguate; for
        the MVP we emit one entity per *unique anchor* by tracking visited
        tiles via a simple "first sighting wins" rule on the (entity,
        direction) pair).
    """
    assert world_CWH.ndim == 3, f"expected (C, W, H), got {world_CWH.shape}"
    _, W, H = world_CWH.shape

    ent_ch = world_CWH[Channel.ENTITIES.value]
    dir_ch = world_CWH[Channel.DIRECTION.value]
    item_ch = world_CWH[Channel.ITEMS.value]
    misc_ch = world_CWH[Channel.MISC.value]

    out_entities: List[dict] = []
    seen = np.zeros((W, H), dtype=bool)

    for x in range(W):
        for y in range(H):
            if seen[x, y]:
                continue
            ent_id = int(ent_ch[x, y])
            if ent_id == 0:
                continue
            ent_meta = entities.get(ent_id)
            if ent_meta is None or not ent_meta.is_placeable:
                continue

            dir_model = int(dir_ch[x, y])
            dir_bp = _DIR_MODEL_TO_BP.get(dir_model, None)

            # Center of an N×M entity is at the geometric midpoint of its
            # tile span. For 1×1 this is +0.5 on both axes.
            cx = x + ent_meta.width / 2.0
            cy = y + ent_meta.height / 2.0

            # Source/sink markers from the request: re-render as
            # constant-combinators carrying virtual-signal labels so the
            # blueprint visibly shows source vs sink and which item.
            if ent_meta.name in (_SOURCE_MARKER_NAME, _SINK_MARKER_NAME):
                item_id = int(item_ch[x, y])
                item_meta = items.get(item_id)
                item_name = item_meta.name if item_meta is not None else None
                signal = (_SIGNAL_SOURCE
                          if ent_meta.name == _SOURCE_MARKER_NAME
                          else _SIGNAL_SINK)
                out_entities.append(_combinator_marker(
                    cx, cy, item_name, signal, dir_model,
                ))
                seen[x, y] = True
                continue

            name = _hyphenate(ent_meta.name)
            recipe: str | None = None
            type_: str | None = None

            item_id = int(item_ch[x, y])
            if name == "assembling-machine-1" and item_id != 0:
                recipe_meta = items.get(item_id)
                if recipe_meta is not None:
                    recipe = _hyphenate(recipe_meta.name)

            if name == "underground-belt":
                misc = int(misc_ch[x, y])
                if misc == Misc.UNDERGROUND_DOWN.value:
                    type_ = "input"
                elif misc == Misc.UNDERGROUND_UP.value:
                    type_ = "output"

            out_entities.append(
                _make_entity(name, cx, cy, dir_bp, recipe=recipe, type_=type_)
            )

            # Mark every tile this entity occupies as visited so we don't
            # emit it again. For axis-rotated entities width/height swap.
            w, h = ent_meta.width, ent_meta.height
            if dir_model in (2, 4):  # EAST or WEST → swap dims
                w, h = h, w
            for dx in range(w):
                for dy in range(h):
                    tx, ty = x + dx, y + dy
                    if 0 <= tx < W and 0 <= ty < H:
                        seen[tx, ty] = True

    # Number entities — Factorio doesn't strictly require entity_number on
    # import but blueprint tooling expects it, so we add one.
    for i, e in enumerate(out_entities, start=1):
        e["entity_number"] = i

    bp: dict = {
        "entities": out_entities,
        "item": "blueprint",
        # Factorio 2.0.x version stamp. The wrong one (1.x: 281479275675648)
        # makes Factorio silently strip 2.0-only entities like
        # constant-combinators with the new sections-of-sections schema.
        "version": 562949958402048,
    }
    if label is not None:
        bp["label"] = label
    if description is not None:
        bp["description"] = description
    return {"blueprint": bp}


def world_tensor_to_blueprint_string(
    world_CWH: np.ndarray,
    *,
    label: str | None = None,
    description: str | None = None,
) -> str:
    return dict2b64(world_tensor_to_blueprint_dict(
        world_CWH, label=label, description=description,
    ))
