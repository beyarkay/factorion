# Splitter

A 1x2 entity that divides, combines, or balances belt throughput. Placed across
two parallel belts to split or merge item flow.

**Source:** https://wiki.factorio.com/Splitter

## Factorio Mechanics

### Basics

- **Size:** 1x2 tiles (spans two belt lanes side by side)
- **Research required:** Logistics
- **Crafting:** 1s + 5 electronic circuits + 5 iron plates + 4 transport belts

### Core Function

A splitter has two input belts and two output belts. It distributes items in a
**1:1 ratio** between its two outputs. The three primary use cases:

1. **Splitting:** One input belt → two output belts (items alternate between
   outputs)
2. **Merging:** Two input belts → one output belt (items interleave onto one
   belt)
3. **Balancing:** Two input belts → two output belts (equalizes throughput
   between the two belts)

### Throughput

Same as [[transport-belt]]: **15 items/second per input belt**. A splitter
can handle two full belts simultaneously (30 items/sec total).

### Lane Behavior

Each [[glossary#lane]] is split independently — left lane items on the input
are distributed to left lanes on the outputs, and same for right lanes. The
splitting decision is independent of item type.

> **Not relevant to Factorion yet.** Lanes aren't modeled.

### Priority Settings

Splitters support **input priority** and **output priority**:

- **Input priority (left/right):** Preferentially drains one input belt before
  the other
- **Output priority (left/right):** Preferentially fills one output belt before
  the other

Combining input + output priority creates a **priority splitter**, commonly
used in main bus designs to keep the primary belt compressed.

> **Not in Factorion.** Priority settings are not planned.

### Filtering

A splitter can be set to filter a specific item type to one output side. The
filtered item goes to the priority output; all other items go to the other
output.

> **Not in Factorion.** Filters are explicitly excluded per README.

### Memory

When one output is blocked, the splitter is forced to send all items to the
other output. It maintains a **memory of up to 5 items** so that when the
blocked side clears, it sends the owed items to balance out.

## Factorion Implementation

### Current Status: Implemented

- **Entity enum:** `EntityKind::Splitter = 7` (Rust), entity value 7 (Python)
- **Size:** 2 tiles wide (perpendicular to flow), 1 tile deep
- **Flow rate:** 30.0 items/sec (2 input belts × 15 items/sec each)
- **Footprint:** Computed via `entity_tiles(x, y, dir, 2, 1)`. Anchor is the
  "left" tile when looking in the flow direction. Tile layout depends on
  direction:
  - **East/West:** anchor (x, y), second tile (x, y+1)
  - **North/South:** anchor (x, y), second tile (x+1, y)

### Connection Logic

For each of the splitter's 2 tiles:

- **Input (behind tile):** Accepts transport belts or underground belts pointing
  in the **same direction** as the splitter. Also accepts sources/sinks with
  matching direction.
- **Output (ahead of tile):** Connects to transport belts, underground belts,
  or sinks that are **not facing the opposite direction** (sideloading allowed).

### Flow Splitting

`transform_flow()` caps input at the splitter's flow rate (30.0). The
throughput calculation then divides output evenly by the number of successors
in the flow graph. With 2 outputs, each gets half the input.

### Lesson Types

Three lesson generators use splitters:

- `SPLITTER_SPLIT`: 1 source → belts → splitter → 2×(belts → sink)
- `SPLITTER_MERGE`: 2×(source → belts) → splitter → belts → 1 sink
- Both are tested across many seeds, grid sizes, and with Rust/Python parity

### Not Implemented

- Priority settings (input/output priority)
- Filtering (item-type routing)
- Memory (rebalancing after blockage clears)

## Interactions

| Entity | Interaction |
|---|---|
| [[transport-belt]] | Primary connection — sits between two belts, splitting or merging flow. |
| [[underground-belt]] | Can connect to underground belt endpoints the same way belts do. |
| [[inserter]] | Not typically used adjacent to splitters in standard designs. |
| [[assembling-machine]] | No direct connection — always has belts + inserters between them. |
