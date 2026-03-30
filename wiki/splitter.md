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

### Current Status: Not Yet Implemented

The splitter is listed as a planned entity in the README ("Factorio environment
rewrite/overhaul" section). It does not currently exist in the `EntityKind`
enum or in `entities.rs`.

### Implementation Notes

When added, the splitter will need:

- A new `EntityKind` variant (value TBD)
- 1x2 footprint handling (like [[assembling-machine]]'s 3x3, but 1x2)
- Connection logic: accept input from the two tiles behind, output to the two
  tiles ahead
- `transform_flow`: split input evenly across outputs (or just pass-through
  at 15 items/sec per side if modeled as a simple flow entity)
- No need for priority or filter mechanics per current project scope

## Interactions

| Entity | Interaction |
|---|---|
| [[transport-belt]] | Primary connection — sits between two belts, splitting or merging flow. |
| [[underground-belt]] | Can connect to underground belt endpoints the same way belts do. |
| [[inserter]] | Not typically used adjacent to splitters in standard designs. |
| [[assembling-machine]] | No direct connection — always has belts + inserters between them. |
