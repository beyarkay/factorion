# Glossary

Definitions of Factorio terms used in these wiki docs. Terms are written from
the perspective of base-game Factorio mechanics; see individual entity docs for
how Factorion simplifies or omits them.

## Belt mechanics

### Lane

Every [[transport-belt]] (and [[underground-belt]], [[splitter]]) has two
independent lanes — a **left lane** and a **right lane**. Each lane carries
items independently at half the belt's total throughput. Items stay on their
lane unless moved by a [[splitter]] or by sideloading.

> **Not yet in Factorion.** Belts are modeled as single-lane pipes.

### Sideloading

Feeding a belt **perpendicularly** into the side of another belt. Items merge
onto only one lane of the receiving belt (the lane on the side the feeder belt
connects to). This is the primary technique for controlling which lane carries
which item type.

Example: A belt pointing East meets the side of a belt pointing North. Items
from the eastbound belt go onto the **right lane** of the northbound belt.

> **Not yet in Factorion.** Requires the lane system.

### Throughput

The rate at which items flow through an entity, measured in **items per
second**. Key throughput numbers:

| Entity | Throughput |
|---|---|
| [[transport-belt]] | 15 items/s |
| [[underground-belt]] | 15 items/s |
| [[splitter]] | 15 items/s (per input belt) |
| [[inserter]] (basic) | 0.86 items/s |
| [[assembling-machine]] | 0.5 crafts/s (base speed) |

### Compression

A belt is **fully compressed** (or "saturated") when it carries items at
maximum density with no gaps. A fully compressed basic belt moves 15 items/sec.
Gaps between items reduce effective throughput.

> **Not in Factorion.** Throughput is calculated via graph flow analysis, not
> discrete item simulation.

## Entity mechanics

### Anchor tile

For multi-tile entities like the [[assembling-machine]] (3x3), the **anchor
tile** is the top-left corner of the entity's footprint. The anchor's (x, y)
coordinate is what's stored in the grid tensor and used for graph node IDs.

### Perimeter

The ring of tiles immediately surrounding a multi-tile entity. For a 3x3
[[assembling-machine]] anchored at (x, y), the perimeter is the set of tiles
adjacent to the 3x3 body (excluding corners). [[Inserter]]s must be placed on
the perimeter to interact with the machine.

### Recipe

A mapping of input items to output items with a crafting time. An
[[assembling-machine]] must be assigned a recipe to produce anything. See
[[items-and-recipes]] for the recipes Factorion implements.

## Underground belt mechanics

### Entrance / Exit (Down / Up)

An [[underground-belt]] has two modes controlled by the `Misc` channel:

- **Down** (`Misc::UndergroundDown = 1`): The entrance — items go underground
- **Up** (`Misc::UndergroundUp = 2`): The exit — items resurface

A down belt searches ahead (in its facing direction) for a matching up belt
within range. The underground segment between them is invisible and passes
through any entities above it.
