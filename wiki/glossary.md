# Glossary

Definitions of Factorio terms used in these wiki docs. Terms are written from
the perspective of base-game Factorio mechanics; see individual entity docs for
how Factorion simplifies or omits them.

## Belt mechanics

### Lane

Every [[transport-belt]] (and [[underground-belt]], [[splitter]]) has two
independent lanes — a **left lane** and a **right lane** (relative to the
belt's facing direction). Each lane carries items at **7.5 items/sec**
independently of the other; density and speed per lane are constant. Items
stay on their lane unless moved by a [[splitter]], by [[glossary#sideloading|
sideloading]], or by inserter placement/extraction.

> **In Factorion:** every belt-ish tile is two lane nodes in the flow
> graph, each capped at 7.5 items/sec.

### Sideloading

Feeding a belt **perpendicularly** into the side of another belt. Items merge
onto only **one lane** of the receiving belt — the lane on the side the feeder
belt connects from. This is the primary technique for controlling which lane
carries which item type.

Example: a belt pointing East into the **west** side of a belt pointing North
lands its items on the **left lane** of the northbound belt. Reversed (West
belt into the **east** side of a North belt) lands on the right lane.

Common uses:
- **Lane isolation:** put copper on the left lane, iron on the right lane, of
  a single belt feeding a smelter row.
- **Lane compression:** sideloading onto a partly-full belt can fully
  saturate it beyond what forward-loading alone achieves.
- **Lane swapping:** chain two sideloads to swap items from left to right.

> **In Factorion:** a perpendicular feed sideloads onto the near-side lane
> whenever the receiving belt has any other belt-connectable input (a lone
> side feed is a curve instead).

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

## Electricity

### Electric network

A set of electric poles joined by copper wires. Every consumer inside any of
the poles' [[glossary#supply-area|supply areas]] draws from the same supply.
See [[electric-network]].

> **In Factorion:** every pair of poles within wire reach is connected, and a
> network is powered iff one of its poles is within wire reach of a tile
> just outside the grid.

### Supply area

The axis-aligned square, centred on an electric pole, inside which entities
are powered by it: 5x5 for a [[small-electric-pole]], 7x7 for a
[[medium-electric-pole]], 18x18 for a [[substation]]. A multi-tile entity is
powered if any one of its tiles lies in the supply area.

### Wire reach

The maximum distance at which two electric poles can be wired together,
measured between the poles' centres; when two pole types connect, the smaller
reach applies. 7.5 for a [[small-electric-pole]], 9 for a
[[medium-electric-pole]], 18 for a [[substation]].
