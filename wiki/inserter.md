# Inserter

A 1x1 robotic arm that picks up items from one tile and places them on an
adjacent tile. The only way to move items between belts and machines.

**Source:** https://wiki.factorio.com/Inserter

## Factorio Mechanics

### Basics

- **Size:** 1x1 tile
- **Available from:** Game start (no research required)
- **Power consumption:** 15.1 kW active, 400 W drain

### Pickup & Drop Behavior

An inserter **picks up** from the tile **behind** it (opposite of its facing
direction) and **drops** onto the tile **ahead** (in its facing direction):

```
   [Source] ← picks from here
   [Inserter →]
   [Destination] ← drops here
```

So an inserter facing East picks up from the tile to its West and drops onto
the tile to its East.

### Throughput

- **Basic inserter:** ~0.83 items/sec (one item per swing)
- **Stack size:** 1 item per swing (basic inserter)

The wiki doesn't give an exact items/sec number for the basic inserter. The
swing speed is governed by rotation speed (302°/sec at normal quality). The
Factorion implementation uses 0.86 items/sec.

### Lane Interaction

In real Factorio, inserters interact with [[glossary#lane|belt lanes]]
asymmetrically:

- **Placing onto a belt:** Items go on the **far lane** (the lane on the far
  side of the belt from the inserter). When belt and inserter face the same
  direction, this is the **right lane from the belt's perspective**.
- **Picking from a belt:** Inserters **prefer the nearest lane** and only
  reach to the far lane if the near one is empty. For belts parallel to the
  inserter, "nearest" resolves to the **left lane from the belt's
  perspective**.

This asymmetry (drop far, pick near) is deliberate: it means an inserter
inserting into a belt and a second inserter extracting from the same belt
downstream will not short-circuit — the extracting inserter sees the item on
the lane it doesn't read first.

> **In Factorion:** drops target the far lane (right lane when the belt is
> parallel/anti-parallel); pickups greedily drain the near lane first
> (left lane when in-line or curved), topping up from the other lane.

### Variants

| Variant | Reach | Stack size | Notes |
|---|---|---|---|
| Inserter (basic) | 1 tile | 1 | Standard |
| Long-handed inserter | 2 tiles | 1 | Reaches over a belt |
| Fast inserter | 1 tile | 1 | 2x speed |
| Stack inserter | 1 tile | up to 12 | Bulk transfer |
| Bulk inserter | 1 tile | up to 12 | Space Age replacement for stack |

> **Only basic inserter in Factorion.** Long-handed inserter is listed as a
> future possibility in the README.

### Filtering

All inserters can be set to filter specific item types (as of Factorio 2.0).

> **Not in Factorion.** Inserter filters are explicitly excluded.

## Factorion Implementation

### Enum & Channel Values

- **Entity enum:** `EntityKind::Inserter = 2`
- **Flow rate:** 0.86 items/sec

### Connection Logic

The inserter connection logic in `entities.rs` (`inserter_connections`) is
shared by Inserter, Source, and Sink:

- **Picks up from behind:** Creates an edge from any non-empty entity at
  `(x - dx, y - dy)` to the inserter
- **Drops onto ahead:** Creates an edge from the inserter to the entity at
  `(x + dx, y + dy)`, but **only** if that entity is a [[transport-belt]],
  [[underground-belt]], or [[assembling-machine]]

An inserter cannot drop onto empty tiles or onto other inserters.

### Blueprint Direction Convention (gotcha)

**Factorio blueprint strings encode an inserter's `direction` as the
direction the arm points toward its *pickup* tile**, not the drop tile.
This is the **opposite** of the Factorion convention (where `Direction`
points toward the drop tile, matching "facing direction").

Example: an inserter in a Factorio blueprint with `direction: 0` (NORTH)
picks from its north neighbour and drops to its south neighbour. The
same inserter inside Factorion would be stored with `Direction::South`.

`blueprint2world` in `factorion.py` handles this by flipping inserter
directions 180° on decode:

```python
if "inserter" in e["name"]:
    e["direction"] = (e.get("direction", 0) + 8) % 16
```

When manually reading blueprint JSON without going through
`blueprint2world`, always apply this flip before comparing to Factorion
semantics. Non-inserter entities (belts, assemblers, etc.) do not need
the flip — only inserters (including Source/Sink, which are bulk/stack
inserters in the blueprint namespace).

### Source & Sink

Source (`EntityKind::Source = 6`) and Sink (`EntityKind::Sink = 5`) are
Factorion-specific entities that reuse inserter connection logic:

- **Source** produces infinite items of a configured type. It uses inserter
  pickup/drop rules to feed items onto belts or into machines.
- **Sink** consumes items with infinite throughput. It uses inserter
  pickup/drop rules to pull items off belts or out of machines.

These are named `stack_inserter` and `bulk_inserter` internally (matching
Python legacy names) but behave as infinite-capacity item endpoints.

### Simplifications vs Real Factorio

| Mechanic | Real Factorio | Factorion |
|---|---|---|
| Lane awareness | Drop on far lane, pick from near lane preferentially | No lanes |
| Stack size | 1 per swing (basic) | Not modeled — flow rate only |
| Power | Requires electricity | No power simulation |
| Variants | 5+ types | Basic only (+ Source/Sink) |
| Filtering | Per-item filters | Not modeled |

## Interactions

| Entity | Interaction |
|---|---|
| [[transport-belt]] | Picks items off belt (from behind) / places items onto belt (ahead). Primary use case. |
| [[assembling-machine]] | Feeds ingredients in / extracts products out. Must be on the machine's perimeter. Direction determines flow: facing away = extracting, facing toward = inserting. |
| [[underground-belt]] | Can pick from / place onto underground belt tiles, same as regular belts. |
| Source / Sink | Share inserter connection logic. Source: infinite output. Sink: infinite input. |
