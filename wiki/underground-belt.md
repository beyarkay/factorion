# Underground Belt

A 1x1 belt entrance/exit pair that transports items underground, allowing belts
to cross over each other without interfering.

**Source:** https://wiki.factorio.com/Underground_belt

## Factorio Mechanics

### Basics

- **Size:** 1x1 tile (each end)
- **Research required:** Logistics
- **Crafting:** 1s + 10 iron plates + 5 transport belts → 2 underground belts

### How It Works

Underground belts come in pairs: an **entrance** (items go underground) and an
**exit** (items resurface). They must face the **same direction** — the
entrance faces the direction items travel, and the exit is placed further along
that direction, also facing the same way.

```
[Belt →] [UG-Down →]  ...underground...  [UG-Up →] [Belt →]
```

The underground segment between entrance and exit is invisible and doesn't
interact with anything above it. This allows belts, other underground belts, or
any other entities to occupy the tiles between the pair.

### Maximum Distance

- **Basic underground belt:** 4 tiles between entrance and exit
- Fast underground belt: 6 tiles
- Express underground belt: 8 tiles

If multiple entrances face the same direction, the exit pairs with the
**nearest** entrance.

### Throughput

Same as [[transport-belt]]: **15 items/second**. The underground segment does
not reduce throughput.

### Lane Behavior

Underground belts preserve the [[glossary#lane]] system — items on the left
lane stay on the left lane through the underground segment.

> **Not yet in Factorion.** Lanes aren't modeled, so this doesn't apply.

## Factorion Implementation

### Enum & Channel Values

- **Entity enum:** `EntityKind::UndergroundBelt = 4`
- **Misc channel:** `Misc::UndergroundDown = 1` (entrance), `Misc::UndergroundUp = 2` (exit)
- **Flow rate:** 15.0 items/sec

### Connection Logic

The entrance and exit have different connection behaviors in `entities.rs`:

**Entrance (UndergroundDown):**
- Searches up to 5 tiles ahead (`max_delta = 6`, range `1..6`) in its facing
  direction for an UndergroundBelt with `Misc::UndergroundUp`
- Creates an edge from entrance → exit when found

**Exit (UndergroundUp):**
- `max_delta = 1`, so `range(1, 1)` is **empty** — the exit creates **no
  edges** from its own `connections()` method
- Instead, the downstream [[transport-belt]]'s connection logic detects the
  exit and creates the edge (belt checks its "behind" tile for belt-type
  entities facing the same direction)

This asymmetry means: entrance → exit edge is created by the entrance;
exit → next belt edge is created by the belt.

### Simplifications vs Real Factorio

| Mechanic | Real Factorio | Factorion |
|---|---|---|
| Max distance | 4 tiles (basic) | ~5 tiles (max_delta = 6) |
| Pairing | Nearest matching entrance | First found in range |
| Lanes | Preserved underground | No lanes |
| Crossing | UG belts can cross perpendicular UG belts | Not explicitly handled |
| Tiers | 3 tiers (4/6/8 tile range) | Single tier |

## Interactions

| Entity | Interaction |
|---|---|
| [[transport-belt]] | Entrance receives from belt behind it; exit feeds belt ahead of it. Belt initiates the exit→belt connection. |
| [[inserter]] | In real Factorio, inserters can pick from / place onto UG belt endpoints. In Factorion, inserter can drop onto UG belts (listed as insertable target). |
| [[assembling-machine]] | No direct connection — needs an [[inserter]]. |
| Other [[underground-belt]]s | Perpendicular pairs can cross underground without interfering (real Factorio). Not explicitly modeled in Factorion. |
