# Transport Belt

A 1x1 conveyor that moves items in one direction. The most basic logistics
entity and the backbone of every factory.

**Source:** https://wiki.factorio.com/Transport_belt

## Factorio Mechanics

### Basics

- **Size:** 1x1 tile
- **Available from:** Game start (no research required)
- **Crafting:** 0.5s + 1 iron gear wheel + 1 iron plate → 2 transport belts

### Movement & Throughput

- **Speed:** 1.875 tiles per second
- **Throughput:** 15 items/second (both lanes combined)
- **Density:** Up to 8 items per tile (4 per lane)

### Two-Lane System

Every belt has a **left lane** and a **right lane** (relative to the belt's
facing direction), and the two are **completely independent**:

> "Belts have two parallel lanes, and the density and speed of each lane is
> constant and independent of the other one." — [Transport belts/Physics](https://wiki.factorio.com/Transport_belts/Physics)

- Each lane carries half the total throughput: **7.5 items/sec per lane**,
  15 items/sec combined
- Each lane holds **4 items per tile** (8 total per tile)
- Items stay on their lane unless moved by a [[splitter]] or by
  [[glossary#sideloading|sideloading]]
- Internally each tile has **256 positions per lane**; saturated items sit
  64 positions apart

**Inserter lane targeting** ([Inserter](https://wiki.factorio.com/Inserter)):

- **Placing onto a belt:** the inserter drops onto the **far lane** (the lane
  on the far side of the belt from the inserter). When the belt and inserter
  face the same direction, this is the **right lane from the belt's
  perspective**.
- **Picking from a belt:** the inserter **prefers the nearest lane** and only
  reaches to the far lane if the near one is empty. For belts parallel to the
  inserter, "nearest" resolves to the left lane from the belt's perspective.

> **Not yet in Factorion:** The two-lane system is not modeled. Belts are
> treated as single-lane pipes. If lanes are added later, sideloading and
> inserter lane targeting would also need to be implemented.

### Curves & Placement

- Belts auto-connect when placed adjacent in compatible directions
- A belt curves when placed at a 90-degree angle to an adjacent belt
- **Curve geometry:** the outer lane of a turn is **1.15234375× longer**
  (295/256) than a straight belt. Items on the outer lane travel a longer
  path at the same speed, so they exit the curve later than items on the
  inner lane — this staggers item arrivals but doesn't reduce per-lane
  throughput.
- Items do not fall off the end of a belt; they stop and wait

> **Not yet in Factorion:** Belt curving and curve geometry are not modeled.

### Sideloading

When a belt feeds into the **side** of another belt (perpendicular), items
merge onto only **one lane** of the receiving belt — the lane on the side the
feeder belt connects from. This is called [[glossary#sideloading|sideloading]]
and is the primary technique for **lane control**:

- Belt pointing East into the west side of a belt pointing North → items go
  onto the **left lane** of the northbound belt
- Belt pointing West into the east side of a belt pointing North → items go
  onto the **right lane** of the northbound belt
- Used to: isolate a single item type onto one lane, merge two item types onto
  separate lanes of one belt, or reach "lane compression" past a long chain

Sideloading also has travel-distance mechanics (late sideloads travel 68
internal positions, early sideloads travel 188) that affect how a sideloading
junction behaves under load.

> **Not yet in Factorion:** Sideloading is not modeled (requires lanes).

### Tiers

There are three belt tiers in base Factorio. Only the first tier is relevant
to Factorion:

| Tier | Throughput | Speed (tiles/s) |
|---|---|---|
| Transport belt | 15 items/s | 1.875 |
| Fast transport belt | 30 items/s | 3.75 |
| Express transport belt | 45 items/s | 5.625 |

## Factorion Implementation

### Enum & Channel Values

- **Entity enum:** `EntityKind::TransportBelt = 1`
- **Direction channel:** `Direction::North = 1, East = 2, South = 3, West = 4`
- **Flow rate:** 15.0 items/sec

### Connection Logic

In the throughput graph (`entities.rs`), a transport belt creates edges based
on its facing direction:

- **From behind:** Connects from the tile behind (opposite of facing direction)
  if that tile is a [[transport-belt]] or [[underground-belt]] facing the same
  direction (and not an underground-down entrance)
- **To ahead:** Connects to the tile ahead if that tile is a belt-type entity
  and is **not** facing the opposite direction (head-on belts don't connect)

A belt does **not** connect to [[inserter]]s or [[assembling-machine]]s — those
entities initiate their own connections.

### Simplifications vs Real Factorio

| Mechanic | Real Factorio | Factorion |
|---|---|---|
| Lanes | Two independent lanes | Single-lane pipe |
| Curves | Affect throughput | No curve modeling |
| Sideloading | Merges onto one lane | Not modeled |
| Item tracking | Discrete items on belt | Graph-based flow analysis |
| Tiers | 3 tiers (15/30/45 items/s) | Single tier (15 items/s) |

## Interactions

| Entity | Interaction |
|---|---|
| [[inserter]] | Picks items off / places items onto belt. In real Factorio this is lane-aware; in Factorion it's a single flow. |
| [[underground-belt]] | Entrance consumes from belt; exit feeds back onto belt. Same throughput (15/s). |
| [[assembling-machine]] | Cannot directly connect — requires an [[inserter]] in between. |
| [[splitter]] | Splits/merges belt flow. Not yet in Factorion. |
| Source / Sink | Factorion-specific: use inserter connection logic to feed onto / consume from belts. |
