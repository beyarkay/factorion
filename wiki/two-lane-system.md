# Two-Lane System

Every belt in Factorio is actually **two parallel lanes** that carry items
independently. This is one of the most important cross-cutting mechanics in
the game — it touches belts, inserters, splitters, sideloading, and factory
design patterns like "main bus".

The throughput engine models every belt-ish tile as two lane nodes in the
flow graph — belts, underground belts, and each of a splitter's two tiles —
with a 7.5 items/sec cap per lane, lane-preserving straight/curve/tunnel/
splitter flow, sideloading onto the near-side lane, and the inserter
drop/pickup lane rules below. This page documents the Factorio mechanics
the engine mirrors.

## Canonical specs

From [Transport belts/Physics](https://wiki.factorio.com/Transport_belts/Physics):

> "Belts have two parallel lanes, and the density and speed of each lane is
> constant and independent of the other one."

| Property | Per lane | Both lanes |
|---|---|---|
| Throughput | 7.5 items/sec | 15 items/sec |
| Density | 4 items/tile | 8 items/tile |
| Speed | 1.875 tiles/sec | 1.875 tiles/sec |
| Internal positions | 256/tile | 512/tile |

Lanes are labelled **left** and **right** relative to the belt's facing
direction.

## Rules for which lane an item ends up on

### Forward belt-to-belt flow

Items travelling straight along connected belts **stay on their lane**. Left
stays left, right stays right. This is the simple case.

### Curves

Curved belts preserve lane assignments, but the outer lane is
**1.15234375× longer** (295/256) than the inner lane. Items exit the curve
at different times but per-lane throughput is unchanged.

### Sideloading (perpendicular feed)

When a belt feeds into the **side** of another belt, its items are merged
onto **only one lane** of the receiving belt — the lane on the side the
feeder belt connects from.

- East belt → west side of North belt → items land on the **left lane** of
  the northbound belt.
- West belt → east side of North belt → items land on the **right lane**.

This is the primary mechanism for **lane control** (isolating an item type
to one lane, swapping lanes, or compressing a partially-full belt).

### Inserters

See [[inserter]].

- **Drop:** always on the **far lane** (the lane on the opposite side of the
  belt from the inserter). When belt and inserter face the same direction,
  that's the right lane from the belt's perspective.
- **Pickup:** **prefers the near lane**; falls back to far lane if near is
  empty.

The drop/pickup asymmetry means a belt loaded by an inserter upstream and
drained by an inserter downstream won't double-count — the drainer looks at
a different lane than the loader targeted.

### Splitters

See [[splitter]]. Splitters **preserve lanes**:

> "The left and right lane splitting is now completely independent."
> — [Splitter](https://wiki.factorio.com/Splitter)

This means the routing decision (which output belt) is made independently
per lane, **not** that items move between lanes. An item on the left lane
of an input exits on the left lane of whichever output it's routed to.

**No passive entity in Factorio moves items between lanes.** Lane swapping
requires a **pair of sideloads**.

## Design patterns that depend on lanes

Several standard Factorio factory patterns exist *only because* belts have
two lanes:

- **Main bus:** a row of belts with distinct item types on each lane of each
  belt. Sideloading is used to pull a partial bandwidth off onto branch
  belts.
- **Lane balancer:** a pattern of splitters + sideloads that evens out
  uneven lane loads (common when a smelter array dumps onto a belt
  asymmetrically).
- **Lane compressor:** sideloading a partly-full belt onto itself via a
  loop to fully saturate both lanes.

None of these patterns are meaningful in Factorion's single-lane model.

## What lanes would cost Factorion to add

Rough blast radius if we wanted to add two-lane support:

### Data model

- Every tile channel carrying item flow would need to become two channels
  (per-lane). Alternatively, keep one channel but double the node count in
  the flow graph (one node per (tile, lane) pair).
- Throughput calculation becomes a graph with ~2× nodes and asymmetric
  edges (sideload = perpendicular cross-lane edge).

### Connection logic (in `entities.rs`)

- Belt→belt forward connection: one edge per lane (2 edges instead of 1).
- Sideload: one edge from feeder belt (both lanes combined) → specific lane
  of receiver (1 edge, but to a per-lane node).
- Inserter drop: edge to `(receiver_belt, far_lane)`.
- Inserter pickup: two edges with priority (near > far).
- Splitter: lane-preserving, so edges only connect left→left and
  right→right, but each lane distributes independently across both outputs
  (4 edges per input belt: left_in→left_out0, left_in→left_out1,
  right_in→right_out0, right_in→right_out1). Roughly doubles splitter
  edge count.
- Underground belt: lanes persist through the underground segment (2
  edges).

### Lesson generators

Every lesson that places a belt, inserter, or splitter would need to reason
about lane assignments. Existing `SPLITTER_SPLIT` / `SPLITTER_MERGE_SIDELOADED`
tests that assert throughput caps would need updating (per-lane caps differ from
combined caps). `SPLITTER_MERGE_SIDELOADED` leans on the per-lane cap directly:
side-loading a saturated source fills only the near lane (7.5), which is what
makes both merge arms necessary.

### Throughput caps

Current cap "15 i/s per belt" remains, but many scenarios become more
restrictive: e.g. a single inserter feeding a belt can saturate at most
7.5 i/s (one lane) rather than 15 i/s.

### Tests

All throughput tests (`tests/test_handcrafted.py`) would need to assert
per-lane throughput, not just combined throughput.

## When to add lanes

Adding lanes is only worth it if:

1. We want the agent to learn **lane control** as a skill (sideloading,
   lane compression). This is a meaningful factory-design primitive.
2. We care about **throughput accuracy** for scenarios where single-lane
   modelling overstates capacity (e.g. an inserter feeding a belt capped at
   7.5 i/s, not 15).
3. We're targeting layouts that rely on main-bus-style designs where each
   lane carries a distinct item type.

If the current curriculum target (single-item factories up to green
circuits) can be expressed without lane reasoning, we can defer lanes
indefinitely.

## Sources

- [Transport belt](https://wiki.factorio.com/Transport_belt)
- [Transport belts/Physics](https://wiki.factorio.com/Transport_belts/Physics)
- [Inserter](https://wiki.factorio.com/Inserter)
- [Splitter](https://wiki.factorio.com/Splitter)
