# Future Lesson Ideas

Brainstormed lesson types beyond the current `MOVE_ONE_ITEM` curriculum. Each section is a self-contained lesson concept that can be tackled independently. They are loosely ordered by implementation priority.

Current state of the world:
- Grid is 5x5, single source + single sink, transport belts only
- Entities defined but unused in lessons: `inserter`, `underground_belt`, `assembling_machine_1`
- Splitters are **not** in the entity list yet (commented out as entity 13)
- Multi-source/multi-sink throughput calculation already works in `factorion_rs`
- There is dead code / plans for a "mask" channel (per-tile 0/1 indicating whether the agent can edit that tile), but it is not yet implemented

---

## 1. Mask Channel

**What**: Add a 5th channel to the world tensor — a binary mask where 1 = editable, 0 = locked. The agent cannot place or remove entities on locked tiles. The mask is part of the observation but not part of the action space.

**Why this is foundational**: Nearly every other lesson idea benefits from the mask. It lets us:
- Create obstacles (lock tiles with entities on them)
- Guide the agent toward correct solutions (only unmask tiles on the solution path)
- Train on a large grid but restrict the active area (start with 4x4 unmasked in a 10x10 grid, gradually expand)
- Present pre-built factory sections the agent must work around

**Scaling**:
- Easy: Unmask only the exact tiles where belts need to go (agent just needs to pick the right entity/direction)
- Medium: Unmask a corridor wider than the solution path (agent must choose which tiles to use)
- Hard: Unmask the entire grid (current behavior, no guidance)

**Implementation sketch**:
1. Add `MASK = 4` to the `Channel` enum
2. `new_world()` initializes mask to all 1s (fully editable) by default
3. In `step()`, reject actions on tiles where mask == 0 (treat as invalid action)
4. Lesson generators set mask based on difficulty

---

## 2. Multi-Source/Multi-Sink (Crossing Paths)

**What**: Place multiple source/sink pairs on the grid, each carrying a different item. The simplest version: source A (item X) on the left, sink A (item X) on the right, source B (item Y) on the top, sink B (item Y) on the bottom. The paths must cross, which at higher difficulty forces underground belts.

**Why**: This is the natural introduction to both multi-item logistics and underground belts. The level-0 version doesn't even require underground belts — it just requires the agent to learn that multiple independent transport tasks exist simultaneously.

**Scaling**:
- Level 0: Two source/sink pairs whose shortest paths don't cross (parallel). Remove 1-N belts. Teaches multi-path awareness.
- Level 1: Two pairs whose paths must cross. Provide a working solution using underground belts. Remove 1 entity.
- Level 2: Remove more entities from crossing solutions.
- Level 3+: Three or four source/sink pairs, more complex crossing patterns.

**Generation algorithm**:
1. Place K source/sink pairs at grid edges, each with a distinct item
2. For non-crossing layouts: find independent shortest paths, fill belts, remove N
3. For crossing layouts: find paths that intersect, resolve crossings with underground belt pairs (down/up), fill solution, remove N entities

**Prerequisites**: The throughput engine already handles multiple sources/sinks. Need to update `generate_lesson()` to place multiple pairs and solve crossing layouts.

**Throughput definition**: Sum of throughput across all source/sink pairs, normalized by the number of pairs. Each pair's throughput is independently measured.

---

## 3. Inserter Chains (Colinear Belt Transfer)

**What**: Two or more belt segments connected by inserters. Items flow along belt A, an inserter picks items off belt A and places them onto belt B, items continue along belt B to the sink.

**Why**: Inserters are the second fundamental mechanic. In real Factorio, inserters are how items move between belts and machines. This lesson teaches the agent inserter placement and direction.

**Scaling**:
- Level 0: Full solution in place (belt-belt-inserter-belt-belt-sink), remove 0 entities. Agent learns "don't break it."
- Level 1: Remove the inserter only. Agent must place 1 inserter with correct direction.
- Level 2: Remove the inserter + adjacent belts.
- Level 3+: Multiple inserter hops in series (belt -> inserter -> belt -> inserter -> belt).

**Generation algorithm**:
1. Pick a grid layout with K segments of belts, each 2-4 tiles long
2. Connect segments with inserters (inserter faces from source belt toward destination belt)
3. Place source at start of first segment, sink at end of last segment
4. Remove N entities (prioritize removing inserters first for easier scaling)

**Key constraint**: Inserters pick from the tile they face away from and place onto the tile they face toward. Direction is critical — a wrong-facing inserter does nothing. The inserter's flow rate (0.86) is lower than belts (15.0), so it becomes the bottleneck. Throughput normalization should account for this.

**Variations**:
- L-shaped: belt goes east, inserter transfers to a belt going south
- Parallel: two belt lines with inserters transferring between them

---

## 4. Splitters (Merge and Split)

**What**: Introduce the splitter entity (2x1, two inputs, two outputs, balances items equally). Use it for both merging (2 inputs -> 1 used output) and splitting (1 input -> 2 outputs) lessons.

**Why**: Splitters are how Factorio handles merging and splitting. They're a single entity that covers both use cases. Essential for any multi-lane factory.

**Scaling**:
- Level 0: Pre-built merge/split with splitter in place, remove 0 entities
- Level 1: Remove the splitter, agent must place it with correct direction
- Level 2: Remove splitter + surrounding belts
- Level 3+: Splitter chains (balancers), multi-stage merging

**Prerequisites**: Splitter is currently commented out as entity 13 (2x1 size, flow 15.0). Need to:
1. Uncomment and integrate into the entity list
2. Add splitter placement logic in `step()` (2x1 entity, occupies 2 tiles)
3. Update throughput engine to handle splitter flow (equal split of input to both outputs)

**Generation algorithm**:
- Merge: Place 2 sources feeding belts into a splitter, one output goes to sink
- Split: Place 1 source feeding a splitter, two outputs go to two sinks

---

## 5. Underground Belt Tunneling

**What**: Standalone underground belt lessons (separate from the crossing-paths lesson above). A straight path with a gap in the middle that must be bridged by an underground belt pair.

**Why**: Underground belts have unique mechanics (down/up pairing, max distance, tunneling through occupied tiles). Worth teaching in isolation before combining with crossings.

**Scaling**:
- Level 0: Full solution with underground pair in place, remove 0 entities
- Level 1: Remove one half of the underground pair
- Level 2: Remove both underground belt tiles
- Level 3: Remove underground pair + surrounding belts
- Level 4+: Multiple underground pairs in sequence; longer tunnels

**Generation algorithm**:
1. Create a straight belt path from source to sink
2. Pick a segment of 2-4 tiles and replace it with: underground_down -> (empty gap) -> underground_up
3. Optionally place obstacles (locked tiles) in the gap to justify the tunnel
4. Remove N entities from the solution

**Key constraint**: Underground belts have a max tunnel distance (4 tiles for basic underground belt in Factorio). The `MISC` channel already supports `UNDERGROUND_DOWN` and `UNDERGROUND_UP` values.

---

## 6. Adaptive Grid Size via Mask

**What**: Train on a 10x10 (or larger) grid from the start, but use the mask channel to restrict the playable area. Gradually expand the unmasked region as the agent improves.

**Why**: Avoids retraining on a new grid size. The agent learns to respect boundaries and work within constraints. The CNN architecture sees the same input dimensions throughout training.

**Scaling**:
- Start: 4x4 unmasked region in a 10x10 grid (simple belt paths)
- Intermediate: 6x6 unmasked, then 8x8
- Advanced: Full 10x10 unmasked
- Can also translate the unmasked region around the grid so the agent doesn't overfit to a specific position

**Prerequisites**: Mask channel (section 1) must be implemented first.

---

## Implementation Priority

| Order | Lesson | New Code Required | Depends On |
|-------|--------|-------------------|------------|
| 1 | Mask channel | New channel + step() guard | Nothing |
| 2 | Multi-source/multi-sink (crossing paths) | New lesson generator | Mask (for obstacles) |
| 3 | Inserter chains | New lesson generator | Nothing |
| 4 | Underground belt tunneling | New lesson generator | Mask (for gap obstacles) |
| 5 | Splitters | New entity + engine support | Nothing |
| 6 | Adaptive grid size | Curriculum logic | Mask |
