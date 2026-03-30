# Items and Recipes

The items and crafting recipes currently modeled in Factorion.

**Sources:**
- https://wiki.factorio.com/Copper_cable
- https://wiki.factorio.com/Electronic_circuit

## Items

| Item | Enum value | Raw material? | Notes |
|---|---|---|---|
| Copper Plate | `Item::CopperPlate = 2` | Yes | Smelted from copper ore (smelting not modeled) |
| Iron Plate | `Item::IronPlate = 3` | Yes | Smelted from iron ore (smelting not modeled) |
| Copper Cable | `Item::CopperCable = 1` | No | Crafted from copper plates |
| Electronic Circuit | `Item::ElectronicCircuit = 4` | No | Crafted from copper cable + iron plates |

Copper plate and iron plate are treated as **raw inputs** in Factorion — they
appear from Source entities. The smelting step (ore → plate) is not modeled.

## Recipes

### Copper Cable

| | Real Factorio | Factorion |
|---|---|---|
| **Input** | 1 copper plate | 2 copper plates |
| **Output** | 2 copper cables | 4 copper cables |
| **Craft time** | 0.5s | Not modeled (continuous flow) |
| **Ratio** | 1:2 | 1:2 (same) |

The Factorion recipe doubles both input and output quantities but preserves the
same ratio. This may be to make flow rates work out to nice numbers in the
throughput graph.

**In code** (`types.rs`):
```rust
Item::CopperCable => Recipe {
    consumes: { CopperPlate: 2.0 },
    produces: { CopperCable: 4.0 },
}
```

### Electronic Circuit (Green Circuit)

| | Real Factorio | Factorion |
|---|---|---|
| **Input** | 3 copper cable + 1 iron plate | 6 copper cable + 2 iron plates |
| **Output** | 1 electronic circuit | 2 electronic circuits |
| **Craft time** | 0.5s | Not modeled (continuous flow) |
| **Ratio** | (3:1):1 | (3:1):1 (same) |

Again, quantities are doubled but ratios are preserved.

**In code** (`types.rs`):
```rust
Item::ElectronicCircuit => Recipe {
    consumes: { CopperCable: 6.0, IronPlate: 2.0 },
    produces: { ElectronicCircuit: 2.0 },
}
```

### Why the doubled quantities?

The recipes in Factorion use doubled quantities compared to real Factorio. The
ratios are identical, so the game mechanics are equivalent. The doubling likely
simplifies flow rate calculations — with an [[assembling-machine]] crafting
speed of 0.5, the effective output rates become whole numbers.

## The Green Circuit Factory

The project's intermediate goal is building a complete **green circuit
(electronic circuit) production line**. This requires:

1. **Copper plate** Source → [[transport-belt]] → [[inserter]] →
   [[assembling-machine]] (copper cable recipe)
2. [[Inserter]] → [[transport-belt]] carrying copper cables
3. **Iron plate** Source → [[transport-belt]] → [[inserter]] →
   [[assembling-machine]] (electronic circuit recipe)
4. Copper cable belt → [[inserter]] → same [[assembling-machine]]
5. [[Inserter]] → [[transport-belt]] → Sink consuming electronic circuits

This fits in an **11x11 grid** and requires all currently modeled entities
working together.

## Items Not Yet Modeled

The real Factorio has hundreds of items. Items that might be added as the
project grows:

- **Iron gear wheel** — intermediate used in many recipes
- **Science packs** — the ultimate goal of most factories
- **Ores** — if mining is ever modeled (copper ore, iron ore)
