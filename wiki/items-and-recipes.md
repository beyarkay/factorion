# Items and Recipes

The items and crafting recipes currently modeled in Factorion. Quantities
match the canonical Factorio wiki values exactly (per craft).

**Sources:**
- https://wiki.factorio.com/Copper_cable
- https://wiki.factorio.com/Electronic_circuit
- https://wiki.factorio.com/Iron_gear_wheel
- https://wiki.factorio.com/Transport_belt
- https://wiki.factorio.com/Assembling_machine_1
- https://wiki.factorio.com/Inserter

**Single source of truth:** items and recipes are defined in
`factorion_rs/src/types.rs` (`all_items()`, `all_recipes()`) and
exposed to Python via the `factorion_rs.py_items()` / `py_recipes()`
PyO3 bindings. To add or change an item or recipe, edit the Rust source
and rebuild the wheel — Python sees it automatically.

## Items

| Item | Enum value | Raw input? | Notes |
|---|---|---|---|
| Empty | `Item::Empty = 0` | — | Sentinel "no item" value |
| Copper Cable | `Item::CopperCable = 1` | No | Crafted from copper plates |
| Copper Plate | `Item::CopperPlate = 2` | Yes | Smelted from copper ore + coal in a stone furnace |
| Iron Plate | `Item::IronPlate = 3` | Yes | Smelted from iron ore + coal in a stone furnace |
| Electronic Circuit | `Item::ElectronicCircuit = 4` | No | Crafted from copper cable + iron plate |
| Iron Gear Wheel | `Item::IronGearWheel = 5` | No | Crafted from iron plate; gates several build recipes |
| Transport Belt | `Item::TransportBelt = 6` | No | The belt entity, as an inventory item |
| Inserter | `Item::Inserter = 7` | No | The inserter entity, as an inventory item |
| Assembling Machine 1 | `Item::AssemblingMachine1 = 8` | No | The assembler entity, as an inventory item |

Copper plate and iron plate remain `Source`-spawnable (so assembler
lessons can feed on them directly), but the smelting step (ore → plate)
is also modeled: see "Smelting" below.

**Item-vs-entity name overlap.** Three items share their string name with
an entity (`transport_belt`, `inserter`, `assembling_machine_1`). They
live in *separate* enums on purpose — `Item::Inserter` (= 7) and
`EntityKind::Inserter` (= 2) have different integer values. The `ITEMS`
and `ENTITIES` channels of the world tensor disambiguate at runtime.
This mirrors Factorio's data model where each placeable entity also
exists as an inventory item.

## Recipes

All recipes are 0.5s craft time on an `Assembling Machine 1`
(`crafting_speed = 0.5`). Throughput math (`transform_flow`) operates on
ratios, so absolute scaling is not material; we keep the wiki values
verbatim.

### Copper Cable

| Input | Output | Craft time |
|---|---|---|
| 1 copper plate | 2 copper cables | 0.5s |

### Electronic Circuit (Green Circuit)

| Inputs | Output | Craft time |
|---|---|---|
| 3 copper cable + 1 iron plate | 1 electronic circuit | 0.5s |

### Iron Gear Wheel

| Input | Output | Craft time |
|---|---|---|
| 2 iron plates | 1 iron gear wheel | 0.5s |

The simplest 1-in-1-out recipe in the table. Useful as the bottom rung
of any assembler-lesson curriculum.

### Transport Belt (build recipe)

| Inputs | Output | Craft time |
|---|---|---|
| 1 iron gear wheel + 1 iron plate | 2 transport belts | 0.5s |

This is the recipe to *craft* a transport belt as an inventory item, not
to operate one. It's modeled as a recipe so an assembler can produce
transport-belt items on a belt.

### Inserter (build recipe)

| Inputs | Output | Craft time |
|---|---|---|
| 1 electronic circuit + 1 iron gear wheel + 1 iron plate | 1 inserter | 0.5s |

### Assembling Machine 1 (build recipe)

| Inputs | Output | Craft time |
|---|---|---|
| 3 electronic circuits + 5 iron gear wheels + 9 iron plates | 1 assembling machine 1 | 0.5s |

The 3:5:9 ratio is awkward and unlikely to be the focus of a single
lesson; mostly recorded for completeness.

## Smelting

Smelting runs in the **stone furnace** (`Item::StoneFurnace`), a
placeable 2×2 crafting machine that behaves exactly like an assembler in
the throughput model (inserter-fed, recipe min-ratio flow, no
crafting-speed cap). The engine has no fuel/energy mechanic, so the
furnace's coal burn is folded into its recipes as an ordinary
ingredient at the true Base-game ratio: a stone furnace draws 90 kW and
coal holds 4 MJ, so a 3.2 s smelt burns `90e3 × 3.2 / 4e6 = 0.072` coal
(a 16 s steel craft burns 5×, i.e. 0.36).

| Inputs | Output | Craft time |
|---|---|---|
| 1 iron ore + 0.072 coal | 1 iron plate | 3.2s |
| 1 copper ore + 0.072 coal | 1 copper plate | 3.2s |
| 2 stone + 0.072 coal | 1 stone brick | 3.2s |
| 5 iron plates + 0.36 coal | 1 steel plate | 16s |

These recipes list `produced_by = [stone_furnace]` only — assemblers
can't smelt, and the furnace can't run assembler recipes.

## The Green Circuit Factory

The project's intermediate goal is a complete **green circuit production
line**:

1. **Copper plate** Source → [[transport-belt]] → [[inserter]] →
   [[assembling-machine]] (copper_cable recipe)
2. [[Inserter]] → [[transport-belt]] carrying copper cables
3. **Iron plate** Source → [[transport-belt]] → [[inserter]] →
   [[assembling-machine]] (electronic_circuit recipe)
4. Copper cable belt → [[inserter]] → same [[assembling-machine]]
5. [[Inserter]] → [[transport-belt]] → Sink consuming electronic circuits

This fits in an **11×11 grid** and exercises every entity currently
modeled.

## Operational Specs (entities)

Reference values from the wiki for entity behavior — not all are
currently used by Factorion's throughput model.

| Entity | Speed / craft speed | Other | Source |
|---|---|---|---|
| Transport belt | 15 i/s combined (2 lanes × 7.5 i/s) | tile speed 1.875 t/s, density 8/tile | [wiki](https://wiki.factorio.com/Transport_belt) |
| Inserter (basic) | ~0.83 i/s (not in infobox) | rotation 302°/s, energy 15.1 kW | [wiki](https://wiki.factorio.com/Inserter) |
| Assembling machine 1 | `0.5×` crafting speed | 0 module slots, 75 kW, 4 pollution/m, no fluid recipes | [wiki](https://wiki.factorio.com/Assembling_machine_1) |

The 0.5× crafting speed of AM1 is why the Rust `assembling_machine_1`
flow rate is `0.5` (`EntityKind::AssemblingMachine1.flow_rate()` in
`factorion_rs/src/types.rs`).
