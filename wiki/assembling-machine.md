# Assembling Machine 1

A 3x3 crafting machine that converts input items into output items according to
a set recipe. The core production entity in any factory.

**Source:** https://wiki.factorio.com/Assembling_machine_1

## Factorio Mechanics

### Basics

- **Size:** 3x3 tiles
- **Research required:** None (available from start)
- **Power consumption:** 75 kW active, 2.5 kW drain
- **Crafting speed:** 0.5 (base multiplier applied to recipe craft time)

### How Crafting Works

1. An [[assembling-machine]] is assigned a **recipe** (see [[items-and-recipes]])
2. [[Inserter]]s feed the required input items into the machine
3. The machine crafts at its crafting speed (0.5x for AM1)
4. [[Inserter]]s extract the produced items

The crafting speed multiplier means AM1 takes **twice as long** as the recipe's
base craft time. For example, a recipe with 0.5s base time takes 1.0s on AM1.

### Item I/O

Assembling machines **cannot** connect directly to [[transport-belt]]s. Items
must always be transferred by [[inserter]]s placed on the machine's
[[glossary#perimeter]].

Inserters can interact with any non-corner perimeter tile of the 3x3 body.

### Limitations (AM1 specifically)

- **No fluid recipes:** Cannot craft recipes that require fluids as input/output
- **No module slots:** Cannot accept speed, efficiency, or productivity modules
- **Slowest tier:** 0.5x crafting speed vs 0.75x (AM2) and 1.25x (AM3)

### Tiers

| Tier | Crafting speed | Module slots | Fluids? |
|---|---|---|---|
| Assembling Machine 1 | 0.5 | 0 | No |
| Assembling Machine 2 | 0.75 | 2 | Yes |
| Assembling Machine 3 | 1.25 | 4 | Yes |

> **Only AM1 in Factorion.** Higher tiers, modules, and fluid recipes are not
> modeled.

## Factorion Implementation

### Enum & Channel Values

- **Entity enum:** `EntityKind::AssemblingMachine1 = 3`
- **Recipe:** Stored in the `Items` channel — the `Item` enum value indicates
  which recipe the machine is set to (e.g., `Item::ElectronicCircuit = 4`)
- **Flow rate:** 0.5 items/sec (base crafting speed)
- **Anchor:** Top-left corner of the 3x3 footprint

### Connection Logic

The assembling machine searches its **perimeter** (the ring of tiles around the
3x3 body, excluding corners) for [[inserter]]-type entities (Inserter, Source,
or Sink):

- If the inserter **faces away** from the machine body → the machine outputs
  to the inserter (machine → inserter)
- If the inserter **faces toward** (or along) the machine body → the inserter
  feeds into the machine (inserter → machine)

Specifically, "faces away" means:
- Inserter faces North and is above the machine (`ddy < 0`)
- Inserter faces South and is below the machine (`ddy > 0`)
- Inserter faces West and is left of the machine (`ddx < 0`)
- Inserter faces East and is right of the machine (`ddx > 0`)

### Transform Flow

The `transform_flow` function implements recipe crafting:

1. Look up the recipe for the assigned item
2. Find the **minimum ratio** of available input to required input across all
   ingredients
3. Scale output production by that ratio

Example: Electronic circuit needs 6 copper cable + 2 iron plate → 2 circuits.
If only 3 copper cable is available, ratio = min(3/6, ∞) = 0.5, output = 1
circuit.

### Simplifications vs Real Factorio

| Mechanic | Real Factorio | Factorion |
|---|---|---|
| Crafting time | Discrete per-item crafting | Continuous flow rate |
| Internal buffer | Stores ingredients and products | No buffer — instant flow |
| Power | Requires electricity | No power simulation |
| Tiers | AM1/AM2/AM3 | AM1 only |
| Modules | Speed/efficiency/productivity | Not modeled |
| Fluid recipes | AM2+ can use fluids | Not modeled |

## Interactions

| Entity | Interaction |
|---|---|
| [[inserter]] | The **only** way to get items in/out. Must be on the perimeter, direction determines flow. |
| [[transport-belt]] | No direct connection — always needs an [[inserter]] between them. |
| Source / Sink | Can interact directly (they use inserter connection logic). Source feeds in, Sink extracts. |
