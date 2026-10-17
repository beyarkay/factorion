# Small Electric Pole

The cheapest electric pole: a 1x1 pole with a short wire reach and a small
supply area. See [[electric-network]] for how poles connect and power
entities.

**Source:** https://wiki.factorio.com/Small_electric_pole (prototype values
cross-checked against `wube/factorio-data` 2.0.28 and 1.1.110)

## Factorio Mechanics

### Basics

| Property | Value |
|---|---|
| Size | 1x1 |
| Supply area | 5x5 (`supply_area_distance` 2.5) |
| Wire reach | 7.5 |
| Max wire connections | Auto-connects to at most 5; no hard cap since 2.0.7 (5 in 1.1) |
| Required technology | Electronics (2.0); available from the start in 1.1 |

### Recipe

| Version | Ingredients | Time | Output |
|---|---|---|---|
| 2.0 | 2 copper cable + 1 wood | 0.5 s | 2 small electric poles |
| 1.1 | Same | 0.5 s | 2 |

Total raw per craft (2 poles): 1 copper plate + 1 wood, 1 s.

### Geometry

The supply area is centred on the pole: the pole's tile plus 2 tiles on each
side. Two small poles connect if their centres are at most 7.5 apart, so
along a row they can be up to 7 tiles apart (6 empty tiles between), and up
to 5 tiles apart on a pure diagonal (`sqrt(5² + 5²) ≈ 7.07`). Wiring to a
pole of another type is limited to the smaller reach, which is always this
pole's 7.5.

## Factorion Implementation

Placeable. Every pole within reach (centre-to-centre distance at most 7.5,
the smaller of the two reaches) is connected; there is no manual wiring and no
connection cap. Each of the 25 tiles in its 5x5 supply area is a powered tile
when the pole's network is powered (see [[electric-network]]).

## Interactions

| Entity | Interaction |
|---|---|
| [[medium-electric-pole]], [[substation]] | Connects within 7.5 tiles. |
| [[inserter]], [[assembling-machine]] | Powered if any of their tiles lie in the 5x5 supply area of a powered network. |
