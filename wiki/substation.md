# Substation

A 2x2 pole with the largest supply area of any pole and a long wire reach.
See [[electric-network]] for how poles connect and power entities.

**Source:** https://wiki.factorio.com/Substation (prototype values
cross-checked against `wube/factorio-data` 2.0.28 and 1.1.110)

## Factorio Mechanics

### Basics

| Property | Value |
|---|---|
| Size | 2x2 |
| Supply area | 18x18 (`supply_area_distance` 9) |
| Wire reach | 18 |
| Max wire connections | Auto-connects to at most 5; no hard cap since 2.0.7 (5 in 1.1) |
| Required technology | Electric energy distribution 2 |

Supply area and wire reach were 16x16 and 16 before 0.15.0.

### Recipe

| Version | Ingredients | Time | Output |
|---|---|---|---|
| 2.0 | 5 advanced circuit + 6 copper cable + 10 steel plate | 0.5 s | 1 |
| 1.1 | 5 advanced circuit + 5 copper plate + 10 steel plate | 0.5 s | 1 |

Total raw (2.0): 28 copper plate + 10 iron plate + 10 plastic bar + 10 steel
plate, 49.5 s.

### Geometry

The substation's centre is the corner shared by its four tiles, so its 18x18
supply area extends 8 tiles beyond its body on every side. Two substations
connect if their centres are at most 18 apart: along a row, up to 18 tiles
apart (16 empty tiles between), which is exactly where their supply areas
abut. Wiring to a [[small-electric-pole]] is limited to 7.5 and to a
[[medium-electric-pole]] to 9, measured from the substation's corner centre
to the other pole's tile centre.

## Factorion Implementation

Placeable as a 2x2 multi-tile entity. Every pole within reach (centre-to-centre
distance at most the smaller of the two reaches) is connected; there is no
manual wiring and no connection cap. Each tile in its 18x18 supply area (all
of it that lies on the grid) is a powered tile when the pole's network is
powered (see [[electric-network]]).

## Interactions

| Entity | Interaction |
|---|---|
| [[small-electric-pole]] | Connects within 7.5 tiles. |
| [[medium-electric-pole]] | Connects within 9 tiles. |
| [[substation]] | Connects within 18 tiles. |
| [[inserter]], [[assembling-machine]] | Powered if any of their tiles lie in the 18x18 supply area of a powered network. |
