# Medium Electric Pole

A 1x1 pole with a larger supply area and longer wire reach than the
[[small-electric-pole]]. See [[electric-network]] for how poles connect and
power entities.

**Source:** https://wiki.factorio.com/Medium_electric_pole (prototype values
cross-checked against `wube/factorio-data` 2.0.28 and 1.1.110)

## Factorio Mechanics

### Basics

| Property | Value |
|---|---|
| Size | 1x1 |
| Supply area | 7x7 (`supply_area_distance` 3.5) |
| Wire reach | 9 |
| Max wire connections | Auto-connects to at most 5; no hard cap since 2.0.7 (5 in 1.1) |
| Required technology | Electric energy distribution 1 |

### Recipe

| Version | Ingredients | Time | Output |
|---|---|---|---|
| 2.0 | 2 copper cable + 4 iron stick + 2 steel plate | 0.5 s | 1 |
| 1.1 | 2 copper plate + 4 iron stick + 2 steel plate | 0.5 s | 1 |

Total raw (2.0): 1 copper plate + 2 iron plate + 2 steel plate, 2 s.

### Geometry

The supply area is centred on the pole: the pole's tile plus 3 tiles on each
side. Two medium poles connect if their centres are at most 9 apart, so along
a row they can be up to 9 tiles apart (8 empty tiles between), and up to 6
tiles apart on a pure diagonal (`sqrt(6² + 6²) ≈ 8.49`). Wiring to a
[[small-electric-pole]] is limited to 7.5.

## Factorion Implementation

Placeable. Every pole within reach (centre-to-centre distance at most 9, or
7.5 to a small pole) is connected; there is no manual wiring and no
connection cap. Each of the 49 tiles in its 7x7 supply area is a powered tile
when the pole's network is powered (see [[electric-network]]).

## Interactions

| Entity | Interaction |
|---|---|
| [[small-electric-pole]] | Connects within 7.5 tiles. |
| [[medium-electric-pole]], [[substation]] | Connects within 9 tiles. |
| [[inserter]], [[assembling-machine]] | Powered if any of their tiles lie in the 7x7 supply area of a powered network. |
