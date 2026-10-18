# Electric Network

How electric poles join into networks, which tiles a network powers, and which
entities need that power.

**Source:** https://wiki.factorio.com/Electric_system (prototype values
cross-checked against `wube/factorio-data` 2.0.28 and 1.1.110,
`base/prototypes/entity/entities.lua`, and
https://lua-api.factorio.com/latest/prototypes/ElectricPolePrototype.html)

## Factorio Mechanics

### Poles and networks

An **electric network** is a set of electric poles joined by copper wires,
together with every generator and consumer inside any of those poles' supply
areas. Every consumer in a network shares the same supply. The poles
Factorion cares about:

| Pole | Size | Supply area | Wire reach | Doc |
|---|---|---|---|---|
| Small electric pole | 1x1 | 5x5 | 7.5 | [[small-electric-pole]] |
| Medium electric pole | 1x1 | 7x7 | 9 | [[medium-electric-pole]] |
| Substation | 2x2 | 18x18 | 18 | [[substation]] |
| Big electric pole | 2x2 | 4x4 | 32 (30 in 1.1) | Not in Factorion |

The big electric pole is a long-distance pole (tiny supply area, longest
reach). **Factorion does not model it.**
Source: https://wiki.factorio.com/Big_electric_pole

### Wire reach

> "Wire reach is the distance across which a pole can connect with another
> pole. If two different types of poles are connected, the smallest of either
> applies." — Electric system wiki page

So two poles A and B can be wired iff

```
dist(A, B) <= min(reach_A, reach_B)
```

Checked in a headless Factorio 2.0.77: the distance is Euclidean between
the two poles' centres and compared with `<=` (small poles 7 apart in a row
wire, 8 apart don't; (5, 5) on a diagonal wires, (6, 6) doesn't), a mixed
pair uses the smaller reach (small–medium at 8 doesn't wire), and each
supply area covers exactly the tiles below. The wiki sources that led there
follow.

The wiki page itself **does not say how the distance is measured**. The
prototype docs define `maximum_wire_distance` as "the maximum distance
between this pole and any other connected pole", and a Factorio developer
(Twinsen, forum thread "[0.17.55-0] Substation wire connection limit") says
"the wires use absolute distance and the electric coverage is a square".
Taken together: wire reach is the **Euclidean distance between the two
poles' centres** (a 1x1 pole's centre is its tile centre; a 2x2 pole's centre
is the corner shared by its four tiles), whereas the supply area is an
axis-aligned square. A medium pole (reach 9) can therefore sit 9 tiles from
another along a row (8 empty tiles between), but only 6 tiles diagonally
(`sqrt(6² + 6²) ≈ 8.49`; `sqrt(7² + 7²) ≈ 9.9` is out of reach).

### Connections per pole

- **Auto-connect.** A newly placed pole wires itself to poles in reach,
  closest first, skipping any pole that would close a triangle with two
  poles already connected to each other, and stops at **5** connections
  (`auto_connect_up_to_n_wires`, default 5). Shift-placing a pole places it
  with no connections.
- **Manual wiring.** The copper wire tool (shortcut bar since 2.0.7, free to
  use) adds or removes single connections; shift-clicking a pole removes all
  of its connections.
- **Limit.** In 1.1 a pole could hold at most **5** copper connections.
  Since 2.0.7 that cap is gone ("Electric poles are no longer limited to 5
  copper connections to other electric poles"); 5 is now only the auto-connect
  limit.

### Supply area

The supply area is an axis-aligned square **centred on the pole's centre**,
with half-width `supply_area_distance` (the tooltip's "supply area" is twice
that). Small pole: 2.5 → 5x5, i.e. the pole tile plus 2 tiles each side.
Medium: 3.5 → 7x7 (3 each side). Substation: 9 → 18x18 (8 tiles each side of
its 2x2 body).

> "A building is connected if one tile of the building is in a covered
> area." — Electric system wiki page

So a 3x3 [[assembling-machine]] is powered if any one of its nine tiles lies
in the supply area. (Confirmed in-game: an assembler with one corner tile in
the area crafts; shifted one tile out, it has no power.) Supply area and wire reach are independent: there is no
rule tying one to the other (Twinsen, same thread).

### What needs electricity

From the base-game prototypes (an entity needs power iff it has an
`energy_source` of type `electric`):

| Entity | Power? | Consumption (active) | Drain |
|---|---|---|---|
| [[transport-belt]] | No | — | — |
| [[underground-belt]] | No | — | — |
| [[splitter]] | No | — | — |
| [[inserter]] | Yes | 15.1 kW (13.6 kW before 2.0.7) | 400 W |
| Long-handed inserter | Yes | 21.4 kW | 400 W |
| [[assembling-machine]] 1 | Yes | 75 kW | 2.5 kW |
| Burner inserter | No — burns fuel | 144 kW (fuel) | — |

Inserter figures are the wiki infobox maxima; the prototype defines them as
5 kJ per movement and 5 kJ per rotation, so draw depends on swing speed. The
assembler's 2.5 kW drain is the electric-source default of `energy_usage /
30`. **Drain** is drawn whether or not the machine is working; **consumption**
only while it works, on top of the drain.
Sources: https://wiki.factorio.com/Inserter,
https://wiki.factorio.com/Long-handed_inserter,
https://wiki.factorio.com/Assembling_machine_1,
https://wiki.factorio.com/Burner_inserter

### Unpowered and underpowered machines

When demand exceeds supply, power is shared out in proportion to demand and
every machine on the network slows by the same factor (the wiki's example: a
210 kW + 90 kW load on 180 kW of solar runs both at 60% speed). A machine
outside every supply area, or on a network with no generation, gets nothing
and **stops**: an unpowered inserter does not swing and an unpowered
assembler does not craft.

## Factorion Implementation

- **Placeable poles:** [[small-electric-pole]], [[medium-electric-pole]],
  [[substation]]. No big electric pole.
- **No manual wiring.** Every pair of poles within `min(reach_A, reach_B)`
  (centre-to-centre) is connected; no connection cap.
- **Power comes from wherever the factory may not build.** A pole network is
  powered iff at least one of its poles could wire to a 1x1 pole on a tile
  just outside the grid or on an unbuildable tile joined to the grid's edge
  through unbuildable tiles (centre-to-centre distance within that pole's own
  reach): the area around a marked-out region is the player's powered base,
  but an enclosed unbuildable island inside it carries no power. There
  are no generators and no supply/demand balance — a network is either fully
  powered or not.
- **Powered tile:** a tile inside the supply area of any pole in a powered
  network.
- **Powered entity:** an inserter, long-handed inserter or assembler is
  powered if any of its tiles is a powered tile. An unpowered entity is
  treated as absent for throughput.
- **No power needed:** belts, underground belts, splitters, and the
  env-spawned source/sink markers.

## Interactions

| Entity | Interaction |
|---|---|
| [[inserter]] | Needs a powered tile; otherwise contributes no flow. |
| [[assembling-machine]] | Powered if any of its 3x3 tiles is a powered tile. |
| [[transport-belt]], [[underground-belt]], [[splitter]] | Unaffected by power. |
