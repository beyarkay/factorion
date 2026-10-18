-- The parity harness's power from off the map: a pole on every tile just
-- outside a factory's grid. Its wire reaches as far as the longest-reaching
-- pole a factory can hold (the substation's 18), so a factory pole wires to
-- it exactly when the engine says that pole reaches off the map, and its
-- supply area is only its own tile, so it powers nothing on the grid.

---@diagnostic disable-next-line: undefined-field
local pole = table.deepcopy(data.raw["electric-pole"]["medium-electric-pole"])
pole.name = "factorion-offmap-pole"
pole.localised_name = "Factorion off-map pole"
pole.minable = nil
pole.fast_replaceable_group = nil
pole.next_upgrade = nil
pole.supply_area_distance = 0.5
pole.maximum_wire_distance = 18

data:extend({ pole })
