-- Blue concrete marking the tiles the model may build on. A placeable tile
-- gets Factorio's own resizable brush, and mining it restores the ground.

local NAME = "factorion-mask"
local blue = { r = 0.25, g = 0.55, b = 1.0, a = 1 }

---@diagnostic disable-next-line: undefined-field
local tile = table.deepcopy(data.raw.tile["refined-concrete"])
tile.name = NAME
tile.localised_name = { "item-name." .. NAME }
tile.minable = { mining_time = 0.1, result = NAME }
tile.tint = blue
tile.map_color = blue
tile.frozen_variant = nil
tile.thawed_variant = nil

---@diagnostic disable-next-line: undefined-field
local item = table.deepcopy(data.raw.item["refined-concrete"])
item.name = NAME
item.icons = { { icon = item.icon, icon_size = item.icon_size, tint = blue } }
item.icon = nil
item.place_as_tile.result = NAME
item.order = "z[factorion]-d[mask]"

data:extend({
  tile,
  item,
  {
    type = "recipe",
    name = NAME,
    enabled = true,
    energy_required = 0.5,
    ingredients = { { type = "item", name = "stone-brick", amount = 1 } },
    results = { { type = "item", name = NAME, amount = 10 } },
  },
})
