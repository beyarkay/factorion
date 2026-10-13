-- Paint and eraser for the tiles the model may build on. They are tiles only
-- to get Factorio's resizable brush: control.lua puts the ground back the
-- moment one is placed and keeps the paint as a drawn overlay instead.

local function paint(name, color, order)
  ---@diagnostic disable-next-line: undefined-field
  local tile = table.deepcopy(data.raw.tile["refined-concrete"])
  tile.name = name
  tile.localised_name = { "item-name." .. name }
  tile.minable = { mining_time = 0.1, result = name }
  tile.tint = color
  tile.map_color = color
  tile.frozen_variant = nil
  tile.thawed_variant = nil

  ---@diagnostic disable-next-line: undefined-field
  local item = table.deepcopy(data.raw.item["refined-concrete"])
  item.name = name
  item.icons = { { icon = item.icon, icon_size = item.icon_size, tint = color } }
  item.icon = nil
  item.place_as_tile.result = name
  item.order = "z[factorion]-" .. order

  -- Placing never uses the paint up, so one stack lasts forever.
  data:extend({
    tile,
    item,
    {
      type = "recipe",
      name = name,
      enabled = true,
      energy_required = 0.5,
      ingredients = {},
      results = { { type = "item", name = name, amount = item.stack_size } },
    },
  })
end

paint("factorion-mask", { r = 0.25, g = 0.55, b = 1.0, a = 1 }, "d[mask]")
paint("factorion-mask-eraser", { r = 1.0, g = 0.3, b = 0.25, a = 1 }, "e[mask-eraser]")
