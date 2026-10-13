-- The planner: kept on the hotbar like the artillery remote; click a tile for
-- the sink, then one tile per ingredient source.

local NAME = "factorion-planner"
local color = { r = 0.15, g = 0.75, b = 1.0 }

data:extend({
  {
    type = "selection-tool",
    name = NAME,
    icon = "__factorion__/graphics/icons/region-tool.png",
    icon_size = 128,
    flags = { "not-stackable" },
    stack_size = 1,
    subgroup = "tool",
    order = "z[factorion]-a[planner]",
    select = {
      border_color = color,
      mode = { "any-tile" },
      cursor_box_type = "copy",
    },
    alt_select = {
      border_color = color,
      mode = { "any-tile" },
      cursor_box_type = "copy",
    },
  },
  {
    type = "recipe",
    name = NAME,
    enabled = true,
    energy_required = 1,
    ingredients = {
      { type = "item", name = "electronic-circuit", amount = 1 },
      { type = "item", name = "iron-plate", amount = 1 },
    },
    results = { { type = "item", name = NAME, amount = 1 } },
  },
})
