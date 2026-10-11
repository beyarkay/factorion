-- 4x2 GPU server. Draws a 4x H200 server's power: 4 x 700 W GPUs plus
-- ~1.7 kW of host.

local NAME = "factorion-gpu-server"
-- Space Age is an optional dependency, so it has loaded before this file
-- whenever it is enabled.
local processor = data.raw.item["quantum-processor"] and "quantum-processor"
  or "processing-unit"

-- The art spans 320 of its 384 px canvas along the machine; scale 0.4 makes
-- that exactly 4 tiles so servers tile end to end without overlapping.
local function sprite(direction)
  return {
    filename = "__factorion__/graphics/entity/gpu-server/" .. direction .. ".png",
    width = 384,
    height = 384,
    scale = 0.4,
    priority = "high",
  }
end

data:extend({
  {
    type = "electric-energy-interface",
    name = NAME,
    icon = "__factorion__/graphics/icons/gpu-server.png",
    icon_size = 128,
    flags = { "placeable-player", "player-creation" },
    minable = { mining_time = 0.5, result = NAME },
    max_health = 300,
    collision_box = { { -1.9, -0.9 }, { 1.9, 0.9 } },
    selection_box = { { -2, -1 }, { 2, 1 } },
    gui_mode = "all",
    energy_source = {
      type = "electric",
      usage_priority = "secondary-input",
      buffer_capacity = "10kJ",
    },
    energy_usage = "4.5kW",
    pictures = {
      north = sprite("north"),
      east = sprite("east"),
      south = sprite("south"),
      west = sprite("west"),
    },
  },
  {
    type = "item",
    name = NAME,
    icon = "__factorion__/graphics/icons/gpu-server.png",
    icon_size = 128,
    subgroup = "energy",
    order = "z[factorion]-a[gpu-server]",
    place_result = NAME,
    stack_size = 10,
  },
  {
    type = "recipe",
    name = NAME,
    enabled = true,
    energy_required = 30,
    ingredients = {
      { type = "item", name = "steel-plate", amount = 20 },
      { type = "item", name = "copper-cable", amount = 50 },
      { type = "item", name = processor, amount = 4 },
      { type = "item", name = "radar", amount = 1 },
    },
    results = { { type = "item", name = NAME, amount = 1 } },
  },
})
