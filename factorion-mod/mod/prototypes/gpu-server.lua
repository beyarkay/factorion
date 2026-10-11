-- Placeholder 4x2 GPU server (purple block until the real art lands). Draws a
-- 4x H200 server's power: 4 x 700 W GPUs plus ~1.7 kW of host.

local NAME = "factorion-gpu-server"
-- Space Age is an optional dependency, so it has loaded before this file
-- whenever it is enabled.
local processor = data.raw.item["quantum-processor"] and "quantum-processor"
  or "processing-unit"

data:extend({
  {
    type = "electric-energy-interface",
    name = NAME,
    icon = "__factorion__/graphics/icons/gpu-server.png",
    icon_size = 64,
    flags = { "placeable-player", "player-creation" },
    minable = { mining_time = 0.5, result = NAME },
    max_health = 300,
    collision_box = { { -1.9, -0.9 }, { 1.9, 0.9 } },
    selection_box = { { -2, -1 }, { 2, 1 } },
    gui_mode = "none",
    energy_source = {
      type = "electric",
      usage_priority = "secondary-input",
      buffer_capacity = "10kJ",
    },
    energy_usage = "4.5kW",
    picture = {
      filename = "__factorion__/graphics/entity/gpu-server.png",
      width = 256,
      height = 128,
      scale = 0.5,
      priority = "high",
    },
  },
  {
    type = "item",
    name = NAME,
    icon = "__factorion__/graphics/icons/gpu-server.png",
    icon_size = 64,
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
