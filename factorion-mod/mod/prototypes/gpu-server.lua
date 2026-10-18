-- 4x2 GPU server. Mods can't add engine types, so it is a constant
-- combinator, the type that takes circuit wires and writes the server's state
-- signals, and nothing else. A hidden electric energy interface under it
-- draws a 4x H200 server's power, which script sets: 4 x 700 W GPUs plus
-- ~1.7 kW of host while predicting, and the host alone when idle.

local NAME = "factorion-gpu-server"
-- Space Age is an optional dependency, so it has loaded before this file
-- whenever it is enabled.
local processor = data.raw.item["quantum-processor"] and "quantum-processor"
  or "processing-unit"
local assembler = data.raw["assembling-machine"]["assembling-machine-2"]

-- Rendered in Blender by art/cable_fold.py and cut by art/pack_cable_fold.py, which
-- writes each orientation's sizes and offsets beside its pictures.
local art = {
  horizontal = require("graphics/entity/gpu-server-horizontal"),
  vertical = require("graphics/entity/gpu-server-vertical"),
}

local function picture(orientation, part)
  local spec = art[orientation][part]
  return {
    filename = spec.filename, width = spec.width, height = spec.height, shift = spec.shift,
    scale = 0.5, priority = "high", draw_as_shadow = part == "shadow",
  }
end

local function sprite(orientation)
  return {
    layers = {
      picture(orientation, "shadow"), picture(orientation, "main"),
    },
  }
end

local server = table.deepcopy(data.raw["constant-combinator"]["constant-combinator"])
server.name = NAME
server.icon = "__factorion__/graphics/icons/gpu-server.png"
server.icon_size = 64
server.flags = { "placeable-player", "player-creation", "not-upgradable", "hide-alt-info" }
server.minable = { mining_time = 0.5, result = NAME }
server.max_health = 300
server.collision_box = { { -1.9, -0.9 }, { 1.9, 0.9 } }
server.selection_box = { { -2, -1 }, { 2, 1 } }
server.fast_replaceable_group = nil
server.next_upgrade = nil
server.sprites = {
  north = sprite("horizontal"), east = sprite("vertical"),
  south = sprite("horizontal"), west = sprite("vertical"),
}
server.activity_led_sprites = nil
server.activity_led_light = nil
server.circuit_wire_connection_points = {
  art.horizontal.circuit, art.vertical.circuit, art.horizontal.circuit, art.vertical.circuit,
}
server.circuit_wire_max_distance = assembler.circuit_wire_max_distance
server.open_sound = assembler.open_sound
server.close_sound = assembler.close_sound

-- The power draw, one per orientation so poles reach it wherever they reach
-- the server.
local function power(orientation, box)
  return {
    type = "electric-energy-interface",
    name = "factorion-gpu-power-" .. orientation,
    localised_name = { "entity-name." .. NAME },
    icon = "__factorion__/graphics/icons/gpu-server.png",
    icon_size = 64,
    hidden = true,
    flags = { "not-on-map", "not-blueprintable", "not-deconstructable", "no-copy-paste", "placeable-off-grid" },
    selectable_in_game = false,
    collision_box = box,
    collision_mask = { layers = {} },
    energy_source = {
      type = "electric",
      usage_priority = "secondary-input",
      buffer_capacity = "150J",
      input_flow_limit = "5kW",
      output_flow_limit = "0W",
    },
    energy_usage = "900W",
    gui_mode = "none",
    allow_copy_paste = false,
  }
end

-- Signals the server writes to its circuit network: a GPU card with
-- vanilla's glyph for what it reports.
local function state_signal(name, glyph, order)
  return {
    type = "virtual-signal",
    name = "factorion-" .. name,
    icons = {
      { icon = "__factorion__/graphics/icons/gpu-glyph.png", icon_size = 64 },
      {
        icon = "__base__/graphics/icons/signal/" .. glyph .. ".png", icon_size = 64,
        scale = 0.3, shift = { 9, 9 },
      },
    },
    subgroup = "virtual-signal",
    order = "z[factorion]-" .. order,
  }
end

data:extend({
  server,
  power("horizontal", { { -1.9, -0.9 }, { 1.9, 0.9 } }),
  power("vertical", { { -0.9, -1.9 }, { 0.9, 1.9 } }),
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
  state_signal("predicting", "signal-hourglass", "a"),
  state_signal("utilisation", "signal-percent", "c"),
  state_signal("cluster", "signal-multiplication", "d"),
  state_signal("throughput", "signal-speed", "f"),
})

-- Runtime swaps only this small atlas frame as utilisation changes. The
-- masks contain the model's visible lamp lenses and share its exact anchor.
for orientation, spec in pairs(art) do
  for level = 0, spec.glow.frame_count - 1 do
    data:extend({ {
      type = "sprite", name = "factorion-gpu-" .. orientation .. "-glow-" .. level,
      filename = spec.glow.filename, width = spec.glow.width, height = spec.glow.height,
      x = level * spec.glow.width, shift = spec.glow.shift, scale = 0.5,
      draw_as_glow = true, priority = "high",
    } })
  end
end
