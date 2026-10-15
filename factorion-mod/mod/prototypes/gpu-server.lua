-- 4x2 GPU server. Draws a 4x H200 server's power: 4 x 700 W GPUs plus
-- ~1.7 kW of host at full load, and the host alone when idle. Mods can't add
-- engine types, so it borrows the assembling machine's: that type takes
-- circuit wires and has a working state. Everything else an assembler does
-- (recipes, modules, beacons, inserters, upgrades) is switched off. Script
-- hands it one forward-pass token per craft while its cluster predicts, and
-- it idles on its drain otherwise.

local NAME = "factorion-gpu-server"
local FORWARD_PASS = "factorion-forward-pass"
-- Space Age is an optional dependency, so it has loaded before this file
-- whenever it is enabled.
local processor = data.raw.item["quantum-processor"] and "quantum-processor"
  or "processing-unit"
local assembler = data.raw["assembling-machine"]["assembling-machine-2"]

-- Purple placeholder blocks exactly 4x2 (or 2x4) tiles at half scale.
local function placeholder(orientation)
  return {
    filename = "__factorion__/graphics/entity/gpu-server-" .. orientation .. ".png",
    width = 384,
    height = 384,
    scale = 0.5,
    priority = "high",
  }
end

-- On the I/O box at the left end (horizontal) or bottom end (vertical).
local function connector(x, y)
  return {
    variation = 18,
    main_offset = util.by_pixel(x, y),
    shadow_offset = util.by_pixel(x + 11, y + 6),
    show_shadow = true,
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

local signals = table.deepcopy(data.raw["constant-combinator"]["constant-combinator"])
signals.name = "factorion-gpu-signals"
signals.hidden = true
signals.flags = { "not-on-map", "not-blueprintable", "not-deconstructable", "placeable-off-grid" }
signals.selectable_in_game = false
signals.minable = nil
signals.collision_box = { { 0, 0 }, { 0, 0 } }
signals.collision_mask = { layers = {} }
signals.selection_box = nil
signals.sprites = nil
signals.activity_led_sprites = nil
signals.activity_led_light = nil
signals.draw_circuit_wires = false
signals.created_smoke = nil

data:extend({
  { type = "recipe-category", name = "factorion-compute" },
  {
    type = "item",
    name = FORWARD_PASS,
    icon = "__factorion__/graphics/icons/gpu-glyph.png",
    icon_size = 64,
    hidden = true,
    stack_size = 100,
  },
  {
    type = "recipe",
    name = FORWARD_PASS,
    icon = "__factorion__/graphics/icons/gpu-glyph.png",
    icon_size = 64,
    category = "factorion-compute",
    hidden = true,
    enabled = true,
    energy_required = 1,
    ingredients = { { type = "item", name = FORWARD_PASS, amount = 1 } },
    results = {},
  },
  {
    type = "assembling-machine",
    name = NAME,
    icon = "__factorion__/graphics/icons/gpu-server.png",
    icon_size = 64,
    flags = {
      "placeable-player", "player-creation", "not-upgradable",
      "no-automated-item-insertion", "no-automated-item-removal",
    },
    minable = { mining_time = 0.5, result = NAME },
    max_health = 300,
    collision_box = { { -1.9, -0.9 }, { 1.9, 0.9 } },
    selection_box = { { -2, -1 }, { 2, 1 } },
    crafting_categories = { "factorion-compute" },
    fixed_recipe = FORWARD_PASS,
    crafting_speed = 1,
    show_recipe_icon = false,
    show_recipe_icon_on_map = false,
    module_slots = 0,
    allowed_effects = {},
    effect_receiver = { uses_module_effects = false, uses_beacon_effects = false, uses_surface_effects = false },
    energy_source = {
      type = "electric",
      usage_priority = "secondary-input",
      drain = "900W",
    },
    energy_usage = "3.6kW",
    circuit_wire_max_distance = assembler.circuit_wire_max_distance,
    circuit_connector = circuit_connector_definitions.create_vector(universal_connector_template, {
      connector(-50, 12), connector(-12, 46), connector(-50, 12), connector(-12, 46),
    }),
    alert_icon_shift = util.by_pixel(0, -12),
    graphics_set = {
      animation = {
        north = placeholder("horizontal"),
        east = placeholder("vertical"),
        south = placeholder("horizontal"),
        west = placeholder("vertical"),
      },
    },
    open_sound = assembler.open_sound,
    close_sound = assembler.close_sound,
    working_sound = data.raw.lab.lab.working_sound,
    impact_category = "metal",
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
  signals,
  state_signal("predicting", "signal-hourglass", "a"),
  state_signal("ready", "signal-checked-green", "b"),
  state_signal("utilisation", "signal-percent", "c"),
  state_signal("cluster", "signal-multiplication", "d"),
  state_signal("step", "signal-number-sign", "e"),
  state_signal("throughput", "signal-speed", "f"),
})
