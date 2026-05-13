-- Two selection tools the player uses to describe the prediction request:
--   factorion-footprint-tool  drag-select an N×N region (the buildable area)
--   factorion-marker-tool     left-click tiles to mark sources,
--                             right-click (alt-select) to mark sinks
--
-- Both are spawned into the player's inventory the first time they join.
-- See control.lua for the actual event wiring.

local empty_sprite = {
  filename = "__core__/graphics/empty.png",
  priority = "extra-high",
  width = 1,
  height = 1,
}

data:extend({
  {
    type = "selection-tool",
    name = "factorion-footprint-tool",
    icon = "__base__/graphics/icons/blueprint.png",
    icon_size = 64,
    flags = { "only-in-cursor", "spawnable", "not-stackable" },
    stack_size = 1,
    stackable = false,
    select = {
      border_color = { r = 0.2, g = 0.6, b = 1.0 },
      mode = { "any-tile" },
      cursor_box_type = "copy",
    },
    alt_select = {
      border_color = { r = 0.6, g = 0.2, b = 1.0 },
      mode = { "any-tile" },
      cursor_box_type = "copy",
    },
  },
  {
    type = "selection-tool",
    name = "factorion-marker-tool",
    icon = "__base__/graphics/icons/iron-plate.png",
    icon_size = 64,
    flags = { "only-in-cursor", "spawnable", "not-stackable" },
    stack_size = 1,
    stackable = false,
    -- Left-click: add sources. Right-click (alt): add sinks.
    select = {
      border_color = { r = 0.2, g = 1.0, b = 0.2 },
      mode = { "any-tile" },
      cursor_box_type = "entity",
    },
    alt_select = {
      border_color = { r = 1.0, g = 0.4, b = 0.2 },
      mode = { "any-tile" },
      cursor_box_type = "entity",
    },
  },
})
