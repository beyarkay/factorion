-- Three focused tools: a fixed-size region brush plus distinct source/sink
-- placers. Runtime behavior and item-picker dialogs live in control.lua.

local function selection(name, icon, color, order)
  return {
    type = "selection-tool",
    name = name,
    icon = icon,
    icon_size = 128,
    flags = { "only-in-cursor", "spawnable", "not-stackable" },
    stack_size = 1,
    stackable = false,
    subgroup = "tool",
    order = order,
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
  }
end

data:extend({
  selection(
    "factorion-footprint-tool",
    "__factorion__/graphics/icons/region-tool.png",
    { r = 0.15, g = 0.75, b = 1.0 },
    "z[factorion]-a[region]"
  ),
  selection(
    "factorion-source-tool",
    "__factorion__/graphics/icons/source-tool.png",
    { r = 0.2, g = 1.0, b = 0.25 },
    "z[factorion]-b[source]"
  ),
  selection(
    "factorion-sink-tool",
    "__factorion__/graphics/icons/sink-tool.png",
    { r = 1.0, g = 0.35, b = 0.1 },
    "z[factorion]-c[sink]"
  ),
})
