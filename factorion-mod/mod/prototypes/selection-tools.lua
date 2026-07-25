-- The region brush stamps the model's fixed-size prediction footprint.
-- Source and sink endpoints are normal placeable belt items.

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
})
