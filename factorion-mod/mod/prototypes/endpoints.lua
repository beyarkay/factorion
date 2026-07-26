-- Factorion endpoints are real one-tile transport belts. Runtime code keeps
-- source lanes supplied and removes the configured item at sink tiles.

local function tint_sprite_tree(value, tint)
  if type(value) ~= "table" then return end
  if value.filename or value.filenames or value.stripes then
    value.tint = tint
  end
  for _, child in pairs(value) do
    if type(child) == "table" then tint_sprite_tree(child, tint) end
  end
end

local function endpoint_belt(name, tint, icon, order)
  ---@diagnostic disable-next-line: undefined-field
  local belt = table.deepcopy(data.raw["transport-belt"]["transport-belt"])
  belt.name = name
  belt.localised_name = { "entity-name." .. name }
  belt.localised_description = { "entity-description." .. name }
  belt.minable = { mining_time = 0.1, result = name }
  belt.placeable_by = { item = name, count = 1 }
  belt.fast_replaceable_group = nil
  belt.next_upgrade = nil
  belt.related_underground_belt = nil
  belt.additional_pastable_entities = {
    "factorion-source-belt", "factorion-sink-belt",
  }

  tint_sprite_tree(belt.belt_animation_set, tint)
  tint_sprite_tree(belt.connector_frame_sprites, tint)

  ---@diagnostic disable-next-line: undefined-field
  local item = table.deepcopy(data.raw.item["transport-belt"])
  item.name = name
  item.icon = icon
  item.icon_size = 128
  item.icons = nil
  item.place_result = name
  item.order = order
  item.stack_size = 50
  return belt, item
end

local source_belt, source_item = endpoint_belt(
  "factorion-source-belt",
  { r = 0.2, g = 1.0, b = 0.25, a = 1 },
  "__factorion__/graphics/icons/source-tool.png",
  "z[factorion]-b[source-belt]"
)
local sink_belt, sink_item = endpoint_belt(
  "factorion-sink-belt",
  { r = 1.0, g = 0.32, b = 0.08, a = 1 },
  "__factorion__/graphics/icons/sink-tool.png",
  "z[factorion]-c[sink-belt]"
)

data:extend({
  source_belt, source_item, sink_belt, sink_item,
})
