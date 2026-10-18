-- Rendered light masks share the body's Blender camera and exported anchors.
-- A blue pilot means powered; five amber segments show smoothed utilisation.
local M = {}
local art = {
  horizontal = require("graphics/entity/gpu-server-horizontal"),
  vertical = require("graphics/entity/gpu-server-vertical"),
}

function M.orientation(entity)
  return (entity.direction == defines.direction.east or entity.direction == defines.direction.west)
    and "vertical" or "horizontal"
end

function M.port(entity)
  return art[M.orientation(entity)].data_port
end

local function object(id)
  return id and rendering.get_object_by_id(id) or nil
end

function M.clear(state)
  for _, id in pairs(state.visuals or {}) do
    local obj = object(id)
    if obj then obj.destroy() end
  end
  state.visuals = nil
end

local function light(ids, key, entity, offset, spec)
  local obj = object(ids[key])
  if not obj then
    spec.target, spec.surface, spec.minimum_darkness = { entity = entity, offset = offset }, entity.surface, 0.3
    obj = rendering.draw_light(spec)
    ids[key] = obj.id
  end
  obj.target = { entity = entity, offset = offset }
  return obj
end

function M.update(state, powered, utilisation, busy)
  local entity = state.entity
  if not entity.valid or not powered then M.clear(state); return end
  local orientation = M.orientation(entity)
  local spec = art[orientation]
  local level = math.min(5, math.max(busy and 1 or 0, math.floor(utilisation * 5 + 0.5)))
  local sprite = "factorion-gpu-" .. orientation .. "-glow-" .. level
  state.visuals = state.visuals or {}
  local ids = state.visuals
  local glow = object(ids.glow)
  if not glow then
    glow = rendering.draw_sprite({
      sprite = sprite, target = entity, surface = entity.surface, render_layer = "object",
    })
    ids.glow = glow.id
  elseif glow.sprite ~= sprite then
    glow.sprite = sprite
  end
  light(ids, "activity", entity, spec.activity_light, {
    sprite = "utility/light_medium", scale = 0.55, color = { r = 1, g = 0.55, b = 0.16 },
  }).intensity = 0.08 + 0.28 * utilisation
  light(ids, "pilot", entity, spec.data_port, {
    sprite = "utility/light_small", scale = 0.35, intensity = 0.18, color = { r = 0.12, g = 0.42, b = 1 },
  })
end

return M
