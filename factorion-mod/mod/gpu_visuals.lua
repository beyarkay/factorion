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
  local activity = object(ids.activity)
  if not activity then
    activity = rendering.draw_light({
      sprite = "utility/light_medium", target = { entity = entity, offset = spec.activity_light },
      surface = entity.surface, scale = 0.55, intensity = 0.12,
      minimum_darkness = 0.3, color = { r = 1, g = 0.55, b = 0.16 },
    })
    ids.activity = activity.id
  end
  activity.target = { entity = entity, offset = spec.activity_light }
  activity.intensity = 0.08 + 0.28 * utilisation
  local pilot = object(ids.pilot)
  if not pilot then
    pilot = rendering.draw_light({
      sprite = "utility/light_small", target = { entity = entity, offset = spec.data_port },
      surface = entity.surface, scale = 0.35, intensity = 0.18,
      minimum_darkness = 0.3, color = { r = 0.12, g = 0.42, b = 1 },
    })
    ids.pilot = pilot.id
  end
  pilot.target = { entity = entity, offset = spec.data_port }
end

return M
