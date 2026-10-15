-- Warning icons the runtime blinks over a source belt with nothing arriving,
-- a GPU server that can't reach its model, and a painted area or GPU server
-- with nothing linked to it.

local function alert(name)
  return {
    type = "sprite",
    name = "factorion-alert-" .. name,
    filename = "__factorion__/graphics/alerts/" .. name .. ".png",
    size = 64,
    scale = 0.5,
    flags = { "icon" },
  }
end

data:extend({ alert("source-empty"), alert("gpu-offline"), alert("unlinked") })
