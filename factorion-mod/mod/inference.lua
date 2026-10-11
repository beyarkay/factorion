-- Inference settings shared by every GPU server, and the window that edits
-- them. Clicking any GPU server opens the same global settings.

local M = {}

M.GPU_SERVER = "factorion-gpu-server"
local FRAME = "factorion-gpu-gui"
local SERVER_KW = 4.5

M.DEFAULTS = {
  checkpoint = "mvbipbr0",
  temperature = 1,
  top_k = 0,
  top_p = 1,
  stop_rule = "sample",
  stop_threshold = 0.5,
  max_steps = 0,
  seed = 0,
}

local STOP_RULES = {
  { "sample", "Sample it, as in training" },
  { "threshold", "Stop above the threshold" },
  { "never", "Never stop early" },
}

local NUMBER_FIELDS = {
  { key = "temperature", caption = "Temperature", min = 0, max = 2, decimal = true,
    tooltip = "0 always takes the most likely choice; 1 samples the model's own distribution, as in training." },
  { key = "top_k", caption = "Top-k", min = 0, max = 1000,
    tooltip = "Only sample from the k most likely choices for each part of an action. 0 turns it off." },
  { key = "top_p", caption = "Top-p", min = 0, max = 1, decimal = true,
    tooltip = "Only sample from the most likely choices that together hold this much probability. 1 turns it off." },
  { key = "stop_threshold", caption = "Stop threshold", min = 0, max = 1, decimal = true,
    tooltip = "With the threshold stop rule, stop once the model's done-probability exceeds this." },
  { key = "max_steps", caption = "Max steps", min = 0, max = 10000,
    tooltip = "Most placements per prediction. 0 uses the training episode length (grid width x height)." },
  { key = "seed", caption = "Seed", min = 0, max = 2147483647,
    tooltip = "0 draws a fresh random seed each prediction; any other value replays that seed exactly." },
}

-- Called with the new checkpoint when the player presses Load.
M.on_load = function(_) end

function M.config()
  storage.inference = storage.inference or {}
  for key, value in pairs(M.DEFAULTS) do
    if storage.inference[key] == nil then storage.inference[key] = value end
  end
  return storage.inference
end

-- A server's 10 kJ buffer drains within a few seconds once its power is cut.
function M.server_counts()
  local total, powered = 0, 0
  for _, surface in pairs(game.surfaces) do
    for _, server in pairs(surface.find_entities_filtered({ name = M.GPU_SERVER })) do
      total = total + 1
      if server.energy > 0 then powered = powered + 1 end
    end
  end
  return total, powered
end

-- nil when predictions may run, else why not.
function M.unavailable_reason()
  local total, powered = M.server_counts()
  if total == 0 then
    return "No GPU server on the map. Build a [item=" .. M.GPU_SERVER ..
      "] and connect it to power to run predictions."
  end
  if powered == 0 then
    return string.format("None of your %d [item=%s] has power. Connect one " ..
      "to your electric network to run predictions.", total, M.GPU_SERVER)
  end
  return nil
end

local function status_caption()
  local total, powered = M.server_counts()
  return string.format("%d GPU server(s), %d powered, drawing %.1f kW",
    total, powered, powered * SERVER_KW)
end

local function model_caption()
  local config = M.config()
  if storage.model and storage.model.checkpoint == config.checkpoint then
    return string.format("Loaded %s (%s), %dx%d grid", storage.model.checkpoint,
      tostring(storage.model.name), storage.model.grid_size,
      storage.model.grid_size)
  end
  if storage.model_error then return "Failed: " .. storage.model_error end
  return "Loading " .. config.checkpoint .. "…"
end

function M.refresh(player)
  local frame = player.gui.screen[FRAME]
  if not frame then return end
  frame.status.caption = status_caption()
  frame.settings.model_status.caption = model_caption()
end

function M.refresh_all()
  for _, player in pairs(game.connected_players) do M.refresh(player) end
end

function M.close(player)
  local frame = player.gui.screen[FRAME]
  if frame then frame.destroy() end
end

function M.open(player)
  M.close(player)
  local config = M.config()
  local frame = player.gui.screen.add({
    type = "frame", name = FRAME, caption = "GPU servers", direction = "vertical",
  })
  frame.auto_center = true
  frame.add({ type = "label", name = "status" })
  frame.add({ type = "line" })

  local settings = frame.add({ type = "table", name = "settings", column_count = 2 })
  settings.add({ type = "label", caption = "Checkpoint" })
  local row = settings.add({
    type = "flow", name = "checkpoint_row", direction = "horizontal",
  })
  row.add({
    type = "textfield", name = "checkpoint", text = config.checkpoint,
    tooltip = "W&B run id (e.g. mvbipbr0, or entity/project/id) or a local .pt path on the server's machine.",
    tags = { factorion_gpu = "checkpoint" },
  })
  row.add({ type = "button", caption = "Load", tags = { factorion_gpu = "load" } })
  settings.add({ type = "empty-widget" })
  settings.add({ type = "label", name = "model_status" })

  local function number_field(field)
    settings.add({ type = "label", caption = field.caption, tooltip = field.tooltip })
    settings.add({
      type = "textfield", text = tostring(config[field.key]), numeric = true,
      allow_decimal = field.decimal or false, allow_negative = false,
      tooltip = field.tooltip, tags = { factorion_gpu = field.key },
    })
  end
  for _, field in ipairs(NUMBER_FIELDS) do
    if field.key == "stop_threshold" then
      settings.add({ type = "label", caption = "Stop rule",
        tooltip = "How the model's \"I'm done\" head ends a prediction." })
      local items, selected = {}, 1
      for i, rule in ipairs(STOP_RULES) do
        items[i] = rule[2]
        if rule[1] == config.stop_rule then selected = i end
      end
      settings.add({
        type = "drop-down", items = items, selected_index = selected,
        tags = { factorion_gpu = "stop_rule" },
      })
    end
    number_field(field)
  end

  M.refresh(player)
  player.opened = frame
end

local function load_checkpoint(player)
  local frame = player.gui.screen[FRAME]
  local field = frame and frame.settings.checkpoint_row.checkpoint
  local checkpoint = field and field.text:match("^%s*(.-)%s*$") or ""
  if checkpoint == "" then return end
  M.config().checkpoint = checkpoint
  storage.model_error = nil
  M.on_load(checkpoint)
  M.refresh_all()
end

-- Each handler returns true when the event belonged to this window.
function M.on_gui_click(event)
  local tag = event.element.valid and event.element.tags.factorion_gpu
  if tag ~= "load" then return false end
  local player = game.get_player(event.player_index)
  if player then load_checkpoint(player) end
  return true
end

function M.on_gui_confirmed(event)
  local tag = event.element.valid and event.element.tags.factorion_gpu
  if tag ~= "checkpoint" then return false end
  local player = game.get_player(event.player_index)
  if player then load_checkpoint(player) end
  return true
end

function M.on_gui_text_changed(event)
  local tag = event.element.valid and event.element.tags.factorion_gpu
  if not tag then return false end
  for _, field in ipairs(NUMBER_FIELDS) do
    if field.key == tag then
      local value = tonumber(event.element.text)
      if value and value >= field.min and value <= field.max
          and (field.decimal or value == math.floor(value)) then
        M.config()[tag] = value
      end
    end
  end
  return true
end

function M.on_gui_selection_state_changed(event)
  local tag = event.element.valid and event.element.tags.factorion_gpu
  if tag ~= "stop_rule" then return false end
  M.config().stop_rule = STOP_RULES[event.element.selected_index][1]
  return true
end

function M.on_gui_closed(event)
  if not event.element or not event.element.valid
      or event.element.name ~= FRAME then
    return false
  end
  event.element.destroy()
  return true
end

return M
