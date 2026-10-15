-- Inference settings shared by every GPU server, and the window that edits
-- them, shows the clicked server's state and builds its painted area.

local gui = require("gui")

local M = {}

M.GPU_SERVER = "factorion-gpu-server"
local FRAME = "factorion-gpu-gui"

M.DEFAULTS = {
  checkpoint = "3y1q7k0v",
  temperature = 1,
  steps = 0,
  refine = 0,
  best_of = 1,
  seed = 0,
}

local NUMBER_FIELDS = {
  { key = "temperature", caption = "Temperature", min = 0, max = 2, decimal = true,
    tooltip = "0 always takes the most likely factory at each denoising step; 1 samples the model's own distribution, as in training." },
  { key = "steps", caption = "Denoising steps", min = 0, max = 100,
    tooltip = "Steps from pure noise to a factory. 0 uses the checkpoint's training schedule." },
  { key = "refine", caption = "Refinement steps", min = 0, max = 100,
    tooltip = "Most extra greedy steps after denoising; they end early once the model leaves its own factory unchanged. 0 uses the checkpoint's training schedule." },
  { key = "best_of", caption = "Best of", min = 1, max = 64,
    tooltip = "Factories denoised side by side; the one with the highest simulated throughput is built, step by step." },
  { key = "seed", caption = "Seed", min = 0, max = 2147483647,
    tooltip = "0 draws a fresh random seed each prediction; any other value replays that seed exactly." },
}

local STATE_SIGNALS = "[virtual-signal=factorion-predicting] 1 while predicting, " ..
  "[virtual-signal=factorion-ready] 1 when a prediction could start, " ..
  "[virtual-signal=factorion-utilisation] utilisation in percent, " ..
  "[virtual-signal=factorion-cluster] powered GPU servers in the cluster, " ..
  "[virtual-signal=factorion-step] denoising steps placed so far, " ..
  "[virtual-signal=factorion-throughput] the sinks' throughput in items per minute."

-- Set by control.lua: on_load gets the new checkpoint when the player presses
-- Load; describe, on_predict and on_clear get the window's GPU server and
-- return a message (describe also returns whether Predict can run and its
-- cluster); info returns the server's live state; settings its circuit
-- settings.
M.on_load = function(_) end
M.describe = function(_) return "", false, "" end
M.on_predict = function(_, _) return "" end
M.on_clear = function(_, _) return "" end
M.info = function(_)
  return { status = "idle", caption = "", utilisation = 0, power_kw = 0, wired = false }
end
M.settings = function(_) return { conditions = {} } end

function M.config()
  storage.inference = storage.inference or {}
  for key, value in pairs(M.DEFAULTS) do
    if storage.inference[key] == nil then storage.inference[key] = value end
  end
  return storage.inference
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

local function window_server(player)
  local server = storage.gpu_windows and storage.gpu_windows[player.index]
  return server and server.valid and server or nil
end

-- The live readouts every few ticks; with full, also the factory the server
-- would build, which floods the paint.
function M.refresh(player, full)
  local frame = player.gui.screen[FRAME]
  if not frame then return end
  local server = window_server(player)
  if not server then
    frame.destroy()
    return
  end
  local main, circuit = frame.body.main, frame.body.circuit
  local info = M.info(server)
  gui.set_status(main, info.status, info.caption)
  main.utilisation.value = info.utilisation
  main.utilisation_label.caption = string.format("Utilisation %d%%, drawing %.1f kW",
    math.floor(info.utilisation * 100 + 0.5), info.power_kw)
  circuit.unwired.visible = not info.wired
  if not full then return end
  circuit.settings.model_status.caption = model_caption()
  local caption, can_predict, cluster = M.describe(server)
  main.region.caption = caption
  main.cluster.caption = cluster
  main.action_row.predict_button.enabled = can_predict
  main.action_row.clear_button.enabled = server.energy > 0
end

function M.refresh_all(full)
  for _, player in pairs(game.connected_players) do M.refresh(player, full) end
end

function M.close(player)
  local frame = player.gui.screen[FRAME]
  if frame then frame.destroy() end
end

function M.open(player, server)
  M.close(player)
  storage.gpu_windows = storage.gpu_windows or {}
  storage.gpu_windows[player.index] = server
  local config = M.config()
  local settings = M.settings(server)
  local frame, main = gui.entity_window(player, FRAME, "GPU server", server)

  local bar = main.add({ type = "progressbar", name = "utilisation", style = "production_progressbar" })
  bar.style.horizontally_stretchable = true
  main.add({ type = "label", name = "utilisation_label" })
  local cluster = main.add({ type = "label", name = "cluster" })
  cluster.style.single_line = false
  cluster.style.maximal_width = 360

  gui.heading(main, "Factory")
  local region = main.add({ type = "label", name = "region" })
  region.style.single_line = false
  region.style.maximal_width = 360
  local actions = main.add({ type = "flow", name = "action_row", direction = "horizontal" })
  actions.add({
    type = "button", name = "predict_button", caption = "Predict", style = "green_button",
    tooltip = "Build a factory on the painted area touching this server with " ..
      "every GPU server sharing it, starting from what they built before.",
    tags = { factorion_gpu = "predict" },
  })
  actions.add({
    type = "button", name = "clear_button", caption = "Clear",
    tooltip = "Stop this cluster's prediction and remove what it built.",
    tags = { factorion_gpu = "clear" },
  })

  local circuit = gui.panel(frame.body, "circuit")
  circuit.add({ type = "label", caption = "Circuit network", style = "caption_label" })
  local unwired = circuit.add({
    type = "label", name = "unwired",
    caption = "Connect a red or green wire to this server to use these.",
  })
  unwired.style.font_color = { 1, 0.75, 0.3 }
  gui.condition_row(circuit, "predict", "Predict when", settings.conditions.predict or {})
  gui.condition_row(circuit, "deconstruct", "Deconstruct region when",
    settings.conditions.deconstruct or {})
  circuit.add({
    type = "checkbox", caption = "Read state", state = settings.read_state ~= false,
    tooltip = STATE_SIGNALS, tags = { factorion_gpu = "read_state" },
  })

  gui.heading(circuit, "Model, shared by every GPU server")
  local table_ = circuit.add({ type = "table", name = "settings", column_count = 2 })
  table_.add({ type = "label", caption = "Checkpoint" })
  local row = table_.add({ type = "flow", name = "checkpoint_row", direction = "horizontal" })
  row.add({
    type = "textfield", name = "checkpoint", text = config.checkpoint,
    tooltip = "W&B run id of a denoiser SFT run (e.g. dr17bmik, or entity/project/id) or a local .pt path on the server's machine.",
    tags = { factorion_gpu = "checkpoint" },
  })
  row.add({ type = "button", caption = "Load", tags = { factorion_gpu = "load" } })
  table_.add({ type = "empty-widget" })
  table_.add({ type = "label", name = "model_status" })
  for _, field in ipairs(NUMBER_FIELDS) do
    table_.add({ type = "label", caption = field.caption, tooltip = field.tooltip })
    table_.add({
      type = "textfield", text = tostring(config[field.key]), numeric = true,
      allow_decimal = field.decimal or false, allow_negative = false,
      tooltip = field.tooltip, tags = { factorion_gpu = field.key },
    })
  end

  M.refresh(player, true)
  player.opened = frame
end

local function load_checkpoint(player)
  local frame = player.gui.screen[FRAME]
  local field = frame and frame.body.circuit.settings.checkpoint_row.checkpoint
  local checkpoint = field and field.text:match("^%s*(.-)%s*$") or ""
  if checkpoint == "" then return end
  M.config().checkpoint = checkpoint
  storage.model_error = nil
  M.on_load(checkpoint)
  M.refresh_all(true)
end

-- Saves the condition row the edited element belongs to.
local function condition_edited(event)
  local key = event.element.valid and event.element.tags.factorion_condition
  local player = game.get_player(event.player_index)
  local server = player and window_server(player)
  if not (key and server) then return false end
  local settings = M.settings(server)
  settings.conditions[key] = gui.read_condition_row(event.element.parent)
  settings.circuit_player = player.index
  return true
end

-- Each handler returns true when the event belonged to this window.
function M.on_gui_click(event)
  local tag = event.element.valid and event.element.tags.factorion_gpu
  local player = game.get_player(event.player_index)
  if not player then return false end
  if tag == "load" then
    load_checkpoint(player)
    return true
  end
  local action = (tag == "predict" and M.on_predict) or (tag == "clear" and M.on_clear)
  if not action then return false end
  local server = window_server(player)
  if server then player.print("[Factorion] " .. action(player, server)) end
  if tag == "predict" then M.close(player) else M.refresh(player, true) end
  return true
end

function M.on_gui_checked_state_changed(event)
  if condition_edited(event) then return true end
  local player = game.get_player(event.player_index)
  local server = player and window_server(player)
  if not (server and event.element.tags.factorion_gpu == "read_state") then return false end
  M.settings(server).read_state = event.element.state
  return true
end

M.on_gui_elem_changed = condition_edited
M.on_gui_selection_state_changed = condition_edited

function M.on_gui_confirmed(event)
  local tag = event.element.valid and event.element.tags.factorion_gpu
  if tag ~= "checkpoint" then return false end
  local player = game.get_player(event.player_index)
  if player then load_checkpoint(player) end
  return true
end

function M.on_gui_text_changed(event)
  if condition_edited(event) then return true end
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

function M.on_gui_closed(event)
  if not event.element or not event.element.valid
      or event.element.name ~= FRAME then
    return false
  end
  event.element.destroy()
  return true
end

return M
