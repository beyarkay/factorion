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
    tooltip = "Factories denoised side by side; the one with the highest simulated throughput is built, step by step. Each step takes this many times as long." },
  { key = "seed", caption = "Seed", min = 0, max = 2147483647,
    tooltip = "0 draws a fresh random seed each prediction; any other value replays that seed exactly." },
}

local TOOLTIPS = {
  predict = "Start a prediction, as the Predict button does, each time this condition becomes true.",
  deconstruct = "Each time this condition becomes true, stop the cluster and mark everything on its " ..
    "build region for deconstruction, as the Clear button does.",
  read_state = "Write this server's state to its wires, each reading on the signal chosen below.",
  endpoints = "Take the factory's sources and sinks from item signals on the wires instead of from " ..
    "source and sink belts. Conditions then ignore item signals.",
}

local ENDPOINT_HELP = "Each item signal on the green wire is a source and on the red wire a sink, " ..
  "at X + 100 × Y + 10000 × D: X and Y count tiles from the build region's top-left corner, " ..
  "0 being just outside it, and D is 1, 2, 3 or 4 for facing north, east, south or west."

-- What Read state writes, each on a signal the player can change.
M.STATE_SIGNALS = {
  { key = "predicting", caption = "Denoising steps left",
    tooltip = "Counts down to 0 as a prediction places its steps; 0 when idle or done." },
  { key = "utilisation", caption = "Utilisation (%)", tooltip = "How busy this server has been lately." },
  { key = "cluster", caption = "Powered servers in cluster", tooltip = "How many servers predict together." },
  { key = "throughput", caption = "Sink throughput (per minute)", tooltip = "Items per minute reaching the sinks." },
}

-- The signal a reading goes out on, or nil when the player cleared it.
function M.state_signal(settings, key)
  local chosen = (settings.state_signals or {})[key]
  if chosen == false then return nil end
  return chosen or { type = "virtual", name = "factorion-" .. key }
end

local function grey_state_rows(section)
  for _, row in ipairs(M.STATE_SIGNALS) do section["row_" .. row.key].signal.enabled = section.active.state end
end

local DEFAULT_CONDITIONS = {
  predict = { signal = { type = "virtual", name = "signal-P" }, comparator = ">", constant = 0 },
  deconstruct = { signal = { type = "virtual", name = "signal-D" }, comparator = ">", constant = 0 },
}

local FACING = { "north", "east", "south", "west" }
local ROLE = { source = "Source", sink = "Sink" }

-- A wire signal: its value, what that decodes to and, when unused, why.
local function signal_line(signal)
  local value = signal.value
  local x, y, d = value % 100, math.floor(value / 100) % 100, math.floor(value / 10000)
  local line = string.format("[item=%s] %d: %s", signal.item, value, ROLE[signal.role])
  if signal.direction then
    line = line .. string.format(" at X %d, Y %d, facing %s", x, y, FACING[d])
  else
    line = line .. string.format(" with X %d, Y %d, D %d", x, y, d)
  end
  if signal.problem then
    line = "[img=utility/warning_icon] " .. line .. ", [color=1,0.6,0.2]unused: " .. signal.problem .. "[/color]"
  end
  return line
end

-- Like vanilla's slot counts: 950, 1.2k, 34k, 1.5M.
local function compact(value)
  local size = math.abs(value)
  if size >= 1e6 then return string.format("%.1fM", value / 1e6) end
  if size >= 1e4 then return string.format("%dk", math.floor(value / 1e3)) end
  if size >= 1e3 then return string.format("%.1fk", value / 1e3) end
  return tostring(value)
end

local function fill(flow, lines)
  flow.clear()
  for _, line in ipairs(lines) do
    local label = flow.add({ type = "label", caption = line })
    label.style.single_line = false
    label.style.maximal_width = 360
  end
end

-- The sources and sinks the model will get, in the main panel, and every
-- wire signal decoded, in the circuit panel.
local function show_endpoints(frame, region, from_wires)
  local used, decoded = {}, {}
  if region then
    table.insert(used, from_wires and "From the wires, green for sources and red for sinks:"
      or "From source and sink belts:")
    for _, endpoint in ipairs(region.endpoints) do
      table.insert(used, string.format("[item=%s] %s at X %d, Y %d, facing %s", endpoint.item,
        ROLE[endpoint.role], endpoint.x - region.origin.x + 1, endpoint.y - region.origin.y + 1,
        FACING[endpoint.direction]))
    end
    for _, signal in ipairs(region.signals or {}) do table.insert(decoded, signal_line(signal)) end
    if from_wires and #decoded == 0 then table.insert(decoded, "No item signals on the wires.") end
  elseif from_wires then
    table.insert(decoded, "Decoded once this server has a build region.")
  end
  fill(frame.panels.window.body.main.endpoints, used)
  local warnings = {}
  for _, warning in ipairs(region and region.warnings or {}) do
    table.insert(warnings, "[img=utility/warning_icon] [color=1,0.75,0.3]" .. warning .. "[/color]")
  end
  fill(frame.panels.window.body.main.warnings, warnings)
  fill(frame.panels.circuit.inner.endpoints.signals, decoded)
end

-- Set by control.lua: on_load gets the new checkpoint when the player presses
-- Load; describe, on_predict and on_clear get the window's GPU server and
-- return a message (describe also returns whether Predict can run, its
-- cluster and its region, if it has one); info returns the server's live state; settings its circuit
-- settings, and settings_changed shares an edit with its cluster.
M.on_load = function(_) end
M.describe = function(_) return "", false, "" end
M.on_predict = function(_, _, _) return "", false end
M.on_clear = function(_, _) return "" end
M.info = function(_)
  return { status = "idle", caption = "", utilisation = 0, power_kw = 0, networks = {}, powered = false, readings = {} }
end
M.settings = function(_) return { conditions = {} } end
M.settings_changed = function(_) end

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

-- Whether each player keeps the circuit panel open, as vanilla remembers.
local function show_circuit(player, frame)
  local open = (storage.gpu_circuit_open or {})[player.index] or false
  frame.panels.circuit.visible = open
  frame.panels.window.titlebar.circuit_button.toggled = open
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
  local main = frame.panels.window.body.main
  local info = M.info(server)
  gui.set_status(main, info.status, info.caption)
  main.utilisation.value = info.utilisation
  main.utilisation_label.caption = string.format("Utilisation %d%%, drawing %.1f kW",
    math.floor(info.utilisation * 100 + 0.5), info.power_kw)
  gui.set_networks(frame.panels.circuit.inner, info.networks)
  local read_state = frame.panels.circuit.inner.read_state
  for _, row in ipairs(M.STATE_SIGNALS) do
    local value = read_state.active.state and info.readings[row.key] or 0
    read_state["row_" .. row.key].count.caption = value == 0 and "" or compact(value)
  end
  if not full then return end
  main.settings.model_status.caption = model_caption()
  local caption, can_predict, cluster, region = M.describe(server)
  show_endpoints(frame, region, M.settings(server).circuit_endpoints)
  main.region.caption = caption
  main.cluster.caption = cluster
  main.action_row.predict_button.enabled = can_predict
  main.action_row.clear_button.enabled = info.powered
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
  local frame, main = gui.entity_window(player, FRAME, "GPU server", server,
    { factorion_gpu = "circuit" })

  local bar = main.add({ type = "progressbar", name = "utilisation", style = "production_progressbar" })
  bar.style.horizontally_stretchable = true
  main.add({ type = "label", name = "utilisation_label" })
  local cluster = main.add({ type = "label", name = "cluster" })
  cluster.style.single_line = false
  cluster.style.maximal_width = 360

  main.add({ type = "flow", name = "warnings", direction = "vertical" })
  gui.heading(main, "Build region")
  local region = main.add({ type = "label", name = "region" })
  region.style.single_line = false
  region.style.maximal_width = 360
  main.add({ type = "flow", name = "endpoints", direction = "vertical" })
  local actions = main.add({ type = "flow", name = "action_row", direction = "horizontal" })
  actions.add({
    type = "button", name = "predict_button", caption = "Predict", style = "green_button",
    tooltip = "Build a factory on the build region touching this server with " ..
      "every GPU server sharing it, starting from whatever is already on it.",
    tags = { factorion_gpu = "predict" },
  })
  actions.add({
    type = "button", name = "clear_button", caption = "Clear",
    tooltip = "Stop this cluster's prediction and mark everything on its build region for deconstruction. Unbuilt ghosts vanish.",
    tags = { factorion_gpu = "clear" },
  })

  gui.heading(main, "Model, shared by every GPU server")
  local table_ = main.add({ type = "table", name = "settings", column_count = 2 })
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

  local circuit = gui.circuit_panel(frame)
  for _, key in ipairs({ "predict", "deconstruct" }) do
    gui.condition_section(circuit, key, key == "predict" and "Predict" or "Deconstruct build region",
      settings.conditions[key] or DEFAULT_CONDITIONS[key], TOOLTIPS[key])
  end
  local read_state = gui.circuit_section(circuit, "read_state", "Read state", settings.read_state ~= false,
    { factorion_gpu = "read_state" }, TOOLTIPS.read_state)
  for _, row in ipairs(M.STATE_SIGNALS) do
    local flow = read_state.add({ type = "flow", name = "row_" .. row.key, direction = "horizontal" })
    flow.style.vertical_align = "center"
    flow.add({ type = "label", caption = row.caption, tooltip = row.tooltip })
    flow.add({ type = "empty-widget" }).style.horizontally_stretchable = true
    flow.add({
      type = "choose-elem-button", name = "signal", elem_type = "signal",
      signal = M.state_signal(settings, row.key), style = "slot_button_in_shallow_frame",
      tags = { factorion_state = row.key },
    })
    -- The live value over the slot's bottom-right corner, as vanilla's are.
    local count = flow.add({ type = "label", name = "count", style = "count_label", ignored_by_interaction = true })
    count.style.width = 36
    count.style.horizontal_align = "right"
    count.style.left_margin = -40
    count.style.top_margin = 16
  end
  grey_state_rows(read_state)
  local endpoints = gui.circuit_section(circuit, "endpoints", "Set sources and sinks",
    settings.circuit_endpoints or false, { factorion_gpu = "circuit_endpoints" }, TOOLTIPS.endpoints)
  local help = endpoints.add({ type = "label", caption = ENDPOINT_HELP })
  help.style.single_line = false
  help.style.maximal_width = 300
  endpoints.add({ type = "flow", name = "signals", direction = "vertical" })
  show_circuit(player, frame)

  M.refresh(player, true)
  player.opened = frame
end

local function load_checkpoint(player)
  local frame = player.gui.screen[FRAME]
  local field = frame and frame.panels.window.body.main.settings.checkpoint_row.checkpoint
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
  local section = event.element.parent
  if section.name ~= key then section = section.parent end
  settings.conditions[key] = gui.read_condition_section(section)
  settings.circuit_player = player.index
  M.settings_changed(server)
  gui.grey_condition(section)
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
  if tag == "circuit" then
    storage.gpu_circuit_open = storage.gpu_circuit_open or {}
    storage.gpu_circuit_open[player.index] = not storage.gpu_circuit_open[player.index]
    local frame = player.gui.screen[FRAME]
    if frame then show_circuit(player, frame) end
    return true
  end
  local action = (tag == "predict" and M.on_predict) or (tag == "clear" and M.on_clear)
  if not action then return false end
  local server = window_server(player)
  if server then player.print("[Factorion] " .. action(player, server), { sound = defines.print_sound.never }) end
  if tag == "predict" then M.close(player) else M.refresh(player, true) end
  return true
end

function M.on_gui_checked_state_changed(event)
  if condition_edited(event) then return true end
  local player = game.get_player(event.player_index)
  local server = player and window_server(player)
  local tag = event.element.tags.factorion_gpu
  if not (server and (tag == "read_state" or tag == "circuit_endpoints")) then return false end
  M.settings(server)[tag] = event.element.state
  M.settings_changed(server)
  if tag == "read_state" then grey_state_rows(event.element.parent) end
  M.refresh(player, true)
  return true
end

function M.on_gui_elem_changed(event)
  if condition_edited(event) then return true end
  local key = event.element.valid and event.element.tags.factorion_state
  local player = game.get_player(event.player_index)
  local server = player and window_server(player)
  if not (key and server) then return false end
  local settings = M.settings(server)
  settings.state_signals = settings.state_signals or {}
  settings.state_signals[key] = event.element.elem_value or false
  M.settings_changed(server)
  return true
end
M.on_gui_selection_state_changed = condition_edited

function M.on_gui_confirmed(event)
  local tag = event.element.valid and event.element.tags.factorion_gpu
  if tag ~= "checkpoint" then return false end
  local player = game.get_player(event.player_index)
  if player then load_checkpoint(player) end
  return true
end

-- A value out of range turns the field red and says what is used instead.
function M.on_gui_text_changed(event)
  if condition_edited(event) then return true end
  local element = event.element
  local tag = element.valid and element.tags.factorion_gpu
  if not tag then return false end
  if tag == "checkpoint" then
    local status = element.parent.parent.model_status
    status.caption = element.text ~= M.config().checkpoint and "Press Load or Enter to load " .. element.text
      or model_caption()
  end
  for _, field in ipairs(NUMBER_FIELDS) do
    if field.key == tag then
      local value = tonumber(element.text)
      local valid = value and value >= field.min and value <= field.max
        and (field.decimal or value == math.floor(value))
      if valid then M.config()[tag] = value end
      element.style = valid and "textbox" or "invalid_value_textfield"
      element.tooltip = valid and field.tooltip or string.format("%s must be %s from %s to %s; using %s.",
        field.caption, field.decimal and "a number" or "a whole number", field.min, field.max, M.config()[tag])
    end
  end
  return true
end

-- After a checkpoint fails to load, the fields show the one still in use.
function M.reset_checkpoint_fields()
  for _, player in pairs(game.connected_players) do
    local frame = player.gui.screen[FRAME]
    if frame then frame.panels.window.body.main.settings.checkpoint_row.checkpoint.text = M.config().checkpoint end
  end
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
