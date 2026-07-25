-- Factorion mod runtime.
--
-- State per player lives in `storage.players[player_index]`:
--   footprint = { x=int, y=int, w=int, h=int }   world-tile bbox
--   pending   = { request_id=string, tick=int }  in-flight request, if any
--
-- Round trip (single-channel RCON, both directions):
--   1. On key `factorion-execute`, we build a request JSON describing the
--      footprint + existing entities + sources + sinks (in footprint-relative
--      coords) and
--      enqueue it on `storage.outbox`. The Python server is polling our
--      `poll_request` remote interface over RCON; when there's something
--      in the queue it gets popped and returned as a JSON string.
--   2. After each model step, the server calls `place_prediction` over RCON.
--      We create that entity directly in the requesting player's world.
--   3. The server calls `finish_prediction` when the model stops.
--
-- No script-output files are involved. The only outbound channel a
-- vanilla mod has is RCON, and we use it both ways.

local parity = require("parity")

local GRID_SIZE = 11

local function get_grid_size()
  return GRID_SIZE
end

local function get_default_item()
  return settings.global["factorion-default-item"].value
end

local function ensure_player_state(player_index)
  storage.players = storage.players or {}
  if not storage.players[player_index] then
    storage.players[player_index] = {
      footprint = nil,
      pending   = nil,
      picker    = nil,
      footprint_render_id = nil,
      predicted_entities = {},
      prediction_placed = 0,
    }
  end
  return storage.players[player_index]
end

local function ensure_pending_lookup()
  -- request_id -> player_index, so the RCON callback can find the player.
  storage.pending_by_request = storage.pending_by_request or {}
  -- FIFO of JSON request strings waiting for the server to pop via RCON.
  storage.outbox = storage.outbox or {}
  storage.endpoints = storage.endpoints or {}
end

-- Forward declarations: these are defined further down but referenced
-- from event handlers (registered earlier in file order). Declaring them
-- as locals here lets the handler closures capture the local slot, which
-- the later assignments fill in.
local try_request_prediction
local json_encode

-- Let a running Python server hot-swap checkpoints without leaving the game.
-- The command only queues the spec; all filesystem/network/model work remains
-- outside Factorio's deterministic Lua sandbox.
commands.add_command("model", "Load a Factorion model: /model <path-or-wandb-id>",
  function(command)
    local player = command.player_index and game.get_player(command.player_index)
    local spec = command.parameter and string.match(command.parameter, "^%s*(.-)%s*$") or ""
    if spec == "" then
      local current = storage.current_model or "(server has not reported one yet)"
      if player then
        player.print("[Factorion] Current model: " .. current)
        if storage.current_model_url then
          player.print("[Factorion] " .. storage.current_model_url)
        end
        player.print("[Factorion] Usage: /model <path-or-wandb-id>")
      end
      return
    end
    storage.model_requests = storage.model_requests or {}
    table.insert(storage.model_requests, {
      spec = spec,
      player_index = command.player_index or 0,
    })
    if player then
      player.print("[Factorion] Loading model " .. spec .. "…")
    end
  end)

-- ----------------------------------------------------------------------------
-- player onboarding
-- ----------------------------------------------------------------------------

local function give_tools(player)
  local inv = player.get_main_inventory()
  if not inv then return end
  if inv.get_item_count("factorion-footprint-tool") == 0 then
    inv.insert({ name = "factorion-footprint-tool", count = 1 })
  end
  if inv.get_item_count("factorion-source-belt") == 0 then
    inv.insert({ name = "factorion-source-belt", count = 10 })
  end
  if inv.get_item_count("factorion-sink-belt") == 0 then
    inv.insert({ name = "factorion-sink-belt", count = 10 })
  end
end

local function show_startup_message(player)
  player.print("[Factorion] Factory-design assistant ready — source/sink belts are active.")
  player.print({ "",
    "1. Press CTRL+T to get the [item=factorion-footprint-tool], ",
    "[item=factorion-source-belt], and [item=factorion-sink-belt]." })
  player.print({ "",
    "2. Stamp the blue 11x11 region, then place the green source belt and ",
    "orange sink belt from your inventory like ordinary belts." })
  player.print(
    "3. Click either endpoint to choose its item. Mine, rotate, copy, paste, " ..
    "and blueprint them normally. Alt mode shows each configured item; sinks " ..
    "show their rolling 5-second throughput above the belt.")
  player.print(
    "4. Hover a source or sink and press R to rotate it. Press CTRL+P to let " ..
    "the model complete the factory around supported entities already in the " ..
    "blue region.")
  player.print(
    "CTRL+R clears the region and model-placed entities; mine endpoint belts " ..
    "normally when you want to pick them up. Keep ./start-mod.sh running.")
end

script.on_event(defines.events.on_player_created, function(event)
  ensure_player_state(event.player_index)
  ensure_pending_lookup()
  local player = game.get_player(event.player_index)
  if player then give_tools(player) end
end)

script.on_event("factorion-give-tools", function(event)
  local player = game.get_player(event.player_index)
  if player then
    give_tools(player)
    player.print("[Factorion] Factorion tools and endpoint belts added to your inventory.")
  end
end)

script.on_event(defines.events.on_player_joined_game, function(event)
  local player = game.get_player(event.player_index)
  if player then
    give_tools(player)
    show_startup_message(player)
  end
end)

-- ----------------------------------------------------------------------------
-- selection-tool handlers
-- ----------------------------------------------------------------------------

local function tile_floor(area_axis)
  -- Selection-area coords are floats at tile *corners*; floor gives the
  -- top-left tile index. Areas come back as {left_top, right_bottom}.
  return math.floor(area_axis + 0.0001)
end

local function area_center_tile(area)
  return tile_floor((area.left_top.x + area.right_bottom.x) / 2),
    tile_floor((area.left_top.y + area.right_bottom.y) / 2)
end

local function fixed_footprint(area)
  local cx, cy = area_center_tile(area)
  local radius = math.floor(GRID_SIZE / 2)
  return { x = cx - radius, y = cy - radius, w = GRID_SIZE, h = GRID_SIZE }
end

local function destroy_footprint_render(state)
  if not state.footprint_render_id then return end
  local object = rendering.get_object_by_id(state.footprint_render_id)
  if object then object.destroy() end
  state.footprint_render_id = nil
end

local function draw_footprint(player, state)
  destroy_footprint_render(state)
  local fp = state.footprint
  local object = rendering.draw_rectangle({
    surface = player.surface,
    left_top = { fp.x, fp.y },
    right_bottom = { fp.x + fp.w, fp.y + fp.h },
    color = { r = 0.1, g = 0.75, b = 1.0, a = 0.95 },
    width = 4,
    filled = false,
    players = { player },
    draw_on_ground = true,
  })
  state.footprint_render_id = object.id
end

local MARKER_DIALOG = "factorion-marker-dialog"
local ITEM_PICKER = "factorion-marker-item"
local FACTORIO_TO_DIRECTION = {
  [defines.direction.north] = 1,
  [defines.direction.east] = 2,
  [defines.direction.south] = 3,
  [defines.direction.west] = 4,
}
local OPPOSITE_DIRECTION = {
  [1] = 3,
  [2] = 4,
  [3] = 1,
  [4] = 2,
}
local ENDPOINT_ENTITY_NAMES = {
  source = "factorion-source-belt",
  sink = "factorion-sink-belt",
}
local MODEL_ENTITY_NAMES = {
  "transport-belt",
  "inserter",
  "long-handed-inserter",
  "assembling-machine-1",
  "underground-belt",
  "splitter",
}
local MODEL_ENTITY_SPECS = {
  ["transport-belt"] = { width = 1, height = 1 },
  ["inserter"] = { width = 1, height = 1, inserter = true },
  ["long-handed-inserter"] = { width = 1, height = 1, inserter = true },
  ["assembling-machine-1"] = {
    width = 3, height = 3, directionless = true, recipe = true,
  },
  ["underground-belt"] = { width = 1, height = 1, underground = true },
  ["splitter"] = { width = 2, height = 1 },
}
local SINK_RATE_SAMPLE_TICKS = 30
local SINK_RATE_WINDOW_TICKS = 5 * 60

local function endpoint_role(entity)
  if not entity or not entity.valid then return nil end
  if entity.name == ENDPOINT_ENTITY_NAMES.source then return "source" end
  if entity.name == ENDPOINT_ENTITY_NAMES.sink then return "sink" end
  return nil
end

local function destroy_endpoint_alt_icon(config)
  if not config or not config.render_id then return end
  local object = rendering.get_object_by_id(config.render_id)
  if object then object.destroy() end
  config.render_id = nil
end

local function destroy_sink_rate_label(config)
  if not config or not config.rate_render_id then return end
  local object = rendering.get_object_by_id(config.rate_render_id)
  if object then object.destroy() end
  config.rate_render_id = nil
end

local function refresh_sink_rate_label(entity, config)
  destroy_sink_rate_label(config)
  if not entity or not entity.valid or not config or config.role ~= "sink" then
    return
  end
  local label = rendering.draw_text({
    text = string.format("%.1f/s", config.throughput_rate or 0),
    surface = entity.surface,
    target = entity,
    target_offset = { 0, -0.7 },
    color = { r = 1.0, g = 0.72, b = 0.2, a = 1.0 },
    scale = 0.8,
    alignment = "center",
    vertical_alignment = "bottom",
    render_mode = "game",
  })
  config.rate_render_id = label.id
end

local function refresh_endpoint_alt_icon(entity, config)
  destroy_endpoint_alt_icon(config)
  if not entity or not entity.valid or not config or not config.item then return end
  local players = {}
  for _, player in pairs(game.connected_players) do
    if player.game_view_settings.show_entity_info then
      table.insert(players, player)
    end
  end
  if #players == 0 then return end
  local icon = rendering.draw_sprite({
    sprite = "item/" .. config.item,
    surface = entity.surface,
    target = entity,
    x_scale = 0.42,
    y_scale = 0.42,
    players = players,
    render_mode = "game",
  })
  config.render_id = icon.id
end

local function clear_downstream_item(source, item)
  local queue = { source }
  local next_index = 1
  local visited = {}
  local removed = 0

  while next_index <= #queue do
    local entity = queue[next_index]
    next_index = next_index + 1
    local unit_number = entity and entity.valid and entity.unit_number or nil
    if unit_number and not visited[unit_number] then
      visited[unit_number] = true
      for line_index = 1, entity.get_max_transport_line_index() do
        local line = entity.get_transport_line(line_index)
        local count = line.get_item_count({ name = item })
        if count > 0 then
          removed = removed + line.remove_item({ name = item, count = count })
        end
        for _, output_line in pairs(line.output_lines) do
          local owner = output_line.owner
          if owner and owner.valid and owner.unit_number
              and not visited[owner.unit_number] then
            table.insert(queue, owner)
          end
        end
      end
    end
  end

  return removed
end

local function register_endpoint(entity, item)
  local role = endpoint_role(entity)
  if not role or not entity.unit_number then return nil end
  storage.endpoints = storage.endpoints or {}
  local config = storage.endpoints[entity.unit_number] or {}
  local item_changed = item and config.item and item ~= config.item
  local cleared_items = 0
  if role == "source" and item_changed then
    cleared_items = clear_downstream_item(entity, config.item)
  end
  config.entity = entity
  config.role = role
  config.item = item or config.item or get_default_item()
  storage.endpoints[entity.unit_number] = config
  if role == "sink" then
    if item_changed or not config.throughput_samples then
      config.throughput_samples = {}
      config.throughput_pending_count = 0
      config.throughput_total_count = 0
      config.throughput_total_ticks = 0
      config.throughput_last_sample_tick = game.tick
      config.throughput_rate = 0
    end
    refresh_sink_rate_label(entity, config)
  else
    destroy_sink_rate_label(config)
  end
  refresh_endpoint_alt_icon(entity, config)
  return config, cleared_items
end

local function endpoint_config(entity)
  if not entity or not entity.unit_number then return nil end
  storage.endpoints = storage.endpoints or {}
  return storage.endpoints[entity.unit_number]
end

local function clear_predicted_entities(state)
  for _, unit_number in ipairs(state.predicted_entities or {}) do
    local entity = game.get_entity_by_unit_number(unit_number)
    if entity and entity.valid then entity.destroy({ raise_destroy = true }) end
  end
  state.predicted_entities = {}
  state.prediction_placed = 0
end

local function close_marker_dialog(player, state)
  local frame = player.gui.screen[MARKER_DIALOG]
  if frame then frame.destroy() end
  state.picker = nil
end

local function open_marker_dialog(player, state, role, entity)
  close_marker_dialog(player, state)
  state.picker = {
    role = role,
    entity_unit_number = entity.unit_number,
  }
  local config = endpoint_config(entity)
  local title = "Configure Factorion " .. role .. " belt"
  local prompt = role == "source"
    and "Choose the item this source belt produces:"
    or "Choose the item this sink counts toward throughput:"
  local frame = player.gui.screen.add({
    type = "frame", name = MARKER_DIALOG, caption = title,
    direction = "vertical",
  })
  frame.auto_center = true
  frame.add({ type = "label", caption = prompt })
  frame.add({
    type = "choose-elem-button", name = ITEM_PICKER,
    elem_type = "item", item = config and config.item or get_default_item(),
  })
  local actions = frame.add({ type = "flow", direction = "horizontal" })
  actions.add({
    type = "button", caption = "Cancel",
    tags = { factorion_action = "marker-cancel" },
  })
  actions.add({
    type = "button", caption = "Save item",
    style = "confirm_button",
    tags = { factorion_action = "marker-save" },
  })
  player.opened = frame
end

local function handle_tool_selection(event)
  if event.item ~= "factorion-footprint-tool" then return end
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  if not player then return end

  clear_predicted_entities(state)
  state.footprint = fixed_footprint(event.area)
  draw_footprint(player, state)
  local fp = state.footprint
  player.print(string.format(
    "[Factorion] Stamped 11x11 region at x=%d..%d, y=%d..%d. " ..
    "Now place a source and sink.",
    fp.x, fp.x + 10, fp.y, fp.y + 10))
end

script.on_event(defines.events.on_player_selected_area, handle_tool_selection)

-- Right-clicking the region tool is a quick clear.
script.on_event(defines.events.on_player_alt_selected_area, function(event)
  if event.item ~= "factorion-footprint-tool" then return end
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  destroy_footprint_render(state)
  clear_predicted_entities(state)
  state.footprint = nil
  if player then
    player.print("[Factorion] Region cleared. Mine placed endpoint belts normally.")
  end
end)

local function save_marker_dialog(player, state)
  if not state.picker then
    player.print(
      "[Factorion] That endpoint dialog had closed. Click the belt and try again.")
    close_marker_dialog(player, state)
    return
  end
  local frame = player.gui.screen[MARKER_DIALOG]
  local picker = frame and frame[ITEM_PICKER]
  local item = picker and picker.elem_value
  if not item then
    player.print("[Factorion] Choose an item first.")
    return
  end
  local config = storage.endpoints
    and storage.endpoints[state.picker.entity_unit_number] or nil
  local entity = config and config.entity or nil
  if not entity or not endpoint_role(entity) then
    player.print("[Factorion] That endpoint belt no longer exists.")
    close_marker_dialog(player, state)
    return
  end
  local cleared_items
  config, cleared_items = register_endpoint(entity, item)
  if not config then return end
  if config.role == "source" and cleared_items > 0 then
    player.print(string.format(
      "[Factorion] Source belt now uses [item=%s]; cleared %d old item(s) " ..
      "from its downstream belts.",
      config.item, cleared_items))
  else
    player.print(string.format(
      "[Factorion] %s belt now uses [item=%s].",
      config.role, config.item))
  end
  close_marker_dialog(player, state)
end

script.on_event(defines.events.on_gui_click, function(event)
  local element = event.element
  if not element or not element.valid then return end
  local action = element.tags and element.tags.factorion_action
  if not action then return end
  local player = game.get_player(event.player_index)
  if not player then return end
  local state = ensure_player_state(event.player_index)
  if action == "marker-cancel" then
    close_marker_dialog(player, state)
    return
  end
  if action ~= "marker-save" then return end
  save_marker_dialog(player, state)
end)

script.on_event("factorion-confirm-dialog", function(event)
  local player = game.get_player(event.player_index)
  if not player then return end
  local state = ensure_player_state(event.player_index)
  if not state.picker or not player.gui.screen[MARKER_DIALOG] then return end
  save_marker_dialog(player, state)
end)

script.on_event(defines.events.on_gui_closed, function(event)
  if not event.element or not event.element.valid
      or event.element.name ~= MARKER_DIALOG then return end
  local player = game.get_player(event.player_index)
  if not player then return end
  local state = ensure_player_state(event.player_index)
  close_marker_dialog(player, state)
end)

script.on_event(defines.events.on_gui_opened, function(event)
  local entity = event.entity
  local role = endpoint_role(entity)
  if not role then return end
  local player = game.get_player(event.player_index)
  if not player then return end
  player.opened = nil
  register_endpoint(entity)
  open_marker_dialog(player, ensure_player_state(event.player_index), role, entity)
end)

local function endpoint_built(event)
  local entity = event.entity or event.created_entity
  local role = endpoint_role(entity)
  if not role then return end
  local item = event.tags and event.tags.factorion_item or nil
  register_endpoint(entity, item)
  if event.player_index then
    local player = game.get_player(event.player_index)
    if player then
      player.print(string.format(
        "[Factorion] Placed %s belt using [item=%s]. Click it to change the item.",
        role, endpoint_config(entity).item))
    end
  end
end

script.on_event(defines.events.on_built_entity, endpoint_built)
script.on_event(defines.events.on_robot_built_entity, endpoint_built)
script.on_event(defines.events.script_raised_built, endpoint_built)
script.on_event(defines.events.script_raised_revive, endpoint_built)

script.on_event(defines.events.on_entity_cloned, function(event)
  local source = endpoint_config(event.source)
  if endpoint_role(event.destination) then
    register_endpoint(event.destination, source and source.item or nil)
  end
end)

script.on_event(defines.events.on_entity_settings_pasted, function(event)
  local source = endpoint_config(event.source)
  if source and endpoint_role(event.destination) then
    local destination = register_endpoint(event.destination, source.item)
    if not destination then return end
    local player = game.get_player(event.player_index)
    if player then
      player.print(string.format(
        "[Factorion] Pasted [item=%s] onto %s belt.",
        destination.item, destination.role))
    end
  end
end)

script.on_event(defines.events.on_player_setup_blueprint, function(event)
  local blueprint = event.stack or event.record
  if not blueprint or not event.mapping then return end
  for index, entity in pairs(event.mapping.get()) do
    local config = endpoint_config(entity)
    if config then
      blueprint.set_blueprint_entity_tag(index, "factorion_item", config.item)
    end
  end
end)

script.on_event(defines.events.on_player_toggled_alt_mode, function()
  for _, config in pairs(storage.endpoints or {}) do
    local entity = config.entity
    if entity then refresh_endpoint_alt_icon(entity, config) end
  end
end)

local function endpoint_entity_removed(event)
  local entity = event.entity
  if not entity or not entity.unit_number then return end
  local config = storage.endpoints and storage.endpoints[entity.unit_number] or nil
  destroy_endpoint_alt_icon(config)
  destroy_sink_rate_label(config)
  if storage.endpoints then storage.endpoints[entity.unit_number] = nil end
end

script.on_event(defines.events.on_player_mined_entity, endpoint_entity_removed)
script.on_event(defines.events.on_robot_mined_entity, endpoint_entity_removed)
script.on_event(defines.events.on_entity_died, endpoint_entity_removed)
script.on_event(defines.events.script_raised_destroy, endpoint_entity_removed)

local function feed_source(mark, entity)
  for line_index = 1, entity.get_max_transport_line_index() do
    local line = entity.get_transport_line(line_index)
    -- Four unstacked items fit on a one-tile lane. Eight attempts also fill
    -- stacked lanes while remaining bounded if another mod changes belt rules.
    for _ = 1, 8 do
      if not line.can_insert_at_back() then break end
      if not line.insert_at_back({ name = mark.item, count = 1 }) then break end
    end
  end
end

local function drain_sink(mark, entity)
  local removed_total = 0
  local left = entity.position.x - 0.51
  local right = entity.position.x + 0.51
  local top = entity.position.y - 0.51
  local bottom = entity.position.y + 0.51
  for line_index = 1, entity.get_max_transport_line_index() do
    local line = entity.get_transport_line(line_index)
    local removals = {}
    for _, detail in ipairs(line.get_detailed_contents()) do
      local stack = detail.stack
      if stack.valid_for_read then
        local position = line.get_line_item_position(detail.position)
        if position.x >= left and position.x <= right
            and position.y >= top and position.y <= bottom then
          local quality = stack.quality and stack.quality.name or nil
          local key = stack.name .. "\0" .. (quality or "")
          local removal = removals[key]
          if not removal then
            removal = { name = stack.name, quality = quality, count = 0 }
            removals[key] = removal
          end
          removal.count = removal.count + stack.count
        end
      end
    end
    for _, removal in pairs(removals) do
      local request = { name = removal.name, count = removal.count }
      if removal.quality then request.quality = removal.quality end
      local removed = line.remove_item(request)
      if removal.name == mark.item then
        removed_total = removed_total + removed
      end
    end
  end
  return removed_total
end

local function sample_sink_throughput(config, entity)
  local elapsed = game.tick - (config.throughput_last_sample_tick or game.tick)
  if elapsed < SINK_RATE_SAMPLE_TICKS then return end

  local sample = {
    count = config.throughput_pending_count or 0,
    ticks = elapsed,
  }
  local samples = config.throughput_samples or {}
  table.insert(samples, sample)
  config.throughput_samples = samples
  config.throughput_pending_count = 0
  config.throughput_last_sample_tick = game.tick
  config.throughput_total_count =
    (config.throughput_total_count or 0) + sample.count
  config.throughput_total_ticks =
    (config.throughput_total_ticks or 0) + sample.ticks

  while config.throughput_total_ticks > SINK_RATE_WINDOW_TICKS
      and #samples > 1 do
    local expired = table.remove(samples, 1)
    config.throughput_total_count =
      config.throughput_total_count - expired.count
    config.throughput_total_ticks =
      config.throughput_total_ticks - expired.ticks
  end

  if config.throughput_total_ticks > 0 then
    config.throughput_rate =
      config.throughput_total_count * 60 / config.throughput_total_ticks
  else
    config.throughput_rate = 0
  end
  refresh_sink_rate_label(entity, config)
end

local function service_endpoint_belts()
  for unit_number, config in pairs(storage.endpoints or {}) do
    local entity = config.entity
    if not entity or not endpoint_role(entity) then
      destroy_endpoint_alt_icon(config)
      storage.endpoints[unit_number] = nil
    elseif config.role == "source" then
      feed_source(config, entity)
    else
      config.throughput_pending_count =
        (config.throughput_pending_count or 0) + drain_sink(config, entity)
      sample_sink_throughput(config, entity)
    end
  end
end

local function discover_endpoint_belts()
  for _, surface in pairs(game.surfaces) do
    local entities = surface.find_entities_filtered({
      name = { ENDPOINT_ENTITY_NAMES.source, ENDPOINT_ENTITY_NAMES.sink },
    })
    for _, entity in pairs(entities) do
      local config = endpoint_config(entity)
      local role = endpoint_role(entity)
      if not config or not config.entity or not config.entity.valid then
        register_endpoint(entity)
      elseif role == "sink" and not config.throughput_samples then
        register_endpoint(entity)
      end
    end
  end
end

script.on_nth_tick(60, discover_endpoint_belts)

script.on_event("factorion-reset", function(event)
  local state = ensure_player_state(event.player_index)
  destroy_footprint_render(state)
  clear_predicted_entities(state)
  state.footprint = nil
  local player = game.get_player(event.player_index)
  if player then
    close_marker_dialog(player, state)
    player.print("[Factorion] Region cleared. Mine placed endpoint belts normally.")
  end
end)

-- ----------------------------------------------------------------------------
-- execute: write a request JSON the server picks up
-- ----------------------------------------------------------------------------

local function world_to_rel(p, fp)
  return p.x - fp.x, p.y - fp.y
end

local function in_footprint(rel_x, rel_y, fp)
  return rel_x >= 0 and rel_x < fp.w and rel_y >= 0 and rel_y < fp.h
end

local function new_request_id()
  return string.format("%d-%d-%d", game.tick,
    math.random(0, 2^30), math.random(0, 2^30))
end

local function build_footprint_mask(fp)
  -- For now: every tile in the bbox is buildable. Later we may let the
  -- player exclude tiles inside the bbox.
  local tiles = {}
  for y = 0, fp.h - 1 do
    for x = 0, fp.w - 1 do
      table.insert(tiles, { x, y })
    end
  end
  return tiles
end

local function gather_existing_entities(state, surface, fp)
  local predicted = {}
  for _, unit_number in ipairs(state.predicted_entities or {}) do
    predicted[unit_number] = true
  end

  local result = {}
  local candidates = surface.find_entities_filtered({
    area = { { fp.x, fp.y }, { fp.x + fp.w, fp.y + fp.h } },
    name = MODEL_ENTITY_NAMES,
  })
  for _, entity in pairs(candidates) do
    if not predicted[entity.unit_number] then
      local spec = MODEL_ENTITY_SPECS[entity.name]
      local direction = 0
      if not spec.directionless then
        direction = FACTORIO_TO_DIRECTION[entity.direction]
      end
      if direction then
        -- Factorio stores an inserter's drop direction; the model stores its
        -- pickup direction.
        if spec.inserter then direction = OPPOSITE_DIRECTION[direction] end

        local width, height = spec.width, spec.height
        if direction == 2 or direction == 4 then
          width, height = height, width
        end
        local world_x = math.floor(entity.position.x - width / 2 + 0.0001)
        local world_y = math.floor(entity.position.y - height / 2 + 0.0001)
        local rx, ry = world_x - fp.x, world_y - fp.y

        -- An entity crossing the region boundary cannot be represented as a
        -- complete model unit, so leave it out rather than supplying a clipped
        -- and misleading footprint.
        if rx >= 0 and ry >= 0 and rx + width <= fp.w and ry + height <= fp.h then
          local entry = {
            name = entity.name,
            x = rx,
            y = ry,
            direction = direction,
          }
          if spec.recipe then
            local recipe = entity.get_recipe()
            if recipe then entry.item = recipe.name end
          end
          if spec.underground then
            if entity.belt_to_ground_type == "input" then
              entry.misc = 1
            elseif entity.belt_to_ground_type == "output" then
              entry.misc = 2
            end
          end
          table.insert(result, entry)
        end
      end
    end
  end
  return result
end

local function gather_request(state, player_index)
  local fp = state.footprint
  local size = get_grid_size()
  local default_item = get_default_item()
  local player = game.get_player(player_index)

  local sources = {}
  local sinks = {}
  local model_entities = gather_existing_entities(state, player.surface, fp)
  local entities = player.surface.find_entities_filtered({
    area = { { fp.x, fp.y }, { fp.x + fp.w, fp.y + fp.h } },
    name = { ENDPOINT_ENTITY_NAMES.source, ENDPOINT_ENTITY_NAMES.sink },
  })
  for _, entity in pairs(entities) do
    local x, y = math.floor(entity.position.x), math.floor(entity.position.y)
    local rx, ry = world_to_rel({ x = x, y = y }, fp)
    if in_footprint(rx, ry, fp) then
      local config = register_endpoint(entity)
      if not config then goto continue end
      local entry = {
        x = rx, y = ry,
        direction = FACTORIO_TO_DIRECTION[entity.direction],
        item = config.item or default_item,
      }
      if config.role == "source" then
        table.insert(sources, entry)
      else
        table.insert(sinks, entry)
      end
    end
    ::continue::
  end

  local provenance = string.format(
    "%d existing supported entities, %d source belt(s), %d sink belt(s)",
    #model_entities, #sources, #sinks)

  local request = {
    request_id    = new_request_id(),
    player_index  = player_index,
    grid_size     = size,
    footprint     = build_footprint_mask(fp),
    entities      = model_entities,
    sources       = sources,
    sinks         = sinks,
    default_item  = default_item,
  }
  return request, provenance
end

-- Build a request from the player's configured endpoint belts and enqueue it
-- for the server. Returns
-- (true, message) on success, (false, message) on a precheck failure.
try_request_prediction = function(player_index)
  local state = ensure_player_state(player_index)
  ensure_pending_lookup()
  if not state.footprint then
    return false, "No footprint set."
  end
  local request, provenance = gather_request(state, player_index)
  if #request.sources == 0 then
    return false, "No source belt inside the region."
  end
  if #request.sinks == 0 then
    return false, "No sink belt inside the region."
  end
  clear_predicted_entities(state)
  state.prediction_placed = 0
  local request_json = json_encode(request)
  table.insert(storage.outbox, request_json)
  storage.pending_by_request[request.request_id] = player_index
  state.pending = { request_id = request.request_id, tick = game.tick }
  log("[Factorion] enqueued request " .. request.request_id ..
      " (" .. provenance .. ") json=" .. request_json)
  local model = storage.current_model or "unknown (server has not identified it yet)"
  return true, string.format(
    "Request %s queued using model %s (%s). Waiting for server…",
    request.request_id, model, provenance)
end

json_encode = function(t)
  -- Factorio 2.0 exposes helpers.table_to_json; the older `game.table_to_json`
  -- is deprecated but kept for compat. Prefer helpers when present.
  if helpers and helpers.table_to_json then
    return helpers.table_to_json(t)
  end
  return game.table_to_json(t)
end

local function describe_placement_failure(surface, player, fp, placement)
  local left = fp.x + placement.tile_x
  local top = fp.y + placement.tile_y
  local right = left + placement.width
  local bottom = top + placement.height

  local blockers = {}
  local found = surface.find_entities({
    { left, top },
    { right, bottom },
  })
  for _, entity in pairs(found) do
    local unit = entity.unit_number and ("#" .. entity.unit_number) or ""
    table.insert(blockers, string.format(
      "%s%s(type=%s,pos=%.2f,%.2f)",
      entity.name,
      unit,
      entity.type,
      entity.position.x,
      entity.position.y))
  end
  table.sort(blockers)
  if #blockers > 8 then
    local extra = #blockers - 8
    while #blockers > 8 do table.remove(blockers) end
    table.insert(blockers, string.format("...(+%d more)", extra))
  end

  local ground_set = {}
  for x = left, right - 1 do
    for y = top, bottom - 1 do
      ground_set[surface.get_tile(x, y).name] = true
    end
  end
  local ground = {}
  for name in pairs(ground_set) do table.insert(ground, name) end
  table.sort(ground)

  local can_place = surface.can_place_entity({
    name = placement.name,
    position = { fp.x + placement.x, fp.y + placement.y },
    direction = placement.direction,
    force = player.force,
  })
  return string.format(
    "error: Factorio refused %s at relative tile (%s,%s), " ..
    "world tile (%d,%d) on %s; can_place_entity=%s; blockers=[%s]; ground=[%s]",
    tostring(placement.name),
    tostring(placement.tile_x),
    tostring(placement.tile_y),
    left,
    top,
    surface.name,
    tostring(can_place),
    #blockers > 0 and table.concat(blockers, ", ") or "none",
    #ground > 0 and table.concat(ground, ", ") or "none")
end

script.on_event("factorion-execute", function(event)
  local player = game.get_player(event.player_index)
  if not player then return end
  local _, msg = try_request_prediction(event.player_index)
  player.print("[Factorion] " .. msg)
end)

-- ----------------------------------------------------------------------------
-- RCON interface: server streams predictions back here
-- ----------------------------------------------------------------------------

-- Re-registration safety: remote.add_interface errors if the name already
-- exists. On /c game.reload_script() the old interface is still bound, so
-- the new methods (e.g. dump_state) would never get added. Remove first.
if remote.interfaces["factorion"] then
  remote.remove_interface("factorion")
end

remote.add_interface("factorion", {
  -- The server's poll: pop and return the oldest pending request JSON, or
  -- "" if the queue is empty. The server wraps the call in
  -- `/silent-command rcon.print(remote.call('factorion','poll_request'))`
  -- so whatever we return here ends up in the RCON response stream.
  poll_request = function()
    storage.outbox = storage.outbox or {}
    if #storage.outbox == 0 then return "" end
    return table.remove(storage.outbox, 1)
  end,

  -- The Python server polls model changes separately from factory requests so
  -- /model can replace the in-memory AgentCNN without restarting either side.
  poll_model = function()
    storage.model_requests = storage.model_requests or {}
    if #storage.model_requests == 0 then return "" end
    return json_encode(table.remove(storage.model_requests, 1))
  end,

  model_status = function(player_index, ok, message, model_name, model_url)
    local player = player_index and player_index > 0
      and game.get_player(player_index) or nil
    if ok then
      storage.current_model = model_name or message
      storage.current_model_url = model_url
    end
    if player then
      player.print((ok and "[Factorion] " or "[Factorion] Model error: ") .. message)
    end
    return true
  end,

  -- Create one model-predicted entity at footprint-relative coordinates.
  place_prediction = function(request_id, placement_json)
    ensure_pending_lookup()
    local player_index = storage.pending_by_request[request_id]
    if not player_index then
      return "error: unknown request_id " .. tostring(request_id)
    end

    if player_index == 0 then
      log("[Factorion] (headless) placement for " .. tostring(request_id) ..
        ": " .. tostring(placement_json))
      return "ok"
    end

    local player = game.get_player(player_index)
    if not player then return "error: requesting player is unavailable" end
    local state = ensure_player_state(player_index)
    local fp = state.footprint
    if not fp then return "error: footprint was cleared" end

    local ok, placement = pcall(
      function() return helpers.json_to_table(placement_json) end)
    if not ok or not placement then return "error: invalid placement JSON" end
    if not placement.name or not placement.tile_x or not placement.tile_y
        or not placement.width or not placement.height then
      return "error: incomplete placement"
    end
    if placement.tile_x < 0 or placement.tile_y < 0
        or placement.tile_x + placement.width > fp.w
        or placement.tile_y + placement.height > fp.h then
      return "error: placement lies outside the footprint"
    end

    local params = {
      name = placement.name,
      position = { fp.x + placement.x, fp.y + placement.y },
      direction = placement.direction,
      force = player.force,
      player = player,
      raise_built = true,
      create_build_effect_smoke = true,
    }
    if placement.type then params.type = placement.type end
    if placement.recipe then params.recipe = placement.recipe end
    local created, entity = pcall(
      function() return player.surface.create_entity(params) end)
    if not created then
      return string.format(
        "error: create_entity raised while placing %s at relative tile " ..
        "(%s,%s): %s",
        tostring(placement.name),
        tostring(placement.tile_x),
        tostring(placement.tile_y),
        tostring(entity))
    end
    if not entity then
      return describe_placement_failure(
        player.surface, player, fp, placement)
    end

    state.predicted_entities = state.predicted_entities or {}
    if entity.unit_number then
      table.insert(state.predicted_entities, entity.unit_number)
    end
    state.prediction_placed = (state.prediction_placed or 0) + 1
    return "ok"
  end,

  -- Complete the streamed request and release its pending state.
  finish_prediction = function(request_id, summary_json)
    ensure_pending_lookup()
    local player_index = storage.pending_by_request[request_id]
    if not player_index then
      return "error: unknown request_id " .. tostring(request_id)
    end
    storage.pending_by_request[request_id] = nil
    if player_index == 0 then
      log("[Factorion] (headless) prediction complete for " ..
        tostring(request_id) .. ": " .. tostring(summary_json))
      return "ok"
    end

    local player = game.get_player(player_index)
    local state = ensure_player_state(player_index)
    state.pending = nil
    local ok, summary = pcall(
      function() return helpers.json_to_table(summary_json) end)
    local reason = ok and summary and summary.stop_reason or "unknown"
    if player then
      player.print(string.format(
        "[Factorion] Prediction complete: placed %d entities (stop: %s).",
        state.prediction_placed or 0, tostring(reason)))
    end
    return "ok"
  end,

  -- Diagnostic: server can call this to check the mod is alive.
  ping = function()
    return "factorion-mod alive at tick " .. tostring(game.tick)
  end,
  protocol_version = function()
    return "4"
  end,

  -- Headless / debug: enqueue a request JSON as if the hotkey had fired.
  -- /silent-command runs in *level scope*, not any mod's, so storage is
  -- inaccessible from RCON directly — this interface is the only way to
  -- poke an outbox entry in from outside the game.
  --
  -- Caller supplies the request JSON and a player_index to deliver the
  -- response to. In headless tests with no player connected, pass 0 to
  -- signal "log streamed placements instead of creating world entities".
  inject_request = function(request_json, deliver_to_player_index)
    storage.outbox = storage.outbox or {}
    storage.pending_by_request = storage.pending_by_request or {}
    table.insert(storage.outbox, request_json)
    -- Parse to extract request_id; helpers.json_to_table is the 2.0+ API.
    local ok, parsed = pcall(function() return helpers.json_to_table(request_json) end)
    if ok and parsed and parsed.request_id then
      storage.pending_by_request[parsed.request_id] =
        deliver_to_player_index or 0
    end
    return "queued, depth=" .. #storage.outbox
  end,

  -- Headless / debug: introspect mod storage from outside.
  introspect = function()
    storage.outbox = storage.outbox or {}
    storage.pending_by_request = storage.pending_by_request or {}
    local pending_n = 0
    for _ in pairs(storage.pending_by_request) do pending_n = pending_n + 1 end
    local endpoint_n = 0
    for _ in pairs(storage.endpoints or {}) do endpoint_n = endpoint_n + 1 end
    return string.format("outbox=%d pending=%d players=%d endpoints=%d",
      #storage.outbox, pending_n,
      storage.players and table_size(storage.players) or 0, endpoint_n)
  end,

  -- Parity harness (server/parity.py): build a factory spec on the
  -- dedicated lab surface, run it, measure per-sink / per-entity
  -- throughput. All three return JSON strings for rcon.print().
  parity_start = function(spec_json)
    return parity.start(spec_json)
  end,
  parity_poll = function()
    return parity.poll()
  end,
  parity_abort = function()
    return parity.abort()
  end,

  -- Headless / debug: dump full state for a given player (or all players).
  dump_state = function(player_index)
    storage.players = storage.players or {}
    local parts = {}
    for k, p in pairs(storage.players) do
      if player_index == nil or k == player_index then
        local fp = p.footprint and string.format("x=%d,y=%d,w=%d,h=%d",
            p.footprint.x, p.footprint.y, p.footprint.w, p.footprint.h)
          or "nil"
        table.insert(parts, string.format(
          "player[%s]: footprint={%s}", tostring(k), fp))
      end
    end
    if #parts == 0 then return "(no player state)" end
    return table.concat(parts, "\n")
  end,
})

-- Endpoint belts drive the player's factory; parity drives its isolated lab
-- surface and early-outs when no run is active.
script.on_event(defines.events.on_tick, function()
  service_endpoint_belts()
  parity.on_tick()
end)

-- ----------------------------------------------------------------------------
-- migrations
-- ----------------------------------------------------------------------------

script.on_init(function()
  storage.players = {}
  storage.pending_by_request = {}
  storage.outbox = {}
  storage.model_requests = {}
  storage.endpoints = {}
end)

script.on_configuration_changed(function()
  storage.players = storage.players or {}
  storage.pending_by_request = storage.pending_by_request or {}
  storage.outbox = storage.outbox or {}
  storage.model_requests = storage.model_requests or {}
  storage.endpoints = storage.endpoints or {}
  for _, player in pairs(game.players) do
    local state = ensure_player_state(player.index)
    state.predicted_entities = state.predicted_entities or {}
    state.prediction_placed = state.prediction_placed or 0
    give_tools(player)
    if state.footprint then draw_footprint(player, state) end
  end
  for _, surface in pairs(game.surfaces) do
    for _, entity in pairs(surface.find_entities_filtered({
      name = { ENDPOINT_ENTITY_NAMES.source, ENDPOINT_ENTITY_NAMES.sink },
    })) do
      register_endpoint(entity)
    end
  end
end)
