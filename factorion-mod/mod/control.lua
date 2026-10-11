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
local inference = require("inference")

local PROTOCOL_VERSION = "9"
-- The server polls every 250 ms, so 5 s of silence means it isn't there.
local SERVER_TIMEOUT_TICKS = 5 * 60
-- Loading a checkpoint (often a W&B download) blocks the server's polls.
local MODEL_LOAD_TIMEOUT_TICKS = 60 * 60
local STOP_REASONS = {
  eot = "the model chose to stop",
  max_steps = "hit the %s-step limit",
  placement_error = "Factorio refused a placement",
  server_error = "the server hit an error",
}

local SETTING_DEFAULTS = {}
for _, spec in pairs(require("settings_spec")) do
  SETTING_DEFAULTS[spec.name] = spec.default_value
end

-- Settings load only when Factorio starts, but this file reloads on every
-- re-host, so a running game may not have a newly added setting yet.
local function setting(name)
  local current = settings.global[name]
  if current then return current.value end
  return SETTING_DEFAULTS[name]
end

-- Sampling settings named in the request message when not at their default.
local SAMPLING_SUMMARY = {
  { "temperature", "temperature %g" },
  { "top_k", "top-k %d" },
  { "top_p", "top-p %g" },
  { "stop_rule", "stop rule: %s" },
  { "stop_threshold", "stop threshold %g" },
  { "max_steps", "max %d steps" },
  { "seed", "seed %d" },
}

local function get_default_item(role)
  return setting("factorion-default-" .. role .. "-item")
end

local function get_checkpoint()
  return inference.config().checkpoint
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

-- ----------------------------------------------------------------------------
-- player onboarding
-- ----------------------------------------------------------------------------

local function show_startup_message(player)
  player.print("[Factorion] Ready. Craft a [item=factorion-planner], click a " ..
    "tile with it to choose what to make, then click a tile for each source " ..
    "and press CTRL+P. Right-click with it to clear the plan. Source and sink " ..
    "belts placed by hand (map editor) work too: press CTRL+P near them.")
end

script.on_event(defines.events.on_player_created, function(event)
  ensure_player_state(event.player_index)
  ensure_pending_lookup()
end)

script.on_event(defines.events.on_player_joined_game, function(event)
  local player = game.get_player(event.player_index)
  if player then show_startup_message(player) end
  -- Re-announce the server's state (after the grace period) to whoever joined.
  if storage.server_connected == false then
    storage.server_connected = nil
    storage.server_watch_tick = game.tick
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
local SAVE_MARKER = "factorion-save-marker"
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
local SINK_LABEL_REFRESH_TICKS = 30

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
  config.item = item or config.item or get_default_item(role)
  storage.endpoints[entity.unit_number] = config
  if role == "sink" then
    if item_changed or not config.throughput_rate then
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

-- The server loads the checkpoint and reports its grid size via model_loaded;
-- until then storage.model is nil and no region can be planned.
local function request_model_load()
  local checkpoint = get_checkpoint()
  if storage.model_loading == checkpoint then return end
  storage.model = nil
  storage.model_loading = checkpoint
  storage.outbox = storage.outbox or {}
  table.insert(storage.outbox, helpers.table_to_json({
    type = "load_model", checkpoint = checkpoint,
  }))
  game.print("[Factorion] Asking the model server to load " .. checkpoint .. "…")
end

-- Ghosts vanish at once. In ghost mode built entities are marked for
-- deconstruction so robots swap them for the ghost placed on top; otherwise
-- they go now, with the player's own mined back into their inventory.
local function remove_entity(entity, player, is_predicted)
  if entity.type == "entity-ghost" then
    entity.destroy({ raise_destroy = true })
  elseif setting("factorion-placement-mode") == "ghost" then
    entity.order_deconstruction(entity.force, player)
  elseif is_predicted then
    entity.destroy({ raise_destroy = true })
  else
    player.mine_entity(entity, true)
  end
end

local function clear_predicted_entities(state, player)
  for _, unit_number in ipairs(state.predicted_entities or {}) do
    local entity = game.get_entity_by_unit_number(unit_number)
    if entity and entity.valid then remove_entity(entity, player, true) end
  end
  state.predicted_entities = {}
end

local function cancel_pending(state)
  if not state.pending then return end
  local request_id = state.pending.request_id
  if storage.pending_by_request then
    storage.pending_by_request[request_id] = nil
  end
  -- A request the server hasn't picked up yet would otherwise be served later
  -- against a request_id the mod no longer knows.
  local outbox = storage.outbox or {}
  for i = #outbox, 1, -1 do
    local queued = helpers.json_to_table(outbox[i])
    if queued and queued.request_id == request_id then
      table.remove(outbox, i)
    end
  end
  state.pending = nil
end

local function close_marker_dialog(player, state)
  local frame = player.gui.screen[MARKER_DIALOG]
  if frame then frame.destroy() end
  state.picker = nil
end

local function open_marker_dialog(player, state, role, entity)
  close_marker_dialog(player, state)
  state.picker = {
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
    elem_type = "item", item = config and config.item or get_default_item(role),
  })
  local actions = frame.add({ type = "flow", direction = "horizontal" })
  actions.add({
    type = "button", name = SAVE_MARKER, caption = "Save item",
    style = "confirm_button",
  })
  player.opened = frame
end

-- ----------------------------------------------------------------------------
-- planner: click a tile and pick the sink item, then click one tile per
-- ingredient source; the region is the model's grid centred on those belts
-- ----------------------------------------------------------------------------

local PLANNER = "factorion-planner"
local PLAN_DIALOG = "factorion-plan-dialog"
local PLAN_ITEM = "factorion-plan-item"

-- The recipe an assembling machine 1 makes item with, preferring the one named
-- after it.
local function recipe_for(item)
  local categories = prototypes.entity["assembling-machine-1"].crafting_categories
  local fallback
  for name, recipe in pairs(prototypes.recipe) do
    if not recipe.hidden and categories[recipe.category] then
      for _, product in pairs(recipe.products) do
        if product.type == "item" and product.name == item then
          if name == item then return recipe end
          fallback = fallback or recipe
        end
      end
    end
  end
  return fallback
end

-- One source per ingredient item, in recipe order; nil plus a reason when
-- belts can't supply it.
local function source_items(item)
  local recipe = recipe_for(item)
  if not recipe then
    return nil, "no recipe an assembling machine 1 can craft makes [item=" .. item .. "]"
  end
  local items = {}
  for _, ingredient in pairs(recipe.ingredients) do
    if ingredient.type ~= "item" then
      return nil, "[recipe=" .. recipe.name .. "] needs a fluid, which belts can't supply"
    end
    table.insert(items, ingredient.name)
  end
  return items
end

local function remove_plan_markers(state)
  for _, unit_number in ipairs(state.plan and state.plan.markers or {}) do
    local entity = game.get_entity_by_unit_number(unit_number)
    if entity and entity.valid then entity.destroy({ raise_destroy = true }) end
  end
  state.plan = nil
end

local PANEL = "factorion-panel"
local NUDGES = { left = { -1, 0 }, up = { 0, -1 }, down = { 0, 1 }, right = { 1, 0 } }

-- Source and sink belts in area, read from the world so belts moved or copied
-- by any means are seen where they are now.
local function endpoints_in(surface, area)
  local sources, sinks = {}, {}
  for _, entity in pairs(surface.find_entities_filtered({
    area = area,
    name = { ENDPOINT_ENTITY_NAMES.source, ENDPOINT_ENTITY_NAMES.sink },
  })) do
    local tile = {
      x = math.floor(entity.position.x), y = math.floor(entity.position.y),
      entity = entity,
    }
    table.insert(endpoint_role(entity) == "source" and sources or sinks, tile)
  end
  return sources, sinks
end

local function inside(tile, fp)
  return tile.x >= fp.x and tile.y >= fp.y
    and tile.x < fp.x + fp.w and tile.y < fp.y + fp.h
end

local function close_panel(player)
  local frame = player and player.gui.screen[PANEL]
  if frame then frame.destroy() end
end

local function open_panel(player)
  close_panel(player)
  local frame = player.gui.screen.add({
    type = "frame", name = PANEL, caption = "Factorion", direction = "vertical",
  })
  frame.location = { 20, 200 }
  return frame
end

local function show_source_panel(player, plan)
  local item = plan.sources[plan.next]
  local row = open_panel(player).add({ type = "flow", direction = "horizontal" })
  row.style.vertical_align = "center"
  row.add({ type = "sprite", sprite = "item/" .. item })
  row.add({
    type = "label",
    caption = { "", string.format("Place source %d of %d: ", plan.next, #plan.sources),
      prototypes.item[item].localised_name },
  })
end

-- The region must stay the model's grid size, so editing it means moving it.
local function open_region_panel(player)
  local frame = open_panel(player)
  frame.add({ type = "label", caption = "Press CTRL+P to build. Move the region:" })
  local pad = frame.add({ type = "table", column_count = 3 })
  for _, cell in ipairs({ "", "up", "", "left", "", "right", "", "down", "" }) do
    if cell == "" then
      pad.add({ type = "empty-widget" })
    else
      pad.add({
        type = "button", caption = cell:sub(1, 1):upper() .. cell:sub(2),
        tags = { factorion_region = cell },
      })
    end
  end
end

local function nudge_region(player, state, direction)
  local fp = state.footprint
  if not fp then return end
  local step = NUDGES[direction]
  local moved = { x = fp.x + step[1], y = fp.y + step[2], w = fp.w, h = fp.h }
  local sources, sinks = endpoints_in(player.surface,
    { { fp.x, fp.y }, { fp.x + fp.w, fp.y + fp.h } })
  for _, tile in ipairs(sources) do table.insert(sinks, tile) end
  for _, tile in ipairs(sinks) do
    if not inside(tile, moved) then
      player.print("[Factorion] Can't move the region that way: a source or " ..
        "sink belt would end up outside it.")
      return
    end
  end
  state.footprint = moved
  draw_footprint(player, state)
end

local function clear_region(player, state)
  cancel_pending(state)
  destroy_footprint_render(state)
  clear_predicted_entities(state, player)
  close_panel(player)
  state.footprint = nil
end

local function place_plan_marker(player, plan, role, item, tile)
  local name = ENDPOINT_ENTITY_NAMES[role]
  local position = { tile.x + 0.5, tile.y + 0.5 }
  if not player.surface.can_place_entity({
        name = name, position = position, force = player.force }) then
    return false
  end
  local entity = player.surface.create_entity({
    name = name, position = position, force = player.force,
  })
  if not entity then return false end
  register_endpoint(entity, item)
  table.insert(plan.markers, entity.unit_number)
  table.insert(plan.tiles, { x = tile.x, y = tile.y, role = role, entity = entity })
  return true
end

local function close_plan_dialog(player)
  local frame = player.gui.screen[PLAN_DIALOG]
  if frame then frame.destroy() end
end

-- The planner leaves the cursor first, or clicking the item picker would drop
-- it into the picker.
local function open_plan_dialog(player)
  close_plan_dialog(player)
  player.clear_cursor()
  local frame = player.gui.screen.add({
    type = "frame", name = PLAN_DIALOG, caption = "Plan a factory",
    direction = "vertical",
  })
  frame.auto_center = true
  frame.add({ type = "label", caption = "What should the factory make?" })
  frame.add({
    type = "choose-elem-button", name = PLAN_ITEM, elem_type = "item",
    item = setting("factorion-default-sink-item"),
  })
  frame.add({
    type = "button", caption = "Place sink", style = "confirm_button",
    tags = { factorion_plan = "place_sink" },
  })
  player.opened = frame
end

-- Puts one of name in the player's hand, from their inventory if they have it.
local function hand_item(player, name)
  local cursor = player.cursor_stack
  if not cursor then return end
  if cursor.valid_for_read and cursor.name == name then return end
  if not player.clear_cursor() then return end
  local inventory = player.get_main_inventory()
  local stack = inventory and inventory.find_item_stack(name)
  if stack then cursor.swap_stack(stack) else cursor.set_stack({ name = name }) end
end

-- Exactly one fresh source belt per source, never a stack from the inventory.
local function hand_source_belt(player)
  local cursor = player.cursor_stack
  if not cursor then return end
  local name = ENDPOINT_ENTITY_NAMES.source
  if not (cursor.valid_for_read and cursor.name == name) and not player.clear_cursor() then
    return
  end
  cursor.set_stack({ name = name, count = 1 })
end

local function prompt_next_source(player, plan)
  hand_source_belt(player)
  show_source_panel(player, plan)
  player.print(string.format(
    "[Factorion] Place source %d of %d: [item=%s] (rotate it with R first if needed)",
    plan.next, #plan.sources, plan.sources[plan.next]))
end

-- The sink is placed for the player, so it is turned to face out of the
-- region as in training; sources keep the rotation they were placed with.
local function orient_sink(plan, fp)
  local cx, cy = fp.x + fp.w / 2, fp.y + fp.h / 2
  for _, marker in ipairs(plan.tiles) do
    if marker.role == "sink" and marker.entity.valid then
      local dx, dy = marker.x + 0.5 - cx, marker.y + 0.5 - cy
      if math.abs(dx) >= math.abs(dy) then
        marker.entity.direction = dx >= 0 and defines.direction.east or defines.direction.west
      else
        marker.entity.direction = dy >= 0 and defines.direction.south or defines.direction.north
      end
    end
  end
end

-- Centres the model's grid on tiles, or says why it can't.
local function set_region(player, state, tiles)
  local size = storage.model.grid_size
  local min_x, min_y, max_x, max_y = math.huge, math.huge, -math.huge, -math.huge
  for _, tile in ipairs(tiles) do
    min_x, max_x = math.min(min_x, tile.x), math.max(max_x, tile.x)
    min_y, max_y = math.min(min_y, tile.y), math.max(max_y, tile.y)
  end
  local span_x, span_y = max_x - min_x + 1, max_y - min_y + 1
  if span_x > size or span_y > size then
    player.print(string.format(
      "[Factorion] These belts span %dx%d tiles, more than the model's %dx%d " ..
      "grid.", span_x, span_y, size, size))
    return false
  end
  state.footprint = {
    x = min_x - math.floor((size - span_x) / 2),
    y = min_y - math.floor((size - span_y) / 2),
    w = size, h = size,
  }
  draw_footprint(player, state)
  open_region_panel(player)
  return true
end

-- Belts the player placed themselves (plan.existing) count towards the region
-- but keep the direction they were given.
local function finish_plan(player, state)
  local plan = state.plan
  plan.done = true
  local tiles = {}
  for _, tile in ipairs(plan.tiles) do table.insert(tiles, tile) end
  for _, tile in ipairs(plan.existing or {}) do table.insert(tiles, tile) end
  if not set_region(player, state, tiles) then return end
  orient_sink(plan, state.footprint)
  player.print("[Factorion] Region ready. Press CTRL+P to build the factory.")
end

-- A source belt the player places while a plan is waiting for sources becomes
-- the next one; their rotation is kept.
local function plan_source_built(player_index, entity)
  local state = ensure_player_state(player_index)
  local plan = state.plan
  if not (plan and plan.sources and not plan.done) then return end
  local player = game.get_player(player_index)
  register_endpoint(entity, plan.sources[plan.next])
  table.insert(plan.markers, entity.unit_number)
  table.insert(plan.tiles, {
    x = math.floor(entity.position.x), y = math.floor(entity.position.y),
    role = "source", entity = entity,
  })
  plan.next = plan.next + 1
  if plan.next > #plan.sources then
    finish_plan(player, state)
  else
    prompt_next_source(player, plan)
  end
end

local function place_sink(player, state)
  local frame = player.gui.screen[PLAN_DIALOG]
  local item = frame and frame[PLAN_ITEM].elem_value
  local plan = state.plan
  if not item or not plan then return end
  local sources, reason
  if not plan.keep_sources then
    sources, reason = source_items(item)
    if not sources then
      player.print("[Factorion] Can't plan that: " .. reason .. ".")
      return
    end
  end
  close_plan_dialog(player)
  if not place_plan_marker(player, plan, "sink", item, plan.sink_tile) then
    player.print("[Factorion] That tile is blocked; click another one.")
    state.plan = nil
    return
  end
  if plan.keep_sources then
    finish_plan(player, state)
    return
  end
  plan.sources, plan.next = sources, 1
  prompt_next_source(player, plan)
end


local function handle_tool_selection(event)
  if event.item ~= PLANNER then return end
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  if not player then return end
  if state.pending then
    player.print("[Factorion] Wait for the current prediction, or right-click with the planner to cancel it.")
    return
  end
  local x, y = area_center_tile(event.area)
  local tile = { x = x, y = y }
  local plan = state.plan

  if plan and plan.awaiting_sink then
    plan.awaiting_sink, plan.sink_tile = nil, tile
    open_plan_dialog(player)
    return
  end

  if not storage.model then
    request_model_load()
    player.print("[Factorion] No model loaded yet; try again once the server reports in.")
    return
  end
  remove_plan_markers(state)
  clear_region(player, state)
  state.plan = { sink_tile = tile, markers = {}, tiles = {} }
  open_plan_dialog(player)
end

script.on_event(defines.events.on_player_selected_area, handle_tool_selection)

-- Right-clicking with the planner cancels the plan and clears its region.
script.on_event(defines.events.on_player_alt_selected_area, function(event)
  if event.item ~= PLANNER then return end
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  remove_plan_markers(state)
  clear_region(player, state)
  if player then
    close_plan_dialog(player)
    player.print("[Factorion] Plan cleared.")
  end
end)

local function save_marker_dialog(player, state)
  if not state.picker then
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
  if inference.on_gui_click(event) then return end
  local nudge = event.element.valid and event.element.tags.factorion_region
  if nudge then
    local player = game.get_player(event.player_index)
    if player then nudge_region(player, ensure_player_state(event.player_index), nudge) end
    return
  end
  if event.element.valid and event.element.tags.factorion_plan == "place_sink" then
    local player = game.get_player(event.player_index)
    if player then place_sink(player, ensure_player_state(event.player_index)) end
    return
  end
  local element = event.element
  if not element or not element.valid or element.name ~= SAVE_MARKER then return end
  local player = game.get_player(event.player_index)
  if not player then return end
  save_marker_dialog(player, ensure_player_state(event.player_index))
end)

script.on_event("factorion-confirm-dialog", function(event)
  local player = game.get_player(event.player_index)
  if not player then return end
  local state = ensure_player_state(event.player_index)
  if player.gui.screen[PLAN_DIALOG] then
    place_sink(player, state)
    return
  end
  if not state.picker or not player.gui.screen[MARKER_DIALOG] then return end
  save_marker_dialog(player, state)
end)

script.on_event(defines.events.on_gui_text_changed, inference.on_gui_text_changed)
script.on_event(defines.events.on_gui_confirmed, inference.on_gui_confirmed)
script.on_event(defines.events.on_gui_selection_state_changed,
  inference.on_gui_selection_state_changed)

-- The server keeps its built-in window only so clicking it raises this event;
-- the shared inference window replaces it straight away.
local function open_gpu_server(player, entity)
  if not entity or not entity.valid or entity.name ~= inference.GPU_SERVER then
    return false
  end
  inference.open(player)
  return true
end

script.on_event(defines.events.on_gui_closed, function(event)
  if inference.on_gui_closed(event) then return end
  if event.element and event.element.valid and event.element.name == PLAN_DIALOG then
    event.element.destroy()
    local state = ensure_player_state(event.player_index)
    if state.plan and not state.plan.sources then state.plan = nil end
    return
  end
  if not event.element or not event.element.valid
      or event.element.name ~= MARKER_DIALOG then return end
  ensure_player_state(event.player_index).picker = nil
end)

script.on_event(defines.events.on_gui_opened, function(event)
  local entity = event.entity
  local player = game.get_player(event.player_index)
  if not player then return end
  if open_gpu_server(player, entity) then return end
  local role = endpoint_role(entity)
  if not role then return end
  player.opened = nil
  register_endpoint(entity)
  open_marker_dialog(player, ensure_player_state(event.player_index), role, entity)
end)

local function entity_built(event)
  local entity = event.entity or event.created_entity
  -- A built model ghost keeps counting as the model's, so rerun/reset finds it.
  local owner = event.tags and event.tags.factorion_player
  if owner and entity and entity.valid and entity.unit_number then
    table.insert(ensure_player_state(owner).predicted_entities, entity.unit_number)
  end
  local role = endpoint_role(entity)
  if not role then return end
  local item = event.tags and event.tags.factorion_item or nil
  register_endpoint(entity, item)
  if role == "source" and event.player_index then
    plan_source_built(event.player_index, entity)
  end
end

script.on_event(defines.events.on_built_entity, entity_built)
script.on_event(defines.events.on_robot_built_entity, entity_built)
script.on_event(defines.events.script_raised_built, entity_built)
script.on_event(defines.events.script_raised_revive, entity_built)

script.on_event(defines.events.on_entity_cloned, function(event)
  local source = endpoint_config(event.source)
  if endpoint_role(event.destination) then
    register_endpoint(event.destination, source and source.item or nil)
  end
end)

script.on_event(defines.events.on_entity_settings_pasted, function(event)
  local source = endpoint_config(event.source)
  if source and endpoint_role(event.destination) then
    register_endpoint(event.destination, source.item)
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

local function service_endpoint_belts()
  -- Per-tick exponential moving average; the setting is its time constant.
  local decay = math.exp(
    -1 / (60 * setting("factorion-sink-smoothing")))
  for unit_number, config in pairs(storage.endpoints or {}) do
    local entity = config.entity
    if not entity or not endpoint_role(entity) then
      destroy_endpoint_alt_icon(config)
      storage.endpoints[unit_number] = nil
    elseif config.role == "source" then
      feed_source(config, entity)
    else
      config.throughput_rate = (config.throughput_rate or 0) * decay
        + drain_sink(config, entity) * 60 * (1 - decay)
      if game.tick % SINK_LABEL_REFRESH_TICKS == 0 then
        refresh_sink_rate_label(entity, config)
      end
    end
  end
end

-- A new checkpoint may have a different grid size, so planned regions go.
inference.on_load = function()
  for player_index, state in pairs(storage.players or {}) do
    local player = game.get_player(player_index)
    if state.footprint and player and player.connected then
      player.print("[Factorion] Checkpoint changed; plan your factory again.")
    end
    destroy_footprint_render(state)
    state.footprint = nil
  end
  storage.model_loading = nil
  request_model_load()
end

-- ----------------------------------------------------------------------------
-- execute: write a request JSON the server picks up
-- ----------------------------------------------------------------------------

local function new_request_id()
  return string.format("%d-%d-%d", game.tick,
    math.random(0, 2^30), math.random(0, 2^30))
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
    if not predicted[entity.unit_number] and not entity.to_be_deconstructed() then
      local direction = entity.type == "assembling-machine"
        and 0 or FACTORIO_TO_DIRECTION[entity.direction]
      if direction then
        -- Factorio stores an inserter's drop direction; the model stores its
        -- pickup direction.
        if entity.type == "inserter" then
          direction = OPPOSITE_DIRECTION[direction]
        end

        local width = entity.prototype.tile_width
        local height = entity.prototype.tile_height
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
          if entity.type == "assembling-machine" then
            local recipe = entity.get_recipe()
            if recipe then entry.item = recipe.name end
          end
          if entity.type == "underground-belt" then
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
    local rx, ry = x - fp.x, y - fp.y
    if rx >= 0 and rx < fp.w and ry >= 0 and ry < fp.h then
      local config = register_endpoint(entity)
      if not config then goto continue end
      local entry = {
        x = rx, y = ry,
        direction = FACTORIO_TO_DIRECTION[entity.direction],
        item = config.item,
      }
      if config.role == "source" then
        table.insert(sources, entry)
      else
        table.insert(sinks, entry)
      end
    end
    ::continue::
  end

  local function icons(endpoints)
    local parts = {}
    for _, endpoint in ipairs(endpoints) do
      table.insert(parts, "[item=" .. endpoint.item .. "]")
    end
    return table.concat(parts, " ")
  end
  local description = string.format("converting %s to %s",
    icons(sources), icons(sinks))
  if #model_entities > 0 then
    description = description .. string.format(
      " around %d existing entities", #model_entities)
  end

  local config = inference.config()
  local request = {
    type          = "predict",
    request_id    = new_request_id(),
    checkpoint    = storage.model.checkpoint,
    grid_size     = fp.w,
    placement_delay_s = setting("factorion-placement-delay"),
    temperature   = config.temperature,
    top_k         = config.top_k,
    top_p         = config.top_p,
    stop_rule     = config.stop_rule,
    stop_threshold = config.stop_threshold,
    max_steps     = config.max_steps,
    seed          = config.seed,
    entities      = model_entities,
    sources       = sources,
    sinks         = sinks,
  }
  return request, description
end

-- Build a request from the player's configured endpoint belts and enqueue it
-- for the server. Returns
-- (true, message) on success, (false, message) on a precheck failure.
local function try_request_prediction(player_index)
  local state = ensure_player_state(player_index)
  ensure_pending_lookup()
  if not state.footprint then
    return false, "No footprint set."
  end
  if state.pending then
    return false, "A prediction is already in progress; right-click with the planner to cancel it."
  end
  local no_gpu = inference.unavailable_reason()
  if no_gpu then return false, no_gpu end
  if not storage.model then
    request_model_load()
    return false, "No model loaded yet."
  end
  if storage.model.grid_size ~= state.footprint.w then
    return false, "This region was planned for another model; plan it again."
  end
  local request, description = gather_request(state, player_index)
  if #request.sources == 0 then
    return false, "No source belt inside the region."
  end
  if #request.sinks == 0 then
    return false, "No sink belt inside the region."
  end
  clear_predicted_entities(state, game.get_player(player_index))
  local request_json = helpers.table_to_json(request)
  table.insert(storage.outbox, request_json)
  storage.pending_by_request[request.request_id] = player_index
  state.pending = {
    request_id = request.request_id, tick = game.tick, description = description,
  }
  log("[Factorion] enqueued request " .. request.request_id ..
      " json=" .. request_json)
  local changed = {}
  for _, knob in ipairs(SAMPLING_SUMMARY) do
    local value = inference.config()[knob[1]]
    if value ~= inference.DEFAULTS[knob[1]] then
      table.insert(changed, string.format(knob[2], value))
    end
  end
  return true, string.format("Requesting a factory %s using checkpoint %s%s.",
    description, request.checkpoint,
    #changed > 0 and " (" .. table.concat(changed, ", ") .. ")" or "")
end

local function describe_placement_failure(surface, fp, placement)
  local left = fp.x + placement.tile_x
  local top = fp.y + placement.tile_y
  local blockers = {}
  local found = surface.find_entities({
    { left, top },
    { left + placement.width, top + placement.height },
  })
  for _, entity in pairs(found) do
    table.insert(blockers, entity.name)
  end
  table.sort(blockers)
  return string.format(
    "error: Factorio refused %s at relative tile (%s,%s), " ..
    "world tile (%d,%d) on %s; blockers=[%s]",
    tostring(placement.name),
    tostring(placement.tile_x),
    tostring(placement.tile_y),
    left,
    top,
    surface.name,
    #blockers > 0 and table.concat(blockers, ", ") or "none")
end

-- Mirrors the model's overwrite rule: every entity the placement touches is
-- removed whole. The model never saw entities crossing the region boundary,
-- so those are left alone and block the placement instead.
local function clear_placement_area(player, state, fp, placement)
  local left = fp.x + placement.tile_x
  local top = fp.y + placement.tile_y
  local predicted = {}
  for _, unit_number in ipairs(state.predicted_entities or {}) do
    predicted[unit_number] = true
  end
  local area = {
    { left + 0.01, top + 0.01 },
    { left + placement.width - 0.01, top + placement.height - 0.01 },
  }
  local overlapped = player.surface.find_entities_filtered({
    area = area, name = MODEL_ENTITY_NAMES,
  })
  for _, ghost in pairs(player.surface.find_entities_filtered({
    area = area, ghost_name = MODEL_ENTITY_NAMES,
  })) do
    table.insert(overlapped, ghost)
  end
  for _, entity in pairs(overlapped) do
    local box = entity.bounding_box
    if box.left_top.x >= fp.x and box.left_top.y >= fp.y
        and box.right_bottom.x <= fp.x + fp.w
        and box.right_bottom.y <= fp.y + fp.h then
      remove_entity(entity, player, predicted[entity.unit_number])
    end
  end
end

-- ----------------------------------------------------------------------------
-- server health: the mod can't connect out, so it watches for the server's
-- polls and explains in chat why they aren't arriving
-- ----------------------------------------------------------------------------

local function recently(tick)
  return tick ~= nil and game.tick - tick < SERVER_TIMEOUT_TICKS
end

-- Every call the server makes counts: it doesn't poll while it is building.
local function heard_from_server()
  storage.server_contact_tick = game.tick
end

local function server_alive()
  local timeout = storage.model_loading and MODEL_LOAD_TIMEOUT_TICKS
    or SERVER_TIMEOUT_TICKS
  return storage.server_contact_tick ~= nil
    and game.tick - storage.server_contact_tick < timeout
end

-- game.is_multiplayer() is false for a GUI host playing alone, even with
-- RCON open, so it can't tell a hosted game from single-player.
local function server_help()
  if recently(storage.server_seen_tick) then
    return string.format(
      "A model server is reaching the game, but it speaks protocol %s and " ..
      "this mod speaks %s. Whichever side is older needs restarting: quit " ..
      "and re-host Factorio to load the latest mod, or restart the server " ..
      "with ./start-mod.sh.",
      tostring(storage.server_protocol or "unknown"), PROTOCOL_VERSION)
  end
  return "Start it from the factorion repo with ./start-mod.sh. Factorio only " ..
    "opens RCON while hosting (Play > Multiplayer > Host), and its config.ini " ..
    "needs local-rcon-socket=127.0.0.1:64502 and " ..
    "local-rcon-password=<password> under [other]; restart Factorio after " ..
    "editing it."
end

local function check_server()
  local connected = server_alive()
  if connected == storage.server_connected then return end
  -- Give a freshly started game time for the server's first poll.
  if storage.server_connected == nil and not connected
      and recently(storage.server_watch_tick) then
    return
  end
  storage.server_connected = connected
  if connected then
    game.print("[Factorion] Model server connected.")
  else
    game.print("[Factorion] Can't reach the model server. " .. server_help())
  end
end

script.on_nth_tick(60, function()
  check_server()
  inference.refresh_all()
end)

local function endpoints_near(player, position)
  local size = storage.model.grid_size
  return endpoints_in(player.surface, {
    { position.x - size, position.y - size },
    { position.x + size, position.y + size },
  })
end

-- Re-reads the belts around the region: it stays put (keeping any nudges)
-- while they all fit inside it, and re-centres on them once one is moved out.
local function refresh_region(player, state)
  local fp = state.footprint
  local sources, sinks = endpoints_in(player.surface, {
    { fp.x - fp.w, fp.y - fp.h }, { fp.x + 2 * fp.w, fp.y + 2 * fp.h },
  })
  local tiles = {}
  for _, tile in ipairs(sources) do table.insert(tiles, tile) end
  for _, tile in ipairs(sinks) do table.insert(tiles, tile) end
  for _, tile in ipairs(tiles) do
    if not inside(tile, fp) then
      if not set_region(player, state, tiles) then return false end
      player.print("[Factorion] Belts have moved; the region is re-centred on them.")
      return true
    end
  end
  return true
end

-- Ingredients of every sink's recipe, each once, in recipe order.
local function sources_for_sinks(sinks)
  local items, seen = {}, {}
  for _, sink in ipairs(sinks) do
    local config = register_endpoint(sink.entity)
    local sources, reason = source_items(config.item)
    if not sources then return nil, reason end
    for _, item in ipairs(sources) do
      if not seen[item] then
        seen[item] = true
        table.insert(items, item)
      end
    end
  end
  return items
end

-- CTRL+P builds the planned region; without one it uses the source and sink
-- belts near the cursor, and guides the player to place whichever is missing.
local function predict_or_guide(player, state, cursor_position)
  local plan = state.plan
  if plan and not plan.done then
    if plan.sources then
      prompt_next_source(player, plan)
      return nil
    end
    hand_item(player, PLANNER)
    return "Click a tile with the planner to place the sink."
  end
  if state.footprint then
    if not refresh_region(player, state) then return nil end
    local _, msg = try_request_prediction(player.index)
    return msg
  end
  if not storage.model then
    request_model_load()
    return "No model loaded yet."
  end
  local sources, sinks = endpoints_near(player, cursor_position)
  if #sinks == 0 then
    state.plan = {
      awaiting_sink = true, keep_sources = #sources > 0, existing = sources,
      markers = {}, tiles = {},
    }
    hand_item(player, PLANNER)
    return "No sink near the cursor. Click a tile with the planner to place one."
  end
  if #sources == 0 then
    local items, reason = sources_for_sinks(sinks)
    if not items then return "Can't work out the sources: " .. reason .. "." end
    state.plan = {
      existing = sinks, sources = items, next = 1, markers = {}, tiles = {},
    }
    player.print("[Factorion] No source near the cursor.")
    prompt_next_source(player, state.plan)
    return nil
  end
  local tiles = {}
  for _, tile in ipairs(sources) do table.insert(tiles, tile) end
  for _, tile in ipairs(sinks) do table.insert(tiles, tile) end
  if not set_region(player, state, tiles) then return nil end
  local _, msg = try_request_prediction(player.index)
  return msg
end

script.on_event("factorion-execute", function(event)
  local player = game.get_player(event.player_index)
  if not player then return end
  local msg = predict_or_guide(
    player, ensure_player_state(event.player_index), event.cursor_position)
  if msg then player.print("[Factorion] " .. msg) end
  if not server_alive() then
    player.print("[Factorion] No model server is connected, so nothing " ..
      "will happen yet. " .. server_help())
  end
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
  poll_request = function(server_protocol)
    storage.server_seen_tick = game.tick
    storage.server_protocol = server_protocol
    if server_protocol ~= PROTOCOL_VERSION then
      return helpers.table_to_json({
        type = "protocol_mismatch", mod_protocol = PROTOCOL_VERSION,
      })
    end
    heard_from_server()
    storage.outbox = storage.outbox or {}
    if #storage.outbox == 0 then return "" end
    return table.remove(storage.outbox, 1)
  end,

  -- Create one model-predicted entity at footprint-relative coordinates,
  -- replacing whatever it overlaps. A placement without a name only deletes.
  place_prediction = function(request_id, placement_json)
    heard_from_server()
    ensure_pending_lookup()
    local player_index = storage.pending_by_request[request_id]
    if not player_index then
      return "cancelled"
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
    if not placement.tile_x or not placement.tile_y
        or not placement.width or not placement.height then
      return "error: incomplete placement"
    end
    if placement.tile_x < 0 or placement.tile_y < 0
        or placement.tile_x + placement.width > fp.w
        or placement.tile_y + placement.height > fp.h then
      return "error: placement lies outside the footprint"
    end

    clear_placement_area(player, state, fp, placement)
    if not placement.name then return "ok" end

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
    if setting("factorion-placement-mode") == "ghost" then
      params.inner_name = params.name
      params.name = "entity-ghost"
      params.tags = { factorion_player = player_index }
    end
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
      return describe_placement_failure(player.surface, fp, placement)
    end

    state.predicted_entities = state.predicted_entities or {}
    if entity.unit_number then
      table.insert(state.predicted_entities, entity.unit_number)
    end
    return "ok"
  end,

  -- Complete the streamed request and release its pending state.
  finish_prediction = function(request_id, summary_json)
    heard_from_server()
    ensure_pending_lookup()
    local player_index = storage.pending_by_request[request_id]
    if not player_index then
      return "cancelled"
    end
    storage.pending_by_request[request_id] = nil
    if player_index == 0 then
      log("[Factorion] (headless) prediction complete for " ..
        tostring(request_id) .. ": " .. tostring(summary_json))
      return "ok"
    end

    local player = game.get_player(player_index)
    local state = ensure_player_state(player_index)
    local description = state.pending and state.pending.description
      or "for an unknown request"
    state.pending = nil
    local ok, summary = pcall(
      function() return helpers.json_to_table(summary_json) end)
    summary = ok and summary or {}
    if player then
      local stop = STOP_REASONS[summary.stop_reason]
        or tostring(summary.stop_reason)
      player.print(string.format(
        "[Factorion] Finished %s: %s steps, %s invalid; %s. Seed %s.",
        description,
        tostring(summary.steps_taken or "?"),
        tostring(summary.invalid_steps or "?"),
        stop:format(summary.steps_taken or "?"),
        tostring(summary.seed or "?")))
      if summary.error then
        player.print("[Factorion] Server error: " .. tostring(summary.error))
      end
    end
    return "ok"
  end,

  -- Replies to a load_model message. A reply for a checkpoint the setting has
  -- since moved away from is ignored; the newer load is already queued.
  model_loaded = function(model_json)
    heard_from_server()
    local model = helpers.json_to_table(model_json)
    if not model or model.checkpoint ~= get_checkpoint() then return "stale" end
    storage.model = model
    storage.model_loading = nil
    storage.model_error = nil
    inference.refresh_all()
    game.print(string.format(
      "[Factorion] Model %s (%s) ready: %dx%d region. Plan a factory with " ..
      "the [item=factorion-planner].",
      model.checkpoint, tostring(model.name), model.grid_size, model.grid_size))
    return "ok"
  end,
  model_failed = function(error_json)
    heard_from_server()
    local failure = helpers.json_to_table(error_json) or {}
    if failure.checkpoint ~= get_checkpoint() then return "stale" end
    storage.model_loading = nil
    storage.model_error = tostring(failure.error)
    inference.refresh_all()
    game.print(string.format(
      "[Factorion] Could not load checkpoint %s: %s",
      tostring(failure.checkpoint), tostring(failure.error)))
    return "ok"
  end,

  -- Diagnostic: server can call this to check the mod is alive.
  ping = function()
    return "factorion-mod alive at tick " .. tostring(game.tick)
  end,
  protocol_version = function()
    storage.server_seen_tick = game.tick
    return PROTOCOL_VERSION
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
  storage.endpoints = {}
  storage.server_watch_tick = game.tick
  request_model_load()
end)

script.on_configuration_changed(function()
  storage.players = storage.players or {}
  storage.pending_by_request = storage.pending_by_request or {}
  storage.outbox = storage.outbox or {}
  storage.endpoints = storage.endpoints or {}
  for _, player in pairs(game.players) do
    local state = ensure_player_state(player.index)
    state.predicted_entities = state.predicted_entities or {}
    if state.footprint then draw_footprint(player, state) end
  end
  for _, surface in pairs(game.surfaces) do
    for _, entity in pairs(surface.find_entities_filtered({
      name = { ENDPOINT_ENTITY_NAMES.source, ENDPOINT_ENTITY_NAMES.sink },
    })) do
      register_endpoint(entity)
    end
  end
  storage.model_loading = nil
  storage.server_connected = nil
  storage.server_watch_tick = game.tick
  request_model_load()
end)
