-- Factorion mod runtime.
--
-- State per player lives in `storage.players[player_index]`:
--   footprint = { x=int, y=int, w=int, h=int }   world-tile bbox
--   sources/sinks = { { x, y, item, direction, entity_unit_number,
--                       render_ids }, ... }
--   pending   = { request_id=string, tick=int }  in-flight request, if any
--
-- Round trip (single-channel RCON, both directions):
--   1. On key `factorion-execute`, we build a request JSON describing the
--      footprint + sources + sinks (in footprint-relative coords) and
--      enqueue it on `storage.outbox`. The Python server is polling our
--      `poll_request` remote interface over RCON; when there's something
--      in the queue it gets popped and returned as a JSON string.
--   2. The server runs the model, then calls our `deliver_blueprint`
--      remote interface (also over RCON). We import the blueprint string
--      into the requesting player's cursor.
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
      sources   = {},
      sinks     = {},
      pending   = nil,
      picker    = nil,
      footprint_render_id = nil,
    }
  end
  return storage.players[player_index]
end

local function ensure_pending_lookup()
  -- request_id -> player_index, so the RCON callback can find the player.
  storage.pending_by_request = storage.pending_by_request or {}
  -- FIFO of JSON request strings waiting for the server to pop via RCON.
  storage.outbox = storage.outbox or {}
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
  if inv.get_item_count("factorion-source-tool") == 0 then
    inv.insert({ name = "factorion-source-tool", count = 1 })
  end
  if inv.get_item_count("factorion-sink-tool") == 0 then
    inv.insert({ name = "factorion-sink-tool", count = 1 })
  end
  player.print({ "", "[Factorion] Tools added: ",
    "click with [item=factorion-footprint-tool] to stamp an 11x11 region, ",
    "then use [item=factorion-source-tool] and [item=factorion-sink-tool] ",
    "to choose items and place endpoints. Press CTRL+P to predict." })
end

script.on_event(defines.events.on_player_created, function(event)
  ensure_player_state(event.player_index)
  ensure_pending_lookup()
  local player = game.get_player(event.player_index)
  if player then give_tools(player) end
end)

script.on_event("factorion-give-tools", function(event)
  local player = game.get_player(event.player_index)
  if player then give_tools(player) end
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

local function inside_footprint(x, y, fp)
  return fp and x >= fp.x and x < fp.x + fp.w
    and y >= fp.y and y < fp.y + fp.h
end

local MARKER_DIALOG = "factorion-marker-dialog"
local ITEM_PICKER = "factorion-marker-item"
local DIRECTION_PICKER = "factorion-marker-direction"
local DIRECTION_LABELS = {
  "Flow north ↑", "Flow east →", "Flow south ↓", "Flow west ←",
}
local DIRECTION_ARROWS = { "↑", "→", "↓", "←" }
local DIRECTION_TO_FACTORIO = {
  defines.direction.north, defines.direction.east,
  defines.direction.south, defines.direction.west,
}
local DIRECTION_SIGNALS = {
  "up-arrow", "right-arrow", "down-arrow", "left-arrow",
}

local function destroy_endpoint_entity(mark)
  if not mark.entity_unit_number then return end
  local entity = game.get_entity_by_unit_number(mark.entity_unit_number)
  if entity and entity.valid then entity.destroy({ raise_destroy = true }) end
  mark.entity_unit_number = nil
end

local function destroy_endpoint_render(mark)
  for _, object_id in ipairs(mark.render_ids or {}) do
    local object = rendering.get_object_by_id(object_id)
    if object then object.destroy() end
  end
  mark.render_ids = nil
end

local function clear_endpoint_markers(state)
  for _, list in pairs({ state.sources or {}, state.sinks or {} }) do
    for _, mark in ipairs(list) do
      destroy_endpoint_render(mark)
      destroy_endpoint_entity(mark)
    end
  end
end

local function create_endpoint_entity(player, mark)
  local entity = player.surface.create_entity({
    name = "constant-combinator",
    position = { mark.x + 0.5, mark.y + 0.5 },
    direction = DIRECTION_TO_FACTORIO[mark.direction],
    force = player.force,
    player = player,
    raise_built = true,
  })
  if not entity then return nil end

  local behavior = entity.get_control_behavior()
  local section = behavior and behavior.get_section(1)
  if behavior and not section then section = behavior.add_section() end
  if not section then
    entity.destroy({ raise_destroy = true })
    return nil
  end
  section.set_slot(1, {
    value = { type = "item", name = mark.item, quality = "normal" },
    min = 1,
  })
  section.set_slot(2, {
    value = {
      type = "virtual",
      name = mark.role == "source" and "signal-output" or "signal-input",
      quality = "normal",
    },
    min = 1,
  })
  section.set_slot(3, {
    value = {
      type = "virtual", name = DIRECTION_SIGNALS[mark.direction],
      quality = "normal",
    },
    min = 1,
  })
  mark.entity_unit_number = entity.unit_number
  return entity
end

local function draw_endpoint(player, mark)
  destroy_endpoint_render(mark)
  local source = mark.role == "source"
  local color = source
    and { r = 0.25, g = 1.0, b = 0.3, a = 1.0 }
    or { r = 1.0, g = 0.4, b = 0.1, a = 1.0 }
  local icon = rendering.draw_sprite({
    sprite = source and "item/factorion-source-tool" or "item/factorion-sink-tool",
    surface = player.surface,
    target = { mark.x + 0.5, mark.y + 0.5 },
    x_scale = 0.22,
    y_scale = 0.22,
    players = { player },
    render_mode = "game",
  })
  local text = rendering.draw_text({
    text = string.format(
      "%s for [item=%s] %s",
      source and "SOURCE" or "SINK", mark.item,
      DIRECTION_ARROWS[mark.direction] or "?"),
    surface = player.surface,
    target = { mark.x + 0.5, mark.y - 0.25 },
    color = color,
    scale = 0.8,
    alignment = "center",
    vertical_alignment = "bottom",
    use_rich_text = true,
    players = { player },
    render_mode = "game",
  })
  mark.render_ids = { icon.id, text.id }
end

local function default_flow_direction(x, y, fp)
  local rel_x, rel_y = x - fp.x, y - fp.y
  local mid = (GRID_SIZE - 1) / 2
  local dx, dy = mid - rel_x, mid - rel_y
  if math.abs(dx) >= math.abs(dy) then
    return dx >= 0 and 2 or 4
  end
  return dy >= 0 and 3 or 1
end

local function close_marker_dialog(player, state)
  local frame = player.gui.screen[MARKER_DIALOG]
  if frame then frame.destroy() end
  state.picker = nil
end

local function open_marker_dialog(player, state, role, x, y)
  close_marker_dialog(player, state)
  local direction = default_flow_direction(x, y, state.footprint)
  state.picker = { role = role, x = x, y = y, direction = direction }
  local title = role == "source" and "Set Factorion source" or "Set Factorion sink"
  local prompt = role == "source"
    and "Choose the item this source provides:"
    or "Choose the item this sink should receive:"
  local frame = player.gui.screen.add({
    type = "frame", name = MARKER_DIALOG, caption = title,
    direction = "vertical",
  })
  frame.auto_center = true
  frame.add({ type = "label", caption = prompt })
  frame.add({
    type = "choose-elem-button", name = ITEM_PICKER,
    elem_type = "item", item = get_default_item(),
  })
  frame.add({ type = "label", caption = "Choose the direction items flow:" })
  frame.add({
    type = "drop-down", name = DIRECTION_PICKER,
    items = DIRECTION_LABELS, selected_index = direction,
  })
  local actions = frame.add({ type = "flow", direction = "horizontal" })
  actions.add({
    type = "button", caption = "Cancel",
    tags = { factorion_action = "marker-cancel" },
  })
  actions.add({
    type = "button", caption = "Set " .. role,
    style = "confirm_button",
    tags = { factorion_action = "marker-save" },
  })
  player.opened = frame
end

local function handle_tool_selection(event)
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  if not player then return end

  if event.item == "factorion-footprint-tool" then
    clear_endpoint_markers(state)
    state.footprint = fixed_footprint(event.area)
    state.sources = {}
    state.sinks = {}
    draw_footprint(player, state)
    local fp = state.footprint
    player.print(string.format(
      "[Factorion] Stamped 11x11 region at x=%d..%d, y=%d..%d. " ..
      "Now place a source and sink.",
      fp.x, fp.x + 10, fp.y, fp.y + 10))
    return
  end

  local role = event.item == "factorion-source-tool" and "source"
    or event.item == "factorion-sink-tool" and "sink" or nil
  if not role then return end
  if not state.footprint then
    player.print("[Factorion] Stamp the 11x11 region first.")
    return
  end
  local x, y = area_center_tile(event.area)
  if not inside_footprint(x, y, state.footprint) then
    player.print("[Factorion] Choose a tile inside the blue 11x11 region.")
    return
  end
  open_marker_dialog(player, state, role, x, y)
end

script.on_event(defines.events.on_player_selected_area, handle_tool_selection)

-- Right-clicking the region tool is a quick clear; source/sink tools behave
-- identically on either mouse button because their roles are explicit.
script.on_event(defines.events.on_player_alt_selected_area, function(event)
  if event.item ~= "factorion-footprint-tool" then
    handle_tool_selection(event)
    return
  end
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  destroy_footprint_render(state)
  clear_endpoint_markers(state)
  state.footprint = nil
  state.sources = {}
  state.sinks = {}
  if player then player.print("[Factorion] Region and endpoints cleared.") end
end)

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
  if action ~= "marker-save" or not state.picker then return end
  local frame = player.gui.screen[MARKER_DIALOG]
  local picker = frame and frame[ITEM_PICKER]
  local item = picker and picker.elem_value
  if not item then
    player.print("[Factorion] Choose an item first.")
    return
  end
  local direction_picker = frame and frame[DIRECTION_PICKER]
  local direction = direction_picker and direction_picker.selected_index or 0
  if direction < 1 or direction > 4 then
    player.print("[Factorion] Choose a flow direction first.")
    return
  end

  local mark = state.picker
  -- One endpoint per tile: saving replaces either previous role at this tile.
  for _, list in pairs({ state.sources, state.sinks }) do
    for i = #list, 1, -1 do
      if list[i].x == mark.x and list[i].y == mark.y then
        destroy_endpoint_render(list[i])
        destroy_endpoint_entity(list[i])
        table.remove(list, i)
      end
    end
  end
  local target = mark.role == "source" and state.sources or state.sinks
  local endpoint = {
    role = mark.role, x = mark.x, y = mark.y,
    item = item, direction = direction,
  }
  if not create_endpoint_entity(player, endpoint) then
    player.print(
      "[Factorion] Could not place the constant-combinator; clear that tile and try again.")
    return
  end
  table.insert(target, endpoint)
  draw_endpoint(player, endpoint)
  player.print(string.format(
    "[Factorion] Set %s at (%d,%d): [item=%s], %s",
    mark.role, mark.x, mark.y, item, DIRECTION_LABELS[direction]))
  close_marker_dialog(player, state)
end)

script.on_event(defines.events.on_gui_closed, function(event)
  if not event.element or event.element.name ~= MARKER_DIALOG then return end
  local state = ensure_player_state(event.player_index)
  state.picker = nil
end)

local function endpoint_entity_removed(event)
  local entity = event.entity
  if not entity or not entity.unit_number then return end
  for _, state in pairs(storage.players or {}) do
    for _, list in pairs({ state.sources or {}, state.sinks or {} }) do
      for i = #list, 1, -1 do
        if list[i].entity_unit_number == entity.unit_number then
          destroy_endpoint_render(list[i])
          table.remove(list, i)
        end
      end
    end
  end
end

script.on_event(defines.events.on_player_mined_entity, endpoint_entity_removed)
script.on_event(defines.events.on_robot_mined_entity, endpoint_entity_removed)
script.on_event(defines.events.on_entity_died, endpoint_entity_removed)

script.on_event("factorion-reset", function(event)
  local state = ensure_player_state(event.player_index)
  destroy_footprint_render(state)
  clear_endpoint_markers(state)
  state.footprint = nil
  state.sources = {}
  state.sinks = {}
  local player = game.get_player(event.player_index)
  if player then
    close_marker_dialog(player, state)
    player.print("[Factorion] Region and endpoints cleared.")
  end
end)

-- ----------------------------------------------------------------------------
-- direction inference: a source on the west edge faces east, etc.
-- ----------------------------------------------------------------------------

local function dir_toward_center(rel_x, rel_y, size)
  -- "Toward center" snapped to cardinal. Larger of the two distances-to-
  -- edge-center wins. Returns the Factorion Direction enum: 1=N,2=E,3=S,4=W.
  local mid = (size - 1) / 2
  local dx = mid - rel_x
  local dy = mid - rel_y
  if math.abs(dx) >= math.abs(dy) then
    if dx >= 0 then return 2 else return 4 end  -- east or west
  else
    if dy >= 0 then return 3 else return 1 end  -- south or north
  end
end

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

-- Map our arrow virtual signals to the Factorion Direction enum.
local ARROW_TO_DIR = {
  ["up-arrow"] = 1, ["right-arrow"] = 2,
  ["down-arrow"] = 3, ["left-arrow"] = 4,
}

local function parse_combinator_marker(combi)
  -- Inspect a constant-combinator's first section for our identifying
  -- filters: signal-output / signal-input + arrow + item. Returns
  --   { role="source"|"sink", item=<name>, direction=<1..4> }
  -- or nil if this combinator isn't one of our markers.
  local cb = combi.get_control_behavior()
  if not cb or not cb.sections then return nil end
  for _, section in pairs(cb.sections) do
    local item_name, role, direction = nil, nil, nil
    local n = section.filters_count or 0
    for i = 1, n do
      local f = section.get_slot(i)
      if f and f.value and f.value.name then
        local name = f.value.name
        local typ = f.value.type
        if name == "signal-output" then role = "source"
        elseif name == "signal-input" then role = "sink"
        elseif ARROW_TO_DIR[name] then direction = ARROW_TO_DIR[name]
        elseif typ == nil or typ == "item" then item_name = name
        end
      end
    end
    if role then
      return { role = role, item = item_name, direction = direction or 2 }
    end
  end
  return nil
end

local function scan_footprint_for_markers(player, fp)
  -- Returns (sources, sinks), each a list of {x_world, y_world, direction, item}.
  -- Coordinates are world tiles (floor of the combinator's centre).
  local sources, sinks = {}, {}
  if not player or not player.surface then return sources, sinks end
  local found = player.surface.find_entities_filtered{
    area = { { fp.x, fp.y }, { fp.x + fp.w, fp.y + fp.h } },
    name = "constant-combinator",
  }
  for _, combi in pairs(found) do
    local m = parse_combinator_marker(combi)
    if m then
      local entry = {
        x = math.floor(combi.position.x),
        y = math.floor(combi.position.y),
        direction = m.direction,
        item = m.item,
      }
      if m.role == "source" then
        table.insert(sources, entry)
      else
        table.insert(sinks, entry)
      end
    end
  end
  return sources, sinks
end

local function gather_request(state, player_index)
  local fp = state.footprint
  local size = get_grid_size()
  local default_item = get_default_item()
  local player = game.get_player(player_index)

  -- Prefer in-world constant-combinators with our identifying filters.
  -- Falls back to the click-tool state lists if no combinators are
  -- found inside the footprint.
  local from_world_src, from_world_snk = scan_footprint_for_markers(player, fp)
  local raw_sources = (#from_world_src > 0) and from_world_src or state.sources
  local raw_sinks = (#from_world_snk > 0) and from_world_snk or state.sinks
  local source_provenance = (#from_world_src > 0) and "combinator" or "source-tool"
  local sink_provenance = (#from_world_snk > 0) and "combinator" or "sink-tool"

  local sources = {}
  for _, s in ipairs(raw_sources) do
    local rx, ry = world_to_rel(s, fp)
    if in_footprint(rx, ry, fp) then
      table.insert(sources, {
        x = rx, y = ry,
        direction = s.direction or dir_toward_center(rx, ry, size),
        item = s.item or default_item,
      })
    end
  end

  local sinks = {}
  for _, s in ipairs(raw_sinks) do
    local rx, ry = world_to_rel(s, fp)
    if in_footprint(rx, ry, fp) then
      table.insert(sinks, {
        x = rx, y = ry,
        direction = s.direction or dir_toward_center(rx, ry, size),
        item = s.item or default_item,
      })
    end
  end

  local provenance = string.format(
    "%d source(s) from %s, %d sink(s) from %s",
    #sources, source_provenance, #sinks, sink_provenance)

  local request = {
    request_id    = new_request_id(),
    player_index  = player_index,
    grid_size     = size,
    footprint     = build_footprint_mask(fp),
    sources       = sources,
    sinks         = sinks,
    default_item  = default_item,
  }
  return request, provenance
end

-- Build a request from the player's current state (combinators preferred,
-- click-marks as fallback) and enqueue it for the server. Returns
-- (true, message) on success, (false, message) on a precheck failure.
try_request_prediction = function(player_index)
  local state = ensure_player_state(player_index)
  ensure_pending_lookup()
  if not state.footprint then
    return false, "No footprint set."
  end
  local request, provenance = gather_request(state, player_index)
  if #request.sources == 0 then
    return false, "No source set. Use the green source tool inside the region."
  end
  if #request.sinks == 0 then
    return false, "No sink set. Use the orange sink tool inside the region."
  end
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

script.on_event("factorion-execute", function(event)
  local player = game.get_player(event.player_index)
  if not player then return end
  local _, msg = try_request_prediction(event.player_index)
  player.print("[Factorion] " .. msg)
end)

-- ----------------------------------------------------------------------------
-- RCON interface: server pushes the prediction back here
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

  -- Called via RCON: remote.call("factorion","deliver_blueprint", req_id, bp_str)
  deliver_blueprint = function(request_id, blueprint_string)
    ensure_pending_lookup()
    local player_index = storage.pending_by_request[request_id]
    if not player_index then
      log("[Factorion] deliver_blueprint: unknown request_id " .. tostring(request_id))
      return false
    end
    storage.pending_by_request[request_id] = nil

    -- player_index == 0 is the headless-test sentinel: log the blueprint
    -- to factorio-current.log instead of trying to inject into a cursor
    -- that doesn't exist (no player connected in --start-server-load-
    -- scenario without admin auth).
    if player_index == 0 then
      log("[Factorion] (headless) blueprint for " .. tostring(request_id) ..
        " (" .. tostring(#blueprint_string) .. " chars): " ..
        string.sub(blueprint_string, 1, 80) .. "...")
      return true
    end

    local player = game.get_player(player_index)
    if not player then return false end

    local cursor = player.cursor_stack
    if not cursor then
      player.print("[Factorion] No cursor_stack available.")
      return false
    end
    -- LuaItemStack:is_empty() was removed in 2.0; use valid_for_read instead.
    if cursor.valid_for_read then
      cursor.clear()
    end
    if not cursor.set_stack({ name = "blueprint", count = 1 }) then
      player.print("[Factorion] Could not place blueprint in cursor.")
      return false
    end

    local rc = cursor.import_stack(blueprint_string)
    if rc < 0 then
      player.print(string.format(
        "[Factorion] Blueprint imported with warnings (rc=%d).", rc))
    elseif rc > 0 then
      player.print("[Factorion] Blueprint import failed.")
      cursor.clear()
      return false
    else
      player.print("[Factorion] Prediction ready. Paste it on your footprint.")
    end

    local state = ensure_player_state(player_index)
    state.pending = nil
    return true
  end,

  -- Diagnostic: server can call this to check the mod is alive.
  ping = function()
    return "factorion-mod alive at tick " .. tostring(game.tick)
  end,

  -- Headless / debug: enqueue a request JSON as if the hotkey had fired.
  -- /silent-command runs in *level scope*, not any mod's, so storage is
  -- inaccessible from RCON directly — this interface is the only way to
  -- poke an outbox entry in from outside the game.
  --
  -- Caller supplies the request JSON and a player_index to deliver the
  -- response to. In headless tests with no player connected, pass 0 to
  -- signal "log the blueprint string instead of injecting into a cursor".
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
    return string.format("outbox=%d pending=%d players=%d",
      #storage.outbox, pending_n,
      storage.players and table_size(storage.players) or 0)
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
          "player[%s]: footprint={%s} #sources=%d #sinks=%d",
          tostring(k), fp, #(p.sources or {}), #(p.sinks or {})))
        if p.sources and #p.sources > 0 then
          local s = {}
          for _, src in ipairs(p.sources) do
            table.insert(s, string.format("(%d,%d)", src.x, src.y))
          end
          table.insert(parts, "  sources: " .. table.concat(s, ","))
        end
        if p.sinks and #p.sinks > 0 then
          local s = {}
          for _, snk in ipairs(p.sinks) do
            table.insert(s, string.format("(%d,%d)", snk.x, snk.y))
          end
          table.insert(parts, "  sinks: " .. table.concat(s, ","))
        end
      end
    end
    if #parts == 0 then return "(no player state)" end
    return table.concat(parts, "\n")
  end,
})

-- Parity runs drive the game every tick while active; the handler
-- early-outs when no run is in flight, so it's free otherwise.
script.on_event(defines.events.on_tick, parity.on_tick)

-- ----------------------------------------------------------------------------
-- migrations
-- ----------------------------------------------------------------------------

script.on_init(function()
  storage.players = {}
  storage.pending_by_request = {}
  storage.outbox = {}
  storage.model_requests = {}
end)

script.on_configuration_changed(function()
  storage.players = storage.players or {}
  storage.pending_by_request = storage.pending_by_request or {}
  storage.outbox = storage.outbox or {}
  storage.model_requests = storage.model_requests or {}
  for _, player in pairs(game.players) do
    local state = ensure_player_state(player.index)
    give_tools(player)
    if state.footprint then draw_footprint(player, state) end
    for role, list in pairs({ source = state.sources, sink = state.sinks }) do
      for _, mark in ipairs(list or {}) do
        mark.role = role
        mark.item = mark.item or get_default_item()
        mark.direction = mark.direction
          or (state.footprint
            and default_flow_direction(mark.x, mark.y, state.footprint) or 2)
        local entity = mark.entity_unit_number
          and game.get_entity_by_unit_number(mark.entity_unit_number) or nil
        if not entity then create_endpoint_entity(player, mark) end
        draw_endpoint(player, mark)
      end
    end
  end
end)
