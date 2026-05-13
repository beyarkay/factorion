-- Factorion mod runtime.
--
-- State per player lives in `storage.players[player_index]`:
--   footprint = { x=int, y=int, w=int, h=int }   world-tile bbox
--   sources   = { { x=int, y=int }, ... }        world-tile coords
--   sinks     = { { x=int, y=int }, ... }
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

local function get_grid_size()
  return settings.global["factorion-grid-size"].value
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

-- ----------------------------------------------------------------------------
-- player onboarding
-- ----------------------------------------------------------------------------

local function give_tools(player)
  local inv = player.get_main_inventory()
  if not inv then return end
  if inv.get_item_count("factorion-footprint-tool") == 0 then
    inv.insert({ name = "factorion-footprint-tool", count = 1 })
  end
  if inv.get_item_count("factorion-marker-tool") == 0 then
    inv.insert({ name = "factorion-marker-tool", count = 1 })
  end
  player.print({ "", "[Factorion] Tools added: ",
    "drag the [item=factorion-footprint-tool] to set an ",
    tostring(get_grid_size()), "x", tostring(get_grid_size()),
    " footprint, then left/right-click with [item=factorion-marker-tool] ",
    "to mark sources/sinks. Press CTRL+SHIFT+P to predict." })
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

local function area_to_bbox(area)
  local x = tile_floor(area.left_top.x)
  local y = tile_floor(area.left_top.y)
  -- right_bottom is exclusive in tile-space; subtract 1 for the inclusive
  -- last tile, then +1 for the count.
  local w = tile_floor(area.right_bottom.x) - x
  local h = tile_floor(area.right_bottom.y) - y
  return { x = x, y = y, w = w, h = h }
end

script.on_event(defines.events.on_player_selected_area, function(event)
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  if not player then return end

  if event.item == "factorion-footprint-tool" then
    local bbox = area_to_bbox(event.area)
    local size = get_grid_size()
    if bbox.w ~= size or bbox.h ~= size then
      player.print(string.format(
        "[Factorion] Footprint must be exactly %dx%d (got %dx%d). " ..
        "Try again, or change `factorion-grid-size` in mod settings.",
        size, size, bbox.w, bbox.h))
      return
    end
    state.footprint = bbox
    state.sources = {}
    state.sinks = {}
    player.print(string.format(
      "[Factorion] Footprint set: x=%d..%d (%d wide), y=%d..%d (%d tall). " ..
      "Mark sources/sinks WITHIN these bounds.",
      bbox.x, bbox.x + bbox.w - 1, bbox.w,
      bbox.y, bbox.y + bbox.h - 1, bbox.h))

  elseif event.item == "factorion-marker-tool" then
    -- left-click = sources
    if not state.footprint then
      player.print("[Factorion] Set a footprint first.")
      return
    end
    local added, skipped = {}, {}
    local fp = state.footprint
    for _, tile in pairs(event.tiles or {}) do
      local tx, ty = tile.position.x, tile.position.y
      if tx >= fp.x and tx < fp.x + fp.w
         and ty >= fp.y and ty < fp.y + fp.h then
        table.insert(state.sources, { x = tx, y = ty })
        table.insert(added, string.format("(%d,%d)", tx, ty))
      else
        table.insert(skipped, string.format("(%d,%d)", tx, ty))
      end
    end
    if #added > 0 then
      player.print(string.format(
        "[Factorion] +%d source(s) at %s; total %d.",
        #added, table.concat(added, ","), #state.sources))
    end
    if #skipped > 0 then
      player.print(string.format(
        "[Factorion] Skipped %d source mark(s) outside footprint: %s. " ..
        "Valid x=%d..%d y=%d..%d.",
        #skipped, table.concat(skipped, ","),
        fp.x, fp.x + fp.w - 1, fp.y, fp.y + fp.h - 1))
    end
  end
end)

-- The "alt" variants of selection events have moved around between
-- Factorio versions. Wire up all three secondary modes so whichever the
-- player's modifier hits (Shift, Ctrl-Shift, etc.) records sinks.

local function handle_sink_select(event, mode_name)
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  if not player then return end

  if event.item == "factorion-footprint-tool" then
    -- any secondary on the footprint = clear it
    state.footprint = nil
    state.sources = {}
    state.sinks = {}
    player.print(string.format("[Factorion] Footprint cleared (via %s).", mode_name))

  elseif event.item == "factorion-marker-tool" then
    if not state.footprint then
      player.print("[Factorion] Set a footprint first.")
      return
    end
    local added, skipped = {}, {}
    local fp = state.footprint
    for _, tile in pairs(event.tiles or {}) do
      local tx, ty = tile.position.x, tile.position.y
      if tx >= fp.x and tx < fp.x + fp.w
         and ty >= fp.y and ty < fp.y + fp.h then
        table.insert(state.sinks, { x = tx, y = ty })
        table.insert(added, string.format("(%d,%d)", tx, ty))
      else
        table.insert(skipped, string.format("(%d,%d)", tx, ty))
      end
    end
    if #added > 0 then
      player.print(string.format(
        "[Factorion] +%d sink(s) at %s; total %d. (via %s)",
        #added, table.concat(added, ","), #state.sinks, mode_name))
    end
    if #skipped > 0 then
      player.print(string.format(
        "[Factorion] Skipped %d sink mark(s) outside footprint: %s. " ..
        "Valid x=%d..%d y=%d..%d.",
        #skipped, table.concat(skipped, ","),
        fp.x, fp.x + fp.w - 1, fp.y, fp.y + fp.h - 1))
    end
  end
end

script.on_event(defines.events.on_player_alt_selected_area, function(e)
  handle_sink_select(e, "alt_select")
end)
script.on_event(defines.events.on_player_reverse_selected_area, function(e)
  handle_sink_select(e, "reverse_select")
end)
script.on_event(defines.events.on_player_alt_reverse_selected_area, function(e)
  handle_sink_select(e, "alt_reverse_select")
end)

script.on_event("factorion-reset", function(event)
  local state = ensure_player_state(event.player_index)
  state.footprint = nil
  state.sources = {}
  state.sinks = {}
  local player = game.get_player(event.player_index)
  if player then player.print("[Factorion] State cleared.") end
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

local function gather_request(state, player_index)
  local fp = state.footprint
  local size = get_grid_size()
  local default_item = get_default_item()

  local sources = {}
  for _, s in ipairs(state.sources) do
    local rx, ry = world_to_rel(s, fp)
    if in_footprint(rx, ry, fp) then
      table.insert(sources, {
        x = rx, y = ry,
        direction = dir_toward_center(rx, ry, size),
        item = default_item,
      })
    end
  end

  local sinks = {}
  for _, s in ipairs(state.sinks) do
    local rx, ry = world_to_rel(s, fp)
    if in_footprint(rx, ry, fp) then
      -- Sink "faces" the same way (into the footprint). The server
      -- knows the difference because we put it in the sinks array.
      table.insert(sinks, {
        x = rx, y = ry,
        direction = dir_toward_center(rx, ry, size),
        item = default_item,
      })
    end
  end

  return {
    request_id    = new_request_id(),
    player_index  = player_index,
    grid_size     = size,
    footprint     = build_footprint_mask(fp),
    sources       = sources,
    sinks         = sinks,
    default_item  = default_item,
  }
end

local function json_encode(t)
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
  local state = ensure_player_state(event.player_index)
  ensure_pending_lookup()

  if not state.footprint then
    player.print("[Factorion] No footprint set.")
    return
  end
  if #state.sources == 0 and #state.sinks == 0 then
    player.print("[Factorion] No sources or sinks marked.")
    return
  end

  local request = gather_request(state, event.player_index)
  local request_json = json_encode(request)
  table.insert(storage.outbox, request_json)
  storage.pending_by_request[request.request_id] = event.player_index
  state.pending = { request_id = request.request_id, tick = game.tick }

  -- Log the full payload to factorio-current.log so we can inspect it
  -- from outside the game when things look wrong.
  log("[Factorion] enqueued request " .. request.request_id ..
      " (footprint=" .. tostring(#request.footprint) ..
      ", sources=" .. tostring(#request.sources) ..
      ", sinks=" .. tostring(#request.sinks) ..
      ") json=" .. request_json)

  player.print(string.format(
    "[Factorion] Request %s queued (footprint=%d sources=%d sinks=%d, " ..
    "outbox depth %d). Waiting for server…",
    request.request_id, #request.footprint,
    #request.sources, #request.sinks, #storage.outbox))
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

-- ----------------------------------------------------------------------------
-- migrations
-- ----------------------------------------------------------------------------

script.on_init(function()
  storage.players = {}
  storage.pending_by_request = {}
  storage.outbox = {}
end)

script.on_configuration_changed(function()
  storage.players = storage.players or {}
  storage.pending_by_request = storage.pending_by_request or {}
  storage.outbox = storage.outbox or {}
end)
