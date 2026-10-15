-- Factorion mod runtime.
--
-- A region is a painted (blue) area with a GPU server against its edge or
-- cabled to it; every server sharing paint with it is its cluster, and they
-- predict together. A server's state lives in `storage.servers[unit_number]`:
--   footprint  = { x=int, y=int, w=int, h=int }   model grid, world tiles
--   pending    = { request_id=string, tick=int }  in-flight request it owns
--   computing  = request_id                       its cluster's request
--   conditions = { predict=…, deconstruct=… }     circuit conditions
--
-- Round trip (single-channel RCON, both directions):
--   1. On the GPU server's Predict button, we build a request JSON describing
--      the painted tiles + existing entities + sources + sinks (in
--      footprint-relative coords) and
--      enqueue it on `storage.outbox`. The Python server is polling our
--      `poll_request` remote interface over RCON; when there's something
--      in the queue it gets popped and returned as a JSON string.
--   2. After each denoising step, the server calls `place_predictions` over
--      RCON with the entities that step changed. We replace them directly in
--      the world.
--   3. The server calls `finish_prediction` when the denoising ends.
--
-- No script-output files are involved. The only outbound channel a
-- vanilla mod has is RCON, and we use it both ways.

local parity = require("parity")
local inference = require("inference")
local gui = require("gui")

local PROTOCOL_VERSION = "11"
-- The server polls every 250 ms, so 5 s of silence means it isn't there.
local SERVER_TIMEOUT_TICKS = 5 * 60
-- Loading a checkpoint (often a W&B download) blocks the server's polls.
local MODEL_LOAD_TIMEOUT_TICKS = 60 * 60
local STOP_REASONS = {
  converged = "the model settled on it",
  max_steps = "it was still changing at the step limit",
  server_error = "the server hit an error",
}
-- A cluster's denoising step takes the forward-pass time divided by its
-- powered servers, counting at most this many.
local MAX_CLUSTER_SPEEDUP = 8
local FORWARD_PASS = "factorion-forward-pass"
local GPU_SIGNALS = "factorion-gpu-signals"
local IDLE_KW, BUSY_KW = 0.9, 4.5
-- Utilisation is a moving average with this time constant.
local UTILISATION_TICKS = 2 * 60
local SOURCE_EMPTY_TICKS = 2 * 60

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

-- Generation settings named in the request message when not at their default.
local SAMPLING_SUMMARY = {
  { "temperature", "temperature %g" },
  { "steps", "%d denoising steps" },
  { "refine", "up to %d refinement steps" },
  { "best_of", "best of %d" },
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
    storage.players[player_index] = { picker = nil }
  end
  return storage.players[player_index]
end

local function ensure_pending_lookup()
  -- request_id -> GPU server unit number (0 for headless injections).
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
    "tile with it to choose what to make, then click a tile for each source. " ..
    "Paint the tiles the factory may use with [item=factorion-mask], put a " ..
    "[item=factorion-gpu-server] against the paint and press Predict in its window. " ..
    "Belt the ingredients into the source belts and carry the product away from the sink.")
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

local MASK_TILE = "factorion-mask"
local MASK_OUTLINE_COLOR = { r = 0.2, g = 0.45, b = 1.0, a = 1 }
local MASK_TOO_BIG_COLOR = { r = 1.0, g = 0.25, b = 0.2, a = 1 }
-- Runs overlap by half a line width so their corners meet.
local MASK_OUTLINE_OVERLAP = 0.08

local function tile_key(x, y) return x .. "," .. y end

-- The tiles of entity's footprint.
local function tile_box(entity)
  local box = entity.bounding_box
  return math.floor(box.left_top.x + 0.01), math.floor(box.left_top.y + 0.01),
    math.ceil(box.right_bottom.x - 0.01) - 1, math.ceil(box.right_bottom.y - 0.01) - 1
end

local NEIGHBOURS = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } }

local DATA_CABLE = "factorion-data-cable"
local CABLE_COLOR = { r = 0.2, g = 0.45, b = 1.0, a = 1 }
-- Reach of a cable hop with no pole at either end, as a medium pole's wire.
local CABLE_REACH = 9
local CABLE_SEGMENTS = 6

local function server_state(server)
  storage.servers = storage.servers or {}
  local state = storage.servers[server.unit_number]
  if not state then
    state = { entity = server, model_entities = {} }
    storage.servers[server.unit_number] = state
  end
  return state
end

-- The painted tiles connected to start, as tile keys; nil once they outgrow
-- the size x size grid, unless confined to window. Tiles under blocked (a GPU
-- server) don't count.
local function flood_paint(surface, start, size, window, blocked)
  local x0, y0, x1, y1 = tile_box(blocked)
  local function free_paint(x, y)
    local under = x >= x0 and x <= x1 and y >= y0 and y <= y1
    local inside = not window or (x >= window.x and x < window.x + size
      and y >= window.y and y < window.y + size)
    return inside and not under and surface.get_tile(x, y).name == MASK_TILE
  end
  local area, queue = { [tile_key(start.x, start.y)] = start }, { start }
  local min_x, min_y, max_x, max_y = start.x, start.y, start.x, start.y
  while #queue > 0 do
    local tile = table.remove(queue)
    min_x, max_x = math.min(min_x, tile.x), math.max(max_x, tile.x)
    min_y, max_y = math.min(min_y, tile.y), math.max(max_y, tile.y)
    if not window and (max_x - min_x >= size or max_y - min_y >= size) then return nil end
    for _, step in ipairs(NEIGHBOURS) do
      local x, y = tile.x + step[1], tile.y + step[2]
      if not area[tile_key(x, y)] and free_paint(x, y) then
        area[tile_key(x, y)] = { x = x, y = y }
        table.insert(queue, { x = x, y = y })
      end
    end
  end
  return area
end

-- The painted area a GPU server touches, as tile keys, or nil and why (no
-- reason when it touches none). Paint bigger than the model's grid is cut to
-- the grid square against the server's most-painted side, returned too.
local function touching_paint(server, size)
  local surface = server.surface
  local x0, y0, x1, y1 = tile_box(server)
  local function painted(x, y) return surface.get_tile(x, y).name == MASK_TILE end
  local sides = { north = {}, south = {}, west = {}, east = {} }
  for x = x0, x1 do
    if painted(x, y0 - 1) then table.insert(sides.north, { x = x, y = y0 - 1 }) end
    if painted(x, y1 + 1) then table.insert(sides.south, { x = x, y = y1 + 1 }) end
  end
  for y = y0, y1 do
    if painted(x0 - 1, y) then table.insert(sides.west, { x = x0 - 1, y = y }) end
    if painted(x1 + 1, y) then table.insert(sides.east, { x = x1 + 1, y = y }) end
  end
  local seeds, best = {}, nil
  for name, side in pairs(sides) do
    for _, seed in ipairs(side) do table.insert(seeds, seed) end
    if #side > 0 and (not best or #side > #sides[best]) then best = name end
  end
  if not best then return nil end
  local area, window = flood_paint(surface, seeds[1], size, nil, server), nil
  if not area then
    local across_x = x0 - math.floor((size - (x1 - x0 + 1)) / 2)
    local across_y = y0 - math.floor((size - (y1 - y0 + 1)) / 2)
    window = ({
      north = { x = across_x, y = y0 - size },
      south = { x = across_x, y = y1 + 1 },
      west = { x = x0 - size, y = across_y },
      east = { x = x1 + 1, y = across_y },
    })[best]
    window.w, window.h = size, size
    seeds = sides[best]
    area = flood_paint(surface, seeds[1], size, window, server)
  end
  for _, seed in ipairs(seeds) do
    if not area[tile_key(seed.x, seed.y)] then
      return nil, "it touches more than one painted area"
    end
  end
  return area, window
end

-- The painted area at a data cable's end, cut to the grid square centred on
-- that end when bigger than the grid; or nil and why.
local function cable_paint(server, cable, size)
  local surface = server.surface
  if surface.get_tile(cable.x, cable.y).name ~= MASK_TILE then
    return nil, "a data cable ends off the paint"
  end
  local start = { x = cable.x, y = cable.y }
  local area = flood_paint(surface, start, size, nil, server)
  if area then return area end
  local half = math.floor(size / 2)
  local window = { x = cable.x - half, y = cable.y - half, w = size, h = size }
  return flood_paint(surface, start, size, window, server), window
end

-- Data links, like vanilla wires, each join two nodes: GPU servers,
-- electric poles and painted tiles. storage.links[key] = { a, b, ka, kb,
-- render_ids }, where ka and kb are the nodes' keys and key joins them.

-- The painted tiles and GPU servers server reaches through data links,
-- passing only through electric poles.
local function linked_nodes(server)
  local start = "e" .. server.unit_number
  local seen, queue, tiles, servers = { [start] = true }, { start }, {}, {}
  while #queue > 0 do
    local at = table.remove(queue)
    for _, link in pairs(storage.links or {}) do
      for _, ends in ipairs({ { link.ka, link.b, link.kb }, { link.kb, link.a, link.ka } }) do
        local from, node, key = ends[1], ends[2], ends[3]
        if from == at and not seen[key] then
          seen[key] = true
          if key:sub(1, 1) == "t" then
            table.insert(tiles, node)
          elseif node.valid and node.type == "electric-pole" then
            table.insert(queue, key)
          elseif node.valid and node.name == inference.GPU_SERVER then
            table.insert(servers, node)
          end
        end
      end
    end
  end
  return tiles, servers
end

-- Every painted piece server reaches itself, { area, window }: the paint it
-- touches and the paint its data links reach. Or nil and why.
local function server_pieces(server, size)
  local pieces = {}
  local area, window = touching_paint(server, size)
  if area then
    table.insert(pieces, { area = area, window = window })
  elseif window then
    return nil, window
  end
  for _, cable in ipairs((linked_nodes(server))) do
    local cabled, cut = cable_paint(server, cable, size)
    if not cabled then return nil, cut end
    table.insert(pieces, { area = cabled, window = cut })
  end
  return pieces
end

-- Blinks, like vanilla's alert icons.
local function draw_alert(kind, target, surface, offset)
  local object = rendering.draw_sprite({
    sprite = "factorion-alert-" .. kind, target = target, surface = surface,
    target_offset = offset, x_scale = 0.8, y_scale = 0.8,
    render_layer = "entity-info-icon-above",
  })
  storage.blinking = storage.blinking or {}
  table.insert(storage.blinking, object.id)
  return object.id
end

-- GPU servers predict together when they touch, share paint, or are joined
-- by data links through any poles. Each server's cluster, by unit number;
-- the members of one cluster share a list.
local function clusters(surface)
  local size = storage.model and storage.model.grid_size
  local servers = surface.find_entities_filtered({ name = inference.GPU_SERVER })
  local parent = {}
  local function find(unit)
    while parent[unit] ~= unit do unit = parent[unit] end
    return unit
  end
  local function join(a, b) parent[find(a)] = find(b) end
  for _, server in ipairs(servers) do parent[server.unit_number] = server.unit_number end
  local painted_by = {}
  for i, server in ipairs(servers) do
    local unit = server.unit_number
    for _, piece in ipairs(size and server_pieces(server, size) or {}) do
      for key in pairs(piece.area) do
        if painted_by[key] then join(painted_by[key], unit) else painted_by[key] = unit end
      end
    end
    local _, linked = linked_nodes(server)
    for _, other in ipairs(linked) do join(unit, other.unit_number) end
    local x0, y0, x1, y1 = tile_box(server)
    for j = i + 1, #servers do
      local a0, b0, a1, b1 = tile_box(servers[j])
      local across = a0 <= x1 and a1 >= x0
      local along = b0 <= y1 and b1 >= y0
      if (across and (b0 == y1 + 1 or b1 == y0 - 1)) or (along and (a0 == x1 + 1 or a1 == x0 - 1)) then
        join(unit, servers[j].unit_number)
      end
    end
  end
  local groups, result = {}, {}
  for _, server in ipairs(servers) do
    local root = find(server.unit_number)
    groups[root] = groups[root] or {}
    table.insert(groups[root], server)
    result[server.unit_number] = groups[root]
  end
  return result
end

-- server's cluster, server first; groups is clusters() of its surface when
-- the caller already has it.
local function cluster(server, groups)
  local members = { server }
  for _, other in ipairs((groups or clusters(server.surface))[server.unit_number] or {}) do
    if other.unit_number ~= server.unit_number then table.insert(members, other) end
  end
  return members
end

-- The painted pieces server's whole cluster reaches, its own first; or nil
-- and why.
local function cluster_pieces(server, size, groups)
  local pieces = {}
  for _, member in ipairs(cluster(server, groups)) do
    local own, reason = server_pieces(member, size)
    if not own and member == server then return nil, reason end
    for _, piece in ipairs(own or {}) do table.insert(pieces, piece) end
  end
  if #pieces == 0 then
    return nil, "neither it nor a GPU server beside or cabled to it touches [item=" ..
      MASK_TILE .. "] paint or has a [item=" .. DATA_CABLE .. "] to any"
  end
  return pieces
end

local function powered(servers)
  local result = {}
  for _, server in ipairs(servers) do
    if server.energy > 0 then table.insert(result, server) end
  end
  return result
end

-- Outlines each painted area of the surface, in alt-mode only, and flags
-- the ones no GPU server reaches. An area bigger than the model's grid is
-- outlined only where GPU servers would build, or in red when none would.
-- removed is a GPU server about to go.
local function draw_mask_outline(surface, removed)
  storage.mask_outlines = storage.mask_outlines or {}
  for _, id in ipairs(storage.mask_outlines[surface.index] or {}) do
    local object = rendering.get_object_by_id(id)
    if object then object.destroy() end
  end
  -- Outlines nothing tracks, such as an older version's, are the only wide
  -- lines the mod draws on the ground.
  for _, object in pairs(rendering.get_all_objects(script.mod_name)) do
    if object.surface.index == surface.index and object.type == "line"
        and object.draw_on_ground and object.width == 5 then
      object.destroy()
    end
  end
  local ids = {}
  storage.mask_outlines[surface.index] = ids
  if not prototypes.tile[MASK_TILE] then return end

  local function outline(tiles, color)
    -- Unit edges grouped by the grid line they lie on.
    local lines = {}
    local function edge(horizontal, at, along)
      local key = (horizontal and "h" or "v") .. at
      lines[key] = lines[key] or { horizontal = horizontal, at = at, along = {} }
      table.insert(lines[key].along, along)
    end
    for _, tile in pairs(tiles) do
      local x, y = tile.x, tile.y
      if not tiles[tile_key(x, y - 1)] then edge(true, y, x) end
      if not tiles[tile_key(x, y + 1)] then edge(true, y + 1, x) end
      if not tiles[tile_key(x - 1, y)] then edge(false, x, y) end
      if not tiles[tile_key(x + 1, y)] then edge(false, x + 1, y) end
    end
    local function point(line, along)
      if line.horizontal then return { along, line.at } end
      return { line.at, along }
    end
    for _, line in pairs(lines) do
      table.sort(line.along)
      local start = line.along[1]
      for i, along in ipairs(line.along) do
        local next_along = line.along[i + 1]
        if next_along ~= along + 1 then
          local object = rendering.draw_line({
            surface = surface,
            from = point(line, start - MASK_OUTLINE_OVERLAP),
            to = point(line, along + 1 + MASK_OUTLINE_OVERLAP),
            color = color,
            width = 5,
            only_in_alt_mode = true,
            draw_on_ground = true,
          })
          table.insert(ids, object.id)
          start = next_along
        end
      end
    end
  end

  local painted = {}
  for _, tile in pairs(surface.find_tiles_filtered({ name = MASK_TILE })) do
    painted[tile_key(tile.position.x, tile.position.y)] = { x = tile.position.x, y = tile.position.y }
  end
  local size = storage.model and storage.model.grid_size
  local covered = {}
  for _, server in pairs(size and surface.find_entities_filtered({ name = inference.GPU_SERVER }) or {}) do
    if server ~= removed then
      for _, piece in ipairs(server_pieces(server, size) or {}) do
        for key, tile in pairs(piece.area) do covered[key] = tile end
      end
    end
  end
  local seen = {}
  for key, start in pairs(painted) do
    if not seen[key] then
      local area, queue = { [key] = start }, { start }
      seen[key] = true
      local min_x, min_y, max_x, max_y = start.x, start.y, start.x, start.y
      while #queue > 0 do
        local tile = table.remove(queue)
        min_x, max_x = math.min(min_x, tile.x), math.max(max_x, tile.x)
        min_y, max_y = math.min(min_y, tile.y), math.max(max_y, tile.y)
        for _, step in ipairs(NEIGHBOURS) do
          local next_key = tile_key(tile.x + step[1], tile.y + step[2])
          if painted[next_key] and not seen[next_key] then
            seen[next_key] = true
            area[next_key] = painted[next_key]
            table.insert(queue, painted[next_key])
          end
        end
      end
      local part = {}
      for tile_at, tile in pairs(area) do
        if covered[tile_at] then part[tile_at] = tile end
      end
      if not size or (max_x - min_x < size and max_y - min_y < size) then
        outline(area, MASK_OUTLINE_COLOR)
      elseif next(part) then
        outline(part, MASK_OUTLINE_COLOR)
      else
        outline(area, MASK_TOO_BIG_COLOR)
      end
      if size and not next(part) then
        local cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        local best
        for _, tile in pairs(area) do
          local d = (tile.x - cx) ^ 2 + (tile.y - cy) ^ 2
          if not best or d < best.d then best = { x = tile.x, y = tile.y, d = d } end
        end
        table.insert(ids, draw_alert("unlinked", { best.x + 0.5, best.y + 0.5 }, surface))
      end
    end
  end
end

-- A cable node is a GPU server, an electric pole or a painted tile { x, y };
-- reading a missing key from an entity raises, so test the type first.
local function is_tile(node) return node.object_name ~= "LuaEntity" end

local function node_key(node, surface)
  if is_tile(node) then return "t" .. surface.index .. ":" .. node.x .. "," .. node.y end
  return "e" .. node.unit_number
end

local function node_valid(node) return is_tile(node) or node.valid end

local function node_position(node)
  if is_tile(node) then return { x = node.x + 0.5, y = node.y + 0.5 } end
  return node.position
end

-- The data port on top of the GPU server's I/O box, which sits at the left
-- end of the horizontal picture and the bottom of the vertical one.
local GPU_CABLE_PORT = {
  [defines.direction.north] = { -1.67, -0.7 },
  [defines.direction.south] = { -1.67, -0.7 },
  [defines.direction.east] = { 0.25, 0.9 },
  [defines.direction.west] = { 0.25, 0.9 },
}

-- Where a cable hangs from a GPU server's port, an electric pole's wire
-- point, or a painted tile's centre.
local function cable_point(node)
  if is_tile(node) then return node_position(node) end
  if node.type == "electric-pole" then
    return { x = node.position.x, y = node.position.y - 2.4 }
  end
  local port = GPU_CABLE_PORT[node.direction]
  return { x = node.position.x + port[1], y = node.position.y + port[2] }
end

local function hop_reach(from, to)
  local function reach(node)
    return not is_tile(node) and node.type == "electric-pole"
      and node.prototype.get_max_wire_distance(node.quality) or nil
  end
  local a, b = reach(from), reach(to)
  if a and b then return math.min(a, b) end
  return a or b or CABLE_REACH
end

local function hop_length(from, to)
  local a, b = node_position(from), node_position(to)
  return math.sqrt((b.x - a.x) ^ 2 + (b.y - a.y) ^ 2)
end

-- Sags like a wire; a tile end gets a plug.
local function draw_cable(surface, from, to, color, players)
  local ids = {}
  local a, b = cable_point(from), cable_point(to)
  local sag = 0.2 + 0.05 * math.sqrt((b.x - a.x) ^ 2 + (b.y - a.y) ^ 2)
  local previous = a
  for k = 1, CABLE_SEGMENTS do
    local t = k / CABLE_SEGMENTS
    local point = {
      x = a.x + (b.x - a.x) * t,
      y = a.y + (b.y - a.y) * t + sag * 4 * t * (1 - t),
    }
    table.insert(ids, rendering.draw_line({
      surface = surface, from = previous, to = point, color = color or CABLE_COLOR, width = 2,
      players = players,
    }).id)
    previous = point
  end
  for _, node in ipairs({ from, to }) do
    if is_tile(node) then
      table.insert(ids, rendering.draw_circle({
        surface = surface, target = cable_point(node), radius = 0.18,
        color = color or CABLE_COLOR, filled = true, draw_on_ground = true, players = players,
      }).id)
    end
  end
  return ids
end

local function destroy_ids(ids)
  for _, id in pairs(ids or {}) do
    local object = rendering.get_object_by_id(id)
    if object then object.destroy() end
  end
end

-- Removes every link cut(link) picks; returns how many.
local function remove_links(cut)
  local removed, surfaces = 0, {}
  for key, link in pairs(storage.links or {}) do
    if cut(link) then
      destroy_ids(link.render_ids)
      storage.links[key] = nil
      removed = removed + 1
      surfaces[link.surface] = true
    end
  end
  for index in pairs(surfaces) do draw_mask_outline(game.surfaces[index]) end
  return removed
end

local function touching(key)
  return function(link) return link.ka == key or link.kb == key end
end

local function holding_cable(player)
  local cursor = player.cursor_stack
  return cursor and cursor.valid_for_read and cursor.name == DATA_CABLE
end

-- The cable in the player's hand: a reach circle around the node it hangs
-- from, and a line to the pole or server under the cursor, red when out of
-- reach. Mods can't follow the mouse itself.
local function draw_cable_preview(player)
  local state = ensure_player_state(player.index)
  destroy_ids(state.cable_preview)
  state.cable_preview = nil
  local from = state.cable
  if not (from and node_valid(from) and holding_cable(player)
      and state.cable_surface == player.surface.index) then
    return
  end
  local players = { player }
  local ids = {
    rendering.draw_circle({
      surface = player.surface, target = node_position(from), radius = hop_reach(from, { x = 0, y = 0 }),
      color = { r = 0.1, g = 0.22, b = 0.5, a = 0.5 }, width = 2, filled = false,
      draw_on_ground = true, players = players,
    }).id,
    rendering.draw_circle({
      surface = player.surface, target = cable_point(from), radius = 0.25,
      color = CABLE_COLOR, width = 2, filled = false, players = players,
    }).id,
  }
  local hovered = player.selected
  if hovered and hovered ~= from and (hovered.name == inference.GPU_SERVER or hovered.type == "electric-pole") then
    local color = hop_length(from, hovered) <= hop_reach(from, hovered) and CABLE_COLOR
      or { r = 1, g = 0.2, b = 0.15, a = 1 }
    for _, id in ipairs(draw_cable(player.surface, from, hovered, color, players)) do table.insert(ids, id) end
  end
  state.cable_preview = ids
end

local function node_at(surface, area)
  local entity = surface.find_entities_filtered({ area = area, name = inference.GPU_SERVER })[1]
    or surface.find_entities_filtered({ area = area, type = "electric-pole" })[1]
  if entity then return entity end
  local x, y = area_center_tile(area)
  if surface.get_tile(x, y).name == MASK_TILE then return { x = x, y = y } end
end

-- As with vanilla wires: click one node, then another, to join them, and
-- the same two again to separate them. The cable stays in hand from the
-- last pole or server, so painted tiles fan out from it.
local function cable_select(event, player)
  local state = ensure_player_state(player.index)
  local surface = event.surface
  local node = node_at(surface, event.area)
  if not node then
    player.print("[Factorion] A data cable joins GPU servers, electric poles and " ..
      "[item=" .. MASK_TILE .. "] paint.")
    return
  end
  local from = state.cable
  if not (from and node_valid(from) and state.cable_surface == surface.index) then
    state.cable, state.cable_surface = node, surface.index
    draw_cable_preview(player)
    return
  end
  local ka, kb = node_key(from, surface), node_key(node, surface)
  if ka == kb then return end
  if is_tile(from) and is_tile(node) then
    player.print("[Factorion] Run the cable from paint to a GPU server or a pole.")
    return
  end
  local key = ka < kb and ka .. "|" .. kb or kb .. "|" .. ka
  storage.links = storage.links or {}
  if storage.links[key] then
    remove_links(function(link) return link == storage.links[key] end)
  else
    local length, reach = hop_length(from, node), hop_reach(from, node)
    if length > reach then
      player.print(string.format("[Factorion] That is %.1f tiles away; this cable reaches %.1f.",
        length, reach))
      return
    end
    storage.links[key] = {
      a = from, b = node, ka = ka, kb = kb, surface = surface.index,
      render_ids = draw_cable(surface, from, node),
    }
    draw_mask_outline(surface)
  end
  if not is_tile(node) then state.cable = node end
  draw_cable_preview(player)
  inference.refresh_all(true)
end

-- Right-click a GPU server, pole or painted tile to remove all its data
-- links, as shift-click does for a pole's wires.
local function cable_cut(event, player)
  local state = ensure_player_state(player.index)
  state.cable = nil
  draw_cable_preview(player)
  local node = node_at(event.surface, event.area)
  local removed = node and remove_links(touching(node_key(node, event.surface))) or 0
  player.print(string.format("[Factorion] Removed %d data link(s).", removed))
  inference.refresh_all(true)
end

local function mask_tiles_changed(event)
  local changed = event.tile ~= nil and event.tile.name == MASK_TILE
  for _, tile in pairs(event.tiles) do
    if tile.name == MASK_TILE or (tile.old_tile and tile.old_tile.name == MASK_TILE) then
      changed = true
    end
  end
  if changed then draw_mask_outline(game.get_surface(event.surface_index)) end
end

script.on_event({
  defines.events.on_player_built_tile,
  defines.events.on_robot_built_tile,
  defines.events.on_player_mined_tile,
  defines.events.on_robot_mined_tile,
  defines.events.script_raised_set_tiles,
}, mask_tiles_changed)

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
  "small-electric-pole",
  "medium-electric-pole",
  "substation",
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
    only_in_alt_mode = true,
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

local function register_endpoint(entity, item)
  local role = endpoint_role(entity)
  if not role or not entity.unit_number then return nil end
  storage.endpoints = storage.endpoints or {}
  local config = storage.endpoints[entity.unit_number] or { seen_tick = game.tick }
  local item_changed = item and config.item and item ~= config.item
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
  return config
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
  elseif is_predicted or not player then
    entity.destroy({ raise_destroy = true })
  else
    player.mine_entity(entity, true)
  end
end

-- Keyed by unit number, holding the entity itself: get_entity_by_unit_number
-- can't find belts, inserters, assemblers or ghosts.
local function clear_predicted_entities(state, player)
  for _, entity in pairs(state.model_entities or {}) do
    if entity.valid then remove_entity(entity, player, true) end
  end
  state.model_entities = {}
end

-- The cluster computing request_id drops its unused forward-pass tokens and
-- idles once the current pass finishes.
local function release_cluster(request_id)
  for _, state in pairs(storage.servers or {}) do
    if state.computing == request_id then
      state.computing = nil
      if state.entity.valid then
        state.entity.get_inventory(defines.inventory.crafter_input).clear()
      end
    end
  end
end

local function cancel_pending(state)
  if not state.pending then return end
  local request_id = state.pending.request_id
  release_cluster(request_id)
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

local function source_receiving(config)
  return game.tick - (config.seen_tick or 0) < SOURCE_EMPTY_TICKS
end

local function refresh_marker_dialog(player, state)
  local frame = player.gui.screen[MARKER_DIALOG]
  local config = state.picker and endpoint_config(state.picker.entity)
  if not (frame and config and config.entity.valid) then
    if frame then close_marker_dialog(player, state) end
    return
  end
  gui.marker_status(frame, config.role, config.item, source_receiving(config), config.throughput_rate or 0)
end

local function open_marker_dialog(player, state, role, entity)
  close_marker_dialog(player, state)
  state.picker = { entity = entity }
  local config = endpoint_config(entity)
  local frame = gui.marker_window(player, MARKER_DIALOG, ITEM_PICKER, entity, role,
    config and config.item or get_default_item(role))
  refresh_marker_dialog(player, state)
  player.opened = frame
end

-- ----------------------------------------------------------------------------
-- planner: click a tile and pick the sink item, then click one tile per
-- ingredient source
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

-- The sink is placed for the player, so it is turned to face away from the
-- sources as in training; sources keep the rotation they were placed with.
local function orient_sink(plan)
  local cx, cy, sources = 0, 0, 0
  for _, marker in ipairs(plan.tiles) do
    if marker.role == "source" then
      cx, cy, sources = cx + marker.x, cy + marker.y, sources + 1
    end
  end
  cx, cy = cx / sources, cy / sources
  for _, marker in ipairs(plan.tiles) do
    if marker.role == "sink" and marker.entity.valid then
      local dx, dy = marker.x - cx, marker.y - cy
      if math.abs(dx) >= math.abs(dy) then
        marker.entity.direction = dx >= 0 and defines.direction.east or defines.direction.west
      else
        marker.entity.direction = dy >= 0 and defines.direction.south or defines.direction.north
      end
    end
  end
end

local function finish_plan(player, state)
  local plan = state.plan
  plan.done = true
  orient_sink(plan)
  close_panel(player)
  player.print("[Factorion] Belts placed. Belt each ingredient into its source " ..
    "and carry the product away from the sink. Paint the tiles the factory may use " ..
    "with [item=factorion-mask], then put a [item=factorion-gpu-server] against " ..
    "the paint and press Predict in its window.")
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
  local sources, reason = source_items(item)
  if not sources then
    player.print("[Factorion] Can't plan that: " .. reason .. ".")
    return
  end
  close_plan_dialog(player)
  if not place_plan_marker(player, plan, "sink", item, plan.sink_tile) then
    player.print("[Factorion] That tile is blocked; click another one.")
    state.plan = nil
    return
  end
  plan.sources, plan.next = sources, 1
  prompt_next_source(player, plan)
end


local function handle_tool_selection(event)
  if event.item == DATA_CABLE then
    local player = game.get_player(event.player_index)
    if player then cable_select(event, player) end
    return
  end
  if event.item ~= PLANNER then return end
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  if not player then return end
  local x, y = area_center_tile(event.area)
  remove_plan_markers(state)
  close_panel(player)
  state.plan = { sink_tile = { x = x, y = y }, markers = {}, tiles = {} }
  open_plan_dialog(player)
end

script.on_event(defines.events.on_player_selected_area, handle_tool_selection)

script.on_event(defines.events.on_selected_entity_changed, function(event)
  local player = game.get_player(event.player_index)
  if player and holding_cable(player) then draw_cable_preview(player) end
end)

script.on_event(defines.events.on_player_cursor_stack_changed, function(event)
  local player = game.get_player(event.player_index)
  if not player then return end
  if not holding_cable(player) then ensure_player_state(player.index).cable = nil end
  draw_cable_preview(player)
end)

-- Right-clicking with the planner cancels the plan.
script.on_event(defines.events.on_player_alt_selected_area, function(event)
  if event.item == DATA_CABLE then
    local player = game.get_player(event.player_index)
    if player then cable_cut(event, player) end
    return
  end
  if event.item ~= PLANNER then return end
  local state = ensure_player_state(event.player_index)
  local player = game.get_player(event.player_index)
  remove_plan_markers(state)
  if player then
    close_panel(player)
    close_plan_dialog(player)
    player.print("[Factorion] Plan cleared.")
  end
end)

local function marker_item_changed(event)
  if event.element.name ~= ITEM_PICKER then return end
  local state = ensure_player_state(event.player_index)
  local entity = state.picker and state.picker.entity
  local item = event.element.elem_value
  if item and entity and entity.valid then register_endpoint(entity, item) end
  local player = game.get_player(event.player_index)
  if player then refresh_marker_dialog(player, state) end
end

script.on_event(defines.events.on_gui_click, function(event)
  local close = event.element.valid and event.element.tags.factorion_close
  local player = game.get_player(event.player_index)
  if close and player then
    local frame = player.gui.screen[close]
    if frame then frame.destroy() end
    if close == MARKER_DIALOG then ensure_player_state(event.player_index).picker = nil end
    return
  end
  if inference.on_gui_click(event) then return end
  if event.element.valid and event.element.tags.factorion_plan == "place_sink" then
    if player then place_sink(player, ensure_player_state(event.player_index)) end
  end
end)

script.on_event("factorion-confirm-dialog", function(event)
  local player = game.get_player(event.player_index)
  if not player then return end
  local state = ensure_player_state(event.player_index)
  if player.gui.screen[PLAN_DIALOG] then place_sink(player, state) end
end)

script.on_event(defines.events.on_gui_text_changed, inference.on_gui_text_changed)
script.on_event(defines.events.on_gui_confirmed, inference.on_gui_confirmed)
script.on_event(defines.events.on_gui_checked_state_changed, inference.on_gui_checked_state_changed)
script.on_event(defines.events.on_gui_selection_state_changed, inference.on_gui_selection_state_changed)
script.on_event(defines.events.on_gui_elem_changed, function(event)
  if not event.element.valid or inference.on_gui_elem_changed(event) then return end
  marker_item_changed(event)
end)

-- The server keeps its built-in window only so clicking it raises this event;
-- the inference window replaces it straight away.
local function open_gpu_server(player, entity)
  if not entity or not entity.valid or entity.name ~= inference.GPU_SERVER then
    return false
  end
  inference.open(player, entity)
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
  event.element.destroy()
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
  -- A built model ghost keeps counting as its server's, so rerun/clear finds it.
  local owner = event.tags and storage.servers and storage.servers[event.tags.factorion_server]
  if owner and entity and entity.valid and entity.unit_number then
    owner.model_entities[entity.unit_number] = entity
  end
  if entity and entity.valid and entity.name == inference.GPU_SERVER then
    local copied = event.tags and event.tags.factorion_gpu
    if copied then
      local state = server_state(entity)
      state.conditions, state.read_state = copied.conditions, copied.read_state
      state.circuit_endpoints = copied.circuit_endpoints
    end
    draw_mask_outline(entity.surface)
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

local function gpu_settings(entity)
  local state = storage.servers and storage.servers[entity.unit_number]
  return state and {
    conditions = state.conditions, read_state = state.read_state,
    circuit_endpoints = state.circuit_endpoints,
  }
end

script.on_event(defines.events.on_entity_settings_pasted, function(event)
  local source = endpoint_config(event.source)
  if source and endpoint_role(event.destination) then
    register_endpoint(event.destination, source.item)
  end
  local copied = event.source.name == inference.GPU_SERVER
    and event.destination.name == inference.GPU_SERVER and gpu_settings(event.source)
  if copied then
    local state = server_state(event.destination)
    ---@diagnostic disable-next-line: undefined-field
    state.conditions = table.deepcopy(copied.conditions)
    state.read_state = copied.read_state
    state.circuit_endpoints = copied.circuit_endpoints
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
    local copied = entity.name == inference.GPU_SERVER and gpu_settings(entity)
    if copied then blueprint.set_blueprint_entity_tag(index, "factorion_gpu", copied) end
  end
end)

-- Outlines show in alt-mode, so redrawing them as it turns on heals any that
-- missed a paint change.
script.on_event(defines.events.on_player_toggled_alt_mode, function(event)
  for _, config in pairs(storage.endpoints or {}) do
    local entity = config.entity
    if entity then refresh_endpoint_alt_icon(entity, config) end
  end
  local player = game.get_player(event.player_index)
  if player then draw_mask_outline(player.surface) end
end)

local function endpoint_entity_removed(event)
  local entity = event.entity
  if not entity or not entity.unit_number then return end
  -- What a removed GPU server built stays, as the player's own entities.
  local server = storage.servers and storage.servers[entity.unit_number]
  if server then
    cancel_pending(server)
    if server.signals and server.signals.valid then server.signals.destroy() end
    for _, endpoint in pairs(server.virtual or {}) do destroy_ids(endpoint.render_ids) end
    storage.servers[entity.unit_number] = nil
  end
  if storage.entity_alerts then storage.entity_alerts[entity.unit_number] = nil end
  if entity.type == "electric-pole" or entity.name == inference.GPU_SERVER then
    remove_links(touching("e" .. entity.unit_number))
  end
  if entity.name == inference.GPU_SERVER then draw_mask_outline(entity.surface, entity) end
  local config = storage.endpoints and storage.endpoints[entity.unit_number] or nil
  destroy_endpoint_alt_icon(config)
  destroy_sink_rate_label(config)
  if storage.endpoints then storage.endpoints[entity.unit_number] = nil end
end

script.on_event(defines.events.on_player_mined_entity, endpoint_entity_removed)

-- Mining a server would otherwise hand over its forward-pass tokens.
local function gpu_pre_mined(event)
  local entity = event.entity
  if entity.valid and entity.name == inference.GPU_SERVER then
    entity.get_inventory(defines.inventory.crafter_input).clear()
  end
end
script.on_event(defines.events.on_pre_player_mined_item, gpu_pre_mined)
script.on_event(defines.events.on_robot_pre_mined, gpu_pre_mined)
script.on_event(defines.events.on_robot_mined_entity, endpoint_entity_removed)
script.on_event(defines.events.on_entity_died, endpoint_entity_removed)
script.on_event(defines.events.script_raised_destroy, endpoint_entity_removed)

-- How many of the endpoint's item newly reached tile x, y of belt entity
-- since the last tick. Transport lines run past the tile, so only items on
-- it count.
local function count_arrivals(config, entity, x, y)
  local left, right, top, bottom = x - 0.01, x + 1.01, y - 0.01, y + 1.01
  local previous, present, arrived = config.on_tile or {}, {}, 0
  for line_index = 1, entity.get_max_transport_line_index() do
    local line = entity.get_transport_line(line_index)
    for _, detail in ipairs(line.get_detailed_contents()) do
      local stack = detail.stack
      if stack.valid_for_read and stack.name == config.item then
        local position = line.get_line_item_position(detail.position)
        if position.x >= left and position.x <= right
            and position.y >= top and position.y <= bottom then
          present[detail.unique_id] = true
          if not previous[detail.unique_id] then arrived = arrived + stack.count end
        end
      end
    end
  end
  config.on_tile = present
  if next(present) then config.seen_tick = game.tick end
  return arrived
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
    else
      config.throughput_rate = (config.throughput_rate or 0) * decay
        + count_arrivals(config, entity, math.floor(entity.position.x),
          math.floor(entity.position.y)) * 60 * (1 - decay)
      if config.role == "sink" and game.tick % SINK_LABEL_REFRESH_TICKS == 0 then
        refresh_sink_rate_label(entity, config)
      end
    end
  end
  -- Circuit endpoints measure whatever belt the player runs over their tile.
  for _, state in pairs(storage.servers or {}) do
    for _, endpoint in pairs(state.virtual or {}) do
      local arrived = endpoint.belt and endpoint.belt.valid
        and count_arrivals(endpoint, endpoint.belt, endpoint.x, endpoint.y) or 0
      endpoint.throughput_rate = (endpoint.throughput_rate or 0) * decay + arrived * 60 * (1 - decay)
      if endpoint.label and game.tick % SINK_LABEL_REFRESH_TICKS == 0 then
        local label = rendering.get_object_by_id(endpoint.label)
        if label then label.text = string.format("%.1f/s", endpoint.throughput_rate) end
      end
    end
  end
end

inference.on_load = function()
  storage.model_loading = nil
  request_model_load()
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

-- Every release publishes the mod and the server under the same version;
-- 0.0.0 is a development checkout.
local function server_command()
  local version = script.active_mods["factorion"]
  if version == "0.0.0" then return "./start-mod.sh from the factorion repo" end
  return "uvx factorion-server@" .. version
end

-- game.is_multiplayer() is false for a GUI host playing alone, even with
-- RCON open, so it can't tell a hosted game from single-player.
local function server_help()
  if recently(storage.server_seen_tick) then
    return string.format(
      "A model server is reaching the game, but it speaks protocol %s and " ..
      "this mod speaks %s. Whichever side is older needs restarting: quit " ..
      "and re-host Factorio to load the latest mod, or restart the server " ..
      "with %s.",
      tostring(storage.server_protocol or "unknown"), PROTOCOL_VERSION,
      server_command())
  end
  return "In a terminal, run " .. server_command() .. " (it needs uv: " ..
    "https://docs.astral.sh/uv). Factorio only lets the server in while " ..
    "hosting, so load your game with Play > Multiplayer > Host saved game " ..
    "(or Host new game)."
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


-- ----------------------------------------------------------------------------
-- execute: write a request JSON the server picks up
-- ----------------------------------------------------------------------------

local function new_request_id()
  return string.format("%d-%d-%d", game.tick,
    math.random(0, 2^30), math.random(0, 2^30))
end

-- The entities wholly inside fp, as the model sees them: the player's, and
-- what this GPU server built before.
local function gather_existing_entities(predicted, surface, fp, endpoint_tiles)
  local result, previous = {}, {}
  local candidates = surface.find_entities_filtered({
    area = { { fp.x, fp.y }, { fp.x + fp.w, fp.y + fp.h } },
    name = MODEL_ENTITY_NAMES,
  })
  for _, entity in pairs(candidates) do
    local at = tile_key(math.floor(entity.position.x), math.floor(entity.position.y))
    if not entity.to_be_deconstructed() and not endpoint_tiles[at] then
      local facingless = entity.type == "assembling-machine" or entity.type == "electric-pole"
      local direction = facingless and 0 or FACTORIO_TO_DIRECTION[entity.direction]
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
          table.insert(predicted[entity.unit_number] and previous or result, entry)
        end
      end
    end
  end
  return result, previous
end

-- A server reading endpoints from its circuit takes each item signal on its
-- red wire as a source and on its green wire as a sink, valued
-- X + 100 Y + 10000 D, which combinators can compute: D is 1-4 for north,
-- east, south or west, and X and Y count tiles from the painted area's
-- top-left corner, so 0 is just outside it. Returns the endpoints, in world
-- tiles, and the item signals that don't decode.
local function circuit_endpoints(server, min_x, min_y)
  local endpoints, ignored = {}, {}
  for role, wire in pairs({
    source = defines.wire_connector_id.circuit_red,
    sink = defines.wire_connector_id.circuit_green,
  }) do
    for _, signal in ipairs(server.get_signals(wire) or {}) do
      if (signal.signal.type or "item") == "item" then
        local value = signal.count
        local d = math.floor(value / 10000)
        if d >= 1 and d <= 4 then
          table.insert(endpoints, {
            x = min_x + value % 100 - 1, y = min_y + math.floor(value / 100) % 100 - 1,
            role = role, item = signal.signal.name, direction = d,
          })
        else
          table.insert(ignored, "[item=" .. signal.signal.name .. "]")
        end
      end
    end
  end
  return endpoints, ignored
end

-- The server's painted areas, the sources and sinks on or beside them, and
-- the model's grid around them all; or nil and why not.
local function region_for(server, groups)
  if not storage.model then
    request_model_load()
    return nil, "no model is loaded yet"
  end
  local size = storage.model.grid_size
  local pieces, reason = cluster_pieces(server, size, groups)
  if not pieces then return nil, reason end
  -- Paint bigger than the grid is cut to the first window, the server's own.
  local window
  for _, piece in ipairs(pieces) do window = window or piece.window end
  local area = {}
  for _, piece in ipairs(pieces) do
    for key, tile in pairs(piece.area) do
      if not window or (tile.x >= window.x and tile.x < window.x + size
          and tile.y >= window.y and tile.y < window.y + size) then
        area[key] = tile
      end
    end
  end
  local min_x, min_y, max_x, max_y = math.huge, math.huge, -math.huge, -math.huge
  local tiles = 0
  for _, tile in pairs(area) do
    min_x, max_x = math.min(min_x, tile.x), math.max(max_x, tile.x)
    min_y, max_y = math.min(min_y, tile.y), math.max(max_y, tile.y)
    tiles = tiles + 1
  end
  local function touches(x, y)
    if window and not (x >= window.x and x < window.x + size
        and y >= window.y and y < window.y + size) then
      return false
    end
    if area[tile_key(x, y)] then return true end
    for _, step in ipairs(NEIGHBOURS) do
      if area[tile_key(x + step[1], y + step[2])] then return true end
    end
    return false
  end
  local candidates, ignored = {}, {}
  if server_state(server).circuit_endpoints then
    candidates, ignored = circuit_endpoints(server, min_x, min_y)
  else
    for _, entity in pairs(server.surface.find_entities_filtered({
      area = { { min_x - 1, min_y - 1 }, { max_x + 2, max_y + 2 } },
      name = { ENDPOINT_ENTITY_NAMES.source, ENDPOINT_ENTITY_NAMES.sink },
    })) do
      local config = register_endpoint(entity)
      table.insert(candidates, {
        x = math.floor(entity.position.x), y = math.floor(entity.position.y),
        role = config.role, item = config.item,
        direction = FACTORIO_TO_DIRECTION[entity.direction], unit = entity.unit_number,
      })
    end
  end
  local endpoints = {}
  for _, endpoint in ipairs(candidates) do
    if touches(endpoint.x, endpoint.y) then
      table.insert(endpoints, endpoint)
      min_x, max_x = math.min(min_x, endpoint.x), math.max(max_x, endpoint.x)
      min_y, max_y = math.min(min_y, endpoint.y), math.max(max_y, endpoint.y)
    elseif not endpoint.unit then
      table.insert(ignored, "[item=" .. endpoint.item .. "]")
    end
  end
  local span_x, span_y = max_x - min_x + 1, max_y - min_y + 1
  if span_x > size or span_y > size then
    return nil, string.format("the paint and its belts span %dx%d tiles, more " ..
      "than the model's %dx%d grid", span_x, span_y, size, size)
  end
  local fp = window or {
    x = min_x - math.floor((size - span_x) / 2),
    y = min_y - math.floor((size - span_y) / 2),
    w = size, h = size,
  }
  local region = {
    fp = fp, tiles = tiles, allowed = {}, sources = {}, sinks = {}, endpoints = endpoints,
    ignored = ignored,
  }
  for _, tile in pairs(area) do
    table.insert(region.allowed, { x = tile.x - fp.x, y = tile.y - fp.y })
  end
  for _, endpoint in ipairs(endpoints) do
    table.insert(endpoint.role == "source" and region.sources or region.sinks, {
      x = endpoint.x - fp.x, y = endpoint.y - fp.y,
      direction = endpoint.direction, item = endpoint.item,
    })
  end
  return region
end

local function icons(endpoints)
  local parts = {}
  for _, endpoint in ipairs(endpoints) do
    table.insert(parts, "[item=" .. endpoint.item .. "]")
  end
  return table.concat(parts, " ")
end

-- The description of the request a server's cluster is computing.
local function computing_description(state)
  local owner = storage.servers[storage.pending_by_request[state.computing] or 0]
  return owner and owner.pending and owner.pending.description or "a factory"
end

-- The region server would build now, or nil and why it can't.
local function predictable_region(server)
  local state = server_state(server)
  if state.computing then
    return nil, "Building " .. computing_description(state) .. "…"
  end
  if server.energy <= 0 then return nil, "This GPU server has no power." end
  local region, reason = region_for(server)
  if not region then return nil, "Can't predict: " .. reason .. "." end
  if #region.sources == 0 then
    return nil, "No source on or beside the painted area."
  end
  if #region.sinks == 0 then
    return nil, "No sink on or beside the painted area."
  end
  return region
end

local function step_seconds(servers)
  return setting("factorion-forward-pass-time") / math.max(1, math.min(#servers, MAX_CLUSTER_SPEEDUP))
end

inference.describe = function(server)
  local members = cluster(server)
  local on = powered(members)
  local speed = #on == 0 and "none powered" or string.format(
    "%d powered: %.2f s per denoising step", #on, step_seconds(on))
  local cluster_caption = #members == 1
    and string.format("No other GPU server touches, shares paint or is cabled to this one; %s.", speed)
    or string.format("Cluster of %d GPU servers predicting together, %s.", #members, speed)
  local region, reason = predictable_region(server)
  if not region then return reason, false, cluster_caption end
  local caption = string.format("Painted area of %d tiles, converting %s to %s.",
    region.tiles, icons(region.sources), icons(region.sinks))
  if #region.ignored > 0 then
    caption = caption .. " Ignoring " .. table.concat(region.ignored, " ") ..
      " on the wires: not X + 100 Y + 10000 D on or beside the paint."
  end
  return caption, true, cluster_caption
end

local function gather_request(state, region, members)
  local predicted = {}
  for _, member in ipairs(members) do
    for unit, entity in pairs(server_state(member).model_entities or {}) do predicted[unit] = entity end
  end
  local model_entities, previous = gather_existing_entities(predicted, state.entity.surface,
    region.fp, state.endpoint_tiles)
  local description = string.format("converting %s to %s on %d painted tiles",
    icons(region.sources), icons(region.sinks), region.tiles)
  if #model_entities > 0 then
    description = description .. string.format(
      " around %d existing entities", #model_entities)
  end
  if #previous > 0 then
    description = description .. string.format(
      ", starting from the %d it built before", #previous)
  end

  local config = inference.config()
  local request = {
    type          = "predict",
    request_id    = new_request_id(),
    checkpoint    = storage.model.checkpoint,
    grid_size     = region.fp.w,
    step_delay_s  = step_seconds(members),
    temperature   = config.temperature,
    steps         = config.steps,
    refine        = config.refine,
    best_of       = config.best_of,
    seed          = config.seed,
    entities      = model_entities,
    start_entities = previous,
    allowed       = region.allowed,
    sources       = region.sources,
    sinks         = region.sinks,
  }
  return request, description
end

-- Rebuilds the factory in the painted area with every powered server of
-- server's cluster, starting from what they built before. Returns the
-- message for the player and whether it started; player is nil when a
-- circuit condition asked.
inference.on_predict = function(player, server, quiet)
  ensure_pending_lookup()
  local region, reason = predictable_region(server)
  if not region then return reason, false end
  local state = server_state(server)
  local members = powered(cluster(server))
  -- The player's own belts feed circuit endpoints, so the model leaves them.
  state.endpoint_tiles = {}
  for _, endpoint in ipairs(region.endpoints) do
    state.endpoint_tiles[tile_key(endpoint.x, endpoint.y)] = true
  end
  local request, description = gather_request(state, region, members)
  local request_json = helpers.table_to_json(request)
  table.insert(storage.outbox, request_json)
  storage.pending_by_request[request.request_id] = server.unit_number
  state.footprint = region.fp
  state.player_index = player and player.index
  state.quiet = quiet
  state.pending = {
    request_id = request.request_id, tick = game.tick, description = description,
  }
  for _, member in ipairs(members) do
    local member_state = server_state(member)
    member_state.computing = request.request_id
    member_state.steps = 0
  end
  log("[Factorion] enqueued request " .. request.request_id ..
      " json=" .. request_json)
  local changed = {}
  for _, knob in ipairs(SAMPLING_SUMMARY) do
    local value = inference.config()[knob[1]]
    if value ~= inference.DEFAULTS[knob[1]] then
      table.insert(changed, string.format(knob[2], value))
    end
  end
  local message = string.format("Requesting a factory %s using checkpoint %s on %d GPU server(s)%s.",
    description, request.checkpoint, #members,
    #changed > 0 and " (" .. table.concat(changed, ", ") .. ")" or "")
  if not server_alive() then
    message = message .. " No model server is connected, so nothing will " ..
      "happen yet. " .. server_help()
  end
  return message, true
end

script.on_event("factorion-predict", function(event)
  local player = game.get_player(event.player_index)
  local server = player and player.selected
  if not (server and server.name == inference.GPU_SERVER) then return end
  player.print("[Factorion] " .. inference.on_predict(player, server))
end)

-- Stops the cluster's prediction and removes everything it built.
inference.on_clear = function(player, server)
  for _, member in ipairs(cluster(server)) do
    local state = server_state(member)
    cancel_pending(state)
    clear_predicted_entities(state, player)
  end
  return "Cleared this cluster's factory."
end

-- Marks every entity of the kinds the model builds on the server's paint
-- for robots to deconstruct, and drops unbuilt ghosts; returns how many.
local function deconstruct_region(server, player)
  if not storage.model then return 0 end
  local members = cluster(server)
  for _, member in ipairs(members) do
    local state = server_state(member)
    cancel_pending(state)
    state.model_entities = {}
  end
  local feeds = {}
  for _, endpoint in pairs(server_state(server).virtual or {}) do
    feeds[tile_key(endpoint.x, endpoint.y)] = true
  end
  local marked = 0
  for _, piece in ipairs(cluster_pieces(server, storage.model.grid_size) or {}) do
    local min_x, min_y, max_x, max_y = math.huge, math.huge, -math.huge, -math.huge
    for _, tile in pairs(piece.area) do
      min_x, max_x = math.min(min_x, tile.x), math.max(max_x, tile.x)
      min_y, max_y = math.min(min_y, tile.y), math.max(max_y, tile.y)
    end
    local area = { { min_x, min_y }, { max_x + 1, max_y + 1 } }
    local found = server.surface.find_entities_filtered({ area = area, name = MODEL_ENTITY_NAMES })
    for _, ghost in pairs(server.surface.find_entities_filtered({ area = area, ghost_name = MODEL_ENTITY_NAMES })) do
      table.insert(found, ghost)
    end
    for _, entity in pairs(found) do
      local at = tile_key(math.floor(entity.position.x), math.floor(entity.position.y))
      if piece.area[at] and not feeds[at] then
        if entity.type == "entity-ghost" then
          entity.destroy({ raise_destroy = true })
        elseif not entity.to_be_deconstructed() then
          entity.order_deconstruction(entity.force, player)
          marked = marked + 1
        end
      end
    end
  end
  return marked
end

-- ----------------------------------------------------------------------------
-- GPU servers: power, utilisation, circuit conditions and state signals
-- ----------------------------------------------------------------------------

local WIRES = { defines.wire_connector_id.circuit_red, defines.wire_connector_id.circuit_green }
local UTILISATION_DECAY = math.exp(-10 / UTILISATION_TICKS)

-- A hidden constant combinator wired to the server writes its state onto
-- whatever networks the player wires to it.
local function ensure_signals(state)
  if state.signals and state.signals.valid then return state.signals end
  local server = state.entity
  local combinator = server.surface.create_entity({
    name = GPU_SIGNALS, position = server.position, force = server.force,
  })
  for _, wire in ipairs(WIRES) do
    combinator.get_wire_connector(wire, true).connect_to(
      server.get_wire_connector(wire, true), false, defines.wire_origin.script)
  end
  state.signals = combinator
  return combinator
end

local function wired(server)
  for _, wire in ipairs(WIRES) do
    local connector = server.get_wire_connector(wire, false)
    for _, connection in ipairs(connector and connector.connections or {}) do
      if connection.origin == defines.wire_origin.player then return true end
    end
  end
  return false
end

local function write_state_signals(state)
  local behavior = ensure_signals(state).get_control_behavior()
  local section = behavior.get_section(1) or behavior.add_section()
  local throughput = 0
  for _, unit in ipairs(state.sink_units or {}) do
    local config = storage.endpoints[unit]
    if config then throughput = throughput + (config.throughput_rate or 0) end
  end
  for _, endpoint in pairs(state.virtual or {}) do
    if endpoint.role == "sink" then throughput = throughput + (endpoint.throughput_rate or 0) end
  end
  local filters = {}
  for _, entry in ipairs(state.read_state ~= false and {
    { "predicting", state.computing and 1 or 0 },
    { "ready", state.ready and 1 or 0 },
    { "utilisation", math.floor((state.utilisation or 0) * 100 + 0.5) },
    { "cluster", state.cluster_size or 0 },
    { "step", state.steps or 0 },
    { "throughput", math.floor(throughput * 60 + 0.5) },
  } or {}) do
    if entry[2] ~= 0 then
      table.insert(filters, {
        value = { type = "virtual", name = "factorion-" .. entry[1], quality = "normal" },
        min = entry[2],
      })
    end
  end
  section.filters = filters
end

local function draw_utilisation(state)
  local u = state.utilisation or 0
  local text = string.format("GPU %d%%", math.floor(u * 100 + 0.5))
  local color = { r = 0.7 - 0.5 * u, g = 0.7 + 0.3 * u, b = 0.7 - 0.5 * u }
  local label = state.utilisation_label and rendering.get_object_by_id(state.utilisation_label)
  if label then
    label.text, label.color = text, color
    return
  end
  state.utilisation_label = rendering.draw_text({
    text = text, surface = state.entity.surface, target = state.entity,
    color = color, scale = 1.2, alignment = "center", vertical_alignment = "middle",
    only_in_alt_mode = true,
  }).id
end

-- Every 10 ticks: a server computing for its cluster always has its next
-- forward-pass token waiting, so it crafts, and draws full power, until the
-- prediction ends.
local function service_gpu_servers()
  for _, state in pairs(storage.servers or {}) do
    local server = state.entity
    if server.valid then
      if state.computing and server.energy > 0 then
        local inventory = server.get_inventory(defines.inventory.crafter_input)
        local short = 2 - inventory.get_item_count(FORWARD_PASS)
        if short > 0 then inventory.insert({ name = FORWARD_PASS, count = short }) end
      end
      local status = server.status
      local busy = (status == defines.entity_status.working
        or status == defines.entity_status.low_power) and 1 or 0
      state.utilisation = (state.utilisation or 0) * UTILISATION_DECAY + busy * (1 - UTILISATION_DECAY)
      write_state_signals(state)
      draw_utilisation(state)
    end
  end
end

local CIRCUIT_ACTIONS = {
  predict = function(server, player) inference.on_predict(player, server, true) end,
  deconstruct = function(server, player) deconstruct_region(server, player) end,
}

-- Every tick, so a one-tick pulse counts; each condition acts once each time
-- it becomes true.
local function check_circuit_conditions()
  for _, state in pairs(storage.servers or {}) do
    if state.conditions and state.entity.valid then
      state.fired = state.fired or {}
      for key, action in pairs(CIRCUIT_ACTIONS) do
        local condition = state.conditions[key]
        local now = condition ~= nil and gui.fulfilled(state.entity, condition, state.circuit_endpoints)
        if now and not state.fired[key] then
          action(state.entity, state.circuit_player and game.get_player(state.circuit_player))
        end
        state.fired[key] = now
      end
    end
  end
end

local ENDPOINT_COLORS = {
  source = { r = 0.2, g = 1.0, b = 0.25, a = 0.9 },
  sink = { r = 1.0, g = 0.32, b = 0.08, a = 0.9 },
}
local ARROWS = { { 0, -1 }, { 1, 0 }, { 0, 1 }, { -1, 0 } }
local BELT_TYPES = { "transport-belt", "underground-belt", "splitter" }

-- An outline in the endpoint belt's colour, its item and its direction.
local function draw_virtual_endpoint(surface, endpoint)
  local x, y = endpoint.x, endpoint.y
  local cx, cy = x + 0.5, y + 0.5
  local color = ENDPOINT_COLORS[endpoint.role]
  local d = ARROWS[endpoint.direction]
  local tip = { cx + d[1] * 0.42, cy + d[2] * 0.42 }
  local back = { cx + d[1] * 0.27, cy + d[2] * 0.27 }
  local ids = {
    rendering.draw_rectangle({
      surface = surface, color = color, width = 3, draw_on_ground = true,
      left_top = { x + 0.06, y + 0.06 }, right_bottom = { x + 0.94, y + 0.94 },
    }).id,
    rendering.draw_sprite({
      sprite = "item/" .. endpoint.item, surface = surface, target = { cx, cy },
      x_scale = 0.4, y_scale = 0.4,
    }).id,
  }
  for _, side in ipairs({ 1, -1 }) do
    table.insert(ids, rendering.draw_line({
      surface = surface, color = color, width = 3, from = tip,
      to = { back[1] - d[2] * 0.14 * side, back[2] + d[1] * 0.14 * side },
    }).id)
  end
  if endpoint.role == "sink" then
    endpoint.label = rendering.draw_text({
      text = "0.0/s", surface = surface, target = { cx, cy - 0.7 },
      color = { r = 1.0, g = 0.72, b = 0.2, a = 1.0 }, scale = 0.8,
      alignment = "center", vertical_alignment = "bottom", only_in_alt_mode = true,
    }).id
    table.insert(ids, endpoint.label)
  end
  return ids
end

-- Keeps the server's circuit endpoints drawn, measured from whatever belt
-- is on their tile, and flagged when a source has nothing arriving.
local function sync_virtual_endpoints(server, state, endpoints)
  local kept = {}
  for _, endpoint in ipairs(endpoints) do
    if not endpoint.unit then
      local key = table.concat({ endpoint.role, endpoint.x, endpoint.y, endpoint.item, endpoint.direction }, ",")
      local entry = state.virtual and state.virtual[key]
      if not entry then
        entry = endpoint
        entry.seen_tick = game.tick
        entry.render_ids = draw_virtual_endpoint(server.surface, entry)
      end
      entry.belt = server.surface.find_entities_filtered({
        position = { entry.x + 0.5, entry.y + 0.5 }, type = BELT_TYPES,
      })[1]
      local empty = entry.role == "source" and game.tick - entry.seen_tick >= SOURCE_EMPTY_TICKS
      if empty and not (entry.alert and rendering.get_object_by_id(entry.alert)) then
        entry.alert = draw_alert("source-empty", { entry.x + 0.5, entry.y + 0.5 }, server.surface)
        entry.render_ids.alert = entry.alert
      elseif not empty and entry.alert then
        destroy_ids({ entry.alert })
        entry.alert, entry.render_ids.alert = nil, nil
      end
      kept[key] = entry
    end
  end
  for key, entry in pairs(state.virtual or {}) do
    if not kept[key] then destroy_ids(entry.render_ids) end
  end
  state.virtual = kept
end

local function set_alert(entity, kind, offset)
  storage.entity_alerts = storage.entity_alerts or {}
  local current = storage.entity_alerts[entity.unit_number]
  local object = current and rendering.get_object_by_id(current.id)
  if current and current.kind == kind and object then return end
  if object then object.destroy() end
  storage.entity_alerts[entity.unit_number] = kind and {
    kind = kind, id = draw_alert(kind, entity, entity.surface, offset),
  } or nil
end

-- Every second: each server's cluster size, sinks and readiness for its
-- signals, and the warnings over servers and sources.
local function survey()
  local offline = storage.server_connected == false or storage.model_error ~= nil
  local size = storage.model and storage.model.grid_size
  for _, surface in pairs(game.surfaces) do
    local groups = clusters(surface)
    for _, server in ipairs(surface.find_entities_filtered({ name = inference.GPU_SERVER })) do
      local state = server_state(server)
      state.cluster_size = #powered(groups[server.unit_number])
      local region = size and region_for(server, groups)
      state.sink_units = {}
      for _, endpoint in ipairs(region and region.endpoints or {}) do
        if endpoint.unit and endpoint.role == "sink" then table.insert(state.sink_units, endpoint.unit) end
      end
      sync_virtual_endpoints(server, state, region and region.endpoints or {})
      state.ready = not state.computing and server.energy > 0 and server_alive()
        and region and #region.sources > 0 and #region.sinks > 0 or false
      local unlinked = size and not region and not cluster_pieces(server, size, groups)
      set_alert(server, offline and "gpu-offline" or (unlinked and "unlinked") or nil, { 0, -0.4 })
    end
  end
  for _, config in pairs(storage.endpoints or {}) do
    if config.entity.valid then
      set_alert(config.entity,
        config.role == "source" and not source_receiving(config) and "source-empty" or nil)
    end
  end
end

local function blink()
  local kept, visible = {}, game.tick % 60 < 30
  for _, id in ipairs(storage.blinking or {}) do
    local object = rendering.get_object_by_id(id)
    if object then
      object.visible = visible
      table.insert(kept, id)
    end
  end
  storage.blinking = kept
end

inference.info = function(server)
  local state = server_state(server)
  local status, caption = "idle", "Idle"
  if server.energy <= 0 then
    status, caption = "broken", "No power"
  elseif storage.server_connected == false then
    status, caption = "broken", "Can't reach the model server"
  elseif storage.model_error then
    status, caption = "broken", "The model failed to load"
  elseif not storage.model then
    status, caption = "waiting", "Loading the model…"
  elseif state.computing then
    status, caption = "working", string.format("Predicting, %d steps placed", state.steps or 0)
  end
  local u = state.utilisation or 0
  return {
    status = status, caption = caption, utilisation = u, wired = wired(server),
    power_kw = server.energy > 0 and IDLE_KW + (BUSY_KW - IDLE_KW) * u or 0,
  }
end

inference.settings = function(server)
  local state = server_state(server)
  state.conditions = state.conditions or {}
  return state
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
    "Factorio refused %s at relative tile (%s,%s), " ..
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
  local surface = state.entity.surface
  local left = fp.x + placement.tile_x
  local top = fp.y + placement.tile_y
  local predicted = state.model_entities or {}
  local area = {
    { left + 0.01, top + 0.01 },
    { left + placement.width - 0.01, top + placement.height - 0.01 },
  }
  local overlapped = surface.find_entities_filtered({
    area = area, name = MODEL_ENTITY_NAMES,
  })
  for _, ghost in pairs(surface.find_entities_filtered({
    area = area, ghost_name = MODEL_ENTITY_NAMES,
  })) do
    table.insert(overlapped, ghost)
  end
  for _, entity in pairs(overlapped) do
    local box = entity.bounding_box
    local at = tile_key(math.floor(entity.position.x), math.floor(entity.position.y))
    if not (state.endpoint_tiles or {})[at] and box.left_top.x >= fp.x and box.left_top.y >= fp.y
        and box.right_bottom.x <= fp.x + fp.w
        and box.right_bottom.y <= fp.y + fp.h then
      remove_entity(entity, player, predicted[entity.unit_number] ~= nil)
    end
  end
end

-- Create one model-predicted entity at footprint-relative coordinates,
-- replacing whatever it overlaps. Returns why Factorio refused it, if it did.
local function place_one(player, state, fp, placement)
  if not placement.tile_x or not placement.tile_y
      or not placement.width or not placement.height then
    return "incomplete placement"
  end
  if placement.tile_x < 0 or placement.tile_y < 0
      or placement.tile_x + placement.width > fp.w
      or placement.tile_y + placement.height > fp.h then
    return "placement lies outside the footprint"
  end

  clear_placement_area(player, state, fp, placement)
  if not placement.name then return nil end

  local surface = state.entity.surface
  local params = {
    name = placement.name,
    position = { fp.x + placement.x, fp.y + placement.y },
    direction = placement.direction,
    force = state.entity.force,
    player = player,
    raise_built = true,
    create_build_effect_smoke = true,
  }
  if placement.type then params.type = placement.type end
  if placement.recipe then params.recipe = placement.recipe end
  if setting("factorion-placement-mode") == "ghost" then
    params.inner_name = params.name
    params.name = "entity-ghost"
    params.tags = { factorion_server = state.entity.unit_number }
  end
  local created, entity = pcall(
    function() return surface.create_entity(params) end)
  if not created then
    return string.format(
      "create_entity raised while placing %s at relative tile " ..
      "(%s,%s): %s",
      tostring(placement.name),
      tostring(placement.tile_x),
      tostring(placement.tile_y),
      tostring(entity))
  end
  if not entity then
    return describe_placement_failure(surface, fp, placement)
  end

  if entity.unit_number then
    state.model_entities[entity.unit_number] = entity
  end
end

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

  -- Apply one denoising step: footprint-relative placements, each replacing
  -- whatever it overlaps (one without a name only deletes). A placement
  -- Factorio refuses doesn't stop the rest of the step.
  place_predictions = function(request_id, placements_json)
    heard_from_server()
    ensure_pending_lookup()
    local server = storage.pending_by_request[request_id]
    if not server then
      return "cancelled"
    end

    if server == 0 then
      log("[Factorion] (headless) placements for " .. tostring(request_id) ..
        ": " .. tostring(placements_json))
      return "ok"
    end

    local state = storage.servers[server]
    if not (state and state.entity.valid) then return "cancelled" end
    local player = state.player_index and game.get_player(state.player_index)
    local fp = state.footprint
    for _, member in pairs(storage.servers) do
      if member.computing == request_id then member.steps = (member.steps or 0) + 1 end
    end

    local ok, placements = pcall(
      function() return helpers.json_to_table(placements_json) end)
    if not ok or type(placements) ~= "table" then
      return "error: invalid placements JSON"
    end
    local refused, first_error = 0, nil
    for _, placement in ipairs(placements) do
      local err = place_one(player, state, fp, placement)
      if err then
        refused = refused + 1
        first_error = first_error or err
      end
    end
    if refused == 0 then return "ok" end
    return string.format("error: %d of %d placements refused; first: %s",
      refused, #placements, first_error)
  end,

  -- Complete the streamed request and release its pending state.
  finish_prediction = function(request_id, summary_json)
    heard_from_server()
    ensure_pending_lookup()
    local server = storage.pending_by_request[request_id]
    if not server then
      return "cancelled"
    end
    storage.pending_by_request[request_id] = nil
    release_cluster(request_id)
    if server == 0 then
      log("[Factorion] (headless) prediction complete for " ..
        tostring(request_id) .. ": " .. tostring(summary_json))
      return "ok"
    end

    local state = storage.servers[server]
    if not state then return "cancelled" end
    local player = state.player_index and game.get_player(state.player_index)
    local description = state.pending and state.pending.description
      or "for an unknown request"
    state.pending = nil
    local ok, summary = pcall(
      function() return helpers.json_to_table(summary_json) end)
    summary = ok and summary or {}
    if player and not state.quiet then
      local stop = STOP_REASONS[summary.stop_reason]
        or tostring(summary.stop_reason)
      player.print(string.format(
        "[Factorion] Finished %s after %s denoising steps, the best of %s " ..
        "chains at %.2f items/s; %s. Seed %s.",
        description,
        tostring(summary.steps_taken or "?"),
        tostring(summary.best_of or "?"),
        tonumber(summary.thput) or 0,
        stop,
        tostring(summary.seed or "?")))
      if (summary.refused_steps or 0) > 0 then
        player.print(string.format(
          "[Factorion] Factorio refused part of %d step(s); see the server log.",
          summary.refused_steps))
      end
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
    for _, surface in pairs(game.surfaces) do draw_mask_outline(surface) end
    inference.refresh_all(true)
    game.print(string.format(
      "[Factorion] Model %s (%s) ready: it builds painted areas up to %dx%d " ..
      "tiles. Plan a factory with the [item=factorion-planner].",
      model.checkpoint, tostring(model.name), model.grid_size, model.grid_size))
    return "ok"
  end,
  model_failed = function(error_json)
    heard_from_server()
    local failure = helpers.json_to_table(error_json) or {}
    if failure.checkpoint ~= get_checkpoint() then return "stale" end
    storage.model_loading = nil
    storage.model_error = tostring(failure.error)
    inference.refresh_all(true)
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

  -- Headless / debug: enqueue a request JSON as if Predict had been pressed.
  -- /silent-command runs in *level scope*, not any mod's, so storage is
  -- inaccessible from RCON directly — this interface is the only way to
  -- poke an outbox entry in from outside the game.
  --
  -- Caller supplies the request JSON and the unit number of the GPU server
  -- whose region it builds. In headless tests, pass 0 to signal "log
  -- streamed placements instead of creating world entities".
  inject_request = function(request_json, deliver_to_server)
    storage.outbox = storage.outbox or {}
    storage.pending_by_request = storage.pending_by_request or {}
    table.insert(storage.outbox, request_json)
    -- Parse to extract request_id; helpers.json_to_table is the 2.0+ API.
    local ok, parsed = pcall(function() return helpers.json_to_table(request_json) end)
    if ok and parsed and parsed.request_id then
      storage.pending_by_request[parsed.request_id] = deliver_to_server or 0
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
})

-- Endpoint belts drive the player's factory; parity drives its isolated lab
-- surface and early-outs when no run is active.
script.on_event(defines.events.on_tick, function()
  service_endpoint_belts()
  check_circuit_conditions()
  parity.on_tick()
end)

script.on_nth_tick(10, function()
  service_gpu_servers()
  inference.refresh_all(false)
  for _, player in pairs(game.connected_players) do
    refresh_marker_dialog(player, ensure_player_state(player.index))
  end
end)
script.on_nth_tick(30, blink)
script.on_nth_tick(60, function()
  check_server()
  survey()
  inference.refresh_all(true)
end)

-- ----------------------------------------------------------------------------
-- migrations
-- ----------------------------------------------------------------------------

script.on_init(function()
  storage.players = {}
  storage.servers = {}
  storage.pending_by_request = {}
  storage.outbox = {}
  storage.endpoints = {}
  storage.server_watch_tick = game.tick
  request_model_load()
end)

script.on_configuration_changed(function()
  storage.players = storage.players or {}
  storage.servers = storage.servers or {}
  storage.pending_by_request = storage.pending_by_request or {}
  storage.outbox = storage.outbox or {}
  storage.endpoints = storage.endpoints or {}
  -- A server whose prototype changed type is gone from the map.
  for unit, state in pairs(storage.servers) do
    -- Cables ran server, poles, tile; each hop becomes a link.
    for _, cable in ipairs(state.cables or {}) do
      destroy_ids(cable.render_ids)
      local nodes = { state.entity }
      for _, pole in ipairs(cable.poles) do table.insert(nodes, pole) end
      table.insert(nodes, { x = cable.x, y = cable.y })
      for i = 1, #nodes - 1 do
        local a, b = nodes[i], nodes[i + 1]
        if node_valid(a) and node_valid(b) and state.entity.valid then
          local surface = state.entity.surface
          local ka, kb = node_key(a, surface), node_key(b, surface)
          storage.links = storage.links or {}
          storage.links[ka < kb and ka .. "|" .. kb or kb .. "|" .. ka] = {
            a = a, b = b, ka = ka, kb = kb, surface = surface.index,
            render_ids = draw_cable(surface, a, b),
          }
        end
      end
    end
    state.cables = nil
    if not state.entity.valid then storage.servers[unit] = nil end
  end
  for _, player in pairs(game.players) do
    -- Saves keep open windows, including ones a newer mod version no longer
    -- manages; every Factorion window is transient, so none should persist.
    for _, element in pairs(player.gui.screen.children) do
      if element.name:find("^factorion%-") then element.destroy() end
    end
    ensure_player_state(player.index)
  end
  for _, surface in pairs(game.surfaces) do
    draw_mask_outline(surface)
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
