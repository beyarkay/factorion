-- Factorion parity runner: build a factory spec inside real Factorio,
-- run it for a fixed number of ticks, and measure per-sink / per-entity
-- throughput so the Python harness can compare against the Factorion
-- engine's steady-state prediction (factorion_rs.py_sink_deliveries).
--
-- Driven entirely over RCON via three remote-interface methods that
-- control.lua merges into the "factorion" interface:
--
--   parity_start(spec_json) -> status JSON ("running" or "error")
--   parity_poll()           -> status JSON; includes the full result once done
--   parity_abort()          -> tear down the active run
--
-- Spec JSON (built by server/parity.py):
--   {
--     run_id = "...",
--     grid_size = 11,
--     game_speed = 32,            -- game.speed while the run is active
--     sample_every = 15,          -- per-entity sampling period (ticks)
--     -- Adaptive measurement (all optional; defaults in M.start). Warmup
--     -- runs until the windowed delivery rate plateaus or warmup_max; then
--     -- counts reset and measure runs until the cumulative rate plateaus
--     -- or measure_max. Legacy warmup_ticks/measure_ticks are accepted as
--     -- warmup_max/measure_max.
--     warmup_min = 300, warmup_max = 1800,
--     measure_min = 600, measure_max = 36000,
--     check_every = 300,          -- convergence-check period (ticks)
--     converge_rel = 0.02,        -- plateau if rate within 2% across checks
--     converge_hits = 3,          -- consecutive stable checks to converge
--     converge_floor = 0.02,      -- abs items/s below which diffs are "stable"
--     entities = {                -- model-placed entities (no source/sink)
--       { name="transport-belt", x=3.5, y=5.5, tile_x=3, tile_y=5,
--         direction=4, type="input"|"output"|nil, recipe="iron-gear-wheel"|nil },
--       ...
--     },
--     sources = { { x=6, y=4, direction=4, item="iron-plate" }, ... },
--     sinks   = { { x=3, y=5, direction=12, item="iron-plate" }, ... },
--   }
--
-- Source/sink representation: the Factorion engine models a source as an
-- infinite item provider and a sink as an infinite consumer, each occupying
-- one directional tile that belts/inserters connect to like any other
-- belt. We reproduce that exactly by placing a real transport-belt on the
-- source/sink tile and scripting its transport lines every tick: source
-- lines are kept full (insert_at_back until refused), sink lines are
-- counted and cleared (so the sink never back-pressures). This keeps the
-- geometry 1:1 with the engine's grid and makes lane semantics
-- (side-loading, curves, inserter drop lanes) come from the real game.
--
-- x/y for entities are Factorio *center* coordinates relative to the grid
-- origin (blueprint convention: 1x1 entity at tile (3,5) sits at 3.5,5.5);
-- tile_x/tile_y echo the Factorion tensor anchor tile so the harness can
-- map results back to the world tensor. Source/sink x/y are tile coords.
--
-- The run happens on a dedicated lab-tiles surface ("factorion-parity")
-- with its own force ("factorion-parity", all recipes enabled, no
-- research so inserter stack size stays 1), powered by an
-- electric-energy-interface + substations placed outside the grid.

local M = {}

local SURFACE_NAME = "factorion-parity"
local FORCE_NAME = "factorion-parity"

-- Grid origin on the parity surface: tile (0,0) of the spec maps to world
-- tile (0,0). Kept at the map origin so a spectator can find it with
-- /c game.player.teleport({5, 5}, "factorion-parity").
local ORIGIN_X = 0
local ORIGIN_Y = 0

local function json_encode(t)
  return helpers.table_to_json(t)
end

local function json_decode(s)
  return helpers.json_to_table(s)
end

local COLOR_ERR = { r = 1.0, g = 0.35, b = 0.35 }
local COLOR_OK = { r = 0.55, g = 1.0, b = 0.55 }
local COLOR_SRC = { r = 0.4, g = 0.9, b = 0.4 }
local COLOR_SNK = { r = 0.4, g = 0.7, b = 1.0 }

-- All human-facing feedback goes through here: game.print reaches the
-- GUI chat AND the headless server console, log() lands in
-- factorio-current.log. A spectator (or Claude tailing the console)
-- sees build errors, phase transitions and live rates without having
-- to poll RCON.
local function announce(msg, color)
  local line = "[Factorion parity] " .. msg
  game.print(line, color and { color = color } or nil)
  log(line)
end

local function draw_label(surface, x, y, text, color, scale)
  return rendering.draw_text({
    text = text,
    surface = surface,
    target = { x, y },
    color = color,
    scale = scale or 1.0,
    alignment = "center",
  })
end

-- defines.entity_status is {name -> int}; results want names.
local STATUS_NAMES = {}
for name, value in pairs(defines.entity_status) do
  STATUS_NAMES[value] = name
end

-- Silence Factorio's alert beeps during a run. Parity factories are built
-- and torn down rapidly and often sit momentarily idle/unpowered, which
-- would otherwise spam the "no power"/"no input" alert sounds. Disabling
-- every alert type per player mutes both the icons and their sounds; it's
-- a persistent per-player setting so it survives across runs.
local function mute_alerts()
  for _, p in pairs(game.players) do
    for _, alert_type in pairs(defines.alert_type) do
      pcall(function() p.disable_alert(alert_type) end)
    end
  end
end

local function ensure_storage()
  storage.parity = storage.parity or {}
  return storage.parity
end

-- ----------------------------------------------------------------------------
-- surface / force setup
-- ----------------------------------------------------------------------------

local function ensure_force()
  if game.forces[FORCE_NAME] then return game.forces[FORCE_NAME] end
  local force = game.create_force(FORCE_NAME)
  -- The spec can carry any recipe the engine knows; none of them are
  -- researched on a fresh force. Enabling everything is safe here because
  -- this force exists only on the parity surface.
  force.enable_all_recipes()
  return force
end

local function ensure_surface()
  local surface = game.surfaces[SURFACE_NAME]
  if not surface then
    surface = game.create_surface(SURFACE_NAME)
    surface.generate_with_lab_tiles = true
    surface.always_day = true
  end
  return surface
end

local function generate_area(surface, grid_size)
  -- Chunks must exist before create_entity; force_generate blocks until done.
  local radius_chunks = math.ceil((grid_size + 24) / 32) + 1
  surface.request_to_generate_chunks({ ORIGIN_X, ORIGIN_Y }, radius_chunks)
  surface.force_generate_chunk_requests()
end

local function clear_surface(surface)
  -- Wipe the previous run. Spare characters so a spectating player who
  -- teleported over doesn't get deleted with the old factory.
  for _, e in pairs(surface.find_entities_filtered({})) do
    if e.valid and e.type ~= "character" then e.destroy() end
  end
end

local function place_power(surface, force, grid_size)
  -- One electric-energy-interface (free infinite power, hidden base
  -- prototype) next to a ring of substations outside the grid. A
  -- substation's supply area is 18x18 centred on it, so corners +
  -- edge-midpoints cover any grid the training sizes use (<= ~26).
  local margin = 3
  local lo = -margin
  local hi = grid_size + margin
  local mid = math.floor(grid_size / 2)
  local spots = {
    { lo, lo }, { hi, lo }, { lo, hi }, { hi, hi },
    { mid, lo }, { mid, hi }, { lo, mid }, { hi, mid },
  }
  local placed = {}
  for _, s in ipairs(spots) do
    local e = surface.create_entity({
      name = "substation",
      position = { ORIGIN_X + s[1], ORIGIN_Y + s[2] },
      force = force,
      raise_built = false,
      create_build_effect_smoke = false,
    })
    if e then table.insert(placed, e) end
  end
  surface.create_entity({
    name = "electric-energy-interface",
    position = { ORIGIN_X - margin - 3, ORIGIN_Y - margin - 3 },
    force = force,
    raise_built = false,
    create_build_effect_smoke = false,
  })
  return placed
end

-- ----------------------------------------------------------------------------
-- build
-- ----------------------------------------------------------------------------

local BELT_LIKE = {
  ["transport-belt"] = true,
  ["fast-transport-belt"] = true,
  ["underground-belt"] = true,
  ["fast-underground-belt"] = true,
  ["splitter"] = true,
  ["fast-splitter"] = true,
}

local INSERTER_LIKE = {
  ["inserter"] = true,
  ["long-handed-inserter"] = true,
  ["fast-inserter"] = true,
  ["burner-inserter"] = true,
}

local function entity_kind(name)
  if BELT_LIKE[name] then return "belt" end
  if INSERTER_LIKE[name] then return "inserter" end
  if string.find(name, "assembling%-machine") then return "machine" end
  return "other"
end

local function build_factory(surface, force, spec)
  local errors = {}
  local watched = {}

  for _, e in ipairs(spec.entities or {}) do
    local created = surface.create_entity({
      name = e.name,
      position = { ORIGIN_X + e.x, ORIGIN_Y + e.y },
      direction = e.direction,
      force = force,
      type = e.type,
      recipe = e.recipe,
      raise_built = false,
      create_build_effect_smoke = false,
    })
    if not created then
      table.insert(errors, string.format(
        "create_entity failed: %s at (%s,%s)", e.name,
        tostring(e.x), tostring(e.y)))
    else
      table.insert(watched, {
        entity = created,
        name = e.name,
        kind = entity_kind(e.name),
        x = e.tile_x,
        y = e.tile_y,
        -- accumulators, filled during the measure phase
        line_totals = {},
        status_counts = {},
        held_ticks = 0,
        products_finished_start = 0,
        samples = 0,
      })
    end
  end

  local sources = {}
  for _, s in ipairs(spec.sources or {}) do
    local belt = surface.create_entity({
      name = "transport-belt",
      position = { ORIGIN_X + s.x + 0.5, ORIGIN_Y + s.y + 0.5 },
      direction = s.direction,
      force = force,
      raise_built = false,
      create_build_effect_smoke = false,
    })
    if not belt then
      table.insert(errors, string.format(
        "source belt failed at (%d,%d)", s.x, s.y))
    else
      table.insert(sources, {
        belt = belt, item = s.item, x = s.x, y = s.y, inserted = 0,
      })
    end
  end

  local sinks = {}
  for _, s in ipairs(spec.sinks or {}) do
    local belt = surface.create_entity({
      name = "transport-belt",
      position = { ORIGIN_X + s.x + 0.5, ORIGIN_Y + s.y + 0.5 },
      direction = s.direction,
      force = force,
      raise_built = false,
      create_build_effect_smoke = false,
    })
    if not belt then
      table.insert(errors, string.format(
        "sink belt failed at (%d,%d)", s.x, s.y))
    else
      table.insert(sinks, {
        belt = belt, item = s.item, x = s.x, y = s.y, counts = {},
      })
    end
  end

  return watched, sources, sinks, errors
end

-- ----------------------------------------------------------------------------
-- per-tick simulation hooks
-- ----------------------------------------------------------------------------

local function feed_sources(run, count_it)
  for _, src in ipairs(run.sources) do
    local belt = src.belt
    if belt.valid then
      for li = 1, belt.get_max_transport_line_index() do
        local line = belt.get_transport_line(li)
        -- A 1-tile lane holds 4 items; 8 is a safe upper bound per tick.
        for _ = 1, 8 do
          if not line.can_insert_at_back() then break end
          if not line.insert_at_back({ name = src.item, count = 1 }) then
            break
          end
          if count_it then src.inserted = src.inserted + 1 end
        end
      end
    end
  end
end

local function drain_sinks(run, count_it)
  for _, snk in ipairs(run.sinks) do
    local belt = snk.belt
    if belt.valid then
      for li = 1, belt.get_max_transport_line_index() do
        local line = belt.get_transport_line(li)
        if line.get_item_count() > 0 then
          if count_it then
            for _, stack in pairs(line.get_contents()) do
              snk.counts[stack.name] = (snk.counts[stack.name] or 0) + stack.count
            end
          end
          line.clear()
        end
      end
    end
  end
end

local function sample_entities(run)
  for _, w in ipairs(run.watched) do
    local e = w.entity
    if e.valid then
      w.samples = w.samples + 1
      if w.kind == "belt" then
        for li = 1, e.get_max_transport_line_index() do
          w.line_totals[li] = (w.line_totals[li] or 0)
            + e.get_transport_line(li).get_item_count()
        end
      else
        local status = e.status
        if status ~= nil then
          local sname = STATUS_NAMES[status] or tostring(status)
          w.status_counts[sname] = (w.status_counts[sname] or 0) + 1
          -- Live overlay: a red status tag above any machine/inserter
          -- that isn't plain "working" right now, so a spectator sees
          -- *where* flow is stalling as it happens.
          if status ~= defines.entity_status.working then
            if w.warn and w.warn.valid then
              w.warn.text = sname
            else
              w.warn = draw_label(e.surface,
                ORIGIN_X + w.x + 0.5, ORIGIN_Y + w.y + 0.2,
                sname, COLOR_ERR, 0.7)
            end
          elseif w.warn and w.warn.valid then
            w.warn.destroy()
            w.warn = nil
          end
        end
        if w.kind == "inserter" and e.held_stack.valid_for_read then
          w.held_ticks = w.held_ticks + 1
        end
      end
    end
  end
end

-- One line per sink with the rate measured so far, e.g.
-- "sink(3,5) iron-plate 14.87/s". Used by chat progress + labels.
local function sink_rate_lines(run, measure_elapsed_ticks)
  local seconds = math.max(measure_elapsed_ticks, 1) / 60.0
  local lines = {}
  for _, snk in ipairs(run.sinks) do
    local count = snk.counts[snk.item] or 0
    table.insert(lines, string.format(
      "sink(%d,%d) %s %.2f/s", snk.x, snk.y, snk.item, count / seconds))
  end
  return lines
end

local function update_sink_labels(run, measure_elapsed_ticks)
  local seconds = math.max(measure_elapsed_ticks, 1) / 60.0
  for _, snk in ipairs(run.sinks) do
    if snk.label and snk.label.valid then
      local count = snk.counts[snk.item] or 0
      snk.label.text = string.format(
        "sink %s %.2f/s", snk.item, count / seconds)
    end
  end
end

-- Total items delivered across all sinks (each sink's configured item).
-- The convergence signal is the aggregate rate, which is stable iff every
-- sink is stable.
local function sink_total(run)
  local total = 0
  for _, snk in ipairs(run.sinks) do
    total = total + (snk.counts[snk.item] or 0)
  end
  return total
end

-- Is `cur` within `rel` (relative) of `prev`? Differences at or below
-- `floor` count as stable regardless, so a near-zero signal doesn't fail
-- on relative noise. Used to detect a plateaued throughput.
local function rel_stable(cur, prev, rel, floor)
  local d = math.abs(cur - prev)
  if d <= floor then return true end
  return d / math.max(math.abs(prev), floor) <= rel
end

-- Reset per-run accumulators at the warmup→measure boundary. Buffers
-- (assembler output slot, output belt, inserter hand) fill during warmup;
-- zeroing here means the measured rate counts only steady-state arrivals,
-- so a one-time buffer flush can't inflate it. This is the measurement
-- that separates a real engine discrepancy from a settling transient.
local function begin_measure(run)
  for _, snk in ipairs(run.sinks) do
    snk.counts = {}
  end
  for _, src in ipairs(run.sources) do
    src.inserted = 0
  end
  for _, w in ipairs(run.watched) do
    w.line_totals = {}
    w.status_counts = {}
    w.held_ticks = 0
    w.samples = 0
    if w.kind == "machine" and w.entity.valid then
      w.products_finished_start = w.entity.products_finished
    end
  end
  run.phase = "measure"
  run.meas_start_tick = game.tick
  run.last_check = 0
  run.prev_meas_rate = -1
  run.meas_stable = 0
end

-- ----------------------------------------------------------------------------
-- result assembly
-- ----------------------------------------------------------------------------

local function finish_run(run)
  local measure_ticks = run.measure_ticks_used or 1
  local measure_seconds = measure_ticks / 60.0
  local result = {
    run_id = run.run_id,
    status = "done",
    warmup_ticks = run.warmup_ticks_used or 0,
    measure_ticks = measure_ticks,
    converged = run.converged or false,
    surface = SURFACE_NAME,
    sinks = {},
    sources = {},
    entities = {},
  }

  for _, snk in ipairs(run.sinks) do
    local expected_count = snk.counts[snk.item] or 0
    table.insert(result.sinks, {
      x = snk.x,
      y = snk.y,
      item = snk.item,
      count = expected_count,
      rate = expected_count / measure_seconds,
      all_items = snk.counts,
    })
  end

  for _, src in ipairs(run.sources) do
    table.insert(result.sources, {
      x = src.x,
      y = src.y,
      item = src.item,
      inserted = src.inserted,
      rate = src.inserted / measure_seconds,
    })
  end

  for _, w in ipairs(run.watched) do
    local entry = {
      x = w.x,
      y = w.y,
      name = w.name,
      kind = w.kind,
      samples = w.samples,
    }
    if w.kind == "belt" then
      local avg = {}
      for li, total in pairs(w.line_totals) do
        avg[li] = total / math.max(w.samples, 1)
      end
      entry.avg_line_items = avg
    else
      entry.status_counts = w.status_counts
      if w.kind == "inserter" then
        entry.held_frac = w.held_ticks / math.max(w.samples, 1)
      end
      if w.kind == "machine" and w.entity.valid then
        entry.products_finished =
          w.entity.products_finished - w.products_finished_start
      end
    end
    table.insert(result.entities, entry)
  end

  local st = ensure_storage()
  st.result = json_encode(result)
  st.run = nil
  game.speed = 1.0
  update_sink_labels(run, measure_ticks)
  announce(string.format(
    "run %s DONE (%s after %d warmup + %d measure ticks) — %s",
    run.run_id,
    run.converged and "converged" or "hit tick cap",
    run.warmup_ticks_used or 0, measure_ticks,
    table.concat(sink_rate_lines(run, measure_ticks), ", ")), COLOR_OK)
end

-- ----------------------------------------------------------------------------
-- public: tick driver + remote methods
-- ----------------------------------------------------------------------------

-- Two adaptive phases, both convergence-gated with a hard tick cap:
--
--   warmup  — flow items until the *windowed* delivery rate plateaus
--             (belts/pipelines saturated), or warmup_max ticks. Counts
--             are then zeroed (begin_measure) so buffer-fill can't leak in.
--   measure — count deliveries until the *cumulative* rate plateaus, or
--             measure_max ticks. The plateaued cumulative rate is the
--             reported steady-state throughput.
--
-- Fast factories (belts) converge in a few checks and finish near-instantly;
-- slow ones (assemblers) rarely satisfy the relative test and run to the cap,
-- which is exactly the extra measuring time they need for an accurate rate.
function M.on_tick()
  local st = storage.parity
  if not st or not st.run then return end
  local run = st.run

  -- Items always flow; the phase only decides what we do with the counts.
  feed_sources(run, true)
  drain_sinks(run, true)

  if run.phase == "warmup" then
    local elapsed = game.tick - run.start_tick
    if elapsed - run.last_check >= run.check_every then
      local total = sink_total(run)
      local windowed = (total - run.warm_last_total)
        / (run.check_every / 60.0)
      run.warm_last_total = total
      run.last_check = elapsed
      if elapsed >= run.warmup_min
        and rel_stable(windowed, run.prev_warm_rate, run.converge_rel,
          run.converge_floor) then
        run.warm_stable = run.warm_stable + 1
      else
        run.warm_stable = 0
      end
      run.prev_warm_rate = windowed
      if run.warm_stable >= run.converge_hits or elapsed >= run.warmup_max then
        run.warmup_ticks_used = elapsed
        begin_measure(run)
        announce(string.format(
          "run %s: saturated after %d warmup ticks; measuring…",
          run.run_id, elapsed))
      end
    end
    return
  end

  -- measure phase
  local meas = game.tick - run.meas_start_tick
  if meas % run.sample_every == 0 then
    sample_entities(run)
    update_sink_labels(run, meas)
  end
  if meas - run.last_check >= run.check_every then
    run.last_check = meas
    local rate = sink_total(run) / math.max(meas / 60.0, 1e-9)
    if meas >= run.measure_min and sink_total(run) > 0
      and rel_stable(rate, run.prev_meas_rate, run.converge_rel,
        run.converge_floor) then
      run.meas_stable = run.meas_stable + 1
    else
      run.meas_stable = 0
    end
    run.prev_meas_rate = rate
    announce(string.format("run %s: measuring %d ticks — %s", run.run_id,
      meas, table.concat(sink_rate_lines(run, meas), ", ")))
    if run.meas_stable >= run.converge_hits or meas >= run.measure_max then
      run.measure_ticks_used = meas
      run.converged = run.meas_stable >= run.converge_hits
      finish_run(run)
    end
  end
end

function M.start(spec_json)
  local st = ensure_storage()
  local ok, spec = pcall(json_decode, spec_json)
  if not ok or type(spec) ~= "table" then
    return json_encode({ status = "error", error = "bad spec JSON" })
  end
  if #(spec.sources or {}) == 0 or #(spec.sinks or {}) == 0 then
    return json_encode({ status = "error",
      error = "spec needs at least one source and one sink" })
  end

  -- Preempt any active run; its partial state is discarded.
  if st.run then
    announce("preempting run " .. st.run.run_id, COLOR_ERR)
  end
  st.run = nil
  st.result = nil

  local force = ensure_force()
  local surface = ensure_surface()
  local grid_size = spec.grid_size or 16
  generate_area(surface, grid_size)
  clear_surface(surface)
  rendering.clear(script.mod_name)
  place_power(surface, force, grid_size)

  local watched, sources, sinks, errors = build_factory(surface, force, spec)
  if #errors > 0 then
    for _, err in ipairs(errors) do
      announce("BUILD FAILED: " .. err, COLOR_ERR)
    end
    return json_encode({ status = "error",
      error = table.concat(errors, "; ") })
  end

  -- Map overlays: grid outline, source/sink labels. Sink labels are
  -- live-updated with the measured rate as the run progresses.
  rendering.draw_rectangle({
    surface = surface,
    left_top = { ORIGIN_X, ORIGIN_Y },
    right_bottom = { ORIGIN_X + grid_size, ORIGIN_Y + grid_size },
    color = { r = 0.5, g = 0.5, b = 0.9, a = 0.6 },
    width = 2,
    filled = false,
  })
  for _, src in ipairs(sources) do
    draw_label(surface, ORIGIN_X + src.x + 0.5, ORIGIN_Y + src.y - 0.6,
      "src " .. src.item, COLOR_SRC, 0.8)
  end
  for _, snk in ipairs(sinks) do
    snk.label = draw_label(surface,
      ORIGIN_X + snk.x + 0.5, ORIGIN_Y + snk.y - 0.6,
      "sink " .. snk.item, COLOR_SNK, 0.8)
  end

  mute_alerts()

  st.run = {
    run_id = spec.run_id or "unnamed",
    start_tick = game.tick,
    -- Adaptive phases (legacy warmup_ticks/measure_ticks accepted as the
    -- caps for back-compat).
    phase = "warmup",
    warmup_min = spec.warmup_min or 300,
    warmup_max = spec.warmup_max or spec.warmup_ticks or 1800,
    measure_min = spec.measure_min or 600,
    measure_max = spec.measure_max or spec.measure_ticks or 36000,
    check_every = math.max(spec.check_every or 300, 1),
    converge_rel = spec.converge_rel or 0.02,
    converge_hits = math.max(spec.converge_hits or 3, 1),
    converge_floor = spec.converge_floor or 0.02,
    sample_every = math.max(spec.sample_every or 15, 1),
    -- warmup convergence trackers
    last_check = 0,
    warm_last_total = 0,
    prev_warm_rate = -1,
    warm_stable = 0,
    watched = watched,
    sources = sources,
    sinks = sinks,
  }
  game.speed = spec.game_speed or 32
  announce(string.format(
    "run %s: built %d entities, %d sources, %d sinks; auto-warmup"
    .. " (≤%d) then measure to convergence (≤%d) at speed %.0fx",
    st.run.run_id, #watched, #sources, #sinks,
    st.run.warmup_max, st.run.measure_max, game.speed))
  return json_encode({
    status = "running",
    run_id = st.run.run_id,
    total_ticks = st.run.warmup_max + st.run.measure_max,
  })
end

function M.poll()
  local st = ensure_storage()
  if st.run then
    local run = st.run
    local ticks_done, total
    if run.phase == "warmup" then
      ticks_done = game.tick - run.start_tick
      total = run.warmup_max
    else
      ticks_done = game.tick - run.meas_start_tick
      total = run.measure_max
    end
    return json_encode({
      status = "running",
      run_id = run.run_id,
      phase = run.phase,
      ticks_done = ticks_done,
      total_ticks = total,
    })
  end
  if st.result then
    return st.result
  end
  return json_encode({ status = "idle" })
end

function M.abort()
  local st = ensure_storage()
  local had = st.run ~= nil
  if had then
    announce("run " .. st.run.run_id .. " ABORTED", COLOR_ERR)
  end
  st.run = nil
  st.result = nil
  game.speed = 1.0
  return json_encode({ status = "aborted", was_running = had })
end

return M
