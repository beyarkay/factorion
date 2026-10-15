-- Windows laid out like vanilla entity windows: a title bar, then a light
-- panel with the entity's status light above a preview of it, and circuit
-- conditions edited the way vanilla's are.

local M = {}

local STATUS_SPRITES = {
  working = "utility/status_working",
  waiting = "utility/status_yellow",
  broken = "utility/status_not_working",
  idle = "utility/status_blue",
}

M.COMPARATORS = { ">", "<", "=", "≥", "≤", "≠" }

-- The root is an invisible frame holding the window and any side panels in
-- root.panels, so they move together. Its close button carries tags.factorion_close = name;
-- with circuit_tags, a circuit network button carrying them sits beside it.
function M.entity_window(player, name, title, entity, circuit_tags)
  local root = player.gui.screen.add({
    type = "frame", name = name, style = "invisible_frame", direction = "horizontal",
  })
  root.auto_center = true
  local panels = root.add({ type = "flow", name = "panels", direction = "horizontal" })
  panels.style.horizontal_spacing = 0
  local frame = panels.add({ type = "frame", name = "window", direction = "vertical" })
  local titlebar = frame.add({ type = "flow", name = "titlebar", direction = "horizontal" })
  titlebar.drag_target = root
  titlebar.add({
    type = "label", caption = title, style = "frame_title", ignored_by_interaction = true,
  })
  local drag = titlebar.add({
    type = "empty-widget", style = "draggable_space_header", ignored_by_interaction = true,
  })
  drag.style.height = 24
  drag.style.horizontally_stretchable = true
  if circuit_tags then
    titlebar.add({
      type = "sprite-button", name = "circuit_button", style = "frame_action_button",
      sprite = "utility/circuit_network_panel", tooltip = "Circuit network", tags = circuit_tags,
    })
  end
  titlebar.add({
    type = "sprite-button", style = "frame_action_button", sprite = "utility/close",
    tooltip = { "gui.close-instruction" }, tags = { factorion_close = name },
  })
  local body = frame.add({ type = "flow", name = "body", direction = "horizontal" })
  local panel = M.panel(body, "main")
  local status = panel.add({ type = "flow", name = "status", direction = "horizontal" })
  status.style.vertical_align = "center"
  status.add({ type = "sprite", name = "status_light", style = "status_image" })
  status.add({ type = "label", name = "status_caption" })
  local preview_frame = panel.add({ type = "frame", style = "deep_frame_in_shallow_frame" })
  local preview = preview_frame.add({ type = "entity-preview", style = "wide_entity_button" })
  preview.entity = entity
  return root, panel
end

-- Vanilla's circuit connection panel: a titled frame beside the window, a
-- subheader naming the networks, then sections divided by lines.
function M.circuit_panel(root)
  local frame = root.panels.add({
    type = "frame", name = "circuit", caption = "Circuit connection", direction = "vertical",
  })
  frame.drag_target = root
  local inner = frame.add({
    type = "frame", name = "inner", style = "inside_shallow_frame", direction = "vertical",
  })
  local header = inner.add({ type = "frame", name = "header", style = "subheader_frame" })
  header.style.horizontally_stretchable = true
  header.style.left_padding = 12
  return inner
end

local NETWORK_COLORS = { red = { 1, 0.35, 0.35 }, green = { 0.35, 1, 0.35 } }

local SIGNAL_TAGS = { item = "item", fluid = "fluid", virtual = "virtual-signal", entity = "entity",
  recipe = "recipe", quality = "quality", ["space-location"] = "space-location", ["asteroid-chunk"] = "asteroid-chunk" }

-- Like vanilla's: the network's id, then each of its signals and count.
local function network_tooltip(network)
  local lines = { "[font=default-bold]Circuit network: " .. network.id .. "[/font]" }
  local row = {}
  for _, entry in ipairs(network.signals or {}) do
    local signal = entry.signal
    table.insert(row, string.format("[%s=%s] %d", SIGNAL_TAGS[signal.type or "item"], signal.name, entry.count))
    if #row == 6 then
      table.insert(lines, table.concat(row, "   "))
      row = {}
    end
  end
  if #row > 0 then table.insert(lines, table.concat(row, "   ")) end
  if #lines > 1 then table.insert(lines, 2, "Signals") end
  return table.concat(lines, "\n")
end

-- networks is a list of { wire = "red" or "green", id, signals }; the header
-- is rebuilt only when the networks change, so a hovered tooltip stays up.
function M.set_networks(inner, networks)
  local header = inner.header
  local key = ""
  for _, network in ipairs(networks) do key = key .. network.wire .. network.id .. " " end
  if header.tags.networks ~= key or #header.children == 0 then
    header.tags = { networks = key }
    header.clear()
    header.add({ type = "label", caption = #networks == 0 and "Not connected" or "Connected to:" })
    for _, network in ipairs(networks) do
      local id = header.add({ type = "label", name = "id_" .. network.wire, caption = tostring(network.id) })
      id.style.font_color = NETWORK_COLORS[network.wire]
      id.style.left_padding = 4
      local info = header.add({ type = "sprite", name = "info_" .. network.wire, sprite = "info_no_border" })
      info.style.stretch_image_to_widget_size = true
      info.style.size = 14
    end
  end
  for _, network in ipairs(networks) do
    local tooltip = network_tooltip(network)
    header["id_" .. network.wire].tooltip = tooltip
    header["info_" .. network.wire].tooltip = tooltip
  end
end

-- A section of a circuit panel headed by a checkbox.
function M.circuit_section(inner, name, caption, state, tags, tooltip)
  if #inner.children > 1 then inner.add({ type = "line" }) end
  local section = inner.add({ type = "flow", name = name, direction = "vertical" })
  section.style.padding = { 8, 12 }
  section.style.vertical_spacing = 8
  section.add({
    type = "checkbox", name = "active", caption = caption, state = state,
    tags = tags, tooltip = tooltip,
  }).style.font = "default-bold"
  return section
end

function M.panel(body, name)
  return body.add({
    type = "frame", name = name, style = "inside_shallow_frame_with_padding",
    direction = "vertical",
  })
end

function M.heading(parent, caption)
  local label = parent.add({ type = "label", caption = caption, style = "caption_label" })
  label.style.top_margin = 8
  return label
end

-- kind is working, waiting, broken or idle.
function M.set_status(panel, kind, caption)
  panel.status.status_light.sprite = STATUS_SPRITES[kind]
  panel.status.status_caption.caption = caption
end

-- A source or sink belt's window: the entity, then its item, picked by the
-- choose-elem-button named picker.
function M.marker_window(player, name, picker, entity, role, item)
  local root, main = M.entity_window(player, name, { "entity-name." .. entity.name }, entity)
  M.heading(main, role == "source" and "Expected item" or "Counted item")
  local row = main.add({ type = "flow", direction = "horizontal" })
  row.style.vertical_align = "center"
  row.add({
    type = "choose-elem-button", name = picker, elem_type = "item", item = item,
    style = "slot_button_in_shallow_frame",
  })
  local hint = row.add({
    type = "label",
    caption = role == "source"
      and "Belt this item into the source; the model builds from it."
      or "The model builds to make this item; its rate is shown in alt-mode.",
  })
  hint.style.single_line = false
  hint.style.maximal_width = 260
  return root
end

function M.marker_status(root, role, item, receiving, rate)
  local icon = "[item=" .. item .. "]"
  local main = root.panels.window.body.main
  if role == "source" then
    if receiving then
      M.set_status(main, "working", "Receiving " .. icon)
    else
      M.set_status(main, "waiting", "No " .. icon .. " arriving; belt some in")
    end
  elseif rate >= 0.05 then
    M.set_status(main, "working", string.format("%.1f %s per second", rate, icon))
  else
    M.set_status(main, "waiting", "No " .. icon .. " passing through")
  end
end

-- A circuit section enabling a condition: signal, comparator and constant,
-- greyed out while it's off; each carries tags.factorion_condition = key so
-- an edit finds its condition.
function M.condition_section(inner, key, caption, condition, tooltip)
  local tags = { factorion_condition = key }
  local section = M.circuit_section(inner, key, caption, condition.enabled or false, tags, tooltip)
  local row = section.add({ type = "flow", name = "condition", direction = "horizontal" })
  row.style.vertical_align = "center"
  row.add({
    type = "choose-elem-button", name = "signal", elem_type = "signal",
    signal = condition.signal, style = "slot_button_in_shallow_frame", tags = tags,
  })
  local selected = 1
  for i, comparator in ipairs(M.COMPARATORS) do
    if comparator == condition.comparator then selected = i end
  end
  row.add({
    type = "drop-down", name = "comparator", items = M.COMPARATORS,
    selected_index = selected, style = "circuit_condition_comparator_dropdown", tags = tags,
  })
  row.add({
    type = "textfield", name = "constant", text = tostring(condition.constant or 0),
    numeric = true, allow_negative = true, style = "short_number_textfield", tags = tags,
  })
  M.grey_condition(section)
end

function M.grey_condition(section)
  for _, child in pairs(section.condition.children) do child.enabled = section.active.state end
end

function M.read_condition_section(section)
  local row = section.condition
  return {
    enabled = section.active.state,
    signal = row.signal.elem_value,
    comparator = M.COMPARATORS[row.comparator.selected_index],
    constant = tonumber(row.constant.text) or 0,
  }
end

local function compare(a, comparator, b)
  if comparator == ">" then return a > b end
  if comparator == "<" then return a < b end
  if comparator == "=" then return a == b end
  if comparator == "≥" then return a >= b end
  if comparator == "≤" then return a <= b end
  return a ~= b
end

-- Whether entity's red and green signals together meet condition, with
-- vanilla's anything and everything; item signals are left out when they
-- place sources and sinks instead.
function M.fulfilled(entity, condition, skip_items)
  local wanted = condition.signal
  if not (condition.enabled and wanted) then return false end
  local signals = {}
  for _, signal in ipairs(entity.get_signals(defines.wire_connector_id.circuit_red,
    defines.wire_connector_id.circuit_green) or {}) do
    if not (skip_items and (signal.signal.type or "item") == "item") then
      table.insert(signals, signal)
    end
  end
  local function holds(count) return compare(count, condition.comparator, condition.constant) end
  if wanted.type == "virtual" and (wanted.name == "signal-anything" or wanted.name == "signal-everything") then
    local anything = wanted.name == "signal-anything"
    for _, signal in ipairs(signals) do
      if holds(signal.count) == anything then return anything end
    end
    return not anything
  end
  for _, signal in ipairs(signals) do
    if signal.signal.name == wanted.name and (signal.signal.type or "item") == (wanted.type or "item") then
      return holds(signal.count)
    end
  end
  return holds(0)
end

return M
