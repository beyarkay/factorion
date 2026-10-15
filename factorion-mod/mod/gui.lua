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

-- Its close button carries tags.factorion_close = name. Further panels go
-- beside the first in frame.body.
function M.entity_window(player, name, title, entity)
  local frame = player.gui.screen.add({ type = "frame", name = name, direction = "vertical" })
  frame.auto_center = true
  local titlebar = frame.add({ type = "flow", direction = "horizontal" })
  titlebar.drag_target = frame
  titlebar.add({
    type = "label", caption = title, style = "frame_title", ignored_by_interaction = true,
  })
  local drag = titlebar.add({
    type = "empty-widget", style = "draggable_space_header", ignored_by_interaction = true,
  })
  drag.style.height = 24
  drag.style.horizontally_stretchable = true
  titlebar.add({
    type = "sprite-button", style = "frame_action_button", sprite = "utility/close",
    tooltip = { "gui.close-instruction" }, tags = { factorion_close = name },
  })
  local body = frame.add({ type = "flow", name = "body", direction = "horizontal" })
  body.style.horizontal_spacing = 12
  local panel = M.panel(body, "main")
  local status = panel.add({ type = "flow", name = "status", direction = "horizontal" })
  status.style.vertical_align = "center"
  status.add({ type = "sprite", name = "light", style = "status_image" })
  status.add({ type = "label", name = "text" })
  local preview_frame = panel.add({ type = "frame", style = "deep_frame_in_shallow_frame" })
  local preview = preview_frame.add({ type = "entity-preview", style = "wide_entity_button" })
  preview.entity = entity
  return frame, panel
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
  panel.status.light.sprite = STATUS_SPRITES[kind]
  panel.status.text.caption = caption
end

-- A checkbox, then signal, comparator and constant; each carries
-- tags.factorion_condition = key so an edit finds its condition.
function M.condition_row(parent, key, caption, condition)
  local tags = { factorion_condition = key }
  local row = parent.add({ type = "flow", name = key, direction = "horizontal" })
  row.style.vertical_align = "center"
  row.add({
    type = "checkbox", name = "enabled", caption = caption,
    state = condition.enabled or false, tags = tags,
  })
  row.add({ type = "empty-widget" }).style.horizontally_stretchable = true
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
end

function M.read_condition_row(row)
  return {
    enabled = row.enabled.state,
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
-- vanilla's anything and everything.
function M.fulfilled(entity, condition)
  local wanted = condition.signal
  if not (condition.enabled and wanted) then return false end
  local signals = entity.get_signals(defines.wire_connector_id.circuit_red,
    defines.wire_connector_id.circuit_green) or {}
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
