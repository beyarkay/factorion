-- A strict stand-in for Factorio's GUI, for tests/test_mod_gui.py. It fails
-- where the game does: an element named after one of LuaGuiElement's own
-- members, two siblings with one name, or setting a property or style field
-- that doesn't exist or that its element type doesn't have. A missing child reads as nil, as in the game.

local M = {}
local members, style_fields = {}, {}

-- Parameters of add() that set a choose-elem-button's value.
local ELEM_VALUES = { item = true, signal = true, fluid = true, recipe = true, entity = true }

function M.setup(member_list, style_list)
  for _, name in ipairs(member_list) do members[name] = true end
  for _, name in ipairs(style_list) do style_fields[name] = true end
end

-- Style fields only some element types have.
local STYLE_TYPES = {
  horizontal_spacing = { flow = true, table = true },
  vertical_spacing = { flow = true, table = true },
}

local function new_style(element_type)
  return setmetatable({}, {
    __newindex = function(t, key, value)
      if not style_fields[key] then error("LuaStyle doesn't contain key " .. tostring(key), 2) end
      if STYLE_TYPES[key] and not STYLE_TYPES[key][element_type] then
        error("LuaStyle of a " .. tostring(element_type) .. " doesn't contain key " .. key, 2)
      end
      rawset(t, key, value)
    end,
  })
end

local function new_element(spec, parent)
  local element = {
    valid = true, type = spec.type, name = spec.name or "", parent = parent,
    tags = spec.tags or {}, style = new_style(spec.type), children = {},
    caption = "", text = "", tooltip = "", visible = true, enabled = true,
  }
  for key, value in pairs(spec) do
    if ELEM_VALUES[key] then
      element.elem_value = value
    elseif key ~= "type" and key ~= "name" and key ~= "style" and key ~= "tags" then
      if not members[key] then error("Unknown add() parameter " .. key, 3) end
      element[key] = value
    end
  end

  function element.add(child_spec)
    if not child_spec.type then error("add() needs a type", 2) end
    local name = child_spec.name
    if name and members[name] then
      error('Invalid name "' .. name .. '": LuaGuiElement contains a property or method with the same name.', 2)
    end
    for _, sibling in ipairs(element.children) do
      if name and sibling.name == name then
        error('Gui element with name "' .. name .. '" already present in the parent element.', 2)
      end
    end
    local child = new_element(child_spec, element)
    table.insert(element.children, child)
    return child
  end

  function element.destroy()
    element.valid = false
    if not parent then return end
    for i, sibling in ipairs(parent.children) do
      if sibling == element then table.remove(parent.children, i) end
    end
  end

  return setmetatable(element, {
    __index = function(_, key)
      for _, child in ipairs(rawget(element, "children")) do
        if child.name == key then return child end
      end
    end,
    __newindex = function(t, key, value)
      if not members[key] then error("LuaGuiElement doesn't contain key " .. tostring(key), 2) end
      rawset(t, key, value)
    end,
  })
end

function M.player(index)
  local player = { index = index, gui = { screen = new_element({ type = "empty-widget" }) } }
  function player.print() end
  return player
end

-- Every element under root, depth first.
function M.descendants(root)
  local all = {}
  local function walk(element)
    for _, child in ipairs(element.children) do
      table.insert(all, child)
      walk(child)
    end
  end
  walk(root)
  return all
end

return M
