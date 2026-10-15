"""The mod's windows, built in Lua 5.2 against a strict stand-in for Factorio's
GUI (tests/mod_gui_mock.lua), so a window that would crash the game fails here.

The member lists are LuaGuiElement's and LuaStyle's from the 2.0.77 runtime API
(doc-html/runtime-api.json beside the game).
"""

import re
from pathlib import Path

import pytest
from lupa import lua52

ROOT = Path(__file__).resolve().parent.parent
MOD = ROOT / "factorion-mod" / "mod"
STYLE_LUA = Path.home() / (
    "Library/Application Support/Steam/steamapps/common/Factorio/factorio.app/Contents/data/core/prototypes/style.lua"
)

GUI_ELEMENT_MEMBERS = """
add add_item add_tab allow_decimal allow_negative allow_none_state anchor auto_center auto_toggle
badge_text bring_to_front caption children children_names clear clear_items clicked_sprite
close_dropdown column_count destroy direction drag_target draw_horizontal_line_after_headers
draw_horizontal_lines draw_vertical_lines elem_filters elem_tooltip elem_type elem_value enabled
entity focus force force_auto_center game_controller_interaction get_index_in_parent get_item
get_mod get_slider_discrete_values get_slider_maximum get_slider_minimum get_slider_value_step gui
horizontal_scroll_policy hovered_sprite icon_selector ignored_by_interaction index is_password
items left_label_caption left_label_tooltip location locked lose_focus_on_confirm
minimap_player_index mouse_button_filter name number numeric object_name parent player_index
position quality raise_hover_events read_only remove_item remove_tab resize_to_sprite
right_label_caption right_label_tooltip scroll_to_bottom scroll_to_element scroll_to_item
scroll_to_left scroll_to_right scroll_to_top select select_all selectable selected_index
selected_tab_index set_item set_slider_discrete_values set_slider_minimum_maximum
set_slider_value_step show_percent_for_small_numbers slider_value sprite state style surface_index
swap_children switch_state tabs tags text toggled tooltip type valid value vertical_centering
vertical_scroll_policy visible word_wrap zoom
""".split()

STYLE_FIELDS = """
badge_font badge_horizontal_spacing bar_width bottom_cell_padding bottom_margin bottom_padding
cell_padding clicked_font_color clicked_vertical_offset color column_alignments
default_badge_font_color disabled_badge_font_color disabled_font_color draw_grayscale_picture
extra_bottom_margin_when_activated extra_bottom_padding_when_activated
extra_left_margin_when_activated extra_left_padding_when_activated extra_margin_when_activated
extra_padding_when_activated extra_right_margin_when_activated extra_right_padding_when_activated
extra_top_margin_when_activated extra_top_padding_when_activated font font_color gui height
horizontal_align horizontal_spacing horizontally_squashable horizontally_stretchable
hovered_font_color left_cell_padding left_margin left_padding margin maximal_height maximal_width
minimal_height minimal_width name natural_height natural_width object_name padding
pie_progress_color rich_text_setting right_cell_padding right_margin right_padding
selected_badge_font_color selected_clicked_font_color selected_font_color
selected_hovered_font_color single_line size stretch_image_to_widget_size strikethrough_color
top_cell_padding top_margin top_padding use_header_filler valid vertical_align vertical_spacing
vertically_squashable vertically_stretchable width
""".split()


def add_calls(source):
    """The table literal of every `.add({ ... })` call in a Lua source."""
    for match in re.finditer(r"\badd\(\s*\{", source):
        depth, i = 0, match.end() - 1
        for i in range(match.end() - 1, len(source)):
            depth += {"{": 1, "}": -1}.get(source[i], 0)
            if depth == 0:
                break
        yield source[match.end() : i]


def top_level_fields(table, key):
    """Values of `key = ...` at the top level of a table literal's body."""
    depth, values = 0, []
    for match in re.finditer(r"[{}]|\b" + key + r"\s*=\s*([^,\n}]+)", table):
        if match.group(0) in "{}":
            depth += 1 if match.group(0) == "{" else -1
        elif depth == 0:
            values.append(match.group(1).strip())
    return values


def literal(value, constants):
    if value.startswith('"'):
        return value.strip('"')
    return constants.get(value)


def mod_sources():
    return {path: path.read_text() for path in sorted(MOD.glob("*.lua"))}


def test_no_element_is_named_after_a_gui_element_member():
    members = set(GUI_ELEMENT_MEMBERS)
    clashes = []
    for path, source in mod_sources().items():
        constants = dict(re.findall(r'^local (\w+) = "([^"]*)"', source, re.M))
        for table in add_calls(source):
            for value in top_level_fields(table, "name"):
                name = literal(value, constants)
                if name in members:
                    clashes.append(f"{path.name}: {name}")
    assert not clashes, f"GUI elements named after LuaGuiElement members: {clashes}"


@pytest.mark.skipif(not STYLE_LUA.exists(), reason="needs Factorio's style.lua")
def test_every_style_exists():
    styles = set(re.findall(r"^    (\w+) =", STYLE_LUA.read_text(), re.M))
    missing = []
    for path, source in mod_sources().items():
        for table in add_calls(source):
            for value in top_level_fields(table, "style"):
                style = literal(value, {})
                if style and style not in styles:
                    missing.append(f"{path.name}: {style}")
    assert not missing, f"Unknown GUI styles: {missing}"


@pytest.fixture
def lua():
    runtime = lua52.LuaRuntime(unpack_returned_tuples=True)
    runtime.execute(f"package.path = '{MOD}/?.lua;{ROOT / 'tests'}/?.lua;' .. package.path")
    runtime.execute('mock = require("mod_gui_mock")')
    runtime.globals().mock.setup(
        runtime.table_from(GUI_ELEMENT_MEMBERS), runtime.table_from(STYLE_FIELDS)
    )
    runtime.execute(
        """
        storage = {}
        defines = { wire_connector_id = { circuit_red = 1, circuit_green = 2 } }
        player = mock.player(1)
        game = {
          tick = 0,
          connected_players = { player },
          get_player = function(index) if index == 1 then return player end end,
        }
        """
    )
    return runtime


def find(lua, root, name):
    for element in lua.globals().mock.descendants(root).values():
        if element.name == name:
            return element
    raise AssertionError(f"no element named {name}")


def event(lua, element):
    return lua.table_from({"element": element, "player_index": 1})


def test_gpu_window_opens_refreshes_and_edits_conditions(lua):
    lua.execute(
        """
        inference = require("inference")
        server = { valid = true, name = "factorion-gpu-server", unit_number = 7 }
        settings_of = { conditions = {} }
        inference.settings = function() return settings_of end
        inference.describe = function() return "Painted area", true, "Cluster of 2" end
        inference.info = function()
          return { status = "working", caption = "Predicting", utilisation = 0.5, power_kw = 2.7, networks = { { wire = "red", id = 41, signals = { { signal = { name = "iron-plate" }, count = 1 } } } }, powered = true }
        end
        inference.open(player, server)
        inference.refresh(player, true)
        inference.refresh(player, false)
        """
    )
    g = lua.globals()
    frame = g.player.gui.screen["factorion-gpu-gui"]
    main = frame.panels.window.body.main
    assert main.status.status_caption.caption == "Predicting"
    assert main.cluster.caption == "Cluster of 2"

    button = frame.panels.window.titlebar.circuit_button
    assert not frame.panels.circuit.visible
    assert g.inference.on_gui_click(event(lua, button))
    assert frame.panels.circuit.visible and button.toggled

    section = frame.panels.circuit.inner.predict
    section.active.state = True
    section.condition.signal.elem_value = lua.eval('{ type = "virtual", name = "signal-P" }')
    section.condition.constant.text = "3"
    assert g.inference.on_gui_text_changed(event(lua, section.condition.constant))
    assert g.inference.on_gui_checked_state_changed(event(lua, section.active))
    condition = g.settings_of.conditions.predict
    assert condition.enabled and condition.signal.name == "signal-P" and condition.constant == 3

    assert section.condition.signal.enabled
    section.active.state = False
    assert g.inference.on_gui_checked_state_changed(event(lua, section.active))
    assert not section.condition.signal.enabled and not section.condition.constant.enabled

    header = [child.caption for child in frame.panels.circuit.inner.header.children.values() if child.type == "label"]
    assert header == ["Connected to:", "41"]
    tooltip = frame.panels.circuit.inner.header.info_red.tooltip
    assert "Circuit network: 41" in tooltip and "[item=iron-plate] 1" in tooltip

    read_state = next(
        element for element in g.mock.descendants(frame).values() if element.tags.factorion_gpu == "read_state"
    )
    read_state.state = False
    assert g.inference.on_gui_checked_state_changed(event(lua, read_state))
    assert g.settings_of.read_state is False


@pytest.mark.parametrize("role", ["source", "sink"])
def test_marker_window_opens_and_shows_status(lua, role):
    lua.execute(
        f"""
        gui = require("gui")
        belt = {{ valid = true, name = "factorion-{role}-belt" }}
        frame = gui.marker_window(player, "factorion-marker-dialog", "factorion-marker-item", belt, "{role}", "iron-plate")
        gui.marker_status(frame, "{role}", "iron-plate", false, 0)
        gui.marker_status(frame, "{role}", "iron-plate", true, 3.5)
        """
    )
    frame = lua.globals().frame
    assert find(lua, frame, "factorion-marker-item").elem_value == "iron-plate"
    assert frame.panels.window.body.main.status.status_caption.caption


def test_condition_matches_signals_like_vanilla(lua):
    lua.execute(
        """
        gui = require("gui")
        function wired(signals)
          return { get_signals = function() return signals end }
        end
        P = { type = "virtual", name = "signal-P" }
        """
    )
    g = lua.globals()
    cases = [
        ('{ { signal = P, count = 2 } }', '{ enabled = true, signal = P, comparator = ">", constant = 1 }', True),
        ("{}", '{ enabled = true, signal = P, comparator = "=", constant = 0 }', True),
        ('{ { signal = P, count = 2 } }', '{ enabled = false, signal = P, comparator = ">", constant = 1 }', False),
        (
            '{ { signal = { name = "iron-plate" }, count = 5 } }',
            '{ enabled = true, signal = { type = "virtual", name = "signal-anything" }, comparator = ">", constant = 1 }',
            True,
        ),
    ]
    for signals, condition, expected in cases:
        assert g.gui.fulfilled(lua.eval(f"wired({signals})"), lua.eval(condition)) is expected, (signals, condition)
    item_only = lua.eval('wired({ { signal = { name = "iron-plate" }, count = 5 } })')
    anything = lua.eval('{ enabled = true, signal = { type = "virtual", name = "signal-anything" }, comparator = ">", constant = 1 }')
    assert g.gui.fulfilled(item_only, anything, True) is False


def test_wire_signals_are_decoded_in_the_window(lua):
    lua.execute(
        """
        inference = require("inference")
        server = { valid = true, name = "factorion-gpu-server", unit_number = 7 }
        settings_of = { conditions = {}, circuit_endpoints = true }
        inference.settings = function() return settings_of end
        local used = { role = "source", item = "iron-plate", value = 20300, x = 9, y = 22, direction = 2 }
        local region = {
          origin = { x = 10, y = 20 }, endpoints = { used },
          signals = {
            used,
            { role = "sink", item = "copper-plate", value = 1001001, problem = "D must be 1, 2, 3 or 4" },
          },
          warnings = { "The [item=copper-plate] sink signal (1001001) is unused: D must be 1, 2, 3 or 4." },
        }
        inference.describe = function() return "Painted area", true, "Cluster of 1", region end
        inference.open(player, server)
        """
    )
    frame = lua.globals().player.gui.screen["factorion-gpu-gui"]
    decoded = [c.caption for c in frame.panels.circuit.inner.endpoints.signals.children.values()]
    assert decoded[0] == "[item=iron-plate] 20300: Source at X 0, Y 3, facing east"
    assert decoded[1].startswith("[img=utility/warning_icon] [item=copper-plate] 1001001: Sink with X 1, Y 10, D 100")
    assert "D must be 1, 2, 3 or 4" in decoded[1]
    used = [c.caption for c in frame.panels.window.body.main.endpoints.children.values()]
    assert "[item=iron-plate] Source at X 0, Y 3, facing east" in used
    warnings = [c.caption for c in frame.panels.window.body.main.warnings.children.values()]
    assert warnings and "[img=utility/warning_icon]" in warnings[0] and "1001001" in warnings[0]


def test_out_of_range_setting_turns_red_and_keeps_the_old_value(lua):
    lua.execute(
        """
        inference = require("inference")
        server = { valid = true, name = "factorion-gpu-server", unit_number = 7 }
        inference.open(player, server)
        """
    )
    g = lua.globals()
    frame = g.player.gui.screen["factorion-gpu-gui"]
    best_of = next(e for e in g.mock.descendants(frame).values() if e.tags.factorion_gpu == "best_of")
    best_of.text = "100"
    assert g.inference.on_gui_text_changed(event(lua, best_of))
    assert best_of.style == "invalid_value_textfield" and "1 to 64; using 1" in best_of.tooltip
    assert g.inference.config().best_of == 1
    best_of.text = "4"
    g.inference.on_gui_text_changed(event(lua, best_of))
    assert best_of.style == "textbox" and g.inference.config().best_of == 4
