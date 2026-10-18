"""Activity lights: exercise power loss, load changes and saved render IDs."""

from pathlib import Path

from lupa import lua52

MOD = Path(__file__).resolve().parent.parent / "factorion-mod/mod"


def runtime():
    lua = lua52.LuaRuntime(unpack_returned_tuples=True)
    lua.globals().mod_path = str(MOD / "?.lua")
    lua.execute("""
      package.path = mod_path .. ';' .. package.path
      defines = { direction = { north=0, east=4, south=8, west=12 } }
      objects, next_id = {}, 0
      local function create(spec)
        next_id = next_id + 1
        spec.id, spec.valid = next_id, true
        spec.destroy = function() spec.valid=false; objects[spec.id]=nil end
        objects[spec.id] = spec
        return spec
      end
      rendering = { draw_sprite=create, draw_light=create,
        get_object_by_id=function(id) return objects[id] end }
      visuals = require('gpu_visuals')
      state = { entity = { valid=true, direction=0, surface={} } }
    """)
    return lua


def test_load_changes_reuse_objects_and_power_loss_removes_all_lights():
    lua = runtime()
    lua.execute("""
      visuals.update(state, true, 0, false)
      assert(objects[state.visuals.glow].sprite == 'factorion-gpu-horizontal-glow-0')
      local original = state.visuals.glow
      visuals.update(state, true, 0, true)
      assert(objects[original].sprite == 'factorion-gpu-horizontal-glow-1')
      visuals.update(state, true, 1, true)
      assert(objects[original].sprite == 'factorion-gpu-horizontal-glow-5')
      assert(next_id == 3, 'load changes must not allocate more render objects')
      visuals.update(state, false, 1, true)
      assert(next(objects) == nil and state.visuals == nil)
      visuals.update(state, true, 0, false)
      assert(objects[state.visuals.glow].sprite == 'factorion-gpu-horizontal-glow-0')
    """)


def test_rotation_and_missing_saved_render_objects_recover_with_matching_anchors():
    lua = runtime()
    lua.execute("""
      visuals.update(state, true, 0.6, false)
      state.entity.direction = defines.direction.east
      visuals.update(state, true, 0.6, false)
      local spec = require('graphics/entity/gpu-server-vertical')
      assert(objects[state.visuals.glow].sprite == 'factorion-gpu-vertical-glow-3')
      assert(objects[state.visuals.activity].target.offset == spec.activity_light)
      assert(visuals.port(state.entity) == spec.data_port)
      objects = {} -- saved IDs whose render objects were removed externally
      visuals.update(state, true, 0.6, false)
      assert(objects[state.visuals.glow] and objects[state.visuals.activity] and objects[state.visuals.pilot])
      state.entity.valid = false
      visuals.update(state, true, 0, false)
      assert(next(objects) == nil)
    """)
