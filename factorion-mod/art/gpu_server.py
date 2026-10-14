"""Model and render the GPU server sprites in Blender.

    Blender --background --factory-startup --python gpu_server.py -- OUT_DIR

Writes main.png and shadow.png at twice the game's resolution into OUT_DIR,
with the footprint's ground centre at the middle of the canvas. pack.py
turns them into sprites.
"""

import math
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Euler, Matrix, Vector

OUT = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
ELEVATION = math.radians(38)
PX_PER_TILE = 128
CANVAS_TILES = (6, 4)
SUN = Vector((1.0, -0.25, -1.2)).normalized()
DIRT = (0.13, 0.09, 0.055)
RUST = (0.30, 0.15, 0.07)

bpy.ops.wm.read_factory_settings(use_empty=True)


def material(name, colour, metallic=0.6, roughness=0.55, grime=1.3, wear=0.25, rust=0.5, emission=None):
    mat = bpy.data.materials.new(name)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    if emission:
        bsdf.inputs["Base Color"].default_value = (*colour, 1)
        bsdf.inputs["Emission Color"].default_value = (*emission[0], 1)
        bsdf.inputs["Emission Strength"].default_value = emission[1]
        return mat

    def node(kind, **inputs):
        n = nodes.new(kind)
        for key, value in inputs.items():
            n.inputs[key].default_value = value
        return n

    def maths(op, a, b):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, float):
                n.inputs[i].default_value = v
            else:
                links.new(v, n.inputs[i])
        return n.outputs["Value"]

    # Dirt collects in crevices (low ambient occlusion), broken up by noise.
    noise = node("ShaderNodeTexNoise", Scale=7.0, Detail=10.0, Roughness=0.65)
    ao = node("ShaderNodeAmbientOcclusion", Distance=0.2)
    crevice = maths("SUBTRACT", 1.0, ao.outputs["AO"])
    blotch = maths("MULTIPLY", maths("SUBTRACT", noise.outputs["Fac"], 0.45), 2.0)
    dirt_fac = maths("MINIMUM", maths("MULTIPLY", maths("ADD", maths("MULTIPLY", crevice, 2.5), blotch), grime), 0.85)
    # Large rust patches under the dirt.
    patch = node("ShaderNodeTexNoise", Scale=2.5, Detail=6.0, Roughness=0.7)
    rust_fac = maths("MULTIPLY", maths("MAXIMUM", maths("MULTIPLY", maths("SUBTRACT", patch.outputs["Fac"], 0.55), 4.0), 0.0), rust)
    rusty = node("ShaderNodeMix")
    rusty.data_type = "RGBA"
    rusty.inputs["A"].default_value = (*colour, 1)
    rusty.inputs["B"].default_value = (*RUST, 1)
    links.new(rust_fac, rusty.inputs["Factor"])
    dirty = node("ShaderNodeMix")
    dirty.data_type = "RGBA"
    links.new(rusty.outputs["Result"], dirty.inputs["A"])
    dirty.inputs["B"].default_value = (*DIRT, 1)
    links.new(dirt_fac, dirty.inputs["Factor"])

    # Worn, lighter edges: where a bevelled normal departs from the true one.
    bevel = node("ShaderNodeBevel", Radius=0.025)
    geometry = nodes.new("ShaderNodeNewGeometry")
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    links.new(bevel.outputs["Normal"], dot.inputs[0])
    links.new(geometry.outputs["Normal"], dot.inputs[1])
    edge = maths("MINIMUM", maths("MULTIPLY", maths("SUBTRACT", 1.0, dot.outputs["Value"]), 12.0 * wear), 1.0)
    worn = node("ShaderNodeMix")
    worn.data_type = "RGBA"
    worn.inputs["B"].default_value = (*[min(1.0, c * 1.8 + 0.12) for c in colour], 1)
    links.new(dirty.outputs["Result"], worn.inputs["A"])
    links.new(edge, worn.inputs["Factor"])
    links.new(worn.outputs["Result"], bsdf.inputs["Base Color"])
    return mat


PAINT = material("paint", (0.19, 0.245, 0.29), metallic=0.2, roughness=0.6, wear=0.6, rust=0.6)
STEEL = material("steel", (0.36, 0.34, 0.31), metallic=0.7, roughness=0.62)
DARK = material("dark", (0.10, 0.095, 0.09), metallic=0.6, roughness=0.5, grime=0.6)
BRIGHT = material("bright", (0.55, 0.53, 0.49), metallic=0.9, roughness=0.35, grime=0.6, rust=0.2)
BLADE = material("blade", (0.22, 0.21, 0.2), metallic=0.8, roughness=0.45, grime=0.8, rust=0.0)
GLASS = material("glass", (0.02, 0.03, 0.035), metallic=0.0, roughness=0.08, grime=0.4, wear=0.0, rust=0.0)
RUBBER = material("rubber", (0.035, 0.033, 0.03), metallic=0.0, roughness=0.7, grime=0.3, wear=0.1)
COPPER = material("copper", (0.62, 0.33, 0.16), metallic=0.9, roughness=0.4, grime=0.6)
HAZARD = material("hazard", (0.80, 0.58, 0.10), metallic=0.2, roughness=0.6, grime=0.7)
LED = material("led", (0.1, 0.3, 0.6), emission=((0.3, 0.6, 1.0), 3))
CERAMIC = material("ceramic", (0.22, 0.38, 0.62), metallic=0.0, roughness=0.3, grime=0.3)


def link(obj, mat, bevel=0.0, parent=None):
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.harden_normals = True
    if parent:
        obj.parent = parent
    return obj


def box(name, x0, x1, y0, y1, z0, z1, mat, bevel=0.01, parent=None):
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1)
    bmesh.ops.transform(bm, verts=bm.verts, matrix=Matrix.Diagonal((x1 - x0, y1 - y0, z1 - z0, 1)))
    bmesh.ops.translate(bm, verts=bm.verts, vec=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    bm.to_mesh(mesh)
    bm.free()
    return link(bpy.data.objects.new(name, mesh), mat, bevel, parent)


def cylinder(name, x, y, z0, z1, r, mat, segments=32, axis="Z", bevel=0.0, parent=None, r_top=None):
    """Along Z by default; axis "X" lays it along X with (x, y) read as (y, z)."""
    mesh = bpy.data.meshes.new(name)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=segments, radius1=r, radius2=r if r_top is None else r_top, depth=z1 - z0)
    rot = {"Z": Matrix(), "X": Matrix.Rotation(math.pi / 2, 4, "Y")}[axis]
    bmesh.ops.transform(bm, verts=bm.verts, matrix=rot)
    centre = {"Z": (x, y, (z0 + z1) / 2), "X": ((z0 + z1) / 2, x, y)}[axis]
    bmesh.ops.translate(bm, verts=bm.verts, vec=centre)
    bm.to_mesh(mesh)
    bm.free()
    obj = link(bpy.data.objects.new(name, mesh), mat, bevel, parent)
    obj.data.shade_smooth()
    return obj


def ring(name, x, y, z0, z1, r_out, r_in, mat):
    outer = cylinder(name, x, y, z0, z1, r_out, mat, segments=40)
    hole = cylinder(name + "_hole", x, y, z0 - 0.01, z1 + 0.01, r_in, mat, segments=40)
    cut = outer.modifiers.new("hole", "BOOLEAN")
    cut.object = hole
    cut.operation = "DIFFERENCE"
    hole.hide_render = True
    outer.modifiers.move(len(outer.modifiers) - 1, 0)
    return outer


def cable(name, points, radius, mat):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = radius
    curve.bevel_resolution = 4
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, p in zip(spline.bezier_points, points):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    return obj


def fan(x, y, z, r):
    """A round fan housing on a square plinth, open on top."""
    box("fan_plinth", x - r - 0.04, x + r + 0.04, y - r - 0.04, y + r + 0.04, z, z + 0.05, STEEL, bevel=0.015)
    ring("fan_housing", x, y, z + 0.05, z + 0.17, r, r - 0.045, STEEL)
    ring("fan_lip", x, y, z + 0.17, z + 0.19, r + 0.01, r - 0.03, STEEL)
    cylinder("fan_well", x, y, z + 0.04, z + 0.06, r - 0.04, RUBBER)
    rotor = bpy.data.objects.new("fan_rotor", None)
    rotor.location = (x, y, z + 0.11)
    bpy.context.collection.objects.link(rotor)
    cylinder("fan_hub", 0, 0, -0.03, 0.04, r * 0.28, DARK, parent=rotor)
    cylinder("fan_cap", 0, 0, 0.04, 0.055, r * 0.16, BRIGHT, parent=rotor)
    for i in range(7):
        blade = box("fan_blade", r * 0.2, r * 0.9, -r * 0.13, r * 0.13, -0.007, 0.007, BLADE, bevel=0, parent=rotor)
        blade.rotation_euler = Euler((0.5, 0, i * 2 * math.pi / 7), "ZXY")
    for k in (-0.5, 0, 0.5):
        box("fan_guard", x - r, x + r, y + k * r - 0.01, y + k * r + 0.01, z + 0.165, z + 0.18, DARK, bevel=0)
    return rotor


def rack(x0, x1, y0, y1, z0, z1):
    """One cabinet holding two cards: two vented doors on the front, two fans on top."""
    box("cabinet", x0, x1, y0, y1, z0, z1, PAINT, bevel=0.035)
    for x in (x0, x1):
        box("post", x - 0.055, x + 0.055, y0 - 0.03, y0 + 0.08, z0, z1 + 0.02, STEEL, bevel=0.015)
    box("top_band", x0, x1, y0 - 0.025, y0 + 0.04, z1 - 0.07, z1, STEEL, bevel=0.012)
    box("kick_plate", x0 + 0.05, x1 - 0.05, y0 - 0.035, y0 + 0.02, z0, z0 + 0.09, DARK, bevel=0.008)
    bay = (x1 - x0) / 2
    for i in range(2):
        bx0 = x0 + i * bay + 0.1
        bx1 = bx0 + bay - 0.2
        box("door", bx0, bx1, y0 - 0.03, y0 + 0.01, z0 + 0.13, z1 - 0.11, PAINT, bevel=0.015)
        # A window onto the card's heat sink.
        wx0, wx1, wz0, wz1 = bx0 + 0.06, bx1 - 0.16, z1 - 0.47, z1 - 0.17
        box("window_frame", wx0 - 0.03, wx1 + 0.03, y0 - 0.05, y0 - 0.02, wz0 - 0.03, wz1 + 0.03, STEEL, bevel=0.01)
        box("window_back", wx0, wx1, y0 - 0.055, y0 - 0.04, wz0, wz1, GLASS, bevel=0)
        for k in range(7):
            fx = wx0 + 0.02 + k * (wx1 - wx0 - 0.04) / 6
            box("fin", fx - 0.009, fx + 0.009, y0 - 0.065, y0 - 0.05, wz0 + 0.02, wz1 - 0.02, COPPER, bevel=0)
        for k in range(2):
            zz = z0 + 0.32 - k * 0.07
            box("vent", wx0, wx1, y0 - 0.045, y0 - 0.025, zz - 0.03, zz, DARK, bevel=0.004)
        for k in range(3):
            zz = z1 - 0.2 - k * 0.085
            box("led", bx1 - 0.12, bx1 - 0.07, y0 - 0.05, y0 - 0.025, zz - 0.03, zz, LED, bevel=0)
    # Panel seams and rivets on the roof between and around the fans.
    for x in (x0 + bay, ):
        box("seam", x - 0.008, x + 0.008, y0 + 0.06, y1 - 0.04, z1, z1 + 0.006, DARK, bevel=0)
    box("seam", x0 + 0.06, x1 - 0.06, y1 - 0.12, y1 - 0.105, z1, z1 + 0.006, DARK, bevel=0)
    for x in (x0 + 0.1, x0 + bay - 0.06, x0 + bay + 0.06, x1 - 0.1):
        for y in (y0 + 0.12, y1 - 0.17):
            cylinder("rivet", x, y, z1, z1 + 0.012, 0.014, BRIGHT, segments=8)
    return [fan(x0 + bay * (i + 0.5), (y0 + y1) / 2 - 0.08, z1, 0.29) for i in range(2)]


def build_horizontal():
    for y in (-0.7, 0.62):
        box("rail", -1.92, 1.92, y - 0.08, y + 0.08, 0.0, 0.13, DARK, bevel=0.02)
    for x in (-1.84, 1.84):
        box("cross", x - 0.07, x + 0.07, -0.78, 0.7, 0.02, 0.12, DARK, bevel=0.01)
        for y in (-0.8, 0.72):
            box("foot", x - 0.14, x + 0.14, y - 0.13, y + 0.13, 0.0, 0.03, STEEL, bevel=0.01)
            cylinder("bolt", x, y, 0.03, 0.06, 0.03, BRIGHT, segments=6)

    z0, z1 = 0.13, 0.92
    fans = rack(-1.75, -0.12, -0.7, 0.45, z0, z1) + rack(0.12, 1.75, -0.7, 0.45, z0, z1)

    # Between the racks: a riser carrying the cards' power bus and cables.
    box("riser", -0.12, 0.12, -0.55, 0.45, z0, z1 - 0.12, DARK, bevel=0.01)
    for k, x in enumerate((-0.07, 0.0, 0.07)):
        cable(f"bus{k}", [(x, -0.62, 0.15), (x, -0.62, 0.55), (x, -0.45, z1 + 0.05), (x, 0.1, z1 + 0.12), (x, 0.42, z1 + 0.1)], 0.028, RUBBER if k != 1 else COPPER)

    # Exhaust duct along the back, rising into a short stack on the east end.
    cylinder("duct", 0.55, z1 + 0.05, -1.75, 1.45, 0.17, STEEL, axis="X")
    for x in (-1.3, -0.45, 0.45, 1.3):
        cylinder("duct_band", 0.55, z1 + 0.05, x - 0.03, x + 0.03, 0.185, DARK, axis="X")
    cylinder("stack", 1.55, 0.5, z0, z1 + 0.45, 0.2, STEEL)
    cylinder("stack_band", 1.55, 0.5, z1 + 0.2, z1 + 0.25, 0.215, DARK)
    ring("stack_mouth", 1.55, 0.5, z1 + 0.45, z1 + 0.5, 0.22, 0.15, BRIGHT)
    cylinder("stack_soot", 1.55, 0.5, z1 + 0.3, z1 + 0.46, 0.15, RUBBER)

    # Power junction on the west end with hazard stripe.
    box("junction", -1.95, -1.7, -0.5, 0.25, z0, 0.7, DARK, bevel=0.02)
    for k in range(4):
        box("hazard", -1.96 + k * 0.0, -1.7, -0.52 + k * 0.19, -0.42 + k * 0.19, 0.6, 0.68, HAZARD, bevel=0)

    # Data port for the blue cable: a short mast and insulator at the back.
    cylinder("mast", -1.5, 0.55, z1, z1 + 0.42, 0.04, STEEL)
    cylinder("insulator", -1.5, 0.55, z1 + 0.42, z1 + 0.54, 0.07, CERAMIC, segments=16, r_top=0.05)
    cylinder("insulator_cap", -1.5, 0.55, z1 + 0.54, z1 + 0.58, 0.04, COPPER, segments=16)
    return fans


def setup_scene():
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.resolution_x = CANVAS_TILES[0] * PX_PER_TILE
    scene.render.resolution_y = CANVAS_TILES[1] * PX_PER_TILE
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"

    cam_data = bpy.data.cameras.new("camera")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = max(CANVAS_TILES)
    cam = bpy.data.objects.new("camera", cam_data)
    scene.collection.objects.link(cam)
    cam.location = Vector((0, -math.cos(ELEVATION), math.sin(ELEVATION))) * 30
    cam.rotation_euler = (math.pi / 2 - ELEVATION, 0, 0)
    scene.camera = cam

    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 5.5
    sun_data.angle = math.radians(3)
    sun_data.color = (1.0, 0.96, 0.9)
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = SUN.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(sun)

    world = bpy.data.worlds.new("world")
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.75, 0.8, 0.9, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.6
    scene.world = world
    return scene


def render(scene, name):
    scene.render.filepath = str(OUT / f"{name}.png")
    bpy.ops.render.render(write_still=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = setup_scene()
    build_horizontal()
    model = [o for o in scene.objects if o.type in {"MESH", "CURVE"}]
    render(scene, "main")

    bpy.ops.mesh.primitive_plane_add(size=20)
    ground = bpy.context.active_object
    ground.is_shadow_catcher = True
    for obj in model:
        obj.visible_camera = False
    render(scene, "shadow")
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "gpu-server.blend"))


main()
