"""Model and render the GPU server sprites in Blender.

    Blender --background --factory-startup --python gpu_server.py -- OUT_DIR [vertical]

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
CANVAS_TILES = (6, 5)
SUN = Vector((1.0, -0.25, -1.2)).normalized()

bpy.ops.wm.read_factory_settings(use_empty=True)


def material(name, colour, metallic=0.6, roughness=0.55, grime=1.0, wear=0.3, chip=None, edge=None, rust=0.6, bump=0.4, radius=0.03, emission=None):
    """chip is wear from inward occlusion, which marks thin parts as all edge;
    edge is the bare metal the paint wears through to."""
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
    chip = wear if chip is None else chip

    def node(kind, **inputs):
        n = nodes.new(kind)
        for key, value in inputs.items():
            n.inputs[key].default_value = value
        return n

    def maths(op, a, b):
        n = nodes.new("ShaderNodeMath")
        n.operation = op
        for i, v in enumerate((a, b)):
            if isinstance(v, (int, float)):
                n.inputs[i].default_value = v
            else:
                links.new(v, n.inputs[i])
        return n.outputs["Value"]

    def mix(a, b, fac):
        n = nodes.new("ShaderNodeMix")
        n.data_type = "RGBA"
        for socket, v in (("A", a), ("B", b)):
            if isinstance(v, tuple):
                n.inputs[socket].default_value = (*v, 1)
            else:
                links.new(v, n.inputs[socket])
        links.new(fac, n.inputs["Factor"])
        return n.outputs["Result"]

    def remap(value, lo, hi, to_lo, to_hi):
        n = node("ShaderNodeMapRange")
        n.inputs["From Min"].default_value = lo
        n.inputs["From Max"].default_value = hi
        n.inputs["To Min"].default_value = to_lo
        n.inputs["To Max"].default_value = to_hi
        links.new(value, n.inputs["Value"])
        return n.outputs["Result"]

    def ramp(value, lo, hi):
        return maths("MINIMUM", maths("MAXIMUM", maths("DIVIDE", maths("SUBTRACT", value, lo), hi - lo), 0.0), 1.0)

    # Rust: fine specks everywhere, gathering into patches. Object
    # coordinates are world coordinates here, so the grain is the same size on
    # every part.
    coords = nodes.new("ShaderNodeTexCoord")
    speck = node("ShaderNodeTexNoise", Scale=30.0, Detail=2.0, Roughness=0.5)
    patch = node("ShaderNodeTexNoise", Scale=4.0, Detail=5.0, Roughness=0.6)
    for n in (speck, patch):
        links.new(coords.outputs["Object"], n.inputs["Vector"])
    rust_fac = maths("MULTIPLY", maths("ADD", ramp(speck.outputs["Fac"], 0.55, 0.62), ramp(patch.outputs["Fac"], 0.5, 0.66)), rust)
    rust_col = mix((0.42, 0.17, 0.06), (0.16, 0.07, 0.03), speck.outputs["Fac"])
    base = mix(colour, rust_col, maths("MINIMUM", rust_fac, 1.0))

    # Grunge: blotchy brightness variation across every surface.
    grunge = node("ShaderNodeTexNoise", Scale=11.0, Detail=8.0, Roughness=0.7)
    links.new(coords.outputs["Object"], grunge.inputs["Vector"])
    shade = remap(grunge.outputs["Fac"], 0.3, 0.7, 1 - 0.6 * min(grime, 1.0), 1 + 0.45 * min(grime, 1.0))
    graded = nodes.new("ShaderNodeMix")
    graded.data_type = "RGBA"
    graded.blend_type = "MULTIPLY"
    graded.inputs["Factor"].default_value = 1.0
    links.new(base, graded.inputs["A"])
    links.new(shade, graded.inputs["B"])
    base = graded.outputs["Result"]

    # Grime in the crevices, and dust settling towards the ground.
    ao = node("ShaderNodeAmbientOcclusion", Distance=0.25)
    crevice = maths("MULTIPLY", maths("SUBTRACT", 1.0, ao.outputs["AO"]), 2.6 * grime)
    dirty = mix(base, (0.02, 0.013, 0.008), maths("MINIMUM", crevice, 0.92))
    height = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Object"], height.inputs["Vector"])
    dust = maths("MULTIPLY", ramp(maths("SUBTRACT", 0.7, height.outputs["Z"]), 0.0, 0.7), 0.55 * min(grime, 1.0))
    dirty = mix(dirty, (0.13, 0.1, 0.07), maths("MULTIPLY", dust, ramp(grunge.outputs["Fac"], 0.35, 0.6)))

    # Worn bright edges, where a bevelled normal departs from the true one.
    bevel = node("ShaderNodeBevel", Radius=radius)
    geometry = nodes.new("ShaderNodeNewGeometry")
    dot = nodes.new("ShaderNodeVectorMath")
    dot.operation = "DOT_PRODUCT"
    links.new(bevel.outputs["Normal"], dot.inputs[0])
    links.new(geometry.outputs["Normal"], dot.inputs[1])
    # Inward-cast occlusion is low only near convex edges.
    inside = node("ShaderNodeAmbientOcclusion", Distance=0.03)
    inside.inside = True
    inside.samples = 8
    chipped = ramp(maths("SUBTRACT", 1.0, inside.outputs["AO"]), 0.15, 0.45)
    edge_fac = maths("MINIMUM", maths("ADD", maths("MULTIPLY", maths("SUBTRACT", 1.0, dot.outputs["Value"]), 14.0 * wear), maths("MULTIPLY", chipped, 1.2 * chip)), 1.0)
    worn = mix(dirty, edge or tuple(min(1.0, c * 2.2 + 0.18) for c in colour), edge_fac)
    links.new(worn, bsdf.inputs["Base Color"])
    if edge:
        links.new(maths("ADD", metallic, maths("MULTIPLY", edge_fac, 0.95 - metallic)), bsdf.inputs["Metallic"])
        links.new(maths("ADD", roughness, maths("MULTIPLY", edge_fac, 0.25 - roughness)), bsdf.inputs["Roughness"])

    if bump:
        grain = node("ShaderNodeTexNoise", Scale=25.0, Detail=6.0)
        links.new(coords.outputs["Object"], grain.inputs["Vector"])
        bumper = node("ShaderNodeBump", Strength=bump, Distance=0.02)
        links.new(grain.outputs["Fac"], bumper.inputs["Height"])
        links.new(bevel.outputs["Normal"], bumper.inputs["Normal"])
        links.new(bumper.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        links.new(bevel.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


BARE = (0.68, 0.63, 0.56)
BLACK = material("black", (0.024, 0.019, 0.015), metallic=0.1, roughness=0.72, rust=0.2, wear=0.45, edge=BARE)
GUNMETAL = material("gunmetal", (0.07, 0.062, 0.052), metallic=0.45, roughness=0.68, rust=0.35, wear=0.5, edge=BARE)
SILVER = material("silver", (0.42, 0.4, 0.36), metallic=0.7, roughness=0.6, rust=0.4, wear=0.3, grime=1.0)
DARK = material("dark", (0.016, 0.015, 0.014), metallic=0.3, roughness=0.7, rust=0.1, wear=0.2, grime=0.8)
VOID = material("void", (0.004, 0.004, 0.005), metallic=0.0, roughness=0.9, rust=0.0, wear=0.0, grime=0.0, bump=0.0)
# Bright fin tips come from the bevel term alone.
FIN = material("fin", (0.04, 0.04, 0.042), metallic=0.6, roughness=0.6, rust=0.2, wear=0.5, chip=0.0, edge=(0.5, 0.48, 0.45), grime=1.0, bump=0.2, radius=0.012)
BLUE = material("blue", (0.05, 0.11, 0.26), metallic=0.4, roughness=0.62, rust=0.2, wear=0.45, edge=BARE, grime=1.0)
COPPER = material("copper", (0.55, 0.27, 0.14), metallic=0.8, roughness=0.55, rust=0.15, wear=0.0, grime=1.0, bump=0.2)
STEEL = material("steel", (0.3, 0.29, 0.27), metallic=0.6, roughness=0.66, rust=0.45, wear=0.35, edge=BARE)
YELLOW = material("yellow", (0.5, 0.38, 0.03), metallic=0.2, roughness=0.6, rust=0.3, wear=0.5, edge=BARE)
ORANGE = material("orange", (0.55, 0.16, 0.02), metallic=0.2, roughness=0.6, rust=0.3, wear=0.5, edge=BARE)
CABLES = {
    "black": material("cable_black", (0.02, 0.02, 0.02), metallic=0.0, roughness=0.5, rust=0.1, bump=0.6),
    "red": material("cable_red", (0.3, 0.035, 0.025), metallic=0.0, roughness=0.5, rust=0.1, bump=0.6),
    "blue": material("cable_blue", (0.03, 0.08, 0.3), metallic=0.0, roughness=0.5, rust=0.1, bump=0.6),
    "green": material("cable_green", (0.04, 0.17, 0.06), metallic=0.0, roughness=0.5, rust=0.1, bump=0.6),
}
BLADE = material("blade", (0.03, 0.03, 0.033), metallic=0.4, roughness=0.6, rust=0.1, wear=0.2, chip=0.0, edge=BARE, bump=0.0)
LED = material("led", (0.05, 0.5, 0.1), emission=((0.25, 1.0, 0.3), 10))


def place(obj, mat, bevel, parent):
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("bevel", "BEVEL")
        mod.width = bevel
        mod.segments = 2
        mod.limit_method = "ANGLE"
        mod.harden_normals = True
    obj.parent = parent
    return obj


def box(x0, x1, y0, y1, z0, z1, mat, bevel=0.012, parent=None):
    mesh = bpy.data.meshes.new("box")
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1)
    bmesh.ops.transform(bm, verts=bm.verts, matrix=Matrix.Diagonal((x1 - x0, y1 - y0, z1 - z0, 1)))
    bmesh.ops.translate(bm, verts=bm.verts, vec=((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    bm.to_mesh(mesh)
    bm.free()
    return place(bpy.data.objects.new("box", mesh), mat, bevel, parent)


def tube(a, b, r, mat, segments=24, r_end=None, bevel=0.0, parent=None, open_ends=False):
    """A cylinder (or cone) from point a to point b."""
    a, b = Vector(a), Vector(b)
    mesh = bpy.data.meshes.new("tube")
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=not open_ends, segments=segments, radius1=r, radius2=r if r_end is None else r_end, depth=(b - a).length)
    turn = (b - a).to_track_quat("Z", "Y").to_matrix().to_4x4()
    bmesh.ops.transform(bm, verts=bm.verts, matrix=Matrix.Translation((a + b) / 2) @ turn)
    bm.to_mesh(mesh)
    bm.free()
    obj = place(bpy.data.objects.new("tube", mesh), mat, bevel, parent)
    obj.data.shade_smooth()
    return obj


def ring(a, b, r, thickness, mat, parent=None, segments=32):
    obj = tube(a, b, r, mat, segments=segments, parent=parent, open_ends=True)
    solid = obj.modifiers.new("solidify", "SOLIDIFY")
    solid.thickness = thickness
    solid.offset = -1
    obj.modifiers.move(len(obj.modifiers) - 1, 0)
    return obj


def cable(points, r, mat, parent=None):
    curve = bpy.data.curves.new("cable", "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = r
    curve.bevel_resolution = 3
    curve.resolution_u = 16
    curve.use_fill_caps = True
    spline = curve.splines.new("BEZIER")
    spline.bezier_points.add(len(points) - 1)
    for bp, p in zip(spline.bezier_points, points):
        bp.co = p
        bp.handle_left_type = bp.handle_right_type = "AUTO"
    obj = bpy.data.objects.new("cable", curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat)
    obj.parent = parent
    return obj


def group(location, rotation=(0, 0, 0), parent=None):
    empty = bpy.data.objects.new("group", None)
    empty.location = location
    empty.rotation_euler = rotation
    empty.parent = parent
    bpy.context.collection.objects.link(empty)
    return empty


def rivets(points, mat=SILVER, r=0.016, axis=(0, -1, 0), parent=None):
    axis = Vector(axis)
    for p in points:
        p = Vector(p)
        tube(p, p + axis * 0.012, r, mat, segments=8, parent=parent)


def fan(centre, r, parent):
    """A fan in a ring frame, facing local -Y."""
    hub = group(centre, (math.pi / 2, 0, 0), parent)
    ring((0, 0, -0.03), (0, 0, 0.05), r, 0.045, GUNMETAL, parent=hub)
    tube((0, 0, -0.03), (0, 0, -0.01), r - 0.01, VOID, parent=hub)
    rotor = group((0, 0, 0.0), parent=hub)
    tube((0, 0, -0.01), (0, 0, 0.03), r * 0.3, DARK, parent=rotor)
    tube((0, 0, 0.03), (0, 0, 0.036), r * 0.2, SILVER, parent=rotor)
    for i in range(7):
        blade = box(r * 0.25, r * 0.9, -r * 0.16, r * 0.16, -0.008, 0.008, BLADE, bevel=0.004, parent=rotor)
        blade.rotation_euler = Euler((0.55, 0, i * 2 * math.pi / 7), "ZXY")
    return rotor


# The chassis lies along the footprint, its front to the camera: x along the
# 4 tiles, y from front (-) to back (+), z up, in tiles.
X0, X1, Y0, Y1, TOP = -1.9, 1.9, -0.82, 0.8, 0.45
FAN_Y = -0.05


def build_chassis():
    """A heavy open-topped tray on rubber feet."""
    for x in (X0 + 0.15, X1 - 0.15):
        for y in (Y0 + 0.12, Y1 - 0.12):
            box(x - 0.1, x + 0.1, y - 0.1, y + 0.1, 0.0, 0.07, BLACK, bevel=0.02)
    box(X0, X1, Y0, Y1, 0.06, 0.13, GUNMETAL, bevel=0.025)
    wall = 0.06
    for y0, y1 in ((Y0, Y0 + wall), (Y1 - wall, Y1)):
        box(X0, X1, y0, y1, 0.1, TOP, STEEL, bevel=0.025)
    for x0, x1 in ((X0, X0 + wall), (X1 - wall, X1)):
        box(x0, x1, Y0, Y1, 0.1, TOP, STEEL, bevel=0.025)
    # A rolled lip along the top of every wall, and rack ears at the front.
    for y in (Y0 + wall / 2, Y1 - wall / 2):
        box(X0 - 0.01, X1 + 0.01, y - 0.05, y + 0.05, TOP - 0.03, TOP + 0.02, STEEL, bevel=0.02)
    for x in (X0 - 0.06, X1 + 0.06):
        box(x - 0.05, x + 0.05, Y0 - 0.02, Y0 + 0.18, 0.18, TOP - 0.04, BLACK, bevel=0.02)
        rivets([(x, Y0 - 0.02, z) for z in (0.24, 0.36)], r=0.03)
    # Vent slots punched in the back wall.
    for k in range(9):
        x = X0 + 0.35 + k * 0.38
        box(x - 0.11, x + 0.11, Y1 + 0.005, Y1 + 0.02, 0.24, 0.32, VOID, bevel=0.01)


def build_fan_wall():
    """Four big fans across the chassis in a black carrier, yellow pull tabs."""
    box(X0 + 0.06, X1 - 0.06, FAN_Y, FAN_Y + 0.18, 0.13, TOP - 0.02, DARK, bevel=0.03)
    rotors = []
    for k in range(4):
        x = -1.2 + k * 0.8
        box(x - 0.36, x + 0.36, FAN_Y - 0.05, FAN_Y, 0.13, TOP - 0.02, BLACK, bevel=0.025)
        rotors.append(fan((x, FAN_Y - 0.05, 0.29), 0.15, None))
        box(x - 0.24, x + 0.24, FAN_Y - 0.08, FAN_Y + 0.02, TOP - 0.05, TOP + 0.03, YELLOW, bevel=0.02)
        box(x + 0.22, x + 0.3, FAN_Y - 0.1, FAN_Y - 0.02, TOP - 0.1, TOP + 0.05, YELLOW, bevel=0.02)
    return rotors


def build_back():
    """Behind the fans, sheet-metal air ducts over the GPUs, with yellow
    grab strips, and a row of memory between them."""
    for x0, x1 in ((X0 + 0.1, -0.15), (0.15, X1 - 0.75)):
        box(x0, x1, FAN_Y + 0.24, Y1 - 0.1, 0.13, TOP - 0.06, STEEL, bevel=0.03)
        box(x0 + 0.08, x1 - 0.08, FAN_Y + 0.3, Y1 - 0.16, TOP - 0.06, TOP - 0.02, STEEL, bevel=0.02)
        mid = (x0 + x1) / 2
        box(mid - 0.35, mid + 0.35, 0.38, 0.46, TOP - 0.02, TOP + 0.01, YELLOW, bevel=0.015)
        for k in range(3):
            xx = x0 + 0.2 + k * 0.18
            box(xx - 0.06, xx + 0.06, FAN_Y + 0.33, FAN_Y + 0.5, TOP - 0.025, TOP - 0.015, VOID, bevel=0.005)
    for k in range(4):
        x = -0.11 + k * 0.075
        box(x - 0.025, x + 0.025, FAN_Y + 0.26, Y1 - 0.12, 0.13, 0.36, BLUE if k % 2 else BLACK, bevel=0.01)
    # Two power supplies at the back right, handles out of the back wall.
    for k in range(2):
        y = 0.2 + k * 0.28
        box(X1 - 0.68, X1 - 0.07, y - 0.12, y + 0.12, 0.13, 0.4, GUNMETAL, bevel=0.025)
        box(X1 - 0.6, X1 - 0.2, y - 0.08, y + 0.08, 0.4, 0.42, VOID, bevel=0.01)
        box(X1 - 0.02, X1 + 0.1, y - 0.08, y + 0.08, 0.2, 0.26, BLACK, bevel=0.02)


def build_front():
    """In front of the fans the board, heat sinks and looms; on the front
    panel drive bays with orange tabs, ports and cables spilling out."""
    box(X0 + 0.06, X1 - 0.06, Y0 + 0.06, FAN_Y - 0.05, 0.13, 0.16, DARK, bevel=0.01)
    for x in (-1.1, 0.4):
        box(x - 0.3, x + 0.3, -0.62, -0.22, 0.16, 0.22, GUNMETAL, bevel=0.02)
        for k in range(5):
            xx = x - 0.24 + k * 0.12
            box(xx - 0.035, xx + 0.035, -0.6, -0.24, 0.22, 0.36, FIN, bevel=0.015)
    for x, y in ((-0.45, -0.35), (-0.3, -0.5), (1.15, -0.35), (1.3, -0.5)):
        tube((x, y, 0.16), (x, y, 0.28), 0.055, BLUE, segments=14)
        tube((x, y, 0.28), (x, y, 0.3), 0.045, STEEL, segments=14)
    # Looms from the fan wall's power to the board.
    for k, colour in enumerate(("black", "red", "black", "black")):
        x = 1.55 + k * 0.07
        cable([(x, FAN_Y - 0.06, 0.26), (x, -0.3, 0.21), (x - 0.4, -0.55, 0.2), (x - 0.9, -0.66, 0.2)], 0.035, CABLES[colour])

    # The front panel: a bay cage on the left half, ports on the right.
    yf = Y0 - 0.01
    box(X0 + 0.12, -0.15, yf - 0.06, yf, 0.15, TOP - 0.06, BLACK, bevel=0.02)
    for k in range(5):
        x = X0 + 0.38 + k * 0.32
        box(x - 0.13, x + 0.13, yf - 0.09, yf - 0.04, 0.18, TOP - 0.08, GUNMETAL, bevel=0.02)
        box(x - 0.1, x + 0.1, yf - 0.11, yf - 0.08, 0.3, 0.35, ORANGE, bevel=0.012)
        box(x + 0.06, x + 0.1, yf - 0.115, yf - 0.09, 0.21, 0.25, LED, bevel=0)
    ports = []
    for k in range(6):
        x = 0.1 + k * 0.28
        box(x - 0.1, x + 0.1, yf - 0.05, yf, 0.2, 0.36, STEEL, bevel=0.015)
        box(x - 0.06, x + 0.06, yf - 0.06, yf - 0.04, 0.24, 0.32, VOID, bevel=0.005)
        ports.append((x, yf - 0.06, 0.28))
    # Thick cables loop out of the ports and sag to the ground in front.
    for k in range(0, 6, 2):
        a, b = ports[k], ports[k + 1]
        for j, colour in enumerate(("black", "black")):
            dy = -0.12 - j * 0.08
            cable([a, (a[0], a[1] + dy, a[2] - 0.08), ((a[0] + b[0]) / 2, a[1] + dy - 0.06, 0.05),
                   (b[0], b[1] + dy, b[2] - 0.08), b], 0.04, CABLES[colour])
    box(-0.1, 0.0, yf - 0.06, yf, 0.15, TOP - 0.06, YELLOW, bevel=0.015)


def build_horizontal():
    build_chassis()
    rotors = build_fan_wall()
    build_back()
    build_front()
    return rotors


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
    sun_data.energy = 7.0
    sun_data.angle = math.radians(3)
    sun_data.color = (1.0, 0.9, 0.76)
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = SUN.to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(sun)
    # The sun is behind, so a shadowless fill from the front left keeps the
    # faces the camera sees from going flat black.
    fill_data = bpy.data.lights.new("fill", "SUN")
    fill_data.energy = 1.6
    fill_data.use_shadow = False
    fill_data.color = (1.0, 0.94, 0.86)
    fill = bpy.data.objects.new("fill", fill_data)
    fill.rotation_euler = Vector((0.5, 1.0, -0.6)).normalized().to_track_quat("-Z", "Y").to_euler()
    scene.collection.objects.link(fill)

    world = bpy.data.worlds.new("world")
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.85, 0.82, 0.76, 1)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.35
    scene.world = world
    return scene


def render(scene, name):
    scene.render.filepath = str(OUT / f"{name}.png")
    bpy.ops.render.render(write_still=True)


def hero(scene):
    """A three-quarter close-up for judging the model against the concepts."""
    cam = scene.camera
    cam.data.type = "PERSP"
    cam.data.lens = 50
    cam.location = Vector((5.0, -6.2, 4.4))
    cam.rotation_euler = (Vector((0.0, 0.0, 0.7)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.resolution_x, scene.render.resolution_y = 1536, 1024
    render(scene, "hero")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = setup_scene()
    build_horizontal()
    if "vertical" in sys.argv:
        # The east and west picture: turned so the I/O box is at the front.
        turn = Matrix.Rotation(math.pi / 2, 4, "Z")
        for obj in scene.objects:
            if obj.type in {"MESH", "CURVE", "EMPTY"} and obj.parent is None:
                obj.matrix_world = turn @ obj.matrix_world
    if "hero" in sys.argv:
        hero(scene)
        return
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
