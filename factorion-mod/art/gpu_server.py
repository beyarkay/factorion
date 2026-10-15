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
DARK = material("dark", (0.016, 0.015, 0.014), metallic=0.3, roughness=0.7, rust=0.0, wear=0.2, grime=0.45)
VOID = material("void", (0.004, 0.004, 0.005), metallic=0.0, roughness=0.9, rust=0.0, wear=0.0, grime=0.0, bump=0.0)
# Bright fin tips come from the bevel term alone.
FIN = material("fin", (0.04, 0.04, 0.042), metallic=0.6, roughness=0.6, rust=0.2, wear=0.5, chip=0.0, edge=(0.5, 0.48, 0.45), grime=1.0, bump=0.2, radius=0.012)
BLUE = material("blue", (0.05, 0.11, 0.26), metallic=0.4, roughness=0.62, rust=0.2, wear=0.45, edge=BARE, grime=1.0)
COPPER = material("copper", (0.55, 0.27, 0.14), metallic=0.8, roughness=0.55, rust=0.15, wear=0.0, grime=1.0, bump=0.2)
STEEL = material("steel", (0.3, 0.29, 0.27), metallic=0.6, roughness=0.66, rust=0.45, wear=0.35, edge=BARE)
YELLOW = material("yellow", (0.5, 0.38, 0.03), metallic=0.2, roughness=0.6, rust=0.3, wear=0.5, edge=BARE)
ORANGE = material("orange", (0.55, 0.16, 0.02), metallic=0.2, roughness=0.6, rust=0.3, wear=0.5, edge=BARE)
# Black anodised aluminium, worn to bare metal on the edges.
ALU = material("aluminium", (0.03, 0.03, 0.034), metallic=0.7, roughness=0.45, rust=0.0, wear=0.45, edge=(0.55, 0.55, 0.57), grime=0.45)
BRIGHT = material("bright_aluminium", (0.45, 0.45, 0.47), metallic=0.85, roughness=0.42, rust=0.05, wear=0.2, grime=0.6)
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


# An open-frame rig: x along the 4 tiles, y from front (-) to back (+), z
# up, in tiles.
X0, X1, Y0, Y1, TOP = -1.85, 1.85, -0.82, 0.82, 1.02
PROFILE = 0.075
CARD_XS = tuple(-1.45 + k * 0.58 for k in range(6))
CARD_FRONT, CARD_BACK, CARD_BOTTOM = -0.62, 0.42, 0.42


def extrusion(a, b):
    """A square aluminium extrusion between two points along one axis."""
    lo = [min(p, q) - PROFILE / 2 for p, q in zip(a, b)]
    hi = [max(p, q) + PROFILE / 2 for p, q in zip(a, b)]
    box(lo[0], hi[0], lo[1], hi[1], lo[2], hi[2], ALU, bevel=0.012)


def build_frame():
    for x in (X0, X1):
        for y in (Y0, Y1):
            extrusion((x, y, 0.04), (x, y, TOP))
            box(x - 0.08, x + 0.08, y - 0.08, y + 0.08, 0.0, 0.04, BLACK, bevel=0.015)
    for z in (0.06, TOP):
        for y in (Y0, Y1):
            extrusion((X0, y, z), (X1, y, z))
        for x in (X0, X1):
            extrusion((x, Y0, z), (x, Y1, z))
    # The rail the cards' brackets screw to.
    extrusion((X0, CARD_FRONT - 0.04, CARD_BOTTOM + 0.2), (X1, CARD_FRONT - 0.04, CARD_BOTTOM + 0.2))
    extrusion((X0, CARD_BACK + 0.06, CARD_BOTTOM - 0.02), (X1, CARD_BACK + 0.06, CARD_BOTTOM - 0.02))
    # Corner brackets where rails meet posts.
    for x in (X0, X1):
        for y in (Y0, Y1):
            for z in (0.12, TOP - 0.07):
                box(x - 0.05, x + 0.05, y - 0.05, y + 0.05, z - 0.04, z + 0.04, STEEL, bevel=0.01)


def build_base():
    """Tray, board and power supply under the cards."""
    box(X0 + 0.04, X1 - 0.04, Y0 + 0.04, Y1 - 0.04, 0.06, 0.1, ALU, bevel=0.015)
    box(-1.75, -0.15, -0.78, 0.1, 0.1, 0.13, DARK, bevel=0.01)
    box(-1.45, -1.05, -0.45, -0.1, 0.13, 0.3, FIN, bevel=0.03)
    for k in range(4):
        x = -0.85 + k * 0.08
        box(x - 0.025, x + 0.025, -0.5, -0.05, 0.13, 0.24, BLUE if k % 2 else BLACK, bevel=0.01)
    # Port panel at the front edge of the board.
    box(-1.7, -0.9, -0.82, -0.74, 0.13, 0.3, STEEL, bevel=0.015)
    for k in range(4):
        x = -1.6 + k * 0.2
        box(x - 0.06, x + 0.06, -0.835, -0.815, 0.18, 0.25, VOID, bevel=0.005)
        box(x + 0.04, x + 0.07, -0.84, -0.82, 0.26, 0.28, LED, bevel=0)
    # The power supply, its fan up.
    box(0.5, 1.75, 0.15, 0.78, 0.1, 0.4, BLACK, bevel=0.03)
    ring((1.1, 0.46, 0.4), (1.1, 0.46, 0.42), 0.2, 0.03, ALU)
    tube((1.1, 0.46, 0.395), (1.1, 0.46, 0.4), 0.18, VOID)
    for k in range(3):
        box(0.92 + k * 0.12, 0.98 + k * 0.12, 0.3, 0.62, 0.41, 0.425, ALU, bevel=0.005)
    box(1.62, 1.76, 0.25, 0.68, 0.14, 0.36, ALU, bevel=0.015)


def build_card(x):
    """A thick card standing on the rails: dark shroud, aluminium end caps,
    a vented bracket at the front and its power socket on top."""
    w = 0.14
    box(x - w, x + w, CARD_FRONT + 0.06, CARD_BACK - 0.06, CARD_BOTTOM, TOP - 0.08, DARK, bevel=0.035)
    for y0, y1 in ((CARD_FRONT, CARD_FRONT + 0.16), (CARD_BACK - 0.12, CARD_BACK)):
        box(x - w - 0.01, x + w + 0.01, y0, y1, CARD_BOTTOM - 0.01, TOP - 0.06, BRIGHT, bevel=0.04)
    box(x - w, x + w, CARD_FRONT - 0.03, CARD_FRONT, CARD_BOTTOM - 0.05, TOP - 0.1, STEEL, bevel=0.01)
    for k in range(3):
        z = CARD_BOTTOM + 0.04 + k * 0.13
        box(x - w + 0.04, x + w - 0.04, CARD_FRONT - 0.04, CARD_FRONT - 0.025, z, z + 0.08, VOID, bevel=0.005)
    box(x + w - 0.08, x + w - 0.03, CARD_FRONT - 0.045, CARD_FRONT - 0.03, TOP - 0.2, TOP - 0.15, LED, bevel=0)
    box(x - 0.08, x + 0.08, 0.05, 0.25, TOP - 0.08, TOP - 0.02, BLACK, bevel=0.015)
    return (x, 0.15, TOP - 0.02)


def build_cables(sockets):
    """Braided power looms arch from each card to the supply, risers drop to
    the board, and spare runs sag in loops along the front."""
    rng = [(-0.04, 0.0), (0.0, 0.04), (0.04, -0.02)]
    for i, (x, y, z) in enumerate(sockets):
        for k, (dx, dz) in enumerate(rng):
            sx = x + dx
            cable([(sx, y, z), (sx, y + 0.02, z + 0.2 + dz), (sx + 0.05, CARD_BACK + 0.05, z + 0.24 + dz),
                   (sx + 0.1, Y1 + 0.06, z), (sx + 0.12, Y1 - 0.06, 0.45)], 0.05, CABLES["black"])
        # The riser from the board up to the card's foot.
        cable([(x, -0.3, CARD_BOTTOM - 0.01), (x, -0.42, CARD_BOTTOM - 0.12), (x * 0.6 - 0.5, -0.4, 0.2),
               (-0.6 - i * 0.05, -0.3, 0.14)], 0.045, CABLES["black"])
    for k in range(5):
        x0 = -1.6 + k * 0.7
        cable([(x0, -0.7, 0.14), (x0 + 0.15, Y0 - 0.12, 0.12), (x0 + 0.35, Y0 - 0.2, 0.05),
               (x0 + 0.55, Y0 - 0.1, 0.12), (x0 + 0.7, -0.7, 0.14)], 0.045, CABLES["black"])
    # Looms snaking across the floor in front of the cards.
    for k in range(4):
        y = -0.72 + k * 0.07
        points = [(-0.9, y, 0.15)]
        for j in range(1, 6):
            points.append((-0.9 + j * 0.5, y + (0.06 if (j + k) % 2 else -0.04), 0.15 + 0.03 * (j % 2)))
        points.append((1.75, y + 0.15, 0.2))
        cable(points, 0.05, CABLES["black"])


def build_horizontal():
    build_frame()
    build_base()
    sockets = [build_card(x) for x in CARD_XS]
    build_cables(sockets)
    return []


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
