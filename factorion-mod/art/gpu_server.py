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
CANVAS_TILES = (6, 5)
SUN = Vector((1.0, -0.25, -1.2)).normalized()
# Cards are turned on their bases so the fixed camera sees a fan face and a
# finned edge at once, as in the three-quarter concepts.
CARD_TURN = math.radians(-28)

bpy.ops.wm.read_factory_settings(use_empty=True)


def material(name, colour, metallic=0.6, roughness=0.55, grime=1.0, wear=0.3, rust=0.6, bump=0.25, emission=None):
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

    # Grime in the crevices.
    ao = node("ShaderNodeAmbientOcclusion", Distance=0.12)
    crevice = maths("MULTIPLY", maths("SUBTRACT", 1.0, ao.outputs["AO"]), 2.0 * grime)
    dirty = mix(base, (0.05, 0.04, 0.03), maths("MINIMUM", crevice, 0.85))

    # Worn bright edges, where a bevelled normal departs from the true one.
    bevel = node("ShaderNodeBevel", Radius=0.02)
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
    edge = maths("MINIMUM", maths("ADD", maths("MULTIPLY", maths("SUBTRACT", 1.0, dot.outputs["Value"]), 14.0 * wear), maths("MULTIPLY", chipped, 1.2 * wear)), 1.0)
    worn = mix(dirty, tuple(min(1.0, c * 2.2 + 0.18) for c in colour), edge)
    links.new(worn, bsdf.inputs["Base Color"])

    if bump:
        grain = node("ShaderNodeTexNoise", Scale=60.0, Detail=6.0)
        links.new(coords.outputs["Object"], grain.inputs["Vector"])
        bumper = node("ShaderNodeBump", Strength=bump, Distance=0.02)
        links.new(grain.outputs["Fac"], bumper.inputs["Height"])
        links.new(bevel.outputs["Normal"], bumper.inputs["Normal"])
        links.new(bumper.outputs["Normal"], bsdf.inputs["Normal"])
    else:
        links.new(bevel.outputs["Normal"], bsdf.inputs["Normal"])
    return mat


def stripes(name, a, b, scale):
    mat = bpy.data.materials.new(name)
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexWave")
    wave.bands_direction = "DIAGONAL"
    wave.inputs["Scale"].default_value = scale
    wave.inputs["Distortion"].default_value = 0.0
    links.new(coords.outputs["Object"], wave.inputs["Vector"])
    cut = nodes.new("ShaderNodeValToRGB")
    cut.color_ramp.interpolation = "CONSTANT"
    cut.color_ramp.elements[0].color = (*a, 1)
    cut.color_ramp.elements[1].position = 0.5
    cut.color_ramp.elements[1].color = (*b, 1)
    links.new(wave.outputs["Fac"], cut.inputs["Fac"])
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = 0.7
    links.new(cut.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


GUNMETAL = material("gunmetal", (0.10, 0.10, 0.105), metallic=0.55, roughness=0.62, rust=1.0, wear=0.5)
STEEL = material("steel", (0.27, 0.26, 0.25), metallic=0.7, roughness=0.5, rust=0.8, wear=0.5)
DARK = material("dark", (0.045, 0.045, 0.05), metallic=0.6, roughness=0.5, rust=0.25, wear=0.2)
BRASS = material("brass", (0.55, 0.37, 0.12), metallic=0.95, roughness=0.4, rust=0.3)
COPPER = material("copper", (0.85, 0.27, 0.06), metallic=0.4, roughness=0.4, rust=0.35, wear=0.4)
OLIVE = material("olive", (0.17, 0.21, 0.10), metallic=0.3, roughness=0.6, rust=0.8)
CERAMIC = material("ceramic", (0.82, 0.79, 0.70), metallic=0.0, roughness=0.25, rust=0.1, grime=0.25, wear=0.0, bump=0.05)
PCB = material("pcb", (0.03, 0.16, 0.065), metallic=0.1, roughness=0.4, grime=0.4, rust=0.0, wear=0.0)
CHIP = material("chip", (0.03, 0.03, 0.03), metallic=0.2, roughness=0.35, rust=0.0, wear=0.3, bump=0.0)
GOLD = material("gold", (0.85, 0.62, 0.22), metallic=1.0, roughness=0.3, rust=0.0, bump=0.0)
CABLES = {
    "black": material("cable_black", (0.03, 0.03, 0.03), metallic=0.0, roughness=0.6, rust=0.15, bump=0.6),
    "red": material("cable_red", (0.33, 0.05, 0.035), metallic=0.0, roughness=0.6, rust=0.2, bump=0.6),
    "blue": material("cable_blue", (0.05, 0.10, 0.26), metallic=0.0, roughness=0.6, rust=0.2, bump=0.6),
    "green": material("cable_green", (0.07, 0.17, 0.07), metallic=0.0, roughness=0.6, rust=0.2, bump=0.6),
}
FAN = material("fan", (0.06, 0.06, 0.065), metallic=0.3, roughness=0.5, rust=0.1, wear=0.5)
LED = material("led", (0.05, 0.5, 0.1), emission=((0.25, 1.0, 0.3), 8))
SWITCH = material("switch", (0.5, 0.06, 0.05), metallic=0.1, roughness=0.4, rust=0.0)
HAZARD = stripes("hazard", (0.75, 0.55, 0.06), (0.03, 0.03, 0.03), 9.0)


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


def rivets(points, mat=STEEL, r=0.016, axis=(0, -1, 0), parent=None):
    axis = Vector(axis)
    for p in points:
        p = Vector(p)
        tube(p, p + axis * 0.012, r, mat, segments=8, parent=parent)


def insulator(x, y, z, height, parent=None):
    """A ribbed ceramic standoff with a brass cap."""
    n = 5
    for k in range(n):
        z0 = z + k * height / n
        tube((x, y, z0), (x, y, z0 + height / n * 0.55), 0.075, CERAMIC, parent=parent)
        tube((x, y, z0 + height / n * 0.55), (x, y, z0 + height / n), 0.05, CERAMIC, parent=parent)
    tube((x, y, z + height), (x, y, z + height + 0.04), 0.035, BRASS, parent=parent)
    tube((x, y, z + height + 0.04), (x, y, z + height + 0.09), 0.015, BRASS, parent=parent)


def fan(centre, r, parent):
    """A fan in a ring frame, facing local -Y."""
    hub = group(centre, (math.pi / 2, 0, 0), parent)
    ring((0, 0, -0.02), (0, 0, 0.03), r, 0.025, DARK, parent=hub)
    tube((0, 0, -0.03), (0, 0, -0.01), r - 0.01, CHIP, parent=hub)
    rotor = group((0, 0, 0.0), parent=hub)
    tube((0, 0, -0.01), (0, 0, 0.03), r * 0.3, FAN, parent=rotor)
    tube((0, 0, 0.03), (0, 0, 0.035), r * 0.18, STEEL, parent=rotor)
    for i in range(9):
        blade = box(r * 0.22, r * 0.93, -r * 0.12, r * 0.12, -0.004, 0.004, FAN, bevel=0, parent=rotor)
        blade.rotation_euler = Euler((0.55, 0, i * 2 * math.pi / 9), "ZXY")
    return rotor


def card(x, y, z, parent=None):
    """One GPU: finned heat sink in a steel frame, olive fan shroud on the front."""
    g = group((x, y, z), (0, 0, CARD_TURN), parent)
    w, t, h = 0.3, 0.15, 1.2
    # Heat sink core and its exposed fins on the right edge and top.
    box(-w, w, -t + 0.03, t, 0.05, h, DARK, bevel=0.008, parent=g)
    for k in range(9):
        yy = -t + 0.06 + k * (2 * t - 0.08) / 8
        box(w - 0.01, w + 0.06, yy - 0.008, yy + 0.008, 0.1, h - 0.04, STEEL, bevel=0, parent=g)
    for k in range(14):
        xx = -w + 0.05 + k * (2 * w - 0.1) / 13
        box(xx - 0.007, xx + 0.007, -t + 0.05, t - 0.02, h, h + 0.04, STEEL, bevel=0, parent=g)
    # Steel frame rails with brass corner caps.
    for side in (-1, 1):
        sx = side * (w + 0.02)
        box(sx - 0.03, sx + 0.03, -t - 0.02, -t + 0.06, 0.0, h + 0.06, STEEL, bevel=0.01, parent=g)
        for zz in (0.0, h - 0.02):
            box(sx - 0.04, sx + 0.04, -t - 0.04, -t + 0.07, zz, zz + 0.08, BRASS, bevel=0.01, parent=g)
        rivets([(sx, -t - 0.02, zz) for zz in (0.3, 0.6, 0.9)], parent=g)
    # The shroud and its two fans.
    box(-w + 0.03, w - 0.03, -t - 0.05, -t + 0.01, 0.07, h - 0.02, OLIVE, bevel=0.015, parent=g)
    rotors = [fan((0, -t - 0.07, zz), 0.205, g) for zz in (0.36, 0.83)]
    for zz in (0.12, 0.6, h - 0.08):
        rivets([(-w + 0.07, -t - 0.05, zz), (w - 0.07, -t - 0.05, zz)], r=0.013, parent=g)
    box(w - 0.1, w - 0.06, -t - 0.07, -t - 0.04, 0.1, 0.13, LED, bevel=0, parent=g)
    box(w - 0.1, w - 0.06, -t - 0.07, -t - 0.04, 0.16, 0.19, LED, bevel=0, parent=g)
    # Power connector and clamp on top, where the looms plug in.
    box(-0.16, 0.1, -0.06, 0.08, h + 0.04, h + 0.12, GUNMETAL, bevel=0.01, parent=g)
    box(-0.18, 0.12, -0.08, 0.1, h + 0.12, h + 0.15, STEEL, bevel=0.008, parent=g)
    # Slot bracket at the foot.
    box(-w - 0.05, w + 0.05, -t - 0.06, t + 0.04, -0.06, 0.04, GUNMETAL, bevel=0.012, parent=g)
    box(-0.05, 0.05, -t - 0.1, -t - 0.04, -0.12, 0.08, BRASS, bevel=0.01, parent=g)
    return g, rotors


def world_point(obj, local):
    return obj.matrix_world @ Vector(local)


def build_horizontal():
    # Plinth: skirt with feet, riveted front panels, top plate.
    box(-1.97, 1.97, -0.86, 0.82, 0.0, 0.07, GUNMETAL, bevel=0.015)
    for x in (-1.8, -0.95, 0.0, 0.95, 1.8):
        box(x - 0.12, x + 0.12, -0.95, -0.8, 0.0, 0.1, GUNMETAL, bevel=0.015)
        rivets([(x - 0.06, -0.95, 0.05), (x + 0.06, -0.95, 0.05)])
    box(-1.1, 1.93, -0.8, 0.78, 0.07, 0.36, GUNMETAL, bevel=0.02)
    box(-1.12, 1.95, -0.82, 0.8, 0.33, 0.38, STEEL, bevel=0.012)
    for x in (-1.1, -0.38, 0.38, 1.13, 1.93):
        box(x - 0.035, x + 0.035, -0.84, -0.78, 0.07, 0.35, STEEL, bevel=0.01)
        rivets([(x, -0.84, z) for z in (0.12, 0.2, 0.28)], r=0.012)
    for x0, x1 in ((-1.03, -0.45), (0.45, 1.06)):
        box(x0, x1, -0.81, -0.79, 0.11, 0.31, DARK, bevel=0)
        for k in range(9):
            xx = x0 + 0.04 + k * (x1 - x0 - 0.08) / 8
            box(xx - 0.012, xx + 0.012, -0.83, -0.8, 0.13, 0.29, GUNMETAL, bevel=0.004)
    for x0, x1 in ((-0.31, 0.31), (1.2, 1.86)):
        box(x0, x1, -0.82, -0.79, 0.12, 0.3, GUNMETAL, bevel=0.01)
        rivets([(x0 + 0.03, -0.82, 0.15), (x1 - 0.03, -0.82, 0.15), (x0 + 0.03, -0.82, 0.27), (x1 - 0.03, -0.82, 0.27)], r=0.01)
    box(1.35, 1.9, -0.98, -0.88, 0.0, 0.03, HAZARD, bevel=0.005)

    # Motherboard strip the cards slot into.
    box(-1.05, 1.9, -0.45, 0.45, 0.38, 0.4, PCB, bevel=0.004)
    for x in (-0.38, 0.38, 1.13):
        box(x - 0.08, x + 0.08, -0.3, -0.15, 0.4, 0.43, CHIP, bevel=0.004)
        box(x - 0.05, x + 0.07, 0.1, 0.25, 0.4, 0.425, CHIP, bevel=0.004)

    # Four cards, each with its fans.
    xs = (-0.74, 0.0, 0.74, 1.48)
    cards = [card(x, -0.12, 0.44) for x in xs]
    bpy.context.view_layer.update()
    rotors = [r for _, rs in cards for r in rs]

    # Copper bus bar on ceramic standoffs, clamped to each card's foot.
    box(-1.08, 1.94, -0.72, -0.6, 0.47, 0.6, COPPER, bevel=0.012)
    box(-1.08, 1.94, -0.76, -0.73, 0.4, 0.44, COPPER, bevel=0.006)
    for x in xs:
        box(x - 0.09, x + 0.09, -0.75, -0.57, 0.44, 0.63, BRASS, bevel=0.012)
        rivets([(x - 0.05, -0.75, 0.6), (x + 0.05, -0.75, 0.6)], r=0.014)
    for x in (-0.37, 0.37, 1.11):
        insulator(x, -0.67, 0.38, 0.32)
    box(1.88, 1.98, -0.74, -0.58, 0.4, 0.66, BRASS, bevel=0.012)

    # Braided looms arching from card to card, and down to the bus.
    colours = ("red", "blue", "green", "black")
    anchors = [world_point(g, (-0.03, 0.0, 1.36)) for g, _ in cards]
    for a, b in zip(anchors, anchors[1:]):
        for k, colour in enumerate(colours):
            off = Vector((0, (k - 1.5) * 0.05, 0))
            mid = (a + b) / 2 + Vector((0, 0.05, 0.42)) + off
            cable([a + off, mid, b + off], 0.032, CABLES[colour])
        for end in (a, b):
            tube(end + Vector((0, -0.09, 0.08)), end + Vector((0, 0.09, 0.08)), 0.05, STEEL, segments=12)
    for g, _ in cards:
        foot = world_point(g, (-0.36, -0.05, 0.05))
        for k, colour in enumerate(("black", "red", "black")):
            off = Vector((k * 0.035, 0, 0))
            cable([world_point(g, (-0.36, -0.05 + k * 0.03, 1.2)), world_point(g, (-0.4, -0.1, 0.7)) + off, foot + off, Vector((foot.x + 0.05 * k, -0.62, 0.62))], 0.02, CABLES[colour])
        band = world_point(g, (-0.38, -0.08, 0.75))
        tube(band - Vector((0, 0, 0.03)), band + Vector((0, 0, 0.03)), 0.07, STEEL)

    # A loom along the back carrying the I/O box's feed to the cards.
    for k, colour in enumerate(("black", "red", "blue", "black", "red")):
        y = 0.52 + k * 0.045
        cable([(-1.15, y, 0.62), (-0.6, y, 0.46), (0.5, y, 0.45), (1.8, y, 0.48)], 0.024, CABLES[colour])
    for x in (-0.5, 0.4, 1.3):
        box(x - 0.04, x + 0.04, 0.47, 0.78, 0.38, 0.52, STEEL, bevel=0.01)

    build_io_box()
    return rotors


def build_io_box():
    x0, x1, y0, y1, z0, z1 = -1.93, -1.1, -0.8, 0.72, 0.0, 1.15
    box(x0, x1, y0, y1, z0, z1 - 0.12, GUNMETAL, bevel=0.025)
    for x in (x0, x1):
        for y in (y0, y1):
            box(x - 0.045, x + 0.045, y - 0.045, y + 0.045, z0, z1 - 0.08, STEEL, bevel=0.012)
            box(x - 0.06, x + 0.06, y - 0.06, y + 0.06, z0, z0 + 0.1, BRASS, bevel=0.012)
    # Open top: a rim around a circuit board.
    for rim in ((x0, x1, y0, y0 + 0.06), (x0, x1, y1 - 0.06, y1), (x0, x0 + 0.06, y0, y1), (x1 - 0.06, x1, y0, y1)):
        box(*rim, z1 - 0.14, z1 - 0.1, STEEL, bevel=0.01)
    box(x0 + 0.06, x1 - 0.06, y0 + 0.06, y1 - 0.06, z1 - 0.14, z1 - 0.11, PCB, bevel=0)
    zb = z1 - 0.11
    box(-1.75, -1.55, -0.45, -0.25, zb, zb + 0.04, CHIP, bevel=0.005)
    box(-1.48, -1.3, -0.55, -0.4, zb, zb + 0.035, CHIP, bevel=0.005)
    box(-1.8, -1.62, 0.05, 0.2, zb, zb + 0.03, CHIP, bevel=0.005)
    for k in range(8):
        xx = -1.55 + k * 0.035
        box(xx - 0.01, xx + 0.01, 0.1, 0.4, zb, zb + 0.12, STEEL, bevel=0)
    for p in ((-1.72, -0.12), (-1.65, -0.08), (-1.4, 0.0), (-1.3, -0.15)):
        tube((p[0], p[1], zb), (p[0], p[1], zb + 0.09), 0.035, STEEL, segments=12)
        tube((p[0], p[1], zb + 0.09), (p[0], p[1], zb + 0.1), 0.03, CHIP, segments=12)
    box(-1.85, -1.3, -0.7, -0.62, zb, zb + 0.05, GOLD, bevel=0.003)
    for k in range(5):
        box(-1.82 + k * 0.1, -1.76 + k * 0.1, 0.48, 0.6, zb, zb + 0.06, CHIP, bevel=0.004)
    # Lid hinged open at the back.
    lid = group((0, y1, z1 - 0.1), (math.radians(-72), 0, 0))
    box(x0 - 0.02, x1 + 0.02, -0.02, 0.04, 0.0, 0.75, GUNMETAL, bevel=0.02, parent=lid)
    for k in range(4):
        box(x0 + 0.08, x1 - 0.08, -0.04, -0.02, 0.1 + k * 0.17, 0.14 + k * 0.17, STEEL, bevel=0.008, parent=lid)
    rivets([(x, -0.04, z) for x in (x0 + 0.04, x1 - 0.04) for z in (0.06, 0.69)], parent=lid)
    for x in (x0 + 0.15, x1 - 0.15):
        tube((x - 0.07, y1, z1 - 0.1), (x + 0.07, y1, z1 - 0.1), 0.03, BRASS, segments=12)

    # Front panel: drive bays, a round connector, switches and sockets.
    yf = y0 - 0.01
    for k in range(4):
        zz = 0.95 - k * 0.12
        box(-1.86, -1.5, yf - 0.03, yf + 0.01, zz - 0.09, zz, STEEL, bevel=0.01)
        box(-1.82, -1.62, yf - 0.045, yf - 0.03, zz - 0.06, zz - 0.035, DARK, bevel=0.003)
        box(-1.56, -1.53, yf - 0.05, yf - 0.03, zz - 0.06, zz - 0.035, LED, bevel=0)
    ring((-1.3, yf - 0.03, 0.85), (-1.3, yf + 0.01, 0.85), 0.1, 0.03, BRASS)
    tube((-1.3, yf - 0.01, 0.85), (-1.3, yf + 0.01, 0.85), 0.075, DARK)
    for a in range(5):
        ang = a * 2 * math.pi / 5
        tube((-1.3 + 0.04 * math.cos(ang), yf - 0.025, 0.85 + 0.04 * math.sin(ang)), (-1.3 + 0.04 * math.cos(ang), yf - 0.005, 0.85 + 0.04 * math.sin(ang)), 0.01, GOLD, segments=8)
    for k in range(3):
        xx = -1.4 + k * 0.08
        box(xx - 0.025, xx + 0.025, yf - 0.03, yf + 0.01, 0.56, 0.66, DARK, bevel=0.005)
        box(xx - 0.016, xx + 0.016, yf - 0.06, yf - 0.02, 0.6, 0.65, SWITCH, bevel=0.004)
    for x in (-1.84, -1.68, -1.48, -1.3):
        for zz in (0.38, 0.22):
            box(x - 0.06, x + 0.06, yf - 0.03, yf + 0.01, zz - 0.05, zz + 0.05, STEEL, bevel=0.008)
            box(x - 0.04, x + 0.04, yf - 0.045, yf - 0.03, zz - 0.025, zz + 0.025, CHIP, bevel=0.003)
    box(x0 - 0.03, x1 + 0.03, y0 - 0.05, y0 + 0.02, 0.0, 0.1, STEEL, bevel=0.012)
    box(-1.9, -1.25, -0.97, -0.88, 0.0, 0.03, HAZARD, bevel=0.005)
    # Feed cables out of the side gland, down to the ground.
    box(x0 - 0.06, x0, -0.35, -0.05, 0.25, 0.6, BRASS, bevel=0.012)
    for k, colour in enumerate(("black", "red", "black")):
        y = -0.3 + k * 0.1
        cable([(x0 - 0.03, y, 0.45), (x0 - 0.15, y, 0.38), (x0 - 0.12, y - 0.1, 0.1), (x0 + 0.1, y - 0.35, 0.03)], 0.03, CABLES[colour])


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
