"""Model and render the GPU server sprites in Blender.

    Blender --background --factory-startup --python gpu_server.py -- OUT_DIR

Writes main.png and shadow.png at twice the game's resolution into OUT_DIR,
with the footprint's ground centre at the middle of the canvas. pack.py
turns them into sprites.
"""

import math
import random
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


def material(name, colour, metallic=0.6, roughness=0.55, grime=1.0, wear=0.3, chip=None, edge=None, rust=0.6, bump=0.25, radius=0.02, emission=None):
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

    # Grime in the crevices.
    ao = node("ShaderNodeAmbientOcclusion", Distance=0.12)
    crevice = maths("MULTIPLY", maths("SUBTRACT", 1.0, ao.outputs["AO"]), 2.0 * grime)
    dirty = mix(base, (0.02, 0.013, 0.008), maths("MINIMUM", crevice, 0.9))

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
        grain = node("ShaderNodeTexNoise", Scale=60.0, Detail=6.0)
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
GUNMETAL = material("gunmetal", (0.058, 0.05, 0.042), metallic=0.5, roughness=0.55, rust=0.25, wear=0.5, edge=BARE)
SILVER = material("silver", (0.66, 0.62, 0.55), metallic=1.0, roughness=0.3, rust=0.05, wear=0.2, grime=0.6)
DARK = material("dark", (0.012, 0.012, 0.014), metallic=0.3, roughness=0.6, rust=0.0, wear=0.0, grime=0.4)
VOID = material("void", (0.004, 0.004, 0.005), metallic=0.0, roughness=0.9, rust=0.0, wear=0.0, grime=0.0, bump=0.0)
# Bright fin tips come from the bevel term alone.
FIN = material("fin", (0.03, 0.03, 0.035), metallic=0.9, roughness=0.3, rust=0.1, wear=0.6, chip=0.0, edge=(0.75, 0.75, 0.78), grime=0.5, bump=0.0, radius=0.005)
BLUE = material("blue", (0.025, 0.11, 0.42), metallic=0.6, roughness=0.35, rust=0.0, wear=0.3, edge=BARE, grime=0.5)
COPPER = material("copper", (0.95, 0.42, 0.2), metallic=1.0, roughness=0.28, rust=0.0, wear=0.0, grime=0.4, bump=0.0)
PCB = material("pcb", (0.03, 0.26, 0.09), metallic=0.1, roughness=0.4, grime=0.2, rust=0.0, wear=0.0)
CHIP = material("chip", (0.02, 0.02, 0.02), metallic=0.2, roughness=0.35, rust=0.0, wear=0.3, bump=0.0)
GOLD = material("gold", (0.85, 0.62, 0.22), metallic=1.0, roughness=0.3, rust=0.0, wear=0.0, bump=0.0)
CABLES = {
    "black": material("cable_black", (0.02, 0.02, 0.02), metallic=0.0, roughness=0.5, rust=0.1, bump=0.6),
    "red": material("cable_red", (0.3, 0.035, 0.025), metallic=0.0, roughness=0.5, rust=0.1, bump=0.6),
    "blue": material("cable_blue", (0.03, 0.08, 0.3), metallic=0.0, roughness=0.5, rust=0.1, bump=0.6),
    "green": material("cable_green", (0.04, 0.17, 0.06), metallic=0.0, roughness=0.5, rust=0.1, bump=0.6),
}
BLADE = material("blade", (0.03, 0.03, 0.033), metallic=0.6, roughness=0.4, rust=0.0, wear=0.15, chip=0.0, edge=BARE, bump=0.0)
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
    ring((0, 0, -0.02), (0, 0, 0.035), r, 0.022, SILVER, parent=hub)
    tube((0, 0, -0.03), (0, 0, -0.01), r - 0.01, VOID, parent=hub)
    rotor = group((0, 0, 0.0), parent=hub)
    tube((0, 0, -0.01), (0, 0, 0.03), r * 0.3, DARK, parent=rotor)
    tube((0, 0, 0.03), (0, 0, 0.036), r * 0.2, SILVER, parent=rotor)
    for i in range(9):
        blade = box(r * 0.22, r * 0.93, -r * 0.12, r * 0.12, -0.004, 0.004, BLADE, bevel=0, parent=rotor)
        blade.rotation_euler = Euler((0.55, 0, i * 2 * math.pi / 9), "ZXY")
    return rotor


CARD_W, CARD_T, CARD_H = 0.24, 0.12, 1.5


def card(x, y, z, parent=None):
    """One GPU: finned heat sink in a silver frame, black three-fan shroud on
    the front, copper heat pipes out of the top, power socket at the foot."""
    g = group((x, y, z), (0, 0, CARD_TURN), parent)
    w, t, h = CARD_W, CARD_T, CARD_H
    box(-w, w, -t + 0.03, t, 0.05, h, DARK, bevel=0.008, parent=g)
    for k in range(7):
        yy = -t + 0.05 + k * (2 * t - 0.07) / 6
        box(w - 0.01, w + 0.075, yy - 0.009, yy + 0.009, 0.1, h - 0.04, FIN, bevel=0.003, parent=g)
    for k in range(12):
        xx = -w + 0.04 + k * (2 * w - 0.08) / 11
        box(xx - 0.008, xx + 0.008, -t + 0.04, t - 0.01, h, h + 0.05, FIN, bevel=0.003, parent=g)
    for k, (xa, xb) in enumerate(((-0.17, 0.05), (-0.07, 0.15))):
        yy = -0.03 + k * 0.07
        cable([(xa, yy, h - 0.02), (xa, yy, h + 0.09), ((xa + xb) / 2, yy, h + 0.14), (xb, yy, h + 0.09), (xb, yy, h - 0.02)], 0.022, COPPER, parent=g)
    # Silver frame rails with dark caps and a blue edge strip.
    for side in (-1, 1):
        sx = side * (w + 0.02)
        box(sx - 0.028, sx + 0.028, -t - 0.02, -t + 0.06, 0.0, h + 0.06, SILVER, bevel=0.01, parent=g)
        for zz in (0.0, h - 0.02):
            box(sx - 0.038, sx + 0.038, -t - 0.035, -t + 0.07, zz, zz + 0.08, GUNMETAL, bevel=0.01, parent=g)
            rivets([(sx, -t - 0.035, zz + 0.04)], r=0.012, parent=g)
    box(-w + 0.03, w - 0.03, -t - 0.05, -t + 0.01, 0.07, h - 0.02, BLACK, bevel=0.015, parent=g)
    box(-w + 0.03, -w + 0.05, -t - 0.06, -t - 0.04, 0.1, h - 0.06, BLUE, bevel=0.004, parent=g)
    rotors = [fan((0.005, -t - 0.07, zz), 0.185, g) for zz in (0.3, 0.72, 1.14)]
    for zz in (0.51, 0.93):
        rivets([(-w + 0.07, -t - 0.05, zz), (w - 0.07, -t - 0.05, zz)], r=0.012, parent=g)
    for zz in (1.38, 1.43):
        box(w - 0.09, w - 0.05, -t - 0.07, -t - 0.04, zz - 0.015, zz + 0.015, LED, bevel=0, parent=g)
    # Slot bracket at the foot and the socket the power loom plugs into.
    box(-w - 0.05, w + 0.05, -t - 0.06, t + 0.04, -0.06, 0.05, GUNMETAL, bevel=0.012, parent=g)
    box(-w + 0.01, -w + 0.19, -t - 0.13, -t - 0.05, -0.04, 0.09, BLACK, bevel=0.01, parent=g)
    box(-w, -w + 0.2, -t - 0.15, -t - 0.12, 0.0, 0.05, SILVER, bevel=0.006, parent=g)
    return g, rotors


def world_point(obj, local):
    return obj.matrix_world @ Vector(local)


def build_plinth(x0, x1):
    y0, y1, top = -0.8, 0.78, 0.4
    box(x0 - 0.03, x1 + 0.03, y0 - 0.04, y1 + 0.04, 0.0, 0.08, BLACK, bevel=0.015)
    # The body sits back from a frame, so the open bays read as recesses.
    box(x0, x1, y0 + 0.06, y1, 0.08, top - 0.04, VOID, bevel=0)
    box(x0, x1, y0 - 0.02, y1, top - 0.06, top, GUNMETAL, bevel=0.018)
    box(x0, x1, y0 - 0.03, y0 + 0.04, 0.08, 0.14, GUNMETAL, bevel=0.012)
    posts = (x0, -0.42, 0.42, 1.24, x1)
    for x in posts:
        box(x - 0.045, x + 0.045, y0 - 0.05, y0 + 0.06, 0.06, top - 0.02, GUNMETAL, bevel=0.012)
        rivets([(x, y0 - 0.05, z) for z in (0.13, 0.26)], r=0.013)
    for x in (x0, 0.0, x1):
        box(x - 0.13, x + 0.13, y0 - 0.12, y0 + 0.02, 0.0, 0.11, GUNMETAL, bevel=0.015)
        rivets([(x - 0.08, y0 - 0.12, 0.055), (x + 0.08, y0 - 0.12, 0.055)], r=0.014)
    for left, right in zip(posts, posts[1:]):
        a, b = left + 0.06, right - 0.06
        if right in (0.42, x1):
            # Louvres, angled so their lips catch the sky against the dark bay.
            n = 5
            for k in range(n):
                zz = 0.17 + k * 0.035
                slat = box(a, b, -0.026, 0.026, -0.005, 0.005, GUNMETAL, bevel=0.002)
                slat.location = (0, y0 + 0.02, zz)
                slat.rotation_euler = (math.radians(-40), 0, 0)
        else:
            box(a, b, y0 - 0.02, y0 + 0.02, 0.16, top - 0.08, BLACK, bevel=0.012)
            rivets([(a + 0.03, y0 - 0.02, z) for z in (0.19, 0.29)] + [(b - 0.03, y0 - 0.02, z) for z in (0.19, 0.29)], r=0.011)
            box(a + 0.08, a + 0.22, y0 - 0.035, y0 - 0.015, 0.21, 0.26, BLUE, bevel=0.004)
    # Rails under each card slot along the top.
    for x in (-0.78, 0.03, 0.84, 1.65):
        box(x - 0.3, x + 0.3, -0.25, 0.15, top, top + 0.03, SILVER, bevel=0.008)


def build_horizontal():
    build_plinth(-1.36, 1.94)

    xs = (-0.78, 0.03, 0.84, 1.65)
    cards = [card(x, -0.05, 0.43) for x in xs]
    bpy.context.view_layer.update()
    rotors = [r for _, rs in cards for r in rs]

    # A loom from the I/O box runs along the deck in front of the cards and
    # peels off into each card's socket.
    colours = ("black", "red", "black", "blue", "black")
    for k, colour in enumerate(colours):
        y = -0.68 + k * 0.045
        z = 0.44 + (k % 2) * 0.012
        cable([(-1.4, y, 0.62), (-1.2, y, z + 0.02), (-0.4, y, z), (0.6, y, z), (1.9, y, z + 0.01)], 0.026, CABLES[colour])
    for x in (-1.08, -0.2, 0.62, 1.45):
        box(x - 0.035, x + 0.035, -0.72, -0.44, 0.4, 0.5, SILVER, bevel=0.01)
    for g, _ in cards:
        socket = world_point(g, (-CARD_W + 0.1, -CARD_T - 0.12, 0.03))
        for k, colour in enumerate(("red", "black")):
            off = Vector((k * 0.05, 0, 0))
            start = Vector((socket.x - 0.12 + k * 0.05, -0.6 + k * 0.04, 0.46))
            cable([start, start + Vector((0.04, 0.02, 0.06)), socket + off + Vector((0, -0.03, 0.02))], 0.022, CABLES[colour])

    # A loom along the back carrying the I/O box's data to the cards.
    for k, colour in enumerate(("black", "green", "black", "red")):
        y = 0.4 + k * 0.05
        cable([(-1.4, y, 0.62), (-0.8, y, 0.45), (0.5, y, 0.44), (1.85, y, 0.47)], 0.024, CABLES[colour])
    for x in (-0.4, 0.45, 1.25):
        box(x - 0.035, x + 0.035, 0.36, 0.6, 0.4, 0.5, SILVER, bevel=0.01)

    # Coolant manifold on a raised back rail, branching to each card.
    box(-1.36, 1.94, 0.64, 0.78, 0.4, 0.5, GUNMETAL, bevel=0.015)
    for k, yy in enumerate((0.67, 0.74)):
        tube((-1.38, yy, 0.54), (1.96, yy, 0.54), 0.03, COPPER, segments=16)
    for x in (-1.1, -0.38, 0.43, 1.24, 1.85):
        box(x - 0.04, x + 0.04, 0.62, 0.79, 0.5, 0.6, SILVER, bevel=0.01)
        rivets([(x, 0.705, 0.6)], axis=(0, 0, 1), r=0.014)
    for g, _ in cards:
        port = world_point(g, (0.0, CARD_T, 0.45))
        cable([Vector((port.x + 0.12, 0.67, 0.54)), Vector((port.x + 0.1, 0.45, 0.6)), port + Vector((0, 0.04, 0))], 0.022, COPPER)
        tube(port, port + Vector((0, 0.06, 0)), 0.04, SILVER, segments=12)
    # Voltage regulators and capacitors in the gaps behind the cards.
    for left, right in zip(xs, xs[1:]):
        x = (left + right) / 2
        box(x - 0.1, x + 0.1, 0.12, 0.3, 0.43, 0.47, DARK, bevel=0.006)
        for k in range(6):
            xx = x - 0.08 + k * 0.032
            box(xx - 0.008, xx + 0.008, 0.13, 0.29, 0.47, 0.56, FIN, bevel=0.002)
        for dx in (-0.06, 0.06):
            tube((x + dx, 0.36, 0.43), (x + dx, 0.36, 0.52), 0.032, BLUE, segments=12)
            tube((x + dx, 0.36, 0.52), (x + dx, 0.36, 0.53), 0.028, SILVER, segments=12)

    build_io_box()
    return rotors


def traces(x0, x1, y0, y1, z, rng):
    """Copper traces in parallel runs with a right-angle turn, and their pads."""
    width, pitch = 0.016, 0.034
    for _ in range(9):
        n = rng.randint(2, 4)
        sx, sy = rng.uniform(x0, x1 - n * pitch), rng.uniform(y0, y1)
        run = rng.uniform(0.12, 0.4) * rng.choice((-1, 1))
        turn = rng.uniform(0.08, 0.25) * rng.choice((-1, 1))
        for k in range(n):
            ax = sx + k * pitch
            by = min(max(sy + run + k * pitch * (1 if turn > 0 else -1), y0), y1)
            bx = min(max(ax + turn, x0), x1)
            box(ax - width / 2, ax + width / 2, min(sy, by), max(sy, by), z, z + 0.004, COPPER, bevel=0)
            box(min(ax, bx), max(ax, bx), by - width / 2, by + width / 2, z, z + 0.004, COPPER, bevel=0)
            box(bx - 0.018, bx + 0.018, by - 0.018, by + 0.018, z, z + 0.005, GOLD, bevel=0)


def build_io_box():
    x0, x1, y0, y1, z1 = -1.95, -1.4, -0.78, 0.7, 0.95
    box(x0, x1, y0, y1, 0.0, z1 - 0.1, BLACK, bevel=0.025)
    for x in (x0, x1):
        for y in (y0, y1):
            box(x - 0.045, x + 0.045, y - 0.045, y + 0.045, 0.0, z1 + 0.04, GUNMETAL, bevel=0.014)
            tube((x, y, z1 + 0.04), (x, y, z1 + 0.055), 0.025, SILVER, segments=10)
    # The board sits down in a tray, so the rim's inner faces show its depth.
    for rim in ((x0, x1, y0, y0 + 0.05), (x0, x1, y1 - 0.05, y1), (x0, x0 + 0.05, y0, y1), (x1 - 0.05, x1, y0, y1)):
        box(*rim, z1 - 0.06, z1 + 0.02, GUNMETAL, bevel=0.01)
    zb = z1 - 0.05
    box(x0 + 0.05, x1 - 0.05, y0 + 0.05, y1 - 0.05, zb - 0.02, zb, PCB, bevel=0)
    traces(x0 + 0.08, x1 - 0.08, y0 + 0.1, y1 - 0.1, zb, random.Random(7))
    box(-1.86, -1.68, -0.42, -0.24, zb, zb + 0.035, CHIP, bevel=0.005)
    box(-1.62, -1.48, -0.6, -0.46, zb, zb + 0.03, CHIP, bevel=0.005)
    box(-1.84, -1.7, 0.06, 0.2, zb, zb + 0.03, CHIP, bevel=0.005)
    for k in range(7):
        xx = -1.64 + k * 0.03
        box(xx - 0.009, xx + 0.009, 0.08, 0.36, zb, zb + 0.1, FIN, bevel=0.002)
    for p in ((-1.82, -0.1), (-1.74, -0.04), (-1.52, -0.12)):
        tube((p[0], p[1], zb), (p[0], p[1], zb + 0.08), 0.03, BLUE, segments=12)
        tube((p[0], p[1], zb + 0.08), (p[0], p[1], zb + 0.09), 0.026, SILVER, segments=12)
    for k in range(4):
        box(-1.86 + k * 0.1, -1.8 + k * 0.1, 0.48, 0.6, zb, zb + 0.05, DARK, bevel=0.004)

    # Front: a drive cage standing proud of the panel, sockets below.
    yf = y0
    box(x0 + 0.06, x1 - 0.06, yf - 0.1, yf + 0.01, 0.46, 0.84, GUNMETAL, bevel=0.015)
    for k in range(3):
        zz = 0.8 - k * 0.11
        box(x0 + 0.1, x1 - 0.1, yf - 0.115, yf - 0.09, zz - 0.085, zz, BLACK, bevel=0.008)
        box(x0 + 0.13, x1 - 0.2, yf - 0.125, yf - 0.11, zz - 0.055, zz - 0.035, VOID, bevel=0)
        box(x1 - 0.17, x1 - 0.13, yf - 0.13, yf - 0.11, zz - 0.06, zz - 0.03, LED, bevel=0)
    box(x0 + 0.06, x1 - 0.06, yf - 0.12, yf - 0.09, 0.44, 0.47, SILVER, bevel=0.006)
    ring((-1.81, yf - 0.03, 0.33), (-1.81, yf + 0.01, 0.33), 0.075, 0.025, SILVER)
    tube((-1.81, yf - 0.01, 0.33), (-1.81, yf + 0.01, 0.33), 0.055, BLUE)
    for a in range(5):
        ang = a * 2 * math.pi / 5
        p = (-1.81 + 0.03 * math.cos(ang), 0.33 + 0.03 * math.sin(ang))
        tube((p[0], yf - 0.025, p[1]), (p[0], yf - 0.005, p[1]), 0.008, GOLD, segments=8)
    for x in (-1.63, -1.5):
        for zz in (0.36, 0.22):
            box(x - 0.05, x + 0.05, yf - 0.03, yf + 0.01, zz - 0.045, zz + 0.045, SILVER, bevel=0.008)
            box(x - 0.033, x + 0.033, yf - 0.045, yf - 0.03, zz - 0.022, zz + 0.022, VOID, bevel=0.003)
    box(x0 - 0.03, x1 + 0.03, y0 - 0.05, y0 + 0.02, 0.0, 0.1, GUNMETAL, bevel=0.012)
    # Glands where the looms leave for the cards, and the feed from the ground.
    for yy in (-0.6, 0.48):
        box(x1 - 0.02, x1 + 0.06, yy - 0.12, yy + 0.12, 0.5, 0.72, SILVER, bevel=0.012)
    box(x0 - 0.06, x0, -0.35, -0.05, 0.25, 0.6, SILVER, bevel=0.012)
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
