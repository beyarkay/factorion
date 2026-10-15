"""Model and render the GPU server sprites in Blender.

    Blender --background --factory-startup --python gpu_server.py -- OUT_DIR [vertical]

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
CARD_TURN = math.radians(-18)

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
PCB = material("pcb", (0.03, 0.26, 0.09), metallic=0.1, roughness=0.4, grime=0.2, rust=0.0, wear=0.0)
CHIP = material("chip", (0.02, 0.02, 0.02), metallic=0.2, roughness=0.35, rust=0.0, wear=0.3, bump=0.0)
GOLD = material("gold", (0.85, 0.62, 0.22), metallic=1.0, roughness=0.3, rust=0.0, wear=0.0, bump=0.0)
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


CARD_W, CARD_T, CARD_H = 0.36, 0.2, 1.25
CARD_XS = (-0.62, 0.5, 1.62)


def card(x, y, z, parent=None):
    """One GPU: finned heat sink in a silver frame, black three-fan shroud on
    the front, copper heat pipes out of the top, power socket at the foot."""
    g = group((x, y, z), (0, 0, CARD_TURN), parent)
    w, t, h = CARD_W, CARD_T, CARD_H
    box(-w, w, -t + 0.03, t, 0.05, h, DARK, bevel=0.03, parent=g)
    for k in range(3):
        yy = -t + 0.07 + k * (2 * t - 0.12) / 2
        box(w - 0.01, w + 0.09, yy - 0.025, yy + 0.025, 0.12, h - 0.06, FIN, bevel=0.012, parent=g)
    for k in range(5):
        xx = -w + 0.07 + k * (2 * w - 0.14) / 4
        box(xx - 0.03, xx + 0.03, -t + 0.05, t - 0.02, h, h + 0.08, FIN, bevel=0.012, parent=g)
    cable([(-0.16, 0.0, h - 0.02), (-0.16, 0.0, h + 0.13), (0.0, 0.0, h + 0.19), (0.16, 0.0, h + 0.13), (0.16, 0.0, h - 0.02)], 0.045, COPPER, parent=g)
    # Heavy frame rails with caps, and a painted edge strip.
    for side in (-1, 1):
        sx = side * (w + 0.03)
        box(sx - 0.05, sx + 0.05, -t - 0.03, -t + 0.08, 0.0, h + 0.08, GUNMETAL, bevel=0.025, parent=g)
        for zz in (0.0, h - 0.04):
            box(sx - 0.065, sx + 0.065, -t - 0.06, -t + 0.1, zz, zz + 0.13, BLACK, bevel=0.025, parent=g)
            rivets([(sx, -t - 0.06, zz + 0.065)], r=0.028, parent=g)
    box(-w + 0.04, w - 0.04, -t - 0.08, -t + 0.01, 0.08, h - 0.03, BLACK, bevel=0.04, parent=g)
    box(-w + 0.04, -w + 0.1, -t - 0.095, -t - 0.06, 0.12, h - 0.08, BLUE, bevel=0.012, parent=g)
    rotors = [fan((0.02, -t - 0.1, zz), 0.27, g) for zz in (0.39, 0.92)]
    box(w - 0.13, w - 0.06, -t - 0.11, -t - 0.07, h - 0.17, h - 0.1, LED, bevel=0, parent=g)
    # Slot bracket at the foot and the socket the power loom plugs into.
    box(-w - 0.07, w + 0.07, -t - 0.08, t + 0.06, -0.08, 0.07, GUNMETAL, bevel=0.025, parent=g)
    box(-w + 0.02, -w + 0.24, -t - 0.17, -t - 0.07, -0.05, 0.11, BLACK, bevel=0.02, parent=g)
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
        rivets([(x, y0 - 0.05, 0.2)], r=0.026)
    for x in (x0, 0.0, x1):
        box(x - 0.13, x + 0.13, y0 - 0.12, y0 + 0.02, 0.0, 0.11, GUNMETAL, bevel=0.015)
        rivets([(x, y0 - 0.12, 0.055)], r=0.028)
    for left, right in zip(posts, posts[1:]):
        a, b = left + 0.06, right - 0.06
        if right in (0.42, x1):
            # Louvres, angled so their lips catch the sky against the dark bay.
            n = 3
            for k in range(n):
                zz = 0.18 + k * 0.06
                slat = box(a, b, -0.04, 0.04, -0.012, 0.012, GUNMETAL, bevel=0.008)
                slat.location = (0, y0 + 0.02, zz)
                slat.rotation_euler = (math.radians(-40), 0, 0)
        else:
            box(a, b, y0 - 0.02, y0 + 0.02, 0.16, top - 0.08, BLACK, bevel=0.012)
            rivets([(a + 0.05, y0 - 0.02, 0.24), (b - 0.05, y0 - 0.02, 0.24)], r=0.024)
            box(a + 0.1, a + 0.3, y0 - 0.04, y0 - 0.015, 0.2, 0.27, BLUE, bevel=0.01)
    # Rails under each card slot along the top.
    for x in CARD_XS:
        box(x - 0.48, x + 0.48, -0.32, 0.22, top, top + 0.05, GUNMETAL, bevel=0.02)


def build_horizontal():
    build_plinth(-1.36, 1.94)

    xs = CARD_XS
    cards = [card(x, -0.05, 0.45) for x in xs]
    bpy.context.view_layer.update()
    rotors = [r for _, rs in cards for r in rs]

    # A loom from the I/O box runs along the deck in front of the cards and
    # peels off into each card's socket.
    colours = ("black", "red", "black")
    for k, colour in enumerate(colours):
        y = -0.66 + k * 0.08
        z = 0.47 + (k % 2) * 0.02
        cable([(-1.4, y, 0.62), (-1.2, y, z + 0.02), (-0.4, y, z), (0.6, y, z), (1.9, y, z + 0.01)], 0.04, CABLES[colour])
    for x in (-1.15, 0.0, 1.1):
        box(x - 0.06, x + 0.06, -0.74, -0.44, 0.42, 0.56, GUNMETAL, bevel=0.02)
    for g, _ in cards:
        socket = world_point(g, (-CARD_W + 0.1, -CARD_T - 0.12, 0.03))
        for k, colour in enumerate(("red", "black")):
            off = Vector((k * 0.05, 0, 0))
            start = Vector((socket.x - 0.12 + k * 0.05, -0.6 + k * 0.04, 0.46))
            cable([start, start + Vector((0.04, 0.02, 0.06)), socket + off + Vector((0, -0.03, 0.02))], 0.035, CABLES[colour])

    # A loom along the back carrying the I/O box's data to the cards.
    for k, colour in enumerate(("black", "green", "black")):
        y = 0.4 + k * 0.08
        cable([(-1.4, y, 0.62), (-0.8, y, 0.47), (0.5, y, 0.46), (1.85, y, 0.49)], 0.038, CABLES[colour])
    for x in (-0.05, 1.05):
        box(x - 0.06, x + 0.06, 0.34, 0.62, 0.42, 0.56, GUNMETAL, bevel=0.02)

    # Coolant manifold on a raised back rail, branching to each card.
    box(-1.36, 1.94, 0.64, 0.78, 0.4, 0.5, GUNMETAL, bevel=0.015)
    for k, yy in enumerate((0.67, 0.74)):
        tube((-1.38, yy, 0.55), (1.96, yy, 0.55), 0.045, COPPER, segments=16)
    for x in (-1.1, 0.0, 1.1, 1.85):
        box(x - 0.07, x + 0.07, 0.61, 0.8, 0.5, 0.63, GUNMETAL, bevel=0.02)
        rivets([(x, 0.705, 0.63)], axis=(0, 0, 1), r=0.026)
    for g, _ in cards:
        port = world_point(g, (0.0, CARD_T, 0.45))
        cable([Vector((port.x + 0.12, 0.67, 0.55)), Vector((port.x + 0.1, 0.45, 0.62)), port + Vector((0, 0.04, 0))], 0.04, COPPER)
        tube(port, port + Vector((0, 0.08, 0)), 0.06, GUNMETAL, segments=12)
    # Voltage regulators and capacitors in the gaps behind the cards.
    for left, right in zip(xs, xs[1:]):
        x = (left + right) / 2
        box(x - 0.14, x + 0.14, 0.1, 0.32, 0.45, 0.52, DARK, bevel=0.02)
        for k in range(3):
            xx = x - 0.09 + k * 0.09
            box(xx - 0.025, xx + 0.025, 0.12, 0.3, 0.52, 0.64, FIN, bevel=0.01)
        tube((x, 0.4, 0.45), (x, 0.4, 0.58), 0.05, BLUE, segments=12)

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
    lid = z1 - 0.03
    box(x0 + 0.05, x1 - 0.05, y0 + 0.05, y1 - 0.05, lid - 0.04, lid, GUNMETAL, bevel=0.02)
    for k in range(5):
        yy = y0 + 0.25 + k * 0.2
        box(x0 + 0.12, x1 - 0.12, yy - 0.05, yy + 0.05, lid, lid + 0.03, BLACK, bevel=0.015)
    rivets([(x, y, lid + 0.01) for x in (x0 + 0.09, x1 - 0.09) for y in (y0 + 0.12, y1 - 0.12)], axis=(0, 0, 1), r=0.026)
    # Front: a drive cage standing proud of the panel, sockets below.
    yf = y0
    box(x0 + 0.06, x1 - 0.06, yf - 0.1, yf + 0.01, 0.46, 0.84, GUNMETAL, bevel=0.015)
    for k in range(2):
        zz = 0.8 - k * 0.17
        box(x0 + 0.1, x1 - 0.1, yf - 0.12, yf - 0.09, zz - 0.14, zz, BLACK, bevel=0.02)
        box(x1 - 0.2, x1 - 0.13, yf - 0.135, yf - 0.115, zz - 0.09, zz - 0.04, LED, bevel=0)
    for x in (-1.81, -1.57):
        box(x - 0.08, x + 0.08, yf - 0.04, yf + 0.01, 0.2, 0.38, GUNMETAL, bevel=0.02)
        box(x - 0.05, x + 0.05, yf - 0.06, yf - 0.03, 0.24, 0.34, VOID, bevel=0.008)
    box(x0 - 0.03, x1 + 0.03, y0 - 0.05, y0 + 0.02, 0.0, 0.1, GUNMETAL, bevel=0.012)
    # Glands where the looms leave for the cards, and the feed from the ground.
    for yy in (-0.6, 0.48):
        box(x1 - 0.02, x1 + 0.06, yy - 0.12, yy + 0.12, 0.5, 0.72, SILVER, bevel=0.012)
    box(x0 - 0.06, x0, -0.35, -0.05, 0.25, 0.6, SILVER, bevel=0.012)
    for k, colour in enumerate(("black", "red", "black")):
        y = -0.3 + k * 0.1
        cable([(x0 - 0.03, y, 0.45), (x0 - 0.15, y, 0.38), (x0 - 0.12, y - 0.1, 0.1), (x0 + 0.1, y - 0.35, 0.03)], 0.045, CABLES[colour])


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
