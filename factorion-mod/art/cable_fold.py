"""Procedural Cable fold model. No image textures or generated artwork.

Blender --background --factory-startup --python cable_fold.py -- OUT [both|horizontal|vertical] [--draft]

One physical 4x2 assembly, rendered at 0 and 90 degrees with camera/sun fixed.
Camera pixel aspect corrects ground foreshortening, never model dimensions.
Writes editable .blend, body, shadow, six light states and anchor metadata.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Matrix, Vector

ELEVATION = math.radians(38)
SUN = Vector((1.0, -0.45, -1.65)).normalized()
TILE, CANVAS = 128, (8, 9)
HEIGHT = 0.82
LAMPS, MARKERS = [], {}


def world(p):
    return Vector((p[0], p[1], p[2] * HEIGHT))


def screen(p):
    return [p[0], -p[1] - p[2] / math.tan(ELEVATION)]


def material(name, colour, metal=0, rough=0.6, grain=0.12):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Metallic"].default_value = metal
    bsdf.inputs["Roughness"].default_value = rough
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 48
    noise.inputs["Detail"].default_value = 2
    ramp = nodes.new("ShaderNodeValToRGB")
    for el, fac in zip(ramp.color_ramp.elements, (1 - grain, 1 + grain)):
        el.color = (*(c * fac for c in colour), 1)
    links.new(noise.outputs["Fac"], ramp.inputs[0])
    ao = nodes.new("ShaderNodeAmbientOcclusion")
    ao.inputs["Distance"].default_value = 0.18
    mix = nodes.new("ShaderNodeMixRGB")
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 0.55
    links.new(ramp.outputs[0], mix.inputs[1])
    links.new(ao.outputs["Color"], mix.inputs[2])
    links.new(mix.outputs[0], bsdf.inputs["Base Color"])
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.12
    bump.inputs["Distance"].default_value = 0.013
    links.new(noise.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs[0], bsdf.inputs["Normal"])
    return mat


def emission(name, colour):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeEmission")
    shader.inputs["Color"].default_value = (*colour, 1)
    mat.node_tree.links.new(shader.outputs[0], out.inputs[0])
    return mat


def finish(obj, name, mat, bevel=0.015):
    obj.name = name
    obj.data.materials.append(mat)
    if bevel:
        mod = obj.modifiers.new("Machined edges", "BEVEL")
        mod.width, mod.segments = bevel, 2
        mod = obj.modifiers.new("Face normals", "WEIGHTED_NORMAL")
        mod.keep_sharp = True
    return obj


def cube(name, centre, size, mat, bevel=0.015, transform=None):
    bpy.ops.mesh.primitive_cube_add(size=1)
    obj = bpy.context.object
    obj.dimensions = size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.location = centre
    if transform is not None:
        obj.matrix_world = transform @ obj.matrix_world
    return finish(obj, name, mat, bevel)


def box(name, centre, size, mat, bevel=0.015):
    return cube(
        name,
        world(centre),
        (size[0], size[1], size[2] * HEIGHT),
        mat,
        bevel,
    )


def cylinder(name, p, radius, depth, mat, axis=(0, 0, 1), vertices=16):
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=vertices, radius=radius, depth=depth, location=p
    )
    obj = bpy.context.object
    obj.rotation_euler = Vector(axis).to_track_quat("Z", "Y").to_euler()
    return finish(obj, name, mat, 0.005)


def bolt(p, axis=(0, 0, 1), radius=0.043):
    p, axis = Vector(p), Vector(axis)
    cylinder("Dark bolt washer", p, radius * 1.35, 0.014, M["dark"], axis)
    cylinder("Hex steel fastener", p + axis * 0.018, radius, 0.032, M["steel"], axis, 6)
    cylinder("Hex recess", p + axis * 0.036, radius * 0.43, 0.003, M["black"], axis, 6)


def pipe(name, points, radius, mat):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions, curve.bevel_depth, curve.bevel_resolution = "3D", radius, 4
    curve.use_fill_caps = True
    spline = curve.splines.new("POLY")
    spline.points.add(len(points) - 1)
    for p, v in zip(spline.points, points):
        p.co = (*v, 1)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    curve.materials.append(mat)


def samples(points):
    pts = [world(p) for p in points]
    ext, result = [pts[0]] + pts + [pts[-1]], []
    for i in range(1, len(ext) - 2):
        a, b, c, d = ext[i - 1 : i + 3]
        for k in range(14):
            t = k / 14
            result.append(
                0.5
                * (
                    (2 * b)
                    + (-a + c) * t
                    + (2 * a - 5 * b + 4 * c - d) * t * t
                    + (-a + 3 * b - 3 * c + d) * t * t * t
                )
            )
    return result + [pts[-1]]


def collar(p, tangent, supported=False):
    rot = tangent.to_track_quat("Y", "Z").to_matrix().to_4x4()
    transform = Matrix.Translation(p) @ rot

    def ring(w, h, c):
        return [
            (-w / 2 + c, -h / 2),
            (w / 2 - c, -h / 2),
            (w / 2, -h / 2 + c),
            (w / 2, h / 2 - c),
            (w / 2 - c, h / 2),
            (-w / 2 + c, h / 2),
            (-w / 2, h / 2 - c),
            (-w / 2, -h / 2 + c),
        ]

    outer, inner = ring(0.63, 0.31, 0.055), ring(0.53, 0.22, 0.035)
    verts = [
        (x, y, z)
        for y in (-0.105, 0.105)
        for outline in (outer, inner)
        for x, z in outline
    ]
    faces = []
    for i in range(8):
        j = (i + 1) % 8
        faces += [
            (i, j, 16 + j, 16 + i),
            (8 + i, 24 + i, 24 + j, 8 + j),
            (i, 8 + i, 8 + j, j),
            (16 + i, 16 + j, 24 + j, 24 + i),
        ]
    mesh = bpy.data.meshes.new("Hollow copper compression sleeve")
    mesh.from_pydata(verts, [], faces)
    obj = bpy.data.objects.new("Copper binding", mesh)
    bpy.context.collection.objects.link(obj)
    obj.matrix_world = transform
    finish(obj, obj.name, M["copper"], 0.012)
    for side in (-1, 1):
        cube(
            "Clamp ear",
            (side * 0.34, 0, -0.015),
            (0.09, 0.18, 0.14),
            M["steel"],
            0.012,
            transform,
        )
        bolt(
            transform @ Vector((side * 0.34, 0, 0.065)),
            rot.to_3x3() @ Vector((0, 0, 1)),
            0.028,
        )
        if supported:
            top = transform @ Vector((side * 0.34, 0, -0.08))
            bottom = Vector((top.x, top.y, 0.28 * HEIGHT))
            axis = top - bottom
            cylinder(
                "Cable saddle support",
                (top + bottom) / 2,
                0.032,
                axis.length,
                M["steel"],
                axis.normalized(),
            )
            cube("Saddle mounting foot", bottom, (0.14, 0.15, 0.055), M["dark"])
            bolt(bottom + Vector((0, 0, 0.035)), radius=0.027)


def loom(name, points, bindings, supports=()):
    path = samples(points)
    for lane in (-1, 0, 1):
        strand = []
        for i, p in enumerate(path):
            tangent = (
                path[min(i + 1, len(path) - 1)] - path[max(0, i - 1)]
            ).normalized()
            side = tangent.cross(Vector((0, 0, 1)))
            side = side.normalized() if side.length > 0.01 else Vector((1, 0, 0))
            # Fan in to actual three-pin plugs at both ends, rather than
            # leaving nine full-diameter cut hoses hovering over the sockets.
            end = min(i, len(path) - 1 - i) / (len(path) - 1)
            fan = min(1, end / 0.12)
            strand.append(p + side * (lane * (0.048 + 0.105 * fan)))
        # Variable-radius hose sweep, with narrower strain relief at plugs.
        curve = bpy.data.curves.new(name + " conductor", "CURVE")
        curve.dimensions, curve.bevel_depth, curve.bevel_resolution = "3D", 0.085, 4
        curve.use_fill_caps = True
        spline = curve.splines.new("POLY")
        spline.points.add(len(strand) - 1)
        for i, (point, position) in enumerate(zip(spline.points, strand)):
            point.co = (*position, 1)
            end = min(i, len(path) - 1 - i) / (len(path) - 1)
            point.radius = 0.48 + 0.52 * min(1, end / 0.10)
        obj = bpy.data.objects.new(name + f" conductor {lane + 2}", curve)
        bpy.context.collection.objects.link(obj)
        curve.materials.append(M["rubber"])
        for index, axis in ((0, (0, 0, 1)), (-1, (1, 0, 0))):
            cylinder("Cable gland", strand[index], 0.052, 0.075, M["copper"], axis)
    for t in bindings:
        i = min(len(path) - 2, max(1, round(t * (len(path) - 1))))
        collar(path[i], (path[i + 1] - path[i - 1]).normalized(), t in supports)


def frame(width, depth):
    # Full footprint with only a .04-tile margin on each side.
    w, d = width - 0.08, depth - 0.08
    for y in (-d / 2 + 0.075, d / 2 - 0.075):
        for z in (0.12, 0.33):
            box("Open steel perimeter rail", (0, y, z), (w, 0.14, 0.09), M["steel"])
        box("Recessed extrusion web", (0, y, 0.23), (w - 0.1, 0.06, 0.17), M["dark"])
        for x in (-w / 2 + 0.16, 0, w / 2 - 0.16):
            box("Bolted rail splice", (x, y, 0.26), (0.18, 0.19, 0.31), M["steel"])
            bolt(world((x, y, 0.43)))
    for x in (-w / 2 + 0.075, w / 2 - 0.075):
        for z in (0.12, 0.33):
            box("Side extrusion", (x, 0, z), (0.14, d, 0.09), M["steel"])
        for y in (-d / 2 + 0.14, d / 2 - 0.14):
            box("Corner foot", (x, y, 0.105), (0.24, 0.24, 0.20), M["dark"], 0.035)
            box("Corner cap", (x, y, 0.24), (0.21, 0.24, 0.09), M["steel"])
            bolt(world((x, y, 0.30)))
    for y in (-d * 0.24, d * 0.24):
        box("Chassis crossmember", (0, y, 0.18), (w - 0.18, 0.12, 0.12), M["dark"])
    box(
        "Recessed equipment pan",
        (0, 0, 0.24),
        (w - 0.3, d - 0.28, 0.05),
        M["pan"],
        0.01,
    )


def terminals(x, y):
    box("Three-channel terminal cradle", (x, y, 0.40), (0.85, 0.86, 0.24), M["dark"])
    for k in range(3):
        tx = x + (k - 1) * 0.25
        box("Olive terminal module", (tx, y, 0.55), (0.22, 0.76, 0.24), M["pcb"])
        box(
            "Copper cable termination",
            (tx, y + 0.15, 0.71),
            (0.19, 0.24, 0.17),
            M["copper"],
        )
        bolt(world((tx, y + 0.15, 0.805)), radius=0.033)
        box(
            "Terminal front vent",
            (tx, y - 0.39, 0.53),
            (0.14, 0.015, 0.11),
            M["black"],
            0.004,
        )


def processor(x, y):
    box("Processor chassis", (x, y, 0.37), (0.83, 0.82, 0.23), M["dark"])
    for z in (0.30, 0.37, 0.44):
        for side in (-1, 1):
            box(
                "Processor side cooling rib",
                (x + side * 0.405, y, z),
                (0.055, 0.72, 0.025),
                M["fin"],
                0.005,
            )
    box("Host board tray", (x, y, 0.49), (0.87, 0.87, 0.08), M["dark"])
    box("Exposed olive PCB", (x, y, 0.55), (0.79, 0.79, 0.045), M["pcb"], 0.01)
    box("Processor socket", (x, y + 0.065, 0.61), (0.48, 0.46, 0.09), M["black"])
    box(
        "Processor retention frame",
        (x, y + 0.065, 0.67),
        (0.45, 0.43, 0.06),
        M["steel"],
    )
    box(
        "Large visible processor", (x, y + 0.065, 0.715), (0.35, 0.34, 0.055), M["chip"]
    )
    for sx in (-1, 1):
        for sy in (-1, 1):
            bolt(world((x + sx * 0.205, y + 0.065 + sy * 0.195, 0.711)), radius=0.015)
    for k in range(5):
        xx = x + (k - 2) * 0.115
        box(
            "Indicator bezel",
            (xx, y - 0.285, 0.592),
            (0.09, 0.105, 0.055),
            M["black"],
            0.008,
        )
        lamp = box(
            "Amber utilisation lamp " + str(k + 1),
            (xx, y - 0.285, 0.626),
            (0.065, 0.075, 0.025),
            M["amber_off"],
            0.007,
        )
        LAMPS.append((lamp, k + 1, M["amber_on"]))
    for side in (-1, 1):
        for k in range(4):
            box(
                "PCB power component",
                (x + side * 0.325, y - 0.03 + k * 0.105, 0.59),
                (0.085, 0.06, 0.04),
                M["chip"],
                0.003,
            )
    MARKERS["activity_light"] = world((x, y - 0.2, 0.68))


def cooler(x, y, width=0.70, depth=0.9):
    box("Radiator housing", (x, y, 0.51), (width, depth, 0.46), M["dark"], 0.035)
    box(
        "Black cooling channels",
        (x, y + 0.04, 0.80),
        (width - 0.1, depth - 0.15, 0.32),
        M["black"],
    )
    for k in range(9):
        xx = x - (width - 0.15) / 2 + k * (width - 0.15) / 8
        box(
            "Aluminium cooling fin",
            (xx, y + 0.04, 0.83),
            (0.024, depth - 0.15, 0.31),
            M["fin"],
            0.004,
        )
    for yy in (-depth / 2 + 0.05, depth / 2 - 0.05):
        box(
            "Radiator retaining bar",
            (x, y + yy, 0.88),
            (width, 0.075, 0.09),
            M["steel"],
        )
        for side in (-1, 1):
            bolt(world((x + side * (width / 2 - 0.065), y + yy, 0.94)), radius=0.027)
    for k in range(3):
        box(
            "Cooling intake grille",
            (x, y - depth / 2 - 0.012, 0.40 + k * 0.085),
            (width - 0.16, 0.025, 0.035),
            M["black"],
            0.004,
        )


def status_bar(x, y):
    # Front-facing repeat of the PCB indicators: cable folds must never hide
    # the activity display when machines are placed in a touching grid.
    box("Front utilisation display", (x, y, 0.44), (0.69, 0.13, 0.15), M["dark"])
    for k in range(5):
        xx = x + (k - 2) * 0.115
        box(
            "Status bezel",
            (xx, y - 0.075, 0.45),
            (0.09, 0.025, 0.095),
            M["black"],
            0.006,
        )
        lamp = box(
            "Front amber utilisation " + str(k + 1),
            (xx, y - 0.092, 0.45),
            (0.065, 0.014, 0.066),
            M["amber_off"],
            0.005,
        )
        LAMPS.append((lamp, k + 1, M["amber_on"]))
    MARKERS["activity_light"] = world((x, y - 0.1, 0.45))


def interface(x, y):
    box("Interface enclosure", (x, y, 0.40), (0.72, 0.36, 0.38), M["dark"], 0.025)
    for sx in (-0.12, 0.12):
        box(
            "Front service cover",
            (x + sx, y - 0.195, 0.43),
            (0.17, 0.025, 0.28),
            M["chip"],
        )
    px = x - 0.27
    box("Data socket rim", (px, y - 0.22, 0.43), (0.15, 0.055, 0.17), M["steel"])
    box(
        "Blue data socket",
        (px, y - 0.253, 0.43),
        (0.105, 0.018, 0.115),
        M["blue"],
        0.006,
    )
    box(
        "Socket interior", (px, y - 0.266, 0.43), (0.07, 0.012, 0.07), M["black"], 0.003
    )
    pilot = box(
        "Blue power pilot",
        (px, y - 0.268, 0.52),
        (0.08, 0.016, 0.025),
        M["blue_off"],
        0.003,
    )
    LAMPS.append((pilot, 0, M["blue_on"]))
    MARKERS["data_port_front"] = world((px, y - 0.28, 0.43))
    for i, colour in enumerate(("red", "green")):
        p = world((x + 0.08 + i * 0.16, y + 0.04, 0.68))
        cylinder("Circuit insulator " + colour, p, 0.045, 0.16, M[colour])
        cylinder(
            "Brass circuit pin " + colour,
            p + Vector((0, 0, 0.105)),
            0.026,
            0.05,
            M["copper"],
        )
        MARKERS[colour] = p + Vector((0, 0, 0.13))


def build():
    frame(4, 2)
    terminals(-1.35, -0.05)
    processor(0.20, -0.14)
    cooler(1.44, 0.03, 0.73, 1.25)
    interface(-1.35, -0.66)
    status_bar(0.20, -0.85)
    MARKERS["activity_light_front"] = MARKERS.pop("activity_light")
    # Physical end-face display and data socket, visible after a quarter turn.
    box("End interface cabinet", (-1.78, -0.12, 0.44), (0.23, 1.40, 0.34), M["dark"])
    for k in range(5):
        y = -0.12 + (k - 2) * 0.14
        box(
            "End indicator bezel",
            (-1.904, y, 0.48),
            (0.025, 0.105, 0.105),
            M["black"],
            0.005,
        )
        lamp = box(
            "End amber utilisation " + str(k + 1),
            (-1.922, y, 0.48),
            (0.012, 0.075, 0.07),
            M["amber_off"],
            0.004,
        )
        LAMPS.append((lamp, k + 1, M["amber_on"]))
    MARKERS["activity_light_end"] = world((-1.93, -0.12, 0.48))
    box("End data socket rim", (-1.91, -0.67, 0.44), (0.05, 0.18, 0.19), M["steel"])
    box(
        "End blue data socket",
        (-1.94, -0.67, 0.44),
        (0.018, 0.13, 0.13),
        M["blue"],
        0.005,
    )
    box(
        "End socket interior",
        (-1.953, -0.67, 0.44),
        (0.012, 0.08, 0.075),
        M["black"],
        0.003,
    )
    lamp = box(
        "End blue power pilot",
        (-1.955, -0.67, 0.56),
        (0.012, 0.08, 0.025),
        M["blue_off"],
        0.003,
    )
    LAMPS.append((lamp, 0, M["blue_on"]))
    MARKERS["data_port_end"] = world((-1.96, -0.67, 0.44))
    # A real receiving backplane: all three bundles plug into this housing.
    box(
        "Interconnect receiving backplane",
        (1.055, 0.04, 0.55),
        (0.17, 1.08, 0.56),
        M["dark"],
    )
    for y, z in ((0.38, 0.65), (-0.30, 0.55), (0.05, 0.45)):
        box("Copper interconnect socket", (0.98, y, z), (0.13, 0.24, 0.18), M["copper"])
    paths = [
        (
            [
                (-1.60, 0.10, 0.795),
                (-1.60, 0.10, 0.98),
                (-1.45, 0.38, 1.12),
                (-0.91, 0.48, 1.33),
                (-0.42, 0.33, 1.07),
                (0.13, 0.35, 0.92),
                (0.70, 0.44, 1.0),
                (1.02, 0.48, 1.12),
                (0.79, 0.38, 0.65),
                (0.965, 0.38, 0.65),
            ],
            [0.16, 0.34, 0.60, 0.79],
            (0.34, 0.60),
        ),
        (
            [
                (-1.35, 0.10, 0.795),
                (-1.35, 0.10, 0.94),
                (-1.07, 0.30, 1.20),
                (-0.77, 0.10, 1.16),
                (-0.46, -0.31, 0.82),
                (0.05, -0.54, 0.67),
                (0.61, -0.48, 0.70),
                (0.79, -0.30, 0.55),
                (0.965, -0.30, 0.55),
            ],
            [0.20, 0.45, 0.73],
            (0.45,),
        ),
        (
            [
                (-1.10, 0.10, 0.795),
                (-1.10, 0.10, 0.91),
                (-0.94, 0.57, 1.04),
                (-0.51, 0.68, 1.02),
                (-0.12, 0.55, 0.74),
                (0.38, 0.55, 0.65),
                (0.84, 0.57, 0.64),
                (0.79, 0.05, 0.45),
                (0.965, 0.05, 0.45),
            ],
            [0.32, 0.68],
            (),
        ),
    ]
    for i in (2, 0, 1):
        loom("Folded neural interconnect " + str(i + 1), *paths[i])


def setup(sample_count, device):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if device != "CPU":
        prefs.compute_device_type = device
        prefs.get_devices()
        for render_device in prefs.devices:
            render_device.use = render_device.type == device
    scene.cycles.device = "CPU" if device == "CPU" else "GPU"
    scene.cycles.samples = sample_count
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.resolution_x, scene.render.resolution_y = [v * TILE for v in CANVAS]
    # One camera correction, shared by both rotations. Physical geometry is
    # genuinely 4x2; square game tiles project to square screen tiles.
    scene.render.pixel_aspect_x = 1 / math.sin(ELEVATION)
    scene.render.pixel_aspect_y = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "Medium High Contrast"
    scene.view_settings.exposure = -0.2
    data = bpy.data.cameras.new("Fixed Factorio orthographic camera")
    data.type, data.ortho_scale = "ORTHO", CANVAS[0]
    data.sensor_fit = "HORIZONTAL"
    cam = bpy.data.objects.new("Game camera", data)
    scene.collection.objects.link(cam)
    cam.location = Vector((0, -math.cos(ELEVATION), math.sin(ELEVATION))) * 30
    cam.rotation_euler = (math.pi / 2 - ELEVATION, 0, 0)
    scene.camera = cam
    for name, direction, energy, colour, shadow in [
        ("Key sun", SUN, 3.4, (1, 0.94, 0.85), True),
        (
            "Shadowless face fill",
            Vector((0.35, 0.8, -0.7)),
            0.85,
            (0.83, 0.90, 1),
            False,
        ),
    ]:
        ld = bpy.data.lights.new(name, "SUN")
        ld.energy, ld.angle, ld.color, ld.use_shadow = (
            energy,
            math.radians(1),
            colour,
            shadow,
        )
        obj = bpy.data.objects.new(name, ld)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        scene.collection.objects.link(obj)
    sky = bpy.data.worlds.new("Soft neutral sky")
    sky.use_nodes = True
    sky.node_tree.nodes["Background"].inputs["Color"].default_value = (
        0.72,
        0.79,
        0.88,
        1,
    )
    sky.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.45
    scene.world = sky
    return scene


def render(scene, out, name):
    scene.render.filepath = str(out / (name + ".png"))
    bpy.ops.render.render(write_still=True)


def icon(scene, out):
    camera = scene.camera
    location, rotation = camera.location.copy(), camera.rotation_euler.copy()
    scale = camera.data.ortho_scale
    resolution = scene.render.resolution_x, scene.render.resolution_y
    aspect = scene.render.pixel_aspect_x
    scene.render.pixel_aspect_x = 1
    camera.location = (6, -8, 8)
    camera.rotation_euler = (
        (Vector((0, 0, 0.55)) - camera.location).to_track_quat("-Z", "Y").to_euler()
    )
    camera.data.ortho_scale = 6.3
    scene.render.resolution_x = scene.render.resolution_y = 768
    render(scene, out, "icon")
    camera.location, camera.rotation_euler = location, rotation
    camera.data.ortho_scale = scale
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.pixel_aspect_x = aspect


def shadows(scene, out, model):
    # Directional projection of evaluated geometry: exact shared sun and no
    # ambient sky in the alpha. pack.py adds a subpixel soft edge.
    deps = bpy.context.evaluated_depsgraph_get()
    verts, faces = [], []
    for obj in model:
        ev = obj.evaluated_get(deps)
        mesh = ev.to_mesh()
        start = len(verts)
        for v in mesh.vertices:
            p = obj.matrix_world @ v.co
            p -= SUN * (p.z / SUN.z)
            p.z = 0.001
            verts.append(p)
        faces.extend(tuple(start + i for i in f.vertices) for f in mesh.polygons)
        ev.to_mesh_clear()
    mesh = bpy.data.meshes.new("Directional sun projection")
    mesh.from_pydata(verts, [], faces)
    shadow = bpy.data.objects.new("Separate ground shadow", mesh)
    scene.collection.objects.link(shadow)
    mesh.materials.append(emission("Pure black shadow", (0, 0, 0)))
    for obj in model:
        obj.hide_render = True
    scene.cycles.samples, scene.cycles.use_denoising = 8, False
    render(scene, out, "shadow")
    shadow.hide_render = True
    for obj in model:
        obj.hide_render = False


def glows(scene, out, model):
    holdout = bpy.data.materials.new("Occluder for light masks")
    holdout.use_nodes = True
    nodes = holdout.node_tree.nodes
    nodes.clear()
    out_node = nodes.new("ShaderNodeOutputMaterial")
    shader = nodes.new("ShaderNodeHoldout")
    holdout.node_tree.links.new(shader.outputs[0], out_node.inputs[0])
    originals = {obj: list(obj.data.materials) for obj in model}
    for obj in model:
        obj.data.materials.clear()
        obj.data.materials.append(holdout)
    for level in range(6):
        for obj, threshold, mat in LAMPS:
            obj.data.materials[0] = mat if threshold <= level else holdout
        render(scene, out, "glow-" + str(level))
    for obj, mats in originals.items():
        obj.data.materials.clear()
        for mat in mats:
            obj.data.materials.append(mat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "orientation",
        choices=["both", "horizontal", "vertical"],
        nargs="?",
        default="both",
    )
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--draft", action="store_true")
    parser.add_argument(
        "--device",
        choices=["CPU", "METAL", "CUDA", "OPTIX", "HIP"],
        default="METAL" if sys.platform == "darwin" else "CPU",
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.out.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    global M
    M = {
        "steel": material("Worn light steel", (0.31, 0.32, 0.29), 0.65, 0.48, 0.20),
        "dark": material(
            "Dark structural steel", (0.075, 0.085, 0.089), 0.65, 0.57, 0.18
        ),
        "pan": material("Equipment pan", (0.039, 0.043, 0.040), 0.45, 0.7),
        "rubber": material(
            "Charcoal cable jacket", (0.022, 0.027, 0.031), 0.08, 0.58, 0.16
        ),
        "copper": material(
            "Dull copper bindings", (0.38, 0.155, 0.052), 0.65, 0.48, 0.23
        ),
        "black": material("Deep recess", (0.004, 0.006, 0.007), 0.1, 0.9),
        "chip": material("Graphite chips", (0.065, 0.078, 0.088), 0.45, 0.5),
        "pcb": material("Olive circuit boards", (0.075, 0.092, 0.032), 0.2, 0.65, 0.18),
        "fin": material("Aluminium fins", (0.22, 0.25, 0.25), 0.8, 0.45),
        "blue": material("Data socket blue", (0.013, 0.064, 0.13), 0.4, 0.5),
        "red": material("Red insulator", (0.23, 0.035, 0.018), 0.1, 0.55),
        "green": material("Green insulator", (0.025, 0.14, 0.055), 0.1, 0.55),
        "amber_off": material("Unlit amber glass", (0.065, 0.025, 0.003), 0.2, 0.3),
        "blue_off": material("Unlit blue glass", (0.003, 0.02, 0.05), 0.2, 0.3),
        "amber_on": emission("Amber utilisation", (1, 0.38, 0.025)),
        "blue_on": emission("Blue pilot", (0.025, 0.38, 1)),
    }
    scene = setup(16 if args.draft else args.samples, args.device)
    build()
    model = [o for o in scene.objects if o.type in {"MESH", "CURVE"}]
    root = bpy.data.objects.new("Cable fold / one physical assembly", None)
    scene.collection.objects.link(root)
    for obj in model:
        obj.parent = root
    for name, position in MARKERS.items():
        marker = bpy.data.objects.new(name, None)
        scene.collection.objects.link(marker)
        marker.parent = root
        marker.location = position
        marker.empty_display_size = 0.06
    bpy.context.view_layer.update()
    original = {o.name: o.matrix_basis.copy() for o in model}
    geometry_hash = hashlib.sha256(
        repr(
            [
                (
                    o.name,
                    list(o.matrix_world),
                    len(o.data.vertices)
                    if o.type == "MESH"
                    else [tuple(p.co) for p in o.data.splines[0].points],
                )
                for o in model
            ]
        ).encode()
    ).hexdigest()
    for frame_number, angle in ((1, 0), (2, math.pi / 2)):
        root.rotation_euler.z = angle
        root.keyframe_insert(data_path="rotation_euler", frame=frame_number)
    scene.frame_start, scene.frame_end = 1, 2
    scene.timeline_markers.new("4x2 / broad side", frame=1)
    scene.timeline_markers.new("2x4 / quarter turn", frame=2)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    root["physical_footprint"] = "4 x 2; no layout changes between rotations"
    scene["render_projection"] = (
        "Shared orthographic camera; pixel aspect corrects ground foreshortening"
    )
    bpy.ops.wm.save_as_mainfile(filepath=str((args.out / "cable-fold.blend").resolve()))
    orientations = (
        ("horizontal", "vertical")
        if args.orientation == "both"
        else (args.orientation,)
    )
    for orientation in orientations:
        vertical = orientation == "vertical"
        scene.frame_set(2 if vertical else 1)
        bpy.context.view_layer.update()
        turn = root.matrix_world.copy()
        for obj in model:
            expected = turn @ original[obj.name]
            error = max(
                abs(obj.matrix_world[r][c] - expected[r][c])
                for r in range(4)
                for c in range(4)
            )
            assert error < 1e-5, (orientation, obj.name, error)
        out = args.out / orientation
        out.mkdir(parents=True, exist_ok=True)
        rotated = {name: turn @ p for name, p in MARKERS.items()}
        rotated["data_port"] = rotated[
            "data_port_end" if vertical else "data_port_front"
        ]
        rotated["activity_light"] = rotated[
            "activity_light_end" if vertical else "activity_light_front"
        ]
        metadata = {
            "orientation": orientation,
            "footprint": [2, 4] if vertical else [4, 2],
            "pixels_per_tile": TILE,
            "canvas": [v * TILE for v in CANVAS],
            "anchor": [v * TILE / 2 for v in CANVAS],
            "elevation_degrees": 38,
            "sun_direction": list(SUN),
            "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
            "assembly_rotation_degrees": 90 if vertical else 0,
            "geometry_sha256": geometry_hash,
            "object_count": len(model),
            "master_model": "../cable-fold.blend",
            "markers": {name: screen(p) for name, p in rotated.items()},
            "wire_shadow": {
                name: screen(p - SUN * (p.z / SUN.z))
                for name, p in rotated.items()
                if name in ("red", "green")
            },
        }
        (out / "render.json").write_text(json.dumps(metadata, indent=2) + "\n")
        scene.cycles.samples = 16 if args.draft else args.samples
        scene.cycles.use_denoising = True
        render(scene, out, "main")
        if not args.draft:
            if not vertical:
                icon(scene, out)
            shadows(scene, out, model)
            glows(scene, out, model)
    scene.frame_set(1)


if __name__ == "__main__":
    main()
