"""Compose the packed GPU server sprites on Factorio ground beside vanilla
buildings, at the game's 64 px per tile, to judge scale, light and colour.

    uv run --no-project --with pillow python preview.py SPRITE_DIR FACTORIO_DATA OUT.png
"""

import re
import sys
from pathlib import Path

from PIL import Image

sprite_dir, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
TILE = 64
W, H = 18, 10
SHADOW_OPACITY = 0.55
base = data / "base/graphics/entity"


def by_pixel(x, y):
    return (x * 2, y * 2)


ours = {}
for name, w, h, x, y in re.findall(
    r'\["([\w-]+)"\] = \{ width = (\d+), height = (\d+), shift = util.by_pixel\(([-\d.]+), ([-\d.]+)\)',
    (sprite_dir / "gpu-server-sprites.lua").read_text(),
):
    ours[name] = (sprite_dir / f"{name}.png", int(w), int(h), by_pixel(float(x), float(y)))

server = ours["gpu-server-horizontal"]
server_shadow = ours["gpu-server-horizontal-shadow"]
vanilla = {
    "lab": ((base / "lab/lab.png", 194, 174, by_pixel(0, 1.5)), (base / "lab/lab-shadow.png", 242, 136, by_pixel(13, 11))),
    "assembler": (
        (base / "assembling-machine-2/assembling-machine-2.png", 214, 218, by_pixel(0, 4)),
        (base / "assembling-machine-2/assembling-machine-2-shadow.png", 196, 163, by_pixel(12, 4.75)),
    ),
    "accumulator": (
        (base / "accumulator/accumulator.png", 130, 189, by_pixel(0, -11)),
        (base / "accumulator/accumulator-shadow.png", 234, 106, by_pixel(29, 6)),
    ),
    "chest": ((base / "steel-chest/steel-chest.png", 64, 80, by_pixel(-0.25, -0.5)), None),
}

# (what, centre in tiles)
placements = [(("server",), (2 + 4 * i, 2)) for i in range(4)]
placements += [
    (("server",), (8, 7)),
    (("vanilla", "lab"), (1.5, 6.5)),
    (("vanilla", "assembler"), (4.5, 6.5)),
    (("vanilla", "accumulator"), (12, 7)),
    (("vanilla", "chest"), (14.5, 7.5)),
    (("server",), (16, 4)),
]

ground_src = Image.open(data / "base/graphics/terrain/concrete/concrete.png").convert("RGBA").crop((0, 0, 256, 256))


def paste(layer, centre, onto):
    path, w, h, (sx, sy) = layer
    sprite = Image.open(path).convert("RGBA").crop((0, 0, w, h))
    cx, cy = centre[0] * TILE + sx, centre[1] * TILE + sy
    onto.alpha_composite(sprite, (round(cx - w / 2), round(cy - h / 2)))


def layers(what):
    if what[0] == "server":
        return server, server_shadow
    main, shadow = vanilla[what[1]]
    return main, shadow


def compose(placements, width, height):
    img = Image.new("RGBA", (width * TILE, height * TILE))
    for gx in range(0, width * TILE, 256):
        for gy in range(0, height * TILE, 256):
            img.alpha_composite(ground_src, (gx, gy))
    # The game merges every shadow into one layer before darkening the ground,
    # so overlaps don't stack; the opacity is a guess, shared by all.
    shadows = Image.new("RGBA", img.size)
    for what, centre in placements:
        shadow = layers(what)[1]
        if shadow:
            paste(shadow, centre, shadows)
    shadows.putalpha(shadows.getchannel("A").point(lambda a: int(a * SHADOW_OPACITY)))
    img.alpha_composite(shadows)
    for what, centre in sorted(placements, key=lambda p: p[1][1]):
        paste(layers(what)[0], centre, img)
    return img


img = compose(placements, W, H)
img.save(out)
big = img.resize((img.width * 2, img.height * 2), Image.NEAREST)
big.crop((0, 0, 9 * TILE * 2, 5 * TILE * 2)).save(out.with_name(out.stem + "-zoom.png"))
# Rows of servers touching, two tiles apart, to check how they stack.
stack = compose([(("server",), (3 + 4 * i, 2 + 2 * j)) for i in range(3) for j in range(4)], 14, 11)
stack.resize((stack.width * 2, stack.height * 2), Image.NEAREST).save(out.with_name(out.stem + "-stack.png"))
