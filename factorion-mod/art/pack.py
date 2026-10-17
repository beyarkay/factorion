"""Turn gpu_server.py's renders into game sprites and a preview beside vanilla.

    uv run --no-project --with pillow python pack.py RENDER_DIR SPRITE_DIR horizontal|vertical

Renders are at 128 px per tile with the footprint's ground centre at the
canvas centre. Sprites are at 64 px per tile (scale = 0.5). Vanilla lines the
front edge of the footprint up with the front edge of the tile, so the anchor
sits above the projected ground centre by the depth the camera squashes away.
"""

import math
import sys
from pathlib import Path

from PIL import Image, ImageFilter

ELEVATION = math.radians(38)
render_dir, sprite_dir, orientation = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
FOOTPRINT = (4, 2) if orientation == "horizontal" else (2, 4)
sprite_dir.mkdir(parents=True, exist_ok=True)


def load(name):
    im = Image.open(render_dir / f"{name}.png").convert("RGBA")
    return im.resize((im.width // 2, im.height // 2), Image.Resampling.LANCZOS)


def sharpen(im):
    """Crisp the edges the way Wube's hand paint-over does."""
    rgb = im.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.2, percent=70, threshold=1))
    out = rgb.convert("RGBA")
    out.putalpha(im.getchannel("A"))
    return out


def crop(im, anchor):
    left, top, right, bottom = im.getchannel("A").point(lambda a: 255 if a > 2 else 0).getbbox()
    left, top, right, bottom = left - 2, top - 2, right + 2, bottom + 2
    if (right - left) % 2:
        right += 1
    if (bottom - top) % 2:
        bottom += 1
    sprite = im.crop((left, top, right, bottom))
    shift = ((left + right) / 2 - anchor[0], (top + bottom) / 2 - anchor[1])
    return sprite, shift


main = sharpen(load("main"))
# The sky darkens the whole catcher a little; only the sun's shadow counts.
shadow_alpha = load("shadow").getchannel("A").point(lambda a: max(0, min(255, (a - 40) * 255 // 200)))
shadow = Image.new("RGBA", main.size, (0, 0, 0, 255))
shadow.putalpha(shadow_alpha)

centre = (main.width / 2, main.height / 2)
anchor = (centre[0], centre[1] - FOOTPRINT[1] / 2 * 64 * (1 - math.sin(ELEVATION)))

lua = ["return {"]
for part, im in (("main", main), ("shadow", shadow)):
    name = f"gpu-server-{orientation}" + ("-shadow" if part == "shadow" else "")
    sprite, shift = crop(im, anchor)
    sprite.save(sprite_dir / f"{name}.png")
    lua.append(f"  {part} = {{ filename = \"__factorion__/graphics/entity/{name}.png\", width = {sprite.width}, "
               f"height = {sprite.height}, shift = {{ {shift[0] / 2 / 32:.3f}, {shift[1] / 2 / 32:.3f} }} }},")
lua.append("}")
(sprite_dir / f"gpu-server-{orientation}.lua").write_text("\n".join(lua) + "\n")
print("\n".join(lua))
