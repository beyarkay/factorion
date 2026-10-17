"""Draw the mod's alert icons and the GPU glyph its circuit signals are built on.

    uv run --no-project --with pillow python icons.py MOD_GRAPHICS_DIR

Alerts copy vanilla's: a flat yellow rounded triangle with a dark symbol.
Everything is drawn at four times size and shrunk, for clean edges.
"""

import sys
from pathlib import Path

from PIL import Image, ImageDraw

SS = 4
SIZE = 64 * SS
YELLOW = (255, 197, 0, 255)
INK = (29, 29, 29, 255)
out = Path(sys.argv[1])


def s(*values):
    return [v * SS for v in values]


# Vanilla centres each symbol's ink here, low in the triangle.
GLYPH_CENTRE = (32, 41)


def triangle():
    im = Image.new("RGBA", (SIZE, SIZE))
    d = ImageDraw.Draw(im)
    r = 7
    # Corners inset by r, then grown by r: a triangle with round corners.
    pts = [(32, 2 + r), (r - 1, 62 - r), (65 - r, 62 - r)]
    d.polygon([tuple(s(*p)) for p in pts], fill=YELLOW)
    for a, b in zip(pts, pts[1:] + pts[:1]):
        d.line([tuple(s(*a)), tuple(s(*b))], fill=YELLOW, width=2 * r * SS)
    for p in pts:
        d.ellipse(s(p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=YELLOW)
    return im


def glyph():
    im = Image.new("RGBA", (SIZE, SIZE))
    return im, ImageDraw.Draw(im)


def save(im, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    im.resize((SIZE // SS, SIZE // SS), Image.Resampling.LANCZOS).save(path)


def save_alert(symbol, path):
    """The symbol's ink, not its yellow gaps, is centred on the triangle."""
    mask = Image.eval(symbol.convert("RGB"), lambda v: 255 if v < 100 else 0).convert("L")
    mask = Image.composite(mask, Image.new("L", mask.size), symbol.getchannel("A").point(lambda a: 255 if a > 128 else 0))
    box = mask.getbbox()
    assert box, f"{path.name} has no ink"
    left, top, right, bottom = box
    dx = round(GLYPH_CENTRE[0] * SS - (left + right) / 2)
    dy = round(GLYPH_CENTRE[1] * SS - (top + bottom) / 2)
    im = triangle()
    im.alpha_composite(symbol, (dx, dy))
    save(im, path)


def card(d, x0, y0, x1, y1, fill, hole):
    """A GPU card from the side: a slab with two fans and a bracket."""
    d.rounded_rectangle(s(x0, y0, x1, y1), radius=2 * SS, fill=fill)
    r = (y1 - y0) / 2 - 2.2
    cy = (y0 + y1) / 2
    for cx in (x0 + (x1 - x0) * 0.3, x0 + (x1 - x0) * 0.7):
        d.ellipse(s(cx - r, cy - r, cx + r, cy + r), fill=hole)
        d.ellipse(s(cx - r * 0.35, cy - r * 0.35, cx + r * 0.35, cy + r * 0.35), fill=fill)
    d.rectangle(s(x0 + 2, y1, x0 + 7, y1 + 3), fill=fill)


def slash(d):
    d.line([tuple(s(15, 58)), tuple(s(49, 24))], fill=YELLOW, width=8 * SS)
    d.line([tuple(s(15, 58)), tuple(s(49, 24))], fill=INK, width=3 * SS)


# A GPU server that can't reach its model: a card, struck out.
im, d = glyph()
card(d, 16, 31, 48, 48, INK, YELLOW)
slash(d)
save_alert(im, out / "alerts/gpu-offline.png")

# A painted area with no GPU server, or a server with no area: a broken link.
im, d = glyph()
for x0 in (11, 34):
    d.rounded_rectangle(s(x0, 38, x0 + 19, 50), radius=6 * SS, outline=INK, width=3 * SS)
d.line([tuple(s(21, 44)), tuple(s(27, 44))], fill=INK, width=3 * SS)
d.line([tuple(s(37, 44)), tuple(s(43, 44))], fill=INK, width=3 * SS)
for a, b in (((29, 33), (30, 37)), ((35, 33), (34, 37)), ((29, 55), (30, 51)), ((35, 55), (34, 51))):
    d.line([tuple(s(*a)), tuple(s(*b))], fill=INK, width=2 * SS)
save_alert(im, out / "alerts/unlinked.png")

# Light grey like vanilla signal glyphs, with a dark rim to read on any slot.
im = Image.new("RGBA", (SIZE, SIZE))
d = ImageDraw.Draw(im)
card(d, 4, 14, 60, 44, (40, 40, 40, 255), (40, 40, 40, 255))
card(d, 6, 16, 58, 42, (215, 215, 215, 255), (60, 60, 60, 255))
save(im, out / "icons/gpu-glyph.png")
