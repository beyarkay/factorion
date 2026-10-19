"""Tests for scripts/render_png.py — the colour-block PNG renderer."""

import os
import sys

import matplotlib.image as mpimg
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import render_png  # noqa: E402
from factorion import Channel, str2ent  # noqa: E402


# Between them: belts, undergrounds, splitters, sources, sinks, assemblers,
# inserters, long-handed inserters and poles.
@pytest.mark.parametrize("kind", ["ROUTE_BELTS", "REACH_OVER_2IN"])
def test_renders_a_lesson_to_png(tmp_path, kind):
    out = tmp_path / "factory.png"
    render_png.main(kind, seed=1, size=20, out=str(out))
    h, w, _ = mpimg.imread(out).shape
    assert w > 500 and h > 500


def test_renders_chests_furnaces_and_inert_entities(tmp_path):
    from factorion import new_world

    w = new_world(width=12, height=4)
    for name, x in (("steel_chest", 0), ("stone_furnace", 1), ("roboport", 3), ("big_electric_pole", 8)):
        e = str2ent(name)
        w[x : x + e.width, 0 : e.height, Channel.ENTITIES.value] = e.value
    out = tmp_path / "factory.png"
    render_png.save(np.transpose(w, (2, 0, 1)), out, "new entities")
    assert mpimg.imread(out).shape[1] > 300
