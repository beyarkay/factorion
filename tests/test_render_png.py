"""Tests for scripts/render_png.py — the colour-block PNG renderer."""

import os
import sys

import matplotlib.image as mpimg
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import render_png  # noqa: E402


# Between them: belts, undergrounds, splitters, sources, sinks, assemblers,
# inserters, long-handed inserters and poles.
@pytest.mark.parametrize("kind", ["ROUTE_BELTS", "REACH_OVER_2IN"])
def test_renders_a_lesson_to_png(tmp_path, kind):
    out = tmp_path / "factory.png"
    render_png.main(kind, seed=1, size=20, out=str(out))
    h, w, _ = mpimg.imread(out).shape
    assert w > 500 and h > 500
