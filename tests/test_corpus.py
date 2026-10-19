"""The human-blueprint corpus converter (`corpus.convert`)."""

import itertools
import random

import numpy as np
import pytest
import torch

from corpus.convert import orient, rasterise, window
from denoise import score_world
from factorion import LESSON_IS_TRIAL, Channel, Footprint, LessonKind, build_factory, str2ent
from factorion_server.blueprint import world_tensor_to_blueprint_dict

KINDS = [k for k in LessonKind if not LESSON_IS_TRIAL[k]]


def _lesson(kind, seed):
    factory = build_factory(size=20, kind=kind, seed=seed)
    if factory is None:
        pytest.skip(f"{kind.name} did not build")
    return factory


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
def test_lesson_round_trips_through_a_blueprint(kind):
    """A generated factory encoded as a blueprint rasterises back to itself
    (the markers, constant combinators in a blueprint, come back masked)."""
    world = _lesson(kind, 0).world_CWH.numpy()
    r = _raster(world_tensor_to_blueprint_dict(world)["blueprint"])
    xs, ys = np.nonzero(world[Channel.ENTITIES.value])
    crop = world[:, xs.min() : xs.max() + 1, ys.min() : ys.max() + 1]
    marker = np.isin(crop[Channel.ENTITIES.value], [str2ent("source").value, str2ent("sink").value])
    for got, c in zip((r.ent, r.dir, r.item, r.misc), (Channel.ENTITIES, Channel.DIRECTION, Channel.ITEMS, Channel.MISC)):
        np.testing.assert_array_equal(got, np.where(marker, 0, crop[c.value]))
    np.testing.assert_array_equal(r.inst < 0, marker)


@pytest.mark.parametrize("kind", KINDS, ids=lambda k: k.name)
@pytest.mark.parametrize("turns", range(4))
@pytest.mark.parametrize("mirror", [False, True])
def test_orienting_a_lesson_keeps_its_throughput(kind, turns, mirror):
    factory = _lesson(kind, 1)
    world = factory.world_CWH.numpy()
    turned = np.stack(orient(list(world), turns, mirror))
    assert score_world(turned, factory.max_throughput, 0)["thput_raw"] == pytest.approx(
        score_world(world, factory.max_throughput, 0)["thput_raw"]
    )


def _raster(bp):
    r = rasterise(bp)
    assert r is not None
    return r


def _bp(*ents, version=2):
    return {"version": version << 48, "entities": [{"name": n, "position": {"x": x, "y": y}, **kw} for n, x, y, kw in ents]}


def test_tiers_collapse_and_unmodelled_entities_mask():
    r = _raster(
        _bp(
            ("express-transport-belt", 0.5, 0.5, {"direction": 4}),
            ("fast-inserter", 1.5, 0.5, {"direction": 4}),
            ("steel-chest", 2.5, 0.5, {}),
            ("assembling-machine-3", 4.5, 1.5, {"recipe": "iron-gear-wheel"}),
            ("assembling-machine-2", 7.5, 1.5, {"recipe": "rocket-fuel"}),
        )
    )
    assert r.ent[0, 0] == str2ent("transport_belt").value
    assert r.ent[1, 0] == str2ent("inserter").value
    assert (r.ent[3:6, 0:3] == str2ent("assembling_machine_1").value).all()
    assert r.inst[2, 0] == -1 and (r.inst[6:9, 0:3] == -1).all()
    assert {m[0] for m in r.masked} == {"steel-chest", "assembling-machine-2"}


def test_mod_entities_reject_the_blueprint():
    assert rasterise(_bp(("transport-belt", 0.5, 0.5, {}), ("se-space-pipe", 1.5, 0.5, {}))) is None


@pytest.mark.parametrize(
    "tier, gap, kept",
    [("", 4, True), ("express-", 4, True), ("express-", 6, False)],
)
def test_undergrounds_mask_unless_yellow_reach_pairs_them_alike(tier, gap, kept):
    """An express tunnel longer than a yellow one can reach would come apart
    collapsed to yellow, so both its ends mask."""
    name = f"{tier}underground-belt"
    r = _raster(
        _bp(
            ("transport-belt", 0.5, 0.5, {"direction": 4}),
            (name, 1.5, 0.5, {"direction": 4, "type": "input"}),
            (name, 2.5 + gap, 0.5, {"direction": 4, "type": "output"}),
        )
    )
    assert ((r.inst[[1, 2 + gap], 0] > 0) == kept).all()


def test_window_masks_the_entities_its_edge_cuts():
    """A window of a row of assemblers only ever keeps whole machines; the
    rest of the window and the grid around it are unbuildable."""
    r = _raster(_bp(*(("assembling-machine-1", 1.5 + 3 * i, 1.5, {"recipe": "iron-gear-wheel"}) for i in range(10))))
    rng = random.Random(0)
    for _ in range(50):
        w = window(r, 16, rng, min_known=0, max_masked=1)
        assert w is not None
        asm = w[Channel.ENTITIES.value] == str2ent("assembling_machine_1").value
        assert asm.sum() % 9 == 0
        assert not (asm & (w[Channel.FOOTPRINT.value] == Footprint.UNAVAILABLE.value)).any()


def test_sft_stream_mixes_in_human_blueprint_windows(tmp_path):
    """Lessons published as blueprints go through the whole corpus path —
    scraped records, the built dataset, the SFT stream — and come out as
    in-vocabulary training states."""
    import gzip
    import json

    from corpus.build import build
    from denoise import _luts, free_mask
    from factorion_server.blueprint import world_tensor_to_blueprint_string
    from sft import _iter_factories

    raw = tmp_path / "lessons.jsonl.gz"
    with gzip.open(raw, "wt") as f:
        for i, kind in enumerate(KINDS):
            factory = build_factory(size=20, kind=kind, seed=2)
            if factory is not None:
                string = world_tensor_to_blueprint_string(factory.world_CWH.numpy())
                f.write(json.dumps({"source": "test", "id": i, "string": string}) + "\n")
    stats = build([raw], tmp_path / "out")
    assert stats["kept/test"] > 0
    files = sorted((tmp_path / "out" / "designs").glob("*.parquet"))
    states = torch.stack(list(itertools.islice(_iter_factories(16, 0, 0, 1, corpus_files=files, corpus_frac=1.0), 64)))
    L = _luts(states.device)
    s = states.long()
    free = free_mask(states)
    assert (L.type_lut[s[:, Channel.ENTITIES.value], s[:, Channel.DIRECTION.value], s[:, Channel.MISC.value]][free] >= 0).all()
    assert (s[:, Channel.ENTITIES.value][free] > 0).float().mean() > 0.1
