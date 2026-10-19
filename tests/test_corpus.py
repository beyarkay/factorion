"""The human-blueprint corpus converter (`corpus.convert`)."""

import itertools
import random

import numpy as np
import pytest
import torch

from corpus.convert import orient, rasterise, to_world, window
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
            ("pipe", 2.5, 0.5, {}),
            ("assembling-machine-3", 4.5, 1.5, {"recipe": "iron-gear-wheel"}),
            ("assembling-machine-2", 7.5, 1.5, {"recipe": "rocket-fuel"}),
        )
    )
    assert r.ent[0, 0] == str2ent("transport_belt").value
    assert r.ent[1, 0] == str2ent("inserter").value
    assert (r.ent[3:6, 0:3] == str2ent("assembling_machine_1").value).all()
    assert r.inst[2, 0] == -1 and (r.inst[6:9, 0:3] == -1).all()
    assert {m[0] for m in r.masked} == {"pipe", "assembling-machine-2"}


# Every entity that only human blueprints use.
HUMAN_ONLY = (
    "wooden_chest", "iron_chest", "steel_chest", "stone_furnace", "steel_furnace", "electric_furnace",
    "big_electric_pole", "solar_panel", "accumulator", "roboport", "radar", "small_lamp", "stone_wall",
)


def test_human_only_entities_round_trip_through_a_blueprint():
    world = np.zeros((len(Channel), 5 * len(HUMAN_ONLY), 4), np.int64)
    for i, name in enumerate(HUMAN_ONLY):
        e = str2ent(name)
        world[Channel.ENTITIES.value, 5 * i : 5 * i + e.width, : e.height] = e.value
    r = _raster(world_tensor_to_blueprint_dict(world)["blueprint"])
    assert not r.masked
    for got, c in zip((r.ent, r.dir, r.item, r.misc), (Channel.ENTITIES, Channel.DIRECTION, Channel.ITEMS, Channel.MISC)):
        np.testing.assert_array_equal(got, world[c.value, : r.ent.shape[0]])


def test_human_only_entities_are_kept_not_masked():
    r = _raster(
        _bp(
            ("stone-furnace", 1, 1, {}),
            ("electric-furnace", 3.5, 1.5, {}),
            ("wooden-chest", 5.5, 0.5, {}),
            ("passive-provider-chest", 6.5, 0.5, {}),
            ("big-electric-pole", 8, 1, {}),
            ("roboport", 11, 2, {}),
        )
    )
    assert not r.masked
    assert (r.ent[0:2, 0:2] == str2ent("stone_furnace").value).all()
    assert (r.ent[2:5, 0:3] == str2ent("electric_furnace").value).all()
    # A furnace picks its recipe from what it is fed.
    assert not r.item.any()
    assert r.ent[5, 0] == str2ent("wooden_chest").value
    assert r.ent[6, 0] == str2ent("steel_chest").value
    assert (r.ent[7:9, 0:2] == str2ent("big_electric_pole").value).all()
    assert (r.ent[9:13, 0:4] == str2ent("roboport").value).all()
    assert not r.dir.any()


def test_1_1_logistic_chests_are_steel_chests():
    r = _raster(_bp(("logistic-chest-requester", 0.5, 0.5, {}), ("inserter", 1.5, 0.5, {"direction": 2}), version=1))
    assert r.ent[0, 0] == str2ent("steel_chest").value


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
    """A window of a row of assemblers only ever keeps whole machines, on
    buildable tiles; the ones its edge cuts and the grid around it are
    masked, never unbuildable."""
    r = _raster(_bp(*(("assembling-machine-1", 1.5 + 3 * i, 1.5, {"recipe": "iron-gear-wheel"}) for i in range(10))))
    rng = random.Random(0)
    for _ in range(50):
        w = window(r, 16, rng, min_known=0, max_masked=1)
        assert w is not None
        asm = w[Channel.ENTITIES.value] == str2ent("assembling_machine_1").value
        assert asm.sum() % 9 == 0
        fp = w[Channel.FOOTPRINT.value]
        assert (fp[asm] == Footprint.BUILDABLE.value).all()
        assert (fp == Footprint.MASKED.value).any() and not (fp == Footprint.UNBUILDABLE.value).any()


def test_windows_need_belts_inserters_or_assemblers():
    """A field of solar panels never trains; the same field with one belt
    does, in a window as big as the blueprint where it is smaller than the
    grid."""
    field = [("solar-panel", 1.5 + 3 * i, 1.5 + 3 * j, {}) for i in range(8) for j in range(8)]
    rng = random.Random(0)
    assert all(window(_raster(_bp(*field)), 32, rng) is None for _ in range(20))
    r = _raster(_bp(*field, ("transport-belt", 24.5, 24.5, {})))
    ws = [w for w in (window(r, 32, rng, max_masked=1) for _ in range(50)) if w is not None]
    assert ws
    for w in ws:
        xs, ys = np.nonzero(w[Channel.FOOTPRINT.value] != Footprint.MASKED.value)
        assert (xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) in (r.ent.shape, r.ent.shape[::-1])


def test_sft_stream_mixes_in_human_blueprint_windows(tmp_path):
    """Lessons published as blueprints go through the whole corpus path —
    scraped records, the built dataset, the SFT stream — and come out as
    in-vocabulary training states."""
    import gzip
    import json

    from corpus.build import build
    from denoise import DenoiserUNet, _luts, free_mask
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
    assert (s[:, Channel.FOOTPRINT.value] == Footprint.MASKED.value).any()
    model = DenoiserUNet(width=16, levels=1, blocks=1, attn_layers=1, attn_heads=1)
    assert model(states).shape[-2:] == states.shape[-2:]


def test_stats_count_a_lessons_entities_and_windows():
    from corpus.stats import _design
    from factorion_server.blueprint import world_tensor_to_blueprint_string

    world = _lesson(LessonKind.OPPOSITE_SIDES_1IN, 0).world_CWH.numpy()
    c = _design(16, {"string": world_tensor_to_blueprint_string(world), "source": "test"})
    assert c["kept/assembling_machine_1"] == (world[Channel.ENTITIES.value] == str2ent("assembling_machine_1").value).sum() // 9
    assert c["masked/constant-combinator"] == 2  # the source and sink markers
    assert c["windows"] >= 1


def test_terrain_is_unbuildable_and_unmodelled_entities_masked():
    """A save's terrain covers only empty tiles and reads as unbuildable; an
    entity the engine doesn't model reads as masked."""
    bp = _bp(("transport-belt", 0.5, 0.5, {}), ("transport-belt", 4.5, 2.5, {}), ("beacon", 2.5, 3.5, {}))
    bp["unbuildable"] = [[-3, 1, 5], [0, 0, 9]]
    fp = to_world(_raster(bp))[Channel.FOOTPRINT.value]
    assert (fp[0:2, 1] == Footprint.UNBUILDABLE.value).all() and (fp[2:, 1] == Footprint.BUILDABLE.value).all()
    assert fp[0, 0] == Footprint.BUILDABLE.value and (fp[1:, 0] == Footprint.UNBUILDABLE.value).all()
    assert (fp[1:4, 2:5] == Footprint.MASKED.value).all()
