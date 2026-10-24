"""The human-blueprint corpus converter (`corpus.convert`)."""

import itertools
import random

import numpy as np
import pytest
import torch

from corpus.convert import orient, rasterise, to_world, window
from denoise import score_world
from factorion import DIR_TO_DELTA, LESSON_IS_TRIAL, Channel, Direction, Footprint, LessonKind, build_factory, items, str2ent
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


def test_build_region_is_buildable_and_the_rest_of_the_view_context():
    """In a view of a long row of assemblers, the build region is a buildable
    rectangle; the other whole machines are unbuildable context; machines the
    view's edge cuts are masked."""
    r = _raster(_bp(*(("assembling-machine-1", 1.5 + 3 * i, 1.5, {"recipe": "iron-gear-wheel"}) for i in range(20))))
    rng = random.Random(0)
    for _ in range(50):
        w = window(r, 32, rng, max_masked=1)
        assert w is not None
        fp = w[Channel.FOOTPRINT.value]
        asm = w[Channel.ENTITIES.value] == str2ent("assembling_machine_1").value
        assert asm.sum() % 9 == 0
        assert np.isin(fp[asm], [Footprint.BUILDABLE.value, Footprint.UNBUILDABLE.value]).all()
        xs, ys = np.nonzero(fp == Footprint.BUILDABLE.value)
        assert len(xs) == (xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1)
        assert (fp == Footprint.MASKED.value).any()


@pytest.mark.parametrize("region", [False, True])
def test_past_a_blueprint_is_unbuildable_but_for_a_masked_band(region):
    """Around a design smaller than the grid lies a one-tile masked band, so
    its inputs and outputs lead somewhere unknown, then unbuildable tiles; a
    save region's world goes on, so all of it is masked."""
    r = _raster(_bp(
        ("assembling-machine-1", 1.5, 1.5, {"recipe": "iron-gear-wheel"}),
        ("inserter", 3.5, 1.5, {"direction": 12}),
        ("transport-belt", 4.5, 1.5, {"direction": 0}),
    ))
    w = window(r, 32, random.Random(0), region=region)
    assert w is not None
    fp = w[Channel.FOOTPRINT.value]
    xs, ys = np.nonzero(w[Channel.ENTITIES.value])
    gx, gy = np.arange(32)[:, None], np.arange(32)[None, :]
    gap = np.maximum(np.maximum(xs.min() - gx, gx - xs.max()), np.maximum(ys.min() - gy, gy - ys.max()))
    assert (fp[gap == 1] == Footprint.MASKED.value).all()
    beyond = Footprint.MASKED.value if region else Footprint.UNBUILDABLE.value
    assert (fp[gap >= 2] == beyond).all()


def test_windows_need_belts_inserters_or_assemblers():
    """A field of solar panels never trains; the same field with one belt
    does, and the belt is in the build region."""
    field = [("solar-panel", 1.5 + 3 * i, 1.5 + 3 * j, {}) for i in range(8) for j in range(8)]
    rng = random.Random(0)
    assert all(window(_raster(_bp(*field)), 32, rng) is None for _ in range(20))
    r = _raster(_bp(*field, ("transport-belt", 24.5, 24.5, {})))
    ws = [w for w in (window(r, 32, rng, max_masked=1) for _ in range(20)) if w is not None]
    assert ws
    for w in ws:
        belt = w[Channel.ENTITIES.value] == str2ent("transport_belt").value
        assert (w[Channel.FOOTPRINT.value][belt] == Footprint.BUILDABLE.value).all()


def _lesson_corpus(tmp_path) -> list:
    """The design shards of a corpus built from one lesson of each kind."""
    import gzip
    import json

    from corpus.build import build
    from factorion_server.blueprint import world_tensor_to_blueprint_string

    raw = tmp_path / "lessons.jsonl.gz"
    with gzip.open(raw, "wt") as f:
        for i, kind in enumerate(KINDS):
            factory = build_factory(size=20, kind=kind, seed=2)
            if factory is not None:
                string = world_tensor_to_blueprint_string(factory.world_CWH.numpy())
                f.write(json.dumps({"source": "test", "id": i, "string": string}) + "\n")
    assert build([raw], tmp_path / "out")["kept/test"] > 0
    return sorted((tmp_path / "out" / "designs").glob("*.parquet"))


def test_sft_stream_mixes_in_human_blueprint_windows(tmp_path):
    """Lessons published as blueprints go through the whole corpus path —
    scraped records, the built dataset, the SFT stream — and come out as
    in-vocabulary training states."""
    from denoise import DenoiserUNet, _luts, free_mask
    from sft import _iter_factories

    files = _lesson_corpus(tmp_path)
    states = torch.stack(list(itertools.islice(_iter_factories(16, 0, 0, 1, corpus_files=files, corpus_frac=1.0), 64)))
    L = _luts(states.device)
    s = states.long()
    free = free_mask(states)
    assert (L.type_lut[s[:, Channel.ENTITIES.value], s[:, Channel.DIRECTION.value], s[:, Channel.MISC.value]][free] >= 0).all()
    assert (s[:, Channel.ENTITIES.value][free] > 0).float().mean() > 0.1
    assert (s[:, Channel.FOOTPRINT.value] == Footprint.MASKED.value).any()
    model = DenoiserUNet(width=16, levels=1, blocks=1, attn_layers=1, attn_heads=1)
    assert model(states).shape[-2:] == states.shape[-2:]


def test_workers_split_the_training_designs_and_never_see_held_out_ones(tmp_path, monkeypatch):
    import corpus.convert as convert

    files = _lesson_corpus(tmp_path)
    monkeypatch.setattr(convert, "held_out", lambda h: int(h[:8], 16) % 3 == 0)
    monkeypatch.setattr(convert, "window", lambda r, *_: r.ent.tobytes())

    def seen(**kw) -> set:
        return set(itertools.islice(convert.stream(files, 16, random.Random(0), **kw), 2000))

    a, b, val = seen(worker=0, workers=2), seen(worker=1, workers=2), seen(val=True)
    assert a and b and val and not (a & b) and not ((a | b) & val)
    assert seen() == a | b


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


def test_a_save_regions_terrain_and_belt_contents_reach_its_design():
    from corpus.build import _rows
    from corpus.convert import designs
    from factorion_server.blueprint import world_tensor_to_blueprint_string

    world = _lesson(LessonKind.OPPOSITE_SIDES_1IN, 0).world_CWH.numpy()
    rec = {"source": "save:test", "string": world_tensor_to_blueprint_string(world),
           "unbuildable": [[0, 0, 2]], "lanes": [[3, 4, 1, "iron-plate"]]}
    (row,) = _rows(rec)
    bp = next(designs(row["string"]))
    assert bp["unbuildable"] == rec["unbuildable"] and bp["lanes"] == rec["lanes"]


def test_a_lesson_stripped_of_its_markers_works_again_as_a_functional_window():
    """A lesson's markers come back masked (`test_lesson_round_trips_through_a_blueprint`):
    the functional window puts a source and sink where its flows meet them,
    and with nothing labelled picks the items that make the machines produce
    as much as the lesson did."""
    import factorion_rs

    from corpus.convert import _machine_items, deliveries, edit, paste

    world = _lesson(LessonKind.OPPOSITE_SIDES_1IN, 0).world_CWH.numpy()
    bp = world_tensor_to_blueprint_dict(world)["blueprint"]
    xs, ys = [e["position"]["x"] for e in bp["entities"]], [e["position"]["y"] for e in bp["entities"]]
    for x, y in ((min(xs) - 12, min(ys) - 12), (max(xs) + 12, max(ys) + 12)):  # room for the window's ring
        bp["entities"].append({"name": "small-electric-pole", "position": {"x": x, "y": y}})
    r = _raster(bp)
    assert r.ent.shape[0] >= 34 and r.ent.shape[1] >= 34
    lesson = sum(rate for *_, rate in factorion_rs.py_sink_deliveries(np.ascontiguousarray(world.transpose(1, 2, 0))))
    w = paste(r, 2, 2, 32, edit(r, 2, 2, 32, random.Random(0)))
    _, products = _machine_items(w)
    assert lesson > 0 and deliveries(w, products)[1] == pytest.approx(lesson)


def test_labelled_belts_across_the_edge_and_store_chests_become_markers_of_what_they_carried():
    import factorion_rs

    from corpus.convert import edit, paste

    east, north = {"direction": 4}, {"direction": 0}
    bp = _bp(
        *(("transport-belt", x + 0.5, 3.5, east) for x in range(40)),
        ("inserter", 20.5, 4.5, north),  # from the belt into the chest
        ("wooden-chest", 20.5, 5.5, {}),
        ("medium-electric-pole", 21.5, 5.5, {}),
        ("small-electric-pole", 0.5, 0.5, {}),
        ("small-electric-pole", 0.5, 39.5, {}),
    )
    bp["lanes"] = [[20, 3, 1, "copper-plate"]]
    edits = edit(r := _raster(bp), 5, 1, 32, random.Random(0))
    w = paste(r, 5, 1, 32, edits)
    ent, item = w[Channel.ENTITIES.value], w[Channel.ITEMS.value]
    copper = str2ent("copper_plate").value
    for (x, y), kind in (((0, 3), "source"), ((31, 3), "sink"), ((16, 5), "sink")):
        assert (ent[x, y], item[x, y]) == (str2ent(kind).value, copper)
    assert len(edits) == 3  # the human's pole already powers the inserter
    got = {(x, y): rate for x, y, _, rate in factorion_rs.py_sink_deliveries(np.ascontiguousarray(w.transpose(1, 2, 0)))}
    assert got[(16, 5)] > 0 and got[(16, 5)] + got[(31, 3)] == pytest.approx(15)


def test_robot_fed_chests_wagons_and_tunnels_out_of_the_region_become_markers():
    """A chest inserters only empty is a source facing no belt, an inserter
    loading nothing loads a wagon, and a tunnel whose exit lies past the
    region's edge ends in a sink."""
    import factorion_rs

    from corpus.convert import edit, paste

    east, north = {"direction": 4}, {"direction": 0}
    bp = _bp(
        *(("transport-belt", x + 0.5, 1.5, east) for x in range(4, 30)),
        ("underground-belt", 30.5, 1.5, {**east, "type": "input"}),
        ("steel-chest", 10.5, 2.5, {}),
        ("inserter", 10.5, 3.5, north),  # from the chest onto the empty track
        ("medium-electric-pole", 11.5, 3.5, {}),
        ("small-electric-pole", 0.5, 0.5, {}),
        ("small-electric-pole", 39.5, 39.5, {}),
    )
    bp["lanes"] = [[12, 1, 1, "copper-plate"]]
    w = paste(r := _raster(bp), 5, 1, 32, edit(r, 5, 1, 32, random.Random(0)))
    ent, facing = w[Channel.ENTITIES.value], w[Channel.DIRECTION.value]
    assert ent[6, 2] == str2ent("source").value
    dx, dy = DIR_TO_DELTA[Direction(facing[6, 2])]
    assert ent[6 + dx, 2 + dy] != str2ent("transport_belt").value
    got = {(x, y): rate for x, y, _, rate in factorion_rs.py_sink_deliveries(np.ascontiguousarray(w.transpose(1, 2, 0)))}
    assert got[(6, 4)] > 0 and got[(26, 1)] == pytest.approx(15)


def _save_corpus(tmp_path, kinds=(LessonKind.OPPOSITE_SIDES_1IN, LessonKind.SPLITTER_1IN)) -> tuple[list, list]:
    """The design and window shards of a corpus of save regions, one per kind:
    a lesson stripped of its markers, its input belt labelled."""
    import gzip
    import json

    from corpus.build import _encode, build
    from corpus.windows import build as build_windows

    raw = tmp_path / "regions.jsonl.gz"
    with gzip.open(raw, "wt") as f:
        for i, kind in enumerate(kinds):
            world = _lesson(kind, 0).world_CWH.numpy()
            bp = world_tensor_to_blueprint_dict(world)["blueprint"]
            xs, ys = [e["position"]["x"] for e in bp["entities"]], [e["position"]["y"] for e in bp["entities"]]
            for x, y in ((min(xs) - 12, min(ys) - 12), (max(xs) + 12, max(ys) + 12)):
                bp["entities"].append({"name": "small-electric-pole", "position": {"x": x, "y": y}})
            (sx, sy), = np.argwhere(world[Channel.ENTITIES.value] == str2ent("source").value)
            dx, dy = DIR_TO_DELTA[Direction(world[Channel.DIRECTION.value, sx, sy])]
            fed = items[int(world[Channel.ITEMS.value, sx, sy])].name.replace("_", "-")
            rec = {"source": "save:test", "id": f"save{i}.zip#nauvis#0,0", "string": _encode(bp), "unbuildable": [],
                   "lanes": [[int(sx + dx), int(sy + dy), 1, fed]]}
            f.write(json.dumps(rec) + "\n")
    build([raw], tmp_path / "out")
    assert build_windows([raw], tmp_path / "out", 1) >= len(kinds)
    return sorted((tmp_path / "out" / "designs").glob("*.parquet")), sorted((tmp_path / "out" / "windows").glob("*.parquet"))


def test_sft_stream_mixes_in_working_windows_of_save_regions(tmp_path):
    """Save regions go through the whole functional-window path — scraped
    records, the windows precompute, the dataset, the SFT stream — and come
    out as factories that deliver."""
    from denoise import score_world
    from sft import _iter_factories

    files, window_files = _save_corpus(tmp_path)
    worlds = list(itertools.islice(_iter_factories(32, 0, 0, 1, corpus_files=files, window_files=window_files, windows_frac=1.0), 16))
    for state in worlds:
        world = state.long().numpy()[: len(Channel)]
        assert score_world(world, 1.0, 0)["thput_raw"] > 0
        assert np.isin(world[Channel.ENTITIES.value], [str2ent("source").value, str2ent("sink").value]).any()


def test_held_out_saves_never_train(tmp_path, monkeypatch):
    import hashlib

    import corpus.convert as convert

    files, window_files = _save_corpus(tmp_path)
    val_save = hashlib.blake2b(b"save0.zip", digest_size=16).hexdigest()
    monkeypatch.setattr(convert, "held_out", lambda digest, every=100: digest == val_save)
    monkeypatch.setattr(convert, "paste", lambda r, *_: np.full((len(Channel), 1, 1), r.ent.sum()))
    monkeypatch.setattr(convert, "orient", lambda chans, *_: chans)

    def seen(**kw) -> set:
        return {int(w[0, 0, 0]) for w in itertools.islice(convert.functional(files, window_files, 32, random.Random(0), **kw), 200)}

    val, train = seen(val=True), seen()
    assert len(val) == len(train) == 1 and val != train


def test_the_human_eval_scores_the_humans_own_build_at_one(tmp_path, monkeypatch):
    import corpus.convert as convert
    from denoise import score_world
    from factorion import blank_entities
    from sft import build_human_eval_factories
    from training_config import SftArgs

    files, window_files = _save_corpus(tmp_path)
    monkeypatch.setattr(convert, "held_out", lambda digest, every=100: True)
    for _, _, f in build_human_eval_factories(SftArgs(eval_seeds_per_kind=4), files, window_files):
        world = f.world_CWH.numpy()
        assert score_world(world, f.max_throughput, 0)["thput"] == pytest.approx(1)
        blank, _ = blank_entities(f)
        markers = np.isin(world[Channel.ENTITIES.value], [str2ent("source").value, str2ent("sink").value])
        assert (blank.numpy()[Channel.ENTITIES.value] > 0).sum() == markers.sum()
