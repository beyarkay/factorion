"""Tests for the Factorio parity harness (factorion-mod/server/parity.py).

These cover everything that runs *without* a Factorio instance: the world
tensor → parity-spec conversion (the same conventions blueprint.py uses,
which have been verified in-game), the engine-side per-sink expectations,
and the sink-rate comparison logic the report is built on.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_SERVER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "factorion-mod", "server"
)
if _SERVER_DIR not in sys.path:
    sys.path.insert(0, _SERVER_DIR)

import factorion_rs  # noqa: E402
from factorion import (  # noqa: E402
    Channel,
    LessonKind,
    Misc,
    build_factory,
    entities,
    recipes,
    str2ent,
)

from parity import (  # noqa: E402
    SinkComparison,
    _PITCH,
    build_batch_spec,
    compare_sinks,
    expected_sink_rates,
    world_to_parity_spec,
)

SIZE = 11

# Every prototype name the spec may emit must be one the mod (and vanilla
# Factorio) knows how to place. Grown deliberately — a new lesson that
# introduces a new entity must also teach the mod's entity_kind() about it.
KNOWN_PROTOTYPES = {
    "transport-belt",
    "underground-belt",
    "splitter",
    "inserter",
    "long-handed-inserter",
    "assembling-machine-1",
    "stone-furnace",
}


def _specs(num_seeds=3):
    for kind in LessonKind:
        for seed in range(num_seeds):
            factory = build_factory(SIZE, kind, seed=seed)
            if factory is None:
                continue
            world = factory.world_CWH
            spec = world_to_parity_spec(world, run_id=f"{kind.name}-{seed}")
            yield kind, seed, world, spec


def test_spec_entities_are_known_prototypes():
    for kind, seed, _, spec in _specs():
        for e in spec["entities"]:
            assert e["name"] in KNOWN_PROTOTYPES, (
                f"{kind.name} seed={seed}: unknown prototype {e['name']!r}"
            )


def test_spec_directions_are_cardinal_16step():
    for kind, seed, _, spec in _specs():
        for e in spec["entities"]:
            assert e["direction"] in (0, 4, 8, 12), (kind.name, seed, e)
        for s in spec["sources"] + spec["sinks"]:
            assert s["direction"] in (0, 4, 8, 12), (kind.name, seed, s)


def test_spec_positions_inside_grid():
    for kind, seed, _, spec in _specs():
        for e in spec["entities"]:
            assert 0 <= e["tile_x"] < SIZE and 0 <= e["tile_y"] < SIZE
            # Center coordinates sit strictly inside the grid.
            assert 0 < e["x"] < SIZE and 0 < e["y"] < SIZE
        for s in spec["sources"] + spec["sinks"]:
            assert 0 <= s["x"] < SIZE and 0 <= s["y"] < SIZE


def test_spec_covers_every_occupied_tile_once():
    """Every non-source/sink occupied tile is claimed by exactly one spec
    entity's footprint (multi-tile entities emit one entry)."""
    src_id = str2ent("source").value
    snk_id = str2ent("sink").value
    for kind, seed, world, spec in _specs():
        ent_ch = np.asarray(world[Channel.ENTITIES.value])
        occupied = {
            (x, y)
            for x in range(SIZE)
            for y in range(SIZE)
            if int(ent_ch[x, y]) not in (0, src_id, snk_id)
        }
        claimed: list[tuple[int, int]] = []
        for e in spec["entities"]:
            meta = next(
                m for m in entities.values()
                if m.name.replace("_", "-") == e["name"]
            )
            w, h = meta.width, meta.height
            # E/W-facing entities swap footprint dims, mirroring the spec's
            # center computation.
            if e["direction"] in (4, 12) and "inserter" not in e["name"]:
                w, h = h, w
            for dx in range(w):
                for dy in range(h):
                    claimed.append((e["tile_x"] + dx, e["tile_y"] + dy))
        assert sorted(claimed) == sorted(occupied), (kind.name, seed)


def test_spec_underground_types_match_misc_channel():
    for kind, seed, world, spec in _specs():
        misc_ch = np.asarray(world[Channel.MISC.value])
        for e in spec["entities"]:
            if e["name"] != "underground-belt":
                assert "type" not in e
                continue
            misc = int(misc_ch[e["tile_x"], e["tile_y"]])
            expected = {
                Misc.UNDERGROUND_DOWN.value: "input",
                Misc.UNDERGROUND_UP.value: "output",
            }[misc]
            assert e["type"] == expected, (kind.name, seed, e)


def test_spec_inserter_direction_is_flipped():
    """Model direction N (bp 0) must become create_entity direction 8 —
    the same pickup/drop flip blueprint.py applies (verified in-game)."""
    for kind, seed, world, spec in _specs():
        dir_ch = np.asarray(world[Channel.DIRECTION.value])
        for e in spec["entities"]:
            model_dir = int(dir_ch[e["tile_x"], e["tile_y"]])
            bp_dir = model_dir * 4 - 4
            if "inserter" in e["name"]:
                assert e["direction"] == (bp_dir + 8) % 16, (kind.name, seed, e)
            else:
                assert e["direction"] == bp_dir, (kind.name, seed, e)


def test_spec_assembler_recipes_exist():
    for kind, seed, _, spec in _specs():
        for e in spec["entities"]:
            if e["name"] == "assembling-machine-1":
                assert "recipe" in e, (kind.name, seed, e)
                assert e["recipe"].replace("-", "_") in recipes, (
                    f"{kind.name} seed={seed}: recipe {e['recipe']!r} "
                    "not in the engine's recipe table"
                )


def test_spec_sources_and_sinks_carry_items():
    for kind, seed, _, spec in _specs():
        assert len(spec["sources"]) >= 1, (kind.name, seed)
        assert len(spec["sinks"]) >= 1, (kind.name, seed)
        for s in spec["sources"] + spec["sinks"]:
            assert s["item"], (kind.name, seed, s)
            assert "_" not in s["item"], "item names must be hyphenated"


def test_spec_json_is_rcon_safe():
    """The spec rides inside a single-quoted Lua string over RCON."""
    import json

    for kind, seed, _, spec in _specs(num_seeds=2):
        payload = json.dumps(spec, separators=(",", ":"))
        assert "'" not in payload, (kind.name, seed)
        assert "\\" not in payload, (kind.name, seed)


def test_expected_sink_rates_match_spec_sinks():
    """Engine per-sink deliveries and the spec must agree on sink identity:
    same positions, same expected item."""
    for kind, seed, world, spec in _specs():
        expected = expected_sink_rates(world)
        spec_sinks = {(s["x"], s["y"]): s["item"] for s in spec["sinks"]}
        assert set(expected) == set(spec_sinks), (kind.name, seed)
        for pos, (item, rate) in expected.items():
            assert item == spec_sinks[pos], (kind.name, seed, pos)
            assert np.isfinite(rate) and rate >= 0, (kind.name, seed, pos)


def test_expected_sink_rates_aggregate_to_engine_score():
    """py_sink_deliveries must be the un-aggregated form of
    simulate_throughput: power mean (p=0.5) of the rates == score."""
    for kind, seed, world, _ in _specs(num_seeds=2):
        world_whc = (
            np.asarray(world).transpose(1, 2, 0).astype(np.int64)
        )
        world_whc = np.ascontiguousarray(world_whc)
        score, _ = factorion_rs.simulate_throughput(world_whc)
        rates = [r for _, r in expected_sink_rates(world).values()]
        power_mean = (sum(r**0.5 for r in rates) / len(rates)) ** 2
        assert score == pytest.approx(power_mean), (kind.name, seed)


# --------------------------------------------------------------------------- #
# compare_sinks
# --------------------------------------------------------------------------- #

def _measured(x, y, item, rate, all_items=None):
    return {"x": x, "y": y, "item": item, "rate": rate,
            "count": rate * 60, "all_items": all_items or {item: rate * 60}}


def test_compare_sinks_within_tolerance_passes():
    expected = {(3, 5): ("iron-plate", 15.0)}
    out = compare_sinks(
        expected, [_measured(3, 5, "iron-plate", 14.5)],
        rel_tol=0.10, abs_tol=0.25,
    )
    assert [s.passed for s in out] == [True]


def test_compare_sinks_out_of_tolerance_fails():
    expected = {(3, 5): ("iron-plate", 15.0)}
    out = compare_sinks(
        expected, [_measured(3, 5, "iron-plate", 7.5)],
        rel_tol=0.10, abs_tol=0.25,
    )
    assert [s.passed for s in out] == [False]


def test_compare_sinks_abs_tol_dominates_near_zero():
    # 0.02 vs 0.0 items/s: relative error is infinite but the absolute
    # tolerance should absorb it (slow assembler lessons live here).
    expected = {(3, 5): ("fast-underground-belt", 0.0)}
    out = compare_sinks(
        expected, [_measured(3, 5, "fast-underground-belt", 0.02)],
        rel_tol=0.10, abs_tol=0.25,
    )
    assert [s.passed for s in out] == [True]


def test_compare_sinks_missing_and_extra_sinks_fail():
    expected = {(3, 5): ("iron-plate", 15.0)}
    out = compare_sinks(
        expected, [_measured(9, 9, "iron-plate", 15.0)],
        rel_tol=0.10, abs_tol=0.25,
    )
    assert len(out) == 2
    assert not any(s.passed for s in out)
    notes = {s.note for s in out}
    assert any("missing" in n for n in notes)
    assert any("not predicted" in n for n in notes)


def test_compare_sinks_notes_unexpected_items():
    expected = {(3, 5): ("iron-plate", 15.0)}
    out = compare_sinks(
        expected,
        [_measured(3, 5, "iron-plate", 15.0,
                   all_items={"iron-plate": 900, "copper-plate": 12})],
        rel_tol=0.10, abs_tol=0.25,
    )
    assert out[0].passed
    assert "copper-plate" in out[0].note


def test_sink_comparison_is_dataclass_roundtrippable():
    s = SinkComparison(1, 2, "iron-plate", 15.0, 14.9, True)
    assert s.item == "iron-plate" and s.passed


# --------------------------------------------------------------------------- #
# build_batch_spec layout
# --------------------------------------------------------------------------- #

def _batch_worlds(n_kinds=4, seeds=3):
    worlds = []
    for kind in list(LessonKind)[:n_kinds]:
        for seed in range(seeds):
            f = build_factory(SIZE, kind, seed=seed)
            if f is not None:
                worlds.append((f"{kind.name}-s{seed}", f.world_CWH))
    return worlds


def _occupied_tiles(spec):
    occ = set()
    for fac in spec["factories"]:
        ox, oy = fac["offset_x"], fac["offset_y"]
        for e in fac["entities"]:
            occ.add((ox + e["tile_x"], oy + e["tile_y"]))
        for s in fac["sources"] + fac["sinks"]:
            occ.add((ox + s["x"], oy + s["y"]))
    return occ


def test_batch_factories_do_not_overlap():
    spec = build_batch_spec(_batch_worlds())
    occ = list(_occupied_tiles(spec))
    assert len(occ) == len(set(occ)), "two factories share a tile"


def test_batch_substations_never_collide_with_factories():
    spec = build_batch_spec(_batch_worlds())
    occ = _occupied_tiles(spec)
    for sx, sy in spec["substations"]:
        # substation is 2x2 with its top-left at (sx-1, sy-1) .. (sx, sy)
        for dx in (-1, 0):
            for dy in (-1, 0):
                assert (sx + dx, sy + dy) not in occ, (
                    f"substation at ({sx},{sy}) collides with a factory tile"
                )


def test_batch_every_tile_is_powered():
    """Every occupied tile must lie within a substation's 18x18 (Chebyshev
    <= 9) supply area, or inserters/assemblers would sit unpowered and skew
    the measurement."""
    spec = build_batch_spec(_batch_worlds())
    subs = spec["substations"]
    for tx, ty in _occupied_tiles(spec):
        assert any(max(abs(tx - sx), abs(ty - sy)) <= 9 for sx, sy in subs), (
            f"tile ({tx},{ty}) is outside every substation supply area"
        )


def test_batch_offsets_are_on_the_pitch_grid():
    spec = build_batch_spec(_batch_worlds(n_kinds=2, seeds=2))
    for fac in spec["factories"]:
        margin = (_PITCH - fac["grid_size"]) // 2
        assert (fac["offset_x"] - margin) % _PITCH == 0
        assert (fac["offset_y"] - margin) % _PITCH == 0


def test_batch_preserves_all_factories_and_ids():
    worlds = _batch_worlds()
    spec = build_batch_spec(worlds)
    assert len(spec["factories"]) == len(worlds)
    assert [f["run_id"] for f in spec["factories"]] == [w[0] for w in worlds]
