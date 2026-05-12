"""Tests for the Items + Recipes single-source-of-truth migration.

Items and recipes are defined in factorion_rs/src/types.rs and exposed to
Python via factorion_rs.py_items() and factorion_rs.py_recipes(). The
Python `items` and `recipes` dicts in factorion.py are built from these
bindings at module load.

These tests cover:
  1. Items: known variants are present and integer values match Rust.
  2. Recipes: end-to-end value correctness for known recipes.
  3. Parity between the Rust binding and the Python view.

Recipe quantities are the canonical Factorio wiki values (per craft).
"""

import factorion_rs

from helpers import items, recipes


# ── Items SOT ────────────────────────────────────────────────────────────────


class TestPyItemsBinding:
    """py_items() returns {int_value: {name, is_placeable, width, height, flow}}."""

    def test_known_items_present(self):
        rs = factorion_rs.py_items()
        names = {props["name"] for props in rs.values()}
        for expected in [
            "copper_cable",
            "copper_plate",
            "iron_plate",
            "electronic_circuit",
            "iron_gear_wheel",
            "transport_belt",
            "inserter",
            "assembling_machine_1",
            "underground_belt",
            "splitter",
            "stack_inserter",  # Source
            "bulk_inserter",   # Sink
        ]:
            assert expected in names, f"missing item {expected}"
        # No "empty" — absence of an item is encoded as channel value 0,
        # not as an Item variant.
        assert "empty" not in names
        assert 0 not in rs

    def test_integer_values_are_stable(self):
        """The integer encoding for the foundational items (1..10) must
        not silently change — world tensors on disk and trained models
        depend on it. Sink/Source are pinned to the LAST two ids
        dynamically so adding intermediate items doesn't break them
        (see TestSourceSinkAreLastTwoIds)."""
        rs = factorion_rs.py_items()
        assert rs[1]["name"] == "transport_belt"
        assert rs[2]["name"] == "inserter"
        assert rs[3]["name"] == "assembling_machine_1"
        assert rs[4]["name"] == "underground_belt"
        assert rs[5]["name"] == "splitter"
        assert rs[6]["name"] == "copper_cable"
        assert rs[7]["name"] == "copper_plate"
        assert rs[8]["name"] == "iron_plate"
        assert rs[9]["name"] == "electronic_circuit"
        assert rs[10]["name"] == "iron_gear_wheel"

    def test_placeability_is_correct(self):
        rs = factorion_rs.py_items()
        ids = sorted(rs.keys())
        sink_id, source_id = ids[-2], ids[-1]
        # Placeable: agent-buildable (1..5) + env-spawned source/sink (last two)
        for value in [1, 2, 3, 4, 5, sink_id, source_id]:
            assert rs[value]["is_placeable"] is True, f"id {value} should be placeable"
        # Everything else (non-placeable items) — recipe ingredients,
        # raw materials, and non-modeled buildings exposed only as items.
        for value in ids:
            if value in {1, 2, 3, 4, 5, sink_id, source_id}:
                continue
            assert rs[value]["is_placeable"] is False, (
                f"id {value} ({rs[value]['name']!r}) should not be placeable"
            )

    def test_assembler_size_is_3x3(self):
        rs = factorion_rs.py_items()
        am1 = rs[3]
        assert am1["width"] == 3
        assert am1["height"] == 3

    def test_splitter_size_is_2x1(self):
        rs = factorion_rs.py_items()
        sp = rs[5]
        assert sp["width"] == 2
        assert sp["height"] == 1


class TestSourceSinkAreLastTwoIds:
    """LOAD-BEARING INVARIANT — DO NOT REMOVE.

    Source and Sink MUST be the last two ids in the unified Item enum.
    The PPO policy in ppo.py sizes its entity head to ``len(entities) - 2``
    to structurally exclude env-spawned source/sink from agent placement
    (see ppo.py line ~780). If anyone reorders the enum and breaks this
    invariant, the head will sample Source/Sink as agent actions, the
    env will reject them, and training will silently regress.

    The Rust mirror is `test_source_and_sink_are_last_two_ids` in
    factorion_rs/src/types.rs — keep both. If you genuinely need to
    remove this protection, you must also rewrite AgentCNN.__init__'s
    entity-head sizing in ppo.py.
    """

    def test_source_is_last_id(self):
        rs = factorion_rs.py_items()
        max_id = max(rs.keys())
        assert rs[max_id]["name"] == "stack_inserter", (
            f"the highest item id ({max_id}) must be Source/stack_inserter, "
            f"got {rs[max_id]['name']!r}. ppo.py:780 depends on this."
        )

    def test_sink_is_second_last_id(self):
        rs = factorion_rs.py_items()
        ids = sorted(rs.keys())
        second_last_id = ids[-2]
        assert rs[second_last_id]["name"] == "bulk_inserter", (
            f"the second-highest item id ({second_last_id}) must be Sink/"
            f"bulk_inserter, got {rs[second_last_id]['name']!r}. "
            f"ppo.py:780 depends on this."
        )

    def test_excluding_last_two_gives_no_source_or_sink(self):
        """Simulate ppo.py's `len(entities) - 2` slice and verify the
        remaining id range contains no source or sink."""
        rs = factorion_rs.py_items()
        ids = sorted(rs.keys())
        # The agent's entity head sees ids[0:len(ids)-2] (the last two are
        # excluded). After unification we have an extra synthetic 0=empty
        # in the Python dict, but this slice is a sanity check that the
        # `entities[:len-2]` pattern keeps the env-only entities out.
        agent_visible = ids[:-2]
        names = {rs[i]["name"] if i in rs else "empty" for i in agent_visible}
        assert "stack_inserter" not in names
        assert "bulk_inserter" not in names


class TestPythonItemsDict:
    def test_dict_has_synthetic_empty_sentinel(self):
        # The Python `items` dict adds a synthetic 0 → "empty" entry on
        # top of the Rust source for tensor-decode convenience. Rust
        # itself has no Empty variant.
        assert 0 in items
        assert items[0].name == "empty"
        assert items[0].is_placeable is False

    def test_dict_built_from_rust(self):
        # All Rust-side keys are present in the Python dict.
        rs = factorion_rs.py_items()
        for value in rs:
            assert value in items

    def test_each_item_has_matching_props(self):
        rs = factorion_rs.py_items()
        for value, props in rs.items():
            item = items[value]
            assert item.name == props["name"]
            assert item.value == value
            assert item.is_placeable == props["is_placeable"]
            assert item.width == props["width"]
            assert item.height == props["height"]
            assert item.flow == props["flow"]


# ── Recipes SOT ──────────────────────────────────────────────────────────────


class TestPyRecipesBinding:
    """Canonical wiki recipe values (per craft, not per second)."""

    def test_copper_cable(self):
        cc = factorion_rs.py_recipes()["copper_cable"]
        assert cc["consumes"] == {"copper_plate": 1.0}
        assert cc["produces"] == {"copper_cable": 2.0}

    def test_electronic_circuit(self):
        ec = factorion_rs.py_recipes()["electronic_circuit"]
        assert ec["consumes"] == {"copper_cable": 3.0, "iron_plate": 1.0}
        assert ec["produces"] == {"electronic_circuit": 1.0}

    def test_iron_gear_wheel(self):
        igw = factorion_rs.py_recipes()["iron_gear_wheel"]
        assert igw["consumes"] == {"iron_plate": 2.0}
        assert igw["produces"] == {"iron_gear_wheel": 1.0}

    def test_transport_belt(self):
        tb = factorion_rs.py_recipes()["transport_belt"]
        assert tb["consumes"] == {"iron_gear_wheel": 1.0, "iron_plate": 1.0}
        assert tb["produces"] == {"transport_belt": 2.0}

    def test_inserter(self):
        ins = factorion_rs.py_recipes()["inserter"]
        assert ins["consumes"] == {
            "electronic_circuit": 1.0,
            "iron_gear_wheel": 1.0,
            "iron_plate": 1.0,
        }
        assert ins["produces"] == {"inserter": 1.0}

    def test_assembling_machine_1(self):
        am1 = factorion_rs.py_recipes()["assembling_machine_1"]
        assert am1["consumes"] == {
            "electronic_circuit": 3.0,
            "iron_gear_wheel": 5.0,
            "iron_plate": 9.0,
        }
        assert am1["produces"] == {"assembling_machine_1": 1.0}

    def test_every_recipe_has_crafting_time(self):
        rs = factorion_rs.py_recipes()
        for name, data in rs.items():
            assert "crafting_time" in data, f"{name} missing crafting_time"
            assert data["crafting_time"] > 0, (
                f"{name}: crafting_time must be positive, got {data['crafting_time']}"
            )

    def test_canonical_crafting_times(self):
        rs = factorion_rs.py_recipes()
        # Spot-check a handful of canonical wiki values.
        assert rs["copper_cable"]["crafting_time"] == 0.5
        assert rs["electronic_circuit"]["crafting_time"] == 0.5
        assert rs["steel_plate"]["crafting_time"] == 16.0
        assert rs["advanced_circuit"]["crafting_time"] == 6.0
        assert rs["engine_unit"]["crafting_time"] == 10.0


class TestPythonRecipesDict:
    def test_recipe_is_dataclass_with_attributes(self):
        ec = recipes["electronic_circuit"]
        assert hasattr(ec, "consumes")
        assert hasattr(ec, "produces")
        assert hasattr(ec, "crafting_time")

    def test_canonical_electronic_circuit(self):
        ec = recipes["electronic_circuit"]
        assert ec.consumes == {"copper_cable": 3.0, "iron_plate": 1.0}
        assert ec.produces == {"electronic_circuit": 1.0}


# ── Parity ───────────────────────────────────────────────────────────────────


class TestRustPythonRecipeParity:
    def test_same_recipe_names(self):
        assert set(recipes.keys()) == set(factorion_rs.py_recipes().keys())

    def test_same_consumes_and_produces(self):
        rs = factorion_rs.py_recipes()
        for name, py_recipe in recipes.items():
            assert py_recipe.consumes == rs[name]["consumes"], (
                f"{name}.consumes mismatch"
            )
            assert py_recipe.produces == rs[name]["produces"], (
                f"{name}.produces mismatch"
            )
            assert py_recipe.crafting_time == rs[name]["crafting_time"], (
                f"{name}.crafting_time mismatch"
            )
