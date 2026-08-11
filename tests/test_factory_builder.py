"""Tests for scripts/factory_builder.py — the interactive UI server.

Covers the model inference path (the parts that aren't covered by
test_sft.py): per-head top-p extraction, the agent cache that lets the
UI resize the grid live, checkpoint loading + cache invalidation on
swap, and the full /predict response schema.

HTTP endpoints aren't exercised here — the underlying functions
(_predict, _model_info, _swap_model) are tested directly, so wiring
them to BaseHTTPRequestHandler would only re-test stdlib socket
behaviour. wandb downloads aren't tested either: they'd require
mocking the wandb client, which is high-effort for low payoff."""

import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
import yaml

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

_NODE = shutil.which("node")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import factorion_rs  # noqa: E402
import factory_builder as fb  # noqa: E402
from factorion import (  # noqa: E402
    LESSON_IS_TRIAL,
    Channel,
    LessonKind,
    entities,
    items,
    render_factory,
)
from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_tiny_checkpoint(size: int = 4, chan: int = 8) -> Path:
    """Build a small AgentCNN at the given size + channel width, save
    its state_dict to a temp .pt, and return the path. The model isn't
    trained — we only care about *shape* compatibility with the inference
    pipeline, not accuracy."""
    env_id = "factorion/FactorioEnv-v0-fbtest"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point="ppo:FactorioEnv")
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, size, "fbtest")])
    try:
        agent = AgentCNN(envs, layers=(chan, chan, chan))
    finally:
        envs.close()
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(agent.state_dict(), path)
    return Path(path)


def _make_compiled_checkpoint(size: int = 4, chan: int = 8) -> Path:
    """Like `_make_tiny_checkpoint`, but every key is prefixed with
    ``_orig_mod.`` to mimic a checkpoint saved by ppo.py *after*
    ``torch.compile`` (which wraps the module and renames params). SFT
    checkpoints are saved uncompiled, PPO ones are not — the builder must
    load both."""
    plain = _make_tiny_checkpoint(size=size, chan=chan)
    try:
        state = torch.load(str(plain), map_location="cpu", weights_only=True)
    finally:
        plain.unlink(missing_ok=True)
    compiled = {f"_orig_mod.{k}": v for k, v in state.items()}
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(compiled, path)
    return Path(path)


@pytest.fixture(autouse=True)
def _reset_fb_state():
    """factory_builder keeps the loaded checkpoint and per-size agents
    as module-level globals. Reset them between tests so one test's
    "no checkpoint loaded" state doesn't leak into the next."""
    fb._CHECKPOINT_STATE = None
    fb._CHECKPOINT_PATH = None
    fb._AGENT_CACHE.clear()
    yield
    fb._CHECKPOINT_STATE = None
    fb._CHECKPOINT_PATH = None
    fb._AGENT_CACHE.clear()


def _empty_grid(size: int) -> list[list[dict]]:
    return [
        [{"entity": "empty", "direction": "NONE", "item": "empty",
          "misc": "NONE", "footprint": "AVAILABLE"} for _ in range(size)]
        for _ in range(size)
    ]


def test_default_wandb_run():
    assert fb.Args().wandb_run == "h76h80yb"


# ── Pure helpers ────────────────────────────────────────────────────────────

class TestTopP:
    def test_top_p_named_includes_until_mass_reached(self):
        # 50/30/15/5 split — top-p=0.95 should include 50+30+15=95, then
        # the 4th item (5%) takes us to 100%, so it's the last one in.
        probs = torch.tensor([0.50, 0.30, 0.15, 0.05])
        names = {0: "a", 1: "b", 2: "c", 3: "d"}
        top, rest = fb._top_p_named(probs, names, top_p=0.95)
        assert [t["name"] for t in top] == ["a", "b", "c"]
        assert top[0]["p"] == pytest.approx(0.50)
        assert rest == pytest.approx(0.05)

    def test_top_p_named_emits_argmax_first(self):
        """Order must be descending so the UI's "top pick" is top[0]."""
        probs = torch.tensor([0.10, 0.70, 0.20])
        names = {0: "a", 1: "b", 2: "c"}
        top, _ = fb._top_p_named(probs, names)
        assert top[0]["name"] == "b"

    def test_top_p_named_concentrated(self):
        """If the top mass already exceeds top_p in one item, only that
        item is returned and rest is whatever's left."""
        probs = torch.tensor([0.99, 0.005, 0.005])
        names = {0: "a", 1: "b", 2: "c"}
        top, rest = fb._top_p_named(probs, names, top_p=0.95)
        assert len(top) == 1 and top[0]["name"] == "a"
        assert rest == pytest.approx(0.01, abs=1e-5)

    def test_tile_top_p_emits_xy(self):
        # H=3, so flat idx 4 -> (x=1, y=1); flat idx 0 -> (x=0, y=0).
        # Probs designed so 4 is top-1 and 0 is top-2.
        probs = torch.tensor([0.30, 0.05, 0.0, 0.0, 0.60, 0.05, 0.0, 0.0, 0.0])
        top, rest = fb._tile_top_p(probs, H=3, top_p=0.85)
        assert top[0] == {"x": 1, "y": 1, "p": pytest.approx(0.60)}
        assert top[1] == {"x": 0, "y": 0, "p": pytest.approx(0.30)}
        assert rest == pytest.approx(0.10, abs=1e-5)


class TestBuildWorld:
    def test_round_trip_entity_value(self):
        """Placing a transport_belt facing EAST writes the entity and
        direction values into the right channels."""
        size = 3
        grid = _empty_grid(size)
        grid[1][2] = {"entity": "transport_belt", "direction": "EAST",
                      "item": "empty", "misc": "NONE", "footprint": "AVAILABLE"}
        world = fb.build_world(grid)
        # fb.items is keyed by Item.value, not by name, so look up via
        # the same name->value map build_world itself constructs.
        name_to_value = {it.name: it.value for it in fb.items.values()}
        # build_world returns WHC; entity channel at (x=2, y=1).
        assert int(world[2, 1, fb.Channel.ENTITIES.value]) == name_to_value["transport_belt"]
        assert int(world[2, 1, fb.Channel.DIRECTION.value]) == fb.Direction.EAST.value

    def test_non_square_raises(self):
        grid = [
            [{"entity": "empty", "direction": "NONE", "item": "empty",
              "misc": "NONE", "footprint": "AVAILABLE"}] * 4,
            [{"entity": "empty", "direction": "NONE", "item": "empty",
              "misc": "NONE", "footprint": "AVAILABLE"}] * 3,
        ]
        with pytest.raises(ValueError, match="square"):
            fb.build_world(grid)

    def test_footprint_unavailable_propagates(self):
        size = 2
        grid = _empty_grid(size)
        grid[0][0]["footprint"] = "UNAVAILABLE"
        world = fb.build_world(grid)
        assert int(world[0, 0, fb.Channel.FOOTPRINT.value]) == \
            fb.Footprint.UNAVAILABLE.value


class TestApplyPrediction:
    """The web UI delegates placement to the rollout's shared mutation path."""

    @staticmethod
    def _prediction(
        *,
        x: int,
        y: int,
        entity: str,
        direction: str,
        item: str = "empty",
        misc: str = "NONE",
    ) -> dict:
        return {
            "x": x,
            "y": y,
            "entity": entity,
            "direction": direction,
            "item": item,
            "misc": misc,
        }

    def test_assembler_prediction_fills_same_3x3_world_footprint_as_rollout(
        self,
    ):
        out = fb._apply_prediction(
            _empty_grid(5),
            self._prediction(
                x=0,
                y=1,
                entity="assembling_machine_1",
                direction="NONE",
                item="electronic_circuit",
            ),
        )

        assert out["applied"] is True
        placed = {
            (x, y)
            for y, row in enumerate(out["grid"])
            for x, cell in enumerate(row)
            if cell["entity"] == "assembling_machine_1"
        }
        assert placed == {(x, y) for x in range(3) for y in range(1, 4)}
        for x, y in placed:
            assert out["grid"][y][x]["item"] == "electronic_circuit"

    @pytest.mark.parametrize(
        ("direction", "expected"),
        [
            ("EAST", {(2, 2), (2, 3)}),
            ("WEST", {(2, 2), (2, 3)}),
            ("NORTH", {(2, 2), (3, 2)}),
            ("SOUTH", {(2, 2), (3, 2)}),
        ],
    )
    def test_splitter_prediction_fills_rotated_two_tile_footprint(
        self, direction, expected
    ):
        out = fb._apply_prediction(
            _empty_grid(5),
            self._prediction(
                x=2, y=2, entity="splitter", direction=direction
            ),
        )
        placed = {
            (x, y)
            for y, row in enumerate(out["grid"])
            for x, cell in enumerate(row)
            if cell["entity"] == "splitter"
        }
        assert out["applied"] is True
        assert placed == expected

    def test_prediction_calls_shared_rollout_placement_function(
        self, monkeypatch
    ):
        called = 0
        shared_apply = fb.apply_placement_action

        def recording_apply(*args, **kwargs):
            nonlocal called
            called += 1
            return shared_apply(*args, **kwargs)

        monkeypatch.setattr(fb, "apply_placement_action", recording_apply)
        fb._apply_prediction(
            _empty_grid(3),
            self._prediction(
                x=1, y=1, entity="transport_belt", direction="EAST"
            ),
        )
        assert called == 1

    def test_invalid_multitile_prediction_is_rejected_atomically(self):
        grid = _empty_grid(5)
        grid[3][2]["entity"] = "transport_belt"
        grid[3][2]["direction"] = "EAST"

        out = fb._apply_prediction(
            grid,
            self._prediction(
                x=2, y=2, entity="splitter", direction="EAST"
            ),
        )

        assert out["applied"] is False
        assert out["invalid_reason"] == "placed_on_existing_entity"
        assert out["grid"] == grid


# ── Graph rendering (Rust-backed) ───────────────────────────────────────────

class TestRenderGraphPng:
    """render_graph_png builds the flow graph via the Rust engine
    (build_graph_nx) and draws it. These cover the migrated visualization
    path end-to-end (issue #178)."""

    def test_empty_grid_reports_nothing_placed(self):
        out = fb.render_graph_png(_empty_grid(4))
        assert out["png"] == ""
        assert out["edges"] == []
        assert "drop something" in out["info"].lower()

    def test_belt_chain_renders_png_and_edges(self):
        grid = _empty_grid(4)
        grid[0][0] = {"entity": "stack_inserter", "direction": "EAST",
                      "item": "copper_cable", "misc": "NONE",
                      "footprint": "AVAILABLE"}
        grid[0][1] = {"entity": "transport_belt", "direction": "EAST",
                      "item": "empty", "misc": "NONE", "footprint": "AVAILABLE"}
        grid[0][2] = {"entity": "transport_belt", "direction": "EAST",
                      "item": "empty", "misc": "NONE", "footprint": "AVAILABLE"}
        grid[0][3] = {"entity": "bulk_inserter", "direction": "EAST",
                      "item": "copper_cable", "misc": "NONE",
                      "footprint": "AVAILABLE"}
        out = fb.render_graph_png(grid)
        assert len(out["png"]) > 0
        assert out["info"] == "6 nodes · 6 edges"
        # Throughput deliberately does NOT ride on this response: it costs
        # ~0.1 ms and this call costs ~1 s of matplotlib, so bundling them
        # made the readout wait on the image.
        assert "thput" not in out
        # The source→belt→belt→sink chain must be present (canonical
        # <char>@x,y node labels).
        flat = " ".join(u + " " + v for u, v in out["edges"])
        assert "S@" in flat and "K@" in flat
        assert flat.count("b@") >= 3


# ── Lesson generation + entity clearing ─────────────────────────────────────

class TestLoadLesson:
    """_load_lesson builds a factory of the chosen lesson kind and,
    when asked, blanks N entity units via the same blank_entities path
    SFT uses to make (partial, completion) training pairs."""

    @staticmethod
    def _placed_count(grid: list[list[dict]]) -> int:
        """Count cells holding any entity (source/sink included)."""
        return sum(c["entity"] != "empty" for row in grid for c in row)

    def test_full_factory_when_zero(self):
        """num_missing_entities=0 (the default) leaves the factory fully
        generated: nothing removed, grid is square at the requested size."""
        out = fb._load_lesson("MOVE_ONE_ITEM", seed=0, size=11)
        assert out["num_removed"] == 0
        assert out["size"] == 11
        assert len(out["grid"]) == 11 and len(out["grid"][0]) == 11
        assert out["next_seed"] == out["used_seed"] + 1
        assert self._placed_count(out["grid"]) > 0

    def test_blanking_removes_requested_units(self):
        """Asking to clear N units removes exactly N (when the factory
        has at least N removable units) — N fewer entities on the grid
        than the fully-generated version at the same seed."""
        full = fb._load_lesson("MOVE_ONE_ITEM", seed=0, size=11)
        partial = fb._load_lesson(
            "MOVE_ONE_ITEM", seed=0, size=11, num_missing_entities=2
        )
        assert partial["num_removed"] == 2
        assert (
            self._placed_count(partial["grid"])
            == self._placed_count(full["grid"]) - 2
        )

    def test_blanking_is_deterministic(self):
        """Same (kind, seed, N) → identical partial grid, so the UI is
        reproducible across repeated clicks."""
        a = fb._load_lesson("MOVE_ONE_ITEM", seed=3, size=11, num_missing_entities=2)
        b = fb._load_lesson("MOVE_ONE_ITEM", seed=3, size=11, num_missing_entities=2)
        assert a["used_seed"] == b["used_seed"]
        assert a["grid"] == b["grid"]

    def test_over_removal_caps_at_removable(self):
        """Requesting more than the factory has removable clears every
        removable unit (capped at total_entities) but keeps the protected
        source/sink, so some entities always survive."""
        full = fb._load_lesson("MOVE_ONE_ITEM", seed=0, size=11)
        out = fb._load_lesson(
            "MOVE_ONE_ITEM", seed=0, size=11, num_missing_entities=999
        )
        assert out["num_removed"] == out["total_entities"]
        # Only the protected (source/sink) entities remain.
        assert (
            self._placed_count(out["grid"])
            == self._placed_count(full["grid"]) - out["total_entities"]
        )
        assert self._placed_count(out["grid"]) > 0

    def test_negative_clamps_to_zero(self):
        """A negative request is clamped to 0 (fully generated) rather
        than raising — the frontend already guards, but the server is
        defensive too."""
        out = fb._load_lesson(
            "MOVE_ONE_ITEM", seed=0, size=11, num_missing_entities=-5
        )
        assert out["num_removed"] == 0


# ── Model loading + cache ───────────────────────────────────────────────────

class TestCheckpointLoading:
    def test_load_checkpoint_populates_state(self):
        path = _make_tiny_checkpoint()
        try:
            fb._load_checkpoint(str(path))
            assert fb._CHECKPOINT_STATE is not None
            assert fb._CHECKPOINT_PATH == str(path)
        finally:
            path.unlink(missing_ok=True)

    def test_load_compiled_checkpoint(self):
        """A PPO checkpoint is saved after torch.compile, so every key
        carries an ``_orig_mod.`` prefix. The builder must strip it:
        otherwise _encoder_arch finds zero conv keys and crashes with
        IndexError (regression for switching to a PPO model in the UI)."""
        path = _make_compiled_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            assert fb._CHECKPOINT_STATE is not None
            # Keys must be normalised — nothing should retain the prefix.
            assert all(
                not k.startswith("_orig_mod.")
                for k in fb._CHECKPOINT_STATE
            )
            info = fb._model_info()
            assert info["loaded"] is True
            assert info["layers"] == [8, 8, 8]
            assert info["kernel_size"] == 3
            # And the agent must build + load the weights without falling
            # back to a fully random net (eot_head kept on size match).
            agent = fb._get_agent(4)
            assert agent is not None
        finally:
            path.unlink(missing_ok=True)

    def test_swap_clears_agent_cache(self):
        """The agent cache must be invalidated on reload — otherwise a
        UI-triggered model swap would silently keep predicting from the
        old weights."""
        ckpt_a = _make_tiny_checkpoint(size=4, chan=8)
        ckpt_b = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(ckpt_a))
            # Prime the cache.
            agent_a = fb._get_agent(4)
            assert 4 in fb._AGENT_CACHE
            # Reload triggers cache invalidation.
            fb._load_checkpoint(str(ckpt_b))
            assert 4 not in fb._AGENT_CACHE, (
                "_load_checkpoint must clear _AGENT_CACHE; otherwise the "
                "next predict() uses stale weights"
            )
            # Building anew yields a fresh object.
            agent_b = fb._get_agent(4)
            assert agent_a is not agent_b
        finally:
            ckpt_a.unlink(missing_ok=True)
            ckpt_b.unlink(missing_ok=True)

    def test_get_agent_caches_per_size(self):
        path = _make_tiny_checkpoint(size=4)
        try:
            fb._load_checkpoint(str(path))
            agent4_first = fb._get_agent(4)
            agent4_second = fb._get_agent(4)
            agent6 = fb._get_agent(6)
            assert agent4_first is agent4_second, "Same size → cached"
            assert agent6 is not agent4_first, "Different size → fresh agent"
        finally:
            path.unlink(missing_ok=True)

    def test_predict_without_checkpoint_raises(self):
        with pytest.raises(RuntimeError, match="no checkpoint"):
            fb._predict(_empty_grid(4))


class TestSwapModel:
    def test_swap_local_path(self):
        ckpt_a = _make_tiny_checkpoint(size=4, chan=8)
        ckpt_b = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(ckpt_a))
            info = fb._swap_model(str(ckpt_b), project="x", entity=None)
            assert info["loaded"] is True
            assert info["path"] == str(ckpt_b)
            assert fb._CHECKPOINT_PATH == str(ckpt_b)
        finally:
            ckpt_a.unlink(missing_ok=True)
            ckpt_b.unlink(missing_ok=True)

    def test_swap_falls_through_to_wandb_when_no_local_file(self, monkeypatch):
        """A value that isn't an existing path is treated as a wandb run id."""
        called_with: dict = {}

        def fake_resolve(run_spec, project, entity):
            called_with["run_spec"] = run_spec
            called_with["project"] = project
            raise RuntimeError("wandb resolver was called as expected")

        monkeypatch.setattr(fb, "_resolve_wandb_checkpoint", fake_resolve)
        with pytest.raises(RuntimeError, match="wandb resolver"):
            fb._swap_model("not-a-path", project="x", entity=None)
        assert called_with == {"run_spec": "not-a-path", "project": "x"}

    def test_swap_empty_value(self):
        with pytest.raises(ValueError, match="empty"):
            fb._swap_model("", project="x", entity=None)


# ── End-to-end /predict schema ──────────────────────────────────────────────

class TestPredictSchema:
    def test_predict_action_returns_compact_placement(self):
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            result = fb._predict_action(_empty_grid(4))
            assert set(result) == {
                "x", "y", "entity", "direction", "item", "misc", "eot_prob",
                # The throughput rides along so the readout can track a
                # held-`a` build without a request of its own.
                "thput", "note",
            }
            assert 0 <= result["x"] < 4
            assert 0 <= result["y"] < 4
        finally:
            path.unlink(missing_ok=True)

    def test_predict_action_matches_detailed_argmax(self):
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            grid = _empty_grid(4)
            compact = fb._predict_action(grid)
            detailed = fb._predict(grid)
            assert compact == {
                key: detailed[key] for key in compact
            }
        finally:
            path.unlink(missing_ok=True)

    def test_predict_returns_full_schema(self):
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            result = fb._predict(_empty_grid(4))
            # Argmax-pick fields drive the dark-blue border + Apply.
            for key in ("x", "y", "entity", "direction", "item", "misc"):
                assert key in result, f"missing argmax field: {key}"
            assert 0 <= result["x"] < 4
            assert 0 <= result["y"] < 4
            # Side-panel top-p distributions per head.
            for head in ("tile", "entity", "direction", "item", "misc"):
                assert f"{head}_top" in result
                assert f"{head}_rest" in result
                top = result[f"{head}_top"]
                rest = result[f"{head}_rest"]
                assert isinstance(top, list) and len(top) >= 1
                # Cumulative mass should account for ~all the probability.
                cum = sum(t["p"] for t in top) + rest
                assert cum == pytest.approx(1.0, abs=1e-4)
            # Ghost overlay candidates list.
            assert "candidates" in result
            for cand in result["candidates"]:
                assert cand["p_tile"] > fb.CANDIDATE_TILE_THRESHOLD
                for key in ("x", "y", "entity", "direction", "item", "misc"):
                    assert key in cand
        finally:
            path.unlink(missing_ok=True)

    def test_predict_returns_eot_prob(self):
        """_predict must surface `eot_prob` in [0, 1] so the UI can show
        the model's "I'm done" probability."""
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            result = fb._predict(_empty_grid(4))
            assert "eot_prob" in result
            assert isinstance(result["eot_prob"], float)
            assert 0.0 <= result["eot_prob"] <= 1.0
        finally:
            path.unlink(missing_ok=True)

    def test_eot_head_loaded_on_size_match_dropped_on_mismatch(self):
        """When the UI grid size matches the checkpoint size, the trained
        eot_head weights must be loaded. When they differ, the head is
        dropped (random-init) so cross-size loading doesn't crash on the
        flat_dim shape mismatch."""
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            assert fb._CHECKPOINT_STATE is not None
            saved_w = fb._CHECKPOINT_STATE["eot_head.1.weight"]

            # Size match → eot_head should match the checkpoint exactly.
            # _get_agent moves the model to _AGENT_DEVICE (mps/cuda on
            # local, cpu on CI); pull weights back to cpu for comparison.
            agent4 = fb._get_agent(4)
            # torch types Sequential[i].weight as `Tensor | Module`, so .cpu() trips ty
            assert torch.equal(agent4.eot_head[1].weight.cpu(), saved_w.cpu()), (  # ty: ignore[invalid-argument-type]
                "eot_head must load when UI size == checkpoint size; "
                "otherwise the UI shows a random-init eot prediction"
            )

            # Size mismatch → eot_head is the model's init, not the saved
            # weights (and shapes differ so they can't be equal anyway).
            agent6 = fb._get_agent(6)
            assert agent6.eot_head[1].weight.shape != saved_w.shape
        finally:
            path.unlink(missing_ok=True)

    def test_predict_argmax_in_tile_top(self):
        """The argmax (x, y) must appear in tile_top[0] — the UI relies
        on this invariant when drawing the dark-blue border."""
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            result = fb._predict(_empty_grid(4))
            tile_top0 = result["tile_top"][0]
            assert (tile_top0["x"], tile_top0["y"]) == (result["x"], result["y"])
        finally:
            path.unlink(missing_ok=True)


class TestRenderIndexApplyWiring:
    """The served HTML must wire clicking a ghosted tile to applying that
    candidate through the server-side rollout placement path."""

    def test_click_applies_visible_candidate(self):
        html = fb.render_index(default_size=11)
        # The shared apply helper exists and the cell click handler calls
        # it for a candidate that's actually drawn here (present + empty),
        # mirroring the ghost-render guard.
        assert "function applyCandidate(" in html
        assert "applyCandidate(cand)" in html
        assert "candByXY[x + ',' + y]" in html
        assert "cand && c.entity === 'empty'" in html

    def test_apply_helper_consumes_candidate_fields(self):
        """applyCandidate destructures exactly the placement fields that
        _predict emits per candidate — keep the two in lockstep."""
        html = fb.render_index(default_size=11)
        assert "const { x, y, entity, direction, item, misc } = cand;" in html
        assert "fetch('/apply_prediction'" in html
        assert "body: JSON.stringify({ grid, prediction:" in html
        assert "grid = data.grid;" in html
        # The same fields _predict guarantees on each candidate (see
        # TestPredictSchema.test_predict_returns_full_schema).
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            result = fb._predict(_empty_grid(4))
        finally:
            path.unlink(missing_ok=True)
        for cand in result["candidates"]:
            for key in ("x", "y", "entity", "direction", "item", "misc"):
                assert key in cand

    def test_hold_a_uses_compact_prediction_loop(self):
        html = fb.render_index(default_size=11)
        assert "function beginApplyKey(" in html
        assert "function endApplyKey(" in html
        assert "function startAutoApply(" in html
        assert "function stopAutoApply(" in html
        assert "detail: 'action'" in html
        assert "document.addEventListener('keyup'" in html


# The page's own script, driven under node against a stub DOM: `fetch` and the
# apply endpoint are replaced, so what runs is the real loop, not a paraphrase.
# `applyCandidate` starts refusing at CAP placements the way a filled grid
# would, which is the only reason a stop-blind loop terminates at all.
_JS_HARNESS = """
import fs from 'node:fs';
const src = fs.readFileSync(process.argv[2], 'utf8');
const mode = process.argv[3];
const stub = () => new Proxy({}, {
  get(t, k) {
    if (k === 'style' || k === 'dataset' || k === 'classList') return stub();
    if (k in t) return t[k];
    return () => stub();
  },
  set(t, k, v) { t[k] = v; return true; },
});
globalThis.document = {
  getElementById: () => stub(), createElement: () => stub(),
  addEventListener: () => {}, querySelectorAll: () => [], body: stub(),
};
globalThis.addEventListener = () => {};
globalThis.window = globalThis;
globalThis.performance = { now: () => 0 };
globalThis.requestAnimationFrame = () => 0;
globalThis.cancelAnimationFrame = () => {};
globalThis.fetch = async () => ({ json: async () => ({}) });

const driver = `
const CAP = 50;
const STOP_AFTER = 3;
let applied = 0;
const stops = ${JSON.stringify(mode)} !== 'never_stops';
requestFastPrediction = async () => ({
  x: 0, y: 0, entity: 'transport-belt', direction: 'NORTH',
  item: 'empty', misc: 'NONE',
  eot_prob: (stops && applied >= STOP_AFTER) ? 0.9 : 0.01,
});
applyCandidate = async () => { applied += 1; return applied < CAP; };
modelLoaded = true;
(async () => {
  if (${JSON.stringify(mode)} === 'tap') {
    prediction = { x: 0, y: 0, entity: 'transport-belt', direction: 'NORTH',
                   item: 'empty', misc: 'NONE', eot_prob: 0.9 };
    beginApplyKey();
    await new Promise((r) => setTimeout(r, 250));
    endApplyKey();
  } else {
    autoApplying = true;
    autoApplyGeneration = 7;
    await runAutoApply(7);
  }
  return { applied, autoApplying, threshold: EOT_STOP_THRESHOLD };
})();
`;
console.log(JSON.stringify(await eval(src + driver)));
"""


@pytest.mark.skipif(_NODE is None, reason="needs node to execute the page's JS")
class TestHoldToApplyRespectsEot:
    """Holding `a` is a rollout, so the model's stop head must end it — the UI
    kept placing entities until the grid refused them, long past eot."""

    def _drive(self, tmp_path: Path, mode: str) -> dict:
        html = fb.render_index(default_size=11)
        js = html.split("<script>", 1)[1].rsplit("</script>", 1)[0]
        (tmp_path / "page.js").write_text(js)
        (tmp_path / "harness.mjs").write_text(_JS_HARNESS)
        assert _NODE is not None  # narrowed by the skipif above
        proc = subprocess.run(
            [_NODE, str(tmp_path / "harness.mjs"), str(tmp_path / "page.js"), mode],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"node failed:\n{proc.stderr}"
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_loop_stops_on_the_step_eot_fires(self, tmp_path):
        out = self._drive(tmp_path, "stops")
        assert out["applied"] == 3, (
            "the loop must apply exactly the placements the model offered "
            "before it declared itself done"
        )
        assert out["autoApplying"] is False

    def test_loop_keeps_building_while_eot_stays_low(self, tmp_path):
        """The stop is the head firing, not the loop being timid: with eot
        pinned low it runs until the grid stops accepting placements."""
        out = self._drive(tmp_path, "never_stops")
        assert out["applied"] == 50

    def test_holding_a_on_a_finished_factory_places_nothing(self, tmp_path):
        """The visible prediction is already suppressed at eot > threshold, so
        the key that consumes it must not apply the hidden placement either."""
        out = self._drive(tmp_path, "tap")
        assert out["applied"] == 0
        assert out["autoApplying"] is False


class TestRenderIndexHelpPopover:
    """The [?] help is a real click-to-toggle popover, not the old native
    `title` tooltip (which browsers rendered unreliably / not at all)."""

    def test_popover_markup_present(self):
        html = fb.render_index(default_size=11)
        assert 'id="help-toggle"' in html
        assert 'id="help-popover"' in html
        assert "function bindHelp(" in html
        assert "bindHelp();" in html

    def test_popover_contains_every_help_line(self):
        html = fb.render_index(default_size=11)
        # Every shortcut line must be reachable in the rendered DOM, joined
        # by <br> inside the popover div.
        for line in fb.HELP_LINES:
            assert line in html
        # The new click-to-apply shortcut is documented.
        assert any("ghost" in line for line in fb.HELP_LINES)


class TestModelInfo:
    def test_unloaded_state(self):
        info = fb._model_info()
        assert info == {"loaded": False}

    def test_loaded_state_exposes_shape(self):
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            info = fb._model_info()
            assert info["loaded"] is True
            assert info["path"] == str(path)
            assert info["layers"] == [8, 8, 8]
        finally:
            path.unlink(missing_ok=True)


class TestBatchRollout:
    """The Scan tab's batched rollout: N blanked factories rebuilt in
    lockstep and streamed back as one event per finished rollout."""

    @staticmethod
    def _scan(payload: dict) -> list[dict]:
        """Drain a scan, asserting every event survives the wire (the
        endpoint serialises each one as a line of NDJSON)."""
        events = list(fb._batch_rollout_request(payload))
        for event in events:
            json.dumps(event)
        return events

    def test_reset_blanks_all_but_protected(self):
        """A scan hands the model a world stripped down to the lesson's
        protected tiles — the source, sink and reserved cells — which is
        the "rebuild from nothing" test the Scan tab exists to run."""
        env = FactorioEnv(size=11, idx=0)
        obs, used_seed = fb._reset_rollout_env(
            env, LessonKind.MOVE_ONE_ITEM, seed=0, num_missing_entities=None
        )
        assert used_seed == 0
        blanked = int((obs[Channel.ENTITIES.value] != 0).sum())
        solved = int((env._solved_world_CWH[Channel.ENTITIES.value] != 0).sum())
        assert 0 < blanked < solved

    def test_streams_a_result_per_seed(self):
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            events = self._scan(
                {"kind": "MOVE_ONE_ITEM", "count": 3, "seed": 0, "size": 5}
            )
        finally:
            path.unlink(missing_ok=True)
        assert events[0] == {"type": "start", "n": 3}
        assert events[-1]["type"] == "done"
        results = [e for e in events if e["type"] == "result"]
        assert sorted(r["seed"] for r in results) == [0, 1, 2]
        assert all(r["kind"] == "MOVE_ONE_ITEM" for r in results)
        assert any(e["type"] == "progress" for e in events)

    def test_result_carries_both_grids(self):
        """Each result ships the factory the model built *and* the
        generator's own solution, so the gallery can show them side by
        side without a second round trip."""
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            events = self._scan(
                {"kind": "MOVE_ONE_ITEM", "count": 1, "seed": 7, "size": 11}
            )
        finally:
            path.unlink(missing_ok=True)
        result = next(e for e in events if e["type"] == "result")
        assert result["stopped_by"] in ("eot", "max_steps")
        for grid in (result["grid"], result["solved_grid"]):
            assert len(grid) == 11 and all(len(row) == 11 for row in grid)
        assert any(
            c["entity"] != "empty" for row in result["solved_grid"] for c in row
        )

    def test_every_kind_cycles_kinds_before_seeds(self):
        """One rollout per LessonKind at the same seed, so a scan sized to
        the kind count is a breadth-first sweep of setups."""
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            events = self._scan({
                "kind": fb.ALL_KINDS_SENTINEL,
                "count": len(list(LessonKind)),
                "seed": 4,
                "size": 11,
            })
        finally:
            path.unlink(missing_ok=True)
        results = [e for e in events if e["type"] == "result"]
        assert {r["kind"] for r in results} == {k.name for k in LessonKind}
        assert all(r["seed"] == 4 for r in results)

    def test_runs_more_rollouts_than_one_batch(self, monkeypatch):
        """A count above the lockstep batch width runs as back-to-back
        groups instead of being truncated to fit a single batch."""
        monkeypatch.setattr(fb, "ROLLOUT_BATCH_SIZE", 2)
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            events = self._scan(
                {"kind": "MOVE_ONE_ITEM", "count": 5, "seed": 0, "size": 5}
            )
        finally:
            path.unlink(missing_ok=True)
        results = [e for e in events if e["type"] == "result"]
        assert sorted(r["seed"] for r in results) == [0, 1, 2, 3, 4]
        # Indices stay unique across groups — the gallery sorts on them.
        assert sorted(r["index"] for r in results) == [0, 1, 2, 3, 4]

    def test_count_is_never_silently_capped(self, monkeypatch):
        """The requested count is what runs. A cap that quietly trimmed the
        scan would read as "that's all there was to find"."""
        monkeypatch.setattr(fb, "ROLLOUT_BATCH_SIZE", 2)
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            events = self._scan(
                {"kind": "MOVE_ONE_ITEM", "count": 7, "seed": 0, "size": 5}
            )
        finally:
            path.unlink(missing_ok=True)
        assert events[0] == {"type": "start", "n": 7}
        assert len([e for e in events if e["type"] == "result"]) == 7

    def test_bad_kind_yields_an_error_event(self):
        """The response headers are already sent by the time the scan
        runs, so failures have to travel in-band rather than as an
        exception."""
        events = self._scan({"kind": "NOT_A_LESSON", "count": 1, "size": 11})
        assert events[-1]["type"] == "error"
        assert "NOT_A_LESSON" in events[-1]["error"]

    def test_missing_checkpoint_yields_an_error_event(self):
        events = self._scan({"kind": "MOVE_ONE_ITEM", "count": 1, "size": 11})
        assert events[-1]["type"] == "error"
        assert "checkpoint" in events[-1]["error"]


class TestRenderIndexScanTab:
    """The served page must expose the scan controls and consume the
    endpoint's NDJSON stream."""

    def test_scan_tab_controls_exist(self):
        html = fb.render_index(default_size=11)
        assert 'data-tab="scan"' in html and 'data-tab="build"' in html
        for element_id in (
            "scan-kind", "scan-count", "scan-seed", "scan-clear", "scan-mask",
            "scan-ref", "scan-sort", "scan-run", "scan-stop", "scan-results",
            "scan-summary", "scan-stats", "scan-clear-results",
        ):
            assert f'id="{element_id}"' in html, element_id
        assert f'<option value="{fb.ALL_KINDS_SENTINEL}">' in html
        # Worst-first by default: a scan is run to find the failures.
        assert '<option value="worst" selected>' in html

    def test_client_streams_from_the_endpoint_the_server_routes(self):
        html = fb.render_index(default_size=11)
        assert "fetch('/batch_rollout'" in html
        assert "resp.body.getReader()" in html
        assert "signal: scanAbort.signal" in html
        # A typo'd path here fails silently as a 404, so pin the client's
        # URL to the handler's route list.
        assert "/batch_rollout" in inspect.getsource(fb.Handler.do_POST)

    def test_result_fields_the_page_reads_are_all_emitted(self):
        """Every `r.<field>` the scan JS reads must exist on a real result
        event. Asserting against a hardcoded list on either side would keep
        passing while the two drifted apart, so this derives one side from
        the served page and the other from an actual rollout."""
        html = fb.render_index(default_size=11)
        read_fields = set(re.findall(r"\br\.([a-z_]+)", html))
        assert read_fields, "no r.<field> reads found in the served page"
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            events = TestBatchRollout._scan(
                {"kind": "MOVE_ONE_ITEM", "count": 1, "seed": 1, "size": 11}
            )
        finally:
            path.unlink(missing_ok=True)
        result = next(e for e in events if e["type"] == "result")
        assert read_fields <= set(result), read_fields - set(result)


class TestThroughputReadout:
    """The number above the grid is what a newbie reads to tell a working
    factory from a broken one, so it must be right and must not wait on the
    graph image."""

    def test_reports_the_engines_number(self):
        factory, _seed = fb._build_with_retry(LessonKind.SPLITTER_SPLIT, 11, 3)
        grid = fb.world_CWH_to_grid(factory.world_CWH)
        out = fb._throughput(fb.build_world(grid))
        want, unreachable = factorion_rs.simulate_throughput(
            fb.build_world(grid).numpy().astype(np.int64)
        )
        assert out == {"thput": want, "unreachable": unreachable}
        assert out["thput"] > 0  # a solved lesson flows

    def test_empty_grid_is_neutral_not_blocked(self):
        """Zero and "nothing here yet" look identical in a number but not to
        a user: a blank grid must not be flagged as a broken factory."""
        out = fb._throughput(fb.build_world(_empty_grid(5)))
        assert out["thput"] is None and out["note"]

    def test_disconnected_factory_reads_as_blocked(self):
        grid = _empty_grid(5)
        grid[0][0] = {"entity": "stack_inserter", "direction": "EAST",
                      "item": "copper_cable", "misc": "NONE",
                      "footprint": "AVAILABLE"}
        grid[4][4] = {"entity": "bulk_inserter", "direction": "EAST",
                      "item": "copper_cable", "misc": "NONE",
                      "footprint": "AVAILABLE"}
        assert fb._throughput(fb.build_world(grid))["thput"] == 0

    def test_rides_on_both_prediction_paths(self):
        """Both are the responses that already built the world, so neither
        should make the readout pay for a request of its own."""
        path = _make_tiny_checkpoint(size=4, chan=8)
        try:
            fb._load_checkpoint(str(path))
            grid = _empty_grid(4)
            assert "thput" in fb._predict(grid)
            assert "thput" in fb._predict_action(grid)
        finally:
            path.unlink(missing_ok=True)


class TestRenderIndexThroughput:
    def test_readout_sits_above_the_grid(self):
        html = fb.render_index(default_size=11)
        assert html.index('id="thput"') < html.index('id="grid-host"')
        # Both verdicts, and the spinner for the wait in between. The icons
        # are inline SVG so a viewer without the right font still gets one.
        assert f"const OK_ICON = '{fb.OK_ICON}'" in html
        assert f"const BAD_ICON = '{fb.BAD_ICON}'" in html
        assert "<svg" in fb.OK_ICON and "<svg" in fb.BAD_ICON
        assert "factory throughput:" in html
        assert "calculating…" in html
        # The card itself carries the verdict, not just the icon.
        assert ".thput.ok" in html and ".thput.bad" in html
        assert "'thput ok' : 'thput bad'" in html
        assert 'class="spinner"' in html
        assert "items per second" in html

    def test_readout_never_queues_behind_the_graph_image(self):
        """The whole point of the split: the graph call is ~1 s of matplotlib
        and the server answers one request at a time, so a readout that waited
        on it would be ~10000x slower than the number costs to compute."""
        html = fb.render_index(default_size=11)
        assert "setTimeout(computePrediction, 25)" in html
        assert "setTimeout(computeGraph, 1000)" in html
        # computeGraph must not be the thing that feeds the readout.
        graph_fn = html.split("async function computeGraph(")[1].split("\n}")[0]
        assert "showThput" not in graph_fn

    def test_held_apply_updates_the_readout_as_it_builds(self):
        html = fb.render_index(default_size=11)
        loop = html.split("async function runAutoApply(")[1].split("\n}")[0]
        assert "showThput(action)" in loop

    def test_falls_back_to_its_own_endpoint_with_no_model(self):
        """Without a checkpoint nothing else asks the server anything, and
        hand-building a factory is exactly when the readout matters most."""
        html = fb.render_index(default_size=11)
        assert "fetch('/throughput'" in html
        assert "/throughput" in inspect.getsource(fb.Handler.do_POST)
        # The body up to the first `return;` is the no-model early exit.
        fn = html.split("async function computePrediction(")[1].split("\n}")[0]
        early_exit = fn.split("return;")[0]
        assert "!modelLoaded" in early_exit
        assert "computeThroughput()" in early_exit


class TestFactoryYaml:
    """`factory_yaml` must emit a document the Rust fixture parser accepts
    (`factorion_rs/src/textual.rs`), asserting what the engine computes for
    that exact world — otherwise a pasted fixture fails the moment it lands
    in `factorion_rs/tests/factories/`."""

    ASSEMBLER_LESSON = LessonKind.MEMORISE_2_INGREDIENT_RECIPES

    @staticmethod
    def _grid(kind: LessonKind, size: int = 11, seed: int = 3):
        factory, _seed = fb._build_with_retry(kind, size, seed)
        return factory, fb.world_CWH_to_grid(factory.world_CWH)

    def test_document_matches_the_fixture_header(self):
        factory, grid = self._grid(self.ASSEMBLER_LESSON)
        text = fb.factory_yaml(grid)
        doc = yaml.safe_load(text)
        # Style, not just value: a quoted `factory:` scalar or block-style
        # bindings parse the same but read nothing like the fixtures.
        assert "factory: |\n" in text
        assert "\n- {x: " in text
        # `Header` is deny_unknown_fields, so a stray key is a parse error.
        assert set(doc) <= {"description", "items", "throughput", "factory"}
        assert all(set(b) == {"x", "y", "item"} for b in doc["items"])
        assert all(set(t) == {"item", "per_second"} for t in doc["throughput"])
        # The grid is the canonical renderer's, not a second implementation.
        assert doc["factory"].rstrip("\n") == render_factory(factory.world_CWH)

    def test_multi_tile_entity_binds_once_at_its_anchor(self):
        """`items:` resolves a coordinate to the whole footprint, so a 3x3
        assembler is one line — and it must be the anchor tile, which is what
        `build_graph` reads the recipe from."""
        _factory, grid = self._grid(self.ASSEMBLER_LESSON)
        doc = yaml.safe_load(fb.factory_yaml(grid))
        tiles = {
            (x, y)
            for y, row in enumerate(grid)
            for x, cell in enumerate(row)
            if cell["entity"] == "assembling_machine_1"
        }
        assert len(tiles) == 9
        anchor = min(tiles, key=lambda p: (p[1], p[0]))
        singles = {
            (x, y)
            for y, row in enumerate(grid)
            for x, cell in enumerate(row)
            if cell["item"] != "empty" and (x, y) not in tiles
        }
        assert {(b["x"], b["y"]) for b in doc["items"]} == singles | {anchor}
        for b in doc["items"]:
            assert grid[b["y"]][b["x"]]["item"] == b["item"]

    def test_throughput_is_the_engines_own_deliveries(self):
        factory, grid = self._grid(LessonKind.SPLITTER_SPLIT)
        doc = yaml.safe_load(fb.factory_yaml(grid))
        world_WHC = np.ascontiguousarray(
            np.transpose(np.asarray(factory.world_CWH), (1, 2, 0)).astype(np.int64)
        )
        # Exact equality, not a tolerance: the emitted rates have to survive
        # the round trip through text.
        assert sorted(
            (t["item"], t["per_second"]) for t in doc["throughput"]
        ) == sorted(
            (item, rate)
            for _x, _y, item, rate in factorion_rs.py_sink_deliveries(world_WHC)
        )

    def test_description_carries_the_provenance_it_can_get(self):
        _factory, grid = self._grid(LessonKind.SPLITTER_SPLIT)
        doc = yaml.safe_load(fb.factory_yaml(grid, "SPLITTER_SPLIT seed 3"))
        assert "SPLITTER_SPLIT seed 3" in doc["description"]
        # This checkout is a git repo, so the sha is a fact the note must
        # carry rather than an optional nicety.
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(fb.__file__).parent.parent,
        ).stdout.strip()
        assert head and head in doc["description"]

    def test_provenance_degrades_instead_of_failing(self, monkeypatch, tmp_path):
        """Plenty of environments have no git, no repo, or no useful clock.
        Each missing part drops out on its own; only when nothing at all is
        known does the key disappear."""
        assert fb._git_commit(tmp_path) is None  # a real directory, not a repo
        monkeypatch.setattr(fb, "_git_commit", lambda *a, **k: None)
        _factory, grid = self._grid(LessonKind.SPLITTER_SPLIT)
        doc = yaml.safe_load(fb.factory_yaml(grid, "SPLITTER_SPLIT seed 3"))
        assert "commit" not in doc["description"]
        assert "SPLITTER_SPLIT seed 3" in doc["description"]

        monkeypatch.setattr(fb, "_provenance", lambda source: None)
        assert "description" not in yaml.safe_load(fb.factory_yaml(grid))

    def test_untagged_sink_is_left_out(self):
        """A throughput entry requires an `item:`, so a sink with nothing
        bound has no fixture form. Emitting one anyway would make the whole
        document unparseable, taking the rest of the factory down with it."""
        blank = {
            "entity": "empty", "direction": "NONE", "item": "empty",
            "misc": "NONE", "footprint": "AVAILABLE",
        }
        grid = [[dict(blank) for _ in range(4)] for _ in range(4)]
        for x, entity in enumerate(
            ["stack_inserter", "transport_belt", "bulk_inserter"]
        ):
            grid[0][x] |= {"entity": entity, "direction": "EAST"}
        assert "throughput" not in yaml.safe_load(fb.factory_yaml(grid))

    def test_every_lesson_round_trips(self):
        """The generators are the main source of factories to copy, so every
        kind must serialise to a parseable document with something to assert
        — a fixture declaring neither `throughput:` nor `graph:` is rejected
        by the sweep in `textual.rs`."""
        for kind in LessonKind:
            factory, _seed = fb._build_with_retry(kind, 11, 0)
            doc = yaml.safe_load(
                fb.factory_yaml(fb.world_CWH_to_grid(factory.world_CWH))
            )
            assert doc["factory"].rstrip("\n") == render_factory(factory.world_CWH)
            # Trials place only markers, so they have no sink deliveries to
            # assert and are not fixture material.
            if not LESSON_IS_TRIAL[kind]:
                assert doc["throughput"], kind.name


class TestRenderIndexCopyYaml:
    """The copy button has to reach the endpoint the handler routes, on both
    tabs."""

    def test_both_tabs_offer_the_button(self):
        html = fb.render_index(default_size=11)
        assert 'id="copy-yaml"' in html                  # by the graph readout
        assert html.count('class="copy-yaml"') == 2      # + one per scan card
        # The button is an icon, so its tooltip is the only thing naming it.
        assert html.count('title="Copy this factory as a YAML test fixture"') == 2
        assert "fetch('/factory_yaml'" in html
        # A typo'd path fails silently as a 404, so pin it to the route list.
        assert "/factory_yaml" in inspect.getsource(fb.Handler.do_POST)

    def test_card_button_does_not_also_open_the_factory(self):
        """A scan card adopts its grid on click; the button sits inside it."""
        html = fb.render_index(default_size=11)
        assert "ev.stopPropagation();" in html

    def test_page_sends_provenance_the_endpoint_reads(self):
        """The lesson/seed is the page's to know, so a rename on either side
        would otherwise silently start copying fixtures with no `description:`."""
        html = fb.render_index(default_size=11)
        assert "JSON.stringify({ grid: g, source })" in html
        assert "source" in inspect.signature(fb.factory_yaml).parameters
        assert 'payload.get("source")' in inspect.getsource(fb.Handler.do_POST)


class TestIconCoverage:
    """Keep the renderer's silently missing icons from reaching the UI."""

    def test_every_entity_has_an_icon(self):
        missing = [
            entity.name
            for entity in entities.values()
            if not fb._icon_b64(entity.name)
        ]
        assert not missing, f"entities missing factorio-icons/*.png: {missing}"

    def test_every_item_has_an_icon(self):
        missing = [
            item.name for item in items.values() if not fb._icon_b64(item.name)
        ]
        assert not missing, f"items missing factorio-icons/*.png: {missing}"
