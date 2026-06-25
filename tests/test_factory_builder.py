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

import os
import sys
import tempfile
from pathlib import Path

import gymnasium as gym
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import factory_builder as fb  # noqa: E402
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
        # A non-empty PNG was produced and throughput was computed.
        assert len(out["png"]) > 0
        assert "throughput" in out["info"]
        # Edges are repr()'d (node names embed a literal newline). The
        # source→belt→belt→sink chain must be present.
        flat = " ".join(u + " " + v for u, v in out["edges"])
        assert "stack_inserter" in flat and "bulk_inserter" in flat
        assert flat.count("transport_belt") >= 3


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
    candidate. This is JS embedded in a Python string, so we assert on the
    rendered markup rather than driving a browser — enough to catch the
    wiring being dropped or the candidate field contract drifting."""

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
