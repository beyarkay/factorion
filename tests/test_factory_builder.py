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
        gym.register(id=env_id, entry_point=FactorioEnv)
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, size, "fbtest")])
    try:
        agent = AgentCNN(envs, chan1=chan, chan2=chan, chan3=chan)
    finally:
        envs.close()
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save(agent.state_dict(), path)
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
            info = fb._swap_model("local", str(ckpt_b), project="x", entity=None)
            assert info["loaded"] is True
            assert info["path"] == str(ckpt_b)
            assert fb._CHECKPOINT_PATH == str(ckpt_b)
        finally:
            ckpt_a.unlink(missing_ok=True)
            ckpt_b.unlink(missing_ok=True)

    def test_swap_local_missing_file(self):
        with pytest.raises(FileNotFoundError):
            fb._swap_model(
                "local", "/tmp/does-not-exist.pt", project="x", entity=None,
            )

    def test_swap_bad_kind(self):
        with pytest.raises(ValueError, match="unknown kind"):
            fb._swap_model("magic", "value", project="x", entity=None)

    def test_swap_empty_value(self):
        with pytest.raises(ValueError, match="empty"):
            fb._swap_model("local", "", project="x", entity=None)


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
            assert info["chan1"] == info["chan2"] == info["chan3"] == 8
        finally:
            path.unlink(missing_ok=True)
