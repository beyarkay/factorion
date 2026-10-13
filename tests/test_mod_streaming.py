"""Streaming-placement tests for the Factorio mod server."""

import json
import subprocess
from typing import cast

import numpy as np
import pytest
import torch

from denoise import DenoiserUNet
from factorion import Channel, Footprint, Misc, str2ent
from factorion_server import server as mod_server
from helpers import TINY
from training_config import SftArgs


def _id(name: str) -> int:
    entry = str2ent(name)
    assert entry is not None
    return entry.value


def _action(
    entity: str,
    *,
    xy=(2, 3),
    direction=1,
    item="empty",
    misc=Misc.NONE.value,
):
    return {
        "xy": xy,
        "entity": _id(entity),
        "direction": direction,
        "item": _id(item),
        "misc": misc,
    }


def test_actions_become_factorio_placements():
    belt = mod_server.action_to_placement(_action("transport_belt"))
    assert belt == {
        "name": "transport-belt",
        "tile_x": 2,
        "tile_y": 3,
        "width": 1,
        "height": 1,
        "x": 2.5,
        "y": 3.5,
        "direction": 0,
    }

    inserter = mod_server.action_to_placement(_action("inserter", direction=2))
    assert inserter["direction"] == 12

    assembler = mod_server.action_to_placement(
        _action(
            "assembling_machine_1",
            xy=(4, 5),
            direction=0,
            item="electronic_circuit",
        ),
    )
    assert assembler["name"] == "assembling-machine-1"
    assert (assembler["width"], assembler["height"]) == (3, 3)
    assert (assembler["x"], assembler["y"]) == (5.5, 6.5)
    assert assembler["recipe"] == "electronic-circuit"

    underground = mod_server.action_to_placement(
        _action(
            "underground_belt",
            misc=Misc.UNDERGROUND_DOWN.value,
        ),
    )
    assert underground["type"] == "input"


def test_request_seeds_existing_entities_with_configuration():
    request = _request()
    request["entities"] = [
        {"name": "transport-belt", "x": 1, "y": 2, "direction": 2},
        {
            "name": "assembling-machine-1",
            "x": 4,
            "y": 3,
            "direction": 0,
            "item": "electronic-circuit",
        },
        {
            "name": "underground-belt",
            "x": 8,
            "y": 8,
            "direction": 4,
            "misc": Misc.UNDERGROUND_UP.value,
        },
        {"name": "splitter", "x": 1, "y": 7, "direction": 2},
    ]

    obs = mod_server.request_to_obs(request)
    ent = obs[Channel.ENTITIES.value]
    direction = obs[Channel.DIRECTION.value]
    item = obs[Channel.ITEMS.value]
    misc = obs[Channel.MISC.value]

    assert ent[1, 2] == _id("transport_belt")
    assert direction[1, 2] == 2

    assembler_tiles = [(x, y) for x in range(4, 7) for y in range(3, 6)]
    for x, y in assembler_tiles:
        assert ent[x, y] == _id("assembling_machine_1")
        assert item[x, y] == _id("electronic_circuit")

    assert ent[8, 8] == _id("underground_belt")
    assert direction[8, 8] == 4
    assert misc[8, 8] == Misc.UNDERGROUND_UP.value

    # An east-facing splitter rotates its 2x1 footprint to 1x2.
    assert ent[1, 7] == _id("splitter")
    assert ent[1, 8] == _id("splitter")

    # The player's entities are walls the denoiser builds around; the
    # markers and every other tile stay buildable.
    built = (ent != _id("empty")) & (ent != _id("source")) & (ent != _id("sink"))
    assert (obs[Channel.FOOTPRINT.value] == ~built).all()


def _request():
    return {
        "grid_size": 11,
        "sources": [
            {"x": 0, "y": 5, "direction": 2, "item": "iron_plate"},
        ],
        "sinks": [
            {"x": 10, "y": 5, "direction": 2, "item": "iron_plate"},
        ],
    }


def _model(**config):
    torch.manual_seed(0)
    net = DenoiserUNet(width=16, levels=2, blocks=1, attn_layers=1, attn_heads=4).eval()
    return mod_server.Model(net, "sft-run", SftArgs(gen_steps=3, gen_refine=3, **config))


def test_empty_action_becomes_a_deletion():
    assert mod_server.action_to_placement(_action("empty", direction=0)) == {
        "tile_x": 2, "tile_y": 3, "width": 1, "height": 1,
    }


def test_frame_placements_delete_what_changed_then_build_the_new():
    before = mod_server.request_to_obs(_request() | {"entities": [
        {"name": "transport-belt", "x": 1, "y": 5, "direction": 2},
        {"name": "assembling-machine-1", "x": 4, "y": 3, "direction": 0, "item": "iron-gear-wheel"},
    ]})
    after = before.copy()
    after[Channel.ENTITIES.value, 4:7, 3:6] = _id("empty")
    after[Channel.ITEMS.value, 4:7, 3:6] = _id("empty")
    after[Channel.ENTITIES.value, 5, 4] = _id("inserter")
    after[Channel.DIRECTION.value, 5, 4] = 1

    placements = mod_server.frame_placements(before, after)

    assert placements == [
        {"tile_x": 4, "tile_y": 3, "width": 3, "height": 3},
        mod_server.action_to_placement(_action("inserter", xy=(5, 4), direction=1)),
    ]
    assert mod_server.frame_placements(after, after) == []


def test_denoising_replays_its_seed_and_keeps_markers_and_walls():
    req = _request() | {"entities": [{"name": "transport-belt", "x": 1, "y": 5, "direction": 2}]}
    world = mod_server.request_to_obs(req)
    gen = mod_server.Generation(best_of=3)

    frames, stats = mod_server.denoise_frames(_model(), world, gen, 7, torch.device("cpu"))
    again, _ = mod_server.denoise_frames(_model(), world, gen, 7, torch.device("cpu"))

    assert all(np.array_equal(a, b) for a, b in zip(frames, again))
    assert len(frames) == stats["steps_taken"] + 1
    fixed = world[Channel.FOOTPRINT.value] == 0
    fixed[0, 5] = fixed[10, 5] = True
    for frame in frames:
        # The model's poles may power them.
        assert (frame[: Channel.POWER.value, fixed] == world[: Channel.POWER.value, fixed]).all()
    assert stats["best_of"] == 3 and stats["stop_reason"] in ("converged", "max_steps")


class _AcceptingRcon:
    def __init__(self, response="ok"):
        self.response = response
        self.commands = []

    def exec(self, command):
        self.commands.append(command)
        return self.response


def _payload(command):
    return json.loads(command.split("','")[-1].removesuffix("'))"))


def test_each_denoising_step_streams_as_one_batch(monkeypatch):
    monkeypatch.setattr(mod_server.time, "sleep", lambda _: None)
    rcon = _AcceptingRcon()
    req = _request() | {"request_id": "r", "checkpoint": "c", "seed": 3, "best_of": 2}

    mod_server.handle_request(
        req, lambda _: _model(), cast(mod_server.RconClient, rcon), device=torch.device("cpu"),
    )

    *steps, finish = rcon.commands
    assert steps and all("'place_predictions','r'" in c for c in steps)
    assert all(isinstance(_payload(c), list) and _payload(c) for c in steps)
    summary = _payload(finish)
    assert "'finish_prediction','r'" in finish
    assert summary["seed"] == 3 and summary["best_of"] == 2 and summary["refused_steps"] == 0
    assert summary["stop_reason"] in ("converged", "max_steps")


def test_request_error_still_releases_pending_state():
    rcon = _AcceptingRcon()

    def get_model(_):
        raise FileNotFoundError("no such run")

    with pytest.raises(FileNotFoundError):
        mod_server.handle_request(
            _request() | {"request_id": "r", "checkpoint": "c"}, get_model,
            cast(mod_server.RconClient, rcon), device=torch.device("cpu"),
        )

    assert "finish_prediction" in rcon.commands[-1]
    assert "server_error" in rcon.commands[-1]
    assert "no such run" in rcon.commands[-1]


def test_load_model_reports_grid_size_or_error():
    rcon = _AcceptingRcon()

    def get_model(spec):
        if spec == "missing":
            raise FileNotFoundError("no such run")
        return _model(size=13)

    mod_server._report_model(cast(mod_server.RconClient, rcon), "abc", get_model, 15)
    mod_server._report_model(cast(mod_server.RconClient, rcon), "missing", get_model, 15)

    loaded, failed = rcon.commands
    assert "'model_loaded'" in loaded and '"grid_size":13' in loaded
    assert '"name":"sft-run"' in loaded
    assert "'model_failed'" in failed and "no such run" in failed


def test_local_checkpoint_loads_with_default_schedule(tmp_path):
    path = tmp_path / "sft_checkpoint.pt"
    net = _model().denoiser
    torch.save({"arch": net.arch, "model": net.state_dict()}, path)

    model = mod_server.load_model(str(path), "factorion", None, torch.device("cpu"))

    assert model.name == "sft_checkpoint.pt" and model.config == SftArgs()


def test_cancel_in_game_stops_quietly(monkeypatch):
    monkeypatch.setattr(mod_server.time, "sleep", lambda _: None)
    rcon = _AcceptingRcon("cancelled")

    mod_server.handle_request(
        _request() | {"request_id": "r", "checkpoint": "c"},
        lambda _: _model(), cast(mod_server.RconClient, rcon), device=torch.device("cpu"),
    )

    place, finish = rcon.commands
    assert "place_predictions" in place
    assert "finish_prediction" in finish and "cancelled" in finish


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload

    def iter_content(self, _size):
        yield self.payload


def test_released_checkpoint_downloads_once_and_misses_unreleased(monkeypatch, tmp_path):
    asset = "https://github.com/beyarkay/factorion/releases/download/v0.7.0/abc123.pt"
    releases = [{"assets": [{"name": "abc123.pt", "browser_download_url": asset}]}]
    urls = []

    def fake_get(url, **_):
        urls.append(url)
        return _FakeResponse(b"weights" if url == asset else releases)

    monkeypatch.setattr(mod_server, "CHECKPOINT_CACHE", tmp_path)
    monkeypatch.setattr(mod_server.requests, "get", fake_get)

    path = mod_server.released_checkpoint("abc123")
    assert path == tmp_path / "abc123.pt"
    assert (tmp_path / "abc123.pt").read_bytes() == b"weights"
    assert mod_server.released_checkpoint("abc123") == path
    assert urls == [mod_server.RELEASES_URL, asset]
    assert mod_server.released_checkpoint("zzz999") is None


def test_gui_rcon_enables_rcon_once_then_reads_it(monkeypatch, tmp_path):
    config = tmp_path / "config.ini"
    config.write_text("[path]\nread-data=__PATH__system-read-data__\n\n[other]\n; local-rcon-socket=0.0.0.0:0\n")
    monkeypatch.setattr(mod_server.subprocess, "run", lambda *_, **__: subprocess.CompletedProcess([], 1))

    host, port, password = mod_server.gui_rcon(config)
    assert (host, port) == ("127.0.0.1", 64502)
    assert f"[other]\nlocal-rcon-socket=127.0.0.1:64502\nlocal-rcon-password={password}\n" in config.read_text()
    assert "read-data=__PATH__system-read-data__" in config.read_text()

    config.write_text(config.read_text().replace("64502", "27015"))
    assert mod_server.gui_rcon(config) == ("127.0.0.1", 27015, password)


def test_gui_rcon_refuses_to_edit_a_running_factorios_config(monkeypatch, tmp_path):
    config = tmp_path / "config.ini"
    config.write_text("[other]\n")
    monkeypatch.setattr(mod_server.subprocess, "run", lambda *_, **__: subprocess.CompletedProcess([], 0))

    with pytest.raises(SystemExit, match="Quit Factorio"):
        mod_server.gui_rcon(config)
    assert config.read_text() == "[other]\n"


def test_load_model_explains_a_missing_wandb(monkeypatch):
    def no_wandb(*_):
        raise ModuleNotFoundError("No module named 'wandb'", name="wandb")

    monkeypatch.setattr(mod_server, "released_checkpoint", lambda _: None)
    monkeypatch.setattr(mod_server, "resolve_checkpoint", no_wandb)

    with pytest.raises(RuntimeError, match="uvx --with wandb factorion-server"):
        mod_server.load_model("abc123", "factorion", None, torch.device("cpu"))
