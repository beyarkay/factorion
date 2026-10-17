"""Streaming-placement tests for the Factorio mod server."""

from typing import cast

import numpy as np
import pytest
import torch

from denoise import DenoiserUNet
from factorion import Channel, Footprint, Misc, str2ent
from factorion_server import server as mod_server
from helpers import TINY


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

    # Existing entities are walls the model builds around; the markers aren't.
    walls = obs[Channel.FOOTPRINT.value] == Footprint.UNAVAILABLE.value
    assert walls.sum() == 1 + 9 + 1 + 2
    assert not walls[0, 5] and not walls[10, 5]


class _AcceptingRcon:
    def __init__(self):
        self.commands = []

    def exec(self, command):
        self.commands.append(command)
        return "ok"


def test_stream_placement_waits_after_factorio_accepts(monkeypatch):
    sleeps = []
    monkeypatch.setattr(mod_server.time, "sleep", sleeps.append)

    accepted = mod_server._stream_placement(
        cast(mod_server.RconClient, _AcceptingRcon()),
        "request-1",
        _action("transport_belt"),
        placement_delay_s=0.01,
    )

    assert accepted is True
    assert sleeps == [0.01]


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


def test_request_error_still_releases_pending_state():
    req = _request() | {"request_id": "request-1", "checkpoint": "ckpt"}
    rcon = _AcceptingRcon()

    def broken(_):
        raise RuntimeError("no such run")

    with pytest.raises(RuntimeError):
        mod_server.handle_request(
            req, broken, cast(mod_server.RconClient, rcon), device=torch.device("cpu"),
        )

    assert "finish_prediction" in rcon.commands[-1]
    assert "server_error" in rcon.commands[-1]
    assert "no such run" in rcon.commands[-1]


def test_load_model_reports_grid_size_or_error():
    rcon = _AcceptingRcon()

    def get_model(spec):
        if spec == "missing":
            raise FileNotFoundError("no such run")
        return cast(DenoiserUNet, None), "sft-run"

    mod_server._report_model(cast(mod_server.RconClient, rcon), "abc", get_model, 15)
    mod_server._report_model(cast(mod_server.RconClient, rcon), "missing", get_model, 15)

    loaded, failed = rcon.commands
    assert "'model_loaded'" in loaded and '"grid_size":15' in loaded
    assert '"name":"sft-run"' in loaded
    assert "'model_failed'" in failed and "no such run" in failed


def _belt_build(_model, obs, *_):
    world = obs.astype(np.int64)
    for x in range(1, 3):
        world[Channel.ENTITIES.value, x, 5] = _id("transport_belt")
        world[Channel.DIRECTION.value, x, 5] = 2
    return world


@pytest.mark.parametrize(("accepted", "stop_reason", "steps"), [
    (True, "eot", 2), (False, "placement_error", 0),
])
def test_inference_streams_the_build_until_done_or_rejected(monkeypatch, accepted, stop_reason, steps):
    monkeypatch.setattr(mod_server, "best_build", _belt_build)
    streamed = []

    _, stats = mod_server.run_inference(
        cast(DenoiserUNet, None),
        _request(),
        device=torch.device("cpu"),
        on_placement=lambda action: streamed.append(action) or accepted,
    )

    belts = [_action("transport_belt", xy=(x, 5), direction=2) for x in (1, 2)]
    assert streamed == belts[: max(steps, 1)]
    assert stats["steps_taken"] == steps
    assert stats["stop_reason"] == stop_reason


def test_empty_action_becomes_a_deletion():
    assert mod_server.action_to_placement(_action("empty", direction=0)) == {
        "tile_x": 2, "tile_y": 3, "width": 1, "height": 1,
    }


def test_build_streams_one_action_per_new_entity_at_its_anchor():
    obs = mod_server.request_to_obs(_request())
    world = obs.astype(np.int64)
    world[Channel.ENTITIES.value, 4:7, 3:6] = _id("assembling_machine_1")
    world[Channel.ITEMS.value, 4:7, 3:6] = _id("electronic_circuit")
    (action,) = mod_server.build_actions(obs, world)
    assert action["xy"] == (4, 3) and action["entity"] == _id("assembling_machine_1")


def _streamed(model, **req):
    streamed = []
    mod_server.run_inference(
        model, _request() | req, device=torch.device("cpu"),
        on_placement=lambda action: streamed.append(action) or True,
    )
    return streamed


def test_seed_replays_a_generated_build_and_new_seeds_differ():
    torch.manual_seed(0)
    model = DenoiserUNet(**TINY).eval()
    assert _streamed(model, seed=1) == _streamed(model, seed=1)
    assert _streamed(model, seed=1) != _streamed(model, seed=2)


class _CancelledRcon:
    def __init__(self):
        self.commands = []

    def exec(self, command):
        self.commands.append(command)
        return "cancelled"


def test_cancel_in_game_stops_quietly(monkeypatch):
    monkeypatch.setattr(mod_server, "best_build", _belt_build)
    rcon = _CancelledRcon()

    mod_server.handle_request(
        _request() | {"request_id": "r", "checkpoint": "c"},
        lambda _: (cast(DenoiserUNet, None), "run"), cast(mod_server.RconClient, rcon),
        device=torch.device("cpu"),
    )

    place, finish = rcon.commands
    assert "place_prediction" in place
    assert "finish_prediction" in finish and "cancelled" in finish
