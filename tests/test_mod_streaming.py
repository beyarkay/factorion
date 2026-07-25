"""Streaming-placement tests for the Factorio mod server."""

from pathlib import Path
import sys
from typing import cast

import torch

from factorion import Channel, Misc, str2ent
from ppo import AgentCNN


SERVER_DIR = Path(__file__).parents[1] / "factorion-mod" / "server"
sys.path.insert(0, str(SERVER_DIR))
import server as mod_server  # noqa: E402


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


def test_transport_belt_action_becomes_world_placement():
    placement = mod_server.action_to_placement(_action("transport_belt"))

    assert placement == {
        "name": "transport-belt",
        "tile_x": 2,
        "tile_y": 3,
        "width": 1,
        "height": 1,
        "x": 2.5,
        "y": 3.5,
        "direction": 0,
    }


def test_inserter_direction_points_toward_its_drop_tile():
    placement = mod_server.action_to_placement(
        _action("inserter", direction=2),
    )

    assert placement["direction"] == 12


def test_assembler_placement_has_center_and_recipe():
    placement = mod_server.action_to_placement(
        _action(
            "assembling_machine_1",
            xy=(4, 5),
            direction=0,
            item="electronic_circuit",
        ),
    )

    assert placement["name"] == "assembling-machine-1"
    assert placement["width"] == 3
    assert placement["height"] == 3
    assert placement["x"] == 5.5
    assert placement["y"] == 6.5
    assert placement["recipe"] == "electronic-circuit"


def test_underground_belt_placement_has_endpoint_type():
    placement = mod_server.action_to_placement(
        _action(
            "underground_belt",
            misc=Misc.UNDERGROUND_DOWN.value,
        ),
    )

    assert placement["type"] == "input"


def test_request_seeds_existing_entities_with_configuration():
    request = _request()
    request["entities"] = [
        {
            "name": "transport-belt",
            "x": 1,
            "y": 2,
            "direction": 2,
        },
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
        {
            "name": "splitter",
            "x": 1,
            "y": 7,
            "direction": 2,
        },
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


class _AcceptingRcon:
    def exec(self, _command):
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


class _OneStepAgent:
    def __init__(self):
        self.calls = 0

    def eot_prob(self, _obs):
        self.calls += 1
        return torch.tensor([0.0 if self.calls == 1 else 1.0])


def _request():
    return {
        "grid_size": 11,
        "footprint": [[x, y] for x in range(11) for y in range(11)],
        "sources": [
            {"x": 0, "y": 5, "direction": 2, "item": "iron_plate"},
        ],
        "sinks": [
            {"x": 10, "y": 5, "direction": 2, "item": "iron_plate"},
        ],
    }


def test_inference_streams_each_accepted_action(monkeypatch):
    action = _action("transport_belt", xy=(1, 5), direction=2)
    monkeypatch.setattr(mod_server, "_argmax_action", lambda *_: action)
    streamed = []

    _, stats = mod_server.run_inference(
        cast(AgentCNN, _OneStepAgent()),
        _request(),
        max_steps=4,
        device=torch.device("cpu"),
        on_placement=lambda predicted: streamed.append(predicted) or True,
    )

    assert streamed == [action]
    assert stats["steps_taken"] == 1
    assert stats["stop_reason"] == "eot"


def test_inference_stops_if_factorio_rejects_a_placement(monkeypatch):
    action = _action("transport_belt", xy=(1, 5), direction=2)
    monkeypatch.setattr(mod_server, "_argmax_action", lambda *_: action)

    _, stats = mod_server.run_inference(
        cast(AgentCNN, _OneStepAgent()),
        _request(),
        max_steps=4,
        device=torch.device("cpu"),
        on_placement=lambda _: False,
    )

    assert stats["steps_taken"] == 1
    assert stats["stop_reason"] == "placement_error"
