"""Tests for the FLOW observation channel.

The observation the policy and critic see is the world's channels plus a
derived FLOW channel: the steady-state items/s the engine computes through
each entity. These cover the whole path — engine → `simulate` → env → network.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorion import (  # noqa: E402
    CH_FLOW,
    FLOW_SCALE,
    MAX_ENTITY_FLOW,
    OBS_CHANNELS,
    LessonKind,
    build_factory,
    simulate,
)
from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402
from helpers import (  # noqa: E402
    TINY_ARCH,
    Channel,
    Direction,
    make_world,
    set_entity,
)

ENV_ID = "factorion/FactorioEnv-v0-flow-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


def belt_line(middle="transport_belt"):
    """Source → `middle` → belt → sink along row 0, as a (C, W, H) world."""
    world = make_world(5)
    set_entity(world, 0, 0, "stack_inserter", Direction.EAST, item_name="copper_cable")
    set_entity(world, 1, 0, middle, Direction.EAST)
    set_entity(world, 2, 0, "transport_belt", Direction.EAST)
    set_entity(world, 3, 0, "bulk_inserter", Direction.EAST, item_name="copper_cable")
    return world.permute(2, 0, 1).contiguous()


class TestSimulate:
    def test_appends_flow_after_the_world_channels(self):
        """The world's channels come through untouched, with FLOW appended."""
        world = belt_line()
        obs, thput, unreachable = simulate(world)

        assert obs.shape == (OBS_CHANNELS, 5, 5)
        assert (obs[:CH_FLOW] == world).all()
        assert thput > 0
        assert unreachable == 0

    def test_flow_follows_the_line_and_stops_at_empty_tiles(self):
        """Every entity on the source→sink path carries the belt's 15 i/s
        (the source clamps to the channel's ceiling); empty tiles carry zero."""
        flow = simulate(belt_line())[0][CH_FLOW] / FLOW_SCALE

        assert flow[0, 0] == MAX_ENTITY_FLOW
        assert flow[1, 0] == 15.0
        assert flow[2, 0] == 15.0
        assert flow[3, 0] == 15.0
        assert flow[4, 0] == 0.0
        assert (flow[:, 1:] == 0).all()

    def test_entity_with_no_feeder_carries_nothing(self):
        """A belt nothing feeds moves nothing, beside a flowing line."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, item_name="copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "bulk_inserter", Direction.EAST, item_name="copper_cable")
        set_entity(world, 2, 2, "transport_belt", Direction.EAST)

        flow = simulate(world.permute(2, 0, 1).contiguous())[0][CH_FLOW]
        assert flow[1, 0] > 0
        assert flow[2, 2] == 0

    def test_a_chain_short_of_the_sink_still_carries(self):
        """The channel advances one tile per belt while throughput stays flat
        zero — the progress signal a value function has no other way to see.
        Delivery stays legible at the sink, which reads its own rate."""
        def partial(n_belts):
            world = make_world(7)
            set_entity(world, 0, 0, "stack_inserter", Direction.EAST, item_name="copper_cable")
            for x in range(1, 1 + n_belts):
                set_entity(world, x, 0, "transport_belt", Direction.EAST)
            set_entity(world, 5, 0, "bulk_inserter", Direction.EAST, item_name="copper_cable")
            return simulate(world.permute(2, 0, 1).contiguous())

        for n_belts in range(1, 4):
            obs, thput, _ = partial(n_belts)
            flow = obs[CH_FLOW] / FLOW_SCALE
            assert thput == 0.0
            assert flow[n_belts, 0] == 15.0
            assert flow[n_belts + 1, 0] == 0.0
            assert flow[5, 0] == 0.0, "nothing has reached the sink"

        obs, thput, _ = partial(4)
        assert thput > 0
        assert obs[CH_FLOW][5, 0] / FLOW_SCALE == 15.0

    def test_a_broken_last_hop_reads_zero_at_the_sink(self):
        """Turning the last belt away keeps the chain carrying but empties the
        sink — the case unreachability flags but the score alone cannot place."""
        world = make_world(6)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, item_name="copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.NORTH)
        set_entity(world, 3, 0, "bulk_inserter", Direction.EAST, item_name="copper_cable")

        obs, thput, _ = simulate(world.permute(2, 0, 1).contiguous())
        flow = obs[CH_FLOW] / FLOW_SCALE
        assert thput == 0.0
        assert flow[1, 0] == 15.0
        assert flow[2, 0] == 15.0
        assert flow[3, 0] == 0.0

    def test_channel_is_integer_valued_so_uint8_storage_is_lossless(self):
        """SFT keeps its demonstrations in uint8 — FLOW's eighths must survive
        the cast, including an inserter's sub-1 i/s rate."""
        obs = simulate(belt_line(middle="inserter"))[0]

        assert (obs[CH_FLOW] == obs[CH_FLOW].round()).all()
        assert obs[CH_FLOW].max() <= 255
        assert (obs.to(torch.uint8).to(obs.dtype) == obs).all()
        # 0.86 i/s to the nearest eighth: distinct from both 0 and a belt's 15.
        assert obs[CH_FLOW, 1, 0] == 7

    def test_keeps_the_world_dtype(self):
        """Callers cast for the network themselves, so the observation must
        not silently promote an int64 world to float."""
        world = belt_line()
        assert simulate(world)[0].dtype == world.dtype
        assert simulate(world.float())[0].dtype == torch.float32

    def test_flow_is_zero_on_a_belt_loop(self):
        """A cycle scores zero throughput while every entity still looks
        connected — unreachability cannot see it, but FLOW reads zero."""
        world = make_world(5)
        set_entity(world, 0, 0, "stack_inserter", Direction.EAST, item_name="copper_cable")
        set_entity(world, 1, 0, "transport_belt", Direction.EAST)
        set_entity(world, 2, 0, "transport_belt", Direction.SOUTH)
        set_entity(world, 2, 1, "transport_belt", Direction.WEST)
        set_entity(world, 1, 1, "transport_belt", Direction.NORTH)
        set_entity(world, 3, 0, "bulk_inserter", Direction.EAST, item_name="copper_cable")

        obs, thput, unreachable = simulate(world.permute(2, 0, 1).contiguous())
        assert thput == 0.0
        assert unreachable == 0
        assert (obs[CH_FLOW] == 0).all()


class TestEnvObservation:
    def test_reset_exposes_the_channel(self):
        """The env's observation carries FLOW from reset onward. Nothing is
        built yet, so only the source markers — an infinite supply, clamped to
        the channel's ceiling — carry anything."""
        env = FactorioEnv(size=11, idx=0)
        obs, _ = env.reset(seed=7, options={"kind": LessonKind.MOVE_ONE_ITEM})

        assert obs.shape == (OBS_CHANNELS, 11, 11)
        source = obs[Channel.ENTITIES.value] == env._source_id
        assert (obs[CH_FLOW][source] == MAX_ENTITY_FLOW * FLOW_SCALE).all()
        assert (obs[CH_FLOW][~source] == 0).all()

        factory = build_factory(size=11, kind=LessonKind.MOVE_ONE_ITEM, seed=7)
        assert factory is not None
        sink = factory.world_CWH[Channel.ENTITIES.value] == env._sink_id
        assert (simulate(factory.world_CWH)[0][CH_FLOW][sink] > 0).all()

    def test_step_refreshes_the_channel(self, registered_env):
        """Placing the belt that completes a line lights the whole line up in
        the very observation the agent acts on next."""
        env = FactorioEnv(size=5, idx=0)
        env.reset(seed=0, options={"kind": LessonKind.MOVE_ONE_ITEM})
        env._world_CWH[:] = torch.as_tensor(belt_line())
        env._world_CWH[Channel.ENTITIES.value, 2, 0] = 0
        env._world_CWH[Channel.DIRECTION.value, 2, 0] = 0

        obs, *_ = env.step({
            "xy": np.array([2, 0]),
            "entity": int(env._world_CWH[Channel.ENTITIES.value, 1, 0]),
            "direction": Direction.EAST.value,
            "item": 0,
            "misc": 0,
            "eot": 0,
        })
        assert (obs[CH_FLOW, :4, 0] > 0).all()


class TestNetworkInput:
    @pytest.fixture()
    def agent(self, registered_env):
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()
        return agent.eval()

    def test_flow_reaches_the_encoder(self, agent):
        """FLOW is a real input, not a dropped channel: changing it alone
        moves both the encoder features and the critic's value."""
        obs = torch.zeros(1, OBS_CHANNELS, 5, 5)
        flowing = obs.clone()
        flowing[0, CH_FLOW, 2, 2] = 15.0 * FLOW_SCALE

        with torch.no_grad():
            assert not torch.allclose(agent.encode(obs)[0], agent.encode(flowing)[0])
            assert not torch.allclose(agent.get_value(obs), agent.get_value(flowing))

    def test_flow_is_log_compressed(self, agent):
        """Raw items/s spans a 35x range; the encoder sees log1p of it so the
        inserter end stays legible next to the one-hot channels."""
        obs = torch.zeros(1, OBS_CHANNELS, 5, 5)
        obs[0, CH_FLOW, 1, 1] = MAX_ENTITY_FLOW * FLOW_SCALE
        # Flow trails the embeddings, the two one-hots and the footprint.
        slot = 2 * agent.cat_embed_dim + agent.num_directions + agent.num_misc + 1

        encoded = agent._encode_input(obs).detach()
        assert encoded[0, slot, 1, 1] == pytest.approx(
            np.log1p(MAX_ENTITY_FLOW), abs=1e-5
        )
        assert encoded[0, slot].sum() == encoded[0, slot, 1, 1]
