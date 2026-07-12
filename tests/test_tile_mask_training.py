"""Invalid-action masking on the tile head (issue #223).

`get_action_and_value` masks illegal tiles for every caller, so the mask is
applied identically when sampling in the rollout and when recomputing
log-probs/entropy in the update — the data-collecting and updated policies
agree. These tests pin the correctness-critical behaviour:

1. `_legal_tile_mask` marks exactly the empty + buildable tiles, in the
   x-major flatten (index = x*H + y) the tile head decodes.
2. `get_action_and_value` never samples an illegal tile (also proves the mask
   layout matches the head's tile-index decode).
3. Masked log-prob/entropy stay finite and the log-prob round-trips (so PPO's
   importance ratio is well-defined under masking).
4. A fully-occupied grid (no legal tile) degrades to a uniform distribution
   rather than NaN.
"""

import os
import sys

import pytest
import torch
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import Channel, str2ent  # noqa: E402
from factorion import Footprint  # noqa: E402
from ppo import AgentCNN, _legal_tile_mask, make_env  # noqa: E402

NUM_CHANNELS = len(Channel)
ENV_ID = "factorion/FactorioEnv-v0-tilemask-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


@pytest.fixture()
def agent(registered_env):
    envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
    a = AgentCNN(envs, layers=(16, 16, 16))
    envs.close()
    return a


def _empty_obs(batch, size):
    """A batch of all-empty, all-buildable observations (B, C, W, H)."""
    obs = torch.zeros(batch, NUM_CHANNELS, size, size)
    obs[:, Channel.ENTITIES.value] = str2ent("empty").value
    obs[:, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
    return obs


def _stack_action(action):
    x_B, y_B = action["xy"].unbind(dim=1)
    return torch.stack(
        [
            x_B, y_B,
            action["entity"], action["direction"],
            action["item"], action["misc"],
            action["eot"].long(),
        ],
        dim=1,
    )


class TestLegalTileMask:
    def test_marks_empty_buildable_in_x_major_layout(self):
        """Occupied and unbuildable tiles are False; everything else True. The
        flat index is x*H + y — the same layout the tile head decodes."""
        size = 3
        obs = _empty_obs(1, size)
        # Occupy (0, 1) -> flat 0*3 + 1 = 1.
        obs[0, Channel.ENTITIES.value, 0, 1] = str2ent("transport_belt").value
        # Empty but unbuildable (2, 0) -> flat 2*3 + 0 = 6.
        obs[0, Channel.FOOTPRINT.value, 2, 0] = Footprint.UNAVAILABLE.value

        mask = _legal_tile_mask(obs)
        assert mask.shape == (1, size * size)
        assert mask.dtype == torch.bool
        legal = {0, 2, 3, 4, 5, 7, 8}
        for n in range(size * size):
            assert bool(mask[0, n]) == (n in legal), f"tile {n}"

    def test_all_empty_grid_is_all_legal(self):
        mask = _legal_tile_mask(_empty_obs(2, 4))
        assert mask.shape == (2, 16)
        assert mask.all()

    def test_fully_occupied_grid_is_all_illegal(self):
        obs = _empty_obs(1, 3)
        obs[0, Channel.ENTITIES.value] = str2ent("transport_belt").value
        assert not _legal_tile_mask(obs).any()


class TestMaskedSampling:
    def test_never_samples_an_illegal_tile(self, agent):
        """The masked tile distribution must place ~0 probability on illegal
        tiles: over many samples the chosen tile is always legal. Also proves
        the mask's flatten matches the head's tile-index decode."""
        torch.manual_seed(0)
        size = 5
        obs = _empty_obs(3, size)
        # Occupy a scattered set of tiles across the three envs.
        obs[0, Channel.ENTITIES.value, 0, 0] = str2ent("transport_belt").value
        obs[1, Channel.FOOTPRINT.value, 2, 3] = Footprint.UNAVAILABLE.value
        obs[2, Channel.ENTITIES.value, 4, 4] = str2ent("transport_belt").value
        mask = _legal_tile_mask(obs)

        for _ in range(64):
            action, _, _, _ = agent.get_action_and_value(obs)
            x_B, y_B = action["xy"].unbind(dim=1)
            flat_B = x_B * size + y_B
            legal_here = mask.gather(1, flat_B.unsqueeze(1)).squeeze(1)
            assert legal_here.all(), "masked sampling proposed an illegal tile"

    def test_masked_logprob_and_entropy_finite_and_roundtrip(self, agent):
        """log-prob/entropy stay finite under masking, and the stored-action
        log-prob recomputes exactly — PPO's ratio needs this equality."""
        torch.manual_seed(1)
        obs = _empty_obs(4, 5)
        obs[0, Channel.ENTITIES.value, 1, 1] = str2ent("transport_belt").value
        obs[2, Channel.FOOTPRINT.value, 0, 4] = Footprint.UNAVAILABLE.value

        action, logp, entropy, _ = agent.get_action_and_value(obs)
        assert torch.isfinite(logp).all()
        assert torch.isfinite(entropy).all()

        action_BA = _stack_action(action)
        _, logp2, _, _ = agent.get_action_and_value(obs, action_BA)
        torch.testing.assert_close(logp, logp2)

    def test_all_illegal_row_is_uniform_not_nan(self, agent):
        """A fully-occupied grid has no legal tile; the tile head must degrade
        to a (finite) uniform rather than an all -inf / NaN distribution."""
        obs = _empty_obs(1, 5)
        obs[0, Channel.ENTITIES.value] = str2ent("transport_belt").value
        assert not _legal_tile_mask(obs).any()

        _, logp, entropy, _ = agent.get_action_and_value(obs)
        assert torch.isfinite(logp).all()
        assert torch.isfinite(entropy).all()

    def test_precomputed_mask_matches_derived(self, agent):
        """Passing a precomputed illegal mask (the PPO update's fast path, which
        derives it once per batch) must give byte-identical log-probs/entropy to
        deriving it inside — indexing commutes with the elementwise derivation,
        so this is a pure speedup with no behaviour change."""
        torch.manual_seed(3)
        obs = _empty_obs(4, 5)
        obs[0, Channel.ENTITIES.value, 1, 1] = str2ent("transport_belt").value
        obs[2, Channel.FOOTPRINT.value, 0, 4] = Footprint.UNAVAILABLE.value

        # A fixed action so both calls score the same choice.
        action, _, _, _ = agent.get_action_and_value(obs)
        action_BA = _stack_action(action)

        _, logp_derived, ent_derived, _ = agent.get_action_and_value(obs, action_BA)
        illegal = ~_legal_tile_mask(obs)
        _, logp_passed, ent_passed, _ = agent.get_action_and_value(
            obs, action_BA, tile_illegal_BN=illegal
        )
        torch.testing.assert_close(logp_derived, logp_passed)
        torch.testing.assert_close(ent_derived, ent_passed)
