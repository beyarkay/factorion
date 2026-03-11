"""Tests for action masking of invalid actions in AgentCNN."""

import os
import sys

import pytest
import torch
import gymnasium as gym
import numpy as np

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402
from helpers import Channel, Direction, entities, str2ent  # noqa: E402


ENV_ID = "factorion/FactorioEnv-v0-masking-test"

SOURCE_ID = len(entities) - 1
SINK_ID = len(entities) - 2


@pytest.fixture(scope="module")
def registered_env():
    """Register the env once for all tests in this module."""
    gym.register(id=ENV_ID, entry_point=FactorioEnv)


@pytest.fixture()
def envs(registered_env):
    """Create a small SyncVectorEnv for testing."""
    return gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, False, 5, "test") for i in range(2)]
    )


@pytest.fixture()
def agent(envs):
    """Create an AgentCNN with default params."""
    return AgentCNN(envs, chan1=32, chan2=64, chan3=64)


class TestTileMask:
    def test_source_sink_never_selected(self, envs, agent):
        """Tiles containing source or sink should never be sampled."""
        # Reset envs to get real observations with source/sink
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        # Find source/sink tile positions from the observation
        ent_channel = obs_t[:, Channel.ENTITIES.value, :, :]  # (B, W, H)
        source_mask = ent_channel == SOURCE_ID
        sink_mask = ent_channel == SINK_ID
        blocked = source_mask | sink_mask

        # Verify there are some blocked tiles
        assert blocked.any(), "Test requires source/sink tiles in observation"

        # Sample many times and verify no blocked tile is selected
        for _ in range(100):
            action_out, _, _, _ = agent.get_action_and_value(obs_t)
            x = action_out["xy"][:, 0]
            y = action_out["xy"][:, 1]
            for b in range(obs_t.shape[0]):
                assert not blocked[b, x[b], y[b]], (
                    f"Sampled blocked tile ({x[b]}, {y[b]}) in batch {b}"
                )

    def test_tile_mask_logits_are_neg_inf(self, envs, agent):
        """Verify that tile logits for source/sink are -inf internally."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)
        B = obs_t.shape[0]

        # Run forward pass manually to inspect logits
        encoded = agent.encoder(obs_t)
        tile_logits_B1WH = agent.tile_logits(encoded)
        tile_logits_BN = tile_logits_B1WH.reshape(B, -1)

        ent_channel = obs_t[:, agent.entity_channel_idx, :, :]
        tile_mask_BN = (
            (ent_channel != agent.source_id) & (ent_channel != agent.sink_id)
        ).reshape(B, -1)
        masked_logits = tile_logits_BN.masked_fill(~tile_mask_BN, float("-inf"))

        # Check that blocked tiles have -inf logits
        blocked_BN = ~tile_mask_BN
        assert (masked_logits[blocked_BN] == float("-inf")).all()
        # Check that valid tiles have finite logits
        assert torch.isfinite(masked_logits[tile_mask_BN]).all()


class TestDirectionMask:
    def test_empty_entity_always_none_direction(self, envs, agent):
        """When entity=empty(0), direction must always be NONE(0)."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        for _ in range(200):
            action_out, _, _, _ = agent.get_action_and_value(obs_t)
            ent = action_out["entity"]
            direc = action_out["direction"]
            # Where entity is empty, direction must be NONE
            empty_mask = ent == 0
            if empty_mask.any():
                assert (direc[empty_mask] == 0).all(), (
                    f"Entity=empty but direction={direc[empty_mask]} (expected NONE=0)"
                )

    def test_belt_entity_never_none_direction(self, envs, agent):
        """When entity=transport_belt(1), direction must not be NONE(0)."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        belt_seen = False
        for _ in range(200):
            action_out, _, _, _ = agent.get_action_and_value(obs_t)
            ent = action_out["entity"]
            direc = action_out["direction"]
            belt_mask = ent == 1
            if belt_mask.any():
                belt_seen = True
                assert (direc[belt_mask] != 0).all(), (
                    f"Entity=belt but direction=NONE (expected cardinal)"
                )
        assert belt_seen, "Never sampled a transport belt in 200 attempts"

    def test_belt_directions_are_cardinal(self, envs, agent):
        """When entity=transport_belt, direction must be in {1,2,3,4}."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        for _ in range(200):
            action_out, _, _, _ = agent.get_action_and_value(obs_t)
            ent = action_out["entity"]
            direc = action_out["direction"]
            belt_mask = ent == 1
            if belt_mask.any():
                dirs = direc[belt_mask]
                assert ((dirs >= 1) & (dirs <= 4)).all(), (
                    f"Belt direction {dirs} not in [1,4]"
                )


class TestNoInvalidActions:
    def test_full_episode_no_invalid_actions(self, envs, agent):
        """Run a full episode and verify frac_invalid_actions is 0."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        for _ in range(20):
            with torch.no_grad():
                action_out, _, _, _ = agent.get_action_and_value(obs_t)
                action_numpy = {k: v.cpu().numpy() for k, v in action_out.items()}

            obs, reward, term, trunc, infos = envs.step(action_numpy)
            obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

            # Check that no invalid action reasons were triggered
            done_mask = np.logical_or(term, trunc)
            for i in range(len(done_mask)):
                if done_mask[i]:
                    assert infos["frac_invalid_actions"][i] == 0.0, (
                        f"Env {i} had invalid actions: frac={infos['frac_invalid_actions'][i]}"
                    )
                    # Reset done env
                    o, _ = envs.envs[i].reset(
                        seed=42 + i,
                        options={"num_missing_entities": 1},
                    )
                    obs[i] = o
                    obs_t = torch.as_tensor(
                        np.array(obs), dtype=torch.float32
                    )


class TestLogProbConsistencyWithMasks:
    def test_log_prob_matches_replay(self, agent, envs):
        """Sample actions with masking, then replay — log probs must match."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        action_out, logp_B, _, _ = agent.get_action_and_value(obs_t)

        x_B = action_out["xy"][:, 0]
        y_B = action_out["xy"][:, 1]
        ent_B = action_out["entity"]
        dir_B = action_out["direction"]
        item_B = action_out["item"]
        misc_B = action_out["misc"]
        action_tensor = torch.stack(
            [x_B, y_B, ent_B, dir_B, item_B, misc_B], dim=1
        )

        _, logp_replay, _, _ = agent.get_action_and_value(
            obs_t, action_tensor.long()
        )
        torch.testing.assert_close(logp_B, logp_replay)

    def test_log_prob_is_negative(self, agent, envs):
        """Log probabilities should be negative (prob < 1)."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)
        _, logp_B, _, _ = agent.get_action_and_value(obs_t)
        assert (logp_B < 0).all()


class TestEntropyWithMasks:
    def test_entropy_is_positive(self, agent, envs):
        """Entropy should be non-negative even with masking."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)
        _, _, entropy_B, _ = agent.get_action_and_value(obs_t)
        assert (entropy_B >= 0).all()

    def test_direction_entropy_reduced_by_masking(self, agent, envs):
        """Direction entropy with masking should be less than max possible.

        With 5 directions, max entropy = ln(5) ≈ 1.61.
        Empty entity → 1 valid dir → entropy = 0.
        Belt → 4 valid dirs → max entropy = ln(4) ≈ 1.39.
        So total direction entropy should always be < ln(5).
        """
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        # Compute direction entropy directly
        encoded = agent.encoder(obs_t)
        B = encoded.shape[0]
        tile_logits = agent.tile_logits(encoded).reshape(B, -1)

        # Apply tile mask
        ent_channel = obs_t[:, agent.entity_channel_idx, :, :]
        tile_mask = (
            (ent_channel != agent.source_id) & (ent_channel != agent.sink_id)
        ).reshape(B, -1)
        tile_logits = tile_logits.masked_fill(~tile_mask, float("-inf"))

        dist_tile = torch.distributions.Categorical(logits=tile_logits)
        tile_idx = dist_tile.sample()
        x_B = tile_idx // agent.height
        y_B = tile_idx % agent.height

        batch_idx = torch.arange(B)
        features = encoded[batch_idx, :, x_B, y_B]
        logits_d = agent.dir_head(features)

        # Unmasked entropy
        dist_unmasked = torch.distributions.Categorical(logits=logits_d)
        unmasked_entropy = dist_unmasked.entropy()

        # Sample entity and apply direction mask
        logits_e = agent.ent_head(features)
        dist_e = torch.distributions.Categorical(logits=logits_e)
        ent = dist_e.sample()
        dir_mask = torch.zeros(B, agent.num_directions, dtype=torch.bool)
        is_empty = ent == 0
        dir_mask[is_empty, 0] = True
        dir_mask[~is_empty, 1:] = True
        masked_logits_d = logits_d.masked_fill(~dir_mask, float("-inf"))
        dist_masked = torch.distributions.Categorical(logits=masked_logits_d)
        masked_entropy = dist_masked.entropy()

        # Masked entropy should be <= unmasked entropy
        assert (masked_entropy <= unmasked_entropy + 1e-6).all()


class TestGradientFlowWithMasks:
    def test_gradients_flow_through_all_params(self, agent, envs):
        """Verify gradients flow through encoder, tile_logits, ent_head, dir_head with masks."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs_t)
        loss = -(logp_B.mean()) + value_B.mean()
        loss.backward()

        assert agent.tile_logits.weight.grad is not None
        assert agent.tile_logits.weight.grad.abs().sum() > 0
        assert agent.ent_head.weight.grad is not None
        assert agent.dir_head.weight.grad is not None

        for name, param in agent.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_during_update(self, agent, envs):
        """Simulate the PPO update path with masks and check grads."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        with torch.no_grad():
            action_out, _, _, _ = agent.get_action_and_value(obs_t)

        x_B = action_out["xy"][:, 0]
        y_B = action_out["xy"][:, 1]
        ent_B = action_out["entity"]
        dir_B = action_out["direction"]
        action_tensor = torch.stack(
            [x_B, y_B, ent_B, dir_B, torch.zeros_like(ent_B), torch.zeros_like(ent_B)],
            dim=1,
        )

        _, logp_B, entropy_B, value_B = agent.get_action_and_value(
            obs_t, action_tensor.long()
        )
        loss = -(logp_B.mean()) + value_B.mean()
        loss.backward()

        assert agent.tile_logits.weight.grad is not None
        assert agent.ent_head.weight.grad is not None
        assert agent.dir_head.weight.grad is not None


class TestOutputShapesWithMasks:
    def test_output_shapes(self, agent, envs):
        """Verify all output tensor shapes are preserved with masking."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)
        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs_t)

        B = obs_t.shape[0]
        assert action_out["xy"].shape == (B, 2)
        assert action_out["entity"].shape == (B,)
        assert action_out["direction"].shape == (B,)
        assert action_out["item"].shape == (B,)
        assert action_out["misc"].shape == (B,)
        assert logp_B.shape == (B,)
        assert entropy_B.shape == (B,)
        assert value_B.shape == (B,)

    def test_xy_within_bounds(self, agent, envs):
        """Verify sampled x, y are within grid bounds."""
        obs, _ = envs.reset(
            seed=42, options={"num_missing_entities": 1}
        )
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)
        W = agent.width

        for _ in range(50):
            action_out, _, _, _ = agent.get_action_and_value(obs_t)
            x = action_out["xy"][:, 0]
            y = action_out["xy"][:, 1]
            assert (x >= 0).all() and (x < W).all(), f"x out of bounds: {x}"
            assert (y >= 0).all() and (y < W).all(), f"y out of bounds: {y}"
