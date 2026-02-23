"""Tests for spatial per-tile action prediction in AgentCNN."""

import os
import sys

import pytest
import torch
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402


ENV_ID = "factorion/FactorioEnv-v0-spatial-test"


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


class TestForwardPass:
    def test_output_shapes(self, agent):
        """Verify all output tensor shapes from get_action_and_value."""
        obs = torch.randn(2, 4, 5, 5)
        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs)

        assert action_out["xy"].shape == (2, 2)
        assert action_out["entity"].shape == (2,)
        assert action_out["direction"].shape == (2,)
        assert action_out["item"].shape == (2,)
        assert action_out["misc"].shape == (2,)
        assert logp_B.shape == (2,)
        assert entropy_B.shape == (2,)
        assert value_B.shape == (2,)

    def test_single_batch(self, agent):
        """Verify forward pass works with batch size 1."""
        obs = torch.randn(1, 4, 5, 5)
        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs)

        assert action_out["xy"].shape == (1, 2)
        assert logp_B.shape == (1,)
        assert entropy_B.shape == (1,)
        assert value_B.shape == (1,)

    def test_xy_within_bounds(self, agent):
        """Verify sampled x, y are within grid bounds."""
        obs = torch.randn(16, 4, 5, 5)
        for _ in range(10):
            action_out, _, _, _ = agent.get_action_and_value(obs)
            x = action_out["xy"][:, 0]
            y = action_out["xy"][:, 1]
            assert (x >= 0).all() and (x < 5).all(), f"x out of bounds: {x}"
            assert (y >= 0).all() and (y < 5).all(), f"y out of bounds: {y}"

    def test_get_value_shape(self, agent):
        """Verify get_value output shape."""
        obs = torch.randn(4, 4, 5, 5)
        value = agent.get_value(obs)
        assert value.shape == (4,)

    def test_item_and_misc_are_zero(self, agent):
        """Item and misc should always be zero (hardcoded)."""
        obs = torch.randn(8, 4, 5, 5)
        action_out, _, _, _ = agent.get_action_and_value(obs)
        assert (action_out["item"] == 0).all()
        assert (action_out["misc"] == 0).all()


class TestLogProbConsistency:
    def test_log_prob_matches_replay(self, agent):
        """Sample actions, then replay them and verify log_prob matches."""
        obs = torch.randn(8, 4, 5, 5)
        action_out, logp_B, _, _ = agent.get_action_and_value(obs)

        # Reconstruct action tensor as the training loop does
        x_B = action_out["xy"][:, 0]
        y_B = action_out["xy"][:, 1]
        ent_B = action_out["entity"]
        dir_B = action_out["direction"]
        item_B = action_out["item"]
        misc_B = action_out["misc"]
        action_tensor = torch.stack(
            [x_B, y_B, ent_B, dir_B, item_B, misc_B], dim=1
        )

        # Replay: pass the same obs and action tensor
        _, logp_replay, _, _ = agent.get_action_and_value(
            obs, action_tensor.long()
        )
        torch.testing.assert_close(logp_B, logp_replay)

    def test_log_prob_is_negative(self, agent):
        """Log probabilities should be negative (prob < 1)."""
        obs = torch.randn(4, 4, 5, 5)
        _, logp_B, _, _ = agent.get_action_and_value(obs)
        assert (logp_B < 0).all()


class TestEntropy:
    def test_entropy_is_positive(self, agent):
        """Entropy should be non-negative."""
        obs = torch.randn(4, 4, 5, 5)
        _, _, entropy_B, _ = agent.get_action_and_value(obs)
        assert (entropy_B >= 0).all()


class TestGradientFlow:
    def test_gradients_flow_through_all_params(self, agent):
        """Verify gradients flow to encoder, tile_logits, ent_head, dir_head."""
        obs = torch.randn(4, 4, 5, 5)
        action_out, logp_B, entropy_B, value_B = agent.get_action_and_value(obs)
        loss = -(logp_B.mean()) + value_B.mean()
        loss.backward()

        # Check tile_logits conv has gradients
        assert agent.tile_logits.weight.grad is not None
        assert agent.tile_logits.weight.grad.abs().sum() > 0

        # Check entity/direction heads have gradients
        assert agent.ent_head.weight.grad is not None
        assert agent.dir_head.weight.grad is not None

        # Check encoder has gradients
        for name, param in agent.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_gradients_flow_during_update(self, agent):
        """Simulate the PPO update path (action not None) and check grads."""
        obs = torch.randn(4, 4, 5, 5)
        with torch.no_grad():
            action_out, _, _, _ = agent.get_action_and_value(obs)
        x_B = action_out["xy"][:, 0]
        y_B = action_out["xy"][:, 1]
        ent_B = action_out["entity"]
        dir_B = action_out["direction"]
        action_tensor = torch.stack(
            [x_B, y_B, ent_B, dir_B, torch.zeros_like(ent_B), torch.zeros_like(ent_B)],
            dim=1,
        )

        _, logp_B, entropy_B, value_B = agent.get_action_and_value(
            obs, action_tensor.long()
        )
        loss = -(logp_B.mean()) + value_B.mean()
        loss.backward()

        assert agent.tile_logits.weight.grad is not None
        assert agent.ent_head.weight.grad is not None
        assert agent.dir_head.weight.grad is not None


class TestBatchConsistency:
    def test_single_vs_batch(self, agent):
        """Processing items one-at-a-time gives same results as batched."""
        obs = torch.randn(3, 4, 5, 5)
        # Create a fixed action to replay (within bounds for 5x5 grid)
        action_tensor = torch.tensor(
            [
                [2, 3, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [4, 4, 1, 4, 0, 0],
            ],
            dtype=torch.long,
        )

        # Batched
        _, logp_batch, entropy_batch, value_batch = agent.get_action_and_value(
            obs, action_tensor
        )

        # One at a time
        for i in range(3):
            _, logp_single, entropy_single, value_single = (
                agent.get_action_and_value(obs[i : i + 1], action_tensor[i : i + 1])
            )
            torch.testing.assert_close(logp_batch[i : i + 1], logp_single)
            torch.testing.assert_close(entropy_batch[i : i + 1], entropy_single)
            torch.testing.assert_close(value_batch[i : i + 1], value_single)
