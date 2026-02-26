"""Tests for torch.compile() integration with AgentCNN."""

import os
import sys

import pytest
import torch
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402


ENV_ID = "factorion/FactorioEnv-v0-compile-test"

# torch.compile requires torch >= 2.0
TORCH_COMPILE_AVAILABLE = hasattr(torch, "compile")


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
    """Create an uncompiled AgentCNN."""
    return AgentCNN(envs, chan1=32, chan2=64, chan3=64)


@pytest.fixture()
def compiled_agent(agent):
    """Create a compiled AgentCNN."""
    return torch.compile(agent, mode="default")


@pytest.mark.skipif(not TORCH_COMPILE_AVAILABLE, reason="torch.compile requires torch >= 2.0")
class TestCompiledForwardPass:
    def test_compiled_output_shapes(self, compiled_agent):
        """Verify compiled agent produces correct output shapes."""
        obs = torch.randn(2, 4, 5, 5)
        action_out, logp_B, entropy_B, value_B = compiled_agent.get_action_and_value(obs)

        assert action_out["xy"].shape == (2, 2)
        assert action_out["entity"].shape == (2,)
        assert action_out["direction"].shape == (2,)
        assert action_out["item"].shape == (2,)
        assert action_out["misc"].shape == (2,)
        assert logp_B.shape == (2,)
        assert entropy_B.shape == (2,)
        assert value_B.shape == (2,)

    def test_compiled_get_value_shape(self, compiled_agent):
        """Verify compiled get_value output shape."""
        obs = torch.randn(4, 4, 5, 5)
        value = compiled_agent.get_value(obs)
        assert value.shape == (4,)

    def test_compiled_xy_within_bounds(self, compiled_agent):
        """Verify compiled agent produces valid coordinates."""
        obs = torch.randn(8, 4, 5, 5)
        for _ in range(5):
            action_out, _, _, _ = compiled_agent.get_action_and_value(obs)
            x = action_out["xy"][:, 0]
            y = action_out["xy"][:, 1]
            assert (x >= 0).all() and (x < 5).all(), f"x out of bounds: {x}"
            assert (y >= 0).all() and (y < 5).all(), f"y out of bounds: {y}"


@pytest.mark.skipif(not TORCH_COMPILE_AVAILABLE, reason="torch.compile requires torch >= 2.0")
class TestCompiledLogProbConsistency:
    def test_compiled_log_prob_replay(self, compiled_agent):
        """Sample actions from compiled agent, replay them, verify log_probs match."""
        obs = torch.randn(4, 4, 5, 5)
        action_out, logp_B, _, _ = compiled_agent.get_action_and_value(obs)

        x_B = action_out["xy"][:, 0]
        y_B = action_out["xy"][:, 1]
        ent_B = action_out["entity"]
        dir_B = action_out["direction"]
        item_B = action_out["item"]
        misc_B = action_out["misc"]
        action_tensor = torch.stack(
            [x_B, y_B, ent_B, dir_B, item_B, misc_B], dim=1
        )

        _, logp_replay, _, _ = compiled_agent.get_action_and_value(
            obs, action_tensor.long()
        )
        torch.testing.assert_close(logp_B, logp_replay)


@pytest.mark.skipif(not TORCH_COMPILE_AVAILABLE, reason="torch.compile requires torch >= 2.0")
class TestCompiledGradientFlow:
    def test_gradients_flow_through_compiled_model(self, compiled_agent):
        """Verify gradients propagate through the compiled agent."""
        obs = torch.randn(4, 4, 5, 5)
        action_out, logp_B, entropy_B, value_B = compiled_agent.get_action_and_value(obs)
        loss = -(logp_B.mean()) + value_B.mean()
        loss.backward()

        # Access underlying module's parameters through the compiled wrapper
        underlying = compiled_agent._orig_mod if hasattr(compiled_agent, "_orig_mod") else compiled_agent
        assert underlying.tile_logits.weight.grad is not None
        assert underlying.ent_head.weight.grad is not None
        assert underlying.dir_head.weight.grad is not None

        for name, param in underlying.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


@pytest.mark.skipif(not TORCH_COMPILE_AVAILABLE, reason="torch.compile requires torch >= 2.0")
class TestCompiledParity:
    def test_deterministic_parity(self, agent, envs):
        """Compiled and uncompiled agents with same weights produce same values."""
        obs = torch.randn(4, 4, 5, 5)
        action_tensor = torch.tensor(
            [
                [2, 3, 1, 2, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [4, 4, 1, 4, 0, 0],
                [1, 1, 0, 0, 0, 0],
            ],
            dtype=torch.long,
        )

        # Get uncompiled results
        with torch.no_grad():
            _, logp_orig, entropy_orig, value_orig = agent.get_action_and_value(
                obs, action_tensor
            )

        # Compile the same agent (same weights)
        compiled = torch.compile(agent, mode="default")
        with torch.no_grad():
            _, logp_compiled, entropy_compiled, value_compiled = compiled.get_action_and_value(
                obs, action_tensor
            )

        torch.testing.assert_close(logp_orig, logp_compiled)
        torch.testing.assert_close(entropy_orig, entropy_compiled)
        torch.testing.assert_close(value_orig, value_compiled)

    def test_compiled_value_parity(self, agent):
        """Compiled and uncompiled agents produce same value estimates."""
        obs = torch.randn(4, 4, 5, 5)

        with torch.no_grad():
            value_orig = agent.get_value(obs)

        compiled = torch.compile(agent, mode="default")
        with torch.no_grad():
            value_compiled = compiled.get_value(obs)

        torch.testing.assert_close(value_orig, value_compiled)


@pytest.mark.skipif(not TORCH_COMPILE_AVAILABLE, reason="torch.compile requires torch >= 2.0")
class TestCompileOptimizer:
    def test_optimizer_works_with_compiled_model(self, compiled_agent):
        """Verify that Adam optimizer can step on a compiled model."""
        optimizer = torch.optim.Adam(compiled_agent.parameters(), lr=1e-3)

        obs = torch.randn(4, 4, 5, 5)
        action_out, logp_B, entropy_B, value_B = compiled_agent.get_action_and_value(obs)
        loss = -(logp_B.mean()) + value_B.mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Second forward pass should still work after optimizer step
        action_out2, logp_B2, entropy_B2, value_B2 = compiled_agent.get_action_and_value(obs)
        assert logp_B2.shape == (4,)
        assert value_B2.shape == (4,)
