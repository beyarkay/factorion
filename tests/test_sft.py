"""Tests for SFT pre-training pipeline."""

import os
import sys

import pytest
import torch
import gymnasium as gym
import numpy as np

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import Channel, Direction, LessonKind, generate_lesson, str2ent
from sft import extract_expert_actions, generate_dataset, SFTArgs
from ppo import FactorioEnv, AgentCNN, make_env


class TestExtractExpertActions:
    def test_reconstructs_solved_world(self):
        """Replaying all extracted actions should reconstruct the solved world."""
        solved, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=0, seed=42,
        )
        task, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=3, seed=42,
        )
        pairs = extract_expert_actions(solved, task)
        assert len(pairs) > 0, "Should have at least one action"

        # Replay actions onto task world
        state = task.clone()
        for obs, tile_idx, entity_id, direction_id in pairs:
            H = state.shape[2]
            x = tile_idx // H
            y = tile_idx % H
            state[Channel.ENTITIES.value, x, y] = entity_id
            state[Channel.DIRECTION.value, x, y] = direction_id
            # Copy other channels from solved
            state[Channel.ITEMS.value, x, y] = solved[Channel.ITEMS.value, x, y]
            state[Channel.MISC.value, x, y] = solved[Channel.MISC.value, x, y]

        # Entity and direction channels should match solved
        assert torch.equal(
            state[Channel.ENTITIES.value], solved[Channel.ENTITIES.value]
        ), "Entities should match after replaying all actions"
        assert torch.equal(
            state[Channel.DIRECTION.value], solved[Channel.DIRECTION.value]
        ), "Directions should match after replaying all actions"

    def test_no_actions_when_identical(self):
        """No actions needed when solved == task."""
        solved, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=0, seed=42,
        )
        pairs = extract_expert_actions(solved, solved.clone())
        assert len(pairs) == 0

    def test_action_count_matches_missing(self):
        """Number of actions should equal num_missing_entities."""
        for seed in [1, 7, 42]:
            solved, _ = generate_lesson(
                size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=0, seed=seed,
            )
            # generate_lesson returns (world, actual_removed) where actual_removed
            # may be less than num_missing_entities if the factory has fewer entities
            task, min_ent = generate_lesson(
                size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=2, seed=seed,
            )
            pairs = extract_expert_actions(solved, task)
            assert len(pairs) == min_ent, (
                f"seed={seed}: expected {min_ent} actions, got {len(pairs)}"
            )

    def test_intermediate_states_are_sequential(self):
        """Each observation should reflect previously applied actions."""
        solved, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=3, seed=99,
        )
        task, _ = generate_lesson(
            size=8, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=3, seed=99,
        )
        pairs = extract_expert_actions(solved, task)

        if len(pairs) < 2:
            pytest.skip("Need at least 2 actions for this test")

        # The second observation should differ from the first
        # (because the first action was applied)
        obs0 = pairs[0][0]
        obs1 = pairs[1][0]
        assert not torch.equal(obs0, obs1), (
            "Sequential observations should differ after action application"
        )

    def test_entity_ids_are_valid(self):
        """All extracted entity IDs should be valid (non-empty) entity values."""
        solved, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=2, seed=42,
        )
        task, _ = generate_lesson(
            size=5, kind=LessonKind.MOVE_ONE_ITEM, num_missing_entities=2, seed=42,
        )
        pairs = extract_expert_actions(solved, task)
        for _, _, entity_id, direction_id in pairs:
            assert entity_id != str2ent("empty").value, "Expert actions shouldn't place empty"
            assert direction_id != Direction.NONE.value, "Expert belt actions need a direction"


class TestGenerateDataset:
    def test_generates_correct_count(self):
        """Dataset should have the requested number of samples."""
        args = SFTArgs(seed=1, size=5, num_samples=100, max_level=2)
        obs, tiles, ents, dirs = generate_dataset(args)
        assert len(obs) == 100
        assert len(tiles) == 100
        assert len(ents) == 100
        assert len(dirs) == 100

    def test_observation_shape(self):
        """Observations should have correct shape (C, W, H)."""
        args = SFTArgs(seed=1, size=5, num_samples=50, max_level=2)
        obs, _, _, _ = generate_dataset(args)
        assert obs.shape[1] == len(Channel)  # channels
        assert obs.shape[2] == 5  # width
        assert obs.shape[3] == 5  # height

    def test_tile_indices_in_range(self):
        """Tile indices should be in [0, W*H)."""
        args = SFTArgs(seed=1, size=5, num_samples=50, max_level=2)
        _, tiles, _, _ = generate_dataset(args)
        assert (tiles >= 0).all()
        assert (tiles < 5 * 5).all()


ENV_ID = "factorion/FactorioEnv-v0-sft-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point=FactorioEnv)


class TestSFTCheckpointLoading:
    def test_checkpoint_roundtrip(self, registered_env, tmp_path):
        """SFT checkpoint should load into AgentCNN and produce valid outputs."""
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, chan1=16, chan2=16, chan3=16, flat_dim=64)

        # Save checkpoint
        ckpt_path = str(tmp_path / "test_sft.pt")
        torch.save(agent.state_dict(), ckpt_path)

        # Load into fresh agent
        agent2 = AgentCNN(envs, chan1=16, chan2=16, chan3=16, flat_dim=64)
        agent2.load_state_dict(torch.load(ckpt_path))

        # Verify identical outputs
        obs, _ = envs.reset(seed=42, options={"num_missing_entities": 1})
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)

        agent.eval()
        agent2.eval()
        with torch.no_grad():
            v1 = agent.get_value(obs_t)
            v2 = agent2.get_value(obs_t)
        assert torch.allclose(v1, v2), "Loaded checkpoint should produce same values"

        envs.close()

    def test_sft_to_ppo_start_from(self, registered_env, tmp_path):
        """SFT checkpoint loaded via --start_from should not crash during RL."""
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, chan1=16, chan2=16, chan3=16, flat_dim=64)

        ckpt_path = str(tmp_path / "test_sft.pt")
        torch.save(agent.state_dict(), ckpt_path)

        # Simulate what ppo.py does with --start_from
        agent2 = AgentCNN(envs, chan1=16, chan2=16, chan3=16, flat_dim=64)
        agent2.load_state_dict(torch.load(ckpt_path))

        # Run a forward pass (simulating RL step)
        obs, _ = envs.reset(seed=42, options={"num_missing_entities": 1})
        obs_t = torch.as_tensor(np.array(obs), dtype=torch.float32)
        action, logprob, entropy, value = agent2.get_action_and_value(obs_t)

        assert "xy" in action
        assert "entity" in action
        assert "direction" in action
        assert logprob.shape == (1,)
        assert value.shape == (1,)

        envs.close()


class TestSFTLossConvergence:
    def test_loss_decreases_on_small_dataset(self, registered_env):
        """SFT loss should decrease when training on a small expert dataset."""
        args = SFTArgs(seed=42, size=5, num_samples=200, max_level=2)
        obs, tiles, ents, dirs = generate_dataset(args)

        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, chan1=16, chan2=16, chan3=16, flat_dim=64)
        envs.close()

        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        ce_loss = torch.nn.CrossEntropyLoss()

        losses = []
        for epoch in range(5):
            agent.train()
            batch_obs = obs.float()
            encoded = agent.encoder(batch_obs)
            B = encoded.shape[0]

            tile_logits = agent.tile_logits(encoded).reshape(B, -1)
            loss_tile = ce_loss(tile_logits, tiles)

            x_B = tiles // agent.height
            y_B = tiles % agent.height
            batch_idx = torch.arange(B)
            tile_features = encoded[batch_idx, :, x_B, y_B]

            ent_logits = agent.ent_head(tile_features)
            dir_logits = agent.dir_head(tile_features)
            loss = loss_tile + ce_loss(ent_logits, ents) + ce_loss(dir_logits, dirs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )
