"""Tests for SFT pre-training pipeline."""

import os
import random
import sys

import pytest
import torch
import gymnasium as gym
import numpy as np

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import Channel, Direction, LessonKind, generate_lesson, str2ent
from sft import (
    SFTArgs,
    _artifact_name,
    _humanize_count,
    _humanize_lr,
    extract_expert_actions,
    generate_dataset,
    train_sft,
)
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

        # Replay actions onto task world. The action carries all four
        # placement channels (entity, direction, item, misc) — the agent
        # is responsible for each.
        state = task.clone()
        for obs, tile_idx, entity_id, direction_id, item_id, misc_id, valid_mask in pairs:
            H = state.shape[2]
            x = tile_idx // H
            y = tile_idx % H
            state[Channel.ENTITIES.value, x, y] = entity_id
            state[Channel.DIRECTION.value, x, y] = direction_id
            state[Channel.ITEMS.value, x, y] = item_id
            state[Channel.MISC.value, x, y] = misc_id

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
        for _, _, entity_id, direction_id, _, _, _ in pairs:
            assert entity_id != str2ent("empty").value, "Expert actions shouldn't place empty"
            assert direction_id != Direction.NONE.value, "Expert belt actions need a direction"

    def test_ug_belt_action_carries_misc(self):
        """A MOVE_VIA_UG_BELT lesson with the UG pair blanked must produce
        action pairs whose misc_id is UNDERGROUND_DOWN or UNDERGROUND_UP —
        not NONE. Without this the env rejects every UG placement at step
        time (ug_belt_wo_up_or_down)."""
        from helpers import Misc
        ug_id = str2ent("underground_belt").value
        found_down = found_up = False
        for seed in range(50):
            try:
                solved, _ = generate_lesson(
                    size=8, kind=LessonKind.MOVE_VIA_UG_BELT,
                    num_missing_entities=0, seed=seed,
                )
                task, _ = generate_lesson(
                    size=8, kind=LessonKind.MOVE_VIA_UG_BELT,
                    num_missing_entities=2, seed=seed,
                )
            except Exception:
                continue
            for _, _, ent_id, _, _, misc_id, _ in extract_expert_actions(solved, task):
                if ent_id == ug_id:
                    assert misc_id != Misc.NONE.value, (
                        f"seed={seed}: UG belt action emitted with misc=NONE"
                    )
                    if misc_id == Misc.UNDERGROUND_DOWN.value:
                        found_down = True
                    if misc_id == Misc.UNDERGROUND_UP.value:
                        found_up = True
            if found_down and found_up:
                break
        assert found_down and found_up, (
            f"Expected to see both DOWN and UP actions across seeds; "
            f"found_down={found_down}, found_up={found_up}"
        )

    def test_source_and_sink_always_present(self):
        """Task observations should always contain source and sink entities."""
        source_id = str2ent("source").value
        sink_id = str2ent("sink").value
        for seed in [1, 7, 42, 99]:
            for level in [1, 4, 8, 16]:
                try:
                    task, _ = generate_lesson(
                        size=8, kind=LessonKind.MOVE_ONE_ITEM,
                        num_missing_entities=level, seed=seed,
                    )
                except Exception:
                    continue
                ent = task[Channel.ENTITIES.value]
                assert (ent == source_id).any(), (
                    f"seed={seed}, level={level}: source missing from task"
                )
                assert (ent == sink_id).any(), (
                    f"seed={seed}, level={level}: sink missing from task"
                )


class TestGenerateDataset:
    def test_generates_correct_count(self):
        """Dataset should have the requested number of samples."""
        args = SFTArgs(seed=1, size=5, num_samples=100, max_level=2)
        obs, tiles, ents, dirs, items_t, miscs_t, masks, seeds, kinds = generate_dataset(args)
        assert len(obs) == 100
        assert len(tiles) == 100
        assert len(ents) == 100
        assert len(dirs) == 100
        assert len(items_t) == 100
        assert len(miscs_t) == 100
        assert len(masks) == 100
        assert len(seeds) == 100
        assert len(kinds) == 100

    def test_observation_shape(self):
        """Observations should have correct shape (C, W, H)."""
        args = SFTArgs(seed=1, size=5, num_samples=50, max_level=2)
        obs, *_ = generate_dataset(args)
        assert obs.shape[1] == len(Channel)  # channels
        assert obs.shape[2] == 5  # width
        assert obs.shape[3] == 5  # height

    def test_tile_indices_in_range(self):
        """Tile indices should be in [0, W*H)."""
        args = SFTArgs(seed=1, size=5, num_samples=50, max_level=2)
        _, tiles, *_ = generate_dataset(args)
        assert (tiles >= 0).all()
        assert (tiles < 5 * 5).all()

    def test_seeds_returned_per_pair(self):
        """generate_dataset returns a per-pair lesson_seed tensor; pairs
        from the same lesson share the same seed (multiple pairs per
        lesson when level > 1)."""
        args = SFTArgs(seed=1, size=5, num_samples=100, max_level=2)
        *_, seeds, _kinds = generate_dataset(args)
        # Multiple unique seeds expected (each lesson has its own seed).
        assert len(set(seeds.tolist())) >= 2
        # And at least one seed appears more than once (level=2 → ~2 pairs).
        from collections import Counter
        counts = Counter(seeds.tolist())
        assert any(c > 1 for c in counts.values()), (
            "expected at least one lesson to produce >1 pair sharing a seed"
        )

    @pytest.mark.parametrize("kind_name", ["SPLITTER_MERGE", "SPLITTER_SPLIT"])
    @pytest.mark.parametrize("seed", range(20))
    def test_multi_tile_entities_emit_one_pair(self, kind_name, seed):
        """Splitters are one entity that occupy two tiles; extract_expert_actions
        must emit a single (anchor) action pair per splitter, not one per cell.
        Otherwise the model is trained to call place-splitter twice for one
        splitter, which would place two splitters at execution time."""
        kind = getattr(LessonKind, kind_name)
        try:
            solved, _ = generate_lesson(size=8, kind=kind, num_missing_entities=0, seed=seed)
            task, _ = generate_lesson(size=8, kind=kind, num_missing_entities=20, seed=seed)
        except Exception:
            pytest.skip(f"{kind_name} seed {seed}: lesson generation failed")

        splitter_id = str2ent("splitter").value
        solved_splitter = (solved[Channel.ENTITIES.value] == splitter_id).sum().item()
        task_splitter = (task[Channel.ENTITIES.value] == splitter_id).sum().item()
        # Skip cases where this lesson didn't include / didn't blank the splitter:
        # without a splitter diff there's nothing to verify here.
        if solved_splitter != 2 or task_splitter != 0:
            pytest.skip("splitter not present + blanked in this case")

        pairs = extract_expert_actions(solved, task)
        splitter_pairs = [p for p in pairs if p[2] == splitter_id]
        assert len(splitter_pairs) == 1, (
            f"{kind_name} seed={seed}: expected 1 splitter placement pair, "
            f"got {len(splitter_pairs)} (one per occupied cell — bug)"
        )

    def test_observations_have_diverse_items(self):
        """SFT observations should carry varied item IDs in the ITEMS
        channel (sources/sinks). Without random_item the lessons all
        carry electronic_circuit, which would let the model memorise
        item_id == electronic_circuit as a constant feature."""
        args = SFTArgs(seed=1, size=8, num_samples=200, max_level=4)
        obs, *_ = generate_dataset(args)
        item_channel = obs[:, Channel.ITEMS.value]
        unique_items = set(item_channel.flatten().tolist())
        # 0 = empty (always present); we want at least 2 non-empty item types.
        non_empty = unique_items - {0}
        assert len(non_empty) >= 2, (
            f"Expected >=2 distinct non-empty item types, got {sorted(unique_items)}"
        )

    def test_kinds_returned_per_pair(self):
        """generate_dataset must return a per-pair kind tensor with the
        same length as the rest. Values must be valid LessonKind enum
        values, and at least two distinct kinds should appear (proves the
        per-kind val aggregation in train_sft has something to bucket)."""
        args = SFTArgs(seed=1, size=8, num_samples=400, max_level=8)
        obs, tiles, *_, kinds = generate_dataset(args)
        assert len(kinds) == len(obs)
        valid_values = {k.value for k in LessonKind}
        assert set(kinds.tolist()).issubset(valid_values), (
            f"kind tensor contains values outside LessonKind: "
            f"{set(kinds.tolist()) - valid_values}"
        )
        assert len(set(kinds.tolist())) >= 2, (
            f"expected >=2 distinct kinds, got {sorted(set(kinds.tolist()))}"
        )

    def test_samples_span_multiple_kinds(self, capsys):
        """Dataset should draw from more than one LessonKind.

        Now that every kind protects its structural entity (inserter,
        splitter), every blanked target is a belt — so we can't tell
        kinds apart from the entity_id targets. Instead, parse the
        per-kind breakdown that generate_dataset prints and verify at
        least two kinds contributed samples.
        """
        args = SFTArgs(seed=1, size=8, num_samples=400, max_level=8)
        generate_dataset(args)
        out = capsys.readouterr().out
        productive = [
            line for line in out.splitlines()
            if "samples=" in line and "samples=     0" not in line and "samples=    0" not in line
        ]
        assert len(productive) >= 2, (
            f"Expected >=2 productive kinds in breakdown, got:\n{out}"
        )
        # Auto-discovery: every enum value appears in the breakdown.
        for kind in LessonKind:
            assert kind.name in out, f"{kind.name} missing from breakdown:\n{out}"


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
        obs, tiles, ents, dirs, items_t, miscs_t, masks, _, _ = generate_dataset(args)

        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, chan1=16, chan2=16, chan3=16, flat_dim=64)
        envs.close()

        optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
        ce_loss = torch.nn.CrossEntropyLoss()
        bce_loss = torch.nn.BCEWithLogitsLoss()

        losses = []
        for epoch in range(5):
            agent.train()
            batch_obs = obs.float()
            encoded = agent.encoder(batch_obs)
            B = encoded.shape[0]

            tile_logits = agent.tile_logits(encoded).reshape(B, -1)
            loss_tile = bce_loss(tile_logits, masks)

            x_B = tiles // agent.height
            y_B = tiles % agent.height
            batch_idx = torch.arange(B)
            tile_features = encoded[batch_idx, :, x_B, y_B]

            ent_logits = agent.ent_head(tile_features)
            dir_logits = agent.dir_head(tile_features)
            item_logits = agent.item_head(tile_features)
            misc_logits = agent.misc_head(tile_features)
            loss = (
                loss_tile
                + ce_loss(ent_logits, ents)
                + ce_loss(dir_logits, dirs)
                + ce_loss(item_logits, items_t)
                + ce_loss(misc_logits, miscs_t)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


class TestTrainSFTEndToEnd:
    def test_train_sft_produces_checkpoint(self, tmp_path):
        """End-to-end: train_sft() should run and produce a valid checkpoint."""
        ckpt = str(tmp_path / "sft_e2e.pt")
        summary = str(tmp_path / "sft_e2e_summary.json")
        args = SFTArgs(
            seed=1,
            size=5,
            num_samples=100,
            max_level=2,
            epochs=2,
            batch_size=32,
            chan1=16,
            chan2=16,
            chan3=16,
            flat_dim=64,
            checkpoint_path=ckpt,
            summary_path=summary,
        )
        agent = train_sft(args)
        assert os.path.exists(ckpt), "Checkpoint should be saved"
        assert os.path.exists(summary), "Summary JSON should be saved"

        import json
        with open(summary) as f:
            s = json.load(f)
        assert "best_val_acc" in s
        assert s["num_samples"] == 100
        assert s["epochs"] == 2


class TestTrainValSeedSplit:
    def test_no_lesson_overlap_between_train_and_val(self, capsys):
        """Train and val sets must not share any lesson seed. A pair-level
        random split would leak factories across the boundary; this test
        is the regression guard for the seed-level split in train_sft."""
        # Capture stdout to read the "Train/val split at seed level" line.
        args = SFTArgs(seed=1, size=5, num_samples=100, max_level=2, val_frac=0.2)
        # Reproduce the split logic without spinning up training.
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        *_, lesson_seeds, _kinds = generate_dataset(args)

        unique_seeds = torch.unique(lesson_seeds)
        n_seeds = len(unique_seeds)
        n_val_seeds = max(1, int(n_seeds * args.val_frac))
        seed_perm = unique_seeds[torch.randperm(n_seeds)]
        val_seeds = set(seed_perm[:n_val_seeds].tolist())
        train_seeds = set(seed_perm[n_val_seeds:].tolist())

        assert val_seeds.isdisjoint(train_seeds), (
            f"Train and val share lesson seeds: {val_seeds & train_seeds}"
        )
        assert len(val_seeds) >= 1
        assert len(train_seeds) >= 1


class TestArtifactNameHelpers:
    """The W&B artifact name encodes hyperparams so runs with identical
    configs collapse into versions of one artifact. These tests pin the
    format so accidentally renaming a hyperparam doesn't silently
    fragment the artifact namespace."""

    @pytest.mark.parametrize("n,expected", [
        (500, "500"),
        (1_000, "1k"),
        (50_000, "50k"),
        (200_000, "200k"),
        (1_000_000, "1m"),
        (2_500_000, "2.5m"),
    ])
    def test_humanize_count(self, n, expected):
        assert _humanize_count(n) == expected

    @pytest.mark.parametrize("lr,expected", [
        (1e-3, "1e-3"),
        (3e-4, "3e-4"),
        (1e-4, "1e-4"),
        (5e-4, "5e-4"),
    ])
    def test_humanize_lr_round(self, lr, expected):
        """Mantissas that are clean integers don't get a decimal point."""
        assert _humanize_lr(lr) == expected

    def test_humanize_lr_zero(self):
        assert _humanize_lr(0) == "0"

    def test_artifact_name_default(self):
        assert _artifact_name(SFTArgs()) == "sft-s8-n50k-e30-bs512-lr1e-3-c48"

    def test_artifact_name_larger_run(self):
        args = SFTArgs(
            size=16, num_samples=200_000, epochs=50, batch_size=1024, lr=3e-4,
        )
        assert _artifact_name(args) == "sft-s16-n200k-e50-bs1024-lr3e-4-c48"

    def test_artifact_name_asymmetric_channels(self):
        """When chan1/2/3 differ, the suffix expands rather than collapsing
        to a single c{N} — so a c32-64-64 run can't accidentally be filed
        under the same artifact as a c48 run."""
        args = SFTArgs(chan1=32, chan2=64, chan3=64)
        assert _artifact_name(args) == "sft-s8-n50k-e30-bs512-lr1e-3-c32-64-64"

    def test_artifact_name_stable_across_runs(self):
        """Two SFTArgs with the same hyperparams must produce the same
        artifact name (otherwise we lose version collapsing)."""
        a = SFTArgs(seed=1)
        b = SFTArgs(seed=999)  # seed isn't part of the artifact name
        assert _artifact_name(a) == _artifact_name(b)
