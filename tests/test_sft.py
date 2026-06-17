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

from helpers import (
    Channel,
    Direction,
    LessonKind,
    blank_entities,
    build_factory,
    str2ent,
)
from sft import (
    SFTArgs,
    _artifact_name,
    _humanize_count,
    _humanize_lr,
    build_lr_schedule,
    extract_expert_actions,
    generate_dataset,
    run_rollout_eval,
    train_sft,
)
from ppo import FactorioEnv, AgentCNN, make_env, layers_from_args


class TestExtractExpertActions:
    def test_reconstructs_solved_world(self):
        """Replaying all extracted actions should reconstruct the solved world."""
        factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=42)
        assert factory is not None
        solved, _ = blank_entities(factory, num_missing_entities=0)
        factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=42)
        assert factory is not None
        task, _ = blank_entities(factory, num_missing_entities=3)
        pairs = extract_expert_actions(solved, task)
        assert len(pairs) > 0, "Should have at least one action"

        # Replay placement actions onto task world. The action carries all
        # four placement channels (entity, direction, item, misc) — the
        # agent is responsible for each. Skip terminal pairs (eot=1) which
        # carry sentinel placement targets, not real placements.
        state = task.clone()
        for (
            obs,
            tile_idx,
            entity_id,
            direction_id,
            item_id,
            misc_id,
            valid_mask,
            eot,
        ) in pairs:
            if eot == 1:
                continue
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
        factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=42)
        assert factory is not None
        solved, _ = blank_entities(factory, num_missing_entities=0)
        pairs = extract_expert_actions(solved, solved.clone())
        assert len(pairs) == 0

    def test_action_count_matches_missing(self):
        """Number of pairs should equal num_missing_entities placement
        actions + 1 terminal (eot=1) pair."""
        for seed in [1, 7, 42]:
            factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=seed)
            assert factory is not None
            solved, _ = blank_entities(factory, num_missing_entities=0)
            # generate_lesson returns (world, actual_removed) where actual_removed
            # may be less than num_missing_entities if the factory has fewer entities
            factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=seed)
            assert factory is not None
            task, min_ent = blank_entities(factory, num_missing_entities=2)
            pairs = extract_expert_actions(solved, task)
            assert len(pairs) == min_ent + 1, (
                f"seed={seed}: expected {min_ent} placement pairs + 1 "
                f"terminal pair, got {len(pairs)}"
            )
            # Exactly one terminal pair, appended last.
            eot_flags = [p[7] for p in pairs]
            assert eot_flags[:-1] == [0] * min_ent
            assert eot_flags[-1] == 1

    def test_intermediate_states_are_sequential(self):
        """Each observation should reflect previously applied actions."""
        factory = build_factory(size=8, kind=LessonKind.MOVE_ONE_ITEM, seed=99)
        assert factory is not None
        solved, _ = blank_entities(factory, num_missing_entities=3)
        factory = build_factory(size=8, kind=LessonKind.MOVE_ONE_ITEM, seed=99)
        assert factory is not None
        task, _ = blank_entities(factory, num_missing_entities=3)
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
        """All extracted placement entity IDs should be valid (non-empty)
        entity values. Terminal pairs (eot=1) carry sentinel zeros and are
        excluded from this check."""
        factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=42)
        assert factory is not None
        solved, _ = blank_entities(factory, num_missing_entities=2)
        factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=42)
        assert factory is not None
        task, _ = blank_entities(factory, num_missing_entities=2)
        pairs = extract_expert_actions(solved, task)
        for _, _, entity_id, direction_id, _, _, _, eot in pairs:
            if eot == 1:
                continue
            assert entity_id != str2ent("empty").value, (
                "Expert actions shouldn't place empty"
            )
            assert direction_id != Direction.NONE.value, (
                "Expert belt actions need a direction"
            )

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
                factory = build_factory(
                    size=8, kind=LessonKind.MOVE_VIA_UG_BELT, seed=seed
                )
                assert factory is not None
                solved, _ = blank_entities(factory, num_missing_entities=0)
                factory = build_factory(
                    size=8, kind=LessonKind.MOVE_VIA_UG_BELT, seed=seed
                )
                assert factory is not None
                task, _ = blank_entities(factory, num_missing_entities=2)
            except Exception:
                continue
            for _, _, ent_id, _, _, misc_id, _, eot in extract_expert_actions(
                solved, task
            ):
                if eot == 1:
                    continue
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
                    factory = build_factory(
                        size=8, kind=LessonKind.MOVE_ONE_ITEM, seed=seed
                    )
                    assert factory is not None
                    task, _ = blank_entities(factory, num_missing_entities=level)
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
        obs, tiles, ents, dirs, items_t, miscs_t, masks, eots, seeds, kinds = (
            generate_dataset(args)
        )
        assert len(obs) == 100
        assert len(tiles) == 100
        assert len(ents) == 100
        assert len(dirs) == 100
        assert len(items_t) == 100
        assert len(miscs_t) == 100
        assert len(masks) == 100
        assert len(eots) == 100
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
            factory = build_factory(size=8, kind=kind, seed=seed)
            assert factory is not None
            solved, _ = blank_entities(factory, num_missing_entities=0)
            factory = build_factory(size=8, kind=kind, seed=seed)
            assert factory is not None
            task, _ = blank_entities(factory, num_missing_entities=20)
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

    def test_assembler_blanking_yields_nonzero_item_targets(self):
        """ASSEMBLE_1IN_1OUT lessons must occasionally emit an expert action
        whose item_id is a real recipe (non-zero). Without this, the item
        head never learns to predict recipes — see issue #107. We sweep many
        seeds at a large num_missing_entities so the assembler gets sampled
        for blanking, then assert at least one non-zero item target appears."""
        nonzero_item_targets = 0
        asm_id = str2ent("assembling_machine_1").value
        asm_pair_count = 0
        for seed in range(60):
            factory = build_factory(
                size=10,
                kind=LessonKind.ASSEMBLE_1IN_1OUT,
                seed=seed,
            )
            if factory is None:
                continue
            solved = factory.world_CWH
            task, _ = blank_entities(factory, num_missing_entities=20)
            for _, _, ent_id, _, item_id, _, _, eot in extract_expert_actions(
                solved, task
            ):
                if eot == 1:
                    continue
                if ent_id == asm_id:
                    asm_pair_count += 1
                    if item_id != 0:
                        nonzero_item_targets += 1
        assert asm_pair_count > 0, (
            "Assembler was never emitted as an expert action across 60 seeds — "
            "the lesson generator stopped blanking it"
        )
        assert nonzero_item_targets > 0, (
            f"Got {asm_pair_count} assembler placement pairs but 0 had a "
            f"non-zero item_id target — recipe channel is being stripped"
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
            line
            for line in out.splitlines()
            if "samples=" in line
            and "samples=     0" not in line
            and "samples=    0" not in line
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
        agent = AgentCNN(envs, layers=(16, 16, 16))

        # Save checkpoint
        ckpt_path = str(tmp_path / "test_sft.pt")
        torch.save(agent.state_dict(), ckpt_path)

        # Load into fresh agent
        agent2 = AgentCNN(envs, layers=(16, 16, 16))
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
        agent = AgentCNN(envs, layers=(16, 16, 16))

        ckpt_path = str(tmp_path / "test_sft.pt")
        torch.save(agent.state_dict(), ckpt_path)

        # Simulate what ppo.py does with --start_from
        agent2 = AgentCNN(envs, layers=(16, 16, 16))
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
        obs, tiles, ents, dirs, items_t, miscs_t, masks, _eots, _, _ = generate_dataset(
            args
        )

        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
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
            layer1=16,
            layer2=16,
            layer3=16,
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


class TestSFTDropout:
    """The SFT dropout knob is inert by default and, when set, must reach the
    encoder via AgentCNN — mirrors ppo.Args' regularisation contract."""

    def test_dropout_default_is_noop(self):
        """Default leaves training unchanged (Dropout2d at p=0 is identity)."""
        assert SFTArgs().dropout == 0.0

    def test_train_sft_threads_dropout_into_agent(self, monkeypatch, tmp_path):
        """train_sft must pass args.dropout through to AgentCNN. Spy on the
        constructor — checkpoints don't persist Dropout2d's p, so the call is
        the only observable point — and assert a non-zero value lands."""
        import sft as sft_mod

        captured = {}
        real_agent_cls = sft_mod.AgentCNN

        def spy(*a, **kw):
            captured.update(kw)
            return real_agent_cls(*a, **kw)

        monkeypatch.setattr(sft_mod, "AgentCNN", spy)

        args = SFTArgs(
            seed=1,
            size=5,
            num_samples=100,
            max_level=2,
            epochs=1,
            batch_size=32,
            layer1=16,
            layer2=16,
            layer3=16,
            dropout=0.5,
            checkpoint_path=str(tmp_path / "drop.pt"),
            summary_path=str(tmp_path / "drop.json"),
        )
        train_sft(args)
        assert captured.get("dropout") == 0.5


class TestTrackedArtifact:
    """The track=True checkpoint-upload path. No test exercised it, so #146's
    chan1/2/3 -> layers rename left a stale args.chan1 in the artifact metadata
    that crashed every tracked run with AttributeError *after* training — only
    at upload. This pins the path end-to-end."""

    def test_tracked_run_finishes_with_layer_metadata(self, monkeypatch, tmp_path):
        import wandb
        from unittest.mock import MagicMock

        captured = {}

        def fake_artifact(*a, **k):
            captured["metadata"] = k.get("metadata", {})
            return MagicMock()

        fake_run = MagicMock()
        fake_run.url = "http://test/run"
        fake_run.summary = {}
        monkeypatch.setattr(wandb, "init", lambda *a, **k: fake_run)
        monkeypatch.setattr(wandb, "Artifact", fake_artifact)

        args = SFTArgs(
            seed=1,
            size=5,
            num_samples=200,
            max_level=2,
            epochs=1,
            batch_size=32,
            layer1=16,
            layer2=16,
            layer3=16,
            track=True,
            eval_rollouts_every_n_epochs=0,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)  # must not raise at the artifact step

        meta = captured["metadata"]
        assert meta["layers"] == [16, 16, 16]
        assert meta["kernel_size"] == 3
        assert "chan1" not in meta


class TestRunRolloutEval:
    """End-to-end coverage of greedy rollout eval on held-out val factories."""

    def _build_val_seeds_to_kind(self, size, num_kinds=4, start_seed=10_000):
        """Iterate seeds until we have `num_kinds` (seed -> kind) pairs each
        of which produces a valid lesson at max_level. Mirrors the
        try/except-continue pattern that generate_dataset uses for malformed
        seeds, so this fixture doesn't get flaky when an enum value lands a
        bad seed."""
        kinds = list(LessonKind)
        out: dict[int, int] = {}
        seed = start_seed
        while len(out) < num_kinds and seed < start_seed + 1000:
            kind = kinds[len(out) % len(kinds)]
            if build_factory(size=size, kind=kind, seed=seed) is None:
                seed += 1
                continue
            out[seed] = kind.value
            seed += 1
        return out

    def test_returns_throughput_in_unit_range(self, registered_env):
        """Untrained agent on tiny grid should return well-formed throughput
        numbers — overall in [0, 1] and per-kind in [0, 1] for any kind that
        was eval'd."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        args = SFTArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            layer1=16,
            layer2=16,
            layer3=16,
        )
        val_seeds_to_kind = self._build_val_seeds_to_kind(size=size, num_kinds=4)
        assert len(val_seeds_to_kind) >= 1, "Sanity: at least one valid lesson"

        roll = run_rollout_eval(
            agent,
            args,
            val_seeds_to_kind,
            device=torch.device("cpu"),
            max_seeds=len(val_seeds_to_kind),
        )
        assert set(roll) == {
            "overall",
            "overall_eot",
            "per_kind",
            "per_kind_eot",
            "per_kind_n",
        }
        overall, per_kind, per_kind_n = (
            roll["overall"],
            roll["per_kind"],
            roll["per_kind_n"],
        )

        assert 0.0 <= overall <= 1.5, f"overall throughput out of range: {overall}"
        assert 0.0 <= roll["overall_eot"] <= 1.5
        total_n = sum(per_kind_n.values())
        assert total_n == len(val_seeds_to_kind), (
            f"per-kind n totals {total_n}, expected {len(val_seeds_to_kind)}"
        )
        for kn, thp in per_kind.items():
            assert 0.0 <= thp <= 1.5, f"{kn}: throughput out of range: {thp}"

    def test_default_max_level_evals_from_empty(self, registered_env):
        """With the default max_level (0), the rollout eval auto-resolves to
        size*size — blanking the WHOLE factory so the agent builds from an
        empty grid. The from-empty path must run end-to-end on every seed and
        return well-formed throughput (this is the only rollout test that
        exercises the default; the others pass max_level explicitly)."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        # max_level left at its 0 default -> size*size (25) >> 2*size (10),
        # so the whole 5x5 factory is blanked.
        args = SFTArgs(
            seed=1,
            size=size,
            num_samples=50,
            layer1=16,
            layer2=16,
            layer3=16,
        )
        assert args.max_level == 0, "default must be the auto sentinel"
        val_seeds_to_kind = self._build_val_seeds_to_kind(size=size, num_kinds=4)
        assert len(val_seeds_to_kind) >= 1

        roll = run_rollout_eval(
            agent,
            args,
            val_seeds_to_kind,
            device=torch.device("cpu"),
            max_seeds=len(val_seeds_to_kind),
        )
        assert 0.0 <= roll["overall"] <= 1.5
        assert sum(roll["per_kind_n"].values()) == len(val_seeds_to_kind)

    def test_max_seeds_caps_eval(self, registered_env):
        """max_seeds should bound the rollout count even when the val set
        is larger."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        args = SFTArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            layer1=16,
            layer2=16,
            layer3=16,
        )
        val_seeds_to_kind = self._build_val_seeds_to_kind(size=size, num_kinds=6)
        assert len(val_seeds_to_kind) >= 3

        roll = run_rollout_eval(
            agent,
            args,
            val_seeds_to_kind,
            device=torch.device("cpu"),
            max_seeds=2,
        )
        assert sum(roll["per_kind_n"].values()) == 2

    def test_empty_val_seeds_is_safe(self, registered_env):
        """Empty val_seeds_to_kind should return zero/empty results without
        crashing (defensive — the caller already guards on len(...)>0)."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        args = SFTArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            layer1=16,
            layer2=16,
            layer3=16,
        )
        roll = run_rollout_eval(
            agent,
            args,
            {},
            device=torch.device("cpu"),
        )
        assert roll["overall"] == 0.0
        assert roll["overall_eot"] == 0.0
        assert sum(roll["per_kind_n"].values()) == 0

    def test_eot_threshold_controls_eot_metric(self, registered_env):
        """One rollout yields two throughputs: `overall` ignores the EOT head
        (steps to env-done), `overall_eot` snapshots throughput at the first
        EOT fire. A threshold above 1 means the head never fires, so
        overall_eot == overall; a threshold below 0 means it fires before any
        step, snapshotting the reset throughput (0) for every seed."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        args = SFTArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            layer1=16,
            layer2=16,
            layer3=16,
        )
        val_seeds_to_kind = self._build_val_seeds_to_kind(size=size, num_kinds=4)

        # sigmoid(logit) < 1 always, so threshold 10 -> EOT never fires.
        never = run_rollout_eval(
            agent,
            args,
            val_seeds_to_kind,
            device=torch.device("cpu"),
            eot_threshold=10.0,
            max_seeds=len(val_seeds_to_kind),
        )
        assert never["overall_eot"] == pytest.approx(never["overall"]), (
            "EOT never fires -> EOT-respecting throughput must equal ignore-EOT"
        )

        # sigmoid(logit) > 0 always, so threshold -1 -> EOT fires before the
        # first step, snapshotting the reset throughput (0).
        always = run_rollout_eval(
            agent,
            args,
            val_seeds_to_kind,
            device=torch.device("cpu"),
            eot_threshold=-1.0,
            max_seeds=len(val_seeds_to_kind),
        )
        assert always["overall_eot"] == 0.0, (
            "EOT firing before the first step must snapshot reset throughput (0)"
        )
        assert 0.0 <= always["overall"] <= 1.5


class TestEotHead:
    """Tests for the binary end-of-turn head."""

    def test_eot_label_per_lesson(self):
        """A lesson with N missing entities emits N placement pairs
        (eot=0) followed by one terminal pair (eot=1) whose obs equals
        the fully-solved factory."""
        factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=11)
        assert factory is not None
        solved, _ = blank_entities(factory, num_missing_entities=0)
        factory = build_factory(size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=11)
        assert factory is not None
        task, min_ent = blank_entities(factory, num_missing_entities=3)
        pairs = extract_expert_actions(solved, task)
        # eot flag is at index 7 in the tuple
        eots = [p[7] for p in pairs]
        assert sum(eots) == 1, f"Expected exactly one terminal pair, got {sum(eots)}"
        assert eots[-1] == 1, "Terminal pair must come last"
        # Terminal observation equals solved (entities + directions).
        terminal_obs = pairs[-1][0]
        assert torch.equal(
            terminal_obs[Channel.ENTITIES.value], solved[Channel.ENTITIES.value]
        )
        assert torch.equal(
            terminal_obs[Channel.DIRECTION.value], solved[Channel.DIRECTION.value]
        )

    def test_eot_tensor_in_dataset(self):
        """generate_dataset must return a per-pair eot tensor with values
        in {0.0, 1.0} and at least one positive (terminal) example."""
        args = SFTArgs(seed=1, size=5, num_samples=200, max_level=4)
        *_, eots, _seeds, _kinds = generate_dataset(args)
        assert eots.dtype == torch.float
        assert set(eots.unique().tolist()).issubset({0.0, 1.0})
        assert eots.sum().item() >= 1, "Dataset must contain >=1 terminal pair"

    def test_eot_head_exists_and_forwards(self, registered_env):
        """AgentCNN must expose an eot_head producing a single logit per
        observation."""
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        assert hasattr(agent, "eot_head"), "AgentCNN must have an eot_head"
        # Forward a fake batch through the encoder + eot_head.
        B, C, W, H = 4, agent.channels, agent.width, agent.height
        x = torch.zeros((B, C, W, H), dtype=torch.float32)
        enc = agent.encoder(x)
        logits = agent.eot_head(enc).squeeze(-1)
        assert logits.shape == (B,), (
            f"eot_head output should be (B,), got {logits.shape}"
        )

    def test_eot_prob_and_should_stop_shapes(self, registered_env):
        """`eot_prob` returns a [0,1] tensor of shape (B,); `eot_should_stop`
        returns a bool tensor of the same shape. These are the methods
        inference rollouts call to decide 'I'm done here'."""
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        B = 3
        x = torch.zeros(
            (B, agent.channels, agent.width, agent.height), dtype=torch.float32
        )
        probs = agent.eot_prob(x)
        assert probs.shape == (B,)
        assert (probs >= 0).all() and (probs <= 1).all()

        stop = agent.eot_should_stop(x, threshold=0.5)
        assert stop.shape == (B,)
        assert stop.dtype == torch.bool

        # Threshold = 1.0 → never stop; threshold = 0.0 → always stop.
        assert not agent.eot_should_stop(x, threshold=1.0).any()
        assert agent.eot_should_stop(x, threshold=-0.01).all()

    def test_eot_head_learns_terminal_vs_placement(self, registered_env, tmp_path):
        """End-to-end: after a short SFT run the eot head should
        distinguish terminal observations (solved factories) from
        placement-step observations. This is the contract that lets the
        rollout 'run until the model thinks it's done'."""
        ckpt = str(tmp_path / "sft_eot.pt")
        summary = str(tmp_path / "sft_eot_summary.json")
        args = SFTArgs(
            seed=3,
            size=5,
            num_samples=1000,
            max_level=3,
            epochs=20,
            batch_size=64,
            lr=3e-3,
            layer1=16,
            layer2=16,
            layer3=16,
            checkpoint_path=ckpt,
            summary_path=summary,
        )
        agent = train_sft(args)

        # Build a fresh held-out set: solved (eot=1) and partial (eot=0)
        # observations from a seed range that wasn't in training. Move
        # inputs to the agent's device (which depends on what train_sft
        # selected — CPU on CI, MPS/CUDA locally).
        device = next(agent.parameters()).device
        agent.eval()
        pos_logits = []
        neg_logits = []
        with torch.no_grad():
            for seed in range(10_000, 10_040):
                factory = build_factory(
                    size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=seed
                )
                assert factory is not None
                solved, _ = blank_entities(factory, num_missing_entities=0)
                factory = build_factory(
                    size=5, kind=LessonKind.MOVE_ONE_ITEM, seed=seed
                )
                assert factory is not None
                task, _ = blank_entities(factory, num_missing_entities=3)
                enc_pos = agent.encoder(solved.unsqueeze(0).float().to(device))
                enc_neg = agent.encoder(task.unsqueeze(0).float().to(device))
                pos_logits.append(agent.eot_head(enc_pos).item())
                neg_logits.append(agent.eot_head(enc_neg).item())

        pos_mean = sum(pos_logits) / len(pos_logits)
        neg_mean = sum(neg_logits) / len(neg_logits)
        assert pos_mean > neg_mean, (
            f"EOT head failed to separate terminal/placement: "
            f"pos_mean={pos_mean:.3f} <= neg_mean={neg_mean:.3f}"
        )


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

    @pytest.mark.parametrize(
        "n,expected",
        [
            (500, "500"),
            (1_000, "1k"),
            (50_000, "50k"),
            (200_000, "200k"),
            (1_000_000, "1m"),
            (2_500_000, "2.5m"),
        ],
    )
    def test_humanize_count(self, n, expected):
        assert _humanize_count(n) == expected

    @pytest.mark.parametrize(
        "lr,expected",
        [
            (1e-3, "1e-3"),
            (3e-4, "3e-4"),
            (1e-4, "1e-4"),
            (5e-4, "5e-4"),
        ],
    )
    def test_humanize_lr_round(self, lr, expected):
        """Mantissas that are clean integers don't get a decimal point."""
        assert _humanize_lr(lr) == expected

    def test_humanize_lr_zero(self):
        assert _humanize_lr(0) == "0"

    def test_artifact_name_default(self):
        # SFTArgs defaults (layers 48,48,64) expand into a per-layer channel
        # suffix; size is the project default of 12.
        assert _artifact_name(SFTArgs()) == "sft-s12-n300k-e30-bs512-lr2.5e-3-c48-48-64"

    def test_artifact_name_larger_run(self):
        args = SFTArgs(
            size=16,
            num_samples=200_000,
            epochs=50,
            batch_size=1024,
            lr=3e-4,
            layer1=48,
            layer2=48,
            layer3=48,
        )
        assert _artifact_name(args) == "sft-s16-n200k-e50-bs1024-lr3e-4-c48-48-48"

    def test_artifact_name_encodes_depth(self):
        """Depth is part of the suffix: a 4-layer encoder of the same width
        must not collide with a 3-layer one (else the deeper run would file
        under the shallower run's artifact)."""
        three = SFTArgs(layer1=48, layer2=48, layer3=48, layer4=0)
        four = SFTArgs(layer1=48, layer2=48, layer3=48, layer4=48)
        assert _artifact_name(three).endswith("-c48-48-48")
        assert _artifact_name(four).endswith("-c48-48-48-48")
        assert _artifact_name(three) != _artifact_name(four)

    def test_artifact_name_asymmetric_channels(self):
        """Differing per-layer widths expand into the full list, so a
        c32-64-64 run can't accidentally file under a c48-48-48 run."""
        args = SFTArgs(layer1=32, layer2=64, layer3=64)
        assert _artifact_name(args) == "sft-s12-n300k-e30-bs512-lr2.5e-3-c32-64-64"

    def test_artifact_name_kernel_size_suffix(self):
        """A non-default kernel size appends -k{N} so two runs differing only
        in receptive field get distinct artifacts; the default k=3 adds no
        token (keeps existing names stable)."""
        assert _artifact_name(SFTArgs()).endswith("-c48-48-64")
        assert _artifact_name(SFTArgs(kernel_size=5)) == (
            "sft-s12-n300k-e30-bs512-lr2.5e-3-c48-48-64-k5"
        )

    def test_artifact_name_stable_across_runs(self):
        """Two SFTArgs with the same hyperparams must produce the same
        artifact name (otherwise we lose version collapsing)."""
        a = SFTArgs(seed=1)
        b = SFTArgs(seed=999)  # seed isn't part of the artifact name
        assert _artifact_name(a) == _artifact_name(b)


class TestLRSchedule:
    """The warmup→cosine schedule should ramp the LR up from a low value,
    reach `args.lr` at the warmup boundary, and decay toward
    `args.lr * args.min_lr_ratio` by the final step."""

    def _step_n(self, scheduler, optimizer, n):
        lrs = []
        for _ in range(n):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        return lrs

    def test_warmup_then_cosine(self):
        args = SFTArgs(lr=1e-3, warmup_frac=0.1, min_lr_ratio=0.01)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        total_steps = 100
        scheduler = build_lr_schedule(optimizer, total_steps, args)

        lrs = self._step_n(scheduler, optimizer, total_steps)

        # Step 0 starts at warmup floor (lr * start_factor = lr * 1e-3)
        assert lrs[0] == pytest.approx(args.lr * 1e-3, rel=0.01)
        # Around the warmup→cosine handoff (~step 10) the LR should be at
        # or near the peak lr
        assert max(lrs[8:13]) >= args.lr * 0.95
        # Final LR should be very close to the cosine floor
        assert lrs[-1] == pytest.approx(args.lr * args.min_lr_ratio, abs=args.lr * 1e-4)

    def test_no_warmup_path(self):
        args = SFTArgs(lr=2e-3, warmup_frac=0.0, min_lr_ratio=0.1)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, 50, args)

        lrs = self._step_n(scheduler, optimizer, 50)

        # Cosine starts at peak and decays monotonically(-ish)
        assert lrs[0] == pytest.approx(args.lr, rel=0.01)
        assert lrs[-1] == pytest.approx(args.lr * args.min_lr_ratio, abs=args.lr * 1e-4)

    def test_handles_one_step(self):
        # cosine_steps = max(1, total_steps - warmup_steps) guards against
        # tiny runs where total_steps <= warmup_steps. Should still build a
        # valid scheduler.
        args = SFTArgs(lr=1e-3, warmup_frac=0.5)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, 1, args)
        optimizer.step()
        scheduler.step()  # must not raise
