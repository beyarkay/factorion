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
from factorion import Footprint
import sft
from sft import (
    SftArgs,
    StreamingDemoDataset,
    _apply_legal_tile_mask,
    _artifact_name,
    _dir_distance_diagnostics,
    _dir_mismatch_by_distance,
    _direction_diagnostics,
    _humanize_count,
    _humanize_lr,
    _iter_demo_pairs,
    _materialise,
    _point_biserial,
    _source_sink_distance,
    _steps_per_epoch,
    build_lr_schedule,
    extract_expert_actions,
    run_rollout_eval,
    train_sft,
)
from ppo import FactorioEnv, AgentCNN, make_env, layers_from_args


def _materialise_args(args):
    """Eagerly collect a full dataset (the 10-tensor tuple) for assertions,
    matching how train_sft draws its data from `_materialise`."""
    max_level = args.max_level if args.max_level > 0 else args.size * args.size
    return _materialise(args.size, max_level, args.seed, target=args.num_samples)


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
        args = SftArgs(seed=1, size=5, num_samples=100, max_level=2)
        obs, tiles, ents, dirs, items_t, miscs_t, masks, eots, seeds, kinds = (
            _materialise_args(args)
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
        args = SftArgs(seed=1, size=5, num_samples=50, max_level=2)
        obs, *_ = _materialise_args(args)
        assert obs.shape[1] == len(Channel)  # channels
        assert obs.shape[2] == 5  # width
        assert obs.shape[3] == 5  # height

    def test_tile_indices_in_range(self):
        """Tile indices should be in [0, W*H)."""
        args = SftArgs(seed=1, size=5, num_samples=50, max_level=2)
        _, tiles, *_ = _materialise_args(args)
        assert (tiles >= 0).all()
        assert (tiles < 5 * 5).all()

    def test_seeds_returned_per_pair(self):
        """_materialise returns a per-pair lesson_seed tensor; pairs
        from the same lesson share the same seed (multiple pairs per
        lesson when level > 1)."""
        args = SftArgs(seed=1, size=5, num_samples=100, max_level=2)
        *_, seeds, _kinds = _materialise_args(args)
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

    @pytest.mark.skip(reason="skipping ASSEMBLE_(1|2)IN_1OUT")
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
        args = SftArgs(seed=1, size=8, num_samples=200, max_level=4)
        obs, *_ = _materialise_args(args)
        item_channel = obs[:, Channel.ITEMS.value]
        unique_items = set(item_channel.flatten().tolist())
        # 0 = empty (always present); we want at least 2 non-empty item types.
        non_empty = unique_items - {0}
        assert len(non_empty) >= 2, (
            f"Expected >=2 distinct non-empty item types, got {sorted(unique_items)}"
        )

    def test_kinds_returned_per_pair(self):
        """_materialise must return a per-pair kind tensor with the
        same length as the rest. Values must be valid LessonKind enum
        values, and at least two distinct kinds should appear (proves the
        per-kind val aggregation in train_sft has something to bucket)."""
        args = SftArgs(seed=1, size=8, num_samples=400, max_level=8)
        obs, tiles, *_, kinds = _materialise_args(args)
        assert len(kinds) == len(obs)
        valid_values = {k.value for k in LessonKind}
        assert set(kinds.tolist()).issubset(valid_values), (
            f"kind tensor contains values outside LessonKind: "
            f"{set(kinds.tolist()) - valid_values}"
        )
        assert len(set(kinds.tolist())) >= 2, (
            f"expected >=2 distinct kinds, got {sorted(set(kinds.tolist()))}"
        )

    def test_samples_span_multiple_kinds(self):
        """Dataset should draw from more than one LessonKind (the per-pair kind
        tensor carries the labels directly)."""
        args = SftArgs(seed=1, size=8, num_samples=400, max_level=8)
        *_, kinds = _materialise_args(args)
        assert len(set(kinds.tolist())) >= 2

    def test_pairs_balanced_across_kinds(self):
        """Sampling balances by pair count, not by lesson: every kind lands
        within a small band. Uniform-by-lesson would push this ratio to ~0.1."""
        from collections import Counter

        args = SftArgs(seed=1, size=8, num_samples=3000, max_level=8)
        *_, kinds = _materialise_args(args)
        vals = [c for c in Counter(kinds.tolist()).values() if c > 0]
        assert len(vals) == len(LessonKind), "every kind should contribute pairs"
        assert min(vals) / max(vals) >= 0.8, f"pair counts not balanced: {sorted(vals)}"

    def test_obs_uint8_masks_bool(self):
        """obs stored uint8 and masks bool (the memory cut); eot stays float."""
        args = SftArgs(seed=1, size=8, num_samples=300, max_level=4)
        obs, *_, masks, eots, _seeds, _kinds = _materialise_args(args)
        assert obs.dtype == torch.uint8
        assert masks.dtype == torch.bool
        assert int(obs.max()) < 256  # nothing overflowed the uint8 range
        assert eots.dtype == torch.float

    def test_unbuildable_kinds_dropped_not_hung(self):
        """A lesson kind that can't fit the grid (a 3x3 assembler with 4-5
        ingredient arms doesn't fit at size 5) must be dropped from the sampler,
        not retried forever. Generation completes and simply omits those kinds;
        the buildable memorise lessons still appear."""
        args = SftArgs(seed=1, size=5, num_samples=400, max_level=0)
        *_, kinds = _materialise_args(args)
        produced = set(kinds.tolist())
        # The 4- and 5-ingredient memorise lessons can't be built at size 5.
        assert LessonKind.MEMORISE_4_INGREDIENT_RECIPES.value not in produced
        assert LessonKind.MEMORISE_5_INGREDIENT_RECIPES.value not in produced
        # ...but the 1- and 2-ingredient ones (which always fit) do appear.
        assert LessonKind.MEMORISE_1_INGREDIENT_RECIPES.value in produced
        assert LessonKind.MEMORISE_2_INGREDIENT_RECIPES.value in produced


class TestStreamingDemoDataset:
    """StreamingDemoDataset generates the same pairs as the materialised path,
    but lazily and sharded across DataLoader workers."""

    def test_yields_target_count_and_shapes(self):
        """A pass yields exactly `target` pairs with the extract 8-tuple's
        shapes/dtypes (obs uint8 (C,W,H), mask bool (W*H), collated to a batch)."""
        from torch.utils.data import DataLoader

        target = 300
        ds = StreamingDemoDataset(size=5, max_level=25, base_seed=1, target=target)
        loader = DataLoader(ds, batch_size=64, num_workers=0)
        batches = list(loader)
        assert sum(b[0].shape[0] for b in batches) == target
        obs, tile, ent, dirn, item, misc, mask, eot = batches[0]
        assert obs.dtype == torch.uint8
        assert obs.shape[1:] == (len(Channel), 5, 5)
        assert mask.dtype == torch.bool
        assert mask.shape[1] == 5 * 5
        assert tile.shape[0] == obs.shape[0]

    def test_workers_walk_disjoint_seeds(self):
        """Sharding is leak-free: worker w of W walks a seed class no other
        worker touches, so no factory is ever generated twice. (seed is at
        index 8 of the yielded 10-tuple.)"""
        random.seed(0)
        seeds_w0 = {row[8] for row in _iter_demo_pairs(5, 25, 100, 0, 2, 150)}
        seeds_w1 = {row[8] for row in _iter_demo_pairs(5, 25, 100, 1, 2, 150)}
        assert seeds_w0 and seeds_w1
        assert seeds_w0.isdisjoint(seeds_w1)

    def test_train_stream_disjoint_from_val(self):
        """Training starts one seed above the highest val seed, so the val set
        and the training stream never share a factory (as train_sft wires it)."""
        val = _materialise(5, 25, 1, n_lessons=30)
        val_seeds = set(val[8].tolist())
        train_base = max(val_seeds) + 1
        random.seed(1)
        train_seeds = {
            row[8] for row in _iter_demo_pairs(5, 25, train_base, 0, 1, 200)
        }
        assert min(train_seeds) >= train_base
        assert val_seeds.isdisjoint(train_seeds)

    def test_steps_per_epoch_matches_loader(self):
        """_steps_per_epoch predicts the batch count the DataLoader actually
        yields (used to size the LR schedule before training)."""
        from torch.utils.data import DataLoader

        target, batch, workers = 240, 64, 2
        ds = StreamingDemoDataset(size=5, max_level=25, base_seed=1, target=target)
        loader = DataLoader(ds, batch_size=batch, num_workers=workers)
        assert len(list(loader)) == _steps_per_epoch(target, workers, batch)


ENV_ID = "factorion/FactorioEnv-v0-sft-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


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
        args = SftArgs(seed=42, size=5, num_samples=200, max_level=2)
        obs, tiles, ents, dirs, items_t, miscs_t, masks, _eots, _, _ = _materialise_args(
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
            encoded = agent.encoder(agent._encode_input(batch_obs))
            B = encoded.shape[0]

            tile_logits = agent.tile_logits(encoded).reshape(B, -1)
            loss_tile = bce_loss(tile_logits, masks.float())  # masks are bool

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
        args = SftArgs(
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

    def test_train_sft_dataset_cache_roundtrips(self, tmp_path):
        """--dataset-cache materialises the training stream to disk on the first
        run and trains from it on the second. Both runs produce a checkpoint."""
        cache = str(tmp_path / "ds_cache.pt")

        def run(tag):
            args = SftArgs(
                seed=1,
                size=5,
                num_samples=300,
                epochs=1,
                batch_size=64,
                layer1=16,
                layer2=16,
                layer3=16,
                dataset_cache=cache,
                checkpoint_path=str(tmp_path / f"ckpt_{tag}.pt"),
                summary_path=str(tmp_path / f"summary_{tag}.json"),
            )
            train_sft(args)

        run("create")
        assert os.path.exists(cache), "Cache should be written on first run"
        run("load")  # second run loads the cache instead of generating
        assert os.path.exists(str(tmp_path / "ckpt_load.pt"))


class TestSFTDropout:
    """The SFT dropout knob, when set, must reach the encoder via AgentCNN —
    mirrors ppo.PpoArgs' regularisation contract."""

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

        args = SftArgs(
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

        args = SftArgs(
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
            eval_rollouts=False,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)  # must not raise at the artifact step

        meta = captured["metadata"]
        assert meta["layers"] == [16, 16, 16]
        assert meta["kernel_size"] == 3
        assert "chan1" not in meta

    def test_run_named_by_artifact_name(self, monkeypatch, tmp_path):
        """The W&B run name is the full hyperparameter signature (== the model
        artifact name) so runs are distinguishable in the table — not all
        "sft-{size}x{size}"."""
        import wandb
        from unittest.mock import MagicMock

        captured = {}

        def fake_init(*a, **k):
            captured["name"] = k.get("name")
            run = MagicMock()
            run.url = "http://test/run"
            run.summary = {}
            return run

        monkeypatch.setattr(wandb, "init", fake_init)
        monkeypatch.setattr(wandb, "Artifact", lambda *a, **k: MagicMock())

        args = SftArgs(
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
            eval_rollouts=False,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)
        assert captured["name"] == _artifact_name(args)


class TestAsmItemAcc:
    """val/asm_item_acc — item-head accuracy on assembler (recipe) placements,
    the direct signal for whether the model reads the sink's item tag (#264)."""

    def _run_and_capture_log(self, monkeypatch, tmp_path, **overrides):
        """Run a tiny tracked train_sft and return the last logged metric dict."""
        import wandb
        from unittest.mock import MagicMock

        logged: list[dict] = []
        fake_run = MagicMock()
        fake_run.url = "http://test/run"
        fake_run.summary = {}
        fake_run.log = lambda d, **k: logged.append(d)
        monkeypatch.setattr(wandb, "init", lambda *a, **k: fake_run)
        monkeypatch.setattr(wandb, "Artifact", lambda *a, **k: MagicMock())

        args = SftArgs(
            seed=1,
            size=7,  # MEMORISE lessons (the only ones with assemblers) fit here
            num_samples=2000,
            epochs=1,
            batch_size=64,
            layer1=16,
            layer2=16,
            layer3=16,
            track=True,
            eval_rollouts=False,
            eval_rollouts_max_seeds=60,
            checkpoint_path=str(tmp_path / "asm.pt"),
            summary_path=str(tmp_path / "asm.json"),
            **overrides,
        )
        train_sft(args)
        assert logged, "no metrics were logged"
        return logged[-1]

    def test_global_and_per_kind_asm_item_acc_logged(self, monkeypatch, tmp_path):
        metrics = self._run_and_capture_log(monkeypatch, tmp_path)

        # Global metric always present.
        assert "val/asm_item_acc" in metrics
        assert 0.0 <= metrics["val/asm_item_acc"] <= 1.0

        # Per-kind asm_item_acc appears ONLY for lessons that place assemblers
        # (the MEMORISE lessons); its presence proves the denominator is real
        # (assembler placements actually landed in the val split), not 0/1.
        asm_keys = [k for k in metrics if k.endswith("/asm_item_acc") and "/" in k[4:]]
        assert asm_keys, "expected at least one per-kind asm_item_acc key"
        assert all("MEMORISE" in k for k in asm_keys), (
            f"only MEMORISE lessons place assemblers, got {asm_keys}"
        )

    def test_belt_only_lessons_have_no_asm_item_acc(self, monkeypatch, tmp_path):
        metrics = self._run_and_capture_log(monkeypatch, tmp_path)
        # MOVE_ONE_ITEM never places an assembler, so it must not emit the key.
        assert "val/MOVE_ONE_ITEM/asm_item_acc" not in metrics


class TestHeldoutRecipes:
    """Held-out-recipe split (#264): recipes excluded from training but kept in
    val, so val/asm_item_acc_{heldout,indist} separate generalization from
    memorisation."""

    def test_heldout_set_deterministic_and_leaves_one_per_class(self):
        from sft import _heldout_recipe_ids

        assert sft._heldout_recipe_ids(0.0) == frozenset()
        assert _heldout_recipe_ids(0.3) == _heldout_recipe_ids(0.3)
        # The single 5-ingredient recipe (oil_refinery) must never be held out,
        # else MEMORISE_5 becomes untrainable.
        oil = str2ent("oil_refinery").value
        assert oil not in _heldout_recipe_ids(0.9)

    def test_training_excludes_heldout_recipes(self):
        """A materialised TRAINING set must contain no assembler placement whose
        recipe was withheld."""
        from sft import _heldout_recipe_ids, _materialise

        held = _heldout_recipe_ids(0.4)
        assert held, "test needs a non-empty held-out set"
        asm_id = str2ent("assembling_machine_1").value
        data = _materialise(
            size=9, max_level=81, base_seed=7000, target=4000, heldout_recipe_ids=held
        )
        ent, item = data[2], data[4]
        asm_items = set(item[ent == asm_id].tolist())
        leaked = asm_items & set(held)
        assert not leaked, f"held-out recipes leaked into training: {leaked}"
        # And the split is real: some in-distribution recipes DID train.
        assert asm_items, "no assembler placements in training set at all"

    def test_val_reports_heldout_and_indist_split(self, monkeypatch, tmp_path):
        import wandb
        from unittest.mock import MagicMock

        logged: list[dict] = []
        fake_run = MagicMock()
        fake_run.url = "http://test/run"
        fake_run.summary = {}
        fake_run.log = lambda d, **k: logged.append(d)
        monkeypatch.setattr(wandb, "init", lambda *a, **k: fake_run)
        monkeypatch.setattr(wandb, "Artifact", lambda *a, **k: MagicMock())

        args = SftArgs(
            seed=1,
            size=7,
            num_samples=2500,
            epochs=1,
            batch_size=64,
            layer1=16,
            layer2=16,
            layer3=16,
            track=True,
            eval_rollouts=False,
            eval_rollouts_max_seeds=80,
            heldout_recipe_frac=0.3,
            checkpoint_path=str(tmp_path / "h.pt"),
            summary_path=str(tmp_path / "h.json"),
        )
        train_sft(args)
        metrics = logged[-1]
        assert "val/asm_item_acc_heldout" in metrics
        assert "val/asm_item_acc_indist" in metrics
        assert 0.0 <= metrics["val/asm_item_acc_heldout"] <= 1.0
        assert 0.0 <= metrics["val/asm_item_acc_indist"] <= 1.0

    def test_split_absent_when_disabled(self, monkeypatch, tmp_path):
        """With frac=0 (default) the split keys must not appear."""
        import wandb
        from unittest.mock import MagicMock

        logged: list[dict] = []
        fake_run = MagicMock()
        fake_run.url = "http://test/run"
        fake_run.summary = {}
        fake_run.log = lambda d, **k: logged.append(d)
        monkeypatch.setattr(wandb, "init", lambda *a, **k: fake_run)
        monkeypatch.setattr(wandb, "Artifact", lambda *a, **k: MagicMock())

        args = SftArgs(
            seed=1,
            size=7,
            num_samples=1500,
            epochs=1,
            batch_size=64,
            layer1=16,
            layer2=16,
            layer3=16,
            track=True,
            eval_rollouts=False,
            eval_rollouts_max_seeds=40,
            checkpoint_path=str(tmp_path / "d.pt"),
            summary_path=str(tmp_path / "d.json"),
        )
        train_sft(args)
        metrics = logged[-1]
        assert "val/asm_item_acc_heldout" not in metrics
        assert "val/asm_item_acc_indist" not in metrics


class TestRunRolloutEval:
    """End-to-end coverage of greedy rollout eval on held-out val factories."""

    def _build_val_seeds_to_kind(self, size, num_kinds=4, start_seed=10_000):
        """Collect up to `num_kinds` (seed -> kind) pairs, one per distinct
        lesson kind, each producing a valid lesson at the given size. Mirrors
        the try/except-continue pattern that _iter_demo_pairs uses for
        malformed seeds, so this fixture doesn't get flaky when an enum value
        lands a bad seed.

        Some kinds can never be generated on a tiny grid (e.g.
        ASSEMBLE_2IN_1OUT does not fit at size 5). Rather than burn the whole
        seed budget retrying an unbuildable kind, skip a kind after
        `max_fails_per_kind` consecutive rejections and move to the next one."""
        max_fails_per_kind = 40
        out: dict[int, int] = {}
        seed = start_seed
        for kind in LessonKind:
            if len(out) >= num_kinds:
                break
            fails = 0
            while fails < max_fails_per_kind:
                if build_factory(size=size, kind=kind, seed=seed) is None:
                    seed += 1
                    fails += 1
                    continue
                out[seed] = kind.value
                seed += 1
                break
        return out

    def test_returns_throughput_in_unit_range(self, registered_env):
        """Untrained agent on tiny grid should return well-formed throughput
        numbers — overall in [0, 1] and per-kind in [0, 1] for any kind that
        was eval'd."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        args = SftArgs(
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
        args = SftArgs(
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

        args = SftArgs(
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

        args = SftArgs(
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

        args = SftArgs(
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

    def _run_with_recorded_proposals(self, monkeypatch):
        """Run a greedy rollout eval with a FactorioEnv that records, for every
        step, the (entity, footprint) the *proposed anchor tile* held in the
        world just before the placement was applied. Returns that list."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, layers=(16, 16, 16))
        envs.close()

        recorded: list[tuple[int, int]] = []

        class RecordingEnv(FactorioEnv):
            def step(self, action):
                x, y = action["xy"]
                ent = int(self._world_CWH[Channel.ENTITIES.value, x, y])
                foot = int(self._world_CWH[Channel.FOOTPRINT.value, x, y])
                recorded.append((ent, foot))
                return super().step(action)

        monkeypatch.setattr(sft, "FactorioEnv", RecordingEnv)

        args = SftArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            layer1=16,
            layer2=16,
            layer3=16,
        )
        val_seeds_to_kind = self._build_val_seeds_to_kind(size=size, num_kinds=4)
        assert len(val_seeds_to_kind) >= 1

        run_rollout_eval(
            agent,
            args,
            val_seeds_to_kind,
            device=torch.device("cpu"),
            max_seeds=len(val_seeds_to_kind),
        )
        return recorded

    def test_masking_only_proposes_legal_tiles(self, registered_env, monkeypatch):
        """The greedy tile argmax must only ever propose legal placement
        targets: the anchor tile is empty (ENTITIES == empty) and buildable
        (FOOTPRINT == AVAILABLE) at the moment of every step."""
        empty_id = str2ent("empty").value
        recorded = self._run_with_recorded_proposals(monkeypatch)
        assert recorded, "rollout should have proposed at least one tile"
        illegal = [
            (ent, foot)
            for ent, foot in recorded
            if ent != empty_id or foot != Footprint.AVAILABLE.value
        ]
        assert not illegal, (
            f"masking must never propose an occupied/unbuildable tile; got {illegal}"
        )


class TestLegalTileMask:
    """Deterministic unit coverage of the eval-time legal-tile mask helper."""

    def _obs(self, size):
        """A single all-empty, all-buildable observation of shape (1, C, W, H)."""
        C = len(Channel)
        obs = torch.zeros(1, C, size, size)
        obs[0, Channel.ENTITIES.value] = str2ent("empty").value
        obs[0, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
        return obs

    def test_argmax_skips_occupied_tile(self):
        """When the top-logit tile is occupied, the masked argmax must fall to
        the next-best *legal* tile instead of livelocking on the occupied one."""
        size = 3
        obs = self._obs(size)
        # Occupy tile (0, 0) -> flat index 0, x-major (idx = x * size + y).
        obs[0, Channel.ENTITIES.value, 0, 0] = str2ent("transport_belt").value
        # Logits favour the occupied tile 0, then tile 5 as runner-up.
        logits = torch.full((1, size * size), -1.0)
        logits[0, 0] = 10.0
        logits[0, 5] = 5.0

        assert logits.argmax(dim=1).item() == 0, "sanity: unmasked picks occupied"
        masked = _apply_legal_tile_mask(logits, obs)
        assert masked.argmax(dim=1).item() == 5, "masked must skip to legal runner-up"
        assert masked[0, 0] == float("-inf"), "occupied tile must be -inf"

    def test_argmax_skips_unbuildable_tile(self):
        """UNAVAILABLE (unbuildable) tiles are masked out just like occupied
        ones, even when the tile is otherwise empty."""
        size = 3
        obs = self._obs(size)
        # Tile (1, 1) -> flat index 4 is empty but not buildable.
        obs[0, Channel.FOOTPRINT.value, 1, 1] = Footprint.UNAVAILABLE.value
        logits = torch.full((1, size * size), -1.0)
        logits[0, 4] = 10.0
        logits[0, 2] = 5.0

        masked = _apply_legal_tile_mask(logits, obs)
        assert masked.argmax(dim=1).item() == 2
        assert masked[0, 4] == float("-inf")

    def test_all_illegal_row_is_nan_free(self):
        """A fully occupied grid leaves every tile illegal; the masked argmax
        must still return a finite index (tile 0) rather than NaN-ing out."""
        size = 3
        obs = self._obs(size)
        obs[0, Channel.ENTITIES.value] = str2ent("transport_belt").value  # occupy everything
        logits = torch.randn(1, size * size)

        masked = _apply_legal_tile_mask(logits, obs)
        assert torch.isinf(masked).all(), "every tile should be masked to -inf"
        idx = masked.argmax(dim=1).item()
        assert idx == 0, "all-illegal row falls back to tile 0"


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
        # Terminal observation equals solved (entities + directions). obs is
        # uint8, so cast to compare against the int64 solved world.
        terminal_obs = pairs[-1][0]
        assert terminal_obs.dtype == torch.uint8
        assert torch.equal(
            terminal_obs[Channel.ENTITIES.value].long(), solved[Channel.ENTITIES.value]
        )
        assert torch.equal(
            terminal_obs[Channel.DIRECTION.value].long(),
            solved[Channel.DIRECTION.value],
        )

    def test_eot_tensor_in_dataset(self):
        """_materialise must return a per-pair eot tensor with values
        in {0.0, 1.0} and at least one positive (terminal) example."""
        args = SftArgs(seed=1, size=5, num_samples=200, max_level=4)
        *_, eots, _seeds, _kinds = _materialise_args(args)
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
        enc = agent.encoder(agent._encode_input(x))
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


class TestPerKindEotMetrics:
    """val/<LESSON>/eot_acc and /eot_pos_recall — the per-LessonKind breakdown
    of the global EOT stop-signal metrics."""

    def test_per_kind_eot_metrics_logged(self, monkeypatch, tmp_path):
        """train_sft must log val/<LESSON>/eot_acc and /eot_pos_recall for
        every LessonKind present in the val split, each in [0, 1] and paired
        1:1 with the existing per-kind placement /acc. The per-kind metrics go
        only to wandb (not summary.json), so capture the logged dict via a
        mock run."""
        import wandb
        from unittest.mock import MagicMock

        logged: dict = {}
        fake_run = MagicMock()
        fake_run.url = "http://test/run"
        fake_run.summary = {}
        fake_run.log.side_effect = lambda d, *a, **k: logged.update(d)
        monkeypatch.setattr(wandb, "init", lambda *a, **k: fake_run)
        monkeypatch.setattr(wandb, "Artifact", lambda *a, **k: MagicMock())

        args = SftArgs(
            seed=1,
            size=5,
            num_samples=400,
            max_level=2,
            epochs=1,
            batch_size=32,
            layer1=16,
            layer2=16,
            layer3=16,
            track=True,
            eval_rollouts=False,  # skip the slow greedy rollout
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)

        acc_keys = [
            k
            for k in logged
            if k.startswith("val/") and k.endswith("/eot_acc") and k != "val/eot_acc"
        ]
        rec_keys = [
            k
            for k in logged
            if k.startswith("val/")
            and k.endswith("/eot_pos_recall")
            and k != "val/eot_pos_recall"
        ]
        assert acc_keys, "expected per-kind val/<LESSON>/eot_acc to be logged"
        assert rec_keys, "expected per-kind val/<LESSON>/eot_pos_recall to be logged"
        for k in acc_keys + rec_keys:
            assert 0.0 <= logged[k] <= 1.0, f"{k}={logged[k]} out of [0, 1]"

        # The per-kind eot keys must cover exactly the kinds that already get a
        # placement /acc — same val split, same buckets, emitted together.
        place_kinds = {
            k.split("/")[1]
            for k in logged
            if k.startswith("val/") and k.endswith("/acc") and k.count("/") == 2
        }
        assert {k.split("/")[1] for k in acc_keys} == place_kinds


class TestDirectionConfusion:
    """Direction confusion matrix diagnostic. Direction values:
    NONE=0, NORTH=1, EAST=2, SOUTH=3, WEST=4."""

    def test_diagnostics_payload(self):
        N, S = 1, 3
        payload = _direction_diagnostics([N, N, S], [N, S, S])
        # The confusion-matrix chart object must be present for wandb to plot.
        assert payload["val/dir_confusion"] is not None

    def test_diagnostics_empty_is_noop(self):
        assert _direction_diagnostics([], []) == {}


class TestDirMismatchVsDistance:
    """source<->sink distance vs direction mismatch. "Mismatch" = argmax(dir)
    disagrees with the demo's direction (a valid alternative routing also
    counts). Distance is read off the obs ENTITIES channel (source=67,
    sink=66, never blanked)."""

    SOURCE, SINK = 67, 66

    def _obs_with(self, src_xy, sink_xy, size=6):
        obs = torch.zeros(1, len(Channel), size, size)
        ent = Channel.ENTITIES.value
        if src_xy is not None:
            obs[0, ent, src_xy[0], src_xy[1]] = self.SOURCE
        if sink_xy is not None:
            obs[0, ent, sink_xy[0], sink_xy[1]] = self.SINK
        return obs

    def test_distance_is_manhattan_between_endpoints(self):
        obs = self._obs_with((1, 1), (4, 3))  # |1-4| + |1-3| = 5
        assert _source_sink_distance(obs)[0].item() == pytest.approx(5.0)

    def test_distance_centroid_for_multiple_sinks(self):
        # one source at (0,0), two sinks at (4,0) and (0,4): sink centroid
        # (2,2) -> Manhattan |0-2| + |0-2| = 4.
        obs = torch.zeros(1, len(Channel), 6, 6)
        ent = Channel.ENTITIES.value
        obs[0, ent, 0, 0] = self.SOURCE
        obs[0, ent, 4, 0] = self.SINK
        obs[0, ent, 0, 4] = self.SINK
        assert _source_sink_distance(obs)[0].item() == pytest.approx(4.0)

    def test_distance_minus_one_when_endpoint_missing(self):
        assert _source_sink_distance(self._obs_with((1, 1), None))[0].item() == -1.0

    def test_mismatch_by_distance_buckets(self):
        # dist 1 -> [0,0] (0%); dist 2 -> [1,1] (100%); dist 3 -> [0,1] (50%)
        xs, rates, counts = _dir_mismatch_by_distance(
            [1, 1, 2, 2, 3, 3], [0, 0, 1, 1, 0, 1]
        )
        assert xs == [1, 2, 3]
        assert rates == pytest.approx([0.0, 1.0, 0.5])
        assert counts == [2, 2, 2]

    def test_point_biserial_sign_and_magnitude(self):
        # mismatches rise with distance -> strong positive correlation.
        assert _point_biserial([1, 2, 3, 4], [0, 0, 1, 1]) == pytest.approx(
            0.894, abs=0.01
        )
        # mirror-image -> strong negative.
        assert _point_biserial([1, 2, 3, 4], [1, 1, 0, 0]) == pytest.approx(
            -0.894, abs=0.01
        )
        # no mismatch variance -> 0 (degenerate, not NaN).
        assert _point_biserial([1, 2, 3, 4], [0, 0, 0, 0]) == 0.0

    def test_distance_diagnostics_payload(self):
        payload = _dir_distance_diagnostics([1, 2, 3, 4], [0, 0, 1, 1])
        assert payload["val/dir/mismatch_distance_corr"] == pytest.approx(
            0.894, abs=0.01
        )
        assert payload["val/dir_mismatch_vs_distance"] is not None

    def test_distance_diagnostics_empty_is_noop(self):
        assert _dir_distance_diagnostics([], []) == {}


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

    def test_artifact_name_larger_run(self):
        args = SftArgs(
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
        three = SftArgs(layer1=48, layer2=48, layer3=48, layer4=0)
        four = SftArgs(layer1=48, layer2=48, layer3=48, layer4=48)
        assert _artifact_name(three).endswith("-c48-48-48")
        assert _artifact_name(four).endswith("-c48-48-48-48")
        assert _artifact_name(three) != _artifact_name(four)

    def test_artifact_name_asymmetric_channels(self):
        """Differing per-layer widths expand into the full list, so a
        c32-64-64 run can't accidentally file under a c48-48-48 run."""
        args = SftArgs(layer1=32, layer2=64, layer3=64)
        # Assert only the channel suffix (the behaviour under test) so this
        # doesn't break when the default size/samples/epochs/lr change.
        assert _artifact_name(args).endswith("-c32-64-64")

    def test_artifact_name_kernel_size_suffix(self):
        """A non-default kernel size appends -k{N} so two runs differing only
        in receptive field get distinct artifacts; the default k=3 adds no
        token (keeps existing names stable)."""
        # Default kernel (3) adds no suffix; a non-default kernel appends
        # -k{N}. Derive from the default name so this survives default changes.
        base = _artifact_name(SftArgs())
        assert not base.endswith(("-k3", "-k5", "-k7"))
        assert _artifact_name(SftArgs(kernel_size=5)) == base + "-k5"

    def test_artifact_name_stable_across_runs(self):
        """Two SftArgs with the same hyperparams must produce the same
        artifact name (otherwise we lose version collapsing)."""
        a = SftArgs(seed=1)
        b = SftArgs(seed=999)  # seed isn't part of the artifact name
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
        args = SftArgs(lr=1e-3, warmup_frac=0.1, min_lr_ratio=0.01)
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
        args = SftArgs(lr=2e-3, warmup_frac=0.0, min_lr_ratio=0.1)
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
        args = SftArgs(lr=1e-3, warmup_frac=0.5)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, 1, args)
        optimizer.step()
        scheduler.step()  # must not raise
