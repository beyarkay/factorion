"""Tests for SFT pre-training pipeline."""

import json
import os
import re
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import gymnasium as gym
import numpy as np

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import (
    TINY_ARCH,
    TINY_ARCH_ARGS,
    Channel,
    LessonKind,
    blank_entities,
    build_factory,
    str2ent,
)
from factorion import LESSON_IS_TRIAL, Footprint
from denoise import DenoiserUNet, load_denoiser, score_world, world_to_state
import sft
from sft import (
    SftArgs,
    _artifact_name,
    _humanize_count,
    _humanize_lr,
    _solved_assembler_recipes,
    build_lr_schedule,
    noise_levels,
    run_rollout_eval,
    select_self_targets,
    self_seeds,
    throughput_pg_loss,
    train_sft,
)
from ppo import (
    FactorioEnv, AgentCNN, make_env, _build_eval_set, _legal_tile_mask,
)
from training_config import PpoArgs

TINY: dict[str, Any] = dict(width=16, levels=2, blocks=1, attn_layers=1, attn_heads=4)
# Generation eval shrunk to a couple of short chains per factory.
FAST_EVAL: dict[str, Any] = dict(eval_seeds_per_kind=1, eval_best_of=2, gen_steps=2, gen_refine=2)


def _batch(n=4, size=15):
    factories = [f for s in range(n) if (f := build_factory(size=size, kind=LessonKind.SAME_SIDE_1IN, seed=s))]
    assert len(factories) == n
    state = torch.stack([torch.from_numpy(world_to_state(f.world_CWH)) for f in factories]).long()
    return state, torch.tensor([f.max_throughput for f in factories])


class TestNoiseLevels:
    def test_point_masses_and_range(self):
        args = SftArgs(noise_zero_frac=0.2, noise_one_frac=0.3)
        t = noise_levels(20_000, args, torch.device("cpu"))
        assert ((t >= 0) & (t <= 1)).all()
        assert (t == 0).float().mean() == pytest.approx(0.2, abs=0.02)
        assert (t == 1).float().mean() == pytest.approx(0.3, abs=0.02)


class TestThroughputLoss:
    @pytest.mark.parametrize("io", ["token", "factored"])
    def test_reinforces_sampled_factories(self, io):
        from denoise import free_mask

        state, max_thput = _batch()
        model = DenoiserUNet(**TINY, io=io)
        logits = model(state)
        args = SftArgs(rl_inputs=2, rl_samples=3, rl_max_noise=0.5)
        loss, score = throughput_pg_loss(model, logits, state, free_mask(state), max_thput, torch.zeros(4), args)
        loss.backward()
        assert torch.isfinite(loss) and score >= 0
        assert model.head[-1].weight.grad is not None

    def test_skips_heavily_corrupted_inputs(self):
        from denoise import free_mask

        state, max_thput = _batch(2)
        model = DenoiserUNet(**TINY)
        logits = model(state)
        loss, score = throughput_pg_loss(model, logits, state, free_mask(state), max_thput, torch.ones(2), SftArgs())
        assert float(loss) == 0.0 and score == 0.0


ENV_ID = "factorion/FactorioEnv-v0-sft-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


class TestSFTCheckpointLoading:
    def test_checkpoint_roundtrip(self, registered_env, tmp_path):
        """SFT checkpoint should load into AgentCNN and produce valid outputs."""
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)

        # Save checkpoint
        ckpt_path = str(tmp_path / "test_sft.pt")
        torch.save(agent.state_dict(), ckpt_path)

        # Load into fresh agent
        agent2 = AgentCNN(envs, **TINY_ARCH)
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
        agent = AgentCNN(envs, **TINY_ARCH)

        ckpt_path = str(tmp_path / "test_sft.pt")
        torch.save(agent.state_dict(), ckpt_path)

        # Simulate what ppo.py does with --start_from
        agent2 = AgentCNN(envs, **TINY_ARCH)
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


class TestTrainSFTEndToEnd:
    @pytest.mark.parametrize("io", ["token", "factored"])
    def test_trains_evaluates_and_checkpoints(self, tmp_path, io):
        ckpt, summary = tmp_path / "ckpt.pt", tmp_path / "summary.json"
        args = SftArgs(
            size=15, num_samples=64, batch_size=16, eval_seeds_per_kind=1,
            eval_best_of=2, gen_steps=2, gen_refine=2, lw_thput=0.1, rl_inputs=2, rl_samples=2, rl_max_noise=1.0,
            checkpoint_path=str(ckpt), summary_path=str(summary), io=io, **TINY,
        )
        model = train_sft(args)
        out = json.loads(summary.read_text())
        assert out["samples_seen"] == 64 and 0.0 <= out["best_val_throughput"] <= 1.0
        loaded = load_denoiser(ckpt)
        assert loaded.arch == model.arch
        for (k, a), b in zip(loaded.state_dict().items(), model.state_dict().values()):
            assert torch.equal(a, b.cpu()), k

        # --start-from resumes those weights (and the noise prior with them).
        args.start_from, args.num_samples = str(ckpt), 16
        args.checkpoint_path = str(tmp_path / "ckpt2.pt")
        assert torch.equal(train_sft(args).noise_prior.cpu(), loaded.noise_prior)


def _fake_wandb(monkeypatch, logged):
    import wandb
    from unittest.mock import MagicMock

    run = MagicMock()
    run.url = "http://test/run"
    run.summary = {}
    run.log.side_effect = lambda d, *a, **k: logged.append(d)
    monkeypatch.setattr(wandb, "init", lambda *a, **k: run)
    monkeypatch.setattr(wandb, "Artifact", lambda *a, **k: MagicMock())


class TestSelfTraining:
    def _build(self, factory, world, chain=0):
        return {**score_world(world, factory.max_throughput, PpoArgs.entity_cost_scale), "world": world, "chain": chain}

    def test_self_seeds_are_disjoint_from_the_eval_set(self):
        for seed in (0, 1, 7):
            args = SftArgs(size=15, seed=seed, eval_seeds_per_kind=2)
            eval_seeds = _build_eval_set(SimpleNamespace(size=15, seed=seed, eval_seeds_per_kind=2))
            # The seeds count up from the first, so every one clears the eval set.
            assert next(self_seeds(args)) > max(eval_seeds)

    def test_keeps_each_kinds_top_flowing_builds(self):
        f = build_factory(size=15, kind=LessonKind.SAME_SIDE_1IN, seed=0)
        assert f is not None
        blank = blank_entities(f)[0].numpy()
        builds = [self._build(f, f.world_CWH.numpy(), 0)] + [self._build(f, blank, j) for j in range(1, 10)]
        # The reference is the top 10%; the rest don't flow even when ranked in.
        for frac in (0.1, 1.0):
            kept = select_self_targets([(0, LessonKind.SAME_SIDE_1IN, f)], [builds], frac)
            assert [(k, r["chain"]) for k, r, _ in kept] == [(LessonKind.SAME_SIDE_1IN, 0)]

        # A lesson build below the top 10% is dropped, while a trial's flowing
        # build is ranked among trials alone and kept.
        worse = self._build(f, blank, 1)
        trial = LessonKind.TRIAL_RECIPE_TREE_DEPTH_1
        kept = select_self_targets(
            [(0, LessonKind.SAME_SIDE_1IN, f), (1, trial, f)],
            [[self._build(f, f.world_CWH.numpy())] + [worse] * 9, [self._build(f, f.world_CWH.numpy(), 3)]],
            0.1,
        )
        assert [(k, r["chain"]) for k, r, _ in kept] == [(LessonKind.SAME_SIDE_1IN, 0), (trial, 3)]

    def test_rejects_dangling_inserters(self):
        f = build_factory(size=15, kind=LessonKind.SAME_SIDE_1IN, seed=0)
        assert f is not None
        world = f.world_CWH.numpy().copy()
        ent = world[Channel.ENTITIES.value]
        empty = np.argwhere(np.lib.stride_tricks.sliding_window_view(np.pad(ent, 1, constant_values=1), (3, 3)).max((2, 3)) == 0)
        x, y = empty[0]
        world[Channel.ENTITIES.value, x, y] = str2ent("inserter").value
        world[Channel.DIRECTION.value, x, y] = 1
        assert self._build(f, world)["thput_raw"] > 0
        assert select_self_targets([(0, LessonKind.SAME_SIDE_1IN, f)], [[self._build(f, world)]], 1.0) == []

    def test_train_sft_fills_the_buffer_and_trains_from_it(self, monkeypatch, tmp_path):
        """Self-training builds trials only. With each one's first build
        swapped for a flowing factory (an untrained model builds nothing that
        flows), it fills the buffer and is trained on."""
        real_chains, real_step = sft.generate_chains, sft.denoise_step
        donor = build_factory(size=15, kind=LessonKind.SAME_SIDE_1IN, seed=0)
        assert donor is not None
        built, trained = [], []

        def fake_chains(model, factories, *a, **k):
            results, n_conv, n_ref = real_chains(model, factories, *a, **k)
            for (_, kind, f), rs in zip(factories, results):
                built.append(kind)
                rs[0] = self._build(f, donor.world_CWH.numpy())
            return results, n_conv, n_ref

        def spy_step(model, state, *a, **k):
            trained.extend(s.to(torch.uint8).numpy().tobytes() for s in state.cpu())
            return real_step(model, state, *a, **k)

        logged: list = []
        _fake_wandb(monkeypatch, logged)
        monkeypatch.setattr(sft, "generate_chains", fake_chains)
        monkeypatch.setattr(sft, "denoise_step", spy_step)
        args = SftArgs(
            size=15, num_samples=96, batch_size=16, eval_every_n_samples=32, self_frac=0.25,
            self_factories=6, self_top_frac=0.5, track=True,
            checkpoint_path=str(tmp_path / "k.pt"), summary_path=str(tmp_path / "k.json"), **TINY, **FAST_EVAL,
        )
        train_sft(args)

        assert built and all(LESSON_IS_TRIAL[k] for k in built)
        self_logs = [m for m in logged if "self/buffer_size" in m]
        assert len(self_logs) == 2 and self_logs[0]["self/buffer_size"] == 1
        assert self_logs[0]["self/TRIAL_RECIPE_TREE_DEPTH_1/accept_frac"] > 0
        assert logged[-1]["self/batch_frac"] == pytest.approx(0.25)
        assert world_to_state(donor.world_CWH).tobytes() in trained

    def test_off_never_generates(self, monkeypatch, tmp_path):
        logged: list = []
        _fake_wandb(monkeypatch, logged)
        monkeypatch.setattr(sft, "self_round", lambda *a, **k: pytest.fail("self-training is off"))
        args = SftArgs(
            size=15, num_samples=64, batch_size=16, eval_every_n_samples=32, self_frac=0.0, track=True,
            checkpoint_path=str(tmp_path / "k.pt"), summary_path=str(tmp_path / "k.json"), **TINY, **FAST_EVAL,
        )
        train_sft(args)
        assert not any(k.startswith("self/") for m in logged for k in m)


class TestTrackedArtifact:
    """The track=True checkpoint-upload path. No test exercised it, so #146's
    chan1/2/3 -> layers rename left a stale args.chan1 in the artifact metadata
    that crashed every tracked run with AttributeError *after* training — only
    at upload. This pins the path end-to-end."""

    def test_tracked_run_finishes_with_arch_metadata(self, monkeypatch, tmp_path):
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
            batch_size=32,
            **TINY,
            **FAST_EVAL,
            track=True,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)  # must not raise at the artifact step

        meta = captured["metadata"]
        assert meta["width"] == 16
        assert meta["attn_layers"] == 1

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
            batch_size=32,
            **TINY,
            **FAST_EVAL,
            track=True,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)
        assert captured["name"] == _artifact_name(args)


    def test_each_new_best_replaces_the_runs_previous_upload(self, monkeypatch, tmp_path):
        """A run uploads each strictly better val/thput checkpoint as it trains,
        so one killed mid-run still leaves its best in W&B, and deletes its own
        previous upload each time, so it holds one version at the end."""
        import itertools
        import wandb
        from unittest.mock import MagicMock

        import sft

        thputs = itertools.chain([0.1, 0.1, 0.3], itertools.repeat(0.2))
        real_generation = sft.generation_metrics

        def fake_generation(model, factories, args, *a, **k):
            metrics = real_generation(model, factories, args, *a, **k)
            metrics[f"val/thput@{args.eval_best_of}"] = next(thputs)
            return metrics

        artifacts, uploads = [], []

        def fake_artifact(*a, **k):
            artifacts.append(MagicMock())
            return artifacts[-1]

        def fake_log_artifact(artifact, aliases):
            uploads.append(MagicMock())
            return uploads[-1]

        fake_run = MagicMock()
        fake_run.url = "http://test/run"
        fake_run.summary = {}
        fake_run.log_artifact.side_effect = fake_log_artifact
        monkeypatch.setattr(wandb, "init", lambda *a, **k: fake_run)
        monkeypatch.setattr(wandb, "Artifact", fake_artifact)
        monkeypatch.setattr(sft, "generation_metrics", fake_generation)

        args = SftArgs(
            seed=1,
            size=5,
            num_samples=400,
            max_level=2,
            batch_size=32,
            **TINY,
            **FAST_EVAL,
            track=True,
            eval_every_n_samples=64,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)

        # 0.1 and 0.3 upload mid-run (the 0.1 tie and the later 0.2s don't),
        # then the final checkpoint + summary.
        assert len(uploads) == 3
        assert all(u.delete.call_count == 1 for u in uploads[:-1])
        uploads[-1].delete.assert_not_called()
        final_files = [c.args[0] for c in artifacts[-1].add_file.call_args_list]
        assert final_files == [args.checkpoint_path, args.summary_path]

    def test_generation_metrics_are_logged_as_best_of_n(self, monkeypatch, tmp_path):
        """Every metric from the best-of-N generation eval carries @N, and no
        plain key (which would read as best-of-1) duplicates it."""
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
            batch_size=32,
            **TINY,
            **FAST_EVAL,
            track=True,
            checkpoint_path=str(tmp_path / "k.pt"),
            summary_path=str(tmp_path / "k.json"),
        )
        train_sft(args)

        n = args.eval_best_of
        assert f"val/thput@{n}" in logged
        assert f"val/inserters_no_input@{n}" in logged
        assert any(k.endswith(f"/MOVE_ONE_ITEM/thput@{n}") for k in logged)
        assert all(re.search(r"@\d+$", k) for k in logged if "thput" in k or "inserters_no" in k)
        assert f"best_val_throughput@{n}" in fake_run.summary


class TestSolvedAssemblerRecipes:
    """_solved_assembler_recipes — the ground truth the rollout scores against."""

    def test_memorise_factory_yields_its_recipe(self):
        f = build_factory(
            size=9, kind=LessonKind.MEMORISE_1_INGREDIENT_RECIPES, seed=3
        )
        assert f is not None
        asm_id = str2ent("assembling_machine_1").value
        recipes = _solved_assembler_recipes(f.world_CWH)
        # Exactly one assembler → exactly one recipe, and it matches the ITEMS
        # tag on the assembler tile.
        assert len(recipes) == 1
        ent = f.world_CWH[Channel.ENTITIES.value]
        item = f.world_CWH[Channel.ITEMS.value]
        asm_items = {int(v) for v in item[ent == asm_id].tolist()}
        assert recipes == asm_items

    def test_belt_factory_has_no_recipe(self):
        f = build_factory(size=9, kind=LessonKind.MOVE_ONE_ITEM, seed=3)
        assert f is not None
        assert _solved_assembler_recipes(f.world_CWH) == set()


class TestRolloutAsmItemAcc:
    """Recipe-pick accuracy inlined into run_rollout_eval: of the assemblers the
    agent actually places, how many got the right recipe (#264)."""

    def _forced_agent(self, size, item_id):
        """An agent whose entity head always places an assembler (with NONE
        dir/misc), whose item head always picks `item_id`, and whose tile head
        prefers central tiles. Zeroing the head weights makes the sculpted bias
        the sole determinant of every argmax; the central tile bias makes the
        first placement land where the blanked assembler used to sit (empty, so
        the 3x3 footprint always fits) — otherwise a random tile argmax can loop
        forever on a corner where the footprint runs off-grid and place nothing.
        """
        asm_id = str2ent("assembling_machine_1").value
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()
        with torch.no_grad():
            for head, idx in (
                (agent.ent_head, asm_id),
                (agent.item_head, item_id),
                (agent.dir_head, 0),   # Direction.NONE
                (agent.misc_head, 0),  # Misc.NONE
            ):
                head.weight.zero_()
                head.bias.fill_(-100.0)
                head.bias[idx] = 100.0

        # Center-weighted tile logits (highest at the grid center), so the legal
        # argmax lands on the most central buildable tile. Wrapped in a Module
        # because nn.Module forbids replacing a submodule with a bare function.
        yy, xx = torch.meshgrid(
            torch.arange(size), torch.arange(size), indexing="ij"
        )
        center = size // 2
        tile_map = (-((xx - center) ** 2 + (yy - center) ** 2)).float().reshape(
            1, 1, size, size
        )

        class _FixedTile(torch.nn.Module):
            def forward(self, encoded):
                return tile_map.expand(encoded.shape[0], 1, size, size)

        agent.tile_logits = _FixedTile()
        agent.eval()
        return agent

    def _seeds_with_recipe(self, size, recipe_id, n, start=600_000):
        """`n` MEMORISE_1 seeds whose (single) assembler recipe is `recipe_id`."""
        out = {}
        s = start
        while len(out) < n and s < start + 20_000:
            f = build_factory(
                size=size, kind=LessonKind.MEMORISE_1_INGREDIENT_RECIPES, seed=s
            )
            if f is not None and _solved_assembler_recipes(f.world_CWH) == {recipe_id}:
                out[s] = LessonKind.MEMORISE_1_INGREDIENT_RECIPES.value
            s += 1
        return out

    def _two_recipes(self, size):
        """Two distinct MEMORISE_1 recipe ids that both occur at this size."""
        seen = []
        s = 600_000
        while len(seen) < 2 and s < 620_000:
            f = build_factory(
                size=size, kind=LessonKind.MEMORISE_1_INGREDIENT_RECIPES, seed=s
            )
            if f is not None:
                r = _solved_assembler_recipes(f.world_CWH)
                if r and next(iter(r)) not in seen:
                    seen.append(next(iter(r)))
            s += 1
        assert len(seen) == 2
        return seen[0], seen[1]

    def test_return_has_asm_keys(self, registered_env):
        """Shape contract used by both SFT val/ and PPO eval/ logging."""
        size = 9
        agent = self._forced_agent(size, item_id=6)  # copper_cable-ish id
        seeds = {s: LessonKind.MEMORISE_1_INGREDIENT_RECIPES.value
                 for s in list(self._seeds_with_recipe(size, self._two_recipes(size)[0], 2))}
        roll = run_rollout_eval(
            agent, SftArgs(seed=1, size=size), seeds, torch.device("cpu"), max_seeds=2
        )
        assert {"asm_item_acc", "per_kind_asm_item_acc", "per_kind_asm_n"} <= set(roll)
        assert 0.0 <= roll["asm_item_acc"] <= 1.0

    def test_correct_recipe_scores_one_wrong_scores_zero(self, registered_env):
        size = 9
        recipe, other = self._two_recipes(size)
        seeds = self._seeds_with_recipe(size, recipe, n=8)
        assert len(seeds) >= 4, "need several matching-recipe factories"
        args = SftArgs(seed=1, size=size)

        # Agent always places an assembler with the CORRECT recipe.
        good = run_rollout_eval(
            self._forced_agent(size, item_id=recipe), args, seeds,
            torch.device("cpu"), max_seeds=len(seeds),
        )
        placed = good["per_kind_asm_n"]["MEMORISE_1_INGREDIENT_RECIPES"]
        assert placed > 0, "forced agent should land at least one assembler"
        assert good["asm_item_acc"] == 1.0
        assert good["per_kind_asm_item_acc"]["MEMORISE_1_INGREDIENT_RECIPES"] == 1.0

        # Same factories, but the agent always picks a DIFFERENT recipe.
        bad = run_rollout_eval(
            self._forced_agent(size, item_id=other), args, seeds,
            torch.device("cpu"), max_seeds=len(seeds),
        )
        assert bad["per_kind_asm_n"]["MEMORISE_1_INGREDIENT_RECIPES"] > 0
        assert bad["asm_item_acc"] == 0.0

    def test_belt_only_lesson_never_scored(self, registered_env):
        """A factory with no assembler contributes no assembler placements, even
        when the agent tries to place assemblers everywhere."""
        size = 9
        seed = next(
            s for s in range(700_000, 700_500)
            if build_factory(size=size, kind=LessonKind.MOVE_ONE_ITEM, seed=s)
            is not None
        )
        roll = run_rollout_eval(
            self._forced_agent(size, item_id=6),
            SftArgs(seed=1, size=size),
            {seed: LessonKind.MOVE_ONE_ITEM.value},
            torch.device("cpu"),
            max_seeds=1,
        )
        assert roll["per_kind_asm_n"]["MOVE_ONE_ITEM"] == 0


class TestRunRolloutEval:
    """End-to-end coverage of the best-of-N rollout eval on held-out val factories."""

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
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()

        args = SftArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            **TINY_ARCH_ARGS,
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
            "trial_overall",
            "trial_n",
            "per_kind",
            "per_kind_n",
            "asm_item_acc",
            "per_kind_asm_item_acc",
            "per_kind_asm_n",
            "eot_acc",
            "eot_pos_recall",
            "per_kind_eot_acc",
            "per_kind_eot_pos_recall",
            "per_kind_eot_step_n",
            "per_kind_eot_pos_n",
            "dangling_inserters",
            "trial_dangling_inserters",
            "per_kind_dangling_inserters",
        }
        overall, per_kind, per_kind_n = (
            roll["overall"],
            roll["per_kind"],
            roll["per_kind_n"],
        )

        assert 0.0 <= overall <= 1.5, f"overall throughput out of range: {overall}"
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
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()

        # max_level left at its 0 default -> size*size (25) >> 2*size (10),
        # so the whole 5x5 factory is blanked.
        args = SftArgs(
            seed=1,
            size=size,
            num_samples=50,
            **TINY_ARCH_ARGS,
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
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()

        args = SftArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            **TINY_ARCH_ARGS,
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
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()

        args = SftArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            **TINY_ARCH_ARGS,
        )
        roll = run_rollout_eval(
            agent,
            args,
            {},
            device=torch.device("cpu"),
        )
        assert roll["overall"] == 0.0
        assert sum(roll["per_kind_n"].values()) == 0

    @staticmethod
    def _force_eot(agent, fire: bool):
        """Pin the EOT head's probability to ~1 (always fires) or ~0 (never)."""
        with torch.no_grad():
            agent.eot_head.weight.zero_()
            agent.eot_head.bias.fill_(100.0 if fire else -100.0)
        return agent

    def test_eot_head_stops_the_rollout(self, registered_env):
        """The sampled EOT action ends the rollout, so the head decides how far
        the agent gets to build. A head that never fires runs to env-done; one
        that always fires stops before the first placement, so every rollout
        stops after one scored step and scores the untouched reset factory
        (throughput 0)."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()

        args = SftArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            **TINY_ARCH_ARGS,
        )
        val_seeds_to_kind = self._build_val_seeds_to_kind(size=size, num_kinds=4)

        never = run_rollout_eval(
            self._force_eot(agent, fire=False),
            args,
            val_seeds_to_kind,
            device=torch.device("cpu"),
            max_seeds=len(val_seeds_to_kind),
        )
        assert 0.0 <= never["overall"] <= 1.5

        always = run_rollout_eval(
            self._force_eot(agent, fire=True),
            args,
            val_seeds_to_kind,
            device=torch.device("cpu"),
            max_seeds=len(val_seeds_to_kind),
        )
        assert always["overall"] == 0.0, (
            "EOT firing before the first placement must score the reset factory (0)"
        )

        # The ASSERT: the head stops the rollout rather than just annotating it,
        # so an immediately-firing head places nothing at all.
        n_rollouts = len(val_seeds_to_kind)
        assert sum(always["per_kind_eot_step_n"].values()) == n_rollouts, (
            "an immediate stop must be exactly one scored step per rollout"
        )
        assert sum(never["per_kind_eot_step_n"].values()) > n_rollouts, (
            "a silent head must keep placing past the first step"
        )

    def test_rollout_eot_head_scoring(self, registered_env):
        """The rollout scores the EOT head on every state it visits, against
        ground truth: a pre-action state is a should-stop positive iff the
        factory is already complete (thput_normed >= 1.0). Forcing the head to
        never / always fire pins both ends. A head that
        always fires stops on the blank reset factory: one scored step per
        rollout, wrong every time, and it never reaches a completed factory to
        recall. A silent head keeps placing to env-done, so it is right on
        exactly the not-done steps and recalls no positive."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()

        args = SftArgs(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
            **TINY_ARCH_ARGS,
        )
        val_seeds_to_kind = self._build_val_seeds_to_kind(size=size, num_kinds=4)
        device = torch.device("cpu")
        max_seeds = len(val_seeds_to_kind)

        never = run_rollout_eval(
            self._force_eot(agent, fire=False), args, val_seeds_to_kind, device, max_seeds=max_seeds
        )
        always = run_rollout_eval(
            self._force_eot(agent, fire=True), args, val_seeds_to_kind, device, max_seeds=max_seeds
        )

        for roll in (never, always):
            assert 0.0 <= roll["eot_acc"] <= 1.0
            assert 0.0 <= roll["eot_pos_recall"] <= 1.0
            for kn, acc in roll["per_kind_eot_acc"].items():
                assert 0.0 <= acc <= 1.0
                if roll["per_kind_eot_step_n"][kn] == 0:
                    assert acc == 0.0, f"{kn}: no steps but acc {acc}"

        # Firing on the blank reset factory is wrong, and stopping there means
        # no completed factory is ever visited — so there is nothing to recall.
        always_steps = sum(always["per_kind_eot_step_n"].values())
        assert always_steps == max_seeds
        assert sum(always["per_kind_eot_pos_n"].values()) == 0
        assert always["eot_acc"] == 0.0
        assert always["eot_pos_recall"] == 0.0

        # A silent head runs the full trajectory: strictly more scored steps,
        # correct on exactly the not-done ones, and no positive recalled.
        never_steps = sum(never["per_kind_eot_step_n"].values())
        never_pos = sum(never["per_kind_eot_pos_n"].values())
        assert never_steps > always_steps
        assert never["eot_acc"] == pytest.approx((never_steps - never_pos) / never_steps)
        assert never["eot_pos_recall"] == 0.0, "silent head recalls no positive"

    def _run_with_recorded_proposals(self, monkeypatch):
        """Run a rollout eval with a FactorioEnv that records, for every
        step, the (entity, footprint) the *proposed anchor tile* held in the
        world just before the placement was applied. Returns that list."""
        size = 5
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
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
            **TINY_ARCH_ARGS,
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
        """The sampled tile pick must only ever propose legal placement
        targets: the anchor tile is not a source/sink and is buildable
        (FOOTPRINT == AVAILABLE) at the moment of every step."""
        markers = (str2ent("source").value, str2ent("sink").value)
        recorded = self._run_with_recorded_proposals(monkeypatch)
        assert recorded, "rollout should have proposed at least one tile"
        illegal = [
            (ent, foot)
            for ent, foot in recorded
            if ent in markers or foot != Footprint.AVAILABLE.value
        ]
        assert not illegal, (
            f"masking must never propose a marker/unbuildable tile; got {illegal}"
        )

    def _random_belt_layer(self, size, n_seeds=8):
        """An agent that lays belts on uniformly random tiles facing random
        ways and never stops, so its samples differ and a few connect; it
        ignores the weights' init, so the expected outcomes are fixed. Returns
        (agent, args, MOVE_ONE_ITEM seeds -> kind)."""
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()
        with torch.no_grad():
            for head in (agent.tile_logits, agent.dir_head, agent.eot_head):
                head.weight.zero_()
                head.bias.zero_()
            agent.eot_head.bias.fill_(-100.0)
            for head, idx in ((agent.ent_head, str2ent("transport_belt").value),
                              (agent.item_head, 0), (agent.misc_head, 0)):
                head.weight.zero_()
                head.bias.fill_(-100.0)
                head.bias[idx] = 100.0
        seeds = {}
        s = 700_000
        while len(seeds) < n_seeds:
            if build_factory(size=size, kind=LessonKind.MOVE_ONE_ITEM, seed=s) is not None:
                seeds[s] = LessonKind.MOVE_ONE_ITEM.value
            s += 1
        return agent, SftArgs(seed=1, size=size, **TINY_ARCH_ARGS), seeds

    @staticmethod
    def _records(agent, args, seeds, **kw):
        records = []
        run_rollout_eval(agent, args, seeds, torch.device("cpu"), max_seeds=len(seeds),
                         records=records, **kw)
        return {r["seed"]: r for r in records}

    def test_eval_is_deterministic_for_a_fixed_seed(self, registered_env):
        """Each sample's randomness is seeded by (run seed, factory, sample), so
        a rerun rebuilds the same factories however the work is batched."""
        agent, args, seeds = self._random_belt_layer(size=5)
        a = self._records(agent, args, seeds, best_of=4, num_envs=8)
        assert a == self._records(agent, args, seeds, best_of=4, num_envs=8)
        assert a == self._records(agent, args, seeds, best_of=4, num_envs=3)

    def test_best_of_n_scores_the_best_sample(self, registered_env):
        """Sample j draws the same noise whatever N is, so N=16's samples include
        N=1's: each factory's score can only rise with N, and here it does."""
        agent, args, seeds = self._random_belt_layer(size=5)
        one = self._records(agent, args, seeds, best_of=1)
        many = self._records(agent, args, seeds, best_of=16)
        assert all(many[s]["thput"] >= one[s]["thput"] for s in seeds)
        assert any(many[s]["thput"] > one[s]["thput"] for s in seeds)


class TestLegalTileMask:
    """Deterministic unit coverage of the eval-time legal-tile mask helper."""

    def _obs(self, size):
        """A single all-empty, all-buildable observation of shape (1, C, W, H)."""
        C = len(Channel)
        obs = torch.zeros(1, C, size, size)
        obs[0, Channel.ENTITIES.value] = str2ent("empty").value
        obs[0, Channel.FOOTPRINT.value] = Footprint.AVAILABLE.value
        return obs

    def test_argmax_skips_marker_tile(self):
        """When the top-logit tile holds a source, the masked argmax must fall
        to the next-best *legal* tile instead of livelocking on the marker."""
        size = 3
        obs = self._obs(size)
        # Source on tile (0, 0) -> flat index 0, x-major (idx = x * size + y).
        obs[0, Channel.ENTITIES.value, 0, 0] = str2ent("source").value
        # Logits favour the marker tile 0, then tile 5 as runner-up.
        logits = torch.full((1, size * size), -1.0)
        logits[0, 0] = 10.0
        logits[0, 5] = 5.0

        assert logits.argmax(dim=1).item() == 0, "sanity: unmasked picks the marker"
        masked = logits.masked_fill(~_legal_tile_mask(obs), float("-inf"))
        assert masked.argmax(dim=1).item() == 5, "masked must skip to legal runner-up"
        assert masked[0, 0] == float("-inf"), "marker tile must be -inf"

    def test_argmax_skips_unbuildable_tile(self):
        """UNAVAILABLE (unbuildable) tiles are masked out just like markers,
        even when the tile is otherwise empty."""
        size = 3
        obs = self._obs(size)
        # Tile (1, 1) -> flat index 4 is empty but not buildable.
        obs[0, Channel.FOOTPRINT.value, 1, 1] = Footprint.UNAVAILABLE.value
        logits = torch.full((1, size * size), -1.0)
        logits[0, 4] = 10.0
        logits[0, 2] = 5.0

        masked = logits.masked_fill(~_legal_tile_mask(obs), float("-inf"))
        assert masked.argmax(dim=1).item() == 2
        assert masked[0, 4] == float("-inf")

    def test_all_illegal_row_is_nan_free(self):
        """A grid of markers leaves every tile illegal; the masked argmax
        must still return a finite index (tile 0) rather than NaN-ing out."""
        size = 3
        obs = self._obs(size)
        obs[0, Channel.ENTITIES.value] = str2ent("sink").value
        logits = torch.randn(1, size * size)

        masked = logits.masked_fill(~_legal_tile_mask(obs), float("-inf"))
        assert torch.isinf(masked).all(), "every tile should be masked to -inf"
        idx = masked.argmax(dim=1).item()
        assert idx == 0, "all-illegal row falls back to tile 0"


class TestEotHead:
    """Tests for the binary end-of-turn head."""

    def test_eot_head_exists_and_forwards(self, registered_env):
        """AgentCNN must expose an eot_head producing a single logit per
        observation."""
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()

        assert hasattr(agent, "eot_head"), "AgentCNN must have an eot_head"
        # Forward a fake batch through the encoder + eot_head.
        B, C, W, H = 4, agent.channels, agent.width, agent.height
        x = torch.zeros((B, C, W, H), dtype=torch.float32)
        enc = agent.encoder(agent._encode_input(x))
        logits = agent.eot_logit(enc)
        assert logits.shape == (B,), (
            f"eot_head output should be (B,), got {logits.shape}"
        )

    def test_eot_prob_and_should_stop_shapes(self, registered_env):
        """`eot_prob` returns a [0,1] tensor of shape (B,); `eot_should_stop`
        returns a bool tensor of the same shape. These are the methods
        inference rollouts call to decide 'I'm done here'."""
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, 5, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
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
            (1_000_000, "1M"),
            (2_500_000, "2.5M"),
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
            batch_size=1024,
            lr=3e-4,
            width=48,
            levels=3,
        )
        assert _artifact_name(args) == "dn-s16-n200k-bs1024-lr3e-4-w48x3b2a4"

    def test_artifact_name_stable_across_runs(self):
        """Two SftArgs with the same hyperparams must produce the same
        artifact name (otherwise we lose version collapsing)."""
        a = SftArgs(seed=1)
        b = SftArgs(seed=999)  # seed isn't part of the artifact name
        assert _artifact_name(a) == _artifact_name(b)


class TestLRSchedule:
    """The WSD schedule should warm up to `args.lr` over an absolute step
    count, hold `args.lr` flat, and cool down to `args.lr * args.min_lr_ratio`
    over the final `cooldown_frac` of the run."""

    def _step_n(self, scheduler, optimizer, n):
        lrs = []
        for _ in range(n):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
        return lrs

    def _lrs(self, args, total_steps):
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, total_steps, args)
        return self._step_n(scheduler, optimizer, total_steps)

    def test_warmup_stable_decay(self):
        args = SftArgs(lr=1e-3, warmup_steps=10, cooldown_frac=0.2, min_lr_ratio=0.01)
        lrs = self._lrs(args, 100)

        assert lrs[0] == pytest.approx(args.lr / 10, rel=1e-6)
        assert lrs[10:81] == pytest.approx([args.lr] * 71, rel=1e-9)
        # (1-sqrt) drops fast: a quarter into the cooldown it is already halfway.
        assert lrs[85] == pytest.approx(args.lr * 0.5, rel=0.02)
        assert lrs[-1] == pytest.approx(args.lr * args.min_lr_ratio, abs=args.lr * 1e-9)

    def test_shared_prefix_across_run_lengths(self):
        """The property the schedule exists for: two runs of different budgets
        are on the same LR at the same step until the shorter one starts
        cooling, so their curves are directly comparable."""
        args = SftArgs(lr=1e-3, warmup_steps=50, cooldown_frac=0.2)
        short = self._lrs(args, 1_000)
        long = self._lrs(args, 100_000)
        assert short[:800] == pytest.approx(long[:800], rel=1e-9)

    def test_handles_run_shorter_than_warmup(self):
        args = SftArgs(lr=1e-3, warmup_steps=300)
        model = torch.nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = build_lr_schedule(optimizer, 1, args)
        optimizer.step()
        scheduler.step()  # must not raise
