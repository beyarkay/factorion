"""Tests for PPO's best-of-N rollout eval (`ppo.run_rollout_eval`)."""

import os
import sys
from types import SimpleNamespace

import gymnasium as gym
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import TINY_ARCH, Channel, LessonKind, build_factory, str2ent  # noqa: E402
from factorion import Footprint  # noqa: E402
import ppo  # noqa: E402
from ppo import (  # noqa: E402
    AgentCNN,
    FactorioEnv,
    _legal_tile_mask,
    _solved_assembler_recipes,
    make_env,
    run_rollout_eval,
)

ENV_ID = "factorion/FactorioEnv-v0-rollout-eval-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


def _eval_args(seed=1, size=5, max_level=0, **_unused):
    """The fields run_rollout_eval reads."""
    return SimpleNamespace(seed=seed, size=size, max_level=max_level)


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
            agent, _eval_args(seed=1, size=size), seeds, torch.device("cpu"), max_seeds=2
        )
        assert {"asm_item_acc", "per_kind_asm_item_acc", "per_kind_asm_n"} <= set(roll)
        assert 0.0 <= roll["asm_item_acc"] <= 1.0

    def test_correct_recipe_scores_one_wrong_scores_zero(self, registered_env):
        size = 9
        recipe, other = self._two_recipes(size)
        seeds = self._seeds_with_recipe(size, recipe, n=8)
        assert len(seeds) >= 4, "need several matching-recipe factories"
        args = _eval_args(seed=1, size=size)

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
            _eval_args(seed=1, size=size),
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

        args = _eval_args(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
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
        args = _eval_args(
            seed=1,
            size=size,
            num_samples=50,
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

        args = _eval_args(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
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

        args = _eval_args(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
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

        args = _eval_args(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
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

        args = _eval_args(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
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

        monkeypatch.setattr(ppo, "FactorioEnv", RecordingEnv)

        args = _eval_args(
            seed=1,
            size=size,
            num_samples=50,
            max_level=2 * size,
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
        return agent, _eval_args(seed=1, size=size), seeds

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
    """The AgentCNN end-of-turn head."""

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

