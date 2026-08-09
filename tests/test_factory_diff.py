"""Tests for the side-by-side policy diff (factory_diff.py)."""

import os
import sys

import gymnasium as gym
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from helpers import TINY_ARCH, LessonKind, build_factory  # noqa: E402

from factory_diff import diff_markdown  # noqa: E402
from ppo import AgentCNN, make_env  # noqa: E402
from sft import SftArgs, run_rollout_eval  # noqa: E402

ENV_ID = "factorion/FactorioEnv-v0-diff-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


def _record(kind, seed, thput, cost=1.0, render="b>"):
    return {
        "seed": seed,
        "kind": kind,
        "thput": thput,
        "entity_cost": cost,
        "render": render,
    }


class TestRolloutRecords:
    """run_rollout_eval's optional per-factory capture — the raw material the
    diff is assembled from."""

    def test_one_record_per_scored_factory(self, registered_env):
        size = 7
        seeds = {}
        s = 700_000
        while len(seeds) < 3 and s < 700_500:
            if build_factory(size=size, kind=LessonKind.MOVE_ONE_ITEM, seed=s) is not None:
                seeds[s] = LessonKind.MOVE_ONE_ITEM.value
            s += 1
        assert len(seeds) == 3

        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()
        agent.eval()

        records = []
        roll = run_rollout_eval(
            agent,
            SftArgs(seed=1, size=size),
            seeds,
            torch.device("cpu"),
            max_seeds=len(seeds),
            records=records,
        )
        assert len(records) == roll["per_kind_n"]["MOVE_ONE_ITEM"] == len(seeds)
        assert {r["seed"] for r in records} == set(seeds)
        for r in records:
            assert r["kind"] == "MOVE_ONE_ITEM"
            assert 0.0 <= r["thput"] <= 1.0
            assert r["entity_cost"] >= 0.0
            # The render is the real grid: `size` rows of space-separated tiles.
            rows = r["render"].splitlines()
            assert len(rows) == size and all(len(row.split()) == size for row in rows)
        assert roll["per_kind"]["MOVE_ONE_ITEM"] == pytest.approx(
            sum(r["thput"] for r in records) / len(records)
        )

    def test_records_default_off(self, registered_env):
        """The eval must stay allocation-free for training runs that don't ask."""
        size = 7
        seeds = {700_000: LessonKind.MOVE_ONE_ITEM.value}
        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "test")])
        agent = AgentCNN(envs, **TINY_ARCH)
        envs.close()
        roll = run_rollout_eval(
            agent, SftArgs(seed=1, size=size), seeds, torch.device("cpu"), max_seeds=1
        )
        assert "overall" in roll


class TestDiffMarkdown:
    def test_only_differing_pairs_rendered_biggest_gap_first(self):
        pr = [
            _record("MOVE_ONE_ITEM", 1, 0.9, render="PRSMALL"),
            _record("MOVE_ONE_ITEM", 2, 0.5),
            _record("SPLITTER_SPLIT", 3, 0.1, render="PRBIG"),
        ]
        main = [
            _record("MOVE_ONE_ITEM", 1, 0.8, render="MAINSMALL"),
            _record("MOVE_ONE_ITEM", 2, 0.5),
            _record("SPLITTER_SPLIT", 3, 0.9, render="MAINBIG"),
        ]
        md = diff_markdown(pr, main)

        assert "3 held-out factories" in md and "2 ended at a different" in md
        # The unchanged pair is tallied but never rendered.
        assert "seed 2" not in md
        assert md.index("seed 3") < md.index("seed 1"), "biggest gap first"
        assert "Δthput -0.800" in md and "Δthput +0.100" in md
        for token in ("PRBIG", "MAINBIG", "PRSMALL", "MAINSMALL"):
            assert token in md
        # Per-lesson tally: one better, one worse, one lesson at each.
        assert "| `SPLITTER_SPLIT` | 1 | 1 | 0 | 1 |" in md
        assert "| `MOVE_ONE_ITEM` | 2 | 1 | 1 | 0 |" in md

    def test_identical_sides_say_so(self):
        pr = [_record("MOVE_ONE_ITEM", 1, 0.4)]
        md = diff_markdown(pr, [_record("MOVE_ONE_ITEM", 1, 0.4)])
        assert "identically-performing" in md
        assert "<details>" not in md

    def test_unpaired_factories_ignored(self):
        md = diff_markdown(
            [_record("MOVE_ONE_ITEM", 1, 0.4)], [_record("MOVE_ONE_ITEM", 9, 0.9)]
        )
        assert md == ""

    def test_truncates_to_the_comment_limit(self):
        pr = [_record("MOVE_ONE_ITEM", i, 1.0, render="x" * 200) for i in range(200)]
        main = [_record("MOVE_ONE_ITEM", i, 0.0, render="y" * 200) for i in range(200)]
        md = diff_markdown(pr, main, max_chars=8000)
        assert len(md) <= 8000
        assert "further factories omitted" in md
        assert md.rstrip().endswith("</details>")

    def test_side_labels_carry_through(self):
        md = diff_markdown(
            [_record("MOVE_ONE_ITEM", 1, 0.4)],
            [_record("MOVE_ONE_ITEM", 1, 0.9)],
            pr_label="cmp-abc-pr",
            main_label="cmp-abc-main",
        )
        assert "cmp-abc-pr vs cmp-abc-main" in md
        assert "cmp-abc-pr  thput 0.400" in md
        assert "cmp-abc-main  thput 0.900" in md
