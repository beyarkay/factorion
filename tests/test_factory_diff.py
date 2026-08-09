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

from ppo import AgentCNN, make_env  # noqa: E402
from sft import SftArgs, run_rollout_eval  # noqa: E402

ENV_ID = "factorion/FactorioEnv-v0-diff-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


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
