"""Tests for the 'uniform' num_missing_entities reset option (partial blanking)."""

import os
import sys

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorion import LessonKind  # noqa: E402
from ppo import FactorioEnv  # noqa: E402

SIZE = 11
OPTS = {"num_missing_entities": "uniform", "kind": LessonKind.FACTORY_1_INGREDIENT}


def _missing(seed: int) -> int:
    env = FactorioEnv(size=SIZE, idx=0)
    env.reset(seed=seed, options=OPTS)
    return env.min_entities_required


def test_uniform_blanking_varies_across_episodes():
    counts = {_missing(seed) for seed in range(20)}
    assert len(counts) > 1, f"blank count never varied: {counts}"
    assert all(c >= 1 for c in counts)


def test_uniform_blanking_deterministic_per_seed():
    assert _missing(7) == _missing(7)


def test_numeric_option_still_exact():
    env = FactorioEnv(size=SIZE, idx=0)
    env.reset(seed=3, options={"num_missing_entities": 2,
                               "kind": LessonKind.FACTORY_1_INGREDIENT})
    assert env.min_entities_required == 2


def test_uniform_can_leave_factory_partially_built():
    # With draws in [1, size*size] and FACTORY_1_INGREDIENT factories holding
    # tens of removable units, some seeds must draw below the removable count.
    for seed in range(40):
        env = FactorioEnv(size=SIZE, idx=0)
        env.reset(seed=seed, options=OPTS)
        env_full = FactorioEnv(size=SIZE, idx=0)
        env_full.reset(seed=seed, options={"num_missing_entities": float("inf"),
                                           "kind": LessonKind.FACTORY_1_INGREDIENT})
        if env.min_entities_required < env_full.min_entities_required:
            return
    raise AssertionError("no seed in 0..39 produced a partial blank")
