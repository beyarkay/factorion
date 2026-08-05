"""PpoArgs.exclude_train_kinds: the training-mix kind filter.

Deep trials (depth 2/3) have delivered zero reward in every recorded run yet
consume ~30% of env steps; the default excludes them from training sampling
while the eval set still scores every kind.
"""

import os
import sys

import pytest

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorion import LessonKind  # noqa: E402
from ppo import FactorioEnv, train_kinds_from_args  # noqa: E402
from training_config import PpoArgs  # noqa: E402


def test_default_excludes_only_the_deep_trials():
    kinds = train_kinds_from_args(PpoArgs())
    assert kinds is not None
    names = {k.name for k in kinds}
    assert "TRIAL_RECIPE_TREE_DEPTH_2" not in names
    assert "TRIAL_RECIPE_TREE_DEPTH_3" not in names
    assert names == {k.name for k in LessonKind} - {
        "TRIAL_RECIPE_TREE_DEPTH_2", "TRIAL_RECIPE_TREE_DEPTH_3"
    }
    assert "TRIAL_RECIPE_TREE_DEPTH_1" in names, "depth-1 must stay trainable"


def test_empty_exclusion_means_all_kinds():
    assert train_kinds_from_args(PpoArgs(exclude_train_kinds="")) is None


def test_unknown_name_is_rejected():
    with pytest.raises(ValueError, match="unknown LessonKind"):
        train_kinds_from_args(PpoArgs(exclude_train_kinds="NOT_A_KIND"))


def test_env_only_samples_allowed_kinds():
    """Across many episode resets the env draws exclusively from train_kinds;
    an explicit per-reset kind still overrides the filter (the eval path)."""
    allowed = [LessonKind.MOVE_ONE_ITEM, LessonKind.MOVE_ONE_ITEM_CHAOS]
    env = FactorioEnv(size=11, max_steps=10, idx=0, train_kinds=allowed)
    seen = set()
    for seed in range(25):
        env.reset(seed=seed)
        seen.add(env._kind)
    assert seen <= set(allowed)
    assert len(seen) > 1, "sampling should reach more than one allowed kind"

    env.reset(seed=0, options={"kind": LessonKind.TRIAL_RECIPE_TREE_DEPTH_2})
    assert env._kind == LessonKind.TRIAL_RECIPE_TREE_DEPTH_2


def test_env_rejects_empty_kind_list():
    with pytest.raises(ValueError, match="non-empty"):
        FactorioEnv(size=11, max_steps=10, idx=0, train_kinds=[])
