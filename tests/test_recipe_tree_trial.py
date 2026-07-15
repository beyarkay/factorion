"""TRIAL_RECIPE_TREE_DEPTH_* — trial kinds: scenarios with no known solution.

A trial's built "factory" is only the markers: a random AM1-craftable sink
item plus one source per item of a randomly-expanded frontier of its
ingredient tree (DEPTH_N = the deepest expanded chain is exactly N crafting
stages long). There is nothing to imitate — trials yield no SFT pairs and are
trained on by RL alone, with throughput normalized by the analytic
`Factory.max_throughput` ceiling instead of a simulated reference solution.
"""

import os
import sys

import numpy as np
import pytest

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorion import (  # noqa: E402
    LESSON_IS_TRIAL,
    Channel,
    LessonKind,
    blank_entities,
    build_factory,
    items,
    recipes,
    str2ent,
)
from ppo import FactorioEnv  # noqa: E402

SIZE = 10

TRIAL_KINDS = {
    LessonKind.TRIAL_RECIPE_TREE_DEPTH_1: 1,
    LessonKind.TRIAL_RECIPE_TREE_DEPTH_2: 2,
    LessonKind.TRIAL_RECIPE_TREE_DEPTH_3: 3,
}

_SRC_ID = str2ent("source").value
_SNK_ID = str2ent("sink").value


def _markers(factory):
    """(sink_item_name, [source_item_names]) of a trial factory, asserting the
    world contains nothing but one sink and its sources."""
    ent = factory.world_CWH[Channel.ENTITIES.value]
    itm = factory.world_CWH[Channel.ITEMS.value]
    sink = None
    sources = []
    for x, y in zip(*np.nonzero(ent.numpy())):
        e = int(ent[x, y])
        name = items[int(itm[x, y])].name
        if e == _SRC_ID:
            sources.append(name)
        elif e == _SNK_ID:
            assert sink is None, "trial must have exactly one sink"
            sink = name
        else:
            raise AssertionError(f"unexpected entity {items[e].name} at ({x},{y})")
    assert sink is not None and sources
    return sink, sources


def _am1_recipe(item_name):
    r = recipes.get(item_name)
    if r is not None and "assembling_machine_1" in r.produced_by:
        return r
    return None


def _covers(item_name, source_names):
    """`source_names` is a valid frontier of `item_name`'s ingredient tree:
    every ingredient is either a source or itself craftable from the sources."""
    r = _am1_recipe(item_name)
    if r is None:
        return False
    return all(
        ing in source_names or _covers(ing, source_names) for ing in r.consumes
    )


class TestTrialFlags:
    def test_exactly_the_trial_kinds_are_flagged(self):
        assert {k for k, t in LESSON_IS_TRIAL.items() if t} == set(TRIAL_KINDS)


class TestTrialFactory:
    @pytest.mark.parametrize("kind", list(TRIAL_KINDS))
    def test_markers_only_and_nothing_to_blank(self, kind):
        for seed in range(10):
            f = build_factory(size=SIZE, kind=kind, seed=seed)
            if f is None:
                continue
            _markers(f)  # asserts sources + one sink and nothing else
            assert f.total_entities == 0
            assert f.protected_positions == frozenset()
            partial, min_required = blank_entities(f, num_missing_entities=float("inf"))
            assert min_required == 0
            assert (partial == f.world_CWH).all()

    @pytest.mark.parametrize("kind", list(TRIAL_KINDS))
    def test_analytic_max_throughput(self, kind):
        """The normalization ceiling is one fully-fed assembler's output rate
        of the sink recipe — positive without any reference solution."""
        f = build_factory(size=SIZE, kind=kind, seed=0)
        assert f is not None
        sink, _ = _markers(f)
        assert f.max_throughput == recipes[sink].produces[sink] > 0

    @pytest.mark.parametrize("kind,depth", list(TRIAL_KINDS.items()))
    def test_sources_are_a_frontier_of_the_sink_tree(self, kind, depth):
        seen_expanded = False
        for seed in range(20):
            f = build_factory(size=SIZE, kind=kind, seed=seed)
            if f is None:
                continue
            sink, sources = _markers(f)
            assert len(set(sources)) == len(sources), "duplicate source items"
            assert _covers(sink, set(sources)), (
                f"seed={seed}: sources {sources} don't resolve {sink}"
            )
            direct = set(recipes[sink].consumes)
            if depth == 1:
                assert set(sources) == direct, (
                    f"seed={seed}: depth 1 must use exactly the direct ingredients"
                )
            else:
                seen_expanded |= set(sources) != direct
        if depth > 1:
            assert seen_expanded, "no seed ever expanded past the direct ingredients"


class TestTrialEnv:
    def test_env_episode_starts_from_the_markers(self):
        """Blanking is a no-op on a trial, so the episode's start state is the
        markers-only world, and normalization uses the analytic ceiling."""
        env = FactorioEnv(size=SIZE, idx=0)
        obs, info = env.reset(
            seed=3, options={"kind": LessonKind.TRIAL_RECIPE_TREE_DEPTH_2}
        )
        assert info["kind"] == LessonKind.TRIAL_RECIPE_TREE_DEPTH_2.value
        assert env._max_throughput > 0
        assert env.min_entities_required == 0
        assert (obs == env._solved_world_CWH.numpy()).all()
        action = {
            "xy": np.array([0, 0]),
            "entity": 0,
            "direction": 0,
            "item": 0,
            "misc": 0,
        }
        _obs, _r, terminated, truncated, info = env.step(action)
        assert not terminated
        assert 0.0 <= info["thput_normed"] <= 1.0


class TestTrialSftExclusion:
    def test_demo_stream_never_draws_a_trial(self):
        from sft import _iter_demo_pairs

        trial_values = {k.value for k in TRIAL_KINDS}
        rows = list(_iter_demo_pairs(SIZE, SIZE * SIZE, base_seed=1, worker_id=0,
                                     num_workers=1, target=60))
        assert rows, "sampler produced no pairs at all"
        assert all(row[9] not in trial_values for row in rows)
