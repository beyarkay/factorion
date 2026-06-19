"""Throughput is normalized by each factory's own max, not a fixed 15.

The env used to divide raw throughput by a hardcoded 15.0. That only made
sense for the simplest belt factory (max == 15 items/s). Lessons like
ASSEMBLE_1IN_1OUT have a max throughput far below 15 (~0.86 items/s, recipe
rate-limited), so a *perfectly* built factory scored ~0.057 under the old
scheme — and, because termination required normalized throughput to reach
1.0, such a factory could never terminate.

Now the env reports two values in `info`:

- ``thput_raw``   — raw items/second (reference-free; what RL optimizes).
- ``thput_normed``— ``thput_raw / per_factory_max`` clamped to [0, 1], so a
  perfectly-rebuilt factory scores exactly 1.0 regardless of absolute speed.

See GH issue #161 for why the per-factory max (which needs the scripted
solution) is fine now but must change for arbitrary RL rollouts.
"""

import os
import sys

import numpy as np
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion_rs  # noqa: E402
from factorion import LessonKind, build_factory  # noqa: E402
from ppo import FactorioEnv  # noqa: E402


def _noop_action():
    """Place ``empty`` at (0,0) — a no-op on tiles that are already empty."""
    return {
        "xy": np.array([0, 0]),
        "entity": 0,
        "direction": 0,
        "item": 0,
        "misc": 0,
    }


def _solved_max(size, kind, seed):
    """Raw items/s of the complete, correct factory for (size, kind, seed)."""
    factory = build_factory(size=size, kind=kind, seed=seed)
    assert factory is not None
    tp, _ = factorion_rs.simulate_throughput(
        factory.world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
    )
    return float(tp)


# A factory whose max throughput is FAR below 15 (recipe rate-limited). Chosen
# deterministically; (0,0) is empty in its solved layout so the noop is safe.
SUB15_KIND = LessonKind.ASSEMBLE_1IN_1OUT
SUB15_SIZE = 7
SUB15_SEED = 0


def test_sub15_factory_max_is_well_below_15():
    """Guard the premise: this lesson's perfect throughput is nowhere near 15."""
    mx = _solved_max(SUB15_SIZE, SUB15_KIND, SUB15_SEED)
    assert 0.0 < mx < 1.0, f"expected a sub-15 max, got {mx}"


def test_solved_sub15_factory_scores_one_not_a_fraction():
    """A perfectly-built sub-15 factory scores 1.0 — the headline fix.

    Under the old fixed /15.0 it would have scored max/15 (~0.057) and could
    never terminate.
    """
    env = FactorioEnv(size=SUB15_SIZE, idx=0)
    env.reset(
        seed=SUB15_SEED,
        options={"num_missing_entities": 0, "kind": SUB15_KIND},
    )
    _, _, terminated, truncated, info = env.step(_noop_action())

    mx = _solved_max(SUB15_SIZE, SUB15_KIND, SUB15_SEED)

    assert info["thput_normed"] == pytest.approx(1.0)
    # Reaching the per-factory max scores 1.0, but on this branch a solve does
    # NOT auto-terminate — the agent ends the episode via the eot action, so a
    # no-op step on a solved factory keeps running.
    assert terminated is False
    assert truncated is False
    # The old scheme would have produced this fraction instead of 1.0.
    assert mx / 15.0 < 0.1
    assert info["thput_normed"] != pytest.approx(mx / 15.0)


def test_thput_raw_is_items_per_second():
    """thput_raw is the unnormalized rate, equal to the solved factory's max."""
    env = FactorioEnv(size=SUB15_SIZE, idx=0)
    env.reset(
        seed=SUB15_SEED,
        options={"num_missing_entities": 0, "kind": SUB15_KIND},
    )
    _, _, _, _, info = env.step(_noop_action())

    mx = _solved_max(SUB15_SIZE, SUB15_KIND, SUB15_SEED)
    # Raw is in items/s (well below 1.0 here), NOT divided by 15.
    assert info["thput_raw"] == pytest.approx(mx)
    assert info["thput_normed"] == pytest.approx(info["thput_raw"] / mx)


def test_info_exposes_new_keys_only():
    """The env exposes thput_raw + thput_normed and no longer a 'throughput' key."""
    env = FactorioEnv(size=SUB15_SIZE, idx=0)
    _, reset_info = env.reset(
        seed=SUB15_SEED,
        options={"num_missing_entities": 0, "kind": SUB15_KIND},
    )
    assert "thput_raw" in reset_info and "thput_normed" in reset_info
    assert "throughput" not in reset_info

    _, _, _, _, step_info = env.step(_noop_action())
    assert "thput_raw" in step_info and "thput_normed" in step_info
    assert "throughput" not in step_info


def test_blanked_factory_scores_below_one():
    """A heavily-blanked factory (nothing rebuilt) scores < 1.0 and >= 0."""
    env = FactorioEnv(size=SUB15_SIZE, idx=0)
    env.reset(
        seed=SUB15_SEED,
        options={"num_missing_entities": 99, "kind": SUB15_KIND},
    )
    _, _, terminated, _, info = env.step(_noop_action())
    assert 0.0 <= info["thput_normed"] < 1.0
    assert terminated is False


@pytest.mark.parametrize(
    "kind",
    [
        LessonKind.MOVE_ONE_ITEM,
        LessonKind.ASSEMBLE_1IN_1OUT,
        LessonKind.ASSEMBLE_2IN_1OUT,
    ],
)
def test_normed_stays_in_unit_interval(kind):
    """thput_normed is always in [0, 1] regardless of the factory's abs speed."""
    size = 7
    for seed in range(6):
        if build_factory(size=size, kind=kind, seed=seed) is None:
            continue
        env = FactorioEnv(size=size, idx=0)
        env.reset(
            seed=seed,
            options={"num_missing_entities": 0, "kind": kind},
        )
        _, _, _, _, info = env.step(_noop_action())
        assert 0.0 <= info["thput_normed"] <= 1.0
        assert info["thput_raw"] >= 0.0
