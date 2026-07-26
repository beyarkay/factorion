"""Static check that the terminal-reward design makes *finishing* a lesson the
discounted-return-optimal policy.

PPO maximises the discounted return, so at any state along a build the agent
compares "EOT now" (bank the current terminal reward r_i immediately) against
"keep building" (a better reward r_j, but j-i steps later, worth
gamma^(j-i) * r_j today). If some low-throughput prefix satisfies
r_i > gamma^(j-i) * r_j for every higher-throughput state j, the reward design
actively teaches premature EOT on that lesson — no amount of RL compute fixes
an incentive that points the wrong way.

The invariant tested per lesson: replay an expert build from a fully-blank
grid and find the stopping index i* = argmax_i gamma^i * r_i (the
discounted-return-optimal stop). That index must land on a (near-)maximum
raw-throughput state. It deliberately does NOT require i* == N: the
entity-cost penalty can make the final expert entity reward-neutral or
slightly negative (e.g. an output inserter that adds cost but no flow), which
is fine — eval scores throughput, not entity count.

Lessons whose partial builds all have zero throughput (a belt line delivers
nothing until it connects) pass under any gamma; the lessons at risk are the
ones with *incremental* throughput — parallel branches (SPLITTER_SPLIT,
FACTORY_1_INGREDIENT) or a pre-existing protected flow (CROSS_UNDER_BELT,
where stopping at step 0 already banks the obstruction line's delivery).
"""

import math
import os
import random
import sys

import numpy as np
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import factorion_rs  # noqa: E402
from factorion import LessonKind, blank_entities, build_factory  # noqa: E402
from ppo import (  # noqa: E402
    _CH_ENT,
    _ENTITY_FOOTPRINT_AREAS,
    _ENTITY_UNIT_COSTS,
    _N_ENTITIES,
)
from sft import extract_expert_actions  # noqa: E402
from training_config import PpoArgs  # noqa: E402

SIZE = 11
SEEDS_PER_KIND = 3
ORDERS_PER_SEED = 2
# i* must reach this fraction of the trajectory's best raw throughput.
THPUT_TOL = 0.99


def _terminal_stats(world_CWH: torch.Tensor) -> tuple[float, float]:
    """(raw throughput, terminal reward) for a world, mirroring
    FactorioEnv.step's terminal-reward computation exactly."""
    thput, _ = factorion_rs.simulate_throughput(
        world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
    )
    ent_np = world_CWH[_CH_ENT].numpy()
    counts = np.bincount(
        np.asarray(ent_np, dtype=np.int64).ravel(), minlength=_N_ENTITIES
    )[:_N_ENTITIES]
    cost = float(np.dot(counts / _ENTITY_FOOTPRINT_AREAS, _ENTITY_UNIT_COSTS))
    reward = thput / (1.0 + PpoArgs.entity_cost_scale * cost)
    if PpoArgs.reward_symlog_r0 > 0:
        reward = math.log1p(reward / PpoArgs.reward_symlog_r0)
    return float(thput), float(reward)


def _build_trajectories(kind: LessonKind):
    """Yield (thputs, rewards) arrays over the prefix states of expert builds
    (index i = state after i placements, i in [0, N])."""
    produced, seed = 0, 300
    while produced < SEEDS_PER_KIND and seed < 500:
        seed += 1
        factory = build_factory(size=SIZE, kind=kind, seed=seed)
        if factory is None:
            continue
        produced += 1
        task, _ = blank_entities(factory, num_missing_entities=float("inf"))
        for order in range(ORDERS_PER_SEED):
            random.seed(seed * 10 + order)
            pairs = extract_expert_actions(factory.world_CWH, task)
            stats = [_terminal_stats(p[0].to(torch.float32)) for p in pairs]
            yield (
                np.array([t for t, _ in stats]),
                np.array([r for _, r in stats]),
            )


def _required_gamma(thputs: np.ndarray, rewards: np.ndarray) -> float:
    """Smallest discount under which some near-max-throughput state beats
    every earlier prefix, i.e. gamma^(j-i) * r_j >= r_i for all i < j."""
    tmax = thputs.max()
    best = 1.0
    for j in range(len(rewards)):
        if thputs[j] < THPUT_TOL * tmax or rewards[j] <= 0:
            continue
        need = 0.0
        for i in range(j):
            if rewards[i] <= 0:
                continue
            need = max(need, (rewards[i] / rewards[j]) ** (1.0 / (j - i)))
        best = min(best, need)
    return best


# Lessons whose incremental-throughput structure needs a higher gamma than the
# current default: at PpoArgs.gamma the discounted-optimal policy stops at a
# throughput-inferior prefix (SPLITTER_SPLIT: one branch of two;
# CROSS_UNDER_BELT: stop at step 0 and bank the protected obstruction line's
# flow; FACTORY_1_INGREDIENT: stop after the first working arm). Measured
# required gammas ~0.973 / 0.978 / 0.996 — strict xfail so raising the default
# past ~0.997 forces removing these marks.
_KNOWN_UNDERDISCOUNTED = {
    LessonKind.SPLITTER_SPLIT,
    LessonKind.CROSS_UNDER_BELT,
    LessonKind.FACTORY_1_INGREDIENT,
}


@pytest.mark.parametrize(
    "kind",
    [
        pytest.param(
            k,
            marks=pytest.mark.xfail(
                reason=(
                    "PpoArgs.gamma is too low for this lesson's build length: "
                    "stopping at a partial-throughput prefix out-earns "
                    "finishing (see module docstring)"
                ),
                strict=True,
            ),
        )
        if k in _KNOWN_UNDERDISCOUNTED
        else k
        for k in LessonKind
    ],
    ids=[k.name for k in LessonKind],
)
def test_discounted_optimal_stop_is_max_throughput(kind):
    gamma = PpoArgs.gamma
    for thputs, rewards in _build_trajectories(kind):
        tmax = thputs.max()
        if tmax == 0:
            continue
        disc = (gamma ** np.arange(len(rewards))) * rewards
        istar = int(np.argmax(disc))
        assert thputs[istar] >= THPUT_TOL * tmax, (
            f"{kind.name}: with gamma={gamma} the discounted-optimal stop is "
            f"after {istar}/{len(rewards) - 1} placements at "
            f"{thputs[istar]:.2f} items/s, below the trajectory max "
            f"{tmax:.2f} items/s — the reward design prefers a partial build. "
            f"This lesson needs gamma >= "
            f"{_required_gamma(thputs, rewards):.4f}."
        )


def test_report_required_gamma_per_lesson(capsys):
    """Always-passing diagnostic: print the minimum gamma each lesson needs so
    the table shows up in verbose test output (`pytest -s`)."""
    rows = []
    for kind in LessonKind:
        need = 0.0
        for thputs, rewards in _build_trajectories(kind):
            if thputs.max() == 0:
                continue
            need = max(need, _required_gamma(thputs, rewards))
        rows.append((kind.name, need))
    with capsys.disabled():
        print(f"\nminimum viable gamma per lesson (PpoArgs.gamma={PpoArgs.gamma}):")
        for name, need in rows:
            flag = "" if PpoArgs.gamma >= need else "  <-- UNDER-DISCOUNTED"
            print(f"  {name:35} gamma_min={need:.4f}{flag}")
    overall = max(need for _, need in rows)
    # Sanity bound only — the per-lesson xfail marks above are the real gate.
    assert overall <= 1.0
