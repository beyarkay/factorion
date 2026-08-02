"""The PPO update must not train on Gymnasium's NEXT_STEP autoreset step (#233).

When a sub-env terminates at ``step()`` call N, call N+1 ignores the action,
resets the env and returns ``reward=0, terminated=False``. The CleanRL-style
rollout stores that call as an ordinary transition, so the batch carries one
junk row per episode: an action the env never executed, paired with the
previous episode's terminal observation.

These tests pin (a) that the autoreset step really behaves that way for a
vectorised ``FactorioEnv``, (b) that ``1 - dones`` — the flag the rollout
already stores — marks exactly those rows, and (c) that ``_masked_mean``
drops them from the loss reductions.
"""

import os
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv, _masked_mean  # noqa: E402

NUM_ENVS = 2
MAX_STEPS = 3


def _eot_action(num_envs, eot):
    """A batched no-op placement; ``eot`` declares end-of-turn per sub-env."""
    return {
        "xy": np.zeros((num_envs, 2), dtype=np.int64),
        "entity": np.zeros(num_envs, dtype=np.int64),
        "direction": np.zeros(num_envs, dtype=np.int64),
        "item": np.zeros(num_envs, dtype=np.int64),
        "misc": np.zeros(num_envs, dtype=np.int64),
        "eot": np.asarray(eot, dtype=np.int64),
    }


def _make_vec_env():
    return gym.vector.SyncVectorEnv([
        lambda i=i: FactorioEnv(size=5, max_steps=MAX_STEPS, idx=i)
        for i in range(NUM_ENVS)
    ])


def test_autoreset_step_ignores_the_action_and_pays_zero():
    """The step after a termination is junk: action dropped, reward forced 0."""
    envs = _make_vec_env()
    envs.reset(seed=0)

    # End both episodes via the eot action.
    _, _, terminations, _, _ = envs.step(_eot_action(NUM_ENVS, [1, 1]))
    assert terminations.all(), "eot should terminate every sub-env"

    # Next call: place a belt somewhere. Autoreset must swallow it.
    action = _eot_action(NUM_ENVS, [0, 0])
    action["entity"][:] = 1  # a real, placeable entity
    action["direction"][:] = 1
    obs, reward, terminations, truncations, _ = envs.step(action)

    assert np.all(reward == 0.0), f"autoreset step paid {reward}, expected 0"
    assert not terminations.any() and not truncations.any()
    # The action was never executed, so the fresh episode's world is untouched
    # by it: every sub-env is back at step 0.
    for env in envs.unwrapped.envs:
        assert env.steps == 0


def test_dones_flag_marks_exactly_the_junk_rows():
    """`dones_SE[step] = next_done` (stored pre-step) flags the junk steps."""
    envs = _make_vec_env()
    next_obs, _ = envs.reset(seed=0)
    next_done = np.zeros(NUM_ENVS, dtype=bool)

    num_steps = 8
    dones = np.zeros((num_steps, NUM_ENVS), dtype=bool)
    rewards = np.zeros((num_steps, NUM_ENVS), dtype=np.float64)

    rng = np.random.default_rng(0)
    for step in range(num_steps):
        dones[step] = next_done  # exactly what the PPO rollout stores
        action = _eot_action(NUM_ENVS, rng.integers(0, 2, size=NUM_ENVS))
        next_obs, reward, terminations, truncations, _ = envs.step(action)
        rewards[step] = reward
        next_done = np.logical_or(terminations, truncations)

    assert dones.any(), "no episode ended; the test is not exercising autoreset"
    # Every flagged row is a zero-reward junk row.
    assert np.all(rewards[dones] == 0.0)


def test_masked_mean_ignores_masked_rows():
    x = torch.tensor([1.0, 100.0, 3.0])
    valid = torch.tensor([1.0, 0.0, 1.0])
    assert _masked_mean(x, valid).item() == pytest.approx(2.0)


def test_masked_mean_of_all_masked_batch_is_zero_not_nan():
    x = torch.tensor([1.0, 2.0])
    valid = torch.zeros(2)
    out = _masked_mean(x, valid)
    assert torch.isfinite(out) and out.item() == pytest.approx(0.0)


def test_masked_mean_gradient_does_not_reach_masked_rows():
    """The junk row must contribute no gradient, not just no value."""
    x = torch.tensor([1.0, 5.0], requires_grad=True)
    _masked_mean(x, torch.tensor([1.0, 0.0])).backward()
    assert x.grad is not None
    assert x.grad[1].item() == pytest.approx(0.0)
