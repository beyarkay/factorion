"""Self-imitation (#358): archive, episode tracker, and loss gating."""

import os
import sys

import numpy as np
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorion import LessonKind  # noqa: E402
from ppo import SilArchive, SilEpisodeTracker, _sil_losses  # noqa: E402

OBS_SHAPE = (5, 3, 3)
TRIAL = LessonKind.TRIAL_RECIPE_TREE_DEPTH_1.value
LESSON = LessonKind.MOVE_ONE_ITEM.value


def _episode(n, fill):
    obs = [np.full(OBS_SHAPE, fill + t, dtype=np.float32) for t in range(n)]
    act = [np.full(7, t, dtype=np.int64) for t in range(n)]
    return obs, act


class TestSilArchive:
    def test_returns_are_terminal_reward_discounted_backwards(self):
        arch = SilArchive(capacity=10, obs_shape=OBS_SHAPE)
        obs, act = _episode(3, fill=0)
        arch.add_episode(obs, act, terminal_reward=2.0, gamma=0.5)
        assert len(arch) == 3
        # gamma^(T-1-t) * r: the terminal step keeps the full reward.
        _, _, ret_np = arch.sample(256, np.random.default_rng(0))
        assert set(np.round(ret_np, 4)) == {0.5, 1.0, 2.0}

    def test_eviction_removes_the_lowest_return_episode(self):
        """Elite semantics: capacity pressure evicts by QUALITY, not age — a
        stale high-return episode outlives a fresh mediocre one (#376)."""
        arch = SilArchive(capacity=4, obs_shape=OBS_SHAPE)
        arch.add_episode(*_episode(3, fill=0), terminal_reward=5.0, gamma=1.0)
        arch.add_episode(*_episode(3, fill=100), terminal_reward=1.0, gamma=1.0)
        assert len(arch) == 3, "the newer but weaker episode must be evicted"
        obs_np, act_np, ret_np = arch.sample(64, np.random.default_rng(0))
        assert obs_np.shape == (64, *OBS_SHAPE)
        assert act_np.shape == (64, 7)
        assert set(np.round(ret_np, 4)) == {5.0}
        assert not (obs_np == 100).any()

    def test_below_floor_episode_is_rejected_when_full(self):
        arch = SilArchive(capacity=6, obs_shape=OBS_SHAPE)
        arch.add_episode(*_episode(3, fill=0), terminal_reward=3.0, gamma=1.0)
        arch.add_episode(*_episode(3, fill=1), terminal_reward=4.0, gamma=1.0)
        arch.add_episode(*_episode(3, fill=2), terminal_reward=1.0, gamma=1.0)
        _, _, ret_np = arch.sample(64, np.random.default_rng(0))
        assert set(np.round(ret_np, 4)) == {3.0, 4.0}


class TestSilEpisodeTracker:
    def _step(self, tracker, reward, done, kind, fill=0.0):
        obs = np.full((1, *OBS_SHAPE), fill, dtype=np.float32)
        act = np.zeros((1, 7), dtype=np.int64)
        tracker.step(obs, act, np.array([reward]), np.array([done]),
                     np.array([kind]))

    def test_successful_trial_is_archived_with_full_episode(self):
        arch = SilArchive(capacity=100, obs_shape=OBS_SHAPE)
        tracker = SilEpisodeTracker(1, arch, gamma=1.0)
        self._step(tracker, 0.0, False, TRIAL)
        self._step(tracker, 0.0, False, TRIAL)
        self._step(tracker, 3.0, True, TRIAL)
        assert tracker.episodes_stored == 1
        assert len(arch) == 3

    def test_zero_reward_trial_and_lesson_success_are_not_archived(self):
        arch = SilArchive(capacity=100, obs_shape=OBS_SHAPE)
        tracker = SilEpisodeTracker(1, arch, gamma=1.0)
        self._step(tracker, 0.0, True, TRIAL)     # trial, no delivery
        self._step(tracker, 0.0, False, LESSON)   # autoreset junk (skipped)
        self._step(tracker, 7.0, True, LESSON)    # lesson success: not SIL's job
        assert tracker.episodes_stored == 0
        assert len(arch) == 0

    def test_autoreset_junk_step_is_not_recorded(self):
        """The step after a done is Gymnasium's swallowed-action reset step
        (#233); it must not become the next episode's first transition."""
        arch = SilArchive(capacity=100, obs_shape=OBS_SHAPE)
        tracker = SilEpisodeTracker(1, arch, gamma=1.0)
        self._step(tracker, 1.0, True, TRIAL, fill=1)   # episode 1 (archived)
        self._step(tracker, 0.0, False, TRIAL, fill=99)  # junk: skipped
        self._step(tracker, 0.0, False, TRIAL, fill=2)
        self._step(tracker, 2.0, True, TRIAL, fill=3)    # episode 2 (archived)
        assert tracker.episodes_stored == 2
        assert len(arch) == 3, "episode 2 has 2 steps; the junk step is absent"
        obs_np, _, _ = arch.sample(256, np.random.default_rng(0))
        assert not (obs_np == 99).any()


class TestSilLosses:
    def test_only_returns_above_value_produce_gradient(self):
        logp = torch.tensor([-1.0, -1.0], requires_grad=True)
        value = torch.tensor([2.0, 0.0])
        returns = torch.tensor([1.0, 3.0])  # below V, above V
        p_loss, v_loss, frac_pos = _sil_losses(logp, value, returns)
        # Only the second transition carries advantage (3 - 0 = 3).
        assert p_loss.item() == pytest.approx(-(-1.0 * 3.0) / 2)
        assert v_loss.item() == pytest.approx(0.5 * 9.0 / 2)
        assert frac_pos.item() == pytest.approx(0.5)

    def test_value_at_or_above_return_is_a_no_op(self):
        logp = torch.tensor([-1.0], requires_grad=True)
        p_loss, v_loss, frac_pos = _sil_losses(
            logp, torch.tensor([5.0]), torch.tensor([5.0])
        )
        assert p_loss.item() == 0.0 and v_loss.item() == 0.0
        p_loss.backward()
        assert logp.grad is not None and float(logp.grad.abs().sum()) == 0.0
