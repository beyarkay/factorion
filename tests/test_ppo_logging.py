"""Tests for the PPO wandb-logging support: run-name signature, the held-out
greedy-eval set, the lesson kind exposed in env info, and the per-head entropy
+ eot prob stashed by get_action_and_value (the policy/* metrics)."""

import os
import sys

import pytest
import torch
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import (  # noqa: E402
    Args,
    AgentCNN,
    FactorioEnv,
    make_env,
    _run_signature,
    _build_eval_set,
    _rollout_episode_metrics,
)
from factorion import LessonKind, build_factory  # noqa: E402
from helpers import Channel  # noqa: E402

NUM_CHANNELS = len(Channel)
ENV_ID = "factorion/FactorioEnv-v0-ppolog-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


# ── run-name signature ──────────────────────────────────────────────────────


class TestRunSignature:
    def test_encodes_core_hyperparams(self):
        sig = _run_signature(Args(size=11, learning_rate=5e-5, seed=3))
        assert sig.startswith("ppo-s11-")
        assert "lr5e-05" in sig
        assert sig.endswith("-seed3")
        # filename-safe: no path/colon/space chars
        assert not (set(sig) & set("/:\\ "))

    def test_includes_start_from_and_warmup(self):
        sig = _run_signature(Args(start_from="j0s5y2mc", critic_warmup=10))
        assert "fromj0s5y2mc" in sig
        assert "cw10" in sig

    def test_omits_optional_when_default(self):
        sig = _run_signature(Args(critic_warmup=0, start_from=None, target_kl=None))
        assert "cw" not in sig
        assert "from" not in sig
        assert "kl" not in sig


# ── greedy-eval held-out set ────────────────────────────────────────────────


class TestBuildEvalSet:
    def test_disjoint_seeds_all_kinds_buildable(self):
        args = Args(size=11, eval_seeds_per_kind=2, seed=1)
        s2k = _build_eval_set(args)
        # Unique seeds (dict keys), and every (seed, kind) actually builds.
        assert len(s2k) == len(set(s2k))
        for seed, kind_val in s2k.items():
            kind = LessonKind(kind_val)
            assert build_factory(size=args.size, kind=kind, seed=seed) is not None
        # Every lesson kind that can build at this size is represented.
        kinds_seen = {LessonKind(v) for v in s2k.values()}
        assert LessonKind.MOVE_ONE_ITEM in kinds_seen
        assert len(kinds_seen) >= 5

    def test_seeds_disjoint_from_training_range(self):
        # Training resets use seeds near args.seed (+ env idx); the eval pool
        # lives in a high range so eval factories are never trained on.
        args = Args(size=11, eval_seeds_per_kind=2, seed=1)
        assert min(_build_eval_set(args)) > 1_000_000


# ── lesson kind exposed in env info ─────────────────────────────────────────


class TestEnvExposesKind:
    def test_reset_and_step_report_kind(self, registered_env):
        env = FactorioEnv(size=8)
        _, info = env.reset(seed=2, options={"kind": LessonKind.SPLITTER_SPLIT})
        assert info["kind"] == LessonKind.SPLITTER_SPLIT.value
        action = {"xy": [0, 0], "entity": 0, "direction": 0, "item": 0,
                  "misc": 0, "eot": 0}
        _, _, _, _, info2 = env.step(action)
        assert info2["kind"] == LessonKind.SPLITTER_SPLIT.value


# ── per-head entropy + eot prob (policy/* metrics) ──────────────────────────


class TestPerHeadEntropyStash:
    def test_get_action_and_value_stashes_head_entropy_and_eot_prob(self, registered_env):
        envs = gym.vector.SyncVectorEnv(
            [make_env(ENV_ID, i, False, 5, "t") for i in range(2)]
        )
        agent = AgentCNN(envs, layers=(16, 16, 16))
        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        agent.get_action_and_value(obs)

        assert set(agent._last_head_entropy) == {
            "tile", "entity", "direction", "item", "misc", "eot"
        }
        for v in agent._last_head_entropy.values():
            assert float(v) >= 0.0  # entropy is non-negative
        # eot prob is a Bernoulli probability in [0, 1].
        assert 0.0 <= float(agent._last_eot_prob) <= 1.0

    def test_total_entropy_equals_sum_of_heads(self, registered_env):
        envs = gym.vector.SyncVectorEnv(
            [make_env(ENV_ID, i, False, 5, "t") for i in range(2)]
        )
        agent = AgentCNN(envs, layers=(16, 16, 16))
        obs = torch.randn(3, NUM_CHANNELS, 5, 5)
        _, _, entropy_B, _ = agent.get_action_and_value(obs)
        head_sum = sum(float(v) for v in agent._last_head_entropy.values())
        # entropy_B is per-sample (summed over heads); its mean should match the
        # sum of the per-head means stashed for logging.
        assert entropy_B.mean().detach().item() == pytest.approx(head_sum, abs=1e-4)


# ── per-episode rollout metrics (overall + per-lesson, raw + normalized) ──────


class TestRolloutEpisodeMetrics:
    def _metrics(self, lesson="MOVE_ONE_ITEM"):
        return _rollout_episode_metrics(
            lesson,
            episode_return=2.5,
            episode_len=42.0,
            thput_normed=0.6,
            thput_raw=9.0,
            ended_by_eot=1.0,
            invalid_frac=0.1,
            num_entities=4.0,
            min_entities_required=3.0,
            frac_reachable=0.75,
        )

    def test_overall_logs_raw_and_normed_throughput(self):
        m = self._metrics()
        assert m["rollout/throughput"] == pytest.approx(0.6)
        assert m["rollout/thput_raw"] == pytest.approx(9.0)

    def test_per_lesson_logs_raw_throughput(self):
        # The ASSERT: raw items/s must be logged per lesson, not just overall.
        m = self._metrics("SPLITTER_SPLIT")
        assert m["rollout/SPLITTER_SPLIT/thput_raw"] == pytest.approx(9.0)
        assert m["rollout/SPLITTER_SPLIT/throughput"] == pytest.approx(0.6)

    def test_every_lesson_kind_gets_a_per_lesson_raw_key(self):
        for kind in LessonKind:
            m = _rollout_episode_metrics(
                kind.name,
                episode_return=0.0,
                episode_len=1.0,
                thput_normed=0.0,
                thput_raw=1.23,
                ended_by_eot=0.0,
                invalid_frac=0.0,
                num_entities=1.0,
                min_entities_required=1.0,
                frac_reachable=0.0,
            )
            assert m[f"rollout/{kind.name}/thput_raw"] == pytest.approx(1.23)

    def test_entity_efficiency_is_required_over_placed(self):
        m = self._metrics()
        assert m["rollout/entity_efficiency"] == pytest.approx(3.0 / 4.0)
