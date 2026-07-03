"""Guard against the "train on the same N factories forever" bug.

PPO used to reset every env to a fixed per-index seed each episode (and
Gymnasium's NEXT_STEP autoreset pinned it further to seed == idx), so the
policy only ever saw ~num_envs factories for the whole run. These tests assert
that both training paths keep drawing *fresh* factories:

  * PPO  — a seed counter handed to each env marches the factory seed forward,
           so the SyncVectorEnv never replays a factory across episodes.
  * SFT  — generate_dataset() marches its seed forward, so the dataset covers
           ~one distinct factory per generated lesson, scaling with num_samples.
"""

import hashlib
import os
import sys
from typing import cast

import numpy as np
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import FactorioEnv, make_env  # noqa: E402

ENV_ID = "factorion/FactorioEnv-v0-diversity-test"


def _factory_hash(env: FactorioEnv) -> str:
    """Hash the target (solved) factory the policy must reconstruct."""
    return hashlib.sha1(env._solved_world_CWH.numpy().tobytes()).hexdigest()


# ── PPO: marching seed counter ───────────────────────────────────────────────


class TestPPOFactoryDiversity:
    def test_seed_marches_and_never_repeats(self):
        env = FactorioEnv(size=7, idx=0)
        env._train_seed = 1000
        env._num_envs = 1  # single env -> +1 each reset sweeps contiguously
        seeds, hashes = [], []
        for _ in range(40):
            env.reset()
            seeds.append(env._seed)
            hashes.append(_factory_hash(env))
        # Seeds march forward every reset — no repeats at all.
        assert seeds == list(range(1000, 1040))
        # And the factories themselves are essentially all distinct (a couple of
        # collisions across 40 random layouts is fine; the bug gave exactly 1).
        assert len(set(hashes)) >= 38

    def test_without_counter_seed_is_deterministic_for_eval(self):
        # Eval/render rely on seed -> factory determinism; no counter set.
        a = FactorioEnv(size=7, idx=0)
        b = FactorioEnv(size=7, idx=0)
        a.reset(seed=4242)
        b.reset(seed=4242)
        assert a._seed == b._seed == 4242
        assert _factory_hash(a) == _factory_hash(b)

    def test_sync_vector_env_does_not_replay_factories(self):
        # The real training path: SyncVectorEnv + NEXT_STEP autoreset. Drive
        # several eot-terminated episodes and assert every env, every episode,
        # builds a never-before-seen factory.
        gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")
        n = 4
        envs = gym.vector.SyncVectorEnv(
            [make_env(ENV_ID, i, False, 7, "t") for i in range(n)]
        )
        for i, sub in enumerate(envs.envs):
            fe = cast(FactorioEnv, sub.unwrapped)
            fe._train_seed = 5000 + i
            fe._num_envs = n
        envs.reset(seed=5000, options={"num_missing_entities": float("inf")})

        def eot_action():
            return {
                "xy": np.zeros((n, 2), dtype=np.int64),
                "entity": np.zeros(n, dtype=np.int64),
                "direction": np.zeros(n, dtype=np.int64),
                "item": np.zeros(n, dtype=np.int64),
                "misc": np.zeros(n, dtype=np.int64),
                "eot": np.ones(n, dtype=np.int64),
            }

        seen = [cast(FactorioEnv, e.unwrapped)._seed for e in envs.envs]
        for _ in range(6):
            envs.step(eot_action())  # eot -> terminate
            envs.step(eot_action())  # next step -> NEXT_STEP autoreset fires
            seen.extend(cast(FactorioEnv, e.unwrapped)._seed for e in envs.envs)

        # 4 envs x (1 + 6) episodes = 28 factory draws, all distinct seeds.
        assert len(seen) == 28
        assert len(set(seen)) == 28


# ── SFT: the demo generator marches its seed ─────────────────────────────────


class TestSFTFactoryDiversity:
    def _distinct_factory_seeds(self, num_samples: int) -> int:
        from sft import _materialise

        out = _materialise(5, 25, 1, target=num_samples)
        seed_tensor = out[8]  # per-pair lesson seed; unique => one per factory
        return len(set(seed_tensor.tolist()))

    def test_dataset_covers_many_distinct_factories(self):
        # Each factory yields only a handful of (state, action) pairs, so a
        # few-hundred-pair dataset must span many distinct factories — not a
        # tiny fixed set.
        assert self._distinct_factory_seeds(200) >= 20

    def test_distinct_factories_scale_with_num_samples(self):
        # The decisive "not the same N forever" check: ask for more data, get
        # more distinct factories (the seed marches; it is not capped at N).
        small = self._distinct_factory_seeds(150)
        large = self._distinct_factory_seeds(450)
        assert large > small
