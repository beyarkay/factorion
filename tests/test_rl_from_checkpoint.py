"""End-to-end tests for the RL-from-SFT-checkpoint setup.

Covers the four behaviours that make `ppo.py --start_from <sft-ckpt>` a
sensible RL fine-tuning run rather than RL-from-scratch:

1. throughput-dominant reward (solution-matching shaping off by default),
2. full-blank build-from-empty task by default (num_missing_entities=inf),
3. end-of-turn as a trained Bernoulli *action* that ends the episode,
4. the critic warm-up actor/critic param split + freeze,
5. the closed-form per-head KL to the frozen SFT reference policy.
"""

import copy
import os
import sys
from typing import cast

import numpy as np
import pytest
import torch
import gymnasium as gym

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ppo import (  # noqa: E402
    AgentCNN, PpoArgs, FactorioEnv, make_env, layer_init, _WARMUP_KL_TOL,
    _KL_REF_PENALIZED_HEADS, _bernoulli_kl, _categorical_kl,
)
from helpers import Channel  # noqa: E402
from factorion import LessonKind  # noqa: E402

NUM_CHANNELS = len(Channel)
ENV_ID = "factorion/FactorioEnv-v0-rlckpt-test"


@pytest.fixture(scope="module")
def registered_env():
    gym.register(id=ENV_ID, entry_point="ppo:FactorioEnv")


@pytest.fixture()
def envs(registered_env):
    return gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, False, 5, "test") for i in range(2)]
    )


@pytest.fixture()
def agent(envs):
    return AgentCNN(envs, layers=(16, 16, 16))


class TestFullBlankDefault:
    def test_reset_without_option_fully_blanks(self, registered_env):
        """Omitting num_missing_entities blanks the whole factory (inf),
        identical to passing inf explicitly, and removes ≥1 entity. The kind
        is pinned to a scripted lesson: a randomly-sampled kind could be a
        trial, which has no entities to blank."""
        env = FactorioEnv(size=8)
        kind = LessonKind.MOVE_ONE_ITEM
        env.reset(seed=3, options={"kind": kind, "num_missing_entities": float("inf")})
        removed_inf = env.min_entities_required
        env.reset(seed=3, options={"kind": kind})  # no num_missing_entities -> default inf
        removed_default = env.min_entities_required
        assert removed_default == removed_inf
        assert removed_default > 0

    def test_default_blank_removes_more_than_partial(self, registered_env):
        """The default (full) blank removes strictly more entity units than a
        1-unit blank. Pin the lesson kind so both resets build the identical
        factory (env.reset reseeds build_factory, but blank_entities draws
        from the global RNG, so counts are the stable invariant to assert)."""
        env = FactorioEnv(size=8)
        env.reset(seed=7, options={"kind": LessonKind.MOVE_ONE_ITEM, "num_missing_entities": 1})
        removed_partial = env.min_entities_required
        env.reset(seed=7, options={"kind": LessonKind.MOVE_ONE_ITEM})  # default full blank
        removed_full = env.min_entities_required
        assert removed_full > removed_partial

    def test_make_env_max_steps_is_size_squared(self, registered_env):
        """A from-empty build needs ~size*size placements, so make_env sizes
        max_steps to size*size (not the old 2*size)."""
        env = make_env(ENV_ID, 0, False, 7, "t")()
        assert env.unwrapped.max_steps == 49


class TestEotAction:
    def test_eot_is_part_of_the_action(self, agent):
        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        action, logp, entropy, value = agent.get_action_and_value(obs)
        assert "eot" in action
        assert action["eot"].shape == (4,)
        assert set(action["eot"].unique().tolist()) <= {0.0, 1.0}
        # eot's log-prob/entropy fold into the joint action distribution.
        assert logp.shape == (4,)
        assert entropy.shape == (4,)

    def test_eot_logprob_roundtrips(self, agent):
        """Recomputing log-prob from a stored 7-dim action (incl. eot at
        index 6) must match the sampled log-prob, so PPO's importance ratio
        is well-defined and the eot head actually gets trained."""
        torch.manual_seed(0)
        obs = torch.randn(3, NUM_CHANNELS, 5, 5)
        action, logp, _, _ = agent.get_action_and_value(obs)
        x_B, y_B = action["xy"].unbind(dim=1)
        action_BA = torch.stack(
            [
                x_B, y_B,
                action["entity"], action["direction"],
                action["item"], action["misc"],
                action["eot"].long(),
            ],
            dim=1,
        )
        assert action_BA.shape == (3, 7)
        _, logp_recomputed, _, _ = agent.get_action_and_value(obs, action_BA)
        torch.testing.assert_close(logp, logp_recomputed)


class TestEotTerminationAndMetrics:
    def _action(self, eot):
        return {
            "xy": np.array([0, 0]),
            "entity": 0, "direction": 0, "item": 0, "misc": 0,
            "eot": int(eot),
        }

    def test_eot_terminates_and_records_episode(self, registered_env):
        """eot=1 must terminate via the env so the RecordEpisodeStatistics
        wrapper (added by make_env) emits "episode" — i.e. eot-ended episodes
        are NOT missing from the metric averages."""
        env = make_env(ENV_ID, 0, False, 5, "t")()
        env.reset(seed=1, options={"num_missing_entities": 99})
        _, _, terminated, truncated, info = env.step(self._action(eot=1))
        assert terminated and not truncated
        assert "episode" in info, "eot-terminated episode must emit RecordEpisodeStatistics info"

    def test_no_eot_keeps_running(self, registered_env):
        """Without eot (and far from solved), the episode keeps going — no
        spurious termination, no episode stats yet."""
        env = make_env(ENV_ID, 0, False, 5, "t")()
        env.reset(seed=1, options={"num_missing_entities": 99})
        _, _, terminated, truncated, info = env.step(self._action(eot=0))
        assert not terminated and not truncated
        assert "episode" not in info

    def test_entity_cost_scale_default(self):
        a = PpoArgs()
        assert a.entity_cost_scale == 0.001


class TestCriticWarmupParamSplit:
    def _split(self, agent):
        critic_params = list(agent.critic_head.parameters())
        critic_ids = {id(p) for p in critic_params}
        actor_params = [p for p in agent.parameters() if id(p) not in critic_ids]
        return actor_params, critic_params

    def test_split_is_a_partition(self, agent):
        actor_params, critic_params = self._split(agent)
        assert len(actor_params) + len(critic_params) == len(list(agent.parameters()))
        assert len(critic_params) > 0 and len(actor_params) > 0

    def test_encoder_and_policy_heads_are_actor(self, agent):
        """The shared encoder and every policy/eot head count as the actor;
        only the value head is the critic. Freezing the encoder too is what
        keeps the policy genuinely fixed during warm-up."""
        actor_params, _ = self._split(agent)
        actor_ids = {id(p) for p in actor_params}
        for module in (agent.encoder, agent.eot_head, agent.tile_logits,
                       agent.ent_head, agent.dir_head):
            assert all(id(p) in actor_ids for p in module.parameters())

    def test_frozen_actor_trains_only_critic(self, agent):
        """With the actor frozen, a value-loss backward populates grads on
        the critic head alone — the warm-up's intended behaviour."""
        actor_params, critic_params = self._split(agent)
        for p in actor_params:
            p.requires_grad_(False)

        obs = torch.randn(4, NUM_CHANNELS, 5, 5)
        value = agent.get_value(obs)
        (value ** 2).mean().backward()

        assert all(p.grad is not None for p in critic_params)
        assert all(p.grad is None for p in actor_params)


class TestCriticHeadStd:
    def test_default_is_positive(self):
        assert PpoArgs().critic_head_std > 0

    def test_construction_scales_value_head_magnitude(self, envs):
        """A smaller critic_head_std yields a smaller-magnitude value head; the
        orthogonal (1, N) init makes the row's L2 norm equal to the std."""
        small = AgentCNN(envs, layers=(16, 16, 16), critic_head_std=0.01)
        big = AgentCNN(envs, layers=(16, 16, 16), critic_head_std=1.0)
        small_weight = cast(torch.Tensor, small.critic_head.weight)
        big_weight = cast(torch.Tensor, big.critic_head.weight)
        assert small_weight.norm().item() < big_weight.norm().item()
        assert small_weight.norm().item() == pytest.approx(0.01, abs=1e-3)
        assert big_weight.norm().item() == pytest.approx(1.0, abs=1e-2)

    def test_post_load_reinit_replaces_weights_in_place_at_new_std(self, agent):
        """The post-load re-init (layer_init on the value head) rewrites it in
        place to the requested std, so a checkpoint's untrained critic doesn't
        clobber --critic-head-std."""
        head_weight = agent.critic_head.weight
        before = head_weight.detach().clone()
        layer_init(agent.critic_head, std=0.02)
        assert head_weight is agent.critic_head.weight  # in place
        assert not torch.equal(before, head_weight)
        assert head_weight.norm().item() == pytest.approx(0.02, abs=1e-3)



# The policy forward must be a deterministic function of (weights, obs, action):
# PPO's importance ratio divides the update's recomputed log-prob by the
# rollout's, so any stochasticity corrupts every ratio, every clipfrac and
# approx_kl — even with the actor frozen. Regression guards for run gs1nqoni,
# where `nn.TransformerEncoderLayer`'s default dropout=0.1 stayed active (it is
# not reached by `--dropout`), giving approx_kl 0.27-0.36 during critic warm-up
# and tripping target_kl after epoch 1 of 8 on every iteration.
#
# Every forward below is hoisted into a module-scoped fixture: the assertions
# are one-liners but a forward is the expensive part, so each is done once.

_ATTN_KW = dict(layers=(16, 16, 16), attn_dim=32, attn_heads=4, attn_layers=1)


def _finetune_agent(envs):
    """An agent whose attention stage contributes at trained-checkpoint scale.

    Anything stochastic inside the attention stage reaches the outputs in
    proportion to the out-projection's magnitude, so the guards below only
    have teeth at the scale a `--start-from` checkpoint has trained to. The
    std is picked so the test arch (1 attention layer, dim 32) reproduces the
    artifact magnitude the production arch showed (approx_kl ~0.3), which is
    what makes the clipfrac and target_kl guards sensitive too.
    """
    agent = AgentCNN(envs, **_ATTN_KW)
    with torch.no_grad():
        torch.nn.init.normal_(agent.attn.out_proj.weight, std=0.15)
    return agent


def _pack_action(action):
    """The rollout's storage layout: [x, y, entity, direction, item, misc, eot]."""
    x_B, y_B = action["xy"].unbind(dim=1)
    return torch.stack([x_B, y_B, action["entity"], action["direction"],
                        action["item"], action["misc"],
                        action["eot"].long()], dim=1)


def _test_envs(registered_env, num_envs=2, size=5):
    return gym.vector.SyncVectorEnv(
        [make_env(ENV_ID, i, False, size, "t") for i in range(num_envs)]
    )


@pytest.fixture(scope="module")
def forward_probe(registered_env):
    """Sample an action, then recompute its log-prob/value four ways: twice in
    train mode, once in eval mode, and the sampled value itself."""
    envs = _test_envs(registered_env)
    agent = _finetune_agent(envs)
    torch.manual_seed(0)
    obs = torch.randn(8, NUM_CHANNELS, 5, 5)

    action, logp_sampled, _, _ = agent.get_action_and_value(obs)
    stored = _pack_action(action)

    agent.train()
    _, logp_1, _, value_1 = agent.get_action_and_value(obs, stored)
    _, logp_2, _, value_2 = agent.get_action_and_value(obs, stored)
    eot_train = agent.eot_prob(obs)
    agent.eval()
    _, logp_eval, _, value_eval = agent.get_action_and_value(obs, stored)
    eot_eval = agent.eot_prob(obs)
    agent.train()

    envs.close()
    return {
        "agent": agent,
        "logp_sampled": logp_sampled, "logp_1": logp_1, "logp_2": logp_2,
        "value_1": value_1, "value_2": value_2,
        "logp_eval": logp_eval, "value_eval": value_eval,
        "eot_train": eot_train, "eot_eval": eot_eval,
    }


class TestDeterministicPolicyForward:
    def test_no_dropout_is_active_at_dropout_zero(self, forward_probe):
        """`--dropout 0` must mean the whole network is deterministic, the
        attention stage included."""
        agent = forward_probe["agent"]
        assert agent.training, "nn.Module defaults to train mode; PPO never flips it"
        active = [
            name for name, m in agent.named_modules()
            if (isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d,
                               torch.nn.Dropout2d, torch.nn.AlphaDropout))
                and m.p > 0)
            or (isinstance(m, torch.nn.MultiheadAttention) and m.dropout > 0)
        ]
        assert active == [], f"dropout active in train mode at dropout=0: {active}"

    def test_replay_logprob_matches_sampled(self, forward_probe):
        """The log-prob round-trip through the rollout's storage layout, on a
        trained-checkpoint agent in train mode — exactly how PPO runs. This is
        the ratio PPO divides by; if it is wrong, everything downstream is."""
        torch.testing.assert_close(
            forward_probe["logp_sampled"], forward_probe["logp_1"]
        )

    def test_repeated_calls_agree(self, forward_probe):
        """Same weights, same obs, same action -> same log-prob and value.

        The `critic/*` symptom: if the encoder features are stochastic then
        V(s) is not a function of s, so the warm-up regresses the value head on
        noise and `critic/explained_variance` cannot climb however long it runs.
        """
        torch.testing.assert_close(forward_probe["logp_1"], forward_probe["logp_2"])
        torch.testing.assert_close(forward_probe["value_1"], forward_probe["value_2"])

    def test_train_and_eval_mode_agree(self, forward_probe):
        """train() and eval() must compute the same function.

        The `eval/thput` vs `rollout/thput` symptom: `run_rollout_eval` calls
        `agent.eval()` while the PPO rollout runs in train mode, so a mode-
        dependent forward shows up as an unexplained gap between the two at
        iteration 1 — before a single gradient step, on identical weights. A
        train/eval gap at step 0 is always a bug, never a property of the model.
        """
        torch.testing.assert_close(forward_probe["logp_1"], forward_probe["logp_eval"])
        torch.testing.assert_close(forward_probe["value_1"], forward_probe["value_eval"])
        torch.testing.assert_close(forward_probe["eot_train"], forward_probe["eot_eval"])


@pytest.fixture(scope="module")
def warmup_iteration(registered_env):
    """One critic-warm-up iteration end to end, mirroring ppo.py's bookkeeping.

    Rolls out a real vector env into the storage layout the loop uses (int64
    packed actions, float32 log-probs, (steps, envs, ...) reshaped steps-major),
    then runs the warm-up's v_loss-only update over shuffled minibatches with
    the actor frozen. Returns the per-minibatch approx_kl and clipfrac series,
    in order, so the assertions can look at both the first update and the worst.
    """
    num_envs, num_steps, size, mb, epochs = 2, 4, 5, 4, 2
    envs = _test_envs(registered_env, num_envs, size)
    agent = _finetune_agent(envs)

    critic_ids = {id(p) for p in agent.critic_head.parameters()}
    for p in agent.parameters():
        if id(p) not in critic_ids:
            p.requires_grad_(False)
    optimizer = torch.optim.Adam(agent.critic_head.parameters(), lr=1e-3)

    obs_buf = torch.zeros((num_steps, num_envs, NUM_CHANNELS, size, size))
    act_buf = torch.zeros((num_steps, num_envs, 7), dtype=torch.int64)
    logp_buf = torch.zeros((num_steps, num_envs))
    returns_buf = torch.zeros((num_steps, num_envs))

    torch.manual_seed(0)
    np.random.seed(0)
    next_obs, _ = envs.reset(seed=0)
    next_obs = torch.as_tensor(np.array(next_obs), dtype=torch.float32)
    for step in range(num_steps):
        obs_buf[step] = next_obs
        with torch.no_grad():
            action, logp, _, _ = agent.get_action_and_value(next_obs)
        act_buf[step] = _pack_action(action)
        logp_buf[step] = logp
        act_np = act_buf[step].numpy()
        next_obs, reward, _term, _trunc, _info = envs.step({
            "xy": act_np[:, 0:2], "entity": act_np[:, 2],
            "direction": act_np[:, 3], "item": act_np[:, 4],
            "misc": act_np[:, 5], "eot": act_np[:, 6],
        })
        next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
        returns_buf[step] = torch.as_tensor(reward, dtype=torch.float32)

    obs_B = obs_buf.reshape(-1, NUM_CHANNELS, size, size)
    act_B = act_buf.reshape(-1, 7)
    logp_B = logp_buf.reshape(-1)
    returns_B = returns_buf.reshape(-1)

    clip_coef = PpoArgs().clip_coef
    idxs_B = np.arange(obs_B.shape[0])
    kls, clipfracs = [], []
    for _epoch in range(epochs):
        np.random.shuffle(idxs_B)
        for start in range(0, len(idxs_B), mb):
            idxs = idxs_B[start:start + mb]
            _a, newlogp, _e, newvalue = agent.get_action_and_value(
                obs_B[idxs], act_B[idxs]
            )
            logratio = newlogp.reshape(-1) - logp_B[idxs]
            ratio = logratio.exp()
            kls.append(float(((ratio - 1) - logratio).mean()))
            clipfracs.append(float(((ratio - 1.0).abs() > clip_coef).float().mean()))
            optimizer.zero_grad(set_to_none=True)
            (0.5 * ((newvalue.view(-1) - returns_B[idxs]) ** 2).mean()).backward()
            optimizer.step()

    envs.close()
    return {"kls": kls, "clipfracs": clipfracs}


class TestWarmupIterationKl:
    """The `losses/*` panels that were carrying the signal in run gs1nqoni,
    asserted on a real warm-up iteration — one test per panel."""

    def test_first_update_is_exactly_on_policy(self, warmup_iteration):
        """The first minibatch of the first epoch runs on weights bit-identical
        to the acting ones (zero optimiser steps so far), so approx_kl and
        clipfrac are pinned to 0 there — warm-up or not. The leftmost point of
        those two curves is the cheapest correctness check a PPO run has.
        """
        assert warmup_iteration["kls"][0] < _WARMUP_KL_TOL
        assert warmup_iteration["clipfracs"][0] == 0.0

    def test_frozen_actor_keeps_kl_at_the_noise_floor(self, warmup_iteration):
        """With the actor frozen the recomputed log-probs equal the acting ones,
        so approx_kl stays at FP noise for the whole warm-up — the invariant
        gs1nqoni violated at 0.27-0.36, ~700x the ~5e-4 FP budget."""
        kls = warmup_iteration["kls"]
        assert max(kls) < _WARMUP_KL_TOL, (
            f"frozen actor produced approx_kl up to {max(kls):.4g} "
            f"(tol {_WARMUP_KL_TOL}); the policy forward is not deterministic"
        )

    def test_clipping_never_engages_while_frozen(self, warmup_iteration):
        """A frozen actor cannot move outside the clip range, so
        `losses/clipfrac` must be flat 0 across the whole warm-up."""
        assert max(warmup_iteration["clipfracs"]) == 0.0

    def test_target_kl_does_not_trip(self, warmup_iteration):
        """`target_kl` must measure policy movement, not a recompute artifact:
        the epoch loop cannot break while the actor is frozen. When it breaks at
        epoch 1 every iteration you are silently running 1 of `--update-epochs`
        epochs, visible as a pinned, constant `perf/update_seconds`.
        """
        target_kl = PpoArgs().target_kl
        assert target_kl is not None
        assert max(warmup_iteration["kls"]) < target_kl, (
            "target_kl would break the epoch loop with the actor frozen"
        )


class TestTrainingLegalMask:
    """Training-time sampling is restricted to legal tiles, matching the
    masked distribution eval/inference always used — invalid placements are
    unrepresentable rather than cheap no-ops the entropy bonus subsidises."""

    def test_sampled_tiles_are_legal(self, agent, registered_env):
        env = FactorioEnv(size=5)
        obs, _ = env.reset(seed=5, options={"kind": LessonKind.MOVE_ONE_ITEM})
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        obs_B = obs_t.unsqueeze(0).expand(64, -1, -1, -1)
        action, logp, _, _ = agent.get_action_and_value(obs_B)
        ent = obs_t[Channel.ENTITIES.value]
        foot = obs_t[Channel.FOOTPRINT.value]
        for x, y in action["xy"].tolist():
            assert ent[x, y] == 0, f"sampled occupied tile ({x},{y})"
            assert foot[x, y] != 0, f"sampled unbuildable tile ({x},{y})"
        assert torch.isfinite(logp).all()

    def test_full_grid_does_not_nan(self, agent):
        obs = torch.zeros(2, NUM_CHANNELS, 5, 5)
        obs[:, Channel.ENTITIES.value] = 1.0  # every tile occupied
        obs[:, Channel.FOOTPRINT.value] = 1.0
        _, logp, entropy, _ = agent.get_action_and_value(obs)
        assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()


def _flat_heads(agent, obs, stored):
    """The per-head distributions the KL block consumes, replayed at the
    stored actions — the exact tensors the PPO loop hands it."""
    out = agent.sample_action(obs, action=stored, compute_value=False)
    return {**out["logp_heads"], "eot_logit": out["eot_logit"]}


def _kl_heads(cur, ref):
    """The PPO loop's per-head KL block, replicated for the tests."""
    kls = {h: _categorical_kl(cur[h], ref[h]) for h in _KL_REF_PENALIZED_HEADS}
    kls["eot"] = _bernoulli_kl(cur["eot_logit"], ref["eot_logit"])
    return kls


class TestKlToRef:
    """The KL(π_θ ‖ π_ref) anchor to the frozen SFT reference."""

    def test_categorical_kl_matches_torch_distributions(self):
        """On rows sharing a -inf mask (how every masked head reaches it), the
        fused KL equals torch.distributions' — including zero-mass entries."""
        torch.manual_seed(0)
        a, b = torch.randn(2, 16, 9, dtype=torch.float64)
        mask = torch.rand(16, 9) < 0.4
        mask[:, 0] = False  # keep every row a valid distribution
        a = a.masked_fill(mask, float("-inf"))
        b = b.masked_fill(mask, float("-inf"))
        logp = torch.log_softmax(a, dim=-1)
        logq = torch.log_softmax(b, dim=-1)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Categorical(logits=a),
            torch.distributions.Categorical(logits=b),
        )
        torch.testing.assert_close(_categorical_kl(logp, logq), expected)
        assert (_categorical_kl(logp, logp) == 0).all()

    def test_bernoulli_kl_matches_torch_distributions(self):
        torch.manual_seed(0)
        z, zq = torch.randn(2, 32, dtype=torch.float64) * 4
        expected = torch.distributions.kl_divergence(
            torch.distributions.Bernoulli(logits=z),
            torch.distributions.Bernoulli(logits=zq),
        )
        torch.testing.assert_close(_bernoulli_kl(z, zq), expected)
        assert (_bernoulli_kl(z, z) == 0).all()

    @pytest.fixture()
    def replay(self, agent):
        """A real partially-occupied obs (so the tile head carries -inf mask
        entries) with actions sampled and packed the way the rollout stores them."""
        env = FactorioEnv(size=5)
        obs_np, _ = env.reset(seed=5, options={"kind": LessonKind.MOVE_ONE_ITEM})
        obs = torch.as_tensor(np.asarray(obs_np), dtype=torch.float32)
        obs = obs.unsqueeze(0).expand(8, -1, -1, -1)
        torch.manual_seed(0)
        action, _, _, _ = agent.get_action_and_value(obs)
        return agent, copy.deepcopy(agent), obs, _pack_action(action)

    def test_kl_to_self_is_zero(self, replay):
        """An unmoved policy has zero drift on every head — the value the
        metric must report at the start of a --start-from run."""
        agent, ref, obs, stored = replay
        kls = _kl_heads(_flat_heads(agent, obs, stored), _flat_heads(ref, obs, stored))
        assert set(kls) == set(_KL_REF_PENALIZED_HEADS) | {"eot"}
        for h, kl_B in kls.items():
            assert torch.isfinite(kl_B).all(), f"{h} KL not finite"
            torch.testing.assert_close(kl_B, torch.zeros_like(kl_B))

    def test_perturbed_head_shows_only_in_its_own_kl(self, replay):
        """Replaying the stored actions conditions every head identically on
        both sides, so nudging one head's weights moves exactly that head's KL."""
        agent, ref, obs, stored = replay
        with torch.no_grad():
            ref.ent_head.weight.add_(torch.randn_like(ref.ent_head.weight) * 0.1)
        kls = _kl_heads(_flat_heads(agent, obs, stored), _flat_heads(ref, obs, stored))
        assert kls["entity"].sum() > 0
        for h in ("tile", "direction", "item", "misc", "eot"):
            torch.testing.assert_close(kls[h], torch.zeros_like(kls[h]))

    def test_penalty_gradient_reaches_the_policy(self, replay):
        """β·KL must be a trainable loss term: its backward populates the
        drifted head's grads while the frozen reference stays untouched."""
        agent, ref, obs, stored = replay
        with torch.no_grad():
            agent.ent_head.weight.add_(torch.randn_like(agent.ent_head.weight) * 0.1)
        ref.requires_grad_(False)
        with torch.no_grad():
            ref_heads = _flat_heads(ref, obs, stored)
        kls = _kl_heads(_flat_heads(agent, obs, stored), ref_heads)
        sum(kls[h].mean() for h in _KL_REF_PENALIZED_HEADS).backward()
        assert agent.ent_head.weight.grad is not None
        assert agent.ent_head.weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in ref.parameters())
