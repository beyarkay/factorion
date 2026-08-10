"""Tests for the GRPO RL finetuning pipeline."""

import json
import os
import random
import sys

import gymnasium as gym
import numpy as np
import pytest
import torch

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from factorion import Channel  # noqa: E402
from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402
from grpo import (  # noqa: E402
    GRPOArgs,
    RolloutBatch,
    _select_eval_grids,
    broadcast_advantages,
    collect_rollout,
    compute_advantages,
    compute_diversity,
    compute_rewards,
    greedy_eval,
    grpo_update,
    policy_step,
    select_grids,
    train_grpo,
)

ENV_ID = "factorion/FactorioEnv-v0-grpo-test"

# Tiny network so tests stay fast; this must match across the fake SFT
# checkpoint and the GRPOArgs that loads it. Passed explicitly (not splatted)
# so the annotated GRPOArgs type-checks cleanly under `ty`.
TINY_LAYERS = (16, 16)


@pytest.fixture(scope="module")
def registered_env():
    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point=FactorioEnv)  # ty: ignore[invalid-argument-type]


def _make_sft_checkpoint(path: str, size: int) -> None:
    """Stand in for a real SFT checkpoint: a freshly-initialised AgentCNN
    state_dict saved exactly the way sft.train_sft writes it."""
    envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "grpo-test")])
    agent = AgentCNN(envs, layers=TINY_LAYERS)
    envs.close()
    torch.save(agent.state_dict(), path)


def _tiny_agent(size=5):
    envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "grpo-test")])
    agent = AgentCNN(envs, layers=TINY_LAYERS)
    envs.close()
    return agent


def _dummy_action(obs):
    B = obs.shape[0]
    return torch.zeros((B, 7), dtype=torch.long)


class TestTrainGRPOEndToEnd:
    def test_requires_start_from(self, registered_env):
        """GRPO finetunes a prior — refuse to run without --start-from."""
        with pytest.raises(ValueError):
            train_grpo(GRPOArgs(seed=1, size=5, start_from="", layers=TINY_LAYERS))

    def test_produces_checkpoint_and_summary(self, registered_env, tmp_path):
        """End-to-end: train_grpo() runs, saves a checkpoint + summary, and
        the checkpoint reloads into a fresh AgentCNN."""
        size = 5
        sft_ckpt = str(tmp_path / "sft.pt")
        _make_sft_checkpoint(sft_ckpt, size)

        out_ckpt = str(tmp_path / "grpo.pt")
        summary = str(tmp_path / "grpo_summary.json")
        args = GRPOArgs(
            seed=1,
            size=size,
            start_from=sft_ckpt,
            num_iterations=2,
            num_grids=2,
            group_size=2,
            num_missing_max=2,
            eval_max_seeds=4,
            n_held_out_seeds=4,
            checkpoint_path=out_ckpt,
            summary_path=summary,
            layers=TINY_LAYERS,
        )
        train_grpo(args)

        assert os.path.exists(out_ckpt), "GRPO checkpoint should be saved"
        assert os.path.exists(summary), "Summary JSON should be saved"

        with open(summary) as f:
            s = json.load(f)
        assert s["size"] == size
        assert s["start_from"] == sft_ckpt
        assert s["num_iterations"] == 2
        assert "samples_seen" in s and s["samples_seen"] > 0
        assert "final_rollout_throughput" in s

        envs = gym.vector.SyncVectorEnv([make_env(ENV_ID, 0, False, size, "grpo-test")])
        agent = AgentCNN(envs, layers=TINY_LAYERS)
        agent.load_state_dict(torch.load(out_ckpt))
        envs.close()


class TestPolicyStep:
    """policy_step must be a faithful, temperature-aware reimplementation of
    AgentCNN.get_action_and_value's action path (including the EOT head)."""

    def _agent_and_obs(self, size=5, batch=6, seed=0):
        agent = _tiny_agent(size)
        torch.manual_seed(seed)
        obs = torch.randint(
            0, 3, (batch, len(Channel), size, size), dtype=torch.float32
        )
        return agent, obs

    def test_t1_matches_native_logp_and_entropy(self, registered_env):
        """At T=1.0, scoring an action with policy_step yields the same
        logp/entropy get_action_and_value reports for that action (both include
        the EOT Bernoulli)."""
        agent, obs = self._agent_and_obs()
        agent.eval()
        with torch.no_grad():
            torch.manual_seed(123)
            action_d, logp_native, ent_native, _v = agent.get_action_and_value(obs)
            x, y = action_d["xy"].unbind(dim=1)
            action_B7 = torch.stack(
                [
                    x,
                    y,
                    action_d["entity"],
                    action_d["direction"],
                    action_d["item"],
                    action_d["misc"],
                    action_d["eot"].long(),
                ],
                dim=1,
            )
            _out, logp_mine, ent_mine = policy_step(
                agent, obs, temperature=1.0, action_B7=action_B7
            )
        assert torch.allclose(logp_mine, logp_native, atol=1e-5), (logp_mine, logp_native)
        assert torch.allclose(ent_mine, ent_native, atol=1e-5)

    def test_scoring_roundtrips_sampled_action(self, registered_env):
        """Sampling then scoring the returned action reproduces its logp."""
        agent, obs = self._agent_and_obs(seed=1)
        agent.eval()
        gen = torch.Generator().manual_seed(7)
        with torch.no_grad():
            action_B7, logp_sampled, _e = policy_step(
                agent, obs, temperature=1.3, generator=gen
            )
            _out, logp_scored, _e2 = policy_step(
                agent, obs, temperature=1.3, action_B7=action_B7
            )
        assert torch.allclose(logp_sampled, logp_scored, atol=1e-6)

    def test_action_is_7_wide_with_eot(self, registered_env):
        agent, obs = self._agent_and_obs(seed=4)
        with torch.no_grad():
            action_B7, _lp, _e = policy_step(agent, obs)
        assert action_B7.shape == (obs.shape[0], 7)
        assert set(action_B7[:, 6].tolist()) <= {0, 1}  # eot column is binary

    def test_logp_depends_on_eot(self, registered_env):
        """EOT is always part of the objective: flipping the scored eot bit
        changes the logp (the EOT Bernoulli term is included)."""
        agent, obs = self._agent_and_obs(seed=5)
        a0 = _dummy_action(obs)  # eot=0
        a1 = a0.clone()
        a1[:, 6] = 1  # eot=1
        with torch.no_grad():
            _o, logp0, _e = policy_step(agent, obs, action_B7=a0)
            _o, logp1, _e2 = policy_step(agent, obs, action_B7=a1)
        assert not torch.allclose(logp0, logp1)

    def test_temperature_increases_entropy(self, registered_env):
        agent, obs = self._agent_and_obs(seed=2)
        with torch.no_grad():
            _a, _lp, ent_cold = policy_step(
                agent, obs, temperature=0.5, action_B7=_dummy_action(obs)
            )
            _a, _lp, ent_hot = policy_step(
                agent, obs, temperature=2.0, action_B7=_dummy_action(obs)
            )
        assert (ent_hot > ent_cold).all()


class TestSelectGrids:
    def test_all_grids_are_realisable(self, registered_env):
        from factorion import build_factory

        args = GRPOArgs(size=5, num_grids=6, num_missing_max=3, start_from="x", layers=TINY_LAYERS)
        grids = select_grids(args, random.Random(0))
        assert len(grids) == 6
        for seed, kind, num_missing in grids:
            assert build_factory(size=5, kind=kind, seed=seed) is not None
            assert 1 <= num_missing <= 3

    def test_eval_grids_disjoint_from_training_seeds(self, registered_env):
        """The held-out eval set must share no grid seed with ANY training
        iteration — including iteration 7, where an earlier integer eval offset
        collided exactly with the training stream."""
        args = GRPOArgs(
            size=5, num_grids=4, num_missing_max=2,
            eval_max_seeds=6, n_held_out_seeds=6,
            start_from="x", layers=TINY_LAYERS,
        )
        eval_seeds = {g[0] for g in _select_eval_grids(args)}
        assert eval_seeds  # non-empty
        for i in range(1, 12):  # spans the previously-colliding iteration 7
            train_seeds = {
                g[0]
                for g in select_grids(args, random.Random(args.seed * 1_000_003 + i))
            }
            assert eval_seeds.isdisjoint(train_seeds), f"iteration {i} overlaps eval"


class TestCollectRollout:
    def _rollout(self, size=5, num_grids=2, group_size=3, seed=0):
        args = GRPOArgs(
            size=size,
            num_grids=num_grids,
            group_size=group_size,
            num_missing_max=2,
            temperature=1.2,
            start_from="x",
            layers=TINY_LAYERS,
        )
        grids = select_grids(args, random.Random(seed))
        policy = _tiny_agent(size)
        ref = _tiny_agent(size)
        gen = torch.Generator().manual_seed(seed)
        batch = collect_rollout(policy, ref, args, grids, torch.device("cpu"), gen)
        return args, batch

    def test_group_shares_initial_grid(self, registered_env):
        args, batch = self._rollout()
        G = args.group_size
        for g in range(args.num_grids):
            ref_world = batch.solved_world_B[g * G]
            for j in range(1, G):
                assert torch.equal(batch.solved_world_B[g * G + j], ref_world)

    def test_groups_differ_from_each_other(self, registered_env):
        args, batch = self._rollout(num_grids=2, group_size=2)
        G = args.group_size
        assert not torch.equal(batch.solved_world_B[0], batch.solved_world_B[G])

    def test_transition_counts_match_episode_length(self, registered_env):
        args, batch = self._rollout()
        B = args.num_grids * args.group_size
        max_transitions = 2 * args.size + 2
        for lane in range(B):
            n = int((batch.lane_of_N == lane).sum())
            assert n == int(batch.steps_B[lane])
            assert 1 <= n <= max_transitions
        assert batch.obs_NCWH.shape[0] == int(batch.steps_B.sum())

    def test_eot_is_always_obeyed(self, registered_env):
        """Force the eot head to always fire -> every lane terminates via eot
        on its first step (the env stops on eot=1)."""
        size = 5
        policy = _tiny_agent(size)
        ref = _tiny_agent(size)
        with torch.no_grad():
            policy.eot_head[1].bias.fill_(50.0)  # sigmoid(50) ~ 1 -> eot=1
        args = GRPOArgs(
            size=size, num_grids=2, group_size=2, num_missing_max=2,
            start_from="x", layers=TINY_LAYERS,
        )
        grids = select_grids(args, random.Random(0))
        gen = torch.Generator().manual_seed(0)
        batch = collect_rollout(policy, ref, args, grids, torch.device("cpu"), gen)
        B = args.num_grids * args.group_size
        assert float(batch.eot_terminated_B.sum()) == B  # all stopped via eot
        assert int(batch.steps_B.max()) == 1  # eot fired on the first step

    def test_stored_obs_are_per_step_snapshots(self, registered_env):
        """A multi-step lane's stored observations must be independent per-step
        snapshots, not all collapsed to the lane's final obs (the failure mode
        if the stored row aliased the in-place-mutated obs buffer)."""
        size = 5
        policy = _tiny_agent(size)
        ref = _tiny_agent(size)
        with torch.no_grad():
            policy.eot_head[1].bias.fill_(-100.0)  # eot never fires -> multi-step lanes
        args = GRPOArgs(
            size=size, num_grids=2, group_size=2, num_missing_max=2,
            start_from="x", layers=TINY_LAYERS,
        )
        grids = select_grids(args, random.Random(0))
        gen = torch.Generator().manual_seed(0)
        batch = collect_rollout(policy, ref, args, grids, torch.device("cpu"), gen)
        saw_multistep = False
        for lane in range(args.num_grids * args.group_size):
            idx = (batch.lane_of_N == lane).nonzero().flatten()
            if len(idx) < 2:
                continue
            saw_multistep = True
            rows = batch.obs_NCWH[idx]
            all_eq_last = all(torch.equal(rows[k], rows[-1]) for k in range(len(rows)))
            assert not all_eq_last, f"lane {lane}: obs rows collapsed to final obs"
        assert saw_multistep, "test ineffective: no multi-step lanes produced"

    def test_shapes_consistent(self, registered_env):
        args, batch = self._rollout()
        N = batch.obs_NCWH.shape[0]
        assert batch.actions_N7.shape == (N, 7)
        assert batch.old_logp_N.shape == (N,)
        assert batch.ref_logp_N.shape == (N,)
        assert batch.lane_of_N.shape == (N,)


def _fake_batch(cum_reward, thput_final=None, eot=None):
    """Minimal RolloutBatch carrying just the reward ingredients."""
    B = len(cum_reward)
    z = torch.zeros((0,), dtype=torch.long)
    return RolloutBatch(
        obs_NCWH=torch.empty((0, 5, 5, 5)),
        actions_N7=torch.empty((0, 7), dtype=torch.long),
        old_logp_N=torch.empty((0,)),
        ref_logp_N=torch.empty((0,)),
        lane_of_N=z,
        group_of_lane_B=torch.arange(B),
        cum_reward_B=torch.tensor(cum_reward, dtype=torch.float32),
        thput_final_B=torch.tensor(
            thput_final if thput_final is not None else [0.0] * B, dtype=torch.float32
        ),
        eot_terminated_B=torch.tensor(
            eot if eot is not None else [0.0] * B, dtype=torch.float32
        ),
        steps_B=torch.zeros(B),
    )


class TestRewards:
    def test_reward_is_scaled_cum_reward(self):
        batch = _fake_batch(cum_reward=[1.0, -0.2, 0.5])
        args = GRPOArgs(reward_scale=0.5, start_from="x", layers=TINY_LAYERS)
        R = compute_rewards(batch, args)
        assert torch.allclose(R, torch.tensor([0.5, -0.1, 0.25]))

    def test_unit_scale_is_identity(self):
        batch = _fake_batch(cum_reward=[0.8, -0.24])
        args = GRPOArgs(reward_scale=1.0, start_from="x", layers=TINY_LAYERS)
        R = compute_rewards(batch, args)
        assert torch.allclose(R, torch.tensor([0.8, -0.24]))


class TestAdvantages:
    def test_per_group_zero_mean(self):
        R = torch.tensor([0.0, 4.0, 1.0, 1.0])  # 2 groups of 2
        A = compute_advantages(R, group_size=2)
        assert torch.allclose(A, torch.tensor([-2.0, 2.0, 0.0, 0.0]))

    def test_no_std_normalization(self):
        """Dr. GRPO does NOT divide by the group std. With group [0, 4]
        (std=2), std-normalized advantages would be [-1, 1]; we want [-2, 2]."""
        R = torch.tensor([0.0, 4.0])
        A = compute_advantages(R, group_size=2)
        assert torch.allclose(A, torch.tensor([-2.0, 2.0]))
        assert not torch.allclose(A, torch.tensor([-1.0, 1.0]))

    def test_each_group_sums_to_zero(self):
        R = torch.tensor([3.0, 1.0, 2.0, 5.0, 9.0, 1.0])  # 3 groups of 2
        A = compute_advantages(R, group_size=2)
        for g in range(3):
            assert abs(float(A[2 * g] + A[2 * g + 1])) < 1e-6


class TestGRPOUpdate:
    def _setup(self, size=5, seed=0, learning_rate=2e-6, update_epochs=2):
        args = GRPOArgs(
            size=size,
            num_grids=2,
            group_size=3,
            num_missing_max=2,
            temperature=1.1,
            start_from="x",
            learning_rate=learning_rate,
            update_epochs=update_epochs,
            layers=TINY_LAYERS,
        )
        grids = select_grids(args, random.Random(seed))
        policy = _tiny_agent(size)
        ref = _tiny_agent(size)
        gen = torch.Generator().manual_seed(seed)
        batch = collect_rollout(policy, ref, args, grids, torch.device("cpu"), gen)
        R = compute_rewards(batch, args)
        A_N = broadcast_advantages(compute_advantages(R, args.group_size), batch.lane_of_N)
        return args, policy, batch, A_N

    def test_ratio_one_and_finite_loss_with_zero_lr(self):
        args, policy, batch, A_N = self._setup(learning_rate=0.0, update_epochs=1)
        opt = torch.optim.Adam(policy.parameters(), lr=0.0)
        m = grpo_update(policy, opt, batch, A_N, args)
        assert abs(m["grpo/ratio_mean"] - 1.0) < 1e-5
        assert abs(m["grpo/approx_kl"]) < 1e-5
        assert m["loss/kl_ref"] >= -1e-6
        assert np.isfinite(m["loss/total"])

    def test_update_changes_weights(self):
        args, policy, batch, _A = self._setup(learning_rate=1e-2, update_epochs=2)
        N = batch.obs_NCWH.shape[0]
        assert N > 0
        A_N = torch.where(torch.arange(N) % 2 == 0, torch.ones(N), -torch.ones(N))
        before = [p.detach().clone() for p in policy.parameters()]
        opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
        m = grpo_update(policy, opt, batch, A_N, args)
        after = list(policy.parameters())
        moved = any(not torch.equal(b, a) for b, a in zip(before, after))
        assert moved, "expected the policy weights to change after an update"
        assert np.isfinite(m["loss/total"])
        assert m["loss/kl_ref"] >= -1e-6

    def test_empty_batch_is_safe(self):
        args = GRPOArgs(size=5, group_size=2, start_from="x", layers=TINY_LAYERS)
        policy = _tiny_agent(5)
        empty = _fake_batch(cum_reward=[])
        opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
        m = grpo_update(policy, opt, empty, torch.empty((0,)), args)
        assert m["grpo/n_transitions"] == 0


class TestGreedyEval:
    def test_metrics_well_formed(self, registered_env):
        size = 5
        args = GRPOArgs(
            size=size, num_grids=6, num_missing_max=2, eval_num_envs=3, start_from="x", layers=TINY_LAYERS
        )
        val_grids = select_grids(args, random.Random(999))
        policy = _tiny_agent(size)
        m = greedy_eval(policy, args, val_grids, torch.device("cpu"))
        assert m["eval/n"] == len(val_grids)
        assert 0.0 <= m["eval/throughput"] <= 1.0
        assert 0.0 <= m["eval/success_rate"] <= 1.0
        assert 0.0 <= m["eval/dir_acc_vs_ref"] <= 1.0

    def test_empty_val_set_is_safe(self, registered_env):
        args = GRPOArgs(size=5, start_from="x", layers=TINY_LAYERS)
        policy = _tiny_agent(5)
        m = greedy_eval(policy, args, [], torch.device("cpu"))
        assert m["eval/n"] == 0
        assert m["eval/throughput"] == 0.0


def _world(size, fill=0):
    return torch.full((len(Channel), size, size), fill, dtype=torch.float32)


def _batch_with_worlds(final_worlds, solved_worlds, thput_final):
    B = len(final_worlds)
    z = torch.zeros((0,), dtype=torch.long)
    return RolloutBatch(
        obs_NCWH=torch.empty((0, 5, 5, 5)),
        actions_N7=torch.empty((0, 7), dtype=torch.long),
        old_logp_N=torch.empty((0,)),
        ref_logp_N=torch.empty((0,)),
        lane_of_N=z,
        group_of_lane_B=torch.arange(B),
        cum_reward_B=torch.zeros(B),
        thput_final_B=torch.tensor(thput_final, dtype=torch.float32),
        eot_terminated_B=torch.zeros(B),
        steps_B=torch.zeros(B),
        final_world_B=final_worlds,
        solved_world_B=solved_worlds,
    )


class TestDiversity:
    def test_off_reference_and_unique_paths(self):
        size = 5
        solved = _world(size)
        solved[Channel.ENTITIES.value, 0, 0] = 1  # the "reference" layout

        on_ref = solved.clone()  # identical to reference
        off_ref = solved.clone()
        off_ref[Channel.ENTITIES.value, 0, 1] = 1  # a different (but working) path

        # 1 group of 2: lane0 on-reference, lane1 off-reference, both working
        batch = _batch_with_worlds(
            final_worlds=[on_ref, off_ref],
            solved_worlds=[solved, solved],
            thput_final=[1.0, 1.0],
        )
        args = GRPOArgs(size=size, group_size=2, start_from="x", layers=TINY_LAYERS)
        R_B = torch.tensor([1.0, 3.0])
        m = compute_diversity(batch, R_B, args)

        assert abs(m["diversity/reward_var"] - 1.0) < 1e-6  # var([1,3]) = 1.0
        assert m["diversity/unique_path_frac"] == 1.0
        assert abs(m["diversity/off_reference_success"] - 0.5) < 1e-6
        assert m["diversity/working_frac"] == 1.0

    def test_identical_group_has_zero_variance_and_no_diversity(self):
        size = 5
        w = _world(size)
        batch = _batch_with_worlds(
            final_worlds=[w.clone(), w.clone()],
            solved_worlds=[w.clone(), w.clone()],
            thput_final=[1.0, 1.0],
        )
        args = GRPOArgs(size=size, group_size=2, start_from="x", layers=TINY_LAYERS)
        m = compute_diversity(batch, torch.tensor([2.0, 2.0]), args)
        assert m["diversity/reward_var"] == 0.0
        assert m["diversity/unique_path_frac"] == 0.5  # 1 unique / 2
        assert m["diversity/off_reference_success"] == 0.0

    def test_non_working_lanes_excluded_from_off_reference(self):
        size = 5
        solved = _world(size)
        solved[Channel.ENTITIES.value, 0, 0] = 1
        off = solved.clone()
        off[Channel.ENTITIES.value, 0, 1] = 1
        # off-reference layout but thput_final < 1 -> not a working factory
        batch = _batch_with_worlds([off, off], [solved, solved], thput_final=[0.5, 0.5])
        args = GRPOArgs(size=size, group_size=2, start_from="x", layers=TINY_LAYERS)
        m = compute_diversity(batch, torch.tensor([0.0, 0.0]), args)
        assert m["diversity/working_frac"] == 0.0
        assert m["diversity/off_reference_success"] == 0.0  # no working factories

    def test_runs_on_real_rollout(self, registered_env):
        args = GRPOArgs(
            size=5, num_grids=2, group_size=3, num_missing_max=2, start_from="x", layers=TINY_LAYERS
        )
        grids = select_grids(args, random.Random(3))
        policy, ref = _tiny_agent(5), _tiny_agent(5)
        gen = torch.Generator().manual_seed(3)
        batch = collect_rollout(policy, ref, args, grids, torch.device("cpu"), gen)
        R_B = compute_rewards(batch, args)
        m = compute_diversity(batch, R_B, args)
        for k in (
            "diversity/reward_var",
            "diversity/unique_path_frac",
            "diversity/off_reference_success",
            "diversity/working_frac",
            "diversity/dir_acc_vs_ref",
        ):
            assert k in m and np.isfinite(m[k])
        assert 0.0 <= m["diversity/unique_path_frac"] <= 1.0
