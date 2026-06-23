"""GRPO (Group Relative Policy Optimization) RL finetuning for AgentCNN.

This is the *Dr. GRPO* variant (Liu et al., "Understanding R1-Zero-Like
Training"): the group-relative advantage is `R_i - mean_group(R)` with **no**
division by the group std and **no** per-response length normalization. Both of
those terms inject bias — std-normalization especially blows up the gradient on
easy/hard grids where reward variance is tiny.

Why GRPO instead of PPO here? The reward is verifiable and outcome-dominated
(throughput jumps to ~1 the moment source connects to sink, plus a terminal
completion bonus). A learned value function buys little against a near-terminal
step-function reward, and the critic is the main source of PPO's tuning surface
and failure modes (value clip, GAE lambda, value-loss coef, shared-encoder
corruption). GRPO deletes the critic entirely: it samples a *group* of G
completions on the *same* seeded grid, scores each by final reward, and
normalizes within the group. Two distinct valid belt paths both score ~1 and
both get non-negative advantage — the baseline is diversity-neutral, which is
exactly the path-agnostic objective we want.

Pipeline position: SFT pretraining gives the policy a strong placement prior
(~0.68 throughput / ~0.78 dir_acc); GRPO finetunes it. Load an SFT checkpoint
with --start-from.

Usage:
    python sft.py --size 8 --checkpoint-path sft.pt ...   # produce the prior
    python grpo.py --start-from sft.pt --size 8 --track ...
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions import Categorical

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import factorion_rs  # noqa: F401,E402  (imported for its side-effect: the env's solver)
from factorion import (  # noqa: E402
    Channel,
    LessonKind,
    blank_entities,
    build_factory,
)

from ppo import Args as PPOArgs  # noqa: E402  (env reward coeffs live as its class defaults)
from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402

GRPO_ENV_ID = "factorion/FactorioEnv-v0-grpo"


@dataclass
class GRPOArgs:
    # ── experiment / tracking ────────────────────────────────────────────
    seed: int = 1
    size: int = 12
    """grid width/height"""
    env_id: str = "factorion/FactorioEnv-v0"
    start_from: str = ""
    """path to the SFT checkpoint to finetune. REQUIRED — GRPO finetunes a
    pretrained actor, it does not train from scratch."""
    track: bool = False
    wandb_project_name: str = "factorion"
    wandb_entity: Optional[str] = None
    wandb_group: Optional[str] = None
    tags: Optional[list[str]] = None

    # ── network (must match the SFT checkpoint) ──────────────────────────
    chan1: int = 48
    chan2: int = 48
    chan3: int = 64
    flat_dim: int = 128
    """unused by AgentCNN (it recomputes chan3*W*H); kept for ctor compat"""
    tile_head_std: float = 0.02208

    # ── GRPO sampling ────────────────────────────────────────────────────
    num_iterations: int = 200
    """outer optimisation iterations"""
    num_grids: int = 16
    """distinct seeded grids sampled per iteration. Keep enough that the
    group baseline isn't dominated by one grid's difficulty."""
    group_size: int = 8
    """G — completions sampled per grid. Suggest 8; sweep {4, 8, 16}."""
    temperature: float = 1.0
    """sampling temperature; >1 widens the group so it covers distinct paths.
    Applied identically to old-logp, new-logp and ref-logp so the ratio stays
    a clean importance ratio of the temperature-T policy."""

    # ── GRPO loss ────────────────────────────────────────────────────────
    clip_coef: float = 0.2
    """PPO-style ratio clip epsilon"""
    beta_kl: float = 0.02
    """KL(pi_theta || pi_ref) penalty. Main guardrail against drifting off the
    SFT manifold. Too high: can't escape the single-path bias. Too low: risk
    collapse / reward hacking. Tune in [0.01, 0.04]."""
    learning_rate: float = 2e-6
    """RL on a pretrained net wants a *much* smaller LR than SFT."""
    update_epochs: int = 2
    """inner PPO-style update epochs per outer batch (1-4)"""
    max_grad_norm: float = 1.0
    ent_coef: float = 0.0
    """small entropy bonus; raise only if the policy collapses prematurely"""
    adam_epsilon: float = 1e-8

    # ── reward (outcome-only by default) ─────────────────────────────────
    reward_mode: str = "outcome"
    """"outcome": R_i = scale*(thp_final + bonus_coef*bonus). "per_step": the
    PPO-faithful sum of per-step (throughput+validity) reward (minus shaping
    when drop_shaping). Outcome is the v1 default — standard GRPO, and the
    advantage is dominated by the same connect/not-connect signal anyway."""
    reward_scale: float = 0.1
    """multiplies R_i. Dr. GRPO drops std-normalization, so we lose its
    automatic scale-invariance; reward_scale keeps |advantage| ~ O(1) against
    a completion bonus that can reach ~2*size."""
    completion_bonus_coef: float = 1.0
    """weight on the terminal (max_steps - steps) bonus inside R_i"""
    drop_shaping: bool = True
    """drop potential-based shaping from R_i (only affects reward_mode=
    "per_step"; shaping is policy-invariant so it ~cancels in the group
    baseline — kept as a knob to verify that empirically, per spec)"""

    # ── grid difficulty ──────────────────────────────────────────────────
    num_missing_min: int = 1
    num_missing_max: int = 0
    """0 => auto-cap at 2*size. Each grid samples num_missing uniformly in
    [num_missing_min, cap] so groups straddle the success boundary."""

    # ── end-of-turn (EOT) ────────────────────────────────────────────────
    # See the rollout stop-condition for the rationale. v1: episodes end only
    # on connection or max_steps; EOT is neither used to stop nor RL-trained.
    honor_eot: bool = False
    eot_threshold: float = 0.5

    # ── held-out eval ────────────────────────────────────────────────────
    eval_every: int = 10
    """run greedy held-out eval every N iterations (0 disables)"""
    eval_max_seeds: int = 64
    eval_num_envs: int = 8
    n_held_out_seeds: int = 256
    """size of the held-out seed pool (disjoint from training seeds)"""

    # ── io ───────────────────────────────────────────────────────────────
    checkpoint_path: str = "grpo_checkpoint.pt"
    summary_path: Optional[str] = None


def _device() -> torch.device:
    dev = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    if dev.type == "mps":
        # metal only likes f32
        torch.set_default_dtype(torch.float32)
    return dev


def _build_agent(args: GRPOArgs, envs, device) -> AgentCNN:
    """Construct an AgentCNN sized to match the SFT checkpoint and load it.

    Used for both the trainable policy and the frozen reference — both start
    from the *same* SFT weights (we never deepcopy a mid-training policy)."""
    agent = AgentCNN(
        envs,
        chan1=args.chan1,
        chan2=args.chan2,
        chan3=args.chan3,
        flat_dim=args.flat_dim,
        tile_head_std=args.tile_head_std,
    )
    if not args.start_from:
        raise ValueError(
            "GRPO finetunes a pretrained actor: pass --start-from <sft_checkpoint.pt>"
        )
    print(f"Loading SFT weights from {args.start_from}")
    agent.load_state_dict(torch.load(args.start_from, map_location=device))
    return agent.to(device)


def _sample(dist: Categorical, generator: Optional[torch.Generator]) -> torch.Tensor:
    """Sample category indices, optionally from an explicit Generator.

    A dedicated generator decouples action sampling from the global torch RNG,
    which build_factory()/blank_entities() reseed on every env.reset — without
    it, action draws would correlate with grid seeds. multinomial runs on CPU
    so the generator works regardless of the policy's device (mps generators
    are finicky)."""
    if generator is None:
        return dist.sample()
    idx = torch.multinomial(dist.probs.to("cpu"), 1, generator=generator).squeeze(-1)
    return idx.to(dist.probs.device)


def policy_step(
    agent: AgentCNN,
    obs_BCWH: torch.Tensor,
    temperature: float = 1.0,
    action_B6: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Temperature-aware replication of AgentCNN.get_action_and_value's action
    path. Sampling (action_B6=None) and scoring (action_B6 given) share ONE
    code path so the temperature is applied identically — to the tile logits
    AND all four per-tile head logits — guaranteeing old-logp, new-logp and
    ref-logp describe the same temperature-T policy and the ratio stays a clean
    importance ratio.

    We deliberately do NOT call agent.get_action_and_value: it has no
    temperature knob and also computes the (unused) value head. At T=1.0 this
    is numerically identical to get_action_and_value's logp/entropy.

    Action layout (B, 6) int columns: [x, y, entity, direction, item, misc] —
    the same convention the env's step() and get_action_and_value's scoring
    path use. Returns (action_B6, logp_B, entropy_B); logp/entropy are SUMS
    over the 5 categorical heads (tile, entity, dir, item, misc). EOT is a
    separate head and is intentionally excluded.
    """
    encoded_BCWH = agent.encoder(obs_BCWH)
    B = encoded_BCWH.shape[0]
    H = agent.height

    # ── tile selection (joint x,y via 1x1 conv) ──
    tile_logits_BN = agent.tile_logits(encoded_BCWH).reshape(B, -1) / temperature
    dist_tile = Categorical(logits=tile_logits_BN)
    if action_B6 is None:
        tile_idx_B = _sample(dist_tile, generator)
    else:
        # Reconstruct tile index from stored (x, y) exactly as
        # get_action_and_value does (ppo.py:915): tile = x*height + y.
        tile_idx_B = action_B6[:, 0] * H + action_B6[:, 1]
    x_B = tile_idx_B // H
    y_B = tile_idx_B % H

    # ── per-tile heads conditioned on features at the selected tile ──
    batch_idx = torch.arange(B, device=encoded_BCWH.device)
    feats_BC = encoded_BCWH[batch_idx, :, x_B, y_B]
    dist_e = Categorical(logits=agent.ent_head(feats_BC) / temperature)
    dist_d = Categorical(logits=agent.dir_head(feats_BC) / temperature)
    dist_i = Categorical(logits=agent.item_head(feats_BC) / temperature)
    dist_m = Categorical(logits=agent.misc_head(feats_BC) / temperature)
    if action_B6 is None:
        ent_B = _sample(dist_e, generator)
        dir_B = _sample(dist_d, generator)
        item_B = _sample(dist_i, generator)
        misc_B = _sample(dist_m, generator)
    else:
        ent_B = action_B6[:, 2]
        dir_B = action_B6[:, 3]
        item_B = action_B6[:, 4]
        misc_B = action_B6[:, 5]

    logp_B = (
        dist_tile.log_prob(tile_idx_B)
        + dist_e.log_prob(ent_B)
        + dist_d.log_prob(dir_B)
        + dist_i.log_prob(item_B)
        + dist_m.log_prob(misc_B)
    )
    entropy_B = (
        dist_tile.entropy()
        + dist_e.entropy()
        + dist_d.entropy()
        + dist_i.entropy()
        + dist_m.entropy()
    )
    out_B6 = torch.stack([x_B, y_B, ent_B, dir_B, item_B, misc_B], dim=1)
    return out_B6, logp_B, entropy_B


def _max_steps(args: GRPOArgs) -> int:
    # Short horizon ("one entity per step over a belt chain"), matching ppo.py's
    # make_env. NOT FactorioEnv's default of size*size, which would make
    # episodes (and the completion bonus) ~size times longer.
    return 2 * args.size


def _make_rollout_env(args: GRPOArgs) -> FactorioEnv:
    # idx=0 so reset(seed) is an exact pass-through: the grid is a pure
    # function of (size, kind, seed), letting a group of lanes share one grid.
    return FactorioEnv(size=args.size, max_steps=_max_steps(args), idx=0)


def select_grids(
    args: GRPOArgs, rng: random.Random
) -> list[tuple[int, LessonKind, int]]:
    """Pick `num_grids` distinct, pre-validated grids `(seed, kind, num_missing)`.

    Each grid gets a random LessonKind and a random num_missing in
    [num_missing_min, cap] so a batch straddles the success boundary (some
    groups all-connect, some none — that's where group-relative advantage has
    signal). We probe build_factory once per candidate and skip seeds it can't
    realise (rejection sampling exhausts on tight grids / hard kinds), because
    FactorioEnv.reset *raises* on a None factory when the kind is pinned."""
    cap = args.num_missing_max if args.num_missing_max > 0 else 2 * args.size
    lo = max(1, args.num_missing_min)
    hi = max(lo, cap)
    kinds = list(LessonKind)
    grids: list[tuple[int, LessonKind, int]] = []
    seed = rng.randrange(1, 2**31 - args.num_grids * 64 - 1)
    attempts = 0
    max_attempts = args.num_grids * 64
    while len(grids) < args.num_grids and attempts < max_attempts:
        attempts += 1
        seed += 1
        kind = kinds[rng.randrange(len(kinds))]
        try:
            ok = build_factory(size=args.size, kind=kind, seed=seed) is not None
        except Exception:
            ok = False
        if not ok:
            continue
        grids.append((seed, kind, rng.randint(lo, hi)))
    if len(grids) < args.num_grids:
        raise RuntimeError(
            f"select_grids: only validated {len(grids)}/{args.num_grids} grids "
            f"in {attempts} attempts (size={args.size})"
        )
    return grids


@dataclass
class RolloutBatch:
    """Flattened transitions from one GRPO rollout, plus the per-lane reward
    ingredients and final/solved worlds the reward + diversity stages consume.

    N = total transitions across all lanes (variable, since episodes vary in
    length). B = num_grids * group_size lanes. A transition is one placement
    step; a lane that terminated early simply contributes fewer rows."""

    obs_NCWH: torch.Tensor
    actions_N6: torch.Tensor
    old_logp_N: torch.Tensor
    ref_logp_N: torch.Tensor
    lane_of_N: torch.Tensor  # (N,) which lane each transition came from
    group_of_lane_B: torch.Tensor  # (B,) lane -> group id (lane // G)
    # per-lane reward ingredients
    thp_final_B: torch.Tensor  # (B,) last throughput the env reported
    terminated_B: torch.Tensor  # (B,) 1.0 if the lane connected (throughput>=1)
    completion_bonus_B: torch.Tensor  # (B,) max_steps - steps_at_terminal
    steps_B: torch.Tensor  # (B,) episode length (transitions stored)
    per_step_reward_B: torch.Tensor  # (B,) sum of PPO-style per-step reward
    # per-lane worlds for diversity metrics
    final_world_B: list = field(default_factory=list)
    solved_world_B: list = field(default_factory=list)


def collect_rollout(
    policy: AgentCNN,
    ref: AgentCNN,
    args: GRPOArgs,
    grids: list[tuple[int, LessonKind, int]],
    device,
    generator: Optional[torch.Generator] = None,
) -> RolloutBatch:
    """Sample a GROUP of `group_size` complete episodes for each grid, in
    lockstep across all B = num_grids*group_size lanes.

    Lockstep (not a refill queue) keeps groups intact: lane g*G+j always
    belongs to grid g, so the Dr. GRPO baseline is computed over completions of
    the *same* grid. Finished lanes are masked out — they keep getting fed
    through the batched forward (cheap; episodes are short) but store no further
    transitions and are never stepped again."""
    G = args.group_size
    B = len(grids) * G
    T = args.temperature
    H = args.size

    envs = [_make_rollout_env(args) for _ in range(B)]

    # Reset ALL lanes before any sampling. build_factory/blank_entities reseed
    # the global RNG on each reset; doing them up front (then sampling via the
    # explicit `generator`) keeps action draws independent of grid seeds.
    obs_B = [None] * B
    last_thp = [0.0] * B
    solved_world_B = [None] * B
    for g, (seed, kind, num_missing) in enumerate(grids):
        for j in range(G):
            lane = g * G + j
            o, info = envs[lane].reset(
                seed=seed,
                options={"num_missing_entities": num_missing, "kind": kind},
            )
            obs_B[lane] = o
            last_thp[lane] = float(info.get("throughput", 0.0))
            solved_world_B[lane] = envs[lane]._solved_world_CWH.clone()

    active = [True] * B
    n_trans = [0] * B
    terminated_B = [0.0] * B
    completion_bonus_B = [0.0] * B
    final_world_B = [None] * B
    steps_B = [0] * B
    # Σ of the env's PPO-style per-step reward, captured so reward_mode=
    # "per_step" is a faithful reconstruction. The env reads its coeffs from
    # the Args *class* defaults (the `'args' not in locals()` branch in
    # FactorioEnv.step), so we read the same attrs here.
    per_step_sum = [0.0] * B
    _ct, _cv = PPOArgs.coeff_throughput, PPOArgs.coeff_validity

    obs_rows: list[torch.Tensor] = []
    act_rows: list[torch.Tensor] = []
    old_lp: list[torch.Tensor] = []
    ref_lp: list[torch.Tensor] = []
    lane_ids: list[int] = []

    max_steps = _max_steps(args)
    with torch.no_grad():
        # Loop until every lane has terminated/truncated. The env truncates
        # when steps>max_steps (checked pre-increment), so a non-connecting
        # lane needs up to max_steps+2 steps; +4 leaves margin so truncation
        # always fires inside the loop, never via the fallback below.
        for _t in range(max_steps + 4):
            if not any(active):
                break
            obs_batch = torch.as_tensor(
                np.stack(obs_B), dtype=torch.float32, device=device
            )
            action_B6, logp_B, _ent = policy_step(
                policy, obs_batch, temperature=T, generator=generator
            )
            _a, ref_logp_B, _e = policy_step(
                ref, obs_batch, temperature=T, action_B6=action_B6
            )

            # ── EOT stop signal — OFF in v1 (honor_eot defaults False) ──
            # Under this reward, connecting auto-terminates the episode
            # (throughput>=1) and stopping early is never beneficial, so the
            # SFT eot head is a pure confound here: honoring it can truncate a
            # near-complete factory and inject reward variance for no gain. We
            # therefore neither stop on it nor RL-train it in v1. `--honor-eot`
            # flips it on for later experiments. Tracking: see the EOT GitHub
            # issue referenced in the project notes.
            eot_stop = [False] * B
            if args.honor_eot:
                encoded = policy.encoder(obs_batch)
                eot_prob_B = torch.sigmoid(policy.eot_head(encoded).squeeze(-1))
                eot_stop = (eot_prob_B > args.eot_threshold).tolist()

            for lane in range(B):
                if not active[lane]:
                    continue
                if eot_stop[lane]:
                    # Agent chose to stop without an env step: no transition,
                    # episode ends with whatever it has placed so far.
                    active[lane] = False
                    terminated_B[lane] = 0.0
                    completion_bonus_B[lane] = 0.0
                    final_world_B[lane] = envs[lane]._world_CWH.clone()
                    steps_B[lane] = n_trans[lane]
                    continue

                obs_rows.append(obs_batch[lane].cpu())
                act_rows.append(action_B6[lane].cpu())
                old_lp.append(logp_B[lane].cpu())
                ref_lp.append(ref_logp_B[lane].cpu())
                lane_ids.append(lane)
                n_trans[lane] += 1

                act = action_B6[lane].tolist()
                action_dict = {
                    "xy": np.array([act[0], act[1]], dtype=int),
                    "entity": int(act[2]),
                    "direction": int(act[3]),
                    "item": int(act[4]),
                    "misc": int(act[5]),
                }
                next_obs, _r, terminated, truncated, info = envs[lane].step(action_dict)
                last_thp[lane] = float(info.get("throughput", 0.0))

                # PPO-style per-step reward (throughput+validity, normalized),
                # + shaping deltas unless drop_shaping. Shaping is
                # potential-based so its episode sum telescopes to a path-
                # independent constant that ~cancels in the group baseline.
                valid_t = 0.0 if any(info.get("invalid_reason", {}).values()) else 1.0
                per_step_sum[lane] += (
                    _ct * last_thp[lane] + _cv * valid_t
                ) / (_ct + _cv)
                if not args.drop_shaping:
                    per_step_sum[lane] += (
                        float(info.get("shaping_location_delta", 0.0))
                        + float(info.get("shaping_entity_delta", 0.0))
                        + float(info.get("shaping_direction_delta", 0.0))
                    )

                if terminated or truncated:
                    active[lane] = False
                    terminated_B[lane] = 1.0 if terminated else 0.0
                    completion_bonus_B[lane] = float(info.get("completion_bonus", 0.0))
                    final_world_B[lane] = envs[lane]._world_CWH.clone()
                    steps_B[lane] = n_trans[lane]
                else:
                    obs_B[lane] = next_obs

    # Lanes that somehow never deactivated (shouldn't happen) get closed out.
    for lane in range(B):
        if final_world_B[lane] is None:
            final_world_B[lane] = envs[lane]._world_CWH.clone()
            steps_B[lane] = n_trans[lane]
    for e in envs:
        e.close()

    group_of_lane = torch.tensor([lane // G for lane in range(B)], dtype=torch.long)
    if obs_rows:
        obs_NCWH = torch.stack(obs_rows).to(device)
        actions_N6 = torch.stack(act_rows).to(device)
        old_logp_N = torch.stack(old_lp).to(device)
        ref_logp_N = torch.stack(ref_lp).to(device)
        lane_of_N = torch.tensor(lane_ids, dtype=torch.long, device=device)
    else:
        obs_NCWH = torch.empty((0, len(Channel), H, H), device=device)
        actions_N6 = torch.empty((0, 6), dtype=torch.long, device=device)
        old_logp_N = torch.empty((0,), device=device)
        ref_logp_N = torch.empty((0,), device=device)
        lane_of_N = torch.empty((0,), dtype=torch.long, device=device)

    return RolloutBatch(
        obs_NCWH=obs_NCWH,
        actions_N6=actions_N6,
        old_logp_N=old_logp_N,
        ref_logp_N=ref_logp_N,
        lane_of_N=lane_of_N,
        group_of_lane_B=group_of_lane,
        thp_final_B=torch.tensor(last_thp, dtype=torch.float32),
        terminated_B=torch.tensor(terminated_B, dtype=torch.float32),
        completion_bonus_B=torch.tensor(completion_bonus_B, dtype=torch.float32),
        steps_B=torch.tensor(steps_B, dtype=torch.float32),
        per_step_reward_B=torch.tensor(per_step_sum, dtype=torch.float32),
        final_world_B=final_world_B,
        solved_world_B=solved_world_B,
    )


def compute_rewards(batch: RolloutBatch, args: GRPOArgs) -> torch.Tensor:
    """Scalar episode return R_i per lane (B,).

    "outcome" (default, standard GRPO): R_i = scale*(thp_final + bonus_coef *
    bonus), where bonus is the terminal (max_steps - steps) only when the lane
    connected. Pure outcome reward — no per-step bookkeeping.

    "per_step" (PPO-faithful, opt-in): the env's summed per-step
    (throughput+validity) reward (with shaping unless drop_shaping) plus the
    same terminal bonus — for an apples-to-apples comparison with PPO."""
    bonus = args.completion_bonus_coef * batch.terminated_B * batch.completion_bonus_B
    if args.reward_mode == "outcome":
        core = batch.thp_final_B + bonus
    elif args.reward_mode == "per_step":
        core = batch.per_step_reward_B + bonus
    else:
        raise ValueError(f"unknown reward_mode {args.reward_mode!r}")
    return args.reward_scale * core


def compute_advantages(R_B: torch.Tensor, group_size: int) -> torch.Tensor:
    """Dr. GRPO advantage: A_i = R_i - mean_group(R).

    NO division by the group std and NO per-response length normalization —
    both inject bias (Liu et al.). std-normalization in particular blows up the
    gradient on easy/hard grids where reward variance is tiny. Lanes are
    group-contiguous (lane = g*G + j, guaranteed by collect_rollout), so a
    reshape recovers the groups."""
    num_groups = R_B.shape[0] // group_size
    grouped = R_B.view(num_groups, group_size)
    baseline = grouped.mean(dim=1, keepdim=True)
    return (grouped - baseline).reshape(-1)


def broadcast_advantages(
    A_B: torch.Tensor, lane_of_N: torch.Tensor
) -> torch.Tensor:
    """Scatter each lane's scalar advantage onto all of its transitions."""
    return A_B.to(lane_of_N.device)[lane_of_N]


def _layout_key(world: torch.Tensor) -> tuple:
    """Hashable identity of a factory layout: its entity + direction channels.
    Two completions with the same key placed the same things facing the same
    way (same path); different keys = genuinely different layouts."""
    ent = world[Channel.ENTITIES.value].reshape(-1).to(torch.int64).tolist()
    dirc = world[Channel.DIRECTION.value].reshape(-1).to(torch.int64).tolist()
    return (tuple(ent), tuple(dirc))


def compute_diversity(batch: RolloutBatch, R_B: torch.Tensor, args: GRPOArgs) -> dict:
    """Diversity + off-reference metrics — where GRPO should distinguish itself
    from SFT and from a reference-anchored critic.

      - diversity/reward_var: mean intra-group reward variance. ~0 across most
        groups means the learning signal is dead (group collapse → advantages
        ~0); raise temperature / lower beta_kl.
      - diversity/unique_path_frac: mean fraction of DISTINCT final layouts
        within a group. GRPO should push this UP vs SFT.
      - diversity/off_reference_success: of the WORKING (connected) factories,
        the fraction whose layout differs from the SFT reference solution. The
        cleanest single number for "did RL escape the single-path anchor."
      - diversity/working_frac: fraction of lanes that connected.
      - diversity/dir_acc_vs_ref: mean full-grid direction match to reference
        (REPORTED, not a target — see greedy_eval)."""
    G = args.group_size
    B = len(batch.final_world_B)
    num_groups = B // G

    grouped = R_B.view(num_groups, G)
    reward_var = grouped.var(dim=1, unbiased=False).mean().item()

    unique_fracs = []
    for g in range(num_groups):
        keys = {_layout_key(batch.final_world_B[g * G + j]) for j in range(G)}
        unique_fracs.append(len(keys) / G)

    working = 0
    off_ref = 0
    dir_accs = []
    for lane in range(B):
        fw = batch.final_world_B[lane]
        sw = batch.solved_world_B[lane]
        fe, se = fw[Channel.ENTITIES.value], sw[Channel.ENTITIES.value]
        fd, sd = fw[Channel.DIRECTION.value], sw[Channel.DIRECTION.value]
        dir_accs.append((fd == sd).float().mean().item())
        if batch.terminated_B[lane] >= 1.0:
            working += 1
            on_reference = torch.equal(fe, se) and torch.equal(fd, sd)
            if not on_reference:
                off_ref += 1

    return {
        "diversity/reward_var": reward_var,
        "diversity/unique_path_frac": float(np.mean(unique_fracs)) if unique_fracs else 0.0,
        "diversity/off_reference_success": (off_ref / working) if working else 0.0,
        "diversity/working_frac": working / B if B else 0.0,
        "diversity/dir_acc_vs_ref": float(np.mean(dir_accs)) if dir_accs else 0.0,
    }


def greedy_eval(
    policy: AgentCNN,
    args: GRPOArgs,
    val_grids: list[tuple[int, LessonKind, int]],
    device,
) -> dict:
    """Batched greedy rollout on held-out grids (argmax every head), mirroring
    sft.run_rollout_eval but self-contained and using the GRPO horizon.

    Episodes end on connection (throughput>=1) or max_steps; EOT is honored
    only when args.honor_eot, so the regime matches training. Reports:
      - eval/throughput: mean final throughput (the real objective)
      - eval/success_rate: fraction that fully connect source->sink
      - eval/dir_acc_vs_ref: mean tile_match_direction (REPORTED, NOT a target
        — under working GRPO this may fall while throughput rises, because the
        actor produces valid OFF-reference paths)."""
    was_training = policy.training
    policy.eval()
    max_steps = _max_steps(args)
    K = max(1, args.eval_num_envs)
    thps: list[float] = []
    dir_accs: list[float] = []

    with torch.no_grad():
        for start in range(0, len(val_grids), K):
            chunk = val_grids[start : start + K]
            n = len(chunk)
            envs = [_make_rollout_env(args) for _ in range(n)]
            obs = [None] * n
            last_thp = [0.0] * n
            last_dir = [0.0] * n
            for i, (seed, kind, num_missing) in enumerate(chunk):
                o, info = envs[i].reset(
                    seed=seed,
                    options={"num_missing_entities": num_missing, "kind": kind},
                )
                obs[i] = o
                last_thp[i] = float(info.get("throughput", 0.0))
                last_dir[i] = float(info.get("tile_match_direction", 0.0))

            active = [True] * n
            for _t in range(max_steps + 4):
                if not any(active):
                    break
                obs_b = torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device)
                encoded = policy.encoder(obs_b)
                tile = policy.tile_logits(encoded).reshape(n, -1).argmax(dim=1)
                x = tile // args.size
                y = tile % args.size
                bidx = torch.arange(n, device=device)
                feats = encoded[bidx, :, x, y]
                ent = policy.ent_head(feats).argmax(dim=1)
                dir_ = policy.dir_head(feats).argmax(dim=1)
                item = policy.item_head(feats).argmax(dim=1)
                misc = policy.misc_head(feats).argmax(dim=1)
                if args.honor_eot:
                    eot = torch.sigmoid(policy.eot_head(encoded).squeeze(-1)) > args.eot_threshold
                else:
                    eot = torch.zeros(n, dtype=torch.bool)

                for i in range(n):
                    if not active[i]:
                        continue
                    if bool(eot[i]):
                        active[i] = False
                        continue
                    action = {
                        "xy": np.array([int(x[i]), int(y[i])], dtype=int),
                        "entity": int(ent[i]),
                        "direction": int(dir_[i]),
                        "item": int(item[i]),
                        "misc": int(misc[i]),
                    }
                    no, _r, term, trunc, info = envs[i].step(action)
                    last_thp[i] = float(info.get("throughput", 0.0))
                    last_dir[i] = float(info.get("tile_match_direction", 0.0))
                    if term or trunc:
                        active[i] = False
                    else:
                        obs[i] = no

            thps.extend(last_thp)
            dir_accs.extend(last_dir)
            for e in envs:
                e.close()

    if was_training:
        policy.train()

    mean_thp = float(np.mean(thps)) if thps else 0.0
    success = float(np.mean([1.0 if t >= 1.0 else 0.0 for t in thps])) if thps else 0.0
    return {
        "eval/throughput": mean_thp,
        "eval/success_rate": success,
        "eval/dir_acc_vs_ref": float(np.mean(dir_accs)) if dir_accs else 0.0,
        "eval/n": len(thps),
    }


def grpo_update(
    policy: AgentCNN,
    optimizer: optim.Optimizer,
    batch: RolloutBatch,
    A_N: torch.Tensor,
    args: GRPOArgs,
) -> dict:
    """One GRPO optimisation pass: update_epochs inner epochs of PPO-style
    clipping plus a k3 KL penalty toward the frozen reference.

    pi_theta_old is the rollout-time logp (batch.old_logp_N), fixed across the
    inner epochs; ratio = exp(new_logp - old_logp). The KL uses the k3
    estimator on the summed joint logps:

        kl = exp(ref - new) - (ref - new) - 1   (>= 0, estimates KL(pi||ref))

    where `new` carries gradient and the stored `ref` is constant. There is no
    critic and no minibatching — the batch is small (sim-bound), so each epoch
    is a full-batch step."""
    N = batch.obs_NCWH.shape[0]
    if N == 0:
        return {"loss/total": float("nan"), "grpo/n_transitions": 0}

    eps = args.clip_coef
    old_logp_N = batch.old_logp_N
    ref_logp_N = batch.ref_logp_N

    last: dict = {}
    for _epoch in range(args.update_epochs):
        _a, new_logp_N, entropy_N = policy_step(
            policy,
            batch.obs_NCWH,
            temperature=args.temperature,
            action_B6=batch.actions_N6,
        )
        logratio_N = new_logp_N - old_logp_N
        ratio_N = logratio_N.exp()

        # clipped surrogate (maximise advantage-weighted ratio => minimise -min)
        pg1 = -A_N * ratio_N
        pg2 = -A_N * torch.clamp(ratio_N, 1 - eps, 1 + eps)
        pg_loss = torch.max(pg1, pg2).mean()

        # k3 KL(pi_theta || pi_ref)
        ref_logratio_N = ref_logp_N - new_logp_N
        kl_N = ref_logratio_N.exp() - ref_logratio_N - 1.0
        kl_loss = kl_N.mean()

        entropy = entropy_N.mean()
        loss = pg_loss + args.beta_kl * kl_loss - args.ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()

        with torch.no_grad():
            # Schulman's low-variance approx of KL(old || new); ~0 early.
            approx_kl = ((ratio_N - 1) - logratio_N).mean()
            clipfrac = ((ratio_N - 1.0).abs() > eps).float().mean()
        last = {
            "loss/total": loss.item(),
            "loss/policy": pg_loss.item(),
            "loss/kl_ref": kl_loss.item(),
            "loss/entropy": entropy.item(),
            "grpo/ratio_mean": ratio_N.mean().item(),
            "grpo/clipfrac": clipfrac.item(),
            "grpo/approx_kl": approx_kl.item(),
            "grpo/grad_norm": float(grad_norm),
            "grpo/n_transitions": N,
        }
    return last


def train_grpo(args: GRPOArgs) -> AgentCNN:
    """GRPO finetuning entry point. Mirrors sft.train_sft's contract: callable
    directly with a GRPOArgs, returns the trained agent, and writes a
    checkpoint + summary JSON.

    NOTE: this is the skeleton (checkpoint/reference plumbing only). The
    rollout → advantage → update loop is added in subsequent commits."""
    t0 = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = _device()
    print(f"running GRPO on {device}")

    # AgentCNN needs a vector env to read grid size / catalog sizes at init.
    if GRPO_ENV_ID not in gym.registry:
        gym.register(id=GRPO_ENV_ID, entry_point=FactorioEnv)
    envs = gym.vector.SyncVectorEnv(
        [make_env(GRPO_ENV_ID, 0, False, args.size, "grpo")]
    )

    policy = _build_agent(args, envs, device)
    # Reference policy: a frozen copy of the SFT prior for KL regularization.
    ref = _build_agent(args, envs, device)
    ref.eval()
    ref.requires_grad_(False)
    envs.close()

    optimizer = optim.Adam(  # noqa: F841  (used by the training loop in later commits)
        policy.parameters(), lr=args.learning_rate, eps=args.adam_epsilon
    )

    run = None
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"grpo-{args.size}x{args.size}",
            group=args.wandb_group,
            tags=["grpo"] + (args.tags or []),
        )

    # ── (training loop added in later commits) ───────────────────────────

    torch.save(policy.state_dict(), args.checkpoint_path)
    runtime = time.time() - t0
    print(f"Checkpoint saved to {args.checkpoint_path}")

    summary = {
        "seed": args.seed,
        "size": args.size,
        "num_iterations": args.num_iterations,
        "num_grids": args.num_grids,
        "group_size": args.group_size,
        "start_from": args.start_from,
        "checkpoint_path": args.checkpoint_path,
        "runtime_seconds": round(runtime, 1),
        "wandb_url": run.url if run is not None else None,
    }
    summary_path = args.summary_path or str(Path(__file__).parent / "grpo_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    if args.track and run is not None:
        run.finish()

    return policy


if __name__ == "__main__":
    train_grpo(tyro.cli(GRPOArgs))
