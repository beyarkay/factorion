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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
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
