"""Supervised Fine-Tuning (SFT) pre-training for AgentCNN.

Uses expert demonstrations from generate_lesson() to teach basic belt
placement patterns before RL training via PPO.

Usage:
    python sft.py --size 8 --num-samples 50000 --epochs 30
    python ppo.py --start_from sft_checkpoint.pt ...
"""

import json
import os
import random
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import factorion  # noqa: E402

from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402


# ── Extract marimo cell objects ──────────────────────────────────────────────
_, _objs = factorion.datatypes.run()
_, _fns = factorion.functions.run()

Channel = _objs["Channel"]
Direction = _objs["Direction"]
LessonKind = _objs["LessonKind"]
generate_lesson = _fns["generate_lesson"]
str2ent = _fns["str2ent"]


def extract_expert_actions(solved_CWH, task_CWH):
    """Extract (state, action) pairs by diffing solved vs task worlds.

    Returns list of (state_CWH, tile_idx, entity_id, direction_id,
    valid_tile_mask) tuples. The valid_tile_mask is a flat binary tensor
    marking ALL tiles that still need entities at this step — not just the
    one chosen. This allows multi-label tile loss to avoid penalizing the
    model for predicting a valid-but-different tile.

    Actions are applied sequentially in random order, so intermediate states
    reflect realistic observations the agent would see.
    """
    C, W, H = solved_CWH.shape
    solved_ent = solved_CWH[Channel.ENTITIES.value]
    task_ent = task_CWH[Channel.ENTITIES.value]

    # Find tiles that differ: solved has entity, task has empty
    diff_mask = (solved_ent != task_ent)
    diff_locs = diff_mask.nonzero(as_tuple=False).tolist()

    if len(diff_locs) == 0:
        return []

    # Shuffle for diversity
    random.shuffle(diff_locs)

    state = task_CWH.clone()
    remaining = set(range(len(diff_locs)))
    pairs = []

    for i, (x, y) in enumerate(diff_locs):
        # Build mask of all remaining valid tiles (including this one)
        valid_mask = torch.zeros(W * H)
        for j in remaining:
            rx, ry = diff_locs[j]
            valid_mask[rx * H + ry] = 1.0

        obs = state.clone()
        tile_idx = x * H + y
        entity_id = int(solved_CWH[Channel.ENTITIES.value, x, y])
        direction_id = int(solved_CWH[Channel.DIRECTION.value, x, y])

        pairs.append((obs, tile_idx, entity_id, direction_id, valid_mask))

        # Apply action to state (so next observation reflects this placement)
        for ch in range(C):
            state[ch, x, y] = solved_CWH[ch, x, y]
        remaining.discard(i)

    return pairs


@dataclass
class SFTArgs:
    seed: int = 1
    """random seed"""
    size: int = 8
    """grid size (width and height)"""
    num_samples: int = 50000
    """number of (state, action) pairs to generate"""
    max_level: int = 0
    """max curriculum level (0 = auto: 2*size)"""
    epochs: int = 30
    """number of training epochs"""
    batch_size: int = 512
    """training batch size"""
    lr: float = 1e-3
    """learning rate"""
    val_frac: float = 0.1
    """fraction of data for validation"""
    checkpoint_path: str = "sft_checkpoint.pt"
    """path to save the trained model"""
    chan1: int = 48
    """CNN encoder channel 1"""
    chan2: int = 48
    """CNN encoder channel 2"""
    chan3: int = 48
    """CNN encoder channel 3"""
    flat_dim: int = 128
    """flat dim (unused, kept for compat with AgentCNN)"""
    tile_head_std: float = 0.06503
    """tile head init std"""
    track: bool = False
    """track with W&B"""
    wandb_project_name: str = "factorion"
    """W&B project name"""
    wandb_entity: Optional[str] = None
    """W&B entity (team or user)"""
    wandb_group: Optional[str] = None
    """W&B run group name"""
    tags: typing.Optional[typing.List[str]] = None
    """Tags to apply to the wandb run"""
    summary_path: Optional[str] = None
    """path to write summary JSON (default: sft_summary.json next to sft.py)"""


def generate_dataset(args: SFTArgs):
    """Generate SFT dataset from expert demonstrations."""
    max_level = args.max_level if args.max_level > 0 else 2 * args.size

    all_obs = []
    all_tile_idx = []
    all_entity = []
    all_direction = []
    all_valid_masks = []

    seed = args.seed
    samples_so_far = 0

    while samples_so_far < args.num_samples:
        level = random.randint(1, max_level)
        seed += 1

        try:
            solved, _ = generate_lesson(
                size=args.size,
                kind=LessonKind.MOVE_ONE_ITEM,
                num_missing_entities=0,
                seed=seed,
            )
            task, _ = generate_lesson(
                size=args.size,
                kind=LessonKind.MOVE_ONE_ITEM,
                num_missing_entities=level,
                seed=seed,
            )
        except Exception:
            continue

        pairs = extract_expert_actions(solved, task)
        for obs, tile_idx, entity_id, direction_id, valid_mask in pairs:
            all_obs.append(obs)
            all_tile_idx.append(tile_idx)
            all_entity.append(entity_id)
            all_direction.append(direction_id)
            all_valid_masks.append(valid_mask)
            samples_so_far += 1
            if samples_so_far >= args.num_samples:
                break

    obs_tensor = torch.stack(all_obs)
    tile_tensor = torch.tensor(all_tile_idx, dtype=torch.long)
    ent_tensor = torch.tensor(all_entity, dtype=torch.long)
    dir_tensor = torch.tensor(all_direction, dtype=torch.long)
    mask_tensor = torch.stack(all_valid_masks)

    return obs_tensor, tile_tensor, ent_tensor, dir_tensor, mask_tensor


def train_sft(args: SFTArgs):
    """Main SFT training loop."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Generating {args.num_samples} expert demonstrations...")
    t0 = time.time()
    obs, tiles, ents, dirs, valid_masks = generate_dataset(args)
    print(f"Generated {len(obs)} samples in {time.time() - t0:.1f}s")

    # Train/val split
    n = len(obs)
    n_val = max(1, int(n * args.val_frac))
    n_train = n - n_val
    perm = torch.randperm(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = TensorDataset(obs[train_idx], tiles[train_idx], ents[train_idx], dirs[train_idx], valid_masks[train_idx])
    val_ds = TensorDataset(obs[val_idx], tiles[val_idx], ents[val_idx], dirs[val_idx], valid_masks[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # Create agent via a temporary env (AgentCNN needs envs for init)
    env_id = "factorion/FactorioEnv-v0-sft"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point=FactorioEnv)
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, args.size, "sft")])

    agent = AgentCNN(
        envs,
        chan1=args.chan1,
        chan2=args.chan2,
        chan3=args.chan3,
        flat_dim=args.flat_dim,
        tile_head_std=args.tile_head_std,
    )
    envs.close()

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    agent.to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()

    run = None
    if args.track:
        import wandb
        sft_tags = ["sft"] + (args.tags or [])
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=f"sft-{args.size}x{args.size}",
            group=args.wandb_group,
            tags=sft_tags,
        )

    best_val_acc = 0.0
    val_loss = 0.0
    val_tile_acc = 0.0
    val_ent_acc = 0.0
    val_dir_acc = 0.0
    print(f"Training for {args.epochs} epochs on {device}...")

    for epoch in range(1, args.epochs + 1):
        agent.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_obs, batch_tile, batch_ent, batch_dir, batch_mask in train_loader:
            batch_obs = batch_obs.float().to(device)
            batch_tile = batch_tile.to(device)
            batch_ent = batch_ent.to(device)
            batch_dir = batch_dir.to(device)
            batch_mask = batch_mask.to(device)

            encoded = agent.encoder(batch_obs)
            B = encoded.shape[0]

            # Tile logits — use BCE with multi-label mask so ALL valid
            # tiles are rewarded, not just the randomly-chosen one
            tile_logits = agent.tile_logits(encoded).reshape(B, -1)
            loss_tile = bce_loss(tile_logits, batch_mask)

            # Extract features at target tile for entity/direction heads
            x_B = batch_tile // agent.height
            y_B = batch_tile % agent.height
            batch_idx = torch.arange(B, device=device)
            tile_features = encoded[batch_idx, :, x_B, y_B]

            ent_logits = agent.ent_head(tile_features)
            dir_logits = agent.dir_head(tile_features)
            loss_ent = ce_loss(ent_logits, batch_ent)
            loss_dir = ce_loss(dir_logits, batch_dir)

            loss = loss_tile + loss_ent + loss_dir

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B
            # Tile accuracy: did the model predict ANY valid tile?
            pred_tile = tile_logits.argmax(dim=1)
            tile_hit = batch_mask[batch_idx, pred_tile] > 0
            pred_ent = ent_logits.argmax(dim=1)
            pred_dir = dir_logits.argmax(dim=1)
            correct = (tile_hit & (pred_ent == batch_ent) & (pred_dir == batch_dir))
            train_correct += correct.sum().item()
            train_total += B

        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation
        agent.eval()
        val_loss = 0.0
        val_correct = 0
        val_tile_correct = 0
        val_ent_correct = 0
        val_dir_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_obs, batch_tile, batch_ent, batch_dir, batch_mask in val_loader:
                batch_obs = batch_obs.float().to(device)
                batch_tile = batch_tile.to(device)
                batch_ent = batch_ent.to(device)
                batch_dir = batch_dir.to(device)
                batch_mask = batch_mask.to(device)

                encoded = agent.encoder(batch_obs)
                B = encoded.shape[0]

                tile_logits = agent.tile_logits(encoded).reshape(B, -1)
                loss_tile = bce_loss(tile_logits, batch_mask)

                x_B = batch_tile // agent.height
                y_B = batch_tile % agent.height
                batch_idx = torch.arange(B, device=device)
                tile_features = encoded[batch_idx, :, x_B, y_B]

                ent_logits = agent.ent_head(tile_features)
                dir_logits = agent.dir_head(tile_features)
                loss_ent = ce_loss(ent_logits, batch_ent)
                loss_dir = ce_loss(dir_logits, batch_dir)

                loss = loss_tile + loss_ent + loss_dir
                val_loss += loss.item() * B

                pred_tile = tile_logits.argmax(dim=1)
                tile_hit = batch_mask[batch_idx, pred_tile] > 0
                pred_ent = ent_logits.argmax(dim=1)
                pred_dir = dir_logits.argmax(dim=1)
                correct = (tile_hit & (pred_ent == batch_ent) & (pred_dir == batch_dir))
                val_correct += correct.sum().item()
                val_tile_correct += tile_hit.sum().item()
                val_ent_correct += (pred_ent == batch_ent).sum().item()
                val_dir_correct += (pred_dir == batch_dir).sum().item()
                val_total += B

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_tile_acc = val_tile_correct / val_total
        val_ent_acc = val_ent_correct / val_total
        val_dir_acc = val_dir_correct / val_total

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"(tile={val_tile_acc:.3f} ent={val_ent_acc:.3f} dir={val_dir_acc:.3f})"
        )

        if args.track and run is not None:
            run.log({
                "train/loss": train_loss,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/acc": val_acc,
                "val/tile_acc": val_tile_acc,
                "val/ent_acc": val_ent_acc,
                "val/dir_acc": val_dir_acc,
                "train/epoch": epoch,
            })

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(agent.state_dict(), args.checkpoint_path)
            print(f"  -> Saved best checkpoint ({val_acc:.3f})")

    total_time = time.time() - t0
    print(f"\nBest validation accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {args.checkpoint_path}")

    # Write summary JSON (total_time includes dataset generation + training)
    summary = {
        "best_val_acc": round(best_val_acc, 4),
        "val_tile_acc": round(val_tile_acc, 4),
        "val_ent_acc": round(val_ent_acc, 4),
        "val_dir_acc": round(val_dir_acc, 4),
        "val_loss": round(val_loss, 4),
        "num_samples": args.num_samples,
        "epochs": args.epochs,
        "size": args.size,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "runtime_seconds": round(total_time, 1),
        "checkpoint_path": args.checkpoint_path,
        "wandb_url": run.url if run is not None else None,
    }
    summary_path = args.summary_path or str(Path(__file__).parent / "sft_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")

    if args.track and run is not None:
        run.finish()

    return agent


if __name__ == "__main__":
    args = tyro.cli(SFTArgs)
    train_sft(args)
