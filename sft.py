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
import factorion_rs  # noqa: E402
from factorion import (  # noqa: E402
    Channel,
    LessonKind,
    entities,
    generate_lesson,
)

from ppo import AgentCNN, FactorioEnv, make_env  # noqa: E402


def extract_expert_actions(solved_CWH, task_CWH):
    """Extract (state, action) pairs by diffing solved vs task worlds.

    Returns list of (state_CWH, tile_idx, entity_id, direction_id, item_id,
    misc_id, valid_tile_mask) tuples. The agent's action covers all four
    placement channels because the env (ppo.py FactorioEnv.step) rejects
    placements with mismatched channels — e.g. an underground belt without
    misc=DOWN/UP, or an assembling_machine_1 without an item (= recipe).

    The valid_tile_mask is a flat binary tensor marking ALL tiles that still
    need entities at this step — not just the one chosen. This allows
    multi-label tile loss to avoid penalizing the model for predicting a
    valid-but-different tile.

    Multi-tile entities (e.g. splitters) emit a single pair at the anchor
    tile, not one per occupied cell — placing the anchor fills the whole
    footprint at execution time.

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

    # Group multi-tile entities so each placement is one (anchor) action,
    # not one per occupied cell. Walk in raster order; when we encounter a
    # multi-tile entity, claim its anchor and mark the rest as secondaries
    # to skip.
    diff_set = {tuple(loc) for loc in diff_locs}
    secondary_tiles: set[tuple[int, int]] = set()
    for x, y in sorted(diff_set):
        if (x, y) in secondary_tiles:
            continue
        ent_val = int(solved_CWH[Channel.ENTITIES.value, x, y])
        proto = entities[ent_val]
        if proto.width == 1 and proto.height == 1:
            continue
        d_val = int(solved_CWH[Channel.DIRECTION.value, x, y])
        tile_list = factorion_rs.py_entity_tiles(x, y, d_val, proto.width, proto.height)
        if tile_list is None:
            continue
        anchor = tuple(tile_list[0])
        for tx, ty in tile_list:
            if (tx, ty) != anchor:
                secondary_tiles.add((tx, ty))

    diff_locs = [loc for loc in diff_locs if tuple(loc) not in secondary_tiles]

    # Shuffle for diversity
    random.shuffle(diff_locs)

    state = task_CWH.clone()
    pairs = []

    # Build per-step valid_mask = all remaining anchor tiles. We pop as we go.
    remaining_locs = [tuple(loc) for loc in diff_locs]

    for step, (x, y) in enumerate(diff_locs):
        valid_mask = torch.zeros(W * H)
        for rx, ry in remaining_locs[step:]:
            valid_mask[rx * H + ry] = 1.0

        obs = state.clone()
        tile_idx = x * H + y
        entity_id = int(solved_CWH[Channel.ENTITIES.value, x, y])
        direction_id = int(solved_CWH[Channel.DIRECTION.value, x, y])
        item_id = int(solved_CWH[Channel.ITEMS.value, x, y])
        misc_id = int(solved_CWH[Channel.MISC.value, x, y])

        pairs.append((obs, tile_idx, entity_id, direction_id, item_id, misc_id, valid_mask))

        # Apply action: copy the entity's full footprint from solved, not
        # just the anchor cell, so the observation reflects what placing
        # the anchor does at execution time.
        proto = entities[entity_id]
        if proto.width == 1 and proto.height == 1:
            tiles_to_apply = [(x, y)]
        else:
            tile_list = factorion_rs.py_entity_tiles(x, y, direction_id, proto.width, proto.height)
            tiles_to_apply = [tuple(t) for t in tile_list] if tile_list is not None else [(x, y)]
        for tx, ty in tiles_to_apply:
            for ch in range(C):
                state[ch, tx, ty] = solved_CWH[ch, tx, ty]

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
    """Generate SFT dataset from expert demonstrations.

    Samples uniformly across every value of `LessonKind` (auto-discovered),
    so adding a new lesson kind to the enum is automatically picked up.
    Per-kind sample/lesson counts are printed to stdout for visibility.

    Also returns a per-pair lesson_seed tensor so callers can split
    train/val at the lesson level (a single lesson with level=L produces
    ~L pairs that all share its seed; splitting at the pair level leaks
    factories across the split).
    """
    max_level = args.max_level if args.max_level > 0 else 2 * args.size
    kinds = list(LessonKind)

    all_obs = []
    all_tile_idx = []
    all_entity = []
    all_direction = []
    all_item = []
    all_misc = []
    all_valid_masks = []
    all_lesson_seeds = []
    all_lesson_kinds = []
    kind_samples = {k.name: 0 for k in kinds}
    kind_lessons = {k.name: 0 for k in kinds}

    seed = args.seed
    samples_so_far = 0

    while samples_so_far < args.num_samples:
        kind = random.choice(kinds)
        # Always blank at the cap. Per-lesson sampling of level was wasteful:
        # extract_expert_actions already produces the full progression
        # (1 entity placed → 2 → … → all-blanked) within a single lesson,
        # so a lower level just means fewer pairs per factory. Pinning to
        # max_level gives us maximum training data per factory layout.
        level = max_level
        seed += 1

        try:
            solved, _ = generate_lesson(
                size=args.size,
                kind=kind,
                num_missing_entities=0,
                seed=seed,
            )
            task, _ = generate_lesson(
                size=args.size,
                kind=kind,
                num_missing_entities=level,
                seed=seed,
            )
        except Exception:
            continue

        kind_lessons[kind.name] += 1
        pairs = extract_expert_actions(solved, task)
        for obs, tile_idx, entity_id, direction_id, item_id, misc_id, valid_mask in pairs:
            all_obs.append(obs)
            all_tile_idx.append(tile_idx)
            all_entity.append(entity_id)
            all_direction.append(direction_id)
            all_item.append(item_id)
            all_misc.append(misc_id)
            all_valid_masks.append(valid_mask)
            all_lesson_seeds.append(seed)
            all_lesson_kinds.append(kind.value)
            kind_samples[kind.name] += 1
            samples_so_far += 1
            if samples_so_far >= args.num_samples:
                break

    print("Per-kind breakdown:")
    name_w = max(len(k) for k in kind_samples)
    for name in sorted(kind_samples):
        print(f"  {name:<{name_w}}  samples={kind_samples[name]:>6}  lessons={kind_lessons[name]:>6}")

    obs_tensor = torch.stack(all_obs)
    tile_tensor = torch.tensor(all_tile_idx, dtype=torch.long)
    ent_tensor = torch.tensor(all_entity, dtype=torch.long)
    dir_tensor = torch.tensor(all_direction, dtype=torch.long)
    item_tensor = torch.tensor(all_item, dtype=torch.long)
    misc_tensor = torch.tensor(all_misc, dtype=torch.long)
    mask_tensor = torch.stack(all_valid_masks)
    seed_tensor = torch.tensor(all_lesson_seeds, dtype=torch.long)
    kind_tensor = torch.tensor(all_lesson_kinds, dtype=torch.long)

    return (
        obs_tensor, tile_tensor, ent_tensor, dir_tensor,
        item_tensor, misc_tensor, mask_tensor, seed_tensor, kind_tensor,
    )


def train_sft(args: SFTArgs):
    """Main SFT training loop."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Generating {args.num_samples} expert demonstrations...")
    t0 = time.time()
    (
        obs, tiles, ents, dirs, items_t, miscs_t,
        valid_masks, lesson_seeds, lesson_kinds,
    ) = generate_dataset(args)
    print(f"Generated {len(obs)} samples in {time.time() - t0:.1f}s")

    # Train/val split at the LESSON SEED level. A pair-level random split
    # would leak factories: a lesson at level=L produces ~L pairs that all
    # share the same factory geometry, just at different intermediate
    # states. Splitting on pairs would put some of those intermediates in
    # train and others in val, so val acc would measure "complete a factory
    # whose other partial states you've trained on", not "complete a
    # factory you've never seen". Splitting on seed guarantees val
    # factories are entirely held out.
    unique_seeds = torch.unique(lesson_seeds)
    n_seeds = len(unique_seeds)
    n_val_seeds = max(1, int(n_seeds * args.val_frac))
    seed_perm = unique_seeds[torch.randperm(n_seeds)]
    val_seeds = set(seed_perm[:n_val_seeds].tolist())

    val_mask = torch.tensor([s.item() in val_seeds for s in lesson_seeds])
    val_idx = val_mask.nonzero(as_tuple=False).squeeze(-1)
    train_idx = (~val_mask).nonzero(as_tuple=False).squeeze(-1)
    print(
        f"Train/val split at seed level: {len(train_idx)} train pairs from "
        f"{n_seeds - n_val_seeds} lessons, {len(val_idx)} val pairs from "
        f"{n_val_seeds} lessons (no factory overlap)"
    )

    train_ds = TensorDataset(
        obs[train_idx], tiles[train_idx], ents[train_idx], dirs[train_idx],
        items_t[train_idx], miscs_t[train_idx], valid_masks[train_idx],
        lesson_kinds[train_idx],
    )
    val_ds = TensorDataset(
        obs[val_idx], tiles[val_idx], ents[val_idx], dirs[val_idx],
        items_t[val_idx], miscs_t[val_idx], valid_masks[val_idx],
        lesson_kinds[val_idx],
    )
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
    # Per-sample variants used in the val loop so per-LessonKind losses can
    # be aggregated without re-running the forward pass.
    ce_loss_none = nn.CrossEntropyLoss(reduction="none")
    bce_loss_none = nn.BCEWithLogitsLoss(reduction="none")

    # Map kind value -> name so per-kind dict keys read as "MOVE_ONE_ITEM"
    # instead of "0" both in print() lines and in wandb panel titles.
    kind_names = {k.value: k.name for k in LessonKind}

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
        # Dataset composition is a one-shot constant for the run — write it
        # to summary (which keeps the value visible in the run sidebar)
        # rather than wandb.log (which would create a flat plottable line).
        from collections import Counter, defaultdict
        samples_by_kind = Counter(kind_names[k] for k in lesson_kinds.tolist())
        seeds_by_kind: dict[str, set] = defaultdict(set)
        for s, k in zip(lesson_seeds.tolist(), lesson_kinds.tolist()):
            seeds_by_kind[kind_names[k]].add(s)
        for k in LessonKind:
            run.summary[f"dataset/samples/{k.name}"] = samples_by_kind.get(k.name, 0)
            run.summary[f"dataset/lessons/{k.name}"] = len(seeds_by_kind.get(k.name, set()))

    best_val_acc = 0.0
    val_loss = 0.0
    val_tile_acc = 0.0
    val_ent_acc = 0.0
    val_dir_acc = 0.0
    val_item_acc = 0.0
    val_misc_acc = 0.0
    print(f"Training for {args.epochs} epochs on {device}...")

    for epoch in range(1, args.epochs + 1):
        agent.train()
        train_loss = 0.0
        train_loss_tile = 0.0
        train_loss_ent = 0.0
        train_loss_dir = 0.0
        train_loss_item = 0.0
        train_loss_misc = 0.0
        train_correct = 0
        train_total = 0

        # batch_kind is carried through the loader so val can aggregate
        # per-kind metrics; we ignore it in the train pass.
        for batch in train_loader:
            (
                batch_obs, batch_tile, batch_ent, batch_dir,
                batch_item, batch_misc, batch_mask, _batch_kind,
            ) = batch
            batch_obs = batch_obs.float().to(device)
            batch_tile = batch_tile.to(device)
            batch_ent = batch_ent.to(device)
            batch_dir = batch_dir.to(device)
            batch_item = batch_item.to(device)
            batch_misc = batch_misc.to(device)
            batch_mask = batch_mask.to(device)

            encoded = agent.encoder(batch_obs)
            B = encoded.shape[0]

            # Tile logits — use BCE with multi-label mask so ALL valid
            # tiles are rewarded, not just the randomly-chosen one
            tile_logits = agent.tile_logits(encoded).reshape(B, -1)
            loss_tile = bce_loss(tile_logits, batch_mask)

            # Extract features at target tile for entity/direction/item/misc heads
            x_B = batch_tile // agent.height
            y_B = batch_tile % agent.height
            batch_idx = torch.arange(B, device=device)
            tile_features = encoded[batch_idx, :, x_B, y_B]

            ent_logits = agent.ent_head(tile_features)
            dir_logits = agent.dir_head(tile_features)
            item_logits = agent.item_head(tile_features)
            misc_logits = agent.misc_head(tile_features)
            loss_ent = ce_loss(ent_logits, batch_ent)
            loss_dir = ce_loss(dir_logits, batch_dir)
            loss_item = ce_loss(item_logits, batch_item)
            loss_misc = ce_loss(misc_logits, batch_misc)

            loss = loss_tile + loss_ent + loss_dir + loss_item + loss_misc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * B
            train_loss_tile += loss_tile.item() * B
            train_loss_ent += loss_ent.item() * B
            train_loss_dir += loss_dir.item() * B
            train_loss_item += loss_item.item() * B
            train_loss_misc += loss_misc.item() * B
            # Whole-action accuracy: every action component correct.
            pred_tile = tile_logits.argmax(dim=1)
            tile_hit = batch_mask[batch_idx, pred_tile] > 0
            pred_ent = ent_logits.argmax(dim=1)
            pred_dir = dir_logits.argmax(dim=1)
            pred_item = item_logits.argmax(dim=1)
            pred_misc = misc_logits.argmax(dim=1)
            correct = (
                tile_hit
                & (pred_ent == batch_ent)
                & (pred_dir == batch_dir)
                & (pred_item == batch_item)
                & (pred_misc == batch_misc)
            )
            train_correct += correct.sum().item()
            train_total += B

        train_loss /= train_total
        train_loss_tile /= train_total
        train_loss_ent /= train_total
        train_loss_dir /= train_total
        train_loss_item /= train_total
        train_loss_misc /= train_total
        train_acc = train_correct / train_total

        # Validation
        agent.eval()
        val_loss = 0.0
        val_loss_tile = 0.0
        val_loss_ent = 0.0
        val_loss_dir = 0.0
        val_loss_item = 0.0
        val_loss_misc = 0.0
        val_correct = 0
        val_tile_correct = 0
        val_ent_correct = 0
        val_dir_correct = 0
        val_item_correct = 0
        val_misc_correct = 0
        val_total = 0

        # Per-LessonKind accumulators. Indexed by kind name to make wandb
        # panel keys read as "val/MOVE_ONE_ITEM/acc" rather than enum ints.
        per_kind_n = {k.name: 0 for k in LessonKind}
        per_kind_correct = {k.name: 0 for k in LessonKind}
        per_kind_tile_correct = {k.name: 0 for k in LessonKind}
        per_kind_ent_correct = {k.name: 0 for k in LessonKind}
        per_kind_dir_correct = {k.name: 0 for k in LessonKind}
        per_kind_item_correct = {k.name: 0 for k in LessonKind}
        per_kind_misc_correct = {k.name: 0 for k in LessonKind}
        per_kind_loss_sum = {k.name: 0.0 for k in LessonKind}

        with torch.no_grad():
            for batch in val_loader:
                (
                    batch_obs, batch_tile, batch_ent, batch_dir,
                    batch_item, batch_misc, batch_mask, batch_kind,
                ) = batch
                batch_obs = batch_obs.float().to(device)
                batch_tile = batch_tile.to(device)
                batch_ent = batch_ent.to(device)
                batch_dir = batch_dir.to(device)
                batch_item = batch_item.to(device)
                batch_misc = batch_misc.to(device)
                batch_mask = batch_mask.to(device)
                batch_kind = batch_kind.to(device)

                encoded = agent.encoder(batch_obs)
                B = encoded.shape[0]

                tile_logits = agent.tile_logits(encoded).reshape(B, -1)
                # Per-sample losses: needed so we can sum them within each
                # LessonKind. mean over the tile axis matches the scale of
                # bce_loss(reduction="mean"), keeping val/loss_tile
                # comparable to train/loss_tile.
                loss_tile_per = bce_loss_none(tile_logits, batch_mask).mean(dim=1)

                x_B = batch_tile // agent.height
                y_B = batch_tile % agent.height
                batch_idx = torch.arange(B, device=device)
                tile_features = encoded[batch_idx, :, x_B, y_B]

                ent_logits = agent.ent_head(tile_features)
                dir_logits = agent.dir_head(tile_features)
                item_logits = agent.item_head(tile_features)
                misc_logits = agent.misc_head(tile_features)
                loss_ent_per = ce_loss_none(ent_logits, batch_ent)
                loss_dir_per = ce_loss_none(dir_logits, batch_dir)
                loss_item_per = ce_loss_none(item_logits, batch_item)
                loss_misc_per = ce_loss_none(misc_logits, batch_misc)

                loss_per_sample = (
                    loss_tile_per + loss_ent_per + loss_dir_per
                    + loss_item_per + loss_misc_per
                )
                val_loss += loss_per_sample.sum().item()
                val_loss_tile += loss_tile_per.sum().item()
                val_loss_ent += loss_ent_per.sum().item()
                val_loss_dir += loss_dir_per.sum().item()
                val_loss_item += loss_item_per.sum().item()
                val_loss_misc += loss_misc_per.sum().item()

                pred_tile = tile_logits.argmax(dim=1)
                tile_hit = batch_mask[batch_idx, pred_tile] > 0
                pred_ent = ent_logits.argmax(dim=1)
                pred_dir = dir_logits.argmax(dim=1)
                pred_item = item_logits.argmax(dim=1)
                pred_misc = misc_logits.argmax(dim=1)
                ent_correct_per = (pred_ent == batch_ent)
                dir_correct_per = (pred_dir == batch_dir)
                item_correct_per = (pred_item == batch_item)
                misc_correct_per = (pred_misc == batch_misc)
                correct_per = (
                    tile_hit & ent_correct_per & dir_correct_per
                    & item_correct_per & misc_correct_per
                )
                val_correct += correct_per.sum().item()
                val_tile_correct += tile_hit.sum().item()
                val_ent_correct += ent_correct_per.sum().item()
                val_dir_correct += dir_correct_per.sum().item()
                val_item_correct += item_correct_per.sum().item()
                val_misc_correct += misc_correct_per.sum().item()
                val_total += B

                # Per-kind aggregation: one bool-mask per kind appearing in
                # this batch. unique() keeps this O(num_kinds_in_batch) so
                # adding new LessonKind enum values has no cost here.
                for k_val in batch_kind.unique().tolist():
                    k_name = kind_names[k_val]
                    mask_k = (batch_kind == k_val)
                    per_kind_n[k_name] += int(mask_k.sum().item())
                    per_kind_correct[k_name] += int(correct_per[mask_k].sum().item())
                    per_kind_tile_correct[k_name] += int(tile_hit[mask_k].sum().item())
                    per_kind_ent_correct[k_name] += int(ent_correct_per[mask_k].sum().item())
                    per_kind_dir_correct[k_name] += int(dir_correct_per[mask_k].sum().item())
                    per_kind_item_correct[k_name] += int(item_correct_per[mask_k].sum().item())
                    per_kind_misc_correct[k_name] += int(misc_correct_per[mask_k].sum().item())
                    per_kind_loss_sum[k_name] += loss_per_sample[mask_k].sum().item()

        val_loss /= val_total
        val_loss_tile /= val_total
        val_loss_ent /= val_total
        val_loss_dir /= val_total
        val_loss_item /= val_total
        val_loss_misc /= val_total
        val_acc = val_correct / val_total
        val_tile_acc = val_tile_correct / val_total
        val_ent_acc = val_ent_correct / val_total
        val_dir_acc = val_dir_correct / val_total
        val_item_acc = val_item_correct / val_total
        val_misc_acc = val_misc_correct / val_total

        # Build per-kind metric dict for both stdout and wandb. Skip kinds
        # absent from the val split (e.g. small datasets where some kinds
        # land entirely in train).
        per_kind_metrics: dict[str, float] = {}
        for k in LessonKind:
            n = per_kind_n[k.name]
            if n == 0:
                continue
            per_kind_metrics[f"val/{k.name}/n"] = n
            per_kind_metrics[f"val/{k.name}/loss"] = per_kind_loss_sum[k.name] / n
            per_kind_metrics[f"val/{k.name}/acc"] = per_kind_correct[k.name] / n
            per_kind_metrics[f"val/{k.name}/tile_acc"] = per_kind_tile_correct[k.name] / n
            per_kind_metrics[f"val/{k.name}/ent_acc"] = per_kind_ent_correct[k.name] / n
            per_kind_metrics[f"val/{k.name}/dir_acc"] = per_kind_dir_correct[k.name] / n
            per_kind_metrics[f"val/{k.name}/item_acc"] = per_kind_item_correct[k.name] / n
            per_kind_metrics[f"val/{k.name}/misc_acc"] = per_kind_misc_correct[k.name] / n

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"(tile={val_tile_acc:.3f} ent={val_ent_acc:.3f} dir={val_dir_acc:.3f} "
            f"item={val_item_acc:.3f} misc={val_misc_acc:.3f})"
        )
        if per_kind_metrics:
            kind_summary = "  ".join(
                f"{k.name}={per_kind_correct[k.name] / per_kind_n[k.name]:.3f}"
                for k in LessonKind
                if per_kind_n[k.name] > 0
            )
            print(f"  per-kind val_acc: {kind_summary}")

        if args.track and run is not None:
            run.log({
                "train/loss": train_loss,
                "train/loss_tile": train_loss_tile,
                "train/loss_ent": train_loss_ent,
                "train/loss_dir": train_loss_dir,
                "train/loss_item": train_loss_item,
                "train/loss_misc": train_loss_misc,
                "train/acc": train_acc,
                "val/loss": val_loss,
                "val/loss_tile": val_loss_tile,
                "val/loss_ent": val_loss_ent,
                "val/loss_dir": val_loss_dir,
                "val/loss_item": val_loss_item,
                "val/loss_misc": val_loss_misc,
                "val/acc": val_acc,
                "val/tile_acc": val_tile_acc,
                "val/ent_acc": val_ent_acc,
                "val/dir_acc": val_dir_acc,
                "val/item_acc": val_item_acc,
                "val/misc_acc": val_misc_acc,
                "train/epoch": epoch,
                **per_kind_metrics,
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
        "val_item_acc": round(val_item_acc, 4),
        "val_misc_acc": round(val_misc_acc, 4),
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
