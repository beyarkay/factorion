"""Supervised Fine-Tuning (SFT) pre-training for AgentCNN.

Uses expert demonstrations from generate_lesson() to teach basic belt
placement patterns before RL training via PPO.

Usage:
    python sft.py --size 11 --num-samples 5000000 --epochs 1
    python ppo.py --start_from sft_checkpoint.pt ...
"""

import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import TypedDict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
from torch.utils.data import DataLoader, IterableDataset, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import factorion_rs  # noqa: E402
from factorion import (  # noqa: E402
    Channel,
    Direction,
    LessonKind,
    blank_entities,
    build_factory,
    entities,
    str2ent,
)

from ppo import (  # noqa: E402
    AgentCNN,
    FactorioEnv,
    make_env,
    layers_from_args,
    assert_device_ok,
    _resolve_start_from,
    _CH_ENT,
    _CH_ITEMS,
    _CH_FOOTPRINT,
    _EMPTY_ENT_ID,
    _ASM_MACHINE_ENT_ID,
    _FOOTPRINT_UNAVAILABLE,
)
from training_config import SftArgs  # noqa: E402


def extract_expert_actions(solved_CWH, task_CWH):
    """Extract (state, action) pairs by diffing solved vs task worlds.

    Returns list of (state_CWH, tile_idx, entity_id, direction_id, item_id,
    misc_id, valid_tile_mask, eot) tuples. The agent's action covers all
    four placement channels because the env (ppo.py FactorioEnv.step)
    rejects placements with mismatched channels — e.g. an underground belt
    without misc=DOWN/UP, or an assembling_machine_1 without an item
    (= recipe).

    The valid_tile_mask is a flat binary tensor marking ALL tiles that still
    need entities at this step — not just the one chosen. This allows
    multi-label tile loss to avoid penalizing the model for predicting a
    valid-but-different tile.

    Multi-tile entities (e.g. splitters) emit a single pair at the anchor
    tile, not one per occupied cell — placing the anchor fills the whole
    footprint at execution time.

    Actions are applied sequentially in random order, so intermediate states
    reflect realistic observations the agent would see.

    `eot` (end-of-turn) is 0 for every placement step (the factory still
    has entities to place) and 1 for a single terminal pair appended at
    the end (state is the fully-solved factory). Placement targets on the
    terminal pair are sentinel zeros — the SFT loop masks placement losses
    on eot=1 samples.
    """
    C, W, H = solved_CWH.shape
    solved_ent = solved_CWH[Channel.ENTITIES.value]
    task_ent = task_CWH[Channel.ENTITIES.value]

    # Find tiles that differ: solved has entity, task has empty
    diff_mask = solved_ent != task_ent
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
        valid_mask = torch.zeros(W * H, dtype=torch.bool)
        for rx, ry in remaining_locs[step:]:
            valid_mask[rx * H + ry] = True

        # uint8 obs / bool mask: ~8x smaller than int64/float32, cast at use.
        obs = state.to(torch.uint8)
        tile_idx = x * H + y
        entity_id = int(solved_CWH[Channel.ENTITIES.value, x, y])
        direction_id = int(solved_CWH[Channel.DIRECTION.value, x, y])
        item_id = int(solved_CWH[Channel.ITEMS.value, x, y])
        misc_id = int(solved_CWH[Channel.MISC.value, x, y])

        pairs.append(
            (obs, tile_idx, entity_id, direction_id, item_id, misc_id, valid_mask, 0)
        )

        # Apply action: copy the entity's full footprint from solved, not
        # just the anchor cell, so the observation reflects what placing
        # the anchor does at execution time.
        proto = entities[entity_id]
        if proto.width == 1 and proto.height == 1:
            tiles_to_apply = [(x, y)]
        else:
            tile_list = factorion_rs.py_entity_tiles(
                x, y, direction_id, proto.width, proto.height
            )
            tiles_to_apply = (
                [tuple(t) for t in tile_list] if tile_list is not None else [(x, y)]
            )
        for tx, ty in tiles_to_apply:
            for ch in range(C):
                state[ch, tx, ty] = solved_CWH[ch, tx, ty]

    # Terminal pair: every placement has been applied, so `state` now equals
    # `solved_CWH`. Emit a sample with eot=1 and sentinel zeros for
    # placement targets; the SFT loop's placement_mask zeroes out the
    # placement losses for this sample. valid_mask=all-zero matches the
    # invariant "no remaining tiles to place".
    terminal_obs = state.to(torch.uint8)
    terminal_valid_mask = torch.zeros(W * H, dtype=torch.bool)
    pairs.append((terminal_obs, 0, 0, 0, 0, 0, terminal_valid_mask, 1))

    return pairs



def _humanize_count(n: int) -> str:
    """50_000 -> '50k', 2_500_000 -> '2.5m'. Used in artifact names so
    `n50k` reads better than `n50000`."""
    if n >= 1_000_000:
        v = n / 1_000_000
        s = f"{v:.1f}".rstrip("0").rstrip(".")
        return f"{s}m"
    if n >= 1_000:
        v = n / 1_000
        s = f"{v:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return str(n)


def _humanize_lr(lr: float) -> str:
    """1e-3 -> '1e-3', 0.0005 -> '5e-4'. Keeps artifact names short and
    visually obvious."""
    if lr == 0:
        return "0"
    exp = int(np.floor(np.log10(lr)))
    mantissa = lr / (10**exp)
    if abs(mantissa - round(mantissa)) < 1e-6:
        return f"{int(round(mantissa))}e{exp}"
    return f"{mantissa:.2g}e{exp}"


def _artifact_name(args: "SftArgs") -> str:
    """Build a descriptive W&B artifact name from the training config.

    Identical-config runs collapse into versions of the same artifact;
    config-changing runs (different size / sample count / lr / channels)
    get their own artifact. `best_val_acc` deliberately goes to the alias
    instead, since baking a varying number into the name would defeat
    versioning."""
    # Encode the full layer list (one token per conv layer) so runs that
    # differ in depth or width get distinct artifacts; append a kernel suffix
    # only for non-default kernels.
    layers = layers_from_args(args)
    chan_str = "c" + "-".join(str(c) for c in layers)
    if args.kernel_size != 3:
        chan_str += f"-k{args.kernel_size}"
    return (
        f"sft-s{args.size}"
        f"-n{_humanize_count(args.num_samples)}"
        f"-e{args.epochs}"
        f"-bs{args.batch_size}"
        f"-lr{_humanize_lr(args.lr)}"
        f"-{chan_str}"
    )


# Max consecutive build failures tolerated for a single lesson kind before it is
# treated as unbuildable at the current grid size and dropped from the sampler.
# build_factory already exhausts a full rejection-sampling budget per call, so a
# None is a strong "can't build" signal; an unbuildable kind (e.g. a 3x3
# assembler with 5 ingredient arms on a size-5 grid) returns None for every seed.
# Without this cap the balance-by-fewest sampler would keep re-drawing that kind
# forever (it stays at 0 samples, always the minimum) and hang.
_MAX_BUILD_FAILURES_PER_KIND = 100


def _iter_demo_pairs(size, max_level, base_seed, worker_id, num_workers, target=None):
    """Yield (obs, tile, ent, dir, item, misc, mask, eot, seed, kind) demos.

    Worker `w` of `num_workers` walks seeds ≡ base_seed+w (mod num_workers), so
    concurrent workers never share a factory. Draws the fewest-pairs kind each
    step so output balances by pair count, not by lesson. Blanks at max_level
    (the full placement progression per lesson); obs/mask are uint8/bool. Runs
    until `target` pairs are produced, or forever when `target` is None. Callers
    own their own RNG seeding.
    """
    kinds = [LessonKind.MEMORISE_2_INGREDIENT_RECIPES]  # HACK: SFT on MEMORISE_2 only
    kind_samples = {k.name: 0 for k in kinds}
    # Kinds still believed buildable at this size, plus a per-kind counter of
    # consecutive build failures. A kind that can't fit the grid returns None for
    # every seed; once it exhausts the failure budget it is dropped so the
    # sampler doesn't spin on it forever (see _MAX_BUILD_FAILURES_PER_KIND).
    available = list(kinds)
    consecutive_fails = {k.name: 0 for k in kinds}

    seed = base_seed + worker_id
    produced = 0
    while target is None or produced < target:
        # Draw the fewest-pairs kind: big factories emit ~10x more pairs, so
        # uniform kind choice starves the rare recipe/assembler heads.
        fewest = min(kind_samples[k.name] for k in available)
        kind = random.choice([k for k in available if kind_samples[k.name] == fewest])
        seed += num_workers
        factory = build_factory(size=size, kind=kind, seed=seed)
        if factory is None:
            consecutive_fails[kind.name] += 1
            if consecutive_fails[kind.name] >= _MAX_BUILD_FAILURES_PER_KIND:
                available.remove(kind)
                print(
                    f"[sft] {kind.name} did not build at size={size} after "
                    f"{_MAX_BUILD_FAILURES_PER_KIND} consecutive attempts; "
                    f"excluding it from this dataset shard."
                )
                if not available:
                    raise RuntimeError(
                        f"No lesson kind could be built at size={size}; cannot "
                        f"generate an SFT dataset. Increase the grid size."
                    )
            continue
        consecutive_fails[kind.name] = 0
        task, _ = blank_entities(factory, num_missing_entities=max_level)

        for pair in extract_expert_actions(factory.world_CWH, task):
            yield (*pair, seed, kind.value)
            kind_samples[kind.name] += 1
            produced += 1
            if target is not None and produced >= target:
                break


def _materialise(size, max_level, base_seed, target=None, n_lessons=None):
    """Eagerly collect demonstrations into stacked tensors (obs, tile, ent, dir,
    item, misc, mask, eot, seed, kind). Stops after `target` pairs, or after
    `n_lessons` distinct factories when that is given instead."""
    random.seed(base_seed)
    rows = []
    seeds = set()
    for row in _iter_demo_pairs(size, max_level, base_seed, 0, 1, target):
        if n_lessons is not None and row[8] not in seeds:
            if len(seeds) >= n_lessons:
                break
            seeds.add(row[8])
        rows.append(row)
    cols = list(zip(*rows))
    return (
        torch.stack(cols[0]),
        torch.tensor(cols[1], dtype=torch.long),
        torch.tensor(cols[2], dtype=torch.long),
        torch.tensor(cols[3], dtype=torch.long),
        torch.tensor(cols[4], dtype=torch.long),
        torch.tensor(cols[5], dtype=torch.long),
        torch.stack(cols[6]),
        torch.tensor(cols[7], dtype=torch.float),
        torch.tensor(cols[8], dtype=torch.long),
        torch.tensor(cols[9], dtype=torch.long),
    )


class StreamingDemoDataset(IterableDataset):
    """Generates SFT demonstrations on the fly, sharded across DataLoader workers.

    Yields the model-input 8-tuple (obs uint8, tile, ent, dir, item, misc,
    mask bool, eot) for `target` pairs per pass; worker w of W walks the disjoint
    seed range base_seed+w mod W so no factory is produced twice. CPU generation
    overlaps GPU training via DataLoader prefetch.
    """

    def __init__(self, size, max_level, base_seed, target):
        self.size = size
        self.max_level = max_level
        self.base_seed = base_seed
        self.target = target

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        random.seed(self.base_seed + worker_id)
        base, rem = divmod(self.target, num_workers)
        my_target = base + (1 if worker_id < rem else 0)
        for row in _iter_demo_pairs(
            self.size, self.max_level, self.base_seed, worker_id, num_workers, my_target
        ):
            # row is (obs, tile, ent, dir, item, misc, mask, eot, seed, kind);
            # training only needs the first 8 — seed/kind are val-only metadata.
            yield row[:8]


def _steps_per_epoch(target, n_workers, batch_size):
    """Optimizer steps one streaming pass yields: each worker batches its own
    per-worker share (partial last batch), so it's the sum of per-worker
    ceil-divisions, not one global ceil-division."""
    n = max(1, n_workers)
    base, rem = divmod(target, n)
    total = 0
    for w in range(n):
        pw = base + (1 if w < rem else 0)
        total += (pw + batch_size - 1) // batch_size
    return max(1, total)


def build_lr_schedule(optimizer, total_steps: int, args: "SftArgs"):
    """Linear warmup → cosine decay scheduler, stepped once per optimizer step.

    Warmup goes from `lr * 1e-3` up to `lr` over the first
    `total_steps * warmup_frac` steps; cosine then decays from `lr` to
    `lr * min_lr_ratio` over the rest. `warmup_frac=0` skips warmup.
    """
    warmup_steps = (
        max(1, int(round(total_steps * args.warmup_frac)))
        if args.warmup_frac > 0
        else 0
    )
    cosine_steps = max(1, total_steps - warmup_steps)
    eta_min = args.lr * args.min_lr_ratio
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_steps,
        eta_min=eta_min,
    )
    if warmup_steps <= 0:
        return cosine
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps],
    )


class RolloutEval(TypedDict):
    """Return shape of :func:`run_rollout_eval`."""

    overall: float  # mean throughput ignoring the EOT head
    overall_eot: float  # mean throughput respecting the EOT head
    per_kind: dict[str, float]  # overall, keyed by LessonKind.name
    per_kind_eot: dict[str, float]  # overall_eot, keyed by LessonKind.name
    per_kind_n: dict[str, int]  # number of val factories per LessonKind.name
    asm_item_acc: float  # frac of placed assemblers given the correct recipe
    per_kind_asm_item_acc: dict[str, float]  # asm_item_acc, keyed by LessonKind.name
    per_kind_asm_n: dict[str, int]  # assembler placements scored per LessonKind.name


def _apply_legal_tile_mask(tile_logits, obs_batch):
    """Mask illegal placement tiles to -inf so an argmax skips them."""
    K = tile_logits.shape[0]
    ent_ch = obs_batch[:, _CH_ENT]
    foot_ch = obs_batch[:, _CH_FOOTPRINT]
    # legal iff no entity & tile is placeable
    legal = (ent_ch == _EMPTY_ENT_ID) & (foot_ch != _FOOTPRINT_UNAVAILABLE)
    return tile_logits.masked_fill(~legal.reshape(K, -1), float("-inf"))


def _solved_assembler_recipes(solved_CWH) -> set[int]:
    """The set of recipes (item ids) carried by assemblers in a solved factory,
    the ground truth for the rollout's recipe-pick check. Empty for factories
    with no assembler (every non-MEMORISE lesson today)."""
    asm_mask = solved_CWH[_CH_ENT] == _ASM_MACHINE_ENT_ID
    recipes = solved_CWH[_CH_ITEMS][asm_mask]
    return set() if not bool(asm_mask.any()) else {int(v) for v in recipes.unique().tolist()}


def run_rollout_eval(
    agent,
    args: SftArgs,
    val_seeds_to_kind: dict[int, int],
    device,
    max_seeds: int = 100,
    eot_threshold: float = 0.5,
    num_envs: int = 8,
) -> RolloutEval:
    """Greedy rollout eval on the held-out val factories.

    Runs K=num_envs FactorioEnvs in parallel and batches the CNN forward
    across them, so the cost per eval is ~K× cheaper than the serial
    version (the big win on GPU/MPS where the batch=1 forward dominates).
    Envs are refilled from the seed queue as they finish, so all K slots
    stay busy until the queue drains.

    For each held-out (seed, kind) we greedy-argmax every head and step
    until the env finishes (throughput==1.0 or max_steps) — we do NOT
    let the EOT head stop us. Throughput is the last `info['thput_normed']`
    the env reported: raw items/sec divided by the per-factory max, in
    [0, 1], so a perfectly-rebuilt factory scores 1.0 regardless of its
    absolute belt speed (the env already calls the Rust solver every step,
    so we don't re-run it). This `overall` number is the model's true
    build skill independent of its stop head.

    In the same single rollout we also track the throughput the model
    *would* have produced if it trusted its EOT head: the first time the
    EOT prob crosses `eot_threshold` for a slot we snapshot that slot's
    current throughput. If the EOT head never fires, the model would have
    built all the way to env-done, so its EOT-respecting value equals the
    final throughput. The mean of those snapshots is `overall_eot`.

    The (seed, kind) pairs are exactly the val_accuracy set — so a rise
    in `val/thput` over training is directly comparable to the
    existing per-kind val accuracy curves.

    The greedy tile argmax is restricted to legal (empty + buildable)
    tiles, so it can't livelock re-proposing an occupied tile the env
    keeps rejecting. Eval-only; training is untouched.

    Returns a dict with:
        overall, overall_eot — mean throughput ignoring / respecting EOT;
        per_kind, per_kind_eot — same, keyed by LessonKind.name;
        per_kind_n — sample count per kind in the eval.
    """
    was_training = agent.training
    agent.eval()

    # Blank the WHOLE factory (size*size) so the greedy eval builds from an
    # empty grid — the honest "can it build from scratch" test, not a partial
    # 2*size blank that leaves most of the factory pre-placed. Matches the
    # data-generation path (which already auto-blanks size*size) and the
    # max_level docstring.
    max_level = args.max_level if args.max_level > 0 else args.size * args.size

    # Deterministic, run-stable subsample of val seeds. Sorting first
    # makes the selection independent of dict iteration order, then we
    # shuffle with args.seed so different runs see the same selection.
    seeds_sorted = sorted(val_seeds_to_kind.keys())
    rng = random.Random(args.seed)
    rng.shuffle(seeds_sorted)
    seeds_sorted = seeds_sorted[:max_seeds]

    per_kind_throughputs: dict[str, list[float]] = {k.name: [] for k in LessonKind}
    per_kind_eot_throughputs: dict[str, list[float]] = {k.name: [] for k in LessonKind}
    all_throughputs: list[float] = []
    all_eot_throughputs: list[float] = []
    per_kind_asm_correct: dict[str, int] = {k.name: 0 for k in LessonKind}
    per_kind_asm_total: dict[str, int] = {k.name: 0 for k in LessonKind}

    if not seeds_sorted:
        if was_training:
            agent.train()
        zero = {kn: 0.0 for kn in per_kind_throughputs}
        per_kind_n = {kn: 0 for kn in per_kind_throughputs}
        return {
            "overall": 0.0,
            "overall_eot": 0.0,
            "per_kind": zero,
            "per_kind_eot": dict(zero),
            "per_kind_n": per_kind_n,
            "asm_item_acc": 0.0,
            "per_kind_asm_item_acc": dict(zero),
            "per_kind_asm_n": dict(per_kind_n),
        }

    # Cap K at the number of seeds — spinning up more envs than work
    # items wastes memory. All envs use idx=0 so the seed we pass to
    # reset() is the seed generate_lesson sees: FactorioEnv.reset adds
    # self.idx to the seed for env-diversity in PPO, but here we need
    # exact seed pass-through to replay the held-out val factories.
    K = max(1, min(num_envs, len(seeds_sorted)))
    envs = [FactorioEnv(size=args.size, idx=0) for _ in range(K)]

    # Pop seeds off the front; each slot harvests its result then pulls
    # the next one. `current` holds the (seed, kind, last_throughput)
    # owned by each slot; `active[i]=False` means the slot has no more
    # work and should be skipped.
    queue = list(seeds_sorted)
    active = [True] * K
    # Per-slot EOT bookkeeping: whether the EOT head has fired yet for the
    # seed this slot is replaying, and the throughput snapshotted when it did.
    eot_fired = [False] * K
    eot_thp = [0.0] * K
    # Correct recipes for the factory each slot is replaying (from its solved
    # world). Empty when that factory has no assembler, which skips the check.
    asm_recipes: list[set[int]] = [set() for _ in range(K)]
    current: list[tuple[int, LessonKind, float]] = [
        (0, LessonKind.MOVE_ONE_ITEM, 0.0)
    ] * K
    obs_stack = []

    for i in range(K):
        s = queue.pop(0)
        k = LessonKind(val_seeds_to_kind[s])
        obs, info = envs[i].reset(
            seed=s,
            options={
                "num_missing_entities": max_level,
                "kind": k,
            },
        )
        current[i] = (s, k, float(info.get("thput_normed", 0.0)))
        asm_recipes[i] = _solved_assembler_recipes(envs[i]._solved_world_CWH)
        obs_stack.append(obs)

    obs_batch = torch.as_tensor(np.stack(obs_stack), dtype=torch.float32, device=device)
    batch_idx_K = torch.arange(K, device=device)

    with torch.no_grad():
        while any(active):
            # Single CNN forward across all K slots. Inactive slots'
            # outputs are discarded; the per-eval cost of K-1 stale
            # forwards at the tail is negligible compared to the
            # batching win earlier in the queue.
            encoded = agent.encoder(agent._encode_input(obs_batch))
            eot_probs = torch.sigmoid(agent.eot_head(encoded).squeeze(-1))

            tile_logits = _apply_legal_tile_mask(
                agent.tile_logits(encoded).reshape(K, -1), obs_batch
            )
            tile_idx_K = tile_logits.argmax(dim=1)
            x_K = tile_idx_K // args.size
            y_K = tile_idx_K % args.size
            tile_features = encoded[batch_idx_K, :, x_K, y_K]

            ent_K = agent.ent_head(tile_features).argmax(dim=1)
            dir_K = agent.dir_head(tile_features).argmax(dim=1)
            item_K = agent.item_head(tile_features).argmax(dim=1)
            misc_K = agent.misc_head(tile_features).argmax(dim=1)

            for i in range(K):
                if not active[i]:
                    continue

                s, k, cur_thp = current[i]

                # Snapshot the EOT-respecting throughput the first time the
                # head crosses threshold — but keep stepping regardless, so
                # `overall` reflects true build skill (EOT ignored).
                if not eot_fired[i] and float(eot_probs[i]) > eot_threshold:
                    eot_fired[i] = True
                    eot_thp[i] = cur_thp

                action = {
                    "xy": np.array([int(x_K[i]), int(y_K[i])], dtype=int),
                    "entity": int(ent_K[i]),
                    "direction": int(dir_K[i]),
                    "item": int(item_K[i]),
                    "misc": int(misc_K[i]),
                }
                next_obs, _r, terminated, truncated, info = envs[i].step(action)
                current[i] = (s, k, float(info.get("thput_normed", 0.0)))

                # Recipe-pick check: the agent tried to place an assembler in a
                # factory that has one. Count it iff the assembler actually
                # landed at the anchor (invalid placements are env no-ops), then
                # score its recipe against the solved factory's recipes.
                if asm_recipes[i] and action["entity"] == _ASM_MACHINE_ENT_ID:
                    ax, ay = int(action["xy"][0]), int(action["xy"][1])
                    if int(envs[i]._world_CWH[_CH_ENT, ax, ay]) == _ASM_MACHINE_ENT_ID:
                        per_kind_asm_total[k.name] += 1
                        if action["item"] in asm_recipes[i]:
                            per_kind_asm_correct[k.name] += 1

                if not (terminated or truncated):
                    obs_batch[i] = torch.as_tensor(
                        next_obs,
                        dtype=torch.float32,
                        device=device,
                    )
                    continue

                # Slot is finished — harvest both metrics, refill or deactivate.
                s, k, last_thp = current[i]
                all_throughputs.append(last_thp)
                per_kind_throughputs[k.name].append(last_thp)
                # EOT never fired → model would have built to env-done, so its
                # EOT-respecting value is the same final throughput.
                eot_value = eot_thp[i] if eot_fired[i] else last_thp
                all_eot_throughputs.append(eot_value)
                per_kind_eot_throughputs[k.name].append(eot_value)

                if queue:
                    s = queue.pop(0)
                    k = LessonKind(val_seeds_to_kind[s])
                    obs, info = envs[i].reset(
                        seed=s,
                        options={
                            "num_missing_entities": max_level,
                            "kind": k,
                        },
                    )
                    current[i] = (s, k, float(info.get("thput_normed", 0.0)))
                    asm_recipes[i] = _solved_assembler_recipes(
                        envs[i]._solved_world_CWH
                    )
                    eot_fired[i] = False
                    eot_thp[i] = 0.0
                    obs_batch[i] = torch.as_tensor(
                        obs,
                        dtype=torch.float32,
                        device=device,
                    )
                else:
                    active[i] = False

    overall = float(np.mean(all_throughputs)) if all_throughputs else 0.0
    overall_eot = float(np.mean(all_eot_throughputs)) if all_eot_throughputs else 0.0
    per_kind = {
        kn: (float(np.mean(ts)) if ts else 0.0)
        for kn, ts in per_kind_throughputs.items()
    }
    per_kind_eot = {
        kn: (float(np.mean(ts)) if ts else 0.0)
        for kn, ts in per_kind_eot_throughputs.items()
    }
    per_kind_n = {kn: len(ts) for kn, ts in per_kind_throughputs.items()}

    asm_total_all = sum(per_kind_asm_total.values())
    asm_item_acc = (
        sum(per_kind_asm_correct.values()) / asm_total_all if asm_total_all else 0.0
    )
    per_kind_asm_item_acc = {
        kn: (per_kind_asm_correct[kn] / n if n else 0.0)
        for kn, n in per_kind_asm_total.items()
    }

    if was_training:
        agent.train()
    return {
        "overall": overall,
        "overall_eot": overall_eot,
        "per_kind": per_kind,
        "per_kind_eot": per_kind_eot,
        "per_kind_n": per_kind_n,
        "asm_item_acc": asm_item_acc,
        "per_kind_asm_item_acc": per_kind_asm_item_acc,
        "per_kind_asm_n": dict(per_kind_asm_total),
    }


# Direction classes for the val confusion matrix, ordered by enum value
# (NONE=0, NORTH=1, EAST=2, SOUTH=3, WEST=4).
_DIRECTION_NAMES = [d.name for d in sorted(Direction, key=lambda d: d.value)]


def _direction_diagnostics(true_dirs, pred_dirs):
    """W&B log payload: the direction confusion matrix rendered as a PNG image.

    We log a matplotlib heatmap as wandb.Image (not wandb.plot.confusion_matrix)
    on purpose: logging an Image every epoch produces a media panel with a step
    slider you can scrub across training, whereas wandb.plot.confusion_matrix
    logs a one-off Table that can't be scrubbed and whose auto-panel renders as
    a bar. Row-normalised so each row is P(pred | target) (the diagonal is
    per-direction recall), which stays comparable across epochs regardless of
    val size. Returns {} when there are no placement samples."""
    import numpy as np
    import wandb
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not true_dirs:
        return {}
    n = len(_DIRECTION_NAMES)
    counts = np.zeros((n, n), dtype=float)
    np.add.at(counts, (np.asarray(true_dirs), np.asarray(pred_dirs)), 1.0)
    row_total = counts.sum(axis=1, keepdims=True)
    norm = np.divide(counts, row_total, out=np.zeros_like(counts), where=row_total > 0)

    fig = Figure(figsize=(5.0, 4.5), dpi=110)
    ax = fig.subplots()
    im = ax.imshow(norm, cmap="turbo", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(_DIRECTION_NAMES, rotation=45, ha="right")
    ax.set_yticklabels(_DIRECTION_NAMES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("target (demo)")
    ax.set_title("Direction confusion (row-normalised: P(pred | target))")
    for i in range(n):
        if row_total[i, 0] == 0:
            continue
        for j in range(n):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] < 0.6 else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return {"val/dir_confusion": wandb.Image(np.asarray(canvas.buffer_rgba()))}


# Source/sink are never blanked (protected in _remove_entities), so every
# observation carries them in the ENTITIES channel — we can read the
# source<->sink separation straight off the obs without plumbing it through
# the dataset.
_SOURCE_ENTITY_ID = str2ent("source").value
_SINK_ENTITY_ID = str2ent("sink").value


def _source_sink_distance(obs_BCWH):
    """Manhattan distance between the source centroid and the sink centroid,
    per sample, read from the ENTITIES channel. Exact for single-source /
    single-sink lessons (MOVE_*); a routing-span proxy for multi-endpoint
    lessons (splitters/assemblers). Returns a [B] float tensor; samples with
    no source or no sink get -1 (caller drops them)."""
    ent = obs_BCWH[:, Channel.ENTITIES.value, :, :]  # [B, W, H]
    W, H = ent.shape[1], ent.shape[2]
    xs = torch.arange(W, device=ent.device).view(1, W, 1).float()
    ys = torch.arange(H, device=ent.device).view(1, 1, H).float()
    src = (ent == _SOURCE_ENTITY_ID).float()
    snk = (ent == _SINK_ENTITY_ID).float()
    src_n = src.sum((1, 2))
    snk_n = snk.sum((1, 2))
    src_x = (src * xs).sum((1, 2)) / src_n.clamp(min=1)
    src_y = (src * ys).sum((1, 2)) / src_n.clamp(min=1)
    snk_x = (snk * xs).sum((1, 2)) / snk_n.clamp(min=1)
    snk_y = (snk * ys).sum((1, 2)) / snk_n.clamp(min=1)
    dist = (src_x - snk_x).abs() + (src_y - snk_y).abs()
    dist[(src_n == 0) | (snk_n == 0)] = -1.0
    return dist


def _dir_mismatch_by_distance(distances, mismatches):
    """Bucket per-sample direction mismatches by (rounded) source<->sink
    distance. A "mismatch" is argmax(dir) != the demo's target direction —
    NOT necessarily a true error, since under multi-path ambiguity a different
    direction can be an equally valid routing. Returns (sorted_distances,
    mismatch_rates, counts) — the correlation curve."""
    from collections import defaultdict

    total = defaultdict(int)
    wrong = defaultdict(int)
    for d, m in zip(distances, mismatches):
        total[int(round(d))] += 1
        wrong[int(round(d))] += int(m)
    xs = sorted(total)
    return xs, [wrong[d] / total[d] for d in xs], [total[d] for d in xs]


def _point_biserial(distances, mismatches):
    """Pearson r between source<->sink distance and per-sample dir mismatch
    (0/1). Positive => the model diverges from the demo more as the endpoints
    get farther apart (the receptive-field hypothesis). 0.0 if either side has
    no variance."""
    n = len(distances)
    if n == 0:
        return 0.0
    md = sum(distances) / n
    mm = sum(mismatches) / n
    cov = sum((d - md) * (m - mm) for d, m in zip(distances, mismatches))
    var_d = sum((d - md) ** 2 for d in distances)
    var_m = sum((m - mm) ** 2 for m in mismatches)
    denom = (var_d * var_m) ** 0.5
    return cov / denom if denom > 0 else 0.0


def _dir_distance_diagnostics(distances, mismatches):
    """W&B payload: the dir-mismatch-vs-distance curve (as a PNG) + a single
    correlation scalar. "Mismatch" = argmax(dir) disagrees with the demo's
    direction (a valid alternative routing also counts as a mismatch).

    The curve is rendered with matplotlib and logged as wandb.Image (not
    wandb.plot.line) so the media panel has a step slider you can scrub across
    training; wandb.plot.line logs a one-off Table that can't be scrubbed. The
    correlation is a plain scalar, so it already trends over time. {} when
    there are no placement samples with a resolvable distance."""
    import numpy as np
    import wandb
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    if not distances:
        return {}
    xs, rates, counts = _dir_mismatch_by_distance(distances, mismatches)
    corr = _point_biserial(distances, mismatches)

    fig = Figure(figsize=(6.0, 4.0), dpi=110)
    ax = fig.subplots()
    ax.plot(xs, rates, marker="o", color="#d62728")
    ax.set_xlabel("source<->sink Manhattan distance")
    ax.set_ylabel("dir mismatch rate (pred != demo)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(f"Dir mismatch vs source-sink distance  (r={corr:+.2f})")
    ax.grid(True, alpha=0.3)
    # annotate each point with its sample count, so sparse (noisy) far-distance
    # buckets are obvious at a glance.
    for x, y, n in zip(xs, rates, counts):
        ax.annotate(
            str(n),
            (x, y),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=7,
            color="gray",
        )
    fig.tight_layout()
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return {
        "val/dir/mismatch_distance_corr": corr,
        "val/dir_mismatch_vs_distance": wandb.Image(np.asarray(canvas.buffer_rgba())),
    }


def train_sft(args: SftArgs):
    """Main SFT training loop."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    max_level = args.max_level if args.max_level > 0 else args.size * args.size

    # Held-out validation: the first eval_rollouts_max_seeds distinct factories
    # from seed `args.seed` upward. The rollout eval replays these same lessons,
    # so val/acc and val/thput stay directly comparable.
    val = _materialise(
        args.size, max_level, args.seed, n_lessons=args.eval_rollouts_max_seeds
    )
    val_tensors_cpu = (*val[:8], val[9])  # obs..eot, kind (per-pair seed dropped)
    val_seeds_to_kind: dict[int, int] = dict(zip(val[8].tolist(), val[9].tolist()))

    # Training draws every seed above the ones validation consumed, so the two
    # sets are disjoint by construction — no factory is ever both trained and
    # validated on. Streams on the fly (bounded memory); --dataset-cache instead
    # materialises the stream to disk once and trains from it, so repeated runs
    # (benchmarks, dev iteration) skip build_factory.
    train_base = int(val[8].max()) + 1
    cached_train = None
    if args.dataset_cache is not None:
        if os.path.exists(args.dataset_cache):
            print(f"Loading cached dataset from {args.dataset_cache} ...")
            # weights_only=False: our own locally-produced, trusted cache.
            cached_train = torch.load(args.dataset_cache, weights_only=False)
        else:
            print(f"Materialising {args.num_samples} demonstrations to cache ...")
            cached_train = _materialise(
                args.size, max_level, train_base, target=args.num_samples
            )[:8]
            torch.save(cached_train, args.dataset_cache)
            print(f"Cached dataset to {args.dataset_cache}")

    # Re-seed so training RNG is identical whether the cache was just created
    # (which consumes the generator RNG) or loaded.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create agent via a temporary env (AgentCNN needs envs for init)
    env_id = "factorion/FactorioEnv-v0-sft"
    if env_id not in gym.registry:
        gym.register(id=env_id, entry_point="ppo:FactorioEnv")
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, False, args.size, "sft")])

    agent = AgentCNN(
        envs,
        layers=layers_from_args(args),
        kernel_size=args.kernel_size,
        tile_head_std=args.tile_head_std,
        dropout=args.dropout,
    )
    envs.close()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert_device_ok(device)

    if args.start_from is not None:
        print(f"Loading model weights from {args.start_from}")
        ckpt_path = _resolve_start_from(
            args.start_from, args.wandb_project_name, args.wandb_entity
        )
        # Load to CPU first; the checkpoint may hold CUDA tensors (saved on a
        # GPU pod) and the agent is moved onto `device` just below, so this
        # keeps --start-from working on CPU/MPS boxes too.
        agent.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    agent.to(device)

    if cached_train is not None:
        # Cached: the whole training set goes GPU-resident once (obs stay uint8,
        # cast to float per-batch), shuffled via a DataLoader over indices so no
        # data is copied per batch.
        (
            tr_obs, tr_tile, tr_ent, tr_dir, tr_item, tr_misc, tr_mask, tr_eot,
        ) = (
            cached_train[0].to(device),
            *(t.to(device) for t in cached_train[1:]),
        )
        n_train = tr_obs.shape[0]
        train_index_loader = DataLoader(
            TensorDataset(torch.arange(n_train)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        steps_per_epoch = max(1, (n_train + args.batch_size - 1) // args.batch_size)
    else:
        # Stream: train batches are generated on the fly by DataLoader workers,
        # so memory is bounded by the prefetch buffer. pin_memory + non_blocking
        # (in the batch source) hide the per-batch host->device copy behind the
        # GPU step.
        stream_workers = min(16, os.cpu_count() or 1)
        train_stream_loader = DataLoader(
            StreamingDemoDataset(args.size, max_level, train_base, args.num_samples),
            batch_size=args.batch_size,
            num_workers=stream_workers,
            pin_memory=(device.type == "cuda"),
            persistent_workers=stream_workers > 0,
            multiprocessing_context="forkserver",
        )
        n_train = args.num_samples
        steps_per_epoch = _steps_per_epoch(
            args.num_samples, stream_workers, args.batch_size
        )
    # Val goes GPU-resident too, iterated via an index DataLoader.
    (
        va_obs, va_tile, va_ent, va_dir, va_item, va_misc, va_mask, va_eot, va_kind,
    ) = (
        val_tensors_cpu[0].to(device),
        *(t.to(device) for t in val_tensors_cpu[1:]),
    )
    n_val = va_obs.shape[0]
    val_index_loader = DataLoader(
        TensorDataset(torch.arange(n_val)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    optimizer = optim.AdamW(
        agent.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Scheduler spans every optimizer step (not every epoch) so warmup_frac
    # is a fraction of the whole run regardless of dataset size.
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_lr_schedule(optimizer, total_steps, args)

    # All losses use reduction="none" so we can (a) mask placement losses
    # off on terminal (eot=1) samples and (b) aggregate per-LessonKind in
    # the val loop without re-running the forward pass.
    ce_loss_none = nn.CrossEntropyLoss(reduction="none")
    bce_loss_none = nn.BCEWithLogitsLoss(reduction="none")
    bce_eot = nn.BCEWithLogitsLoss()

    # Map kind value -> name so per-kind dict keys read as "MOVE_ONE_ITEM"
    # instead of "0" both in print() lines and in wandb panel titles.
    kind_names = {k.value: k.name for k in LessonKind}

    run = None
    if args.track:
        import wandb

        sft_tags = ["sft"] + (args.tags or [])
        if args.start_from is not None:
            sft_tags.append(f"start_from:{args.start_from}")
        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=_artifact_name(args),
            group=args.wandb_group,
            tags=sft_tags,
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
        )

    best_val_acc = 0.0
    best_val_throughput = 0.0
    ever_saved = False
    val_loss = 0.0
    val_tile_acc = 0.0
    val_ent_acc = 0.0
    val_dir_acc = 0.0
    val_item_acc = 0.0
    val_misc_acc = 0.0
    val_eot_acc = 0.0
    val_eot_pos_recall = 0.0
    # Cumulative optimisation pressure, used as the wandb x-axis (see the
    # define_metric calls above). global_step counts optimiser updates;
    # samples_seen counts training pairs the model has been updated on.
    global_step = 0
    samples_seen = 0
    total_samples = args.epochs * n_train
    print(f"Training for {args.epochs} epochs ({total_samples} samples) on {device}...")

    # Train batch source yielding the 8 model-input tensors per batch (obs, tile,
    # ent, dir, item, misc, mask, eot; obs/eot already float). Cached: gather
    # GPU-resident tensors by shuffled index (no host->device copy). Stream: pull
    # CPU batches from the generating workers and copy them to device
    # (non_blocking, overlapping the GPU step).
    def _train_batches():
        if cached_train is not None:
            for (idx_cpu,) in train_index_loader:
                idx = idx_cpu.to(device)
                yield (
                    tr_obs[idx].float(), tr_tile[idx], tr_ent[idx], tr_dir[idx],
                    tr_item[idx], tr_misc[idx], tr_mask[idx], tr_eot[idx],
                )
        else:
            for b_obs, b_tile, b_ent, b_dir, b_item, b_misc, b_mask, b_eot in (
                train_stream_loader
            ):
                yield (
                    b_obs.to(device, non_blocking=True).float(),
                    b_tile.to(device, non_blocking=True),
                    b_ent.to(device, non_blocking=True),
                    b_dir.to(device, non_blocking=True),
                    b_item.to(device, non_blocking=True),
                    b_misc.to(device, non_blocking=True),
                    b_mask.to(device, non_blocking=True),
                    b_eot.to(device, non_blocking=True).float(),
                )

    def _batch_stream():
        """(epoch, is_epoch_end, batch) across every epoch; a fresh
        _train_batches() per epoch re-runs the stream / reshuffles the cache.
        Streamed epochs can vary by a batch, so is_epoch_end is keyed to the
        predicted steps_per_epoch; the final eval is guaranteed by exhaustion."""
        for ep in range(1, args.epochs + 1):
            for i, batch in enumerate(_train_batches()):
                yield ep, i == steps_per_epoch - 1, batch

    stream = _batch_stream()
    epoch = 0
    next_eval_at = args.eval_every_n_samples
    stream_done = False
    pbar = tqdm.tqdm(total=total_samples, unit="smpl", unit_scale=True)
    while not stream_done:
        t_train = time.time()
        agent.train()
        # On-GPU accumulators converted ONCE per eval window (logging-only),
        # instead of ~10 .item() syncs/batch; see tests/benchmarks/EXPERIMENT_LOG.md.
        acc_loss = torch.zeros(7, device=device)  # total,tile,ent,dir,item,misc,eot
        acc_correct = torch.zeros((), device=device)
        acc_eot_correct = torch.zeros((), device=device)
        acc_grad_norm = torch.zeros((), device=device)
        train_total = 0
        grad_norm_count = 0
        # DataLoader-wait attribution (CPU side; needs no GPU sync). The compute
        # share is derived at window end as train_seconds - train_data_s.
        train_data_s = 0.0

        t_batch = time.time()
        for epoch, is_epoch_end, batch in stream:
            t_ready = time.time()
            train_data_s += t_ready - t_batch
            (
                batch_obs, batch_tile, batch_ent, batch_dir,
                batch_item, batch_misc, batch_mask, batch_eot,
            ) = batch

            encoded = agent.encoder(agent._encode_input(batch_obs))
            B = encoded.shape[0]
            # Placement loss is only meaningful for non-terminal samples;
            # eot=1 samples carry sentinel placement targets. Normalise by
            # the placement-sample count so the loss scale is independent
            # of the per-batch mix of terminal / placement samples.
            placement_mask = (batch_eot < 0.5).float()
            n_place = placement_mask.sum().clamp(min=1.0)

            # EOT head — BCE on every sample.
            eot_logits = agent.eot_head(encoded).squeeze(-1)
            loss_eot = bce_eot(eot_logits, batch_eot)

            # Tile logits — use BCE with multi-label mask so ALL valid
            # tiles are rewarded, not just the randomly-chosen one. Reduce
            # to per-sample, mask off terminal samples, then average.
            tile_logits = agent.tile_logits(encoded).reshape(B, -1)
            loss_tile_per = bce_loss_none(tile_logits, batch_mask.float()).mean(dim=1)
            loss_tile = (loss_tile_per * placement_mask).sum() / n_place

            # Extract features at target tile for entity/direction/item/misc heads
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
            loss_ent = (loss_ent_per * placement_mask).sum() / n_place
            loss_dir = (loss_dir_per * placement_mask).sum() / n_place
            loss_item = (loss_item_per * placement_mask).sum() / n_place
            loss_misc = (loss_misc_per * placement_mask).sum() / n_place

            loss = (
                args.lw_tile * loss_tile
                + args.lw_ent * loss_ent
                + args.lw_dir * loss_dir
                + args.lw_item * loss_item
                + args.lw_misc * loss_misc
                + args.lw_eot * loss_eot
            )

            optimizer.zero_grad()
            loss.backward()
            if args.max_grad_norm > 0:
                grad_norm = nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm
                )
                acc_grad_norm += grad_norm
                grad_norm_count += 1
            optimizer.step()
            scheduler.step()
            global_step += 1
            samples_seen += B
            pbar.update(B)

            acc_loss += torch.stack([
                loss, loss_tile, loss_ent, loss_dir, loss_item, loss_misc, loss_eot
            ]).detach() * B
            # Whole-action accuracy: the model agrees with the demo on
            # every output for this sample. For placement samples (eot=0)
            # that means all 5 placement heads correct AND EOT predicted
            # "not done". For terminal samples (eot=1) that means EOT
            # predicted "done"; placement targets are sentinels so they
            # don't enter the check.
            pred_tile = tile_logits.argmax(dim=1)
            tile_hit = batch_mask[batch_idx, pred_tile] > 0
            pred_ent = ent_logits.argmax(dim=1)
            pred_dir = dir_logits.argmax(dim=1)
            pred_item = item_logits.argmax(dim=1)
            pred_misc = misc_logits.argmax(dim=1)
            place_heads_correct = (
                tile_hit
                & (pred_ent == batch_ent)
                & (pred_dir == batch_dir)
                & (pred_item == batch_item)
                & (pred_misc == batch_misc)
            )
            is_place = placement_mask.bool()
            eot_pred_bool = eot_logits > 0
            eot_correct_t = eot_pred_bool == (batch_eot > 0.5)
            correct = torch.where(
                is_place,
                place_heads_correct & eot_correct_t,
                eot_correct_t,
            )
            acc_correct += correct.sum()
            train_total += B
            acc_eot_correct += eot_correct_t.sum()
            # No per-batch sync: t_batch just bounds the next DataLoader wait.
            t_batch = time.time()

            hit_sample_cadence = (
                args.eval_every_n_samples > 0 and samples_seen >= next_eval_at
            )
            if hit_sample_cadence:
                next_eval_at += args.eval_every_n_samples
            if hit_sample_cadence or is_epoch_end:
                break
        else:
            stream_done = True

        # A window that consumed nothing (stream exhausted right on a boundary)
        # has no stats to sync or log; skip straight to loop exit.
        if train_total == 0:
            break

        # Single GPU->CPU sync for the whole window's accumulated stats.
        (
            train_loss,
            train_loss_tile,
            train_loss_ent,
            train_loss_dir,
            train_loss_item,
            train_loss_misc,
            train_loss_eot,
        ) = (acc_loss / train_total).tolist()
        train_acc = (acc_correct / train_total).item()
        train_eot_acc = (acc_eot_correct / train_total).item()
        grad_norm_sum = acc_grad_norm.item()
        train_seconds = time.time() - t_train
        train_compute_s = max(0.0, train_seconds - train_data_s)

        # Validation
        t_val = time.time()
        agent.eval()
        val_loss = 0.0
        val_loss_tile = 0.0
        val_loss_ent = 0.0
        val_loss_dir = 0.0
        val_loss_item = 0.0
        val_loss_misc = 0.0
        val_loss_eot = 0.0
        val_correct = 0
        val_tile_correct = 0
        val_ent_correct = 0
        val_dir_correct = 0
        val_item_correct = 0
        val_misc_correct = 0
        val_eot_correct = 0
        val_eot_pos_correct = 0
        val_eot_pos_total = 0
        val_total = 0
        val_place_total = 0
        # Direction (target, pred) over placement samples, for the confusion
        # matrix logged below.
        dir_true_eval: list[int] = []
        dir_pred_eval: list[int] = []
        # source<->sink distance + dir mismatch per placement sample, for the
        # dir-mismatch-vs-distance correlation diagnostic.
        dist_eval: list[float] = []
        dir_mismatch_eval: list[int] = []

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
        # EOT is scored over EVERY sample of a kind (the placement eot=0 steps
        # plus the one terminal eot=1), so it needs full-sample counts, not the
        # placement-only per_kind_n. The pos_* counters track recall on the rare
        # terminal (eot=1) samples — the ones that actually trigger "stop".
        per_kind_eot_correct = {k.name: 0 for k in LessonKind}
        per_kind_eot_total = {k.name: 0 for k in LessonKind}
        per_kind_eot_pos_correct = {k.name: 0 for k in LessonKind}
        per_kind_eot_pos_total = {k.name: 0 for k in LessonKind}

        with torch.no_grad():
            for (idx_cpu,) in val_index_loader:
                idx = idx_cpu.to(device)
                batch_obs = va_obs[idx].float()
                batch_tile = va_tile[idx]
                batch_ent = va_ent[idx]
                batch_dir = va_dir[idx]
                batch_item = va_item[idx]
                batch_misc = va_misc[idx]
                batch_mask = va_mask[idx]
                batch_eot = va_eot[idx]
                batch_kind = va_kind[idx]

                encoded = agent.encoder(agent._encode_input(batch_obs))
                B = encoded.shape[0]
                placement_mask = (batch_eot < 0.5).float()
                is_place = placement_mask.bool()

                eot_logits = agent.eot_head(encoded).squeeze(-1)
                loss_eot_per = bce_loss_none(eot_logits, batch_eot)

                tile_logits = agent.tile_logits(encoded).reshape(B, -1)
                # Per-sample losses: needed so we can sum them within each
                # LessonKind and so terminal samples can be masked out of
                # placement losses. mean over the tile axis matches the
                # scale of bce_loss(reduction="mean"), keeping
                # val/loss_tile comparable to train/loss_tile.
                loss_tile_per = (
                    bce_loss_none(tile_logits, batch_mask.float()).mean(dim=1)
                    * placement_mask
                )

                x_B = batch_tile // agent.height
                y_B = batch_tile % agent.height
                batch_idx = torch.arange(B, device=device)
                tile_features = encoded[batch_idx, :, x_B, y_B]

                ent_logits = agent.ent_head(tile_features)
                dir_logits = agent.dir_head(tile_features)
                item_logits = agent.item_head(tile_features)
                misc_logits = agent.misc_head(tile_features)
                loss_ent_per = ce_loss_none(ent_logits, batch_ent) * placement_mask
                loss_dir_per = ce_loss_none(dir_logits, batch_dir) * placement_mask
                loss_item_per = ce_loss_none(item_logits, batch_item) * placement_mask
                loss_misc_per = ce_loss_none(misc_logits, batch_misc) * placement_mask

                loss_per_sample = (
                    args.lw_tile * loss_tile_per
                    + args.lw_ent * loss_ent_per
                    + args.lw_dir * loss_dir_per
                    + args.lw_item * loss_item_per
                    + args.lw_misc * loss_misc_per
                    + args.lw_eot * loss_eot_per
                )
                val_loss += loss_per_sample.sum().item()
                val_loss_tile += loss_tile_per.sum().item()
                val_loss_ent += loss_ent_per.sum().item()
                val_loss_dir += loss_dir_per.sum().item()
                val_loss_item += loss_item_per.sum().item()
                val_loss_misc += loss_misc_per.sum().item()
                val_loss_eot += loss_eot_per.sum().item()

                pred_tile = tile_logits.argmax(dim=1)
                tile_hit = batch_mask[batch_idx, pred_tile] > 0
                pred_ent = ent_logits.argmax(dim=1)
                pred_dir = dir_logits.argmax(dim=1)
                pred_item = item_logits.argmax(dim=1)
                pred_misc = misc_logits.argmax(dim=1)
                ent_correct_per = pred_ent == batch_ent
                dir_correct_per = pred_dir == batch_dir
                item_correct_per = pred_item == batch_item
                misc_correct_per = pred_misc == batch_misc
                # Collect placement-sample directions for the confusion matrix
                # (terminal samples carry sentinel dir=0, so exclude them).
                dir_true_eval.extend(batch_dir[is_place].tolist())
                dir_pred_eval.extend(pred_dir[is_place].tolist())
                # source<->sink distance vs dir mismatch, placement samples with
                # a resolvable distance (both endpoints present in the obs).
                sample_dist = _source_sink_distance(batch_obs)
                place_valid = is_place & (sample_dist >= 0)
                dist_eval.extend(sample_dist[place_valid].tolist())
                dir_mismatch_eval.extend((~dir_correct_per[place_valid]).int().tolist())
                # EOT-head accuracy + recall on positives separately.
                # Recall on positives matters because the positives are
                # rare; "always predict 0" would give high accuracy but
                # never trigger episode termination.
                eot_pred_bool = eot_logits > 0
                eot_correct_per = eot_pred_bool == (batch_eot > 0.5)

                # Whole-sample accuracy. Placement sample (eot=0): all 5
                # placement heads correct AND EOT predicted "not done".
                # Terminal sample (eot=1): EOT predicted "done"; placement
                # targets are sentinels so they don't enter the check.
                place_heads_correct = (
                    tile_hit
                    & ent_correct_per
                    & dir_correct_per
                    & item_correct_per
                    & misc_correct_per
                )
                correct_per = torch.where(
                    is_place,
                    place_heads_correct & eot_correct_per,
                    eot_correct_per,
                )
                val_correct += int(correct_per.sum().item())
                val_tile_correct += tile_hit[is_place].sum().item()
                val_ent_correct += ent_correct_per[is_place].sum().item()
                val_dir_correct += dir_correct_per[is_place].sum().item()
                val_item_correct += item_correct_per[is_place].sum().item()
                val_misc_correct += misc_correct_per[is_place].sum().item()
                val_total += B
                val_place_total += int(is_place.sum().item())

                val_eot_correct += int(eot_correct_per.sum().item())
                is_pos = batch_eot > 0.5
                val_eot_pos_correct += int(eot_correct_per[is_pos].sum().item())
                val_eot_pos_total += int(is_pos.sum().item())

                # Per-kind aggregation: bucket placement metrics by kind on
                # placement samples only (terminal samples carry no kind-
                # specific placement signal; their eot loss is folded in
                # via loss_per_sample). unique() keeps this
                # O(num_kinds_in_batch) so adding new LessonKind enum
                # values has no cost here.
                for k_val in batch_kind.unique().tolist():
                    k_name = kind_names[k_val]
                    mask_k = (batch_kind == k_val) & is_place
                    per_kind_n[k_name] += int(mask_k.sum().item())
                    per_kind_correct[k_name] += int(correct_per[mask_k].sum().item())
                    per_kind_tile_correct[k_name] += int(tile_hit[mask_k].sum().item())
                    per_kind_ent_correct[k_name] += int(
                        ent_correct_per[mask_k].sum().item()
                    )
                    per_kind_dir_correct[k_name] += int(
                        dir_correct_per[mask_k].sum().item()
                    )
                    per_kind_item_correct[k_name] += int(
                        item_correct_per[mask_k].sum().item()
                    )
                    per_kind_misc_correct[k_name] += int(
                        misc_correct_per[mask_k].sum().item()
                    )
                    per_kind_loss_sum[k_name] += loss_per_sample[mask_k].sum().item()
                    # EOT spans the whole kind (placement + terminal), so use
                    # the full kind mask here, not the placement-only mask_k.
                    kind_k = batch_kind == k_val
                    kind_k_pos = kind_k & is_pos
                    per_kind_eot_correct[k_name] += int(
                        eot_correct_per[kind_k].sum().item()
                    )
                    per_kind_eot_total[k_name] += int(kind_k.sum().item())
                    per_kind_eot_pos_correct[k_name] += int(
                        eot_correct_per[kind_k_pos].sum().item()
                    )
                    per_kind_eot_pos_total[k_name] += int(kind_k_pos.sum().item())

        # Placement losses were already masked off on terminal samples (their
        # per-sample contribution is zero), so dividing by val_place_total
        # gives the average over the placement subset. The eot loss spans
        # every sample, so it's divided by val_total.
        place_norm = max(1, val_place_total)
        val_loss /= val_total
        val_loss_tile /= place_norm
        val_loss_ent /= place_norm
        val_loss_dir /= place_norm
        val_loss_item /= place_norm
        val_loss_misc /= place_norm
        val_loss_eot /= val_total
        val_acc = val_correct / val_total
        val_tile_acc = val_tile_correct / place_norm
        val_ent_acc = val_ent_correct / place_norm
        val_dir_acc = val_dir_correct / place_norm
        val_item_acc = val_item_correct / place_norm
        val_misc_acc = val_misc_correct / place_norm
        val_eot_acc = val_eot_correct / val_total
        val_eot_pos_recall = (
            val_eot_pos_correct / val_eot_pos_total if val_eot_pos_total > 0 else 0.0
        )

        # Build per-kind metric dict for both stdout and wandb. Skip kinds
        # absent from the val split (e.g. small datasets where some kinds
        # land entirely in train).
        per_kind_metrics: dict[str, float] = {}
        for k in LessonKind:
            n = per_kind_n[k.name]
            if n == 0:
                continue
            per_kind_metrics[f"val/{k.name}/loss"] = per_kind_loss_sum[k.name] / n
            per_kind_metrics[f"val/{k.name}/acc"] = per_kind_correct[k.name] / n
            per_kind_metrics[f"val/{k.name}/tile_acc"] = (
                per_kind_tile_correct[k.name] / n
            )
            per_kind_metrics[f"val/{k.name}/ent_acc"] = per_kind_ent_correct[k.name] / n
            per_kind_metrics[f"val/{k.name}/dir_acc"] = per_kind_dir_correct[k.name] / n
            per_kind_metrics[f"val/{k.name}/item_acc"] = (
                per_kind_item_correct[k.name] / n
            )
            per_kind_metrics[f"val/{k.name}/misc_acc"] = (
                per_kind_misc_correct[k.name] / n
            )
            # EOT acc/recall use full-sample / positive-only denominators (NOT
            # n, which counts placement samples only). Mirrors the global
            # val/eot_acc + val/eot_pos_recall, but per LessonKind so we can see
            # which lessons the stop-signal misfires on.
            eot_n = per_kind_eot_total[k.name]
            per_kind_metrics[f"val/{k.name}/eot_acc"] = (
                per_kind_eot_correct[k.name] / eot_n if eot_n > 0 else 0.0
            )
            eot_pos_n = per_kind_eot_pos_total[k.name]
            per_kind_metrics[f"val/{k.name}/eot_pos_recall"] = (
                per_kind_eot_pos_correct[k.name] / eot_pos_n if eot_pos_n > 0 else 0.0
            )

        val_seconds = time.time() - t_val

        # Rollout eval: greedy-play the held-out factories and record final
        # throughput. Same lessons as val accuracy, so val/thput is directly
        # comparable to val/acc curves. Runs on every eval window unless
        # disabled (it's the slow part of an eval).
        do_rollout = args.eval_rollouts
        if do_rollout and len(val_seeds_to_kind) > 0:
            t_rollout = time.time()
            roll = run_rollout_eval(
                agent,
                args,
                val_seeds_to_kind,
                device,
                max_seeds=args.eval_rollouts_max_seeds,
                eot_threshold=args.rollout_eot_threshold,
                num_envs=args.eval_rollouts_num_envs,
            )
            rollout_seconds = time.time() - t_rollout
            overall_thp = roll["overall"]
            per_kind_thp_n = roll["per_kind_n"]
            # val/thput ignores the EOT head (true build skill, and the
            # default checkpoint-selection metric); val/thput_eot stops
            # at the first EOT fire (what the model's own stop head produces).
            per_kind_metrics["val/thput"] = overall_thp
            per_kind_metrics["val/thput_eot"] = roll["overall_eot"]
            per_kind_metrics["val/rollout_seconds"] = rollout_seconds
            for kn, thp in roll["per_kind"].items():
                if per_kind_thp_n[kn] > 0:
                    per_kind_metrics[f"val/{kn}/thput"] = thp
            for kn, thp in roll["per_kind_eot"].items():
                if per_kind_thp_n[kn] > 0:
                    per_kind_metrics[f"val/{kn}/thput_eot"] = thp
            # Recipe-pick accuracy from the same rollout: fraction of the
            # assemblers the agent placed that got the right recipe. Only logged
            # for factories that actually have an assembler (so it appears once
            # the agent starts placing them, and never for belt-only lessons).
            per_kind_asm_n = roll["per_kind_asm_n"]
            if sum(per_kind_asm_n.values()) > 0:
                per_kind_metrics["val/asm_item_acc"] = roll["asm_item_acc"]
            for kn, acc in roll["per_kind_asm_item_acc"].items():
                if per_kind_asm_n[kn] > 0:
                    per_kind_metrics[f"val/{kn}/asm_item_acc"] = acc
        else:
            overall_thp = None
            rollout_seconds = None
            per_kind_thp_n = {}

        pbar.write(
            f"{samples_seen:>{len(str(total_samples))}}/{total_samples} samples "
            f"(epoch {epoch}/{args.epochs}) | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"train_eot_acc={train_eot_acc:.3f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} "
            f"(tile={val_tile_acc:.3f} ent={val_ent_acc:.3f} dir={val_dir_acc:.3f} "
            f"item={val_item_acc:.3f} misc={val_misc_acc:.3f} "
            f"eot={val_eot_acc:.3f} eot+={val_eot_pos_recall:.3f})"
            + (
                f"  val_thp={overall_thp:.3f} ({rollout_seconds:.1f}s)"
                if overall_thp is not None
                else ""
            )
            + (
                f" | t: train={train_seconds:.1f}s "
                f"(data={train_data_s:.1f}s compute={train_compute_s:.1f}s) "
                f"val={val_seconds:.1f}s"
            )
        )
        if per_kind_metrics:
            kind_summary = "  ".join(
                f"{k.name}={per_kind_correct[k.name] / per_kind_n[k.name]:.3f}"
                for k in LessonKind
                if per_kind_n[k.name] > 0
            )
            pbar.write(f"  per-kind val_acc: {kind_summary}")

        if args.track and run is not None:
            # Drive wandb's underlying _step with samples_seen (cumulative
            # training pairs the model has been updated on) instead of
            # letting it auto-increment once per epoch. This is what makes
            # the x-axis measure "optimisation pressure applied": two runs
            # with the same epoch count but different num_samples now spread
            # across proportionally different x-ranges. Crucially, _step is
            # the axis EVERY default panel plots against — including ones
            # already pinned to "Step" in an existing workspace — so this
            # fixes panels (e.g. val/<kind>/thput) that define_metric's
            # step_metric alone could not retarget. samples_seen rather than
            # global_step because it stays comparable across batch_size
            # changes; both are still logged as metrics so either can be
            # picked as a custom x-axis in the UI. (_step must increase
            # monotonically — samples_seen does, by construction.)
            run.log(
                {
                    "train/loss": train_loss,
                    "train/loss_tile": train_loss_tile,
                    "train/loss_ent": train_loss_ent,
                    "train/loss_dir": train_loss_dir,
                    "train/loss_item": train_loss_item,
                    "train/loss_misc": train_loss_misc,
                    "train/loss_eot": train_loss_eot,
                    "train/acc": train_acc,
                    "train/eot_acc": train_eot_acc,
                    "val/loss": val_loss,
                    "val/loss_tile": val_loss_tile,
                    "val/loss_ent": val_loss_ent,
                    "val/loss_dir": val_loss_dir,
                    "val/loss_item": val_loss_item,
                    "val/loss_misc": val_loss_misc,
                    "val/loss_eot": val_loss_eot,
                    "val/acc": val_acc,
                    "val/tile_acc": val_tile_acc,
                    "val/ent_acc": val_ent_acc,
                    "val/dir_acc": val_dir_acc,
                    "val/item_acc": val_item_acc,
                    "val/misc_acc": val_misc_acc,
                    "val/eot_acc": val_eot_acc,
                    "val/eot_pos_recall": val_eot_pos_recall,
                    "train/epoch": epoch,
                    "train/samples_seen": samples_seen,
                    "train/global_step": global_step,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": (grad_norm_sum / grad_norm_count)
                    if grad_norm_count > 0
                    else float("nan"),
                    # Real wall-clock (no added syncs); train splits into
                    # DataLoader wait vs compute to expose data-vs-compute bound.
                    "perf/train_seconds": train_seconds,
                    "perf/train_data_seconds": train_data_s,
                    "perf/train_compute_seconds": train_compute_s,
                    "perf/val_seconds": val_seconds,
                    **per_kind_metrics,
                    **_direction_diagnostics(dir_true_eval, dir_pred_eval),
                    **_dir_distance_diagnostics(dist_eval, dir_mismatch_eval),
                },
                step=samples_seen,
                commit=True,
            )

        # Track best val accuracy for the summary, but DON'T select on it.
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Default checkpoint-selection metric: greedy throughput (EOT ignored).
        # overall_thp is None on evals where no rollout ran.
        if overall_thp is not None and overall_thp >= best_val_throughput:
            best_val_throughput = overall_thp
            torch.save(agent.state_dict(), args.checkpoint_path)
            ever_saved = True
            pbar.write(f"  -> Saved best checkpoint (val/thput {overall_thp:.3f})")

    pbar.close()
    if not ever_saved:
        # Rollouts disabled or no val factories — fall back to the final model
        # so a checkpoint always exists.
        torch.save(agent.state_dict(), args.checkpoint_path)
    total_time = time.time() - t0
    print(
        f"\nBest val/thput: {best_val_throughput:.3f}  (best val_acc {best_val_acc:.3f})"
    )
    print(f"Checkpoint saved to: {args.checkpoint_path}")

    # Write summary JSON (total_time includes dataset generation + training)
    summary = {
        "best_val_throughput": round(best_val_throughput, 4),
        "best_val_acc": round(best_val_acc, 4),
        "val_tile_acc": round(val_tile_acc, 4),
        "val_ent_acc": round(val_ent_acc, 4),
        "val_dir_acc": round(val_dir_acc, 4),
        "val_item_acc": round(val_item_acc, 4),
        "val_misc_acc": round(val_misc_acc, 4),
        "val_eot_acc": round(val_eot_acc, 4),
        "val_eot_pos_recall": round(val_eot_pos_recall, 4),
        "val_loss": round(val_loss, 4),
        "num_samples": args.num_samples,
        "epochs": args.epochs,
        # Total optimisation pressure applied — the magnitudes the wandb
        # x-axis ends at. Records how much training actually happened
        # independent of the epoch count.
        "samples_seen": samples_seen,
        "optimizer_steps": global_step,
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
        # Headline metric in the W&B run table: greedy throughput (EOT
        # ignored), the same number that selected the checkpoint.
        run.summary["best_val_throughput"] = best_val_throughput
        run.summary["best_val_acc"] = best_val_acc

        # Upload the best checkpoint + summary so a Mac can grab the trained
        # model with `wandb artifact get` without rummaging through the pod.
        #
        # Name encodes hyperparams so runs with identical config collapse
        # into versions of one artifact, while different configs land in
        # separate artifacts — easier to navigate than a wall of
        # `sft-checkpoint-vN`. val_acc varies per run, so it stays as an
        # alias instead of being baked into the name.
        artifact = wandb.Artifact(
            name=_artifact_name(args),
            type="model",
            metadata={
                "best_val_throughput": best_val_throughput,
                "best_val_acc": best_val_acc,
                "size": args.size,
                "layers": layers_from_args(args),
                "kernel_size": args.kernel_size,
                "num_samples": args.num_samples,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
            },
        )
        artifact.add_file(args.checkpoint_path)
        artifact.add_file(summary_path)
        run.log_artifact(
            artifact,
            aliases=[
                "latest",
                f"thp{best_val_throughput:.3f}",
                f"val{best_val_acc:.3f}",
            ],
        )
        print(f"Logged W&B artifact: {artifact.name}")
        run.finish()

    return agent


if __name__ == "__main__":
    args = tyro.cli(SftArgs)
    train_sft(args)
