"""Denoising pretraining of the whole-factory UNet (`denoise.py`).

Every training example is a known-correct factory from `build_factory` with a
random fraction t ~ U[0, 1] of its free tiles resampled from noise; the model
always regresses the clean factory. The eval generates factories for held-out
lessons from pure noise (best of `eval_best_of` chains each) and scores them
in the throughput simulator. With `self_frac` > 0 each eval also builds
training-seed trial factories and replays the model's top builds by PPO's
score as extra clean targets (expert iteration): trials have no reference to
denoise towards otherwise.

Usage:
    python sft.py --num-samples 2000000
"""

import collections
import copy
import io
import itertools
import json
import math
import multiprocessing
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional, TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import tyro
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from denoise import (  # noqa: E402
    CH_PART,
    DenoiserUNet,
    N_TYPES,
    _luts,
    corrupt,
    free_mask,
    generate_chains,
    improve_world,
    legalize,
    load_denoiser,
    run_denoise_eval,
    score_world,
    state_to_tokens,
    tokens_to_state,
    world_to_state,
)
from factorion import (  # noqa: E402
    LESSON_IS_TRIAL,
    Channel,
    Factory,
    TRAINING_KINDS,
    LessonKind,
    build_factory,
    count_dangling_inserters,
    render_factory,
)

from ppo import (  # noqa: E402
    FactorioEnv,
    assert_device_ok,
    cuda_env_info,
    _build_eval_set,
    _resolve_start_from,
    _upload_replacing,
    _CH_ENT,
    _CH_ITEMS,
    _ASM_MACHINE_ENT_ID,
)
from training_config import PpoArgs, SftArgs, wsd_multiplier  # noqa: E402

# Training factories come from far above the held-out eval seeds (9M-11M).
_TRAIN_SEED_BASE = 1_000_000_000
# Self-generated targets are built from the upper half of the run's training
# seed block: past any seed its loader reaches, far above the eval seeds.
_SELF_SEED_OFFSET = 50_000_000
# Noise levels the val/ denoising metrics are measured at.
_VAL_NOISE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _humanize_count(n: int) -> str:
    """50_000 -> '50k', 2_500_000 -> '2.5M'. Used in artifact names so
    `n50k` reads better than `n50000`."""
    if n >= 1_000_000:
        v = n / 1_000_000
        s = f"{v:.1f}".rstrip("0").rstrip(".")
        return f"{s}M"
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
    config-changing runs (different size / sample count / lr / architecture)
    get their own artifact."""
    return (
        f"dn-s{args.size}"
        f"-n{_humanize_count(args.num_samples)}"
        f"-bs{args.batch_size}"
        f"-lr{_humanize_lr(args.lr)}"
        f"-w{args.width}x{args.levels}b{args.blocks}a{args.attn_layers}"
    )


# Max consecutive build failures tolerated for a single lesson kind before it is
# treated as unbuildable at the current grid size and dropped from the sampler.
# build_factory already exhausts a full rejection-sampling budget per call, so a
# None is a strong "can't build" signal; an unbuildable kind (e.g. a 3x3
# assembler with 5 ingredient arms on a size-5 grid) returns None for every seed.
# Without this cap the sampler would keep re-drawing that kind forever and hang.
_MAX_BUILD_FAILURES_PER_KIND = 100


def _iter_factories(size, base_seed, worker_id, num_workers):
    """Yield (state uint8, max_throughput) for fresh lesson factories forever.

    Worker `w` of `num_workers` walks seeds ≡ base_seed+w (mod num_workers), so
    concurrent workers never share a factory. Kinds are drawn uniformly.
    """
    # Trial kinds have no known solution, so there is no clean factory to
    # denoise towards. Trials are RL-only; PPO samples them directly.
    kinds = [k for k in TRAINING_KINDS if not LESSON_IS_TRIAL[k]]
    # Kinds still believed buildable at this size, plus a per-kind counter of
    # consecutive build failures. A kind that can't fit the grid returns None for
    # every seed; once it exhausts the failure budget it is dropped so the
    # sampler doesn't spin on it forever (see _MAX_BUILD_FAILURES_PER_KIND).
    available = list(kinds)
    consecutive_fails = {k.name: 0 for k in kinds}
    rng = random.Random(base_seed + worker_id)

    seed = base_seed + worker_id
    while True:
        kind = rng.choice(available)
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
        yield torch.from_numpy(world_to_state(factory.world_CWH)), float(factory.max_throughput)


class FactoryStream(IterableDataset):
    """Generates training factories on the fly, sharded across DataLoader workers.

    CPU generation overlaps GPU training via DataLoader prefetch.
    """

    def __init__(self, size, base_seed):
        self.size = size
        self.base_seed = base_seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        return _iter_factories(self.size, self.base_seed, worker_id, num_workers)


def build_lr_schedule(optimizer, total_steps: int, args: "SftArgs"):
    """Warmup-Stable-Decay (`wsd_multiplier`), stepped once per optimizer step.

    Only the cooldown depends on `total_steps`, so runs of different budgets
    share their LR curve until their cooldowns diverge — and a cooldown can be
    branched off a stable-phase checkpoint
    (`--start-from <ckpt> --warmup-steps 0 --cooldown-frac 1`) instead of
    repeating the shared prefix.
    """
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: wsd_multiplier(
            step, total_steps, args.warmup_steps, args.cooldown_frac, args.min_lr_ratio
        ),
    )


class RolloutEval(TypedDict):
    """Return shape of :func:`run_rollout_eval`."""

    # The overall means pool lessons only. Trials measure a different thing —
    # inventing a factory with nothing to imitate, versus reproducing a known
    # one — and pooling them would silently move the headline number that the
    # SFT baseline and every historical run are quoted in.
    overall: float  # mean lesson throughput at the moment the model stopped
    trial_overall: float  # overall over trial kinds only (0.0 when none ran)
    trial_n: int  # number of trial factories the eval scored
    per_kind: dict[str, float]  # overall, keyed by LessonKind.name
    per_kind_n: dict[str, int]  # number of val factories per LessonKind.name
    asm_item_acc: float  # frac of placed assemblers given the correct recipe
    per_kind_asm_item_acc: dict[str, float]  # asm_item_acc, keyed by LessonKind.name
    per_kind_asm_n: dict[str, int]  # assembler placements scored per LessonKind.name
    eot_acc: float  # frac of rollout steps whose EOT prediction matches "is the factory done?"
    eot_pos_recall: float  # frac of done states where the EOT head actually fired
    per_kind_eot_acc: dict[str, float]  # eot_acc, keyed by LessonKind.name
    per_kind_eot_pos_recall: dict[str, float]  # eot_pos_recall, keyed by LessonKind.name
    per_kind_eot_step_n: dict[str, int]  # rollout steps scored per LessonKind.name
    per_kind_eot_pos_n: dict[str, int]  # done (should-stop) states seen per LessonKind.name
    # `count_dangling_inserters` totals (no-input, no-output) over the scored
    # factories — should be zero — lessons/trials pooled apart like `overall`.
    dangling_inserters: tuple[int, int]
    trial_dangling_inserters: tuple[int, int]
    per_kind_dangling_inserters: dict[str, tuple[int, int]]


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
    num_envs: int = 8,
    best_of: int = 1,
    records: Optional[list[dict]] = None,
) -> RolloutEval:
    """Best-of-N sampled rollout eval on the held-out val factories.

    Each held-out (seed, kind) gets `best_of` rollouts sampled from the policy
    (temperature 1, tile pick restricted to legal tiles, as a PPO rollout
    samples) and scores the best of them. A rollout steps until the sampled EOT
    action fires — which ends it, exactly as it terminates a PPO episode — or
    the env finishes (throughput==1.0 or max_steps) without it ever firing.
    Throughput is the last `info['thput_normed']` the env reported for the
    state the model stopped at: raw items/sec divided by the per-factory
    max, in [0, 1], so a perfectly-rebuilt factory scores 1.0 regardless of
    its absolute belt speed (the env already calls the Rust solver every
    step, so we don't re-run it).

    Every (factory, sample) pair is a work item for K=num_envs*best_of
    FactorioEnvs run in parallel with the CNN forward batched across them;
    slots refill from the queue as they finish, so all K stay busy until it
    drains. Each sample draws from its own noise stream seeded by (args.seed,
    factory seed, sample index), so the result is reproducible, independent
    of K and of batching order, and every checkpoint faces the same random
    numbers.

    The (seed, kind) pairs are exactly the val_accuracy set. The result is
    logged as `val/thput@N`, directly comparable to the existing per-kind val
    accuracy curves. Eval-only; training is untouched.

    Every other output (EOT and recipe-pick scoring, dangling inserters,
    `records`) comes from each factory's selected sample. `records`, when
    given, collects one dict per scored factory (seed, kind, thput,
    entity_cost and the ASCII render of the world the model stopped at) — the
    raw material for a qualitative diff of two policies (`factory_diff.py`),
    which the aggregates alone cannot show.

    Returns a dict with:
        overall — mean best-of-N throughput at the state the model stopped at;
        per_kind — the same, keyed by LessonKind.name;
        per_kind_n — sample count per kind in the eval;
        eot_acc, eot_pos_recall — EOT-head accuracy / positive-class recall;
        per_kind_eot_acc, per_kind_eot_pos_recall — same, keyed by kind.
    """
    was_training = agent.training
    agent.eval()

    # Blank the WHOLE factory (size*size) so the eval builds from an
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
    # Lessons and trials accumulate separately — see RolloutEval.
    all_throughputs: list[float] = []
    trial_throughputs: list[float] = []
    per_kind_asm_correct: dict[str, int] = {k.name: 0 for k in LessonKind}
    per_kind_asm_total: dict[str, int] = {k.name: 0 for k in LessonKind}
    # EOT-head scoring: every scored step counts toward *_step; done states
    # (should-stop positives) also count toward *_pos.
    per_kind_eot_correct: dict[str, int] = {k.name: 0 for k in LessonKind}
    per_kind_eot_step_total: dict[str, int] = {k.name: 0 for k in LessonKind}
    per_kind_eot_pos_correct: dict[str, int] = {k.name: 0 for k in LessonKind}
    per_kind_eot_pos_total: dict[str, int] = {k.name: 0 for k in LessonKind}
    per_kind_dangling: dict[str, tuple[int, int]] = {k.name: (0, 0) for k in LessonKind}

    if not seeds_sorted:
        if was_training:
            agent.train()
        zero = {kn: 0.0 for kn in per_kind_throughputs}
        per_kind_n = {kn: 0 for kn in per_kind_throughputs}
        return {
            "overall": 0.0,
            "trial_overall": 0.0,
            "trial_n": 0,
            "per_kind": zero,
            "per_kind_n": per_kind_n,
            "asm_item_acc": 0.0,
            "per_kind_asm_item_acc": dict(zero),
            "per_kind_asm_n": dict(per_kind_n),
            "eot_acc": 0.0,
            "eot_pos_recall": 0.0,
            "per_kind_eot_acc": dict(zero),
            "per_kind_eot_pos_recall": dict(zero),
            "per_kind_eot_step_n": dict(per_kind_n),
            "per_kind_eot_pos_n": dict(per_kind_n),
            "dangling_inserters": (0, 0),
            "trial_dangling_inserters": (0, 0),
            "per_kind_dangling_inserters": per_kind_dangling,
        }

    # One work item per (factory, sample), a factory's samples adjacent so they
    # share batched forwards: K slots hold num_envs factories' samples at once.
    # Cap K at the number of work items — spinning up more envs than that
    # wastes memory. All envs use idx=0 so the seed we pass to reset() is the
    # seed generate_lesson sees: FactorioEnv.reset adds self.idx to the seed
    # for env-diversity in PPO, but here we need exact seed pass-through to
    # replay the held-out val factories.
    queue = [(s, j) for s in seeds_sorted for j in range(best_of)]
    K = max(1, min(num_envs * best_of, len(queue)))
    # A factory's samples all start from the same blank grid, so it is reset
    # once into `template` and each sample deep-copies that (~40x cheaper).
    template = FactorioEnv(size=args.size, idx=0)
    template_reset: dict[int, tuple] = {}
    envs = [template] * K
    head_sizes = {"tile": args.size * args.size, "entity": agent.num_entities,
                  "direction": agent.num_directions, "item": agent.num_items,
                  "misc": agent.num_misc, "eot": 1}
    active = [True] * K
    # Each slot's in-flight sample: its stats so far and its own noise stream,
    # seeded by (run seed, factory, sample) so neither the batch it rides in nor
    # the checkpoint being scored changes which random numbers it draws.
    cur: list[dict] = [{} for _ in range(K)]
    rngs = [np.random.default_rng() for _ in range(K)]
    finished: dict[int, list[dict]] = {s: [] for s in seeds_sorted}
    # Staged on the host so each step is one host->device copy, not one per env.
    obs_batch = torch.zeros((K, len(Channel), args.size, args.size))

    def load_slot(i: int) -> None:
        """Reset slot `i` onto the next work item, deactivating it once the
        queue is empty."""
        if not queue:
            active[i] = False
            return
        s, j = queue.pop(0)
        k = LessonKind(val_seeds_to_kind[s])
        if s not in template_reset:
            template_reset.clear()
            template_reset[s] = template.reset(
                seed=s,
                options={
                    "num_missing_entities": max_level,
                    "kind": k,
                },
            )
        obs, info = template_reset[s]
        envs[i] = copy.deepcopy(template)
        cur[i] = {
            "seed": s,
            "sample": j,
            "kind": k,
            "thput": float(info.get("thput_normed", 0.0)),
            # Correct recipes from the solved world; empty when that factory has
            # no assembler, which skips the recipe check.
            "asm_recipes": _solved_assembler_recipes(envs[i]._solved_world_CWH),
            **dict.fromkeys(
                ("asm_n", "asm_correct", "eot_steps", "eot_correct", "eot_pos", "eot_pos_correct"), 0
            ),
        }
        rngs[i] = np.random.default_rng((args.seed, s, j))
        obs_batch[i] = torch.as_tensor(obs, dtype=torch.float32)

    def finish_slot(i: int) -> None:
        """Record slot `i`'s finished sample, then refill the slot."""
        c = cur[i]
        c["entity_cost"] = float(envs[i]._entity_cost)
        c["dangling"] = count_dangling_inserters(envs[i]._world_CWH)
        if records is not None:
            c["render"] = render_factory(envs[i]._world_CWH)
        finished[c["seed"]].append(c)
        load_slot(i)

    for i in range(K):
        load_slot(i)

    with torch.no_grad():
        while any(active):
            # Single CNN forward across all K slots. Inactive slots'
            # outputs are discarded; the per-eval cost of K-1 stale
            # forwards at the tail is negligible compared to the
            # batching win earlier in the queue. An inactive slot draws from
            # its finished sample's stream, which nothing reads again.
            u_KL = torch.as_tensor(
                np.stack([r.random(sum(head_sizes.values()), dtype=np.float32) for r in rngs])
            ).to(device)
            noise = dict(zip(head_sizes, u_KL.split(list(head_sizes.values()), dim=1)))
            noise["eot"] = noise["eot"].squeeze(1)
            # The policy's own sample, as a PPO rollout draws it (legal_mask
            # keeps it off walled and source/sink tiles); the critic is unused here.
            out = agent.sample_action(
                obs_batch.to(device), noise=noise, legal_mask=True, compute_value=False
            )
            # One device->host copy per head per step, not one per env per head.
            act = out["action"]
            eot_K = act["eot"].tolist()
            x_K = act["xy"][:, 0].tolist()
            y_K = act["xy"][:, 1].tolist()
            ent_K = act["entity"].tolist()
            dir_K = act["direction"].tolist()
            item_K = act["item"].tolist()
            misc_K = act["misc"].tolist()

            for i in range(K):
                if not active[i]:
                    continue

                c = cur[i]
                pred_stop = bool(eot_K[i])

                # Score the head against ground truth on this pre-action state:
                # it should fire iff the factory is already complete.
                is_done = c["thput"] >= 1.0
                c["eot_steps"] += 1
                c["eot_correct"] += int(pred_stop == is_done)
                if is_done:
                    c["eot_pos"] += 1
                    c["eot_pos_correct"] += int(pred_stop)

                # The sampled EOT action ends the rollout, exactly as it
                # terminates a PPO episode: we score the factory the model
                # declared finished, never one it was forced past.
                if pred_stop:
                    finish_slot(i)
                    continue

                action = {
                    "xy": np.array([int(x_K[i]), int(y_K[i])], dtype=int),
                    "entity": int(ent_K[i]),
                    "direction": int(dir_K[i]),
                    "item": int(item_K[i]),
                    "misc": int(misc_K[i]),
                }
                next_obs, _r, terminated, truncated, info = envs[i].step(action)
                c["thput"] = float(info.get("thput_normed", 0.0))

                # Recipe-pick check: the agent tried to place an assembler in a
                # factory that has one. Count it iff the assembler actually
                # landed at the anchor (invalid placements are env no-ops), then
                # score its recipe against the solved factory's recipes.
                if c["asm_recipes"] and action["entity"] == _ASM_MACHINE_ENT_ID:
                    ax, ay = int(action["xy"][0]), int(action["xy"][1])
                    if int(envs[i]._world_CWH[_CH_ENT, ax, ay]) == _ASM_MACHINE_ENT_ID:
                        c["asm_n"] += 1
                        c["asm_correct"] += int(action["item"] in c["asm_recipes"])

                if not (terminated or truncated):
                    obs_batch[i] = torch.as_tensor(
                        next_obs,
                        dtype=torch.float32,
                    )
                    continue

                # The env ran out of room or steps before the head ever fired.
                finish_slot(i)

    # Each factory scores its best sample; ties go to the cheaper build, then
    # the lower sample index, so the pick doesn't depend on finish order. Every
    # other output comes from that same sample.
    for s in seeds_sorted:
        c = max(finished[s], key=lambda r: (r["thput"], -r["entity_cost"], -r["sample"]))
        kn = c["kind"].name
        per_kind_throughputs[kn].append(c["thput"])
        (trial_throughputs if LESSON_IS_TRIAL[c["kind"]] else all_throughputs).append(c["thput"])
        per_kind_asm_total[kn] += c["asm_n"]
        per_kind_asm_correct[kn] += c["asm_correct"]
        per_kind_eot_step_total[kn] += c["eot_steps"]
        per_kind_eot_correct[kn] += c["eot_correct"]
        per_kind_eot_pos_total[kn] += c["eot_pos"]
        per_kind_eot_pos_correct[kn] += c["eot_pos_correct"]
        no_input, no_output = c["dangling"]
        acc_in, acc_out = per_kind_dangling[kn]
        per_kind_dangling[kn] = (acc_in + no_input, acc_out + no_output)
        if records is not None:
            records.append(
                {
                    "seed": s,
                    "kind": kn,
                    "thput": c["thput"],
                    "entity_cost": c["entity_cost"],
                    "inserters_no_input": no_input,
                    "inserters_no_output": no_output,
                    "render": c["render"],
                }
            )

    overall = float(np.mean(all_throughputs)) if all_throughputs else 0.0
    trial_overall = float(np.mean(trial_throughputs)) if trial_throughputs else 0.0
    per_kind = {
        kn: (float(np.mean(ts)) if ts else 0.0)
        for kn, ts in per_kind_throughputs.items()
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

    eot_step_all = sum(per_kind_eot_step_total.values())
    eot_pos_all = sum(per_kind_eot_pos_total.values())
    eot_acc = sum(per_kind_eot_correct.values()) / eot_step_all if eot_step_all else 0.0
    eot_pos_recall = (
        sum(per_kind_eot_pos_correct.values()) / eot_pos_all if eot_pos_all else 0.0
    )
    per_kind_eot_acc = {
        kn: (per_kind_eot_correct[kn] / n if n else 0.0)
        for kn, n in per_kind_eot_step_total.items()
    }
    per_kind_eot_pos_recall = {
        kn: (per_kind_eot_pos_correct[kn] / n if n else 0.0)
        for kn, n in per_kind_eot_pos_total.items()
    }

    def pooled_dangling(trial: bool) -> tuple[int, int]:
        pairs = [per_kind_dangling[k.name] for k in LessonKind if LESSON_IS_TRIAL[k] == trial]
        return sum(p[0] for p in pairs), sum(p[1] for p in pairs)

    if was_training:
        agent.train()
    return {
        "overall": overall,
        "trial_overall": trial_overall,
        "trial_n": len(trial_throughputs),
        "per_kind": per_kind,
        "per_kind_n": per_kind_n,
        "asm_item_acc": asm_item_acc,
        "per_kind_asm_item_acc": per_kind_asm_item_acc,
        "per_kind_asm_n": dict(per_kind_asm_total),
        "eot_acc": eot_acc,
        "eot_pos_recall": eot_pos_recall,
        "per_kind_eot_acc": per_kind_eot_acc,
        "per_kind_eot_pos_recall": per_kind_eot_pos_recall,
        "per_kind_eot_step_n": dict(per_kind_eot_step_total),
        "per_kind_eot_pos_n": dict(per_kind_eot_pos_total),
        "dangling_inserters": pooled_dangling(trial=False),
        "trial_dangling_inserters": pooled_dangling(trial=True),
        "per_kind_dangling_inserters": per_kind_dangling,
    }


def noise_levels(n, args: SftArgs, device):
    """Per-example corruption level: U[0, 1] plus point masses at 0 (a clean
    factory must map to itself, which is what stops generation) and 1 (pure
    noise, where generation starts)."""
    u = torch.rand((2, n), device=device)
    t = u[0]
    t = torch.where(u[1] < args.noise_zero_frac, 0.0, t)
    return torch.where(u[1] > 1 - args.noise_one_frac, 1.0, t)


def noise_prior_from(states: torch.Tensor) -> torch.Tensor:
    """The noise distribution over types: their frequency on the free tiles of
    `states`, mixed with 5% uniform so every type can appear."""
    typ = state_to_tokens(states)[0][free_mask(states)]
    freq = torch.bincount(typ, minlength=N_TYPES).float()
    return 0.95 * freq / freq.sum() + 0.05 / N_TYPES


def tokens_logp(model, logits, toks, free):
    """Per-example log-probability of `toks` under `logits`, averaged over
    free tiles."""
    freef = free.float()
    return (model.logp(logits, toks) * freef).sum((1, 2)) / freef.sum((1, 2)).clamp(min=1)


def throughput_pg_loss(model, logits, state, free, max_thput, t, args: SftArgs):
    """REINFORCE on simulated throughput. Up to `rl_inputs` lightly-corrupted
    examples (t ≤ rl_max_noise, where a sample is a plausible factory) each get
    `rl_samples` factories sampled from the prediction; each is legalized and
    scored with PPO's reward (raw throughput × cost efficiency ÷ reference) and
    reinforced against its siblings' mean (RLOO). Returns (loss, mean score)."""
    idx = torch.nonzero(t <= args.rl_max_noise).squeeze(1)[: args.rl_inputs]
    if len(idx) == 0:
        return logits.new_zeros(()), 0.0
    K = args.rl_samples
    rep = idx.repeat_interleave(K)
    lg, fr = logits[rep], free[rep]
    with torch.no_grad():
        toks = model.pick(lg.detach(), temperature=1.0)
        conf = model.conf(lg.detach(), toks)
    cond = state[rep, :CH_PART].cpu().numpy().astype(np.int64)
    typ, rec, part = (x.cpu().numpy() for x in toks)
    conf_np, mt = conf.float().cpu().numpy(), max_thput[rep].tolist()
    scores = torch.tensor(
        [
            score_world(
                legalize(cond[i], typ[i], rec[i], part[i] if model.io == "token" else None, conf_np[i]),
                mt[i],
                PpoArgs.entity_cost_scale,
            )["score"]
            for i in range(len(rep))
        ],
        device=logits.device,
    ).view(-1, K)
    adv = (scores - (scores.sum(1, keepdim=True) - scores) / (K - 1)).view(-1)
    return -(adv * tokens_logp(model, lg, toks, fr)).mean(), float(scores.mean())


def denoise_step(model, state, max_thput, args: SftArgs):
    """One training batch: corrupt, predict, (unroll), loss. Returns (loss,
    stats) where stats holds detached per-term values."""
    free = free_mask(state)
    clean = state_to_tokens(state)
    t = noise_levels(state.shape[0], args, state.device)
    noisy = corrupt(clean, free, t, model.noise_prior)
    logits = model(tokens_to_state(state, free, *noisy))
    terms = model.losses(logits, clean, free)
    if args.unroll:
        # SUNDAE: also denoise the model's own sample of its prediction, the
        # kind of input every generation step after the first sees.
        with torch.no_grad():
            own = model.pick(logits.detach(), temperature=1.0)
        terms2 = model.losses(model(tokens_to_state(state, free, *own)), clean, free)
        terms = {k: (v + terms2[k]) / 2 for k, v in terms.items()}
    weights = {"type": args.lw_type, "recipe": args.lw_recipe, "part": args.lw_part, "cost": args.lw_cost}
    loss = sum(weights.get(k, 1.0) * v for k, v in terms.items())
    stats = {k: v.detach() for k, v in terms.items()}
    if args.lw_thput > 0:
        pg, mean_score = throughput_pg_loss(model, logits, state, free, max_thput, t, args)
        loss = loss + args.lw_thput * pg
        stats["thput_pg"] = pg.detach()
        stats["rl_score"] = torch.tensor(mean_score)
    with torch.no_grad():
        typ_pred = model.pick(logits.detach())[0]
        stats["type_acc"] = ((typ_pred == clean[0]) & free).sum() / free.sum().clamp(min=1)
    return loss, stats


@torch.no_grad()
def val_denoise_metrics(model, val_states, args: SftArgs, device) -> dict:
    """Denoising accuracy of the clean-factory prediction at fixed noise
    levels on held-out factories (same noise draw every eval)."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(args.seed)
    out: dict = {}
    L = _luts(device)
    # `factored` predicts no parts: its footprints are re-derived by `legalize`.
    heads = ("type", "recipe", "part") if model.io == "token" else ("type", "recipe")
    for t in _VAL_NOISE_LEVELS:
        hits = dict.fromkeys(heads, 0.0)
        counts = dict.fromkeys(heads, 0.0)
        exact, loss_sum, n_tiles = 0, 0.0, 0.0
        for lo in range(0, len(val_states), args.batch_size):
            s = val_states[lo : lo + args.batch_size].to(device).long()
            free = free_mask(s)
            clean = state_to_tokens(s)
            noisy = corrupt(clean, free, torch.full((len(s),), t, device=device), model.noise_prior, gen)
            logits = model(tokens_to_state(s, free, *noisy))
            pred = model.pick(logits)
            masks = {"type": free, "recipe": L.is_asm[clean[0]] & free, "part": (L.n_parts[clean[0]] > 1) & free}
            ok_all = torch.ones(len(s), dtype=torch.bool, device=device)
            for name, p, c in zip(heads, pred, clean):
                m = masks[name]
                ok = p == c
                hits[name] += float((ok & m).sum())
                counts[name] += float(m.sum())
                ok_all &= (ok | ~m).flatten(1).all(1)
            exact += int(ok_all.sum())
            n = float(free.sum())
            loss_sum += n * float(sum(v for k, v in model.losses(logits, clean, free).items() if k != "cost"))
            n_tiles += n
        tag = f"val/t{t:.2f}"
        out[f"{tag}/loss"] = loss_sum / max(n_tiles, 1)
        out[f"{tag}/exact"] = exact / len(val_states)
        for name in hits:
            out[f"{tag}/{name}_acc"] = hits[name] / max(counts[name], 1)
    for key in ("loss", *(f"{h}_acc" for h in heads)):
        out[f"val/{key}"] = float(np.mean([out[f"val/t{t:.2f}/{key}"] for t in _VAL_NOISE_LEVELS]))
    # A clean factory left untouched: the stopping criterion of generation.
    out["val/identity"] = out["val/t0.00/exact"]
    return out


def generation_metrics(model, factories, args: SftArgs, device, records=None) -> dict:
    """val/ metrics of `run_denoise_eval`, keyed `<metric>@N` (N = chains)."""
    N = args.eval_best_of
    roll = run_denoise_eval(
        model,
        factories,
        device,
        best_of=N,
        steps=args.gen_steps,
        refine=args.gen_refine,
        temperature=args.gen_temperature,
        cost_scale=PpoArgs.entity_cost_scale,
        seed=args.seed,
        batch_chains=args.eval_batch_chains,
        records=records,
    )
    out = {
        f"val/converged_frac@{N}": roll["converged_frac"],
        f"val/refine_steps@{N}": roll["refine_steps"],
        f"val/inserters_no_input@{N}": roll["dangling"][0],
        f"val/inserters_no_output@{N}": roll["dangling"][1],
    }
    if roll["n_trials"]:
        out[f"val/trial_inserters_no_input@{N}"] = roll["dangling"][2]
        out[f"val/trial_inserters_no_output@{N}"] = roll["dangling"][3]
    for key, v in roll.items():
        if key.startswith("thput@"):
            out[f"val/{key}"] = v
        elif key.startswith("trial_thput@") and roll["n_trials"]:
            out[f"val/{key}"] = v
        elif key.startswith("per_kind@"):
            n = key.split("@")[1]
            out |= {f"val/{kind}/thput@{n}": x for kind, x in v.items()}
    return out


def self_seeds(args: SftArgs) -> Iterator[int]:
    return itertools.count(_TRAIN_SEED_BASE + args.seed * 100_000_000 + _SELF_SEED_OFFSET)


def select_self_targets(factories, results, top_frac: float) -> list[tuple[LessonKind, dict, Factory]]:
    """The top `top_frac` of each kind's builds by PPO's score (the cheaper
    first among ties), keeping only those that flow and leave no inserter
    dangling, and at most one per factory so no layout is replayed as several
    near-copies. Kinds are ranked apart so the hard ones, trials above all,
    still contribute their best. Returns (kind, build, Factory) per target."""
    by_kind: dict = {}
    for (s, kind, factory), rs in zip(factories, results):
        by_kind.setdefault(kind, []).extend((r, s, factory) for r in rs)
    out, taken = [], set()
    for kind, builds in by_kind.items():
        builds.sort(key=lambda b: (-b[0]["score"], b[0]["entity_cost"]))
        for r, s, factory in builds[: math.ceil(top_frac * len(builds))]:
            if s in taken or r["thput_raw"] <= 0 or count_dangling_inserters(torch.from_numpy(r["world"])) != (0, 0):
                continue
            taken.add(s)
            out.append((kind, r, factory))
    return out


def repair_targets(targets, args: SftArgs) -> list[tuple[LessonKind, dict, Factory]]:
    """`improve_world` on every target, across a process pool (each takes
    ~1 s of simulation per step). A repair that leaves an inserter dangling
    is dropped, keeping the build as the model made it."""
    jobs = [(r["world"], f.max_throughput, PpoArgs.entity_cost_scale, args.self_repair_steps) for _, r, f in targets]
    with multiprocessing.get_context("forkserver").Pool(min(32, os.cpu_count() or 1)) as pool:
        worlds = pool.starmap(improve_world, jobs)
    out = []
    for (kind, r, f), w in zip(targets, worlds):
        if count_dangling_inserters(torch.from_numpy(w)) == (0, 0):
            r = {**score_world(w, f.max_throughput, PpoArgs.entity_cost_scale), "world": w, "model_score": r["score"]}
        out.append((kind, r, f))
    return out


def self_round(ema, kinds, seeds, buffer, args: SftArgs, device, generator) -> dict:
    """Build `self_factories` training-seed factories (cycling `kinds`) best
    of `eval_best_of` with the EMA model, repair the selected builds and add
    them to `buffer` (keyed by world bytes, so a repeat refreshes its place
    rather than duplicating), and return the round's self/ metrics."""
    factories = []
    while len(factories) < args.self_factories:
        s = next(seeds)
        kind = kinds[s % len(kinds)]
        if (f := build_factory(size=args.size, kind=kind, seed=s)) is not None:
            factories.append((s, kind, f))
    results, _, _ = generate_chains(
        ema, factories, device, best_of=args.eval_best_of, steps=args.gen_steps, refine=args.gen_refine,
        temperature=args.gen_temperature, cost_scale=PpoArgs.entity_cost_scale, generator=generator,
        batch_chains=args.eval_batch_chains,
    )
    targets = select_self_targets(factories, results, args.self_top_frac)
    if args.self_repair_steps and targets:
        targets = repair_targets(targets, args)
    for kind, r, factory in targets:
        key = r["world"].tobytes()
        buffer.pop(key, None)
        buffer[key] = (torch.from_numpy(world_to_state(r["world"])), factory.max_throughput, kind)
    while len(buffer) > args.self_buffer_size:
        buffer.popitem(last=False)

    metrics: dict = {"self/buffer_size": len(buffer)}
    for kind in kinds:
        kept = [r for k, r, _ in targets if k == kind]
        n = sum(k == kind for _, k, _ in factories)
        metrics[f"self/{kind.name}/factory_accept_frac"] = len(kept) / max(n, 1)
        metrics[f"self/{kind.name}/target_thput"] = float(np.mean([r["thput"] for r in kept])) if kept else 0.0
        gains = [r["score"] - r["model_score"] for r in kept if "model_score" in r]
        metrics[f"self/{kind.name}/repair_gain"] = float(np.mean(gains)) if gains else 0.0
        metrics[f"self/{kind.name}/buffer_frac"] = sum(e[2] == kind for e in buffer.values()) / max(len(buffer), 1)
    return metrics


def build_eval_factories(args: SftArgs) -> list:
    """The held-out (seed, kind, Factory) set: PPO's eval seeds (trials
    included), so a generation eval is directly comparable to PPO's."""
    seeds_to_kind = _build_eval_set(SimpleNamespace(size=args.size, seed=args.seed, eval_seeds_per_kind=args.eval_seeds_per_kind))
    return [(s, LessonKind(k), build_factory(size=args.size, kind=LessonKind(k), seed=s)) for s, k in seeds_to_kind.items()]


def _upload_checkpoint(run, args, best_thput, files, previous):
    """Upload the run's best checkpoint via `_upload_replacing`."""
    return _upload_replacing(
        run,
        _artifact_name(args),
        files,
        {"best_val_throughput": best_thput, **{k: v for k, v in vars(args).items() if isinstance(v, (int, float, str))}},
        ["latest", f"thp{best_thput:.3f}"],
        previous,
    )


def train_sft(args: SftArgs):
    """Main denoising training loop."""
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(line_buffering=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    if args.batch_size % args.noise_repeats:
        raise ValueError("batch_size must be a multiple of noise_repeats")
    if not 0 <= args.self_frac < 0.5:
        raise ValueError("self_frac must be in [0, 0.5): real factories must stay the majority")
    if args.lw_thput > 0 and args.rl_samples < 2:
        raise ValueError("the throughput loss needs rl_samples >= 2 for its leave-one-out baseline")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    assert_device_ok(device)

    train_seed = _TRAIN_SEED_BASE + args.seed * 100_000_000
    factories = build_eval_factories(args)
    val_states = torch.stack(
        [torch.from_numpy(world_to_state(f.world_CWH)) for _, k, f in factories if not LESSON_IS_TRIAL[k]]
    )

    if args.start_from is not None:
        print(f"Loading model weights from {args.start_from}")
        model = load_denoiser(_resolve_start_from(args.start_from, args.wandb_project_name, args.wandb_entity))
    else:
        model = DenoiserUNet(
            width=args.width,
            levels=args.levels,
            blocks=args.blocks,
            attn_layers=args.attn_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
            io=args.io,
        )
        prior_states = [s for s, _ in itertools.islice(_iter_factories(args.size, train_seed, 0, 1), 256)]
        model.noise_prior.copy_(noise_prior_from(torch.stack(prior_states)))
    print(f"Denoiser has {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    model.to(device)
    model.amp = args.amp and device.type == "cuda"
    ema = copy.deepcopy(model).requires_grad_(False)
    ema.eval()
    # The compiled module shares the model's parameters; generation and the
    # val metrics run on the EMA copy, uncompiled (their shapes vary).
    fwd = torch.compile(model, dynamic=False) if args.compile and device.type == "cuda" else model

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    total_steps = max(1, math.ceil(args.num_samples / args.batch_size))
    scheduler = build_lr_schedule(optimizer, total_steps, args)

    workers = min(16, os.cpu_count() or 1)
    n_real = args.batch_size // args.noise_repeats
    # Self-generated targets overwrite extra rows of a fixed-shape batch (real
    # factories until the buffer has any): growing the batch once the buffer
    # filled halved training speed on GPU.
    n_self = round(args.self_frac * n_real)
    loader = DataLoader(
        FactoryStream(args.size, train_seed),
        batch_size=n_real + n_self,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
        multiprocessing_context="forkserver",
    )

    run = None
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config={**vars(args), **cuda_env_info()},
            name=_artifact_name(args),
            group=args.wandb_group,
            tags=["sft", "denoise"] + (args.tags or []),
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
        )

    eval_every = args.eval_every_n_samples
    if eval_every is None:
        eval_every = 250_000 if args.num_samples <= 2_000_000 else 1_000_000
    next_eval = eval_every
    best_thput, uploaded = -1.0, None
    key = f"val/thput@{args.eval_best_of}"
    # The self-training replay buffer, oldest first; its own RNGs, so
    # self_frac=0 leaves training's random stream untouched.
    buffer: collections.OrderedDict = collections.OrderedDict()
    self_kinds = sorted({k for _, k, _ in factories if k.name in args.self_kinds}, key=lambda k: k.value)
    seeds, self_rng = self_seeds(args), random.Random(args.seed)
    self_gen = torch.Generator(device=device).manual_seed(args.seed)
    self_rows = window_rows = 0
    samples_seen = step = 0
    acc: dict[str, torch.Tensor] = {}
    n_acc = 0
    t_window, data_s, window_samples = time.time(), 0.0, 0
    pbar = tqdm.tqdm(total=args.num_samples, unit="smpl", unit_scale=True)
    it = iter(loader)
    while samples_seen < args.num_samples:
        t_data = time.time()
        state_cpu, max_cpu = next(it)
        data_s += time.time() - t_data
        window_rows += len(max_cpu)
        if buffer and n_self:
            rows = self_rng.choices(list(buffer.values()), k=n_self)
            state_cpu[n_real:] = torch.stack([r[0] for r in rows])
            max_cpu[n_real:] = torch.tensor([r[1] for r in rows], dtype=max_cpu.dtype)
            self_rows += n_self
        state = state_cpu.to(device, non_blocking=True).long().repeat_interleave(args.noise_repeats, 0)
        max_thput = max_cpu.to(device, torch.float32).repeat_interleave(args.noise_repeats, 0)

        model.train()
        loss, stats = denoise_step(fwd, state, max_thput, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm or float("inf"))
        optimizer.step()
        scheduler.step()
        decay = min(args.ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        step += 1
        # Only the first n_real factories count, so the schedule and x-axis
        # match a run without self-training.
        samples_seen += n_real * args.noise_repeats
        window_samples += n_real * args.noise_repeats
        pbar.update(n_real * args.noise_repeats)
        stats["loss"], stats["grad_norm"] = loss.detach(), grad_norm.detach()
        for k, v in stats.items():
            acc[k] = acc.get(k, 0) + v.to(device)
        n_acc += 1

        if samples_seen < next_eval and samples_seen < args.num_samples:
            continue
        next_eval += eval_every
        train_seconds = time.time() - t_window
        metrics = {f"train/{k}": float(v) / n_acc for k, v in acc.items()}
        acc, n_acc = {}, 0
        t_eval = time.time()
        metrics |= val_denoise_metrics(ema, val_states, args, device)
        t_gen = time.time()
        metrics |= generation_metrics(ema, factories, args, device)
        metrics |= {
            "train/samples_seen": samples_seen,
            "train/global_step": step,
            "train/lr": optimizer.param_groups[0]["lr"],
            "perf/train_seconds": train_seconds,
            "perf/train_data_seconds": data_s,
            "perf/samples_per_second": window_samples / max(train_seconds, 1e-9),
            "perf/val_seconds": t_gen - t_eval,
            "perf/gen_seconds": time.time() - t_gen,
        }
        if args.self_frac > 0:
            metrics["self/batch_frac"] = self_rows / max(window_rows, 1)
            self_rows = window_rows = 0
        if args.self_frac > 0 and self_kinds and samples_seen < args.num_samples:
            t_self = time.time()
            metrics |= self_round(ema, self_kinds, seeds, buffer, args, device, self_gen)
            metrics["perf/self_seconds"] = time.time() - t_self
        pbar.write(
            f"{samples_seen} samples | loss {metrics['train/loss']:.4f} | val type_acc {metrics['val/type_acc']:.3f} "
            f"identity {metrics['val/identity']:.3f} | {key} {metrics[key]:.3f} "
            f"trial {metrics.get(f'val/trial_thput@{args.eval_best_of}', 0.0):.3f} "
            f"converged {metrics[f'val/converged_frac@{args.eval_best_of}']:.2f} | "
            f"train {train_seconds:.0f}s (data {data_s:.0f}s) gen {metrics['perf/gen_seconds']:.0f}s "
            f"self {metrics.get('perf/self_seconds', 0.0):.0f}s buffer {len(buffer)}"
        )
        if run is not None:
            run.log(metrics, step=samples_seen)
        if metrics[key] > best_thput:
            best_thput = metrics[key]
            torch.save({"arch": ema.arch, "model": ema.state_dict()}, args.checkpoint_path)
            pbar.write(f"  -> Saved best checkpoint ({key} {best_thput:.3f})")
            if run is not None:
                uploaded = _upload_checkpoint(run, args, best_thput, [args.checkpoint_path], uploaded)
        t_window, data_s, window_samples = time.time(), 0.0, 0
    pbar.close()

    summary = {
        "best_val_throughput": round(best_thput, 4),
        "num_samples": args.num_samples,
        "samples_seen": samples_seen,
        "optimizer_steps": step,
        "size": args.size,
        "seed": args.seed,
        "runtime_seconds": round(time.time() - t0, 1),
        "checkpoint_path": args.checkpoint_path,
        "wandb_url": run.url if run is not None else None,
    }
    summary_path = args.summary_path or str(Path(__file__).parent / "sft_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")
    print(f"\nBest {key}: {best_thput:.3f}")
    print(f"Checkpoint saved to: {args.checkpoint_path}")
    if run is not None:
        run.summary[f"best_val_throughput@{args.eval_best_of}"] = best_thput
        _upload_checkpoint(run, args, best_thput, [args.checkpoint_path, summary_path], uploaded)
        run.finish()
    return ema


if __name__ == "__main__":
    args = tyro.cli(SftArgs)
    train_sft(args)
