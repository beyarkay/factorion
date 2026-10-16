"""RL fine-tuning of an SFT denoiser (`denoise.py`) by online rejection-sampling
fine-tuning (RAFT; Dong et al., "RAFT: Reward rAnked FineTuning", 2023).

Every iteration builds `groups` fresh factories, blanks each to its markers,
and generates `group_size` builds of each the way the eval does, only hotter
(`temperature`). Each build earns the reward (raw throughput × cost efficiency
÷ reference), and the model takes a step of SFT's denoising loss toward each
factory's best build. SFT's loss on the iteration's lesson references rides
along (`sft_coef`): without it entity cost climbs and greedy refinement
decays. A policy gradient on the same builds learned far slower: one score
spread over a thousand sampled tiles is mostly noise (#484).

Usage:
    python rl.py --start-from <sft run id or .pt> --total-timesteps 3000000
"""

import os
import random
import time
from typing import Iterator

import numpy as np
import torch
import torch.optim as optim
import tqdm
import tyro

from denoise import CH_PART, generate, legalize, load_denoiser, score_world, world_to_state
from factorion import LESSON_IS_TRIAL, TRAINING_KINDS, Factory, LessonKind, blank_entities, build_factory
from sft import (
    assert_device_ok,
    build_eval_factories,
    cuda_env_info,
    denoise_step,
    generation_metrics,
    resolve_checkpoint,
    upload_replacing,
)
from training_config import RlArgs, SftArgs, wsd_multiplier

# Far above the eval seeds (9M-11M); build_factory seeds numpy, which caps
# seeds at 2**32 - 1, so a run gets 10M seeds and run seeds up to 229.
_TRAIN_SEED_BASE = 2_000_000_000
# Builds per batched forward: bounds memory only.
_CHUNK = 256


def _signature(args: RlArgs) -> str:
    return (
        f"raft-s{args.size}-lr{args.lr:g}-g{args.groups}x{args.group_size}"
        f"-t{args.temperature:g}-sft{args.sft_coef:g}"
        f"-from{os.path.basename(str(args.start_from)).removesuffix('.pt')}-seed{args.seed}"
    )


def _factories(args: RlArgs) -> Iterator[tuple[LessonKind, Factory]]:
    """Fresh training factories forever, kinds drawn uniformly."""
    rng = random.Random(args.seed)
    seed = _TRAIN_SEED_BASE + args.seed * 10_000_000
    while True:
        kind = rng.choice(TRAINING_KINDS)
        seed += 1
        factory = build_factory(size=args.size, kind=kind, seed=seed)
        if factory is not None:
            yield kind, factory


def rollout(model, batch, args: RlArgs, gen: SftArgs, generator) -> dict:
    """`group_size` builds of each (kind, factory) in `batch`, their
    scores, and the training targets: each factory's best build (none when all
    its builds scored alike) and the lesson references."""
    device = next(model.parameters()).device
    G = args.group_size
    cond = torch.as_tensor(np.stack([world_to_state(blank_entities(f)[0].numpy()) for _, f in batch]))
    cond = cond.to(device).repeat_interleave(G, 0)
    model.eval()
    outs = [generate(model, c, gen.gen_steps, gen.gen_refine, args.temperature, generator) for c in cond.split(_CHUNK)]
    typ, rec, part = (torch.cat([o[0][j] for o in outs]).cpu().numpy() for j in range(3))
    conf = torch.cat([o[1] for o in outs]).float().cpu().numpy()
    cond_np = cond[:, :CH_PART].cpu().numpy().astype(np.int64)
    worlds = [
        legalize(cond_np[i], typ[i], rec[i], part[i], conf[i]) for i in range(len(cond))
    ]
    scores = [score_world(w, batch[i // G][1].max_throughput, args.entity_cost_scale) for i, w in enumerate(worlds)]
    reward = np.array([s["score"] for s in scores]).reshape(-1, G)
    best = [world_to_state(worlds[f * G + int(r.argmax())]) for f, r in enumerate(reward) if r.max() > r.min()]
    refs = [world_to_state(f.world_CWH) for kind, f in batch if not LESSON_IS_TRIAL[kind]]
    targets = {k: torch.as_tensor(np.stack(xs)).to(device) if xs else None for k, xs in (("best", best), ("refs", refs))}
    return {"scores": scores, **targets}


def update(model, optimizer, roll: dict, args: RlArgs, gen: SftArgs) -> dict:
    """One optimizer step of SFT's denoising loss toward the best builds, plus
    `sft_coef` times it toward the lesson references."""
    stats: dict[str, float] = {}
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for key, coef, targets in (("losses/raft", 1.0, roll["best"]), ("losses/sft", args.sft_coef, roll["refs"])):
        if targets is None:
            continue
        for t in targets.long().split(_CHUNK):
            loss, _ = denoise_step(model, t, gen)
            w = len(t) / len(targets)
            (coef * w * loss).backward()
            stats[key] = stats.get(key, 0.0) + w * float(loss.detach())
    stats["optim/grad_norm"] = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm))
    optimizer.step()
    return stats


def rollout_metrics(batch, roll: dict, G: int) -> dict:
    """rollout/ stats, lessons and trials pooled apart as the eval pools them."""
    out: dict = {}
    pools: dict[str, dict[str, list[float]]] = {}
    for f, (kind, _) in enumerate(batch):
        group = roll["scores"][f * G : (f + 1) * G]
        thput = [s["thput"] for s in group]
        for pool in ("trial_" if LESSON_IS_TRIAL[kind] else "", f"{kind.name}/"):
            p = pools.setdefault(pool, {})
            p.setdefault("thput", []).extend(thput)
            p.setdefault(f"thput@{G}", []).append(max(thput))
            p.setdefault("reward", []).extend(s["score"] for s in group)
            p.setdefault("entity_cost", []).extend(s["entity_cost"] for s in group)
    for pool, p in pools.items():
        out |= {f"rollout/{pool}{k}": float(np.mean(v)) for k, v in p.items()}
    return out


def train_rl(args: RlArgs):
    if args.start_from is None:
        raise SystemExit("--start-from is required: the SFT denoiser to fine-tune")
    t_start = time.time()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    assert_device_ok(device)
    model = load_denoiser(resolve_checkpoint(args.start_from, args.wandb_project_name, args.wandb_entity)[0], device)
    model.amp = args.amp and device.type == "cuda"
    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=args.adam_epsilon)
    # Generation schedule and eval set as SFT's: eval/ at iteration 0 is the
    # base's own val/.
    gen = SftArgs(seed=args.seed, size=args.size, eval_best_of=args.eval_best_of)
    eval_factories = build_eval_factories(gen)
    G = args.group_size
    steps_per_iter = args.groups * G * gen.gen_steps
    num_iterations = max(1, args.total_timesteps // steps_per_iter)
    run_name = _signature(args)
    print(f"RAFT on {device}: {num_iterations} iterations of {args.groups} factories x {G} builds")

    run = None
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config={**vars(args), **cuda_env_info(), "algo": "raft"},
            name=run_name,
            group=args.wandb_group,
            tags=["raft", *(args.tags or [])],
            id=args.wandb_run_id,
            resume="allow" if args.wandb_run_id else None,
        )
        wandb.define_metric("*", step_metric="global_step")

    key = f"eval/thput@{args.eval_best_of}"
    ckpt_out = f"artifacts/{run_name}.pt"
    best, uploaded, pending = -1.0, None, False
    stream = _factories(args)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    global_step = 0
    for it in tqdm.trange(0, num_iterations + 1):
        metrics: dict = {"global_step": global_step}
        if it:
            lr = args.lr * wsd_multiplier(it - 1, num_iterations, args.warmup_iters, args.cooldown_frac, args.min_lr_ratio)
            optimizer.param_groups[0]["lr"] = lr
            t0 = time.time()
            batch = [next(stream) for _ in range(args.groups)]
            roll = rollout(model, batch, args, gen, generator)
            t1 = time.time()
            metrics |= update(model, optimizer, roll, args, gen)
            t2 = time.time()
            global_step += steps_per_iter
            metrics |= rollout_metrics(batch, roll, G)
            metrics |= {
                "global_step": global_step,
                "optim/lr": lr,
                "perf/rollout_seconds": t1 - t0,
                "perf/update_seconds": t2 - t1,
                "perf/sps": steps_per_iter / (t2 - t0),
            }
        if it % args.eval_every == 0 or it == num_iterations:
            t0 = time.time()
            # SFT's generation eval (`val/` there), keyed `eval/`.
            metrics |= {"eval/" + k.removeprefix("val/"): v for k, v in generation_metrics(model, eval_factories, gen, device).items()}
            metrics["perf/eval_seconds"] = time.time() - t0
            tqdm.tqdm.write(
                f"iter {it} step {global_step} | {key} {metrics[key]:.3f} "
                f"trial {metrics.get(f'eval/trial_thput@{args.eval_best_of}', 0.0):.3f}"
            )
            if metrics[key] > best:
                best = metrics[key]
                os.makedirs("artifacts", exist_ok=True)
                torch.save({"arch": model.arch, "model": model.state_dict()}, ckpt_out)
                pending = True
            if run is not None and pending and time.time() - t_start >= 60 * args.min_upload_minutes:
                uploaded = upload_replacing(
                    run, run_name, [ckpt_out], {"best_eval_throughput": best}, ["latest", f"thp{best:.3f}"], uploaded
                )
                pending = False
        if run is not None:
            run.log(metrics)
    if run is not None:
        run.summary[f"best_{key}"] = best
        run.finish()
    return model


if __name__ == "__main__":
    train_rl(tyro.cli(RlArgs))
