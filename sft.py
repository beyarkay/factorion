"""Denoising pretraining of the whole-factory UNet (`denoise.py`).

Every training example is a known-correct factory from `build_factory` with a
random fraction t ~ U[0, 1] of its free tiles resampled from noise; the model
always regresses the clean factory. The eval generates factories for held-out
lessons from pure noise (best of `eval_best_of` chains each) and scores them
in the throughput simulator.

Usage:
    python sft.py --num-samples 2000000
"""

import copy
import io
import itertools
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from denoise import (  # noqa: E402
    DenoiserUNet,
    N_TYPES,
    _luts,
    corrupt,
    free_mask,
    legalize,
    load_denoiser,
    run_denoise_eval,
    score_world,
    state_to_tokens,
    tokens_to_state,
    world_to_state,
)
from factorion import LESSON_IS_TRIAL, TRAINING_KINDS, LessonKind, build_factory  # noqa: E402
from ppo import (  # noqa: E402
    _build_eval_set,
    _resolve_start_from,
    _upload_replacing,
    assert_device_ok,
    cuda_env_info,
)
from training_config import PpoArgs, SftArgs, wsd_multiplier  # noqa: E402

# Training factories come from far above the held-out eval seeds (9M-11M).
_TRAIN_SEED_BASE = 1_000_000_000
# Consecutive build failures after which a kind counts as unbuildable at this
# grid size and leaves the sampler.
_MAX_BUILD_FAILURES_PER_KIND = 100
# Noise levels the val/ denoising metrics are measured at.
_VAL_NOISE_LEVELS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _humanize_count(n: int) -> str:
    """50_000 -> '50k', 2_500_000 -> '2.5M'."""
    for div, suffix in ((1_000_000, "M"), (1_000, "k")):
        if n >= div:
            return f"{n / div:.1f}".rstrip("0").rstrip(".") + suffix
    return str(n)


def _artifact_name(args: SftArgs) -> str:
    """W&B run/artifact name: identical configs collapse into versions of one
    artifact, config changes get their own."""
    return (
        f"dn-s{args.size}-n{_humanize_count(args.num_samples)}-bs{args.batch_size}"
        f"-lr{args.lr:.0e}-w{args.width}x{args.levels}b{args.blocks}a{args.attn_layers}"
    )


def _iter_factories(size, base_seed, worker_id, num_workers):
    """Yield (state uint8, max_throughput) for fresh lesson factories forever,
    kinds uniform. Worker w walks seeds ≡ base_seed + w (mod num_workers)."""
    kinds = [k for k in TRAINING_KINDS if not LESSON_IS_TRIAL[k]]
    fails = dict.fromkeys(kinds, 0)
    rng = random.Random(base_seed + worker_id)
    seed = base_seed + worker_id
    while True:
        kind = rng.choice(kinds)
        seed += num_workers
        factory = build_factory(size=size, kind=kind, seed=seed)
        if factory is None:
            fails[kind] += 1
            if fails[kind] >= _MAX_BUILD_FAILURES_PER_KIND:
                kinds.remove(kind)
                if not kinds:
                    raise RuntimeError(f"No lesson kind builds at size={size}")
            continue
        fails[kind] = 0
        yield torch.from_numpy(world_to_state(factory.world_CWH)), float(factory.max_throughput)


class FactoryStream(IterableDataset):
    """Clean training factories, generated on the fly by DataLoader workers."""

    def __init__(self, size, base_seed):
        self.size = size
        self.base_seed = base_seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid, n = (0, 1) if info is None else (info.id, info.num_workers)
        return _iter_factories(self.size, self.base_seed, wid, n)


def build_lr_schedule(optimizer, total_steps: int, args: SftArgs):
    """Warmup-Stable-Decay (`wsd_multiplier`), stepped once per optimizer step."""
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: wsd_multiplier(step, total_steps, args.warmup_steps, args.cooldown_frac, args.min_lr_ratio),
    )


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
    cond = state[rep, :5].cpu().numpy().astype(np.int64)
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


def build_eval_factories(args: SftArgs) -> list:
    """The held-out (seed, kind, Factory) set: PPO's eval seeds (trials
    included), so a generation eval is directly comparable to PPO's."""
    seeds_to_kind = _build_eval_set(SimpleNamespace(size=args.size, seed=args.seed, eval_seeds_per_kind=args.eval_seeds_per_kind))
    return [(s, LessonKind(k), build_factory(size=args.size, kind=LessonKind(k), seed=s)) for s, k in seeds_to_kind.items()]


def _upload_checkpoint(run, args, best_thput, files, previous):
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
    if args.lw_thput > 0 and args.rl_samples < 2:
        raise ValueError("the throughput loss needs rl_samples >= 2 for its leave-one-out baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99))
    total_steps = max(1, math.ceil(args.num_samples / args.batch_size))
    scheduler = build_lr_schedule(optimizer, total_steps, args)

    workers = min(16, os.cpu_count() or 1)
    loader = DataLoader(
        FactoryStream(args.size, train_seed),
        batch_size=args.batch_size // args.noise_repeats,
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
        state = state_cpu.to(device, non_blocking=True).long().repeat_interleave(args.noise_repeats, 0)
        max_thput = max_cpu.to(device, torch.float32).repeat_interleave(args.noise_repeats, 0)

        model.train()
        loss, stats = denoise_step(fwd, state, max_thput, args)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        decay = min(args.ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)
        step += 1
        samples_seen += state.shape[0]
        window_samples += state.shape[0]
        pbar.update(state.shape[0])
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
        pbar.write(
            f"{samples_seen} samples | loss {metrics['train/loss']:.4f} | val type_acc {metrics['val/type_acc']:.3f} "
            f"identity {metrics['val/identity']:.3f} | {key} {metrics[key]:.3f} "
            f"trial {metrics.get(f'val/trial_thput@{args.eval_best_of}', 0.0):.3f} "
            f"converged {metrics[f'val/converged_frac@{args.eval_best_of}']:.2f} | "
            f"train {train_seconds:.0f}s (data {data_s:.0f}s) gen {metrics['perf/gen_seconds']:.0f}s"
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
    print(f"\nBest {key}: {best_thput:.3f}\nCheckpoint saved to: {args.checkpoint_path}")
    if run is not None:
        run.summary[f"best_val_throughput@{args.eval_best_of}"] = best_thput
        _upload_checkpoint(run, args, best_thput, [args.checkpoint_path, summary_path], uploaded)
        run.finish()
    return ema


if __name__ == "__main__":
    train_sft(tyro.cli(SftArgs))
