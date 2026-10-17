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
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
from torch.utils.data import DataLoader, IterableDataset

from denoise import (
    DenoiserUNet,
    N_TYPES,
    _luts,
    corrupt,
    free_mask,
    load_denoiser,
    run_denoise_eval,
    state_to_tokens,
    tokens_to_state,
    world_to_state,
)
from factorion import (
    LESSON_IS_TRIAL,
    Channel,
    Factory,
    Footprint,
    TRAINING_KINDS,
    LessonKind,
    build_factory,
    count_unpowered,
    refresh_power,
)

from training_config import RlArgs, SftArgs, wsd_multiplier

# Training factories come from far above the held-out eval seeds (9M-11M).
_TRAIN_SEED_BASE = 1_000_000_000
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


_MIN_BOX = 5


def boxed_factory(kind: LessonKind, seed: int, size: int) -> Optional[Factory]:
    """A lesson inside a random w × h box (5 ≤ w, h ≤ size) of a size × size
    grid, every tile outside the box unbuildable. Generators only build
    squares, so the factory is min(w, h) square, placed at random within the
    box. None when the kind doesn't fit or the move unpowers it."""
    rng = random.Random(seed)
    w, h = rng.randint(_MIN_BOX, size), rng.randint(_MIN_BOX, size)
    s = min(w, h)
    factory = build_factory(size=s, kind=kind, seed=seed)
    if factory is None:
        return None
    bx, by = rng.randint(0, size - w), rng.randint(0, size - h)
    fx, fy = bx + rng.randint(0, w - s), by + rng.randint(0, h - s)
    world = torch.zeros((factory.world_CWH.shape[0], size, size), dtype=factory.world_CWH.dtype)
    world[Channel.FOOTPRINT.value, bx : bx + w, by : by + h] = Footprint.AVAILABLE.value
    world[:, fx : fx + s, fy : fy + s] = factory.world_CWH
    refresh_power(world)
    # A pole that drew power from the square's edge may now face buildable
    # box tiles instead.
    if count_unpowered(world):
        return None
    protected = frozenset((x + fx, y + fy) for x, y in factory.protected_positions)
    return replace(factory, world_CWH=world, protected_positions=protected)


def _iter_factories(size, base_seed, worker_id, num_workers, box_frac=0.0):
    """Yield the uint8 state of fresh lesson factories forever, `box_frac` of
    them built inside a random box (`boxed_factory`).

    Worker `w` of `num_workers` walks seeds ≡ base_seed+w (mod num_workers), so
    concurrent workers never share a factory. Kinds are drawn uniformly.
    """
    # Trial kinds have no known solution, so there is no clean factory to
    # denoise towards. Trials are RL-only.
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
        if rng.random() < box_frac:
            # A box too small for the kind says nothing about the full grid, so
            # it doesn't count towards dropping the kind.
            boxed = boxed_factory(kind, seed, size)
            if boxed is not None:
                yield torch.from_numpy(world_to_state(boxed.world_CWH))
            continue
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
        yield torch.from_numpy(world_to_state(factory.world_CWH))


class FactoryStream(IterableDataset):
    """Generates training factories on the fly, sharded across DataLoader workers.

    CPU generation overlaps GPU training via DataLoader prefetch.
    """

    def __init__(self, size, base_seed, box_frac):
        self.size = size
        self.base_seed = base_seed
        self.box_frac = box_frac

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id = 0 if info is None else info.id
        num_workers = 1 if info is None else info.num_workers
        return _iter_factories(self.size, self.base_seed, worker_id, num_workers, self.box_frac)


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


def assert_device_ok(device) -> None:
    if device.type in ("cuda", "mps"):
        return
    # CI and the local test suite run CPU-only smoke tests; everywhere else a
    # CPU fallback is a bug. pytest sets PYTEST_CURRENT_TEST for every test, so
    # `uv run pytest` trains on CPU without needing CI=true in the environment.
    if os.environ.get("CI") or os.environ.get("PYTEST_CURRENT_TEST"):
        return
    raise RuntimeError(
        f"Refusing to train on '{device.type}': no CUDA GPU was found. This "
        "usually means the host NVIDIA driver is too old for the installed torch "
        f"CUDA build (this wheel ships CUDA {torch.version.cuda}). Fix the "
        "host/driver, or pick a GPU+host that supports it."
    )


def cuda_env_info() -> dict[str, str]:
    """Wheel CUDA build + host GPU, recorded in every run's W&B config: CI
    schedules across whatever driver versions RunPod has free, so a failure
    that only reproduces on some hosts is otherwise untraceable after the pod
    is gone. The host driver's CUDA version is in W&B's own run metadata
    (`cudaVersion`)."""
    info = {"env/torch_version": torch.__version__, "env/torch_cuda_build": torch.version.cuda or "none"}
    if torch.cuda.is_available():
        info["env/gpu_name"] = torch.cuda.get_device_name(0)
        info["env/gpu_capability"] = "%d.%d" % torch.cuda.get_device_capability(0)
    return info


def resolve_checkpoint(spec: str, project: str, entity: Optional[str] = None) -> tuple[str, dict]:
    """(local .pt path, provenance) of a checkpoint given as a local path or a
    W&B run id ("abc123" or "entity/project/abc123"), whose newest committed
    model artifact is downloaded to /tmp/factorion-checkpoints."""
    if os.path.exists(spec):
        return spec, {"kind": "local", "path": spec}
    if spec.endswith(".pt"):
        raise FileNotFoundError(f"checkpoint {spec!r} does not exist")
    import wandb

    # Callers that disabled W&B (the factory builder, tests) still need the download.
    prev = {k: os.environ.pop(k) for k in ("WANDB_MODE", "WANDB_DISABLED") if k in os.environ}
    try:
        api = wandb.Api()
        run = api.run(spec if spec.count("/") == 2 else f"{entity or api.default_entity}/{project}/{spec}")
        # A run replaces its checkpoint as its best improves, so skip a
        # version whose upload hasn't finished.
        arts = [a for a in run.logged_artifacts() if a.type == "model" and a.state == "COMMITTED"]
        if not arts:
            raise RuntimeError(f"run {run.id} has no committed model artifact")
        art = max(arts, key=lambda a: a.created_at)
        dest = Path("/tmp/factorion-checkpoints") / run.id / art.name.replace(":", "_")
        pt_files = sorted(Path(art.download(root=str(dest))).glob("*.pt"))
        if not pt_files:
            raise RuntimeError(f"artifact {art.name} contains no .pt file")
        print(f"Resolved {spec} -> {art.name} -> {pt_files[0]}")
        source = {
            "kind": "wandb",
            "run_id": run.id,
            "run_url": run.url,
            "run_name": run.name,
            "artifact": art.name,
            "config": dict(getattr(run, "config", {})),
        }
        return str(pt_files[0]), source
    finally:
        os.environ.update(prev)


def upload_replacing(run, name, files, metadata, aliases, previous):
    """Log `files` as a new version of model artifact `name` and delete
    `previous`, this run's last upload, so a run holds one checkpoint in W&B
    however often its best improves, and a run killed mid-training still
    leaves its best behind. Returns the new version, to pass as `previous`."""
    import wandb

    artifact = wandb.Artifact(name=name, type="model", metadata=metadata)
    for f in files:
        artifact.add_file(f)
    logged = run.log_artifact(artifact, aliases=aliases)
    logged.wait()
    if previous is not None:
        previous.delete(delete_aliases=True)
    return logged


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


def denoise_step(model, state, args: SftArgs):
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
    heads = ("type", "recipe", "part")
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
        cost_scale=RlArgs.entity_cost_scale,
        seed=args.seed,
        batch_chains=args.eval_batch_chains,
        records=records,
    )
    out = {
        f"val/converged_frac@{N}": roll["converged_frac"],
        f"val/refine_steps@{N}": roll["refine_steps"],
        f"val/inserters_no_input@{N}": roll["dangling"][0],
        f"val/inserters_no_output@{N}": roll["dangling"][1],
        f"val/unpowered@{N}": roll["unpowered"][0],
    }
    if roll["n_trials"]:
        out[f"val/trial_inserters_no_input@{N}"] = roll["dangling"][2]
        out[f"val/trial_inserters_no_output@{N}"] = roll["dangling"][3]
        out[f"val/trial_unpowered@{N}"] = roll["unpowered"][1]
    for key, v in roll.items():
        if key.startswith("thput@"):
            out[f"val/{key}"] = v
        elif key.startswith("trial_thput@") and roll["n_trials"]:
            out[f"val/{key}"] = v
        elif key.startswith("per_kind@"):
            n = key.split("@")[1]
            out |= {f"val/{kind}/thput@{n}": x for kind, x in v.items()}
    return out


def build_eval_factories(args) -> list[tuple[int, LessonKind, Factory]]:
    """The held-out (seed, kind, Factory) set: `eval_seeds_per_kind` factories
    of every training kind (trials included), each kind from its own seed
    range far below the training seeds."""
    out = []
    for ki, kind in enumerate(LessonKind):
        if kind not in TRAINING_KINDS:
            continue
        base = 9_000_000 + args.seed + ki * 100_000
        found = 0
        for s in range(base, base + 5000):
            if found == args.eval_seeds_per_kind:
                break
            factory = build_factory(size=args.size, kind=kind, seed=s)
            if factory is not None:
                out.append((s, kind, factory))
                found += 1
    return out


def _upload_checkpoint(run, args, best_thput, files, previous):
    """Upload the run's best checkpoint via `_upload_replacing`."""
    return upload_replacing(
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
        model = load_denoiser(resolve_checkpoint(args.start_from, args.wandb_project_name, args.wandb_entity)[0])
    else:
        model = DenoiserUNet(
            width=args.width,
            levels=args.levels,
            blocks=args.blocks,
            attn_layers=args.attn_layers,
            attn_heads=args.attn_heads,
            dropout=args.dropout,
        )
        prior_states = list(itertools.islice(_iter_factories(args.size, train_seed, 0, 1, args.box_frac), 256))
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
    loader = DataLoader(
        FactoryStream(args.size, train_seed, args.box_frac),
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
    best_thput, uploaded, pending = -1.0, None, False
    key = f"val/thput@{args.eval_best_of}"
    samples_seen = step = 0
    acc: dict[str, torch.Tensor] = {}
    n_acc = 0
    t_window, data_s, window_samples = time.time(), 0.0, 0
    pbar = tqdm.tqdm(total=args.num_samples, unit="smpl", unit_scale=True)
    it = iter(loader)
    while samples_seen < args.num_samples:
        t_data = time.time()
        state_cpu = next(it)
        data_s += time.time() - t_data
        state = state_cpu.to(device, non_blocking=True).long().repeat_interleave(args.noise_repeats, 0)

        model.train()
        loss, stats = denoise_step(fwd, state, args)
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
            pending = True
        if run is not None and pending and time.time() - t0 >= 60 * args.min_upload_minutes:
            uploaded = _upload_checkpoint(run, args, best_thput, [args.checkpoint_path], uploaded)
            pending = False
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
        if time.time() - t0 >= 60 * args.min_upload_minutes:
            _upload_checkpoint(run, args, best_thput, [args.checkpoint_path, summary_path], uploaded)
        run.finish()
    return ema


if __name__ == "__main__":
    args = tyro.cli(SftArgs)
    train_sft(args)
