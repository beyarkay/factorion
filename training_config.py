"""Single source of truth for training hyperparameters.

`RlArgs` and `SftArgs` share their identical defaults through the
`SharedArgs` base so a value like the grid size or the W&B project lives in
exactly one place. `rl.py` and `sft.py` import the dataclasses from here. CI
jobs (`ci/jobs.py`) pass only their whitelisted overrides on the command line,
so everything else falls through to these defaults — never copy a value out
of here into CI config.

Keep this module a leaf: it must not import torch / rl / sft / factorion, so
importing it (from Python or a shell one-liner) stays cheap.
"""

import math
import typing
from dataclasses import dataclass
from typing import Optional


def wsd_multiplier(
    step: int,
    total_steps: int,
    warmup_steps: int,
    cooldown_frac: float,
    min_ratio: float,
) -> float:
    """Warmup-Stable-Decay LR multiplier at 0-indexed `step` of `total_steps`.

    Linear warmup over `warmup_steps` up to 1.0, a stable phase holding 1.0,
    then a (1 - sqrt) cooldown over the final `cooldown_frac` of the run down
    to `min_ratio`. Only the cooldown depends on `total_steps`, so runs of
    different budgets share their curve until their cooldowns diverge.
    Cooldown shape from Hägele et al., "Scaling Laws and Compute-Optimal
    Training Beyond Fixed Training Durations" (NeurIPS 2024). Shared by
    `sft.build_lr_schedule` (per optimizer step) and `rl.train_rl` (per
    iteration).
    """
    warmup = min(warmup_steps, total_steps - 1)
    cooldown = max(
        1, min(int(round(total_steps * cooldown_frac)), total_steps - warmup)
    )
    stable_end = total_steps - cooldown
    if step < warmup:
        return (step + 1) / warmup
    if step < stable_end:
        return 1.0
    t = min(1.0, (step - stable_end) / cooldown)
    return min_ratio + (1.0 - min_ratio) * (1.0 - math.sqrt(t))


@dataclass
class SharedArgs:
    """Defaults common to RL (`RlArgs`) and SFT (`SftArgs`)."""

    seed: int = 1
    """seed of the experiment"""
    size: int = 32
    """the width and height of the factory grid"""

    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "factorion"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: Optional[str] = None
    """W&B run group name (groups parallel seeds together in the dashboard)"""
    wandb_run_id: Optional[str] = None
    """Pre-assigned W&B run id. CI sets this at launch so the run URL is known
    (and linkable from the PR) before the pod even boots. None = W&B mints one."""
    tags: typing.Optional[typing.List[str]] = None
    """Tags to apply to the wandb run."""
    min_upload_minutes: float = 30.0
    """No checkpoint goes to W&B before the run is this old (the pending best
    uploads at the first eval after): short smoke/sweep runs are rarely
    resumed from, and their checkpoints were half the storage quota."""
    box_frac: float = 0.75
    """fraction of lessons (SFT and RL, training and held-out alike) built
    inside a random box, every tile outside it unbuildable
    (`sft.boxed_factory`): the buildable area in the game is whatever the
    player marks out"""
    eval_best_of: int = 8
    """chains generated per held-out factory in the generation eval (SFT val/,
    RL eval/); each factory scores the best of them. Every metric that eval
    produces is logged as `<key>@N`; an unsuffixed metric is best-of-1."""


@dataclass
class RlArgs(SharedArgs):
    """Online rejection-sampling fine-tuning of an SFT denoiser (`rl.py`).
    A timestep is one ancestral denoising step of one build."""

    start_from: Optional[str] = None
    """denoiser checkpoint to fine-tune (required): a local .pt path or a W&B
    run id"""
    total_timesteps: int = 500000
    """total timesteps of the experiment"""
    lr: float = 2.5e-4
    """peak learning rate"""
    warmup_iters: int = 22
    """iterations of linear LR warmup up to `lr`"""
    cooldown_frac: float = 0.4097
    """fraction of the iterations for the final (1 - sqrt) LR cooldown"""
    min_lr_ratio: float = 0.07785
    """LR at the final iteration, as a fraction of `lr`"""
    adam_epsilon: float = 6.891e-06
    """The epsilon parameter for Adam"""
    max_grad_norm: float = 1.221
    """the maximum norm for the gradient clipping"""
    groups: int = 64
    """factories per iteration"""
    group_size: int = 32
    """builds per factory; the model trains toward the best of them"""
    temperature: float = 1.5
    """sampling temperature of the builds: hotter widens the best of a
    group without hurting the typical build (the eval keeps 1.0)"""
    sft_coef: float = 0.3
    """weight of SFT's denoising loss on each iteration's lesson references
    (the best builds' loss weighs 1)"""
    entity_cost_scale: float = 0.001
    """strength of the multiplicative entity-cost discount on the reward. Each
    entity costs its per-output recursively-expanded raw-item count plus
    cumulative craft time."""
    eval_every: int = 50
    """run the held-out generation eval every N iterations (and on the last)"""
    amp: bool = True
    """bf16 autocast for the network on CUDA; losses stay fp32"""


@dataclass
class SftArgs(SharedArgs):
    start_from: Optional[str] = None
    """Checkpoint to resume training from instead of a fresh model: either a
    local .pt path OR a W&B run id (e.g. a prior SFT run like 'qv4uei74')."""

    num_samples: int = 20_000_000
    """corrupted training examples to train on. Factories are generated on the
    fly by DataLoader workers, each one used for `noise_repeats` examples."""
    batch_size: int = 128
    """corrupted examples per optimizer step"""
    noise_repeats: int = 1
    """examples per generated factory, each at its own noise level; amortises
    factory generation, the CPU-side bottleneck"""
    noise_zero_frac: float = 0.05
    """fraction of examples left clean (noise level 0): the model must learn to
    leave a finished factory alone, which is what stops generation"""
    noise_one_frac: float = 0.1
    """fraction of examples that are pure noise (level 1), where generation
    starts; the rest are uniform in [0, 1]"""
    unroll: bool = True
    """also train on the model's own sampled prediction (SUNDAE's unrolled
    denoising), the input every generation step after the first sees"""
    width: int = 128
    """denoiser channels at full resolution; each lower level doubles them"""
    levels: int = 2
    """UNet resolutions (levels - 1 stride-2 downsamples)"""
    blocks: int = 2
    """residual blocks per level, on each side of the UNet"""
    attn_layers: int = 4
    """rotary self-attention blocks over the bottleneck cells"""
    attn_heads: int = 8
    """heads of each attention block (head dim must be a multiple of 4)"""
    lr: float = 1.5e-3
    """peak learning rate, held flat for the whole stable phase"""
    warmup_steps: int = 500
    """optimizer steps of linear warmup up to lr — absolute, not a fraction, so
    runs of different lengths warm up identically"""
    cooldown_frac: float = 0.3
    """fraction of total steps for the final (1-sqrt) cooldown — the only phase
    that depends on run length"""
    min_lr_ratio: float = 0.05
    """LR at the final step, as a fraction of lr"""
    weight_decay: float = 0.01
    """AdamW weight decay"""
    dropout: float = 0.0
    """dropout in residual and attention blocks. Every example is fresh."""
    amp: bool = True
    """bf16 autocast for the network on CUDA; losses stay fp32"""
    compile: bool = True
    """torch.compile the training forward (CUDA only)"""
    max_grad_norm: float = 1.0
    """grad L2-norm clip (0 disables clipping)"""
    ema_decay: float = 0.999
    """decay of the weight EMA that is evaluated and checkpointed"""
    lw_type: float = 1.0
    """weight of the per-tile entity/facing cross entropy"""
    lw_recipe: float = 1.0
    """weight of the recipe cross entropy (assembler tiles)"""
    lw_part: float = 1.0
    """weight of the footprint-offset cross entropy (multi-tile tiles)"""
    lw_cost: float = 0.05
    """weight of the predicted factory's expected per-tile entity cost: a
    pull toward leaving unneeded tiles empty"""
    gen_steps: int = 8
    """ancestral steps of a chain (noise level 1 → 0)"""
    gen_refine: int = 8
    """max greedy refinement steps after them; a chain stops early once its
    prediction reproduces its input"""
    gen_temperature: float = 1.0
    """sampling temperature of the ancestral steps"""
    eval_seeds_per_kind: int = 64
    """held-out factories per training lesson kind (and per trial kind)"""
    eval_batch_chains: int = 2048
    """chains denoised per batched forward"""
    eval_every_n_samples: Optional[int] = None
    """evaluate (and select checkpoints) every N training examples. None =
    250k for runs of at most 2M samples, else 1M."""
    checkpoint_path: str = "sft_checkpoint.pt"
    """path to save the trained model"""
    summary_path: Optional[str] = None
    """path to write summary JSON (default: sft_summary.json next to sft.py)"""
