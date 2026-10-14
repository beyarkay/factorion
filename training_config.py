"""Single source of truth for training hyperparameters.

`PpoArgs` and `SftArgs` share their identical defaults through the
`SharedArgs` base so a value like the grid size, the CNN encoder shape, or the
W&B project lives in exactly one place. `ppo.py` and `sft.py` import the
dataclasses from here (`from training_config import PpoArgs` / `SftArgs`). CI
jobs (`ci/jobs.py`) pass only their whitelisted overrides on the command line,
so everything else falls through to these defaults — never copy a value out
of here into CI config.

Keep this module a leaf: it must not import torch / ppo / sft / factorion, so
importing it (from Python or a shell one-liner) stays cheap.

Fields that share a *name* but not a *value* across PPO (`PpoArgs`) and SFT (they are tuned
independently — e.g. `learning_rate`/`lr`, `dropout`, `weight_decay`,
`max_grad_norm`) live on each dataclass, not on `SharedArgs`.
"""

import math
import typing
from dataclasses import dataclass
from typing import Optional

# Number of CNN encoder width slots (layer1..layer{NUM_LAYER_SLOTS}); every slot
# with positive width becomes one conv layer (see ppo.layers_from_args).
NUM_LAYER_SLOTS = 8


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
    `sft.build_lr_schedule` (per optimizer step) and `ppo._iteration_lrs`
    (per PPO iteration, one envelope each for actor and critic).
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
    """Defaults common to PPO (`PpoArgs`) and SFT (`SftArgs`) with identical values.

    The CNN encoder shape (`layer1..8`, `kernel_size`, `attn_*`) is PPO's
    `AgentCNN`; SFT's denoiser takes its own architecture fields.
    """

    seed: int = 1
    """seed of the experiment"""
    size: int = 15
    """the width and height of the factory grid"""

    # CNN encoder width per layer slot. The encoder uses every slot with
    # positive width, in order; a slot of 0 drops that layer. Exposing depth +
    # per-layer width as independent numeric slots (rather than one categorical
    # "64,64,64" string) lets a W&B Bayesian sweep optimise the architecture
    # ordinally. RF = 1 + n_layers * (kernel_size - 1).
    layer1: int = 96
    layer2: int = 0
    layer3: int = 0
    layer4: int = 0
    layer5: int = 0
    layer6: int = 0
    layer7: int = 0
    layer8: int = 0
    kernel_size: int = 3
    """CNN conv kernel size (odd); padding pinned to kernel_size // 2 ("same")"""
    attn_dim: int = 288
    """model dim of the self-attention stage over the encoded map."""
    attn_heads: int = 8
    """self-attention head count (snapped down to a divisor of attn_dim)."""
    attn_layers: int = 8
    """number of stacked transformer-encoder blocks in the attention stage."""
    attn_pos_embed: int = 1
    """1 = add a learned per-cell positional embedding to the attention tokens
    (self-attention is otherwise permutation-invariant). Int not bool for W&B
    sweeps."""

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
    eval_best_of: int = 8
    """sampled rollouts per held-out factory in the rollout eval (SFT val/,
    PPO eval/); each factory scores the best of them. Every metric that rollout
    produces is logged as `<key>@N`; an unsuffixed metric is best-of-1."""


@dataclass
class PpoArgs(SharedArgs):
    exp_name: str = "ppo"
    """the name of this experiment"""
    torch_deterministic: bool = False
    """deterministic CUDA kernels (cudnn + `torch.use_deterministic_algorithms`).
    Off because the deterministic FlashAttention backward is slower; the
    ppo-speed benchmark turns it on for its bit-identical signature gate."""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    metal: bool = True
    """if toggled, Apple MPS will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    start_from: Optional[str] = None
    """SFT checkpoint to start training from instead of from scratch: either a
    local .pt path OR a W&B run id."""
    start_from_wandb: Optional[str] = None
    """wandb run id to continue from"""

    # Algorithm specific arguments
    env_id: str = "factorion/FactorioEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 6.379e-05
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments. More envs -> less likely to fit on GPU"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    lr_warmup_iters: int = 22
    """iterations of linear actor-LR warmup up to `learning_rate` at the start
    of the post-critic-warmup window — absolute, not a fraction, so runs of
    different lengths warm up identically (0 = start at peak)"""
    lr_cooldown_frac: float = 0.4097
    """fraction of the post-critic-warmup window for the actor LR's final
    (1 - sqrt) cooldown — the only phase that depends on run length (see
    `wsd_multiplier`). Schedule defaults from sweep 663awt86 (run b3hbcpvf)."""
    lr_min_ratio: float = 0.07785
    """actor LR at the final iteration, as a fraction of `learning_rate`"""
    critic_lr_cooldown_frac: float = 0.3607
    """as `lr_cooldown_frac`, for the critic's independent envelope"""
    critic_lr_min_ratio: float = 0.1232
    """critic LR at the final iteration, as a fraction of `critic_lr`"""
    gamma: float = 0.997
    """the discount factor gamma"""
    gae_lambda: float = 0.9187
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches. more minibatches -> smaller minibatch size -> more likely to fit on GPU"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1987
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""

    ent_coef_start: float = 0.008034
    """entropy coefficient at the start of training (high = more exploration)"""
    ent_coef_end: float = 0.0007372
    """entropy coefficient at the end of training (low = more exploitation)"""
    vf_coef: float = 0.794
    """coefficient of the value function"""
    entity_cost_scale: float = 0.001
    """strength of the multiplicative terminal entity-cost penalty. Each
    entity costs its per-output recursively-expanded raw-item count plus
    cumulative craft time."""
    max_grad_norm: float = 1.221
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = 0.02
    """the target KL divergence threshold, checked per minibatch."""
    divergence_penalty: float = 0.008
    """Weight β on the KL(π_θ ‖ π_ref) loss penalty anchoring the policy to the
    frozen --start-from SFT reference. Covers the five placement heads;
    the EOT head's KL is logged but never penalized (it sets the episode
    horizon). Inert without --start-from (there is no reference to anchor to);
    at 0 the penalty is off but the policy/kl_to_ref* drift metrics still log
    whenever --start-from is set."""
    adam_epsilon: float = 6.891e-06
    """The epsilon parameter for Adam"""
    weight_decay: float = 0.0
    """L2 weight decay for the Adam optimiser."""
    tile_head_std: float = 0.03492
    """Initialization std for the tile selection conv head."""
    dropout: float = 0.0
    """Dropout probability in the CNN encoder."""
    summary_path: Optional[str] = None
    """path to write summary JSON (default: summary.json next to ppo.py)"""
    critic_warmup: int = 9
    """Freeze the actor (encoder + all policy heads) for this many PPO iterations and train only the critic head, then unfreeze."""
    critic_lr: float = 4.798e-06
    """peak learning rate of the critic (value head); the actor's peak is
    `learning_rate`."""
    critic_head_std: float = 0.1169
    """Initialization std for the value head."""
    eval_every: int = 50
    """Run the held-out eval (eval/thput and per-lesson breakdowns) every N PPO iterations (and on the final iteration).
    An eval costs ~11 training iterations (A4000, bf16), so 50 keeps it under
    ~20% of wall time while a 1M-timestep run still gets 5 points."""
    eval_seeds_per_kind: int = 4
    """Held-out factories per LessonKind in the eval set. An eval's cost is
    linear in it; 4 still bounds eval/thput@N's standard error at ~0.05."""
    eval_num_envs: int = 8
    """Held-out factories rolled out in parallel, each over eval_best_of envs.
    Every step forwards all num_envs*eval_best_of slots, finished or not, so 64
    was slower: the wide batch pays for idle slots until its last rollout ends."""
    amp: bool = True
    """run the encoder forward under bf16 autocast on CUDA; the heads, log-probs
    and losses stay fp32. bf16 is what unlocks the flash-attention kernel."""
    async_envs: bool = False
    """Run the training envs in worker processes (gym AsyncVectorEnv) instead of
    serially (SyncVectorEnv)."""

    # ── Time-to-quality benchmarking (offline; see tests/benchmarks/bench_run.sh ppo-quality) ──────────
    target_metric: Optional[str] = None
    """If set, stop training the first time an EMA of this iter-metric key
    (e.g. 'rollout/reward', 'rollout/thput', 'eval/thput') reaches
    --target-value, and record the wall-clock time-to-quality. None disables
    (normal fixed-iteration training)."""
    target_value: Optional[float] = None
    """Quality threshold for --target-metric (EMA-smoothed)."""
    quality_ema_alpha: float = 0.4
    """EMA weight on the newest sample when smoothing --target-metric (higher =
    less smoothing). Smoothing tames the per-iteration metric noise so the
    time-to-quality crossing is repeatable."""
    max_seconds: Optional[float] = None
    """Safety cap for --target-metric runs: stop (quality not reached) after this
    many wall-clock seconds so a stuck/regressing run can't hang the benchmark."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (num_envs * num_steps)"""
    minibatch_size: int = 0
    """the mini-batch size (batch_size // num_minibatches)"""
    num_iterations: int = 0
    """the number of iterations (total_timesteps // batch_size)"""


@dataclass
class SftArgs(SharedArgs):
    start_from: Optional[str] = None
    """Checkpoint to resume training from instead of a fresh model: either a
    local .pt path OR a W&B run id (e.g. a prior SFT run like 'qv4uei74')."""

    num_samples: int = 20_000_000
    """corrupted training examples to train on. Factories are generated on the
    fly by DataLoader workers, each one used for `noise_repeats` examples."""
    max_level: int = 0
    """max curriculum level (0 = auto: size*size)"""
    batch_size: int = 256
    """corrupted examples per optimizer step"""
    noise_repeats: int = 4
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
    """rotary self-attention blocks over the bottleneck cells (head dim must
    be a multiple of 4)"""
    io: str = "token"
    """tile encoding in and out: 'token' (entity × facing × underground type,
    recipe, footprint part) or 'factored' (AgentCNN's encoding and its
    per-channel entity / direction / item / misc heads)"""
    lr: float = 3e-4
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
    lw_thput: float = 0.0
    """weight of the REINFORCE loss on simulated throughput × cost efficiency
    of factories sampled from the prediction (0 disables). Off: at 5M samples
    it left val/thput@64 unchanged (0.612 vs 0.609, runs gyh6chx8/r7inipmm)
    while its CPU-side legalize + simulate halved training speed."""
    rl_inputs: int = 32
    """examples per batch the throughput loss samples from"""
    rl_samples: int = 4
    """factories sampled per example (leave-one-out baseline, so >= 2)"""
    rl_max_noise: float = 0.5
    """only examples at most this corrupted feed the throughput loss: from a
    noisier input a one-step sample is rarely a working factory"""
    gen_steps: int = 8
    """ancestral steps of a chain (noise level 1 → 0)"""
    gen_refine: int = 8
    """max greedy refinement steps after them; a chain stops early once its
    prediction reproduces its input"""
    gen_temperature: float = 1.0
    """sampling temperature of the ancestral steps"""
    self_frac: float = 0.125
    """self-generated targets from the replay buffer added to each batch, as a
    fraction of its real factories (expert iteration; 0 disables it). Added,
    not substituted, so lessons train on exactly the data they would without
    it; below 0.5 so real factories stay the majority and the model can't
    collapse onto its own mode."""
    self_kinds: tuple[str, ...] = ("TRIAL_RECIPE_TREE_DEPTH_2", "TRIAL_RECIPE_TREE_DEPTH_3")
    """kinds self-training builds. Not lessons, which train on their
    references, nor depth-1 trials, which the model already solves perfectly
    and whose targets left their eval unchanged."""
    self_factories: int = 4096
    """training-seed factories (blanked to their markers) the EMA model builds
    at every eval, to refill the buffer"""
    self_best_of: int = 16
    """chains per self-training factory: more than the eval's, so the expert
    finds builds the model alone rarely makes"""
    self_top_frac: float = 0.1
    """fraction of each kind's builds kept as targets, ranked by PPO's score
    (only builds that flow and leave no inserter dangling qualify)"""
    self_buffer_size: int = 8192
    """most self-generated targets kept, oldest evicted first"""
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
