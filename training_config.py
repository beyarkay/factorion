"""Single source of truth for training hyperparameters.

`PpoArgs` and `SftArgs` share their identical defaults through the
`SharedArgs` base so a value like the grid size, the CNN encoder shape, or the
W&B project lives in exactly one place. `ppo.py` and `sft.py` import the
dataclasses from here (`from training_config import PpoArgs` / `SftArgs`); the CI
shell scripts read the same defaults with
``python -c "from training_config import PpoArgs; print(PpoArgs().<field>)"`` instead
of hardcoding their own copies.

Keep this module a leaf: it must not import torch / ppo / sft / factorion, so
importing it (from Python or a shell one-liner) stays cheap.

Fields that share a *name* but not a *value* across PPO (`PpoArgs`) and SFT (they are tuned
independently — e.g. `learning_rate`/`lr`, `dropout`, `weight_decay`,
`max_grad_norm`, `tile_head_std`) live on each dataclass, not on `SharedArgs`.
"""

import typing
from dataclasses import dataclass
from typing import Optional

# Number of CNN encoder width slots (layer1..layer{NUM_LAYER_SLOTS}); every slot
# with positive width becomes one conv layer (see ppo.layers_from_args).
NUM_LAYER_SLOTS = 8


@dataclass
class SharedArgs:
    """Defaults common to PPO (`PpoArgs`) and SFT (`SftArgs`) with identical values.

    The CNN encoder shape (`layer1..8`, `kernel_size`) and `size` MUST match
    between the two so an SFT checkpoint loads into the PPO policy unchanged.
    """

    seed: int = 1
    """seed of the experiment"""
    size: int = 11
    """the width and height of the factory grid"""

    # CNN encoder width per layer slot. The encoder uses every slot with
    # positive width, in order; a slot of 0 drops that layer. Exposing depth +
    # per-layer width as independent numeric slots (rather than one categorical
    # "64,64,64" string) lets a W&B Bayesian sweep optimise the architecture
    # ordinally. RF = 1 + n_layers * (kernel_size - 1).
    layer1: int = 93
    layer2: int = 69
    layer3: int = 96
    layer4: int = 0
    layer5: int = 0
    layer6: int = 0
    layer7: int = 0
    layer8: int = 0
    kernel_size: int = 3
    """CNN conv kernel size (odd); padding pinned to kernel_size // 2 ("same")"""

    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "factorion"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
    wandb_group: Optional[str] = None
    """W&B run group name (groups parallel seeds together in the dashboard)"""
    tags: typing.Optional[typing.List[str]] = None
    """Tags to apply to the wandb run."""


@dataclass
class PpoArgs(SharedArgs):
    exp_name: str = "ppo"
    """the name of this experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    metal: bool = True
    """if toggled, Apple MPS will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    start_from: Optional[str] = None
    """SFT checkpoint to start training from instead of from scratch: either a
    local .pt path OR a W&B run id (e.g. an SFT run like 'j0s5y2mc', whose
    model artifact is downloaded automatically)."""
    start_from_wandb: Optional[str] = None
    """wandb run id to continue from"""

    # Algorithm specific arguments
    env_id: str = "factorion/FactorioEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 7e-4
    """the learning rate of the optimizer (default = the confirmed SFT->PPO
    finetune optimum; see tests/benchmarks/EXPERIMENT_LOG.md)"""
    num_envs: int = 16
    """the number of parallel game environments. More envs -> less likely to fit on GPU"""
    num_steps: int = 256
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.9857
    """the discount factor gamma"""
    gae_lambda: float = 0.8014
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches. more minibatches -> smaller minibatch size -> more likely to fit on GPU"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2746
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""

    ent_coef_start: float = 0.02041
    """entropy coefficient at the start of training (high = more exploration)"""
    ent_coef_end: float = 0.0004576
    """entropy coefficient at the end of training (low = more exploitation)"""
    vf_coef: float = 0.7426
    """coefficient of the value function"""
    throughput_reward_scale: float = 1.0
    """scales the terminal throughput reward (paid once when the episode ends, on eot or max_steps); throughput is in [0, 1] so this sets the max terminal reward."""
    step_penalty: float = 0.01
    """penalty subtracted every step, so dragging the build out costs reward and the eot head learns to fire once the factory can't improve. Small relative to throughput_reward_scale."""
    max_grad_norm: float = 1.979
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = 0.02
    """the target KL divergence threshold; early-stops the update's epochs. None
    = always run all update_epochs. (Why this default: EXPERIMENT_LOG.md.)"""
    adam_epsilon: float = 6.866e-06
    """The epsilon parameter for Adam"""
    weight_decay: float = 0.0
    """L2 weight decay for the Adam optimiser."""
    tile_head_std: float = 0.06503
    """Initialization std for the tile selection conv head (smaller = more uniform initial exploration)"""
    dropout: float = 0.0
    """Dropout probability in the CNN encoder."""
    summary_path: Optional[str] = None
    """path to write summary JSON (default: summary.json next to ppo.py)"""
    critic_warmup: int = 5
    """Freeze the actor (encoder + all policy heads) for this many PPO iterations and train only the critic head, then unfreeze. An SFT checkpoint loads a trained actor but a random critic; without a warm-up the random critic's garbage advantages wreck the SFT policy in the first updates. Set 0 to disable for from-scratch runs. LR + entropy annealing start at unfreeze. (Why the default of 5: tests/benchmarks/EXPERIMENT_LOG.md.)"""
    critic_lr_mult: float = 1.0
    """Multiplier on the critic (value-head) learning rate relative to the actor's. >1 warms the value head faster — useful to shorten --critic-warmup (the warmup is dead time for the actor). 1.0 = unchanged (critic LR == actor LR)."""
    eval_every: int = 7
    """Run the greedy held-out eval (eval/thput, eval/thput_eot, per-lesson) every N PPO iterations (and on the final iteration). Mirrors the SFT rollout eval so the curves overlay the SFT baseline. 0 disables."""
    eval_seeds_per_kind: int = 12
    """Held-out factories per LessonKind in the greedy eval set."""
    eval_num_envs: int = 8
    """Parallel envs for the greedy eval rollout."""
    amp: bool = False
    """Run the policy/value forward passes under bf16 autocast (mixed precision).
    Speeds up the GPU matmuls; helps most when the GPU is the bottleneck (less so
    here, where the rollout is CPU-bound). Changes numerics, so the trajectory
    (and time-to-quality) can shift."""
    async_envs: bool = False
    """Run the training envs in worker processes (gym AsyncVectorEnv) instead of
    serially (SyncVectorEnv). At high --num-envs the serial CPU env-stepping is
    the rollout bottleneck; AsyncVectorEnv fans it across cores. (At 16 envs the
    IPC overhead makes it slower — only worth it with many envs.)"""

    # ── Time-to-quality benchmarking (offline; see tests/benchmarks/bench_run.sh ppo-quality) ──────────
    target_metric: Optional[str] = None
    """If set, stop training the first time an EMA of this iter-metric key
    (e.g. 'rollout/reward', 'rollout/thput', 'eval/thput_eot') reaches
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
    # One epoch over fresh data; architecture matches checkpoint kkcv6xe3.
    num_samples: int = 45_000_000
    """(state, action) pairs streamed per epoch. Generated on the fly by
    DataLoader workers, so this is never held in memory all at once."""
    max_level: int = 0
    """max curriculum level (0 = auto: size*size)"""
    epochs: int = 1
    """number of training epochs"""
    batch_size: int = 512
    """training batch size"""
    lr: float = 3.242e-3
    """peak learning rate (after warmup, before cosine decay)"""
    warmup_frac: float = 0.0
    """fraction of total steps for linear warmup from lr*1e-3 up to lr. 0 disables warmup."""
    min_lr_ratio: float = 0.02869
    """cosine decay floor as a fraction of lr (final LR = lr * min_lr_ratio)"""
    weight_decay: float = 1.661e-3
    """AdamW weight decay"""
    dropout: float = 0.1827
    """spatial dropout (Dropout2d) after each encoder conv. 0.0 = off (no-op)."""
    max_grad_norm: float = 2.104
    """grad L2-norm clip (0 disables clipping)"""
    lw_tile: float = 1.162
    """loss weight for the tile-selection (BCE) head"""
    lw_ent: float = 0.6673
    """loss weight for the entity (CE) head"""
    lw_dir: float = 0.948
    """loss weight for the direction (CE) head"""
    lw_item: float = 0.6349
    """loss weight for the item / recipe (CE) head"""
    lw_misc: float = 0.6236
    """loss weight for the misc (CE) head"""
    lw_eot: float = 1.302
    """loss weight for the EOT (end-of-trajectory) BCE head"""
    eval_every_n_samples: int = 100_000
    """run validation + rollout eval + logging + checkpoint selection every N
    optimiser-seen samples rather than once per epoch (0 = evaluate only once,
    after the final batch). Samples, not epochs, so a single-epoch run over a
    huge dataset still yields a real training curve instead of one point."""
    eval_rollouts: bool = True
    """run the greedy rollout eval (the default checkpoint-selection metric) on
    each eval. Disable to skip the slow rollout (val accuracy still logged)."""
    eval_rollouts_max_seeds: int = 400
    """cap on val seeds per rollout eval — the sample size of the selection
    metric (val/thput), so it sets its noise floor. Drawn from val lessons."""
    eval_rollouts_num_envs: int = 8
    """parallel envs for rollout eval; batches the CNN forward across them"""
    rollout_eot_threshold: float = 0.5
    """EOT-head prob above which we mark the model "would stop" (for val/thput_eot)"""
    checkpoint_path: str = "sft_checkpoint.pt"
    """path to save the trained model"""
    tile_head_std: float = 0.02208
    """tile head init std"""
    summary_path: Optional[str] = None
    """path to write summary JSON (default: sft_summary.json next to sft.py)"""
    dataset_cache: Optional[str] = None
    """if set, materialise the training stream to this path once and reuse it on
    later runs (torch.save/torch.load), trading the streaming generation for a
    fixed on-disk dataset. Lets repeated runs (benchmarks, dev iteration) skip
    build_factory; the tensors are a pure function of (size, num_samples, seed)."""
