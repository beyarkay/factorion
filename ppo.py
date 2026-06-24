import json
import os
import typing
import random
import time
from dataclasses import dataclass
from typing import Any, Optional, cast
from collections import deque
from datetime import datetime
from pathlib import Path
import tempfile
from pathlib import Path
import shutil
import subprocess

import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
import factorion_rs
from factorion import (
    Channel,
    Direction,
    Footprint,
    LessonKind,
    Misc,
    blank_entities,
    build_factory,
    entities,
    items,
    str2ent,
    str2item,
)
from PIL import Image, ImageDraw, ImageFont

moving_average_length = 500
end_of_episode_thputs = deque(maxlen=moving_average_length)
for _ in range(moving_average_length):
    end_of_episode_thputs.append(0)
min_belts_thoughputs = [deque(maxlen=100) for _ in range(10)]

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    metal: bool = True
    """if toggled, Apple MPS will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "factorion"
    """the wandb's project name"""
    wandb_entity: Optional[str] = None
    """the entity (team) of wandb's project"""
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
    learning_rate: float = 5.86e-4
    """the learning rate of the optimizer"""
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
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    adam_epsilon: float = 6.866e-06
    """The epsilon parameter for Adam"""
    weight_decay: float = 0.0
    """L2 weight decay for the Adam optimiser."""
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
    tile_head_std: float = 0.06503
    """Initialization std for the tile selection conv head (smaller = more uniform initial exploration)"""
    dropout: float = 0.0
    """Dropout probability in the CNN encoder."""
    size: int = 11
    """The width and height of the factory"""
    summary_path: Optional[str] = None
    """path to write summary JSON (default: summary.json next to ppo.py)"""
    wandb_group: Optional[str] = None
    """W&B run group name (groups parallel seeds together in the dashboard)"""
    tags: typing.Optional[typing.List[str]] = None
    """Tags to apply to the wandb run."""
    critic_warmup: int = 0
    """Freeze the actor (encoder + all policy heads) for this many PPO iterations and train only the critic head, then unfreeze. An SFT checkpoint loads a trained actor but a random critic; without a warm-up the random critic's garbage advantages wreck the SFT policy in the first updates. 0 disables (default, preserves from-scratch behaviour). LR + entropy annealing start at unfreeze."""
    eval_every: int = 7
    """Run the greedy held-out eval (eval/thput, eval/thput_eot, per-lesson) every N PPO iterations (and on the final iteration). Mirrors the SFT rollout eval so the curves overlay the SFT baseline. 0 disables."""
    eval_seeds_per_kind: int = 12
    """Held-out factories per LessonKind in the greedy eval set."""
    eval_num_envs: int = 8
    """Parallel envs for the greedy eval rollout."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (num_envs * num_steps)"""
    minibatch_size: int = 0
    """the mini-batch size (batch_size // num_minibatches)"""
    num_iterations: int = 0
    """the number of iterations (total_timesteps // batch_size)"""


def _append_run_tags(run, *tags: str) -> None:
    """Append tags to a (possibly None / disabled) W&B run."""
    if run is not None:
        run.tags = (run.tags or ()) + tags


def _run_signature(args) -> str:
    """Filename-safe W&B run name encoding the key hyperparameters, so runs are
    identifiable at a glance instead of by timestamp (mirrors SFT's naming).
    e.g. ``ppo-s11-lr5e-05-ent0-cw10-fromj0s5y2mc-c93-69-96-seed1``."""
    layers = "-".join(str(c) for c in layers_from_args(args))
    sig = f"ppo-s{args.size}-lr{args.learning_rate:g}-ent{args.ent_coef_start:g}"
    if args.ent_coef_end != args.ent_coef_start:
        sig += f"_{args.ent_coef_end:g}"
    if args.target_kl is not None:
        sig += f"-kl{args.target_kl:g}"
    if args.critic_warmup:
        sig += f"-cw{args.critic_warmup}"
    if args.start_from:
        sig += f"-from{args.start_from}"
    sig += f"-c{layers}-seed{args.seed}"
    return sig


def _build_eval_set(args) -> dict:
    """Fixed held-out (seed -> LessonKind.value) factories for the greedy eval,
    disjoint from the training seeds. Each LessonKind gets its own high seed
    range so seeds never collide across kinds; only seeds where build_factory
    succeeds are kept (rejection sampling fails on some seed/kind/grid combos)."""
    out: dict = {}
    for ki, kind in enumerate(LessonKind):
        base = 9_000_000 + args.seed + ki * 100_000
        found, s = 0, base
        while found < args.eval_seeds_per_kind and s < base + 5000:
            if build_factory(size=args.size, kind=kind, seed=s) is not None:
                out[s] = kind.value
                found += 1
            s += 1
    return out


def _run_greedy_eval(agent, args, eval_seeds_to_kind, device) -> dict:
    """Greedy held-out throughput eval, mirroring SFT's val/thput[_eot] so
    the curves overlay. Returns a flat dict of eval/* metrics. Reuses SFT's
    run_rollout_eval (lazy import: sft imports ppo, so a top-level import would
    be circular); it only reads .size/.seed/.max_level off args, hence the shim."""
    from types import SimpleNamespace

    from sft import run_rollout_eval

    # Duck-typed args shim: run_rollout_eval only reads .size/.seed/.max_level.
    eval_args = SimpleNamespace(size=args.size, seed=args.seed, max_level=0)
    roll = run_rollout_eval(
        agent,
        eval_args,  # ty: ignore[invalid-argument-type]
        eval_seeds_to_kind,
        device,
        max_seeds=len(eval_seeds_to_kind),
        eot_threshold=0.5,
        num_envs=args.eval_num_envs,
    )
    metrics = {
        "eval/thput": roll["overall"],
        "eval/thput_eot": roll["overall_eot"],
    }
    for kn, thp in roll["per_kind"].items():
        if roll["per_kind_n"].get(kn, 0) > 0:
            metrics[f"eval/{kn}/thput"] = thp
    for kn, thp in roll["per_kind_eot"].items():
        if roll["per_kind_n"].get(kn, 0) > 0:
            metrics[f"eval/{kn}/thput_eot"] = thp
    return metrics


def _rollout_episode_metrics(
    lesson: str,
    *,
    episode_return: float,
    episode_len: float,
    thput_normed: float,
    thput_raw: float,
    ended_by_eot: float,
    invalid_frac: float,
    num_entities: float,
    min_entities_required: float,
    frac_reachable: float,
) -> dict:
    """Build the rollout/* metrics for one finished episode (overall + per-lesson).

    Pure (no wandb/torch) so it can be unit-tested. The per-lesson keys carry
    the lesson name, so each averages over only that lesson's episodes. Both the
    overall and per-lesson views log thput_raw (items/s) alongside the
    normalized throughput, so lessons with very different ceilings (belts ~15/s
    vs assemblers <1/s) stay comparable in raw terms.
    """
    return {
        "rollout/thput": float(thput_normed),
        "rollout/thput_raw": float(thput_raw),
        "rollout/reward": float(episode_return),
        "rollout/length": float(episode_len),
        "rollout/eot_rate": float(ended_by_eot),
        "rollout/invalid_frac": float(invalid_frac),
        "rollout/num_entities": float(num_entities),
        "rollout/entity_efficiency": float(min_entities_required) / float(num_entities),
        "rollout/frac_reachable": float(frac_reachable),
        # Per-lesson breakdown — each averages over only this lesson's episodes.
        f"rollout/{lesson}/thput": float(thput_normed),
        f"rollout/{lesson}/thput_raw": float(thput_raw),
        f"rollout/{lesson}/reward": float(episode_return),
        f"rollout/{lesson}/length": float(episode_len),
    }


def _resolve_wandb_checkpoint(
    run_spec: str, project: str, entity: Optional[str]
) -> tuple[str, dict]:
    """Resolve a W&B run id to (local_path, source_metadata). Downloads
    the run's most recent model-type artifact to /tmp/factorion-checkpoints.

    The metadata dict (run_id, run_url, run_name, artifact name) lets callers
    record/display provenance (e.g. the factory_builder UI) instead of the
    anonymous tmp download path.

    `run_spec` is either a bare id ("abc123") or a full path
    ("user/factorion/abc123"). Sets WANDB_MODE/WANDB_DISABLED back to online
    for the duration of the call — a caller that disabled W&B (e.g. the
    factory_builder local server) still needs to fetch the artifact."""
    import wandb

    prev_mode = os.environ.pop("WANDB_MODE", None)
    prev_disabled = os.environ.pop("WANDB_DISABLED", None)
    try:
        api = wandb.Api()
        if run_spec.count("/") == 2:
            run = api.run(run_spec)
        else:
            ent = entity or api.default_entity
            run = api.run(f"{ent}/{project}/{run_spec}")
        dest = Path("/tmp/factorion-checkpoints") / run.id
        dest.mkdir(parents=True, exist_ok=True)

        model_arts = [a for a in run.logged_artifacts() if a.type == "model"]
        if not model_arts:
            raise RuntimeError(
                f"run {run.id} has no artifacts of type=model — "
                f"was it trained with --track and the artifact-upload code?"
            )
        # Newest first. Each `download()` returns the local dir holding
        # the artifact's files.
        art = max(model_arts, key=lambda a: a.created_at)
        local_dir = Path(art.download(root=str(dest / art.name.replace(":", "_"))))
        pt_files = sorted(local_dir.glob("*.pt"))
        if not pt_files:
            raise RuntimeError(f"artifact {art.name} contains no .pt file")
        path = str(pt_files[0])
        print(f"Resolved {run_spec} -> {art.name} -> {path}")
        source = {
            "kind": "wandb",
            "run_id": run.id,
            "run_url": run.url,
            "run_name": run.name,
            "artifact": art.name,
        }
        return path, source
    finally:
        if prev_mode is not None:
            os.environ["WANDB_MODE"] = prev_mode
        if prev_disabled is not None:
            os.environ["WANDB_DISABLED"] = prev_disabled


def _resolve_start_from(
    start_from: str, project: str, entity: Optional[str] = None
) -> str:
    """Resolve ``--start-from`` to a local ``.pt`` checkpoint path.

    ``--start-from`` accepts either a local checkpoint path or a W&B run id
    (e.g. an SFT run like ``j0s5y2mc``). A path that exists on disk is returned
    unchanged. Otherwise the value is treated as a W&B run id and the run's
    model artifact (SFT uploads its best checkpoint there) is downloaded via
    the shared :func:`_resolve_wandb_checkpoint`.
    """
    if os.path.exists(start_from):
        return start_from
    # A value ending in .pt is clearly meant as a path — surface the missing
    # file plainly instead of misinterpreting it as a W&B run id.
    if start_from.endswith(".pt"):
        raise FileNotFoundError(
            f"--start-from='{start_from}' looks like a checkpoint path but does "
            f"not exist."
        )
    path, _source = _resolve_wandb_checkpoint(start_from, project, entity)
    return path


def make_env(env_id, idx, capture_video, size, run_name, throughput_reward_scale=1.0, step_penalty=0.01):
    def thunk():
        kwargs: dict[str, Any] = {"render_mode": "rgb_array"} if capture_video else {}
        kwargs.update({'size': size, 'max_steps': size*size, 'idx': idx,
                       'throughput_reward_scale': throughput_reward_scale,
                       'step_penalty': step_penalty})
        env = gym.make(env_id, **kwargs)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}/env_{idx}", episode_trigger=lambda e: (e+1) % 10 == 0)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

mapping = {
    # transport belt
    (1, 1): '↑',
    (1, 2): '→',
    (1, 3): '↓',
    (1, 4): '←',
    # sink
    (5, 1):  '📥',
    (5, 2):  '📥',
    (5, 3):  '📥',
    (5, 4): '📥',
    # source
    (6, 1):  '📤',
    (6, 2):  '📤',
    (6, 3):  '📤',
    (6, 4): '📤',
}

def get_pretty_format(tensor, entity_dir_map):
    assert isinstance(tensor, torch.Tensor), f"Input must be a torch tensor but is {tensor}"
    assert tensor.ndim == 3 and tensor.shape[0] >= 2, f"Tensor must have shape (2+, W, H) but has shape {tensor.shape}"
    assert tensor.shape[1] == tensor.shape[2], f"Expected world to be square, but is of shape {tensor.shape}"

    _, W, H = tensor.shape
    entities = tensor[0]
    directions = tensor[1]

    lines = []
    for y in range(H):
        line = []
        for x in range(W):
            ent = int(entities[x, y])
            direc = int(directions[x, y])
            char = entity_dir_map.get((ent, direc), str(ent))
            if char == '0':
                char = '.'
            if char in ('📤', '📥'):
                line.append(f"{char:^2}")
            else:
                line.append(f"{char:^3}")
        lines.append(" ".join(line))
    return "\n".join(lines)


class FactorioEnv(gym.Env):
    def __init__(
        self,
        size: int = 11,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        idx: Optional[int] = None,
        options: Optional[dict] = None,
        throughput_reward_scale: float = 1.0,
        step_penalty: float = 0.01,
    ):
        super().__init__()
        self.throughput_reward_scale = throughput_reward_scale
        self.step_penalty = step_penalty
        if render_mode is not None:
            self.metadata = {"render_modes": [render_mode], "render_fps": 2}
            self.render_mode = render_mode
            self.render_modes = [render_mode]

        self.size = size
        if max_steps is None:
            max_steps = self.size * self.size

        if idx is None:
            idx = 0
        self.idx = idx

        print(f"FactorioEnv({size=}, {max_steps=}, {render_mode=}, {idx=})")
        self.max_steps = max_steps

        self._world_CWH = torch.zeros((len(Channel), self.size, self.size))

        self.max_id_in_tensor = max(len(items), len(entities), len(Direction))
        # Observation is the world, with a square grid of tiles and one channel
        # representing the entity ID, the other representing the direction
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.max_id_in_tensor,
            shape=(len(Channel), self.size, self.size),
            dtype=np.int64,
        )


        self.action_space = gym.spaces.Dict({
            "xy": gym.spaces.Box(low=0, high=self.size, shape=(2,), dtype=np.int64),
            "entity": gym.spaces.Discrete(len(entities)),
            "direction": gym.spaces.Discrete(len(Direction)),
            "item": gym.spaces.Discrete(len(items)),
            "misc": gym.spaces.Discrete(len(Misc)),
            "eot": gym.spaces.Discrete(2),
        })
        # Cache source/sink IDs (non-placeable prototypes)
        self._source_id = str2ent('stack_inserter').value
        self._sink_id = str2ent('bulk_inserter').value
        self._reset_options = options if options is not None else {}

        self.steps = 0
        # Overwritten in reset() with the episode's actual lesson; default keeps
        # _get_info() safe if it is ever called before the first reset.
        self._kind = LessonKind.MOVE_ONE_ITEM

    def _get_obs(self):
        return self._world_CWH

    def _get_info(self):
        return {
            # Lesson kind as an int (LessonKind.value) so SyncVectorEnv can
            # stack it across envs; the rollout loop buckets per-lesson metrics
            # by LessonKind(int).name.
            'kind': self._kind.value,
            # thput_raw: raw items/second (reference-free). thput_normed:
            # raw / per-factory max, clamped to [0, 1] (1.0 == reproduced the
            # reference factory's throughput). See FIXME(#161) in step().
            'thput_raw': self._thput_raw,
            'thput_normed': self._thput_normed,
            'frac_reachable': self._frac_reachable,
            'frac_hallucin': self._frac_hallucin,
            'final_dir_reward': self._final_dir_reward,
            'material_cost': self._material_cost,
            'reward': self._reward,
            'cum_reward': self._cum_reward,
        }

    def _compute_solution_match(self):
        """Compute similarity to solved factory over solution-nonempty tiles only.
        Returns (location_match, entity_match, direction_match) each in [0, 1]."""
        orig_ent = self._solved_world_CWH[Channel.ENTITIES.value]
        curr_ent = self._world_CWH[Channel.ENTITIES.value]
        orig_dir = self._solved_world_CWH[Channel.DIRECTION.value]
        curr_dir = self._world_CWH[Channel.DIRECTION.value]

        mask = (orig_ent != 0)
        n = mask.sum().item()
        if n == 0:
            return 1.0, 1.0, 1.0

        location_match = (curr_ent[mask] != 0).float().sum().item() / n
        entity_match = (curr_ent[mask] == orig_ent[mask]).float().sum().item() / n
        direction_match = (curr_dir[mask] == orig_dir[mask]).float().sum().item() / n

        return location_match, entity_match, direction_match

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is None:
            seed = 0
        self._seed = int(seed + self.idx)
        super().reset(seed=self._seed)
        if options is not None:
            self._reset_options = options
        self._cum_reward = 0

        self.invalid_actions = 0
        self._thput_raw = 0.0
        self._thput_normed = 0.0
        self._frac_reachable = 0
        self._frac_hallucin = 0
        self._final_dir_reward = 0
        self._material_cost = 0
        self._reward = 0
        self._terminated = False
        self._truncated = False
        self.max_entities = 2
        self._num_missing_entities = self._reset_options.get('num_missing_entities', float('inf'))

        self.actions = []
        # Lesson kind is settable per-reset. Default (omitted or None) is
        # uniform random sampling across LessonKind on every reset —
        # matches the project direction where lessons are data generators
        # rather than a fixed curriculum, so a single env naturally sees
        # all kinds. Pass a concrete LessonKind to pin one. The sampler
        # uses self.np_random which super().reset() seeded above, so the
        # choice is deterministic per episode seed.
        #
        # build_factory returns None when rejection sampling exhausts
        # itself for a given (size, kind, seed); for the random-kind
        # path we just sample a different kind, since they have very
        # different rejection rates on small grids.
        kind_opt = self._reset_options.get('kind', None)
        factory = None
        if kind_opt is None:
            kinds_list = list(LessonKind)
            for _ in range(16):
                kind = kinds_list[int(self.np_random.integers(0, len(kinds_list)))]
                factory = build_factory(
                    size=self.size, kind=kind, seed=self._seed,
                )
                if factory is not None:
                    break
            else:
                raise RuntimeError(
                    f"Failed to sample a valid LessonKind for seed={self._seed} "
                    f"after 16 attempts"
                )
        else:
            kind = kind_opt
            factory = build_factory(
                size=self.size, kind=kind, seed=self._seed,
            )
            if factory is None:
                raise RuntimeError(
                    f"build_factory returned None for kind={kind} "
                    f"seed={self._seed}"
                )
        self._kind = kind
        self._solved_world_CWH = factory.world_CWH
        # Per-factory maximum throughput: the raw items/second the complete,
        # correct factory carries. We normalize the agent's raw throughput by
        # this so a perfectly-rebuilt factory scores 1.0 regardless of its
        # absolute speed (different lessons/layouts have very different maxima;
        # the old fixed /15.0 mislabeled any factory whose max was < 15).
        # FIXME(#161): this depends on having the full scripted solution to
        # compute the max. Fine while every episode is a scripted lesson, but
        # arbitrary RL rollouts won't have a reference factory — see GH #161.
        max_tp, _ = factorion_rs.simulate_throughput(
            self._solved_world_CWH.permute(1, 2, 0).to(torch.int64).numpy()
        )
        self._max_throughput = float(max_tp)
        self._world_CWH, min_entities_required = blank_entities(
            factory, num_missing_entities=self._num_missing_entities
        )

        self.min_entities_required = min_entities_required
        self._original_world_CWH = torch.clone(self._world_CWH)
        self._prev_match = self._compute_solution_match()
        self.steps = 0
        return self._world_CWH.cpu().numpy(), self._get_info()

    def step(self, action):
        x, y = action["xy"]
        entity_id = action["entity"]
        direc = action["direction"]
        item_id = action["item"]
        misc = action["misc"]


        assert 0 <= x < self._world_CWH.shape[1], f"x={x} out of bounds [0, {self._world_CWH.shape[1]})"
        assert 0 <= y < self._world_CWH.shape[2], f"y={y} out of bounds [0, {self._world_CWH.shape[2]})"
        source_id = self._source_id
        sink_id = self._sink_id

        # Mutate the world with the agent's actions
        entity_to_be_replaced = self._world_CWH[Channel.ENTITIES.value, x, y]

        self.actions.append(None)
        action_is_invalid = False
        invalid_reason = {
            'placed_on_masked_tile': False,
            'replaced_source_or_sink': False,
            'placed_source_or_sink': False,
            'place_asm_mach_wo_recipe': False,
            'placement_wo_direction': False,
            'direction_wo_entity': False,
            'ug_belt_wo_up_or_down': False,
            'placement_with_unneeded_misc': False,
            'too_wide': False,
            'too_tall': False,
        }

        # Check that the action is actually valid
        if not (0 <= entity_id < len(entities)):
            action_is_invalid = True
        elif not (0 <= direc < len(Direction)):
            action_is_invalid = True
        elif entity_id in (source_id, sink_id):
            # agent tried to place a source or sink
            invalid_reason['placed_source_or_sink'] = True
            action_is_invalid = True
        elif entity_id == str2ent('assembling_machine_1').value and item_id == str2item('empty'):
            # Model is trying to place an assembling machine without a recipe
            invalid_reason['place_asm_mach_wo_recipe'] = True
            action_is_invalid = True
            pass
        elif entity_id not in (str2ent('empty').value, str2ent('assembling_machine_1').value) and direc == Direction.NONE.value:
            # Model is trying to put a thing without giving a direction
            invalid_reason['placement_wo_direction'] = True
            action_is_invalid = True
            pass
        elif entity_id == str2ent('empty').value and direc != Direction.NONE.value:
            # Model is trying to put a thing without giving a direction
            invalid_reason['direction_wo_entity'] = True
            action_is_invalid = True
            pass
        elif (misc == Misc.NONE.value) and (entity_id == str2ent('underground_belt').value):
            # model is trying to place an underground belt without giving a down/up
            invalid_reason['ug_belt_wo_up_or_down'] = True
            action_is_invalid = True
            pass
        elif (misc != Misc.NONE.value) and (entity_id != str2ent('underground_belt').value):
            # model is trying to place a thing that doesn't need a Misc but
            # still giving it a Misc
            invalid_reason['placement_with_unneeded_misc'] = True
            action_is_invalid = True
            pass
        else:
            # Compute the entity's full footprint (anchor for 1x1, all
            # occupied tiles for multi-tile). py_entity_tiles handles
            # rotation correctly; the previous x+width/y+height bounds
            # checks were direction-agnostic and wrong for rotated splitters.
            proto = entities[entity_id]
            tiles_list = factorion_rs.py_entity_tiles(x, y, direc, proto.width, proto.height)
            if tiles_list is None:
                tiles_list = [(x, y)]
            tiles_list = [tuple(t) for t in tiles_list]

            # Validate every tile of the footprint. Multi-tile placements
            # were previously only validated at the anchor, so a splitter
            # could overlap an existing belt at its secondary tile or
            # extend off-grid undetected.
            out_of_bounds = any(
                not (0 <= tx < self.size and 0 <= ty < self.size)
                for tx, ty in tiles_list
            )
            if out_of_bounds:
                invalid_reason['too_wide'] = True
                action_is_invalid = True
            elif any(
                self._world_CWH[Channel.FOOTPRINT.value, tx, ty]
                    == Footprint.UNAVAILABLE.value
                for tx, ty in tiles_list
            ):
                invalid_reason['placed_on_masked_tile'] = True
                action_is_invalid = True
            elif any(
                int(self._world_CWH[Channel.ENTITIES.value, tx, ty]) in (source_id, sink_id)
                for tx, ty in tiles_list
            ):
                invalid_reason['replaced_source_or_sink'] = True
                action_is_invalid = True
            else:
                action_is_invalid = False
                # Write entity_id and direction at every tile of the
                # footprint, matching how lessons and place_multi_tile
                # represent multi-tile entities. Items + misc go on the
                # anchor only (those channels are anchor-scoped).
                for tx, ty in tiles_list:
                    self._world_CWH[Channel.ENTITIES.value, tx, ty] = entity_id
                    self._world_CWH[Channel.DIRECTION.value, tx, ty] = direc
                self._world_CWH[Channel.ITEMS.value, x, y] = item_id
                self._world_CWH[Channel.MISC.value, x, y] = misc
                self.actions[-1] = {
                    'entity': entities[entity_id].name,
                    'xy': (x, y),
                    'direction': Direction(direc),
                    'item': items[item_id].name,
                    'misc': Misc(misc),
                }

        self.invalid_actions += 1 if action_is_invalid else 0

        thput_raw, num_unreachable = factorion_rs.simulate_throughput(self._world_CWH.permute(1, 2, 0).to(torch.int64).numpy())
        # thput_raw is items/second — the reference-free signal arbitrary RL
        # should ultimately optimize. thput_normed in [0, 1] is raw / per-factory
        # max, so a perfectly-rebuilt factory scores 1.0 no matter its absolute
        # speed (the old fixed /15.0 scored a perfect sub-15 factory as e.g. 0.33).
        # FIXME(#161): thput_normed, the termination check, and the reward below
        # all depend on self._max_throughput, which we only know because the
        # lesson is scripted and we have the full solution. Arbitrary RL rollouts
        # won't have that reference; this must move to a reference-free signal.
        thput_normed = (
            min(1.0, thput_raw / self._max_throughput)
            if self._max_throughput > 0
            else 0.0
        )

        # Calculate a "reachable" fraction that penalises the model for leaving
        # entities disconnected from the graph (almost certainly useless)
        num_entities = self._world_CWH[Channel.ENTITIES.value].count_nonzero()
        # NOTE: weird bug with num_unreachable calculations, not planning on
        # fixing any time super soon. really the calculation should be to
        # calculate frac_reachable directly, not go via frac_unreachable
        frac_reachable = 0 if num_entities == 2 else max(0, 1.0 - (float(num_unreachable) / (num_entities - 2)))
        frac_hallucin = 0

        # Give some small reward for having the belt be the right direction.
        # Only meaningful when the layout has exactly one sink (the
        # MOVE_ONE_ITEM case). Other lesson kinds (e.g. SPLITTER_SPLIT)
        # place multiple sinks or none, so this shaping term gets zeroed
        # instead of asserting.
        sink_locs = torch.where(self._world_CWH[Channel.ENTITIES.value] == self._sink_id)
        C, W, H = self._world_CWH.shape
        if len(sink_locs[0]) == 1:
            w_sink, h_sink = sink_locs[0][0], sink_locs[1][0]
            w_belt = torch.clamp(w_sink, 1, W-2)
            h_belt = torch.clamp(h_sink, 1, H-2)
            final_belt_dir = self._world_CWH[Channel.DIRECTION.value, w_belt, h_belt]
            sink_dir = self._world_CWH[Channel.DIRECTION.value, w_sink, h_sink]
            final_dir_reward = 1.0 if final_belt_dir == sink_dir else 0.0
        else:
            final_dir_reward = 0.0

        material_cost = (
            1.0 * (self._world_CWH[Channel.DIRECTION.value] == str2ent('transport_belt').value).sum()
            + 1.5 * (self._world_CWH[Channel.DIRECTION.value] == str2ent('underground_belt').value).sum()
            + 2.0 * (self._world_CWH[Channel.DIRECTION.value] == str2ent('assembling_machine_1').value).sum()
        )

        # ── Diagnostic tile-match metrics (for logging, NOT used in reward) ──
        orig_ent = self._solved_world_CWH[Channel.ENTITIES.value]
        curr_ent = self._world_CWH[Channel.ENTITIES.value]
        orig_dir = self._solved_world_CWH[Channel.DIRECTION.value]
        curr_dir = self._world_CWH[Channel.DIRECTION.value]

        solution_nonempty = (orig_ent != 0)
        current_nonempty = (curr_ent != 0)
        num_solution_nonempty = solution_nonempty.sum().item()
        if num_solution_nonempty > 0:
            tile_match_location = (solution_nonempty & current_nonempty).sum().item() / num_solution_nonempty
        else:
            tile_match_location = 1.0
        tile_match_entity = (curr_ent == orig_ent).float().mean().item()
        tile_match_direction = (curr_dir == orig_dir).float().mean().item()

        # ── Delta-based shaping diagnostics (logged in info, NOT in reward) ──
        # Computed over solution-nonempty tiles only.
        curr_match = self._compute_solution_match()
        loc_delta = curr_match[0] - self._prev_match[0]
        ent_delta = curr_match[1] - self._prev_match[1]
        dir_delta = curr_match[2] - self._prev_match[2]
        self._prev_match = curr_match

        eot_declared = int(action.get("eot", 0)) == 1
        terminated = eot_declared
        truncated = (not terminated) and (self.steps > self.max_steps)

        reward = -self.step_penalty
        if terminated or truncated:
            reward += self.throughput_reward_scale * thput_normed

        self._thput_raw = thput_raw
        self._thput_normed = thput_normed
        self._frac_reachable = frac_reachable
        self._frac_hallucin = frac_hallucin
        self._final_dir_reward = final_dir_reward
        self._material_cost = material_cost
        self._reward = reward
        self._terminated = terminated
        self._truncated = truncated

        observation = self._get_obs()
        info = self._get_info()
        if terminated or truncated:
            info.update({ 'steps_taken': self.steps })

        num_placed_entities = len([a for a in self.actions if a is not None and a['entity'] != 'empty'])

        info.update({
            'thput_raw': thput_raw,
            'thput_normed': thput_normed,
            'frac_reachable': frac_reachable,
            'frac_hallucin': frac_hallucin,
            'final_dir_reward': final_dir_reward,
            'material_cost': material_cost,
            'completion_bonus': self.max_steps - self.steps,
            'min_entities_required': self.min_entities_required,
            'num_entities': num_entities,
            'num_placed_entities': num_placed_entities,
            'frac_invalid_actions': self.invalid_actions / self.max_steps,
            'max_entities': self.max_entities,
            'invalid_reason': invalid_reason,
            'tile_match_location': tile_match_location,
            'tile_match_entity': tile_match_entity,
            'tile_match_direction': tile_match_direction,
            'shaping_location_match': curr_match[0],
            'shaping_entity_match': curr_match[1],
            'shaping_direction_match': curr_match[2],
            'shaping_location_delta': loc_delta,
            'shaping_entity_delta': ent_delta,
            'shaping_direction_delta': dir_delta,
        })

        self._cum_reward += reward
        self.steps += 1

        return observation.numpy(), float(reward), terminated, truncated, info

    def render(self):
        """
        Grid renderer with icons, direction arrow, and recipe icon.
        Returns an RGB uint8 array compatible with Gymnasium video wrappers.
        """
        # ------------------- constants -------------------
        ICON_DIR       = Path("factorio-icons")   # where all *.png live
        CELL_PX        = 64                       # tile size in pixels
        MINI_PX        = 18                       # size for arrow + recipe icon
        GRID_COLOR     = (0, 0, 0)                # black grid lines

        # ---------------- lazy one-time setup ------------
        if not hasattr(self, "_render_cache"):
            # cache will store resized sprites + generated arrows
            self._render_cache = {"entity": {}, "item": {}, "arrow": {}}

            # entity icons (resized to CELL_PX)
            for ent_id, ent in entities.items():
                p = ICON_DIR / f"{ent.name}.png"
                if p.exists():
                    img = Image.open(p).convert("RGBA").resize(((CELL_PX // 10) * 8, (CELL_PX // 10) * 8), Image.Resampling.BICUBIC)
                    self._render_cache["entity"][ent_id] = img

            # item (recipe) icons (resized to MINI_PX)
            for itm_id, itm in items.items():
                if itm_id == 0:  # 0 = empty → no icon
                    continue
                p = ICON_DIR / f"{itm.name}.png"
                if p.exists():
                    img = Image.open(p).convert("RGBA").resize((MINI_PX, MINI_PX), Image.Resampling.BICUBIC)
                    self._render_cache["item"][itm_id] = img

            # tiny triangular arrow, rotated for each cardinal direction
            base = Image.new("RGBA", (MINI_PX, MINI_PX), (0, 0, 0, 0))
            draw = ImageDraw.Draw(base)
            # shaft: thin vertical rectangle
            shaft_w = max(1, MINI_PX // 12)
            shaft_x0 = (MINI_PX - shaft_w) // 2
            draw.rectangle(
                [shaft_x0, MINI_PX // 3, shaft_x0 + shaft_w, MINI_PX - 1],
                fill=(0, 0, 0, 255)
            )
            # head: triangle at the top
            draw.polygon(
                [
                    (MINI_PX // 2, 0),           # tip
                    (shaft_x0 + shaft_w*4, MINI_PX // 3),
                    (shaft_x0 - shaft_w*4, MINI_PX // 3),
                ],
                fill=(0, 0, 0, 255)
            )
            # cache rotations for each cardinal direction
            self._render_cache["arrow"][Direction.NORTH.value] = base
            self._render_cache["arrow"][Direction.EAST.value]  = base.rotate(-90, expand=True)
            self._render_cache["arrow"][Direction.SOUTH.value] = base.rotate(180, expand=True)
            self._render_cache["arrow"][Direction.WEST.value]  = base.rotate(90,  expand=True)

            # default font for fallback label
            self._font = ImageFont.load_default()

        cache = self._render_cache
        ENT, DIR, ITEM = Channel.ENTITIES.value, Channel.DIRECTION.value, Channel.ITEMS.value

        # ---------------- canvas -------------------------
        HUD_PX = 32*4                                     # height of stats strip
        canvas_w = canvas_h = self.size * CELL_PX
        canvas = Image.new("RGB", (canvas_w, canvas_h + HUD_PX), (255, 255, 255))
        draw   = ImageDraw.Draw(canvas)

        ent_layer  = self._world_CWH[ENT].cpu().numpy()
        dir_layer  = self._world_CWH[DIR].cpu().numpy()
        item_layer = self._world_CWH[ITEM].cpu().numpy()

        for gx in range(self.size):          # grid row
            for gy in range(self.size):      # grid col
                x0, y0 = gx * CELL_PX, gy * CELL_PX
                x1, y1 = x0 + CELL_PX, y0 + CELL_PX
                draw.rectangle([x0, y0, x1, y1], outline=GRID_COLOR, width=1)

                ent_id = int(ent_layer[gx, gy])
                sprite = cache["entity"].get(ent_id)
                if sprite is not None:
                    canvas.paste(sprite, (x0 + CELL_PX//10, y0 + CELL_PX//10), sprite)
                else:
                    # fallback: draw first letter
                    letter = entities[ent_id].name[0].upper()
                    font   = self._font
                    canvas_w, canvas_h   = font.getbbox(letter)[2:]
                    draw.text(
                        (x0 + (CELL_PX - canvas_w) // 2, y0 + (CELL_PX - canvas_h) // 2),
                        letter,
                        fill=(0, 0, 0),
                        font=font,
                    )
                # Draw the xy-coords onto the cell
                text = f"{gx},{gy}"
                font   = self._font
                draw.text(
                    (x0 + (CELL_PX // 10) * 8, y0 + CELL_PX // 10),
                    text,
                    fill=(0, 0, 0),
                    font=font,
                )

                itm_id = int(item_layer[gx, gy])
                mini   = cache["item"].get(itm_id)
                if mini is not None:
                    off = CELL_PX - MINI_PX - 2
                    canvas.paste(mini, (x0 + off, y0 + off), mini)

                dir_id = Direction(dir_layer[gx, gy]).value
                arrow  = cache["arrow"].get(dir_id)
                if arrow is not None:
                    canvas.paste(arrow, (x0 + 2, y0 + 2), arrow)

        info = self._get_info() or {}
        unreach = info.get("frac_reachable", "?")
        cum_reward = info.get("cum_reward", 0)
        thput_raw = info.get("thput_raw", "?")
        thput_normed = info.get("thput_normed", "?")
        reward = info.get("reward", "?")

        lines = [
            f"Step {self.steps}/{self.max_steps}",
            f"SumReward: {cum_reward:.3f}  |  Reward: {reward:.3f}",
            f"thrput: {thput_normed} ({thput_raw}/s)  |  Unreachable: {unreach:.3f}",
        ]
        if self.actions and self.actions[-1] is not None:
            lines.append(f"Placing {self.actions[-1]['entity']}")
            lines.append(f"    at {self.actions[-1]['xy']}")
            lines.append(f"    facing {self.actions[-1]['direction']}")

        font = self._font
        bbox = font.getbbox(lines[0])
        txt_h = bbox[3] - bbox[1]

        draw.rectangle([(0, canvas_h), (canvas_w, canvas_h + HUD_PX)], fill=(230, 230, 230))
        for i, line in enumerate(lines):
            draw.text(
                (4, 4 + canvas_h + i * txt_h * 1.25),
                line,
                fill=(0, 0, 0),
                font=font,
            )

        return np.asarray(canvas, dtype=np.uint8)


NUM_LAYER_SLOTS = 8


def layers_from_args(args) -> list[int]:
    """Compact the ``layer1..layer{NUM_LAYER_SLOTS}`` width slots on an
    ``Args``/``SFTArgs`` into the CNN encoder's channel list: every slot with
    positive width becomes one conv layer, in slot order; a zero-width slot is
    dropped. This lets a W&B Bayesian sweep tune the encoder's depth *and*
    per-layer width as independent numeric dimensions (drive a slot toward 0 to
    remove a layer) instead of an opaque categorical "64,64,64" string."""
    slots = [getattr(args, f"layer{i}") for i in range(1, NUM_LAYER_SLOTS + 1)]
    layers = [c for c in slots if c > 0]
    if not layers:
        raise ValueError("at least one of layer1..layer8 must have positive width")
    return layers


class AgentCNN(nn.Module):
    def __init__(self, envs, layers=(48, 48, 64), kernel_size=3, tile_head_std=0.01, dropout=0.0):
        super().__init__()
        base_env = envs.envs[0].unwrapped
        self.width = base_env.size
        self.height = base_env.size
        self.channels = len(Channel)
        # Source/sink (bulk_inserter, stack_inserter) live as the last two
        # catalog entries; they are env-spawned, never agent-placeable. Sizing
        # the head to len(entities)-2 makes them structurally impossible to
        # sample, so we never waste samples on placements the env rejects.
        self.num_entities = len(entities) - 2
        self.num_directions = len(Direction)
        self.num_items = len(items)
        self.num_misc = len(Misc)
        # Variable-depth conv encoder: one conv layer per entry in `layers`,
        # that entry giving the layer's channel width. `kernel_size` sets each
        # layer's receptive field (RF = 1 + len(layers) * (kernel_size - 1));
        # padding is pinned to kernel_size // 2 so every conv preserves the
        # W x H spatial dims ("same" convolution). That invariant is
        # load-bearing: the tile head emits one logit per grid cell and the
        # per-tile heads index encoded[:, :, x, y], both of which require the
        # feature map to stay exactly grid-sized.
        #
        # nn.Dropout2d(dropout) (spatial dropout) trails each ReLU; p=0.0 is a
        # no-op and dropout is inert in eval(). Because it inserts a non-conv
        # module between convs, conv weights no longer sit at encoder.0/2/4 —
        # anything reading them back from a checkpoint must locate by shape.
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        if len(layers) == 0:
            raise ValueError("layers must contain at least one conv layer")
        padding = kernel_size // 2
        conv_stack = []
        in_ch = self.channels
        for ch in layers:
            conv_stack.append(
                layer_init(nn.Conv2d(in_ch, ch, kernel_size=kernel_size, padding=padding))
            )
            conv_stack.append(nn.ReLU())
            conv_stack.append(nn.Dropout2d(dropout))
            in_ch = ch
        self.encoder = nn.Sequential(*conv_stack)
        self.layers = tuple(layers)
        self.kernel_size = kernel_size
        last_chan = layers[-1]  # encoder output channels — feeds every head
        num_params_encoder = sum(p.numel() for p in self.encoder.parameters())
        print(
            f"Encoder has {num_params_encoder} params "
            f"({len(layers)} layers {tuple(layers)}, kernel_size={kernel_size})"
        )

        flat_dim = last_chan * self.width * self.height

        # Project encoded state to value
        self.critic_head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(flat_dim, 1), std=1.0)
        )

        # Bias init at -2 so an untrained model defaults to "not finished"
        # (sigmoid(-2) ≈ 0.12).
        self.eot_head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(flat_dim, 1), std=1.0, bias_const=-2.0),
        )

        # Tile selection: 1x1 conv producing one logit per spatial position
        self.tile_logits = layer_init(nn.Conv2d(last_chan, 1, kernel_size=1), std=tile_head_std)

        # Per-tile entity/direction/item/misc heads (conditioned on selected
        # tile features). The env's step() requires all four to be set
        # consistently — e.g. an underground_belt placement must carry
        # misc=UNDERGROUND_DOWN/UP, and an assembling_machine_1 placement
        # must carry a recipe in `item`.
        self.ent_head = layer_init(nn.Linear(last_chan, self.num_entities))
        self.dir_head = layer_init(nn.Linear(last_chan, self.num_directions))
        self.item_head = layer_init(nn.Linear(last_chan, self.num_items))
        self.misc_head = layer_init(nn.Linear(last_chan, self.num_misc))
        self.time_for_get_value = None
        self.time_for_get_action_and_value = None

        # Bias every head toward its "empty / NONE" slot (value 0) so a
        # freshly-initialised policy mostly proposes no-ops, matching the
        # prior used for entity/direction.
        with torch.no_grad():
            for head in (self.ent_head, self.dir_head, self.item_head, self.misc_head):
                head.bias.fill_(0.0)
                head.bias.data[0] = 1.0

    def get_value(self, x_BCWH):
        t0 = time.time()
        encoded = self.encoder(x_BCWH)
        value_B = self.critic_head(encoded).squeeze(-1)
        self.time_for_get_value = time.time() - t0
        return value_B

    def eot_prob(self, x_BCWH):
        """End-of-turn probability per observation, in [0, 1].

        Kept off the `get_action_and_value` return tuple so the ~20 PPO /
        test callsites unpacking that 4-tuple don't have to change. Use
        this from inference rollouts to decide whether the agent thinks
        the factory is finished.
        """
        encoded = self.encoder(x_BCWH)
        return torch.sigmoid(self.eot_head(encoded).squeeze(-1))

    def eot_should_stop(self, x_BCWH, threshold: float = 0.5):
        """Boolean stop signal per observation. Threshold defaults to 0.5;
        lower it if the model rambles, raise it if it stops short."""
        return self.eot_prob(x_BCWH) > threshold

    def get_action_and_value(self, x_BCWH, action=None):
        t0 = time.time()

        # Encode input once and reuse for both action and value heads
        encoded_BCWH = self.encoder(x_BCWH)  # (B, chan3, W, H)
        value_B = self.critic_head(encoded_BCWH).squeeze(-1)

        B = encoded_BCWH.shape[0]

        # --- Tile selection: joint (x, y) via 1x1 conv ---
        tile_logits_B1WH = self.tile_logits(encoded_BCWH)      # (B, 1, W, H)
        tile_logits_BN = tile_logits_B1WH.reshape(B, -1)       # (B, W*H)
        dist_tile = Categorical(logits=tile_logits_BN)

        if action is None:
            tile_idx_B = dist_tile.sample()                     # (B,)
        else:
            # Reconstruct tile index from stored (x, y)
            tile_idx_B = action[:, 0] * self.height + action[:, 1]

        # Decode tile index back to (x, y)
        x_B = tile_idx_B // self.height
        y_B = tile_idx_B % self.height

        # --- Extract per-tile features at selected (x, y) ---
        batch_idx = torch.arange(B, device=encoded_BCWH.device)
        tile_features_BC = encoded_BCWH[batch_idx, :, x_B, y_B]  # (B, chan3)

        # --- Entity / direction / item / misc heads (conditioned on tile features) ---
        logits_e_BE = self.ent_head(tile_features_BC)
        logits_d_BD = self.dir_head(tile_features_BC)
        logits_i_BI = self.item_head(tile_features_BC)
        logits_m_BM = self.misc_head(tile_features_BC)
        dist_e = Categorical(logits=logits_e_BE)
        dist_d = Categorical(logits=logits_d_BD)
        dist_i = Categorical(logits=logits_i_BI)
        dist_m = Categorical(logits=logits_m_BM)

        eot_logit_B = self.eot_head(encoded_BCWH).squeeze(-1)
        dist_eot = Bernoulli(logits=eot_logit_B)

        if action is None:
            ent_B = dist_e.sample()
            dir_B = dist_d.sample()
            item_B = dist_i.sample()
            misc_B = dist_m.sample()
            eot_B = dist_eot.sample()
        else:
            ent_B = action[:, 2]
            dir_B = action[:, 3]
            item_B = action[:, 4]
            misc_B = action[:, 5]
            eot_B = action[:, 6].float()

        # --- Log probs and entropy ---
        logp_B = (
            dist_tile.log_prob(tile_idx_B) +
            dist_e.log_prob(ent_B) +
            dist_d.log_prob(dir_B) +
            dist_i.log_prob(item_B) +
            dist_m.log_prob(misc_B) +
            dist_eot.log_prob(eot_B)
        )
        # Per-head entropies, summed into the joint entropy used by PPO. Kept
        # individually so the rollout can log policy/entropy_{head} (which heads
        # are still exploring vs collapsed) — the RL analog of SFT's per-head
        # accuracy. Stashed as detached scalars (cheap; mirrors the
        # self.time_for_* attributes already set here, so it stays eager-safe).
        ent_tile = dist_tile.entropy()
        ent_e = dist_e.entropy()
        ent_d = dist_d.entropy()
        ent_i = dist_i.entropy()
        ent_m = dist_m.entropy()
        ent_eot = dist_eot.entropy()
        entropy_B = ent_tile + ent_e + ent_d + ent_i + ent_m + ent_eot
        self._last_head_entropy = {
            "tile": ent_tile.mean().detach(),
            "entity": ent_e.mean().detach(),
            "direction": ent_d.mean().detach(),
            "item": ent_i.mean().detach(),
            "misc": ent_m.mean().detach(),
            "eot": ent_eot.mean().detach(),
        }
        # Bernoulli(logits).probs == sigmoid(logits); compute directly (ty's
        # torch stubs don't expose the lazy .probs property).
        self._last_eot_prob = torch.sigmoid(eot_logit_B).mean().detach()

        action_out = {
            "xy": torch.stack([x_B, y_B], dim=1),
            "direction": dir_B,
            "entity": ent_B,
            "item": item_B,
            "misc": misc_B,
            "eot": eot_B,
        }
        self.time_for_get_action_and_value = time.time() - t0
        return action_out, logp_B, entropy_B, value_B

if __name__ == "__main__":
    print("Starting...")
    start_time = time.time()
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    num_gsteps = args.num_envs * args.num_steps * args.num_iterations
    print(f"batch_size: {args.batch_size}, minibatch_size: {args.minibatch_size}, num_iterations: {args.num_iterations}, num_gsteps: {num_gsteps}")

    run_name = _run_signature(args)
    run = None
    if args.track:
        import wandb
        if args.start_from_wandb is not None and args.start_from is None:
            raise Exception(f"args.start_from_wandb is not None ({args.start_from_wandb}) and args.start_from is None")

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            group=args.wandb_group,
            save_code=True,
            tags=args.tags,
            id=args.start_from_wandb,
            resume="must" if args.start_from_wandb is not None else None,
        )
        _append_run_tags(
            run,
            f"batch_size:{args.batch_size}",
            f"minibatch_size:{args.minibatch_size}",
            f"num_iterations:{args.num_iterations}",
            f"seed:{args.seed}",
            f"size:{args.size}",
            f"timesteps:{args.total_timesteps//1000}K",
            f"layers:{'-'.join(map(str, layers_from_args(args)))}",
            f"k:{args.kernel_size}",
        )

        # Define metric axes and summary aggregation. global_step (env steps)
        # is the x-axis for every panel. eval/* (greedy held-out throughput) is
        # the headline progress signal, so it summarises to its max.
        wandb.define_metric("*", step_metric="global_step")
        _LESSONS = [k.name for k in LessonKind]
        wandb.define_metric("eval/thput", summary="max")
        wandb.define_metric("eval/thput_eot", summary="max")
        wandb.define_metric("eval/seconds", summary="last")
        for ln in _LESSONS:
            wandb.define_metric(f"eval/{ln}/thput", summary="max")
            wandb.define_metric(f"eval/{ln}/thput_eot", summary="max")
        for m in ["thput", "thput_raw", "reward", "length", "eot_rate",
                  "invalid_frac", "num_entities", "entity_efficiency",
                  "frac_reachable"]:
            wandb.define_metric(f"rollout/{m}", summary="last")
        for ln in _LESSONS:
            for m in ["thput", "thput_raw", "reward", "length"]:
                wandb.define_metric(f"rollout/{ln}/{m}", summary="last")
        for m in ["entropy", "eot_prob"]:
            wandb.define_metric(f"policy/{m}", summary="last")
        for h in ["tile", "entity", "direction", "item", "misc", "eot"]:
            wandb.define_metric(f"policy/entropy_{h}", summary="last")
        for m in ["policy", "value", "entropy", "total", "approx_kl",
                  "clipfrac", "explained_variance"]:
            wandb.define_metric(f"losses/{m}", summary="last")
        for m in ["lr", "critic_lr", "ent_coef", "grad_norm", "critic_warmup"]:
            wandb.define_metric(f"optim/{m}", summary="last")
        for m in ["sps", "rollout_seconds", "update_seconds", "eval_seconds"]:
            wandb.define_metric(f"perf/{m}", summary="last")
    print("Registering factorio Gym env")
    # Register the factorio env. This runs only under __main__, so pass the
    # class directly rather than "ppo:FactorioEnv" — a string entry_point would
    # make gym re-import ppo as a second module (this one is __main__).
    # gymnasium accepts a class entry_point; its EnvCreator stub is just too
    # strict (wants **kwargs), hence the ignore.
    gym.register(
        id="factorion/FactorioEnv-v0",
        entry_point=FactorioEnv,  # ty: ignore[invalid-argument-type]
    )

    print(f"Seeding with {args.seed=}")
    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else ("mps" if torch.backends.mps.is_available() and args.metal else "cpu"))
    if device.type == "mps":
        # metal doesn't like anything but f32
        torch.set_default_dtype(torch.float32)
    print(f"running on {device}")

    print(f"Setting up envs with {args}")
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.size, run_name, args.throughput_reward_scale, args.step_penalty) for i in range(args.num_envs)],
    )

    encoder_layers = layers_from_args(args)
    print(f"Creating agent with layers={encoder_layers}, {args.kernel_size=}, {args.tile_head_std=}, {args.dropout=} ")
    agent = AgentCNN(
        envs,
        layers=encoder_layers,
        kernel_size=args.kernel_size,
        tile_head_std=args.tile_head_std,
        dropout=args.dropout,
    )

    if args.start_from is not None:
        if args.track:
            _append_run_tags(run, f"start_from:{args.start_from}")
        print(f"Loading model weights from {args.start_from}")
        ckpt_path = _resolve_start_from(
            args.start_from, args.wandb_project_name, args.wandb_entity
        )
        # Load to CPU first; the SFT checkpoint may hold CUDA tensors (saved on
        # a GPU pod) and the agent is moved onto `device` just below. This keeps
        # --start-from working on CPU/MPS boxes, not only the GPU CI pod.
        agent.load_state_dict(torch.load(ckpt_path, map_location="cpu"))

    agent.to(device)

    # Split params into critic (the value head) vs actor (encoder + every
    # policy/eot head). Captured BEFORE torch.compile so the names are clean
    # (compile prepends "_orig_mod."); the param tensors themselves are the
    # same objects the optimiser and compiled module share, so freezing one
    # list freezes the live params. The critic head is the only part SFT
    # never trained, so the warm-up trains it in isolation against the
    # frozen SFT features before letting gradients touch the actor.
    critic_params = list(agent.critic_head.parameters())
    critic_param_ids = {id(p) for p in critic_params}
    actor_params = [p for p in agent.parameters() if id(p) not in critic_param_ids]

    print("Compiling agent with torch.compile()")
    # torch.compile returns an OptimizedModule that proxies attribute access to
    # the wrapped AgentCNN; cast back so the policy's methods stay typed.
    agent = cast(AgentCNN, torch.compile(agent))

    # Two param groups so the critic warm-up + LR annealing can address actor
    # and critic LRs independently; group[0]=actor keeps the existing
    # logging/annealing callsites (which read param_groups[0]) correct.
    optimizer = optim.Adam(
        [
            {"params": actor_params, "lr": args.learning_rate},
            {"params": critic_params, "lr": args.learning_rate},
        ],
        lr=args.learning_rate,
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )

    # Freeze the actor for the warm-up window. Freezing the encoder too (not
    # just the policy heads) is deliberate: the encoder feeds the value head,
    # so leaving it trainable would let the value loss reshape the features
    # the policy reads — i.e. the actor would not actually be frozen. With it
    # frozen, the critic learns a value readout of the fixed SFT features.
    def set_actor_requires_grad(flag: bool) -> None:
        for p in actor_params:
            p.requires_grad_(flag)

    if args.critic_warmup > 0:
        set_actor_requires_grad(False)
        print(f"Critic warm-up: actor frozen for the first {args.critic_warmup} iterations")

    print("Allocating storage space")
    # ALGO Logic: Storage setup
    obs_shape = envs.single_observation_space.shape
    assert obs_shape is not None, "vector env must expose a concrete observation shape"
    obs_SECWH = torch.zeros((args.num_steps, args.num_envs) + obs_shape, dtype=torch.float32, device=device)
    ACTION_SPACE_SHAPE = (7,)  # xy(2), entity, direction, item, misc, eot
    actions_SEA = torch.zeros((args.num_steps, args.num_envs) + ACTION_SPACE_SHAPE, dtype=torch.int64, device=device)
    logprobs_SE = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards_SE = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones_SE = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values_SE = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    next_obs_ECWH, _ = envs.reset(
        seed=args.seed,
        options={
            'num_missing_entities': float('inf'),
        }
    )
    next_obs_ECWH = torch.as_tensor(np.array(next_obs_ECWH), dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    final_thputs_100ma = sum(end_of_episode_thputs) / len(end_of_episode_thputs)
    unclipped_grad_norm = np.nan
    approx_kl = np.nan

    # Accumulate episode-level metrics during the rollout, then log means once
    # per iteration. Per-lesson keys (rollout/{LESSON}/*) are recorded only by
    # episodes of that lesson, so each averages over just its own episodes.
    _episode_metrics: dict[str, list[float]] = {}

    def _record_episode(metrics: dict[str, float]) -> None:
        for k, v in metrics.items():
            _episode_metrics.setdefault(k, []).append(v)

    def _flush_episode_means() -> dict[str, float]:
        if not _episode_metrics:
            return {}
        means = {k: sum(v) / len(v) for k, v in _episode_metrics.items()}
        _episode_metrics.clear()
        return means

    # Fixed held-out greedy-eval set (disjoint from training seeds), used to log
    # eval/* — directly comparable to the SFT baseline's val/thput[_eot].
    eval_seeds_to_kind = _build_eval_set(args) if args.eval_every > 0 else {}
    if eval_seeds_to_kind:
        print(f"Greedy eval: {len(eval_seeds_to_kind)} held-out factories, "
              f"every {args.eval_every} iters")

    print(f"Starting {args.num_iterations} iterations")
    pbar = tqdm.trange(1, args.num_iterations + 1)
    for iteration in pbar:
        # Critic warm-up: the actor stays frozen for the first
        # critic_warmup iterations (only the value head trains), then we
        # unfreeze once and run normal PPO from there.
        in_warmup = iteration <= args.critic_warmup
        if args.critic_warmup > 0 and iteration == args.critic_warmup + 1:
            set_actor_requires_grad(True)
            print(f"\nCritic warm-up complete; unfreezing actor at iteration {iteration}")

        # Annealing schedules run over the post-warm-up window, so the actor
        # gets its full LR/entropy schedule starting from the unfreeze point
        # rather than burning it while frozen. During warm-up the critic
        # trains at the un-annealed peak LR.
        anneal_total = max(1, args.num_iterations - args.critic_warmup)
        anneal_iter = max(0, iteration - args.critic_warmup)
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 if in_warmup else 1.0 - (anneal_iter - 1.0) / anneal_total
            lrnow = frac * args.learning_rate
            for group in optimizer.param_groups:
                group["lr"] = lrnow

        # Entropy coefficient annealing: linear from ent_coef_start to ent_coef_end
        ent_frac = 1.0 if in_warmup else 1.0 - (anneal_iter - 1.0) / anneal_total
        ent_coef = args.ent_coef_end + ent_frac * (args.ent_coef_start - args.ent_coef_end)

        # Per-iteration accumulators for the acting policy's distribution shape
        # (the policy/* metrics): summed over rollout steps, meaned at log time.
        _head_ent_sum = {h: 0.0 for h in ["tile", "entity", "direction", "item", "misc", "eot"]}
        _eot_prob_sum = 0.0
        rollout_start = time.time()

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            if (step+1) % 10 == 0 or step + 1 == args.num_steps:
                pbar.set_description(f"taking step {step+1: 4}/{args.num_steps}; gstep:{global_step: 6}; thput:{final_thputs_100ma:.3f}")
            obs_SECWH[step] = next_obs_ECWH
            dones_SE[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # TODO update for new action logic
                # D for dictionary
                action_ED, logprobs_E, _entropy_E, value_E = agent.get_action_and_value(next_obs_ECWH)
                values_SE[step] = value_E
                # Accumulate the acting policy's per-head entropy + eot prob
                # (stashed by get_action_and_value) for the policy/* metrics.
                for h, e in agent._last_head_entropy.items():
                    _head_ent_sum[h] += float(e)
                _eot_prob_sum += float(agent._last_eot_prob)

                x_B, y_B = action_ED["xy"].unbind(dim=1)
                ent_B = action_ED["entity"]
                dir_B = action_ED["direction"]
                item_B = action_ED["item"]
                misc_B = action_ED["misc"]
                eot_B = action_ED["eot"].long()  # Bernoulli 0/1, stored as int

                # Combine the actions together such that the environments are
                # grouped first and then each component of the action
                action_EA = torch.stack([x_B, y_B, ent_B, dir_B, item_B, misc_B, eot_B], dim=1)

            actions_SEA[step] = action_EA
            logprobs_SE[step] = logprobs_E

            # eot ("declare done") is part of the env action now: the env
            # terminates on it, so RecordEpisodeStatistics counts eot-ended
            # episodes and next_done picks them up via `terminations`.
            action_ED_numpy = {k: v.cpu().numpy() for k, v in action_ED.items()}
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_ECWH, reward, terminations, truncations, infos = envs.step(action_ED_numpy)
            next_done = np.logical_or(terminations, truncations)
            rewards_SE[step] = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device)

            # Reset done envs back to a fully-blank (build-from-empty) factory.
            done_indices = np.where(next_done)[0]
            for idx in done_indices:
                obs, _ = envs.envs[idx].reset(seed=args.seed + idx, options={
                    'num_missing_entities': float('inf'),
                })
                next_obs_ECWH[idx] = obs

            next_obs_ECWH = torch.as_tensor(np.array(next_obs_ECWH), dtype=torch.float32, device=device)
            next_done = torch.as_tensor(np.array(next_done), dtype=torch.float32, device=device)

            if "episode" in infos:
                # The "_episode" mask indicates which environments finished this step
                # We can use any of the masks (_r, _l, _t) as they should be the same for a finished env
                finished_envs_mask = infos["_episode"]

                # Iterate through the boolean mask
                for i in range(args.num_envs):
                    if not finished_envs_mask[i]:
                        continue
                    # This environment finished, extract its stats
                    episode_return = infos["episode"]["r"][i]
                    episode_len = infos["episode"]["l"][i]
                    end_of_episode_thput = infos["thput_normed"][i]
                    end_of_episode_thput_raw = infos["thput_raw"][i]
                    # eot_rate: ended by the EOT action (termination) vs hitting
                    # max_steps (truncation).
                    ended_by_eot = 1.0 if bool(terminations[i]) else 0.0
                    lesson = LessonKind(int(infos["kind"][i])).name

                    end_of_episode_thputs.append(end_of_episode_thput)

                    _record_episode(_rollout_episode_metrics(
                        lesson,
                        episode_return=episode_return,
                        episode_len=episode_len,
                        thput_normed=end_of_episode_thput,
                        thput_raw=end_of_episode_thput_raw,
                        ended_by_eot=ended_by_eot,
                        invalid_frac=infos['frac_invalid_actions'][i],
                        num_entities=infos['num_entities'][i],
                        min_entities_required=infos['min_entities_required'][i],
                        frac_reachable=infos["frac_reachable"][i],
                    ))

        rollout_seconds = time.time() - rollout_start

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs_ECWH).reshape(1, -1)
            advantages_SE = torch.zeros_like(rewards_SE).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_SE[t + 1]
                    nextvalues = values_SE[t + 1]
                delta = rewards_SE[t] + args.gamma * nextvalues * nextnonterminal - values_SE[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages_SE[t] = lastgaelam
            returns_SE = advantages_SE + values_SE

        obs_B = obs_SECWH.reshape((-1,) + obs_shape)
        logprobs_B = logprobs_SE.reshape(-1)
        # NOTE: maybe have to convert back to tuple of batches
        actions_B = actions_SEA.reshape((-1,) + ACTION_SPACE_SHAPE)
        advantages_B = advantages_SE.reshape(-1)
        returns_B = returns_SE.reshape(-1)
        values_B = values_SE.reshape(-1)

        # Optimizing the policy and value network
        update_start = time.time()
        idxs_B = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            pbar.set_description(f"optimiser epoch {epoch+1}/{args.update_epochs}; grad norm:{unclipped_grad_norm:5.2f}; kl:{approx_kl:5.3f}; thput:{final_thputs_100ma:.3f}")
            np.random.shuffle(idxs_B)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                idxs = idxs_B[start:end]

                _action_BA, newlogprobs_B, entropy_B, newvalue_B = agent.get_action_and_value(
                    obs_B[idxs],
                    actions_B.long()[idxs]
                )
                newlogprobs_B = newlogprobs_B.reshape(-1)
                logratio_B = newlogprobs_B - logprobs_B[idxs].reshape(-1)
                ratio_B = logratio_B.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio_B).mean()
                    approx_kl = ((ratio_B - 1) - logratio_B).mean()
                    clipfracs += [((ratio_B - 1.0).abs() > args.clip_coef).float().mean().item()]

                assert not torch.isnan(advantages_B).any(), f"Some advantages are NaN: {advantages_B=}"
                advantages_mB = advantages_B[idxs]
                if args.norm_adv:
                    advantages_mB = (advantages_mB - advantages_mB.mean()) / (advantages_mB.std() + 1e-8)

                # Policy loss
                pg_loss1 = -advantages_mB * ratio_B
                pg_loss2 = -advantages_mB * torch.clamp(ratio_B, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue_B.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - returns_B[idxs]) ** 2
                    v_clipped = values_B[idxs] + torch.clamp(
                        newvalue - values_B[idxs],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - returns_B[idxs]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - returns_B[idxs]) ** 2).mean()

                entropy_loss = entropy_B.mean()
                assert not torch.isnan(pg_loss), "pg_loss is NaN, probably a bug"
                assert not torch.isnan(v_loss), "v_loss is NaN, probably a bug"
                if in_warmup:
                    # Actor frozen: train the value head only. The policy- and
                    # entropy-gradient paths run through frozen params anyway,
                    # but dropping them keeps the warm-up's objective explicit.
                    loss = v_loss * args.vf_coef
                else:
                    loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad(set_to_none=True)
                assert not torch.isnan(loss), "Loss is NaN, probably a bug"
                loss.backward()
                unclipped_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = values_B.cpu().numpy(), returns_B.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        update_seconds = time.time() - update_start

        # ── Greedy held-out eval (every eval_every iters + the final one) ──
        eval_metrics: dict = {}
        eval_seconds = 0.0
        if eval_seeds_to_kind and (
            iteration % args.eval_every == 0 or iteration == args.num_iterations
        ):
            t_eval = time.time()
            eval_metrics = _run_greedy_eval(agent, args, eval_seeds_to_kind, device)
            eval_seconds = time.time() - t_eval
            eval_metrics["eval/seconds"] = eval_seconds

        # ── Per-iteration logging ──────────────────────────────────────
        n_steps = max(1, args.num_steps)
        iter_metrics = {
            "global_step": global_step,
            "losses/policy": pg_loss.item(),
            "losses/value": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/total": loss.item(),
            "losses/approx_kl": float(approx_kl),
            "losses/clipfrac": float(np.mean(clipfracs)),
            "losses/explained_variance": explained_var,
            # policy/* describe the ACTING policy's distribution (meaned over the
            # rollout steps), the RL analog of SFT's per-head metrics.
            "policy/entropy": sum(_head_ent_sum.values()) / n_steps,
            "policy/eot_prob": _eot_prob_sum / n_steps,
            "optim/lr": optimizer.param_groups[0]["lr"],
            "optim/critic_lr": optimizer.param_groups[1]["lr"],
            "optim/ent_coef": ent_coef,
            "optim/grad_norm": float(unclipped_grad_norm),
            "optim/critic_warmup": 1.0 if in_warmup else 0.0,
            "perf/sps": int(global_step / (time.time() - start_time)),
            "perf/rollout_seconds": rollout_seconds,
            "perf/update_seconds": update_seconds,
            "perf/eval_seconds": eval_seconds,
        }
        for h, s in _head_ent_sum.items():
            iter_metrics[f"policy/entropy_{h}"] = s / n_steps
        iter_metrics.update(eval_metrics)

        # Flush rollout episode means (empty if no episodes ended this iter)
        iter_metrics.update(_flush_episode_means())

        # Moving-average rollout throughput drives the pbar + final summary.
        if len(end_of_episode_thputs) > 0:
            final_thputs_100ma = sum(end_of_episode_thputs) / len(end_of_episode_thputs)

        # Single wandb.log() call per iteration
        if args.track:
            wandb.log(iter_metrics, step=global_step)

        if args.capture_video and ((iteration-1) % 50 == 0 or iteration + 1 == args.num_iterations):
            print(f"Recording agent progress at {iteration}")
            num_render_envs = 5
            render_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, False, args.size, run_name, args.throughput_reward_scale, args.step_penalty) for i in range(num_render_envs)])
            next_obs_ECWH_render, _ = render_envs.reset(seed=args.seed, options={'num_missing_entities': float('inf')})

            temp_dirs = [tempfile.mkdtemp() for _ in range(num_render_envs)]
            frame_counts = [0] * num_render_envs

            try:
                # Save initial frames
                for env_idx, img in enumerate(render_envs.render() or []):
                    image = Image.fromarray(img, mode="RGB")
                    frame_path = os.path.join(temp_dirs[env_idx], f'frame_{frame_counts[env_idx]:06d}.png')
                    image.save(frame_path, format="png", optimize=True)
                    frame_counts[env_idx] += 1

                # Run simulation and save frames
                for i in range(1, 10):
                    with torch.no_grad():
                        action_ED_render, _logprobs_E, _entropy_E, _value_E = agent.get_action_and_value(torch.Tensor(next_obs_ECWH_render).to(device))
                        action_ED_numpy = {k: v.cpu().numpy() for k, v in action_ED_render.items()}
                        next_obs_ECWH_render, _reward, terminations_render, truncations_render, _infos = render_envs.step(action_ED_numpy)

                    for env_idx, img in enumerate(render_envs.render() or []):
                        image = Image.fromarray(img, mode="RGB")
                        frame_path = os.path.join(temp_dirs[env_idx], f'frame_{frame_counts[env_idx]:06d}.png')
                        image.save(frame_path, format="png", optimize=True)
                        frame_counts[env_idx] += 1

                # Create videos for each environment
                iso8601 = datetime.now().replace(microsecond=0).isoformat(sep='T').replace(":", "-")
                Path('videos').mkdir(parents=True, exist_ok=True)

                for env_idx in range(num_render_envs):
                    output_path = f'videos/world_inits/{iso8601}_size{args.size}_blank_iter{iteration:06}_env{env_idx}.mp4'

                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-y',
                        '-framerate', '2',
                        '-i', os.path.join(temp_dirs[env_idx], 'frame_%06d.png'),
                        '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p',
                        '-crf', '23',
                        output_path
                    ]

                    try:
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                        print(f"Video saved: {output_path}")
                    except FileNotFoundError:
                        print(f"ffmpeg not found, skipping video for env {env_idx}")

            finally:
                for temp_dir in temp_dirs:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                render_envs.close()
    final_thput = 0 if len(end_of_episode_thputs) == 0 else sum(end_of_episode_thputs) / len(end_of_episode_thputs)
    def format_duration(seconds: float) -> str:
        total_seconds = int(round(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    runtime = time.time() - start_time
    if args.track:
        _append_run_tags(run, f"thput:{final_thput*100:.0f}", f"duration:{format_duration(runtime)}")
    envs.close()
    if runtime > 60 * 5: # 5 minutes
        run_name_dir_safe = run_name.replace('/', '-').replace(':', '-').replace(' ', '_')
        agent_name = f"agent-{run_name_dir_safe}"
        print(f"Saving model to artifacts/{agent_name}.pt")
        os.makedirs("artifacts", exist_ok=True)
        torch.save(agent.state_dict(), f"artifacts/{agent_name}.pt")
        if args.track:
            artifact = wandb.Artifact(name=agent_name, type="model")
            artifact.add_file(f"artifacts/{agent_name}.pt")
            wandb.log_artifact(artifact)
    else:
        print(f'Not saving because: {time.time() - start_time:.2f} <= {60 * 5}')

    # Write summary JSON (used by CI to post results to PR)
    summary = {
        "global_step": global_step,
        "total_timesteps": args.total_timesteps,
        "moving_avg_throughput": round(final_thput, 4),
        "runtime_seconds": round(runtime, 1),
        "runtime_human": format_duration(runtime),
        "sps": int(global_step / runtime) if runtime > 0 else 0,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "grid_size": args.size,
        "wandb_url": run.url if args.track and run else None,
    }
    summary_path = args.summary_path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "summary.json")
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary written to {summary_path}")


