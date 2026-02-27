import json
import os
import typing
import random
import time
from dataclasses import dataclass
from typing import Optional
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
import sys
sys.path.insert(1, '/Users/brk/projects/factorion') # NOTE: must be before import factorion
import factorion
from PIL import Image, ImageDraw, ImageFont
import factorion_rs

# episodic_returns = deque(maxlen=100)
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
    """Path of a model from which to start the training (instead of from scratch)"""
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
    coeff_throughput: float = 0.8785
    """coefficient of the throughput when calculating reward"""
    coeff_frac_reachable: float = 0.01
    """coefficient of the fraction of unreachable nodes when calculating reward"""
    coeff_frac_hallucin: float = 0.00
    """coefficient of the fraction of tiles that had to be changed after normalisation"""
    coeff_final_dir_reward: float = 0.01
    """coefficient of reward given to the final belt being correctly oriented"""
    coeff_material_cost: float = 0.01
    """coefficient of reward given to the cost of materials used to solve the problem"""
    coeff_validity: float = 0.01
    """coefficient of reward given to the action being valid"""
    coeff_shaping_location: float = 1.0
    """delta reward shaping: reward for placing entities at correct positions (solution-nonempty tiles only)"""
    coeff_shaping_entity: float = 1.0
    """delta reward shaping: reward for correct entity types (solution-nonempty tiles only)"""
    coeff_shaping_direction: float = 1.0
    """delta reward shaping: reward for correct directions (solution-nonempty tiles only)"""
    max_grad_norm: float = 1.979
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    adam_epsilon: float = 6.866e-06
    """The epsilon parameter for Adam"""
    chan1: int = 48
    """Number of channels in the first layer of the CNN encoder"""
    chan2: int = 48
    """Number of channels in the second layer of the CNN encoder"""
    chan3: int = 48
    """Number of channels in the third layer of the CNN encoder"""
    flat_dim: int = 128
    """Output size of the fully connected layer after the encoder"""
    tile_head_std: float = 0.06503
    """Initialization std for the tile selection conv head (smaller = more uniform initial exploration)"""
    size: int = 8
    """The width and height of the factory"""
    summary_path: Optional[str] = None
    """path to write summary JSON (default: summary.json next to ppo.py)"""
    wandb_group: Optional[str] = None
    """W&B run group name (groups parallel seeds together in the dashboard)"""
    tags: typing.Optional[typing.List[str]] = None
    """Tags to apply to the wandb run."""
    curriculum_geometric_p: float = 0.5
    """Geometric distribution parameter for curriculum sampling. Higher = more weight on frontier difficulty. 0 = uniform sampling (legacy behavior)."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (num_envs * num_steps)"""
    minibatch_size: int = 0
    """the mini-batch size (batch_size // num_minibatches)"""
    num_iterations: int = 0
    """the number of iterations (total_timesteps // batch_size)"""


def sample_difficulty(max_missing_entities: int, geometric_p: float) -> int:
    """Sample a difficulty level biased toward the frontier.

    When geometric_p > 0, uses a geometric distribution so ~50% of episodes
    are at the frontier difficulty, ~25% at frontier-1, etc.  When
    geometric_p == 0, falls back to uniform sampling (legacy behavior).

    Returns an int in [0, max_missing_entities].
    """
    if geometric_p <= 0 or max_missing_entities <= 0:
        return int(np.random.randint(0, max_missing_entities + 1))
    # Clamp to <1.0 so difficulty-0 scaffolding episodes are always possible.
    # At p=1.0, geometric always returns 1, making every episode frontier-only,
    # which eliminates the "don't break things" training signal (see PR #13).
    geometric_p = min(geometric_p, 0.95)
    # geometric(p) returns 1,2,3,... so subtract 1 to get 0,1,2,...
    # This gives P(frontier) = p, P(frontier-1) = p(1-p), etc.
    difficulty = max_missing_entities - (int(np.random.geometric(p=geometric_p)) - 1)
    return max(0, min(difficulty, max_missing_entities))


def make_env(env_id, idx, capture_video, size, run_name):
    def thunk():
        kwargs = {"render_mode": "rgb_array"} if capture_video else {}
        kwargs.update({'size': size, 'max_steps': 2*size, 'idx': idx})
        # kwargs.update({'size': size, 'max_steps': 6})
        env = gym.make(env_id, **kwargs)
        if capture_video:
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}/env_{idx}", episode_trigger=lambda e: (e+1) % 10 == 0)
            # env = gym.wrappers.RecordVideo(env, f"videos/{run_name}/env_{idx}", episode_trigger=lambda _: True)
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
    # assert torch.is_integral(tensor), "Tensor must contain integers"

    _, W, H = tensor.shape
    # entities = tensor[Channel.ENTITIES.value]
    # directions = tensor[Channel.DIRECTION.value]
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
        size: int = 5,
        max_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
        idx: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().__init__()
        # Setup the renderer if requested
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

        # Import the functions from factorion
        outputs, objs = factorion.datatypes.run()
        for obj_name, obj in objs.items():
            setattr(self, obj_name, obj)
        outputs, functions = factorion.functions.run()
        self.fns = {}
        for func_name, func in functions.items():
            self.fns[func_name] = func
            setattr(self, func_name, func)

        self._world_CWH = torch.zeros((len(self.Channel), self.size, self.size))

        self.max_id_in_tensor = max(len(self.items), len(self.entities), len(self.Direction))
        # Observation is the world, with a square grid of tiles and one channel
        # representing the entity ID, the other representing the direction
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.max_id_in_tensor,
            shape=(len(self.Channel), self.size, self.size),
            dtype=int,
        )


        self.action_space = gym.spaces.Dict({
            "xy": gym.spaces.Box(low=0, high=self.size, shape=(2,), dtype=int),
            "entity": gym.spaces.Discrete(len(self.entities)),
            "direction": gym.spaces.Discrete(len(self.Direction)),
            "item": gym.spaces.Discrete(len(self.items)),
            "misc": gym.spaces.Discrete(len(self.Misc))
        })
        self._reset_options = options if options is not None else {}

        self._num_missing_entities = 0
        self.steps = 0

    def _get_obs(self):
        return self._world_CWH

    def _get_info(self):
        return {
            'num_missing_entities': self._num_missing_entities,
            'throughput': self._throughput,
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
        orig_ent = self._solved_world_CWH[self.Channel.ENTITIES.value]
        curr_ent = self._world_CWH[self.Channel.ENTITIES.value]
        orig_dir = self._solved_world_CWH[self.Channel.DIRECTION.value]
        curr_dir = self._world_CWH[self.Channel.DIRECTION.value]

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
        self._throughput = 0
        self._frac_reachable = 0
        self._frac_hallucin = 0
        self._final_dir_reward = 0
        self._material_cost = 0
        self._reward = 0
        self._terminated = False
        self._truncated = False
        self.max_entities = 2
        self._num_missing_entities = self._reset_options['num_missing_entities'] # TODO also change max_steps in tandem

        self.actions = []
        # Generate the solved factory first (same seed → same layout),
        # then re-generate with missing entities for the actual episode.
        self._solved_world_CWH, _ = self.generate_lesson(
            size=self.size,
            kind=self.LessonKind.MOVE_ONE_ITEM,
            num_missing_entities=0,
            seed=self._seed,
        )
        self._world_CWH, min_entities_required = self.generate_lesson(
            size=self.size,
            kind=self.LessonKind.MOVE_ONE_ITEM,
            num_missing_entities=self._num_missing_entities,
            seed=self._seed,
        )

        self.min_entities_required = min_entities_required
        self._original_world_CWH = torch.clone(self._world_CWH)
        self._prev_match = self._compute_solution_match()
        self.steps = 0
        return self._world_CWH.cpu().numpy(), self._get_info()

    def step(self, action):
        # print(f"Stepping env with action {action}")
        x, y = action["xy"]
        entity_id = action["entity"]
        direc = action["direction"]
        item_id = action["item"]
        misc = action["misc"]


        # (x, y), entity_id, direc = action
        assert 0 <= x < self._world_CWH.shape[1], f"{x} isn't between 0 and {self._world_CWH.shape[1]}"
        assert 0 <= y < self._world_CWH.shape[2], f"{y} isn't between 0 and {self._world_CWH.shape[2]}"
        # account for two non-placeable prototypes: source and sink
        assert 0 <= entity_id < len(self.entities) - 2, f"{entity_id} isn't between 0 and {len(self.entities)-2}"
        assert 0 <= direc < len(self.Direction), f"{direc} isn't between 0 and {len(self.Direction)}"
        # TODO update for new action logic

        # Mutate the world with the agent's actions
        entity_to_be_replaced = self._world_CWH[self.Channel.ENTITIES.value, x, y]

        self.actions.append(None)
        action_is_invalid = False
        invalid_reason = {
            'replaced_source_or_sink': False,
            'place_asm_mach_wo_recipe': False,
            'placement_wo_direction': False,
            'direction_wo_entity': False,
            'ug_belt_wo_up_or_down': False,
            'placement_with_unneeded_misc': False,
            'too_wide': False,
            'too_tall': False,
        }

        # Check that the action is actually valid
        if entity_to_be_replaced in (len(self.entities)-1, len(self.entities)-2):
            # disallow the replacement of the source+sink
            invalid_reason['replaced_source_or_sink'] = True
            action_is_invalid = True
            pass
        elif entity_id == self.str2ent('assembling_machine_1').value and item_id == self.str2item('empty'):
            # Model is trying to place an assembling machine without a recipe
            invalid_reason['place_asm_mach_wo_recipe'] = True
            action_is_invalid = True
            pass
        elif entity_id not in (self.str2ent('empty').value, self.str2ent('assembling_machine_1').value) and direc == self.Direction.NONE.value:
            # Model is trying to put a thing without giving a direction
            invalid_reason['placement_wo_direction'] = True
            action_is_invalid = True
            pass
        elif entity_id == self.str2ent('empty').value and direc != self.Direction.NONE.value:
            # Model is trying to put a thing without giving a direction
            invalid_reason['direction_wo_entity'] = True
            action_is_invalid = True
            pass
        elif (misc == self.Misc.NONE.value) and (entity_id == self.str2ent('underground_belt').value):
            # model is trying to place an underground belt without giving a down/up
            invalid_reason['ug_belt_wo_up_or_down'] = True
            action_is_invalid = True
            pass
        elif (misc != self.Misc.NONE.value) and (entity_id != self.str2ent('underground_belt').value):
            # model is trying to place a thing that doesn't need a Misc but
            # still giving it a Misc
            invalid_reason['placement_with_unneeded_misc'] = True
            action_is_invalid = True
            pass
        elif x + self.entities[entity_id].width > self.size:
            # The thing is too wide to be placed here
            invalid_reason['too_wide'] = True
            action_is_invalid = True
            pass
        elif y + self.entities[entity_id].height > self.size:
            # The thing is too tall to be placed here
            invalid_reason['too_tall'] = True
            action_is_invalid = True
            pass
        else:
            action_is_invalid = False
            # If all the above guards didn't catch anything, allow the
            # placement of the entity
            self._world_CWH[self.Channel.ENTITIES.value, x, y] = entity_id
            self._world_CWH[self.Channel.DIRECTION.value, x, y] = direc
            self._world_CWH[self.Channel.ITEMS.value, x, y] = item_id
            self._world_CWH[self.Channel.MISC.value, x, y] = misc
            self.actions[-1] = {
                'entity': self.entities[entity_id].name,
                'xy': (x, y),
                'direction': self.Direction(direc),
                'item': self.items[item_id].name,
                'misc': self.Misc(misc),
            }

        self.invalid_actions += 1 if action_is_invalid else 0

        throughput, num_unreachable = factorion_rs.simulate_throughput(self._world_CWH.permute(1, 2, 0).to(torch.int64).numpy())
        # TODO don't always divide by 15
        throughput /= 15.0

        # Calculate a "reachable" fraction that penalises the model for leaving
        # entities disconnected from the graph (almost certainly useless)
        num_entities = self._world_CWH[self.Channel.ENTITIES.value].count_nonzero()
        # NOTE: weird bug with num_unreachable calculations, not planning on
        # fixing any time super soon. really the calculation should be to
        # calculate frac_reachable directly, not go via frac_unreachable
        frac_reachable = 0 if num_entities == 2 else max(0, 1.0 - (float(num_unreachable) / (num_entities - 2)))
        frac_hallucin = 0

        # Give some small reward for having the belt be the right direction
        sink_id = self.str2ent('bulk_inserter').value
        sink_locs = torch.where(self._world_CWH[self.Channel.ENTITIES.value] == sink_id)
        assert len(sink_locs[0]) == len(sink_locs[1]) == 1, f"Expected 1 bulk inserter, found {sink_locs} in world {self._world_CWH}"
        C, W, H = self._world_CWH.shape
        w_sink, h_sink = sink_locs[0][0], sink_locs[1][0]
        w_belt = torch.clamp(w_sink, 1, W-2)
        h_belt = torch.clamp(h_sink, 1, H-2)

        final_belt_dir = self._world_CWH[self.Channel.DIRECTION.value, w_belt, h_belt]
        sink_dir = self._world_CWH[self.Channel.DIRECTION.value, w_sink, h_sink]

        final_dir_reward = 1.0 if final_belt_dir == sink_dir else 0.0

        material_cost = (
            1.0 * (self._world_CWH[self.Channel.DIRECTION.value] == self.str2ent('transport_belt').value).sum()
            + 1.5 * (self._world_CWH[self.Channel.DIRECTION.value] == self.str2ent('underground_belt').value).sum()
            + 2.0 * (self._world_CWH[self.Channel.DIRECTION.value] == self.str2ent('assembling_machine_1').value).sum()
        )

        # ── Diagnostic tile-match metrics (for logging, NOT used in reward) ──
        orig_ent = self._solved_world_CWH[self.Channel.ENTITIES.value]
        curr_ent = self._world_CWH[self.Channel.ENTITIES.value]
        orig_dir = self._solved_world_CWH[self.Channel.DIRECTION.value]
        curr_dir = self._world_CWH[self.Channel.DIRECTION.value]

        solution_nonempty = (orig_ent != 0)
        current_nonempty = (curr_ent != 0)
        num_solution_nonempty = solution_nonempty.sum().item()
        if num_solution_nonempty > 0:
            tile_match_location = (solution_nonempty & current_nonempty).sum().item() / num_solution_nonempty
        else:
            tile_match_location = 1.0
        tile_match_entity = (curr_ent == orig_ent).float().mean().item()
        tile_match_direction = (curr_dir == orig_dir).float().mean().item()

        # ── Normalized weighted reward (throughput + validity only) ──
        reward_components = {
            'throughput': {
                'coeff': Args.coeff_throughput if 'args' not in locals() else args.coeff_throughput,
                'value': throughput,
            },
            'validity': {
                'coeff': Args.coeff_validity if 'args' not in locals() else args.coeff_validity,
                'value': 0 if action_is_invalid else 1,
            },
        }
        pre_reward = 0.0
        normalisation = 0.0
        for name, item in reward_components.items():
            pre_reward += item['coeff'] * item['value']
            normalisation += item['coeff']

        pre_reward /= normalisation

        # ── Delta-based reward shaping (PBRS, additive) ──
        # Computed over solution-nonempty tiles only for ~10x stronger signal.
        curr_match = self._compute_solution_match()
        loc_delta = curr_match[0] - self._prev_match[0]
        ent_delta = curr_match[1] - self._prev_match[1]
        dir_delta = curr_match[2] - self._prev_match[2]
        self._prev_match = curr_match

        coeff_loc = Args.coeff_shaping_location if 'args' not in locals() else args.coeff_shaping_location
        coeff_ent = Args.coeff_shaping_entity if 'args' not in locals() else args.coeff_shaping_entity
        coeff_dir = Args.coeff_shaping_direction if 'args' not in locals() else args.coeff_shaping_direction

        pre_reward += coeff_loc * loc_delta
        pre_reward += coeff_ent * ent_delta
        pre_reward += coeff_dir * dir_delta

        # Terminate early when the agent connects source to sink
        # Terminate early when the agent connects source to sink
        terminated = throughput >= 1.0
        # Halt the run if the agent runs out of steps (only if not already solved)
        truncated = (not terminated) and (self.steps > self.max_steps)

        if terminated:
            # If the agent solved before the end, give extra reward
            reward = pre_reward + (self.max_steps - self.steps)
        else:
            reward = pre_reward

        self._throughput = throughput
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
            'throughput': throughput,
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
            for ent_id, ent in self.entities.items():
                p = ICON_DIR / f"{ent.name}.png"
                if p.exists():
                    img = Image.open(p).convert("RGBA").resize(((CELL_PX // 10) * 8, (CELL_PX // 10) * 8), Image.BICUBIC)
                    self._render_cache["entity"][ent_id] = img

            # item (recipe) icons (resized to MINI_PX)
            for itm_id, itm in self.items.items():
                if itm_id == 0:  # 0 = empty → no icon
                    continue
                p = ICON_DIR / f"{itm.name}.png"
                if p.exists():
                    img = Image.open(p).convert("RGBA").resize((MINI_PX, MINI_PX), Image.BICUBIC)
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
            self._render_cache["arrow"][self.Direction.NORTH.value] = base
            self._render_cache["arrow"][self.Direction.EAST.value]  = base.rotate(-90, expand=True)
            self._render_cache["arrow"][self.Direction.SOUTH.value] = base.rotate(180, expand=True)
            self._render_cache["arrow"][self.Direction.WEST.value]  = base.rotate(90,  expand=True)

            # default font for fallback label
            self._render_cache["font"] = ImageFont.load_default()

        cache = self._render_cache
        ENT, DIR, ITEM = self.Channel.ENTITIES.value, self.Channel.DIRECTION.value, self.Channel.ITEMS.value

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
                    letter = self.entities[ent_id].name[0].upper()
                    font   = cache["font"]
                    canvas_w, canvas_h   = font.getbbox(letter)[2:]
                    draw.text(
                        (x0 + (CELL_PX - canvas_w) // 2, y0 + (CELL_PX - canvas_h) // 2),
                        letter,
                        fill=(0, 0, 0),
                        font=font,
                    )
                # Draw the xy-coords onto the cell
                text = f"{gx},{gy}"
                font   = cache["font"]
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

                dir_id = self.Direction(dir_layer[gx, gy]).value
                arrow  = cache["arrow"].get(dir_id)
                if arrow is not None:
                    canvas.paste(arrow, (x0 + 2, y0 + 2), arrow)

        info = self._get_info() or {}
        unreach = info.get("frac_reachable", "?")
        cum_reward = info.get("cum_reward", 0)
        throughput = info.get("throughput", "?")
        reward = info.get("reward", "?")

        lines = [
            f"Step {self.steps}/{self.max_steps}",
            f"SumReward: {cum_reward:.3f}  |  Reward: {reward:.3f}",
            f"thrput: {throughput}  |  Unreachable: {unreach:.3f}",
        ]
        if self.actions and self.actions[-1] is not None:
            lines.append(f"Placing {self.actions[-1]['entity']}")
            lines.append(f"    at {self.actions[-1]['xy']}")
            lines.append(f"    facing {self.actions[-1]['direction']}")

        font = cache["font"]
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


class AgentCNN(nn.Module):
    def __init__(self, envs, chan1=32, chan2=64, chan3=64, flat_dim=256, tile_head_std=0.01):
        super().__init__()
        base_env = envs.envs[0].unwrapped
        self.width = base_env.size
        self.height = base_env.size
        self.channels = len(base_env.Channel)
        # minus two for the source and the sink
        self.num_entities = 2 # len(base_env.entities) - 2 TODO for now, only allow empty or transport_belt
        self.num_directions = len(base_env.Direction)
        self.num_items = len(base_env.items)
        self.num_misc = len(base_env.Misc)
        self.chan3 = chan3

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(self.channels, chan1, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(chan1, chan2, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(chan2, chan3, kernel_size=3, padding=1)),
            nn.ReLU(),
        )
        num_params_encoder = sum(p.numel() for p in self.encoder.parameters())
        print(f"Encoder has {num_params_encoder} params")

        flat_dim = chan3 * self.width * self.height

        # Project encoded state to value
        self.critic_head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(flat_dim, 1), std=1.0)
        )

        # Tile selection: 1x1 conv producing one logit per spatial position
        self.tile_logits = layer_init(nn.Conv2d(chan3, 1, kernel_size=1), std=tile_head_std)

        # Per-tile entity/direction heads (conditioned on selected tile features)
        self.ent_head = layer_init(nn.Linear(chan3, self.num_entities))
        self.dir_head = layer_init(nn.Linear(chan3, self.num_directions))
        self.time_for_get_value = None
        self.time_for_get_action_and_value = None

        # Bias the entity/direction heads towards predicting empty space
        with torch.no_grad():
            self.ent_head.bias.fill_(0.0)
            self.ent_head.bias.data[0] = 1.0
            self.dir_head.bias.fill_(0.0)
            self.dir_head.bias.data[0] = 1.0

    def get_value(self, x_BCWH):
        t0 = time.time()
        encoded = self.encoder(x_BCWH)
        value_B = self.critic_head(encoded).squeeze(-1)
        self.time_for_get_value = time.time() - t0
        return value_B

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

        # --- Entity and direction heads (conditioned on tile features) ---
        logits_e_BE = self.ent_head(tile_features_BC)
        logits_d_BD = self.dir_head(tile_features_BC)
        dist_e = Categorical(logits=logits_e_BE)
        dist_d = Categorical(logits=logits_d_BD)

        if action is None:
            ent_B = dist_e.sample()
            dir_B = dist_d.sample()
        else:
            ent_B = action[:, 2]
            dir_B = action[:, 3]

        # --- Log probs and entropy ---
        logp_B = (
            dist_tile.log_prob(tile_idx_B) +
            dist_e.log_prob(ent_B) +
            dist_d.log_prob(dir_B)
        )
        entropy_B = (
            dist_tile.entropy() +
            dist_e.entropy() +
            dist_d.entropy()
        )

        # --- Output action dict (format unchanged) ---
        action_out = {
            "xy": torch.stack([x_B, y_B], dim=1),
            "direction": dir_B,
            "entity": ent_B,
            "item": torch.zeros_like(ent_B),
            "misc": torch.zeros_like(ent_B),
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
    # 16 * 256 * (50000/(16*256))
    num_gsteps = args.num_envs * args.num_steps * args.num_iterations
    print(f"batch_size: {args.batch_size}, minibatch_size: {args.minibatch_size}, num_iterations: {args.num_iterations}, num_gsteps: {num_gsteps}")

    iso8601 = datetime.now().replace(microsecond=0).isoformat(sep='T')
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{iso8601}"
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
        run.tags = run.tags + (
            f"batch_size:{args.batch_size}",
            f"minibatch_size:{args.minibatch_size}",
            f"num_iterations:{args.num_iterations}",
            f"seed:{args.seed}",
            f"size:{args.size}",
            f"timesteps:{args.total_timesteps//1000}K",
            f"chan1:{args.chan1}",
            f"chan2:{args.chan2}",
            f"chan3:{args.chan3}",
            f"flat_dim:{args.flat_dim}",
        )

        # Define metric axes and summary aggregation
        wandb.define_metric("*", step_metric="global_step")
        for m in ["throughput", "reward", "length", "invalid_frac",
                   "num_entities", "entity_efficiency", "frac_reachable"]:
            wandb.define_metric(f"episode/{m}", summary="last")
        for m in ["tile_location", "tile_entity", "tile_direction",
                   "delta_location", "delta_entity", "delta_direction"]:
            wandb.define_metric(f"shaping/{m}", summary="last")
        wandb.define_metric("curriculum/score", summary="max")
        wandb.define_metric("curriculum/level", summary="max")
        wandb.define_metric("curriculum/throughput_avg", summary="last")
        wandb.define_metric("curriculum/episode_difficulty", summary="last")
        for m in ["policy", "value", "entropy", "approx_kl", "clipfrac", "explained_var"]:
            wandb.define_metric(f"losses/{m}", summary="last")
        for m in ["lr", "ent_coef", "grad_norm"]:
            wandb.define_metric(f"optim/{m}", summary="last")
        wandb.define_metric("perf/sps", summary="last")
    print("Registering factorio Gym env")
    # Register the factorio env
    gym.register(
        id="factorion/FactorioEnv-v0",
        entry_point=FactorioEnv,
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
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.size, run_name) for i in range(args.num_envs)],
    )

    print(f"Creating agent with {args.chan1=}, {args.chan2=}, {args.chan3=}, {args.flat_dim=}, {args.tile_head_std=} ")
    agent = AgentCNN(
        envs,
        chan1=args.chan1,
        chan2=args.chan2,
        chan3=args.chan3,
        flat_dim=args.flat_dim,
        tile_head_std=args.tile_head_std,
    )

    if args.start_from is not None:
        if args.track:
            run.tags = run.tags + (f"start_from:{args.start_from}",)
        print(f"Loading model weights from {args.start_from}")
        agent.load_state_dict(torch.load(args.start_from))

    agent.to(device)

    print("Compiling agent with torch.compile()")
    agent = torch.compile(agent)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    print("Allocating storage space")
    # ALGO Logic: Storage setup
    obs_SECWH = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape, dtype=torch.float32, device=device)
    ACTION_SPACE_SHAPE = (6,)
    actions_SEA = torch.zeros((args.num_steps, args.num_envs) + ACTION_SPACE_SHAPE, dtype=int, device=device)
    logprobs_SE = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    rewards_SE = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    dones_SE = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)
    values_SE = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float32, device=device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    max_missing_entities = 1
    next_obs_ECWH, _ = envs.reset(
        seed=args.seed,
        options={
            'num_missing_entities': sample_difficulty(max_missing_entities, args.curriculum_geometric_p),
        }
    )
    next_obs_ECWH = torch.as_tensor(np.array(next_obs_ECWH), dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    final_thputs_100ma = sum(end_of_episode_thputs) / len(end_of_episode_thputs)
    unclipped_grad_norm = np.nan
    approx_kl = np.nan

    # Log initial eval metrics at step 0 (before any training)
    if args.track:
        wandb.log({
            "curriculum/level": max_missing_entities,
            "curriculum/score": (max_missing_entities - 1) + final_thputs_100ma,
            "curriculum/throughput_avg": final_thputs_100ma,
        }, step=0)

    # Accumulate episode-level metrics during rollout, log means once per iteration
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

    _next_eval_log_step = 256
    print(f"Starting {args.num_iterations} iterations")
    iteration_of_last_increase = 0
    pbar = tqdm.trange(1, args.num_iterations + 1)
    for iteration in pbar:
        # print(f"{iteration=}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Entropy coefficient annealing: linear from ent_coef_start to ent_coef_end
        ent_frac = 1.0 - (iteration - 1.0) / args.num_iterations
        ent_coef = args.ent_coef_end + ent_frac * (args.ent_coef_start - args.ent_coef_end)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            if (step+1) % 10 == 0 or step + 1 == args.num_steps:
                pbar.set_description(f"taking step {step+1: 4}/{args.num_steps}; gstep:{global_step: 6}; score:{(max_missing_entities - 1) + final_thputs_100ma:.2f} (lvl:{max_missing_entities} thput:{final_thputs_100ma:.2f})")
            obs_SECWH[step] = next_obs_ECWH
            dones_SE[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # TODO update for new action logic
                # D for dictionary
                action_ED, logprobs_E, _entropy_E, value_E = agent.get_action_and_value(next_obs_ECWH)
                values_SE[step] = value_E
                # Flatten action
                # (xy_B2, direc_B1, entities_B1) = action_ED

                # Unpack the action
                x_B, y_B = action_ED["xy"].unbind(dim=1)  # Unbind xy into x and y
                ent_B = action_ED["entity"]
                dir_B = action_ED["direction"]
                item_B = action_ED["item"]
                misc_B = action_ED["misc"]

                # Combine the actions together such that the environments are
                # grouped first and then each component of the action
                action_EA = torch.stack([x_B, y_B, ent_B, dir_B, item_B, misc_B], dim=1)

            actions_SEA[step] = action_EA
            logprobs_SE[step] = logprobs_E

            action_ED_numpy = {k: v.cpu().numpy() for k, v in action_ED.items()}
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_ECWH, reward, terminations, truncations, infos = envs.step(action_ED_numpy)
            next_done = np.logical_or(terminations, truncations)
            rewards_SE[step] = torch.as_tensor(np.array(reward), dtype=torch.float32, device=device)

            # Reset only the done environments with updated num_missing_entities
            done_indices = np.where(next_done)[0]
            for idx in done_indices:
                obs, _ = envs.envs[idx].reset(seed=args.seed + idx, options={
                    'num_missing_entities': sample_difficulty(max_missing_entities, args.curriculum_geometric_p)
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
                    end_of_episode_thput = infos["throughput"][i]
                    final_frac_reachable = infos["frac_reachable"][i]
                    # final_frac_hallucin = infos["frac_hallucin"][i]

                    # episodic_returns.append(episode_return)
                    # avg_return = sum(episodic_returns) / len(episodic_returns)
                    # writer.add_scalar("old/charts/episodic_return_ma", avg_return, global_step)

                    end_of_episode_thputs.append(end_of_episode_thput)

                    _record_episode({
                        "episode/throughput": float(end_of_episode_thput),
                        "episode/reward": float(episode_return),
                        "episode/length": float(episode_len),
                        "episode/invalid_frac": float(infos['frac_invalid_actions'][i]),
                        "episode/num_entities": float(infos['num_entities'][i]),
                        "episode/entity_efficiency": float(infos['min_entities_required'][i]) / float(infos['num_entities'][i]),
                        "episode/frac_reachable": float(final_frac_reachable),
                        "curriculum/episode_difficulty": float(infos['num_missing_entities'][i]),
                        "shaping/tile_location": float(infos['tile_match_location'][i]),
                        "shaping/tile_entity": float(infos['tile_match_entity'][i]),
                        "shaping/tile_direction": float(infos['tile_match_direction'][i]),
                        "shaping/delta_location": float(infos['shaping_location_delta'][i]),
                        "shaping/delta_entity": float(infos['shaping_entity_delta'][i]),
                        "shaping/delta_direction": float(infos['shaping_direction_delta'][i]),
                    })

            # Log eval metrics every 256 global steps during rollout
            if args.track and global_step >= _next_eval_log_step:
                final_thputs_100ma = sum(end_of_episode_thputs) / len(end_of_episode_thputs)
                eval_metrics = {
                    "curriculum/level": max_missing_entities,
                    "curriculum/score": (max_missing_entities - 1) + final_thputs_100ma,
                    "curriculum/throughput_avg": final_thputs_100ma,
                }
                eval_metrics.update(_flush_episode_means())
                wandb.log(eval_metrics, step=global_step)
                _next_eval_log_step = global_step + 256

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

        # flatten the batch
        obs_B = obs_SECWH.reshape((-1,) + envs.single_observation_space.shape)
        logprobs_B = logprobs_SE.reshape(-1)
        # NOTE: maybe have to convert back to tuple of batches
        actions_B = actions_SEA.reshape((-1,) + ACTION_SPACE_SHAPE)
        advantages_B = advantages_SE.reshape(-1)
        returns_B = returns_SE.reshape(-1)
        values_B = values_SE.reshape(-1)

        # print()

        # Optimizing the policy and value network
        idxs_B = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            pbar.set_description(f"optimiser epoch {epoch+1}/{args.update_epochs}; grad norm:{unclipped_grad_norm:5.2f}; kl:{approx_kl:5.3f}; score:{(max_missing_entities - 1) + final_thputs_100ma:.2f}")
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
                loss = pg_loss - ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad(set_to_none=True)
                assert not torch.isnan(loss), "Loss is NaN, probably a bug"
                loss.backward()
                unclipped_grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break
        # print()

        y_pred, y_true = values_B.cpu().numpy(), returns_B.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # ── Per-iteration logging ──────────────────────────────────────
        iter_metrics = {
            "global_step": global_step,
            "losses/policy": pg_loss.item(),
            "losses/value": v_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_var": explained_var,
            "optim/lr": optimizer.param_groups[0]["lr"],
            "optim/ent_coef": ent_coef,
            "optim/grad_norm": float(unclipped_grad_norm),
            "perf/sps": int(global_step / (time.time() - start_time)),
        }

        # Flush episode means (empty dict if no episodes ended this iteration)
        iter_metrics.update(_flush_episode_means())

        # Curriculum metrics — moving averages preserved exactly as before
        if len(end_of_episode_thputs) > 0:
            final_thputs_100ma = sum(end_of_episode_thputs) / len(end_of_episode_thputs)
            if len(end_of_episode_thputs) > int(moving_average_length * 0.9):
                if final_thputs_100ma > 0.95 and iteration - iteration_of_last_increase > 10:
                    iteration_of_last_increase = iteration
                    end_of_episode_thputs.clear()
                    for _ in range(moving_average_length):
                        end_of_episode_thputs.append(0)
                    max_missing_entities = min(max_missing_entities + 1, args.size*2)
                    print(f"\nNow working with {max_missing_entities=}")
            iter_metrics["curriculum/level"] = max_missing_entities
            iter_metrics["curriculum/score"] = (max_missing_entities - 1) + final_thputs_100ma
            iter_metrics["curriculum/throughput_avg"] = final_thputs_100ma

        # Single wandb.log() call per iteration
        if args.track:
            wandb.log(iter_metrics, step=global_step)

        if (iteration-1) % 50 == 0 or iteration + 1 == args.num_iterations:
            print(f"Recording agent progress at {iteration}")
            num_render_envs = 5
            render_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, False, args.size, run_name) for i in range(num_render_envs)])
            next_obs_ECWH_render, _ = render_envs.reset(seed=args.seed, options={'num_missing_entities': max_missing_entities})

            temp_dirs = [tempfile.mkdtemp() for _ in range(num_render_envs)]
            frame_counts = [0] * num_render_envs

            try:
                # Save initial frames
                for env_idx, img in enumerate(render_envs.render()):
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

                    for env_idx, img in enumerate(render_envs.render()):
                        image = Image.fromarray(img, mode="RGB")
                        frame_path = os.path.join(temp_dirs[env_idx], f'frame_{frame_counts[env_idx]:06d}.png')
                        image.save(frame_path, format="png", optimize=True)
                        frame_counts[env_idx] += 1

                # Create videos for each environment
                iso8601 = datetime.now().replace(microsecond=0).isoformat(sep='T').replace(":", "-")
                Path('videos').mkdir(parents=True, exist_ok=True)

                for env_idx in range(num_render_envs):
                    output_path = f'videos/world_inits/{iso8601}_size{args.size}_missing{max_missing_entities}_iter{iteration:06}_env{env_idx}.mp4'

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
    # curriculum_score: monotonically increasing with agent capability.
    # Advancing a curriculum level is always worth more than any throughput
    # gain within a level (since throughput is in [0, 1]).
    curriculum_score = (max_missing_entities - 1) + final_thput
    def format_duration(seconds: float) -> str:
        total_seconds = int(round(seconds))
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        return f"{hours:02d}h{minutes:02d}m{secs:02d}s"
    runtime = time.time() - start_time
    if args.track:
        run.tags = run.tags + (f"score:{curriculum_score:.2f}", f"thput:{final_thput*100:.0f}", f"duration:{format_duration(runtime)}")
    envs.close()
    if runtime > 60 * 5: # 5 minutes
        # avg_throughput = 0 if len(final_throughputs) == 0 else float(sum(final_throughputs) / len(final_throughputs))
        # Save the model to a file
        run_name_dir_safe = run_name.replace('/', '-').replace(':', '-')
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
        "curriculum_score": round(curriculum_score, 4),
        "max_missing_entities": max_missing_entities,
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


