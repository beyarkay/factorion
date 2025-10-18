# TODO: also log throughput, frac hallucin, frac_reachable, to wandb
# TODO: convert to many steps, each predicting the placement of an item
# TODO: integrate with actual factorio
import os
import random
import time
from dataclasses import dataclass
from typing import Optional
from collections import deque
from datetime import datetime
from pathlib import Path

import tqdm
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import sys
sys.path.insert(1, '/Users/brk/projects/factorion') # NOTE: must be before import factorion
import factorion
from PIL import Image, ImageDraw, ImageFont

episodic_returns = deque(maxlen=100)
end_of_episode_thputs = deque(maxlen=100)
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
    metal: bool = False # TODO set to true and convert all floats to f32
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

    # Algorithm specific arguments
    env_id: str = "factorion/FactorioEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 2
    """the number of parallel game environments. More envs -> less likely to fit on GPU"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.93
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 128
    """the number of mini-batches. more minibatches -> smaller minibatch size -> more likely to fit on GPU"""
    update_epochs: int = 8
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.26
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""

    ent_coef: float = 0.00065
    """coefficient of the entropy"""
    vf_coef: float = 0.6
    """coefficient of the value function"""
    coeff_throughput: float = 0.97
    """coefficient of the throughput when calculating reward"""
    coeff_frac_reachable: float = 0.01
    """coefficient of the fraction of unreachable nodes when calculating reward"""
    coeff_frac_hallucin: float = 0.00
    """coefficient of the fraction of tiles that had to be changed after normalisation"""
    coeff_final_dir_reward: float = 0.01
    """coefficient of reward given to the final belt being correctly oriented"""
    coeff_material_cost: float = 0.01
    """coefficient of reward given to the cost of materials used to solve the problem"""

    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    adam_epsilon: float = 1e-5
    """The epsilon parameter for Adam"""
    chan1: int = 32
    """Number of channels in the first layer of the CNN encoder"""
    chan2: int = 32
    """Number of channels in the second layer of the CNN encoder"""
    chan3: int = 32
    """Number of channels in the third layer of the CNN encoder"""
    flat_dim: int = 128
    """Output size of the fully connected layer after the encoder"""
    size: int = 10
    """The width and height of the factory"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, size, run_name):
    def thunk():
        kwargs = {"render_mode": "rgb_array"} if capture_video else {}
        kwargs.update({'size': size, 'max_steps':(2*size)})
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
        render_mode: Optional[str] = None
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
        print(f"FactorioEnv({size=}, {max_steps=}, {render_mode=})")
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

        self.steps = 0

    def _get_obs(self):
        return self._world_CWH

    def _get_info(self):
        return {
            # 'num_missing_entities': self.num_missing_entities,
            'throughput': self._throughput,
            'frac_reachable': self._frac_reachable,
            'frac_hallucin': self._frac_hallucin,
            'final_dir_reward': self._final_dir_reward,
            'material_cost': self._material_cost,
            'reward': self._reward,
            'cum_reward': self._cum_reward,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        options = options if options is not None else {}
        self._cum_reward = 0
        self._seed = seed

        self.invalid_actions = 0
        self._throughput = 0
        self._frac_reachable = 0
        self._frac_hallucin = 0
        # self._num_missing_entities = 0
        self._final_dir_reward = 0
        self._material_cost = 0
        self._reward = 0
        self._terminated = False
        self._truncated = False
        self.max_entities = 2
        # print(f"Resetting env with options {options}")
        # self.num_missing_entities = float('inf') if options is None else options.get('num_missing_entities', float('inf'))
        self.actions = []
        self._world_CWH, min_entities_required = self.generate_lesson(
            size=self.size,
            kind=self.LessonKind.MOVE_ONE_ITEM,
            num_missing_entities=0, #self.num_missing_entities,
            seed=seed,
            # max_entities=self.max_entities,
        )
        image = Image.fromarray(self.render(), mode="RGB")
        iso8601 = datetime.now().replace(microsecond=0).isoformat(sep='T').replace(":", "-")
        w = self._world_CWH.shape[1]
        h = self._world_CWH.shape[2]
        # print(f"world {min_entities_required}: ")
        # print(get_pretty_format(self._world_CWH, mapping))
        image.save(f'videos/world_inits/{iso8601}_seed{seed}_{w}x{h}.png', format="png", optimize=True)

        self.min_entities_required = min_entities_required
        self._original_world_CWH = torch.clone(self._world_CWH)
        # self._world_CWH = self.get_new_world(seed, n=self.size, min_belts=list(range(0,  17))).permute(2, 0, 1).to(int)
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
        # direc_to_be_replaced = self._world_CWH[self.Channel.ENTITIES.value, x, y]

        self.actions.append(None)
        self.invalid_actions += 1
        invalid_reason = {
            'replaced_source_or_sink': False,
            'replace_empty_with_empty': False,
            'place_empty_w_direction': False,
            'place_empty_w_recipe': False,
            'place_asm_mach_wo_recipe': False,
            'placement_wo_direction': False,
            'ug_belt_wo_up_or_down': False,
            'placement_with_unneeded_misc': False,
            'too_wide': False,
            'too_tall': False,
        }

        # Check that the action is actually valid
        if entity_to_be_replaced in (len(self.entities)-1, len(self.entities)-2):
            # disallow the replacement of the source+sink
            invalid_reason['replaced_source_or_sink'] = True
            pass
        elif entity_id == self.str2ent('empty').value and entity_to_be_replaced == self.str2item('empty').value:
            # Model is trying to replace empty space with more empty space
            invalid_reason['replace_empty_with_empty'] = True
            pass
        elif entity_id == self.str2ent('empty').value and direc != self.Direction.NONE.value:
            # Model is trying to place empty space with a direction
            invalid_reason['place_empty_w_direction'] = True
            pass
        elif entity_id == self.str2ent('empty').value and item_id != self.str2item('empty').value:
            # Model is trying to place empty space with a recipe item
            invalid_reason['place_empty_w_recipe'] = True
            pass
        elif entity_id == self.str2ent('assembling_machine_1').value and item_id == self.str2item('empty'):
            # Model is trying to place an assembling machine without a recipe
            invalid_reason['place_asm_mach_wo_recipe'] = True
            pass
        elif entity_id not in (self.str2ent('empty').value, self.str2ent('assembling_machine_1').value) and direc == self.Direction.NONE.value:
            # Model is trying to put a thing without giving a direction
            invalid_reason['placement_wo_direction'] = True
            pass
        elif (misc == self.Misc.NONE.value) and (entity_id == self.str2ent('underground_belt').value):
            # model is trying to place an underground belt without giving a down/up
            invalid_reason['ug_belt_wo_up_or_down'] = True
            pass
        elif (misc != self.Misc.NONE.value) and (entity_id != self.str2ent('underground_belt').value):
            # model is trying to place a thing that doesn't need a Misc but
            # still giving it a Misc
            invalid_reason['placement_with_unneeded_misc'] = True
            pass
        elif x + self.entities[entity_id].width > self.size:
            # The thing is too wide to be placed here
            invalid_reason['too_wide'] = True
            pass
        elif y + self.entities[entity_id].height > self.size:
            # The thing is too tall to be placed here
            invalid_reason['too_tall'] = True
            pass
        else:
            self.invalid_actions -= 1
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

        # world_old_CWH = self._world_CWH.clone().detach()

        throughput, num_unreachable = self.funge_throughput(self._world_CWH.permute(1, 2, 0))
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

        reward_components = {
            'throughput': {
                'coeff': args.coeff_throughput,
                'value': throughput,
            },
            # 'frac_reachable': {
            #     'coeff': args.coeff_frac_reachable,
            #     'value': frac_reachable,
            # },
            # 'frac_hallucin': {
            #     'coeff': args.coeff_frac_hallucin,
            #     'value': frac_hallucin,
            # },
            # 'final_dir_reward': {
            #     'coeff': args.coeff_final_dir_reward,
            #     'value': final_dir_reward,
            # },
            # 'material_cost': {
            #     'coeff': args.coeff_material_cost,
            #     'value': material_cost,
            # },
            # 'finished_before_deadline': {
            #     'coeff': args.coeff_finished_before_deadline,
            #     'value': (self.max_steps - self.steps) if terminated else 0,
            # },
        }
        pre_reward = 0.0
        normalisation = 0.0
        for name, item in reward_components.items():
            pre_reward += item['coeff'] * item['value']
            normalisation += item['coeff']

        pre_reward /= normalisation

        # Terminate early when the agent connects source to sink
        terminated = False # TODO revide early-stopping, but for now we'll remove it `throughput == 1.0`
        # Halt the run if the agent runs out of steps
        truncated = self.steps >= self.max_steps
        # TODO remove this
        terminated = truncated

        if terminated:
            # If the agent solved before the end, give extra reward
            reward = pre_reward + (self.max_steps - self.steps)
        else:
            reward = pre_reward

        # if throughput == 1.0:
        #     print(f"\033[1;36mSUCCESS throughput={throughput} (reward={reward:.6f}, step {self.steps}/{self.max_steps} (min: {self.min_entities_required}))\033[0m\n" +
        #         f"\nActions:\n" +
        #         '\n'.join([
        #             f"  {i:0>2}. Placing {a['entity']: <20} at {a['xy']} facing {a['direction']}" for i, a in enumerate(self.actions) if a is not None
        #         ]) +
        #         f"\nAfter:\n" +
        #         get_pretty_format(self._world_CWH, mapping) +
        #         f"\n(pre_reward) {pre_reward:.6f} = (thput){throughput:.6f}*{args.coeff_throughput}" +
        #         f" + (reachable){frac_reachable:.6f}*{args.coeff_frac_reachable}" +
        #         f" + (halluc){frac_hallucin:.6f}*{args.coeff_frac_hallucin} " +
        #         f" + (final_direc){final_dir_reward:.6f}*{args.coeff_final_dir_reward} " +
        #         f" + (material_cost){material_cost:.6f}*{args.coeff_material_cost}\n" +
        #         f"completion bonus: {self.max_steps - self.steps}\n"
        #         '\n--------------\n'
        #     )
        # elif truncated:
        # if truncated:
        #     print(f"\033[1;31mTRUNCATED: throughput={throughput} (reward={reward:.6f}, step {self.steps}/{self.max_steps} (min: {self.min_entities_required}))\n" +
        #           # f"\nActions:\n" +
        #           # '\n'.join([
        #           #     f"  {i:0>2}. Placing {a['entity']: <20} at {a['xy']} facing {a['direction']}" for i, a in enumerate(self.actions) if a is not None
        #           # ]) +
        #           # f"\nAfter:\n" +
        #         get_pretty_format(self._world_CWH, mapping) +
        #         '\n--------------\033[0m\n'
        #     )

        self._throughput = throughput
        self._frac_reachable = frac_reachable
        self._frac_hallucin = frac_hallucin
        # self._num_missing_entities = self.num_missing_entities
        self._final_dir_reward = final_dir_reward
        self._material_cost = material_cost
        self._reward = reward
        self._terminated = terminated
        self._truncated = truncated

        observation = self._get_obs()
        info = self._get_info()
        if terminated:
            info.update({ 'steps_taken': self.steps })

        info.update({
            'throughput': throughput,
            'frac_reachable': frac_reachable,
            'frac_hallucin': frac_hallucin,
            # 'num_missing_entities': self.num_missing_entities,
            'final_dir_reward': final_dir_reward,
            'material_cost': material_cost,
            'completion_bonus': self.max_steps - self.steps,
            'min_entities_required': self.min_entities_required,
            'num_entities': num_entities,
            'frac_invalid_actions': self.invalid_actions / self.max_steps,
            'max_entities': self.max_entities,
            'invalid_reason': invalid_reason,
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
    def __init__(self, envs, chan1=32, chan2=64, chan3=64, flat_dim=256):
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

        # TODO the flat dim as the parameter actually does nothing
        flat_dim = chan3 * self.width * self.height

        # Project encoded state to value
        self.critic_head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(flat_dim, 1), std=1.0)
        )

        # Action heads: output logits for x, y, direction, entity_id, etc
        self.action_head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(flat_dim, flat_dim)),
            nn.ReLU()
        )

        self.x_head = layer_init(nn.Linear(flat_dim, self.width))
        self.y_head = layer_init(nn.Linear(flat_dim, self.height))
        self.ent_head = layer_init(nn.Linear(flat_dim, self.num_entities))
        self.dir_head = layer_init(nn.Linear(flat_dim, self.num_directions))
        self.time_for_get_value = None
        self.time_for_get_action_and_value = None

        # Bias the entity/direction heads towards predicting empty space
        with torch.no_grad():
            self.ent_head.bias.fill_(0.0)
            self.ent_head.bias.data[0] = 1.0
            self.dir_head.bias.fill_(0.0)
            self.dir_head.bias.data[0] = 1.0


        # self.item_head = layer_init(nn.Linear(flat_dim, self.num_items))
        # self.misc_head = layer_init(nn.Linear(flat_dim, self.num_misc))

    def get_value(self, x_BCWH):
        t0 = time.time()
        encoded = self.encoder(x_BCWH)
        value_B = self.critic_head(encoded).squeeze(-1)
        self.time_for_get_value = time.time() - t0
        return value_B

    def get_action_and_value(self, x_BCWH, action=None):
        t0 = time.time()
        # B = x_BCWH.shape[0]

        # Encode input
        encoded_BCWH = self.encoder(x_BCWH)
        value_B = self.get_value(x_BCWH)

        # Flatten for action head
        features_BF = self.action_head(encoded_BCWH)

        # Predict logits
        logits_x_BW = self.x_head(features_BF)
        logits_y_BH = self.y_head(features_BF)
        logits_e_BE = self.ent_head(features_BF)
        logits_d_BD = self.dir_head(features_BF)
        # logits_i_BD = self.item_head(features_BF)
        # logits_m_BD = self.misc_head(features_BF)

        # Build distributions
        dist_x = Categorical(logits=logits_x_BW)
        dist_y = Categorical(logits=logits_y_BH)
        dist_e = Categorical(logits=logits_e_BE)
        dist_d = Categorical(logits=logits_d_BD)
        # dist_i = Categorical(logits=logits_i_BD)
        # dist_m = Categorical(logits=logits_m_BD)

        # Sample or unpack provided actions
        if action is None:
            x_B = dist_x.sample()
            y_B = dist_y.sample()
            ent_B = dist_e.sample()
            dir_B = dist_d.sample()
            # item_B = dist_i.sample()
            # misc_B = dist_m.sample()
        else:
            # Surprisingly enough, action is a tensor here, not a dict
            x_B = action[:, 0]
            y_B = action[:, 1]
            ent_B = action[:, 2]
            dir_B = action[:, 3]
            # item_B = action[:, 4]
            # misc_B = action[:, 5]


        # Compute log probs and entropy
        logp_B = (
            dist_x.log_prob(x_B) +
            dist_y.log_prob(y_B) +
            dist_e.log_prob(ent_B) +
            dist_d.log_prob(dir_B) # +
            # dist_i.log_prob(item_B) +
            # dist_m.log_prob(misc_B)
        )
        entropy_B = (
            dist_x.entropy() +
            dist_y.entropy() +
            dist_e.entropy() +
            dist_d.entropy() # +
            # dist_i.entropy() +
            # dist_m.entropy()
        )

        # Final action format: tuple of tensors
        # action_out = (torch.stack([x_B, y_B], dim=1), ent_B, dir_B)
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
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    print(f"batch_size: {args.batch_size}, minibatch_size: {args.minibatch_size}, num_iterations: {args.num_iterations}")
    iso8601 = datetime.now().replace(microsecond=0).isoformat(sep='T')
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{iso8601}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    print("Setting up writer")
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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

    print(f"Setting up envs with {args}")
    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.size, run_name) for i in range(args.num_envs)],
    )

    print(f"Creating agent with {args.chan1=}, {args.chan2=}, {args.chan3=}, {args.flat_dim=} ")
    agent = AgentCNN(
        envs,
        chan1=args.chan1,
        chan2=args.chan2,
        chan3=args.chan3,
        flat_dim=args.flat_dim
    )

    if args.start_from is not None:
        print(f"Loading model weights from {args.start_from}")
        agent.load_state_dict(torch.load(args.start_from))

    agent.to(device)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)

    print("Allocating storage space")
    # ALGO Logic: Storage setup
    obs_SECWH = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    ACTION_SPACE_SHAPE = (6,)
    actions_SEA = torch.zeros((args.num_steps, args.num_envs) + ACTION_SPACE_SHAPE, dtype=int).to(device)
    logprobs_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_ECWH, _ = envs.reset(
        seed=args.seed,
        options={'max_entities': 2}
        # options={'max_entities': global_step // 2_000 + 1}
    )
    next_obs_ECWH = torch.Tensor(next_obs_ECWH).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    print("Starting iterations")
    pbar = tqdm.trange(1, args.num_iterations + 1)
    for iteration in pbar:
        # print(f"{iteration=}")
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_SECWH[step] = next_obs_ECWH
            dones_SE[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                # TODO update for new action logic
                # D for dictionary
                t0 = time.time()
                action_ED, logprobs_E, _entropy_E, value_E = agent.get_action_and_value(next_obs_ECWH)
                writer.add_scalar(
                    f"per_second/get_action_for_rollout_div_{len(next_obs_ECWH)}",
                    ((1.0/(time.time() - t0))/len(next_obs_ECWH)),
                     global_step
                )
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
            rewards_SE[step] = torch.tensor(reward).to(device).view(-1)

            # Add one to the environments that succeeded at getting some throughput
            # if 'num_missing_entities' not in infos:
            #     breakpoint()
            # new_missing_entities = infos['num_missing_entities'] + (reward >= args.coeff_throughput).astype(int)

            # # Compute num_missing_entities based on global progress, from 1 to width*height
            # for k, v in infos.items():
            #     if k.startswith("_") or k == 'episode':
            #         continue
            #     for i, value in enumerate(v):
            #         if value is None:
            #             continue
            #         try:
            #             writer.add_scalar(f"old/charts/info_{k}_{i:0>2}", value, global_step)
            #         except:
            #             breakpoint()

            # # Prepare reset options per env
            # options = [{'num_missing_entities': new_missing_entities[i]} if d else None for i, d in enumerate(next_done)]
            # options = [{'num_missing_entities': float('inf')} if d else None for i, d in enumerate(next_done)]

            # Reset only the done environments with updated num_missing_entities
            done_indices = np.where(next_done)[0]
            for idx in done_indices:
                # obs, _ = envs.envs[idx].reset(options=options[idx])
                obs, _ = envs.envs[idx].reset(options={'max_entities': 2})
                next_obs_ECWH[idx] = obs

            next_obs_ECWH = torch.Tensor(next_obs_ECWH).to(device)
            next_done = torch.Tensor(next_done).to(device)


            for reason, values in infos.get('invalid_reason', {}).items():
                if reason[0] == '_':
                    continue
                for value in values:
                    writer.add_scalar(f"per_episode_invalid_reasons/{reason}", value, global_step)

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

                    episodic_returns.append(episode_return)
                    avg_return = sum(episodic_returns) / len(episodic_returns)
                    writer.add_scalar("old/charts/episodic_return_ma", avg_return, global_step)

                    end_of_episode_thputs.append(end_of_episode_thput)
                    avg_throughput = sum(end_of_episode_thputs) / len(end_of_episode_thputs)
                    writer.add_scalar("moving_avg/throughput", avg_throughput, global_step)
                    writer.add_scalar("at_end_of_episode/throughput", end_of_episode_thput, global_step)
                    writer.add_scalar("at_end_of_episode/num_steps", episode_len, global_step)
                    writer.add_scalar("at_end_of_episode/frac_invalid_actions", infos['frac_invalid_actions'][i], global_step)
                    writer.add_scalar("at_end_of_episode/episode_reward", episode_return, global_step)
                    writer.add_scalar("old/charts/final_throughput_ma", avg_throughput, global_step)

                    # min_belts = infos['min_belts'][i]
                    # writer.add_scalar(f"min_belts/d{min_belts}_throughput", final_throughput, global_step)

                    # min_belts_thoughputs[min_belts].append(final_throughput)
                    # avg_min_belts_throughput = (
                    #     0
                    #     if not min_belts_thoughputs[min_belts]
                    #     else sum(min_belts_thoughputs[min_belts]) / len(min_belts_thoughputs[min_belts])
                    # )
                    # writer.add_scalar(f"min_belts/d{min_belts}_throughput_ma", avg_min_belts_throughput, global_step)

                    writer.add_scalar("old/charts/episodic_entity_efficiency",  infos['min_entities_required'][i] / infos['num_entities'][i], global_step)
                    writer.add_scalar("old/charts/episodic_completion_bonus", infos['completion_bonus'][i], global_step)
                    writer.add_scalar("old/charts/episodic_final_dir_reward", infos['final_dir_reward'][i], global_step)
                    writer.add_scalar("old/charts/episodic_frac_reachable", final_frac_reachable, global_step)
                    writer.add_scalar("old/charts/episodic_length", episode_len, global_step)
                    writer.add_scalar("old/charts/episodic_material_cost", infos['material_cost'][i], global_step)
                    writer.add_scalar("old/charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("old/charts/final_through", end_of_episode_thput, global_step)
                    writer.add_scalar("old/charts/episodic_max_entities", infos['max_entities'][i], global_step)

                    # writer.add_scalar("old/charts/episodic_frac_hallucin", final_frac_hallucin, global_step)

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

        # Optimizing the policy and value network
        idxs_B = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(idxs_B)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                idxs = idxs_B[start:end]

                t0 = time.time()
                _action_BA, newlogprobs_B, entropy_B, newvalue_B = agent.get_action_and_value(
                    obs_B[idxs],
                    actions_B.long()[idxs]
                )
                writer.add_scalar(
                    f"per_second/get_action_for_optim_div_{len(obs_B[idxs])}",
                    ((1.0/(time.time() - t0))/len(obs_B[idxs])),
                    global_step
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
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                t0 = time.time()
                optimizer.zero_grad()
                assert not torch.isnan(loss), "Loss is NaN, probably a bug"
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                writer.add_scalar("per_second/backward_and_step", 1.0/(time.time() - t0), global_step)

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = values_B.cpu().numpy(), returns_B.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("old/charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("old/charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if len(end_of_episode_thputs) > 0:
            final_thputs_100ma = sum(end_of_episode_thputs) / len(end_of_episode_thputs)
            pbar.set_description(f"gstep: {global_step}, thput: {final_thputs_100ma:.2f}")

    envs.close()
    writer.close()
    # if args.total_timesteps > 10_000:
    #     avg_throughput = 0 if len(final_throughputs) == 0 else float(sum(final_throughputs) / len(final_throughputs))
    #     # Save the model to a file
    #     run_name_dir_safe = run_name.replace('/', '-').replace(':', '-')
    #     agent_name = f"agent-{avg_throughput:.6f}-{run_name_dir_safe}"
    #     print(f"Saving model with MA final throughput of {avg_throughput:.8f} to artifacts/{agent_name}.pt")
    #     os.makedirs("artifacts", exist_ok=True)
    #     torch.save(agent.state_dict(), f"artifacts/{agent_name}.pt")
    #     if args.track:
    #         artifact = wandb.Artifact(name=agent_name, type="model")
    #         artifact.add_file(f"artifacts/{agent_name}.pt")
    #         wandb.log_artifact(artifact)


