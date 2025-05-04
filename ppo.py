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
sys.path.insert(1, '/Users/brk/projects/factorion')
import factorion

episodic_returns = deque(maxlen=100)
final_throughputs = deque(maxlen=100)
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
    num_envs: int = 24
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.93
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
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
    coeff_throughput: float = 0.98
    """coefficient of the throughput when calculating reward"""
    coeff_frac_reachable: float = 0.01
    """coefficient of the fraction of unreachable nodes when calculating reward"""
    coeff_frac_hallucin: float = 0.00
    """coefficient of the fraction of tiles that had to be changed after normalisation"""
    coeff_final_dir_reward: float = 0.01
    """coefficient of reward given to the final belt being correctly oriented"""

    max_grad_norm: float = 1.0
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""
    adam_epsilon: float = 1e-5
    """The epsilon parameter for Adam"""
    chan1: int = 256
    """Number of channels in the first layer of the CNN encoder"""
    chan2: int = 256
    """Number of channels in the second layer of the CNN encoder"""
    chan3: int = 128
    """Number of channels in the third layer of the CNN encoder"""
    flat_dim: int = 128
    """Output size of the fully connected layer after the encoder"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    size: int = 0
    """The width and height of the factory (computed at runtime)"""


def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        # if capture_video and idx == 0:
        #     env = gym.make(env_id, render_mode="rgb_array")
        #     env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        # else:
        return gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))

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
    (2, 1):  '📥',
    (2, 2):  '📥',
    (2, 3):  '📥',
    (2, 4): '📥',
    # source
    (3, 1):  '📤',
    (3, 2):  '📤',
    (3, 3):  '📤',
    (3, 4): '📤',
}

def get_pretty_format(tensor, entity_dir_map):
    assert isinstance(tensor, torch.Tensor), f"Input must be a torch tensor but is {tensor}"
    assert tensor.ndim == 3 and tensor.shape[0] == 2, f"Tensor must have shape (2, W, H) but has shape {tensor.shape}"
    assert tensor.shape[1] == tensor.shape[2], f"Expected world to be square, but is of shape {tensor.shape}"
    # assert torch.is_integral(tensor), "Tensor must contain integers"

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
            if char in ('📤', '📥'):
                line.append(f"{char:^2}")
            else:
                line.append(f"{char:^3}")
        lines.append(" ".join(line))
    return "\n".join(lines)


class FactorioEnv(gym.Env):
    def __init__(
        self,
        size: int = 16,
        max_steps: Optional[int] = None,
    ):
        self.size = size
        if max_steps is None:
            max_steps = self.size * self.size
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

        self.max_ = max(len(self.prototypes), len(self.Direction))
        # Observation is the world, with a square grid of tiles and one channel
        # representing the entity ID, the other representing the direction
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.max_,
            shape=(len(self.Channel), self.size, self.size),
            dtype=int,
        )

        self.action_space = gym.spaces.Tuple((
            # x,y coordinates
            gym.spaces.Box(low=0, high=self.size, shape=(2,), dtype=int),
            # Direction: None, North, South, East, West
            gym.spaces.Discrete(len(self.Direction)),
            # Entity ID: None or belt
            gym.spaces.Discrete(len(self.prototypes))
        ))

        self.steps = 0

    def _get_obs(self):
        # return self._world_CWH.numpy()
        return self._world_CWH

    def _get_info(self):
        return { }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._world_CWH = self.get_new_world(seed, n=self.size, min_belts=list(range(0,  17))).permute(2, 0, 1).to(int)
        self.steps = 0
        return self._world_CWH.cpu().numpy(), self._get_info()


    def step(self, action):
        (x, y), entity_id, direc = action
        assert 0 <= x < self._world_CWH.shape[1], f"{x} isn't between 0 and {self._world_CWH.shape[1]}"
        assert 0 <= y < self._world_CWH.shape[2], f"{y} isn't between 0 and {self._world_CWH.shape[2]}"
        # account for two non-placeable prototypes: source and sink
        assert 0 <= entity_id < len(self.prototypes) - 2, f"{entity_id} isn't between 0 and {len(self.prototypes)-2}"
        assert 0 <= direc < len(self.Direction), f"{direc} isn't between 0 and {len(self.Direction)}"


        # Mutate the world with the agent's actions
        entity_to_be_replaced = self._world_CWH[self.Channel.ENTITIES.value, x, y]
        direc_to_be_replaced = self._world_CWH[self.Channel.ENTITIES.value, x, y]

        if entity_to_be_replaced in (len(self.prototypes)-1, len(self.prototypes)-2):
            # disallow the replacement of the source+sink
            pass
        elif entity_id == 1 and direc == 0:
            # Disallow the placement of belts without a direction
            pass
        elif entity_id != 0 and direc == 0:
            # Model is trying to put a thing without giving a direction
            pass
        else:
            self._world_CWH[self.Channel.ENTITIES.value, x, y] = entity_id
            self._world_CWH[self.Channel.DIRECTION.value, x, y] = direc
        # TODO need to change the world to be like how funge_throughput expects


        world_old_CWH = self._world_CWH.clone().detach()

        throughput, num_unreachable = self.funge_throughput(self._world_CWH.permute(1, 2, 0))
        throughput /= 15.0

        # Calculate a "reachable" fraction that penalises the model for leaving
        # entities disconnected from the graph (almost certainly useless)
        num_entities = self._world_CWH[self.Channel.ENTITIES.value].count_nonzero()
        frac_reachable = 0 if num_entities == 2 else (1.0 - (float(num_unreachable) / (num_entities - 2)))
        frac_hallucin = 0

        # Give some small reward for having the belt be the right direction
        sink_id = self.prototype_from_str('bulk_inserter').value
        sink_locs = torch.where(self._world_CWH[self.Channel.ENTITIES.value] == sink_id)
        assert len(sink_locs[0]) == len(sink_locs[1]) == 1, f"Expected 1 bulk inserter, found {sink_locs} in world {self._world_CWH}"
        C, W, H = self._world_CWH.shape
        w_sink, h_sink = sink_locs[0][0], sink_locs[1][0]
        w_belt = torch.clamp(w_sink, 1, W-2)
        h_belt = torch.clamp(h_sink, 1, H-2)

        final_belt_dir = self._world_CWH[self.Channel.DIRECTION.value, w_belt, h_belt]
        sink_dir = self._world_CWH[self.Channel.DIRECTION.value, w_sink, h_sink]

        final_dir_reward = 1.0 if final_belt_dir == sink_dir else 0.0

        # if throughput == 1.0:
        #     # print(self._world_CWH)
        #     print(get_pretty_format(self._world_CWH, mapping))
        #     print(f"{throughput=} {frac_reachable=}")
        reward = (
            throughput * args.coeff_throughput
            + frac_reachable * args.coeff_frac_reachable
            + frac_hallucin * args.coeff_frac_hallucin
            + final_dir_reward * args.coeff_final_dir_reward
        )
        reward /= (
            args.coeff_throughput
            + args.coeff_frac_hallucin
            + args.coeff_frac_reachable
            + args.coeff_final_dir_reward
        )

        # Terminate early when the agent connects source to sink
        terminated = throughput == 1.0
        # Halt the run if the agent runs out of steps
        truncated = self.steps >= self.max_steps

        if terminated:
            # If the agent solved before the end, give extra reward
            reward += (self.max_steps - self.steps)

        min_belts = self.get_min_belts(self._world_CWH)

        # if np.random.rand() > 0.999 or (np.random.rand() > 0.99 and throughput == 1.0):
        if throughput == 1.0:
            print(f"{self.steps=} {min_belts=}")
            print(get_pretty_format(self._world_CWH, mapping))
            print(f'!!! {throughput}i/s + {num_unreachable} unreachable + {final_dir_reward} direction = {reward:.3f} reward!!!')
            print('--------------')


        # Calculate the Manhattan distance between the source and the sink
        # stack_inserter_id = self.prototype_from_str("stack_inserter").value
        # bulk_inserter_id = self.prototype_from_str("bulk_inserter").value
        # coords1 = torch.where(self._world_CWH[self.Channel.ENTITIES.value] == bulk_inserter_id)
        # assert len(coords1[0]) == len(coords1[1]) == 1, f"Expected 1 bulk inserter, found {coords1} in world {self._world_CWH}"
        # w1, h1 = coords1[0][0], coords1[1][0]

        # coords2 = torch.where(self._world_CWH[self.Channel.ENTITIES.value] == stack_inserter_id)
        # assert len(coords2[0]) == len(coords2[1]) == 1, f"Expected 1 stack inserter, found {coords2} in world {self._world_CWH}"
        # w2, h2 = coords2[0][0], coords2[1][0]
        # min_belts = torch.abs(w1 - w2) + torch.abs(h1 - h2)

        observation = self._get_obs()
        info = self._get_info()
        info.update({
            'throughput': throughput,
            'frac_reachable': frac_reachable,
            'frac_hallucin': frac_hallucin,
            'min_belts': int(min_belts),
        })

        self.steps += 1
        assert not torch.isnan(torch.tensor(reward)).any(), f"Reward is nan or inf: {reward}"
        assert not torch.isinf(torch.tensor(reward)).any(), f"Reward is nan or inf: {reward}"

        return observation.numpy(), float(reward), terminated, truncated, info

class AgentCNN(nn.Module):
    def __init__(self, envs, chan1=32, chan2=64, chan3=64, flat_dim=256):
        super().__init__()
        base_env = envs.envs[0].unwrapped
        self.width = base_env.size
        self.height = base_env.size
        self.channels = len(base_env.Channel)
        # minus two for the source and the sink
        self.num_entities = len(base_env.prototypes) - 2
        self.num_directions = len(base_env.Direction)

        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(self.channels, chan1, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(chan1, chan2, kernel_size=3, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(chan2, chan3, kernel_size=3, padding=1)),
            nn.ReLU(),
        )

        flat_dim = chan3 * self.width * self.height

        # Project encoded state to value
        self.critic_head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(flat_dim, 1), std=1.0)
        )

        # Action heads: output logits for x, y, direction, entity_id
        self.action_head = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(flat_dim, flat_dim)),
            nn.ReLU()
        )

        self.x_head = layer_init(nn.Linear(flat_dim, self.width))
        self.y_head = layer_init(nn.Linear(flat_dim, self.height))
        self.ent_head = layer_init(nn.Linear(flat_dim, self.num_entities))
        self.dir_head = layer_init(nn.Linear(flat_dim, self.num_directions))

    def get_value(self, x_BCWH):
        encoded = self.encoder(x_BCWH)
        value_B = self.critic_head(encoded).squeeze(-1)
        return value_B

    def get_action_and_value(self, x_BCWH, action=None):
        B = x_BCWH.shape[0]

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

        # Build distributions
        dist_x = Categorical(logits=logits_x_BW)
        dist_y = Categorical(logits=logits_y_BH)
        dist_e = Categorical(logits=logits_e_BE)
        dist_d = Categorical(logits=logits_d_BD)

        # Sample or unpack provided actions
        if action is None:
            x_B = dist_x.sample()
            y_B = dist_y.sample()
            ent_B = dist_e.sample()
            dir_B = dist_d.sample()
        else:
            x_B = action[:, 0]
            y_B = action[:, 1]
            ent_B = action[:, 2]
            dir_B = action[:, 3]


        # Compute log probs and entropy
        logp_B = (
            dist_x.log_prob(x_B) +
            dist_y.log_prob(y_B) +
            dist_e.log_prob(ent_B) +
            dist_d.log_prob(dir_B)
        )
        entropy_B = (
            dist_x.entropy() +
            dist_y.entropy() +
            dist_e.entropy() +
            dist_d.entropy()
        )

        # Final action format: tuple of tensors
        action_out = (torch.stack([x_B, y_B], dim=1), ent_B, dir_B)
        return action_out, logp_B, entropy_B, value_B

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # get the default size from the FactorioEnv
    args.size = FactorioEnv().size
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
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # Register the factorio env
    gym.register(
        id="factorion/FactorioEnv-v0",
        entry_point=FactorioEnv,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    torch.use_deterministic_algorithms(args.torch_deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else ("mps" if torch.backends.mps.is_available() and args.metal else "cpu"))

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

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

    # ALGO Logic: Storage setup
    obs_SECWH = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    ACTION_SPACE_SHAPE = (4,)
    actions_SEA = torch.zeros((args.num_steps, args.num_envs) + ACTION_SPACE_SHAPE, dtype=int).to(device)
    logprobs_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_ECWH, _ = envs.reset(seed=args.seed)
    next_obs_ECWH = torch.Tensor(next_obs_ECWH).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    pbar = tqdm.trange(1, args.num_iterations + 1)
    for iteration in pbar:
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
                # T for a tuple
                action_ET, logprobs_E, _entropy_E, value_E = agent.get_action_and_value(next_obs_ECWH)
                values_SE[step] = value_E
                # Flatten action
                (xy_B2, direc_B1, entities_B1) = action_ET
                action_EA = torch.cat([xy_B2, direc_B1.unsqueeze(1), entities_B1.unsqueeze(1)], dim=1)

            actions_SEA[step] = action_EA
            logprobs_SE[step] = logprobs_E

            action_ET_np = (
                action_ET[0].cpu().numpy(),
                action_ET[1].cpu().numpy(),
                action_ET[2].cpu().numpy(),
            )
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_ECWH, reward, terminations, truncations, infos = envs.step(action_ET_np)
            next_done = np.logical_or(terminations, truncations)
            rewards_SE[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_ECWH = torch.Tensor(next_obs_ECWH).to(device)
            next_done = torch.Tensor(next_done).to(device)

            if "episode" in infos:
                # The "_episode" mask indicates which environments finished this step
                # We can use any of the masks (_r, _l, _t) as they should be the same for a finished env
                finished_envs_mask = infos["_episode"]

                # Iterate through the boolean mask
                for i in range(args.num_envs):
                    if not finished_envs_mask[i]: continue
                    # This environment finished, extract its stats
                    episode_return = infos["episode"]["r"][i]
                    episode_len = infos["episode"]["l"][i]
                    final_throughput = infos["throughput"][i]
                    final_frac_reachable = infos["frac_reachable"][i]
                    final_frac_hallucin = infos["frac_hallucin"][i]

                    episodic_returns.append(episode_return)
                    avg_return = sum(episodic_returns) / len(episodic_returns)
                    writer.add_scalar("charts/episodic_return_ma", avg_return, global_step)

                    final_throughputs.append(final_throughput)
                    avg_throughput = sum(final_throughputs) / len(final_throughputs)
                    writer.add_scalar("charts/final_throughput_ma", avg_throughput, global_step)

                    min_belts = infos['min_belts'][i]
                    writer.add_scalar(f"min_belts/d{min_belts}_throughput", final_throughput, global_step)

                    min_belts_thoughputs[min_belts].append(final_throughput)
                    avg_min_belts_throughput = (
                        0
                        if not min_belts_thoughputs[min_belts]
                        else sum(min_belts_thoughputs[min_belts]) / len(min_belts_thoughputs[min_belts])
                    )
                    writer.add_scalar(f"min_belts/d{min_belts}_throughput_ma", avg_min_belts_throughput, global_step)

                    writer.add_scalar("charts/episodic_return", episode_return, global_step)
                    writer.add_scalar("charts/episodic_length", episode_len, global_step)
                    writer.add_scalar("charts/episodic_throughput", final_throughput, global_step)
                    writer.add_scalar("charts/episodic_frac_reachable", final_frac_reachable, global_step)
                    writer.add_scalar("charts/episodic_frac_hallucin", final_frac_hallucin, global_step)

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
                assert not torch.isnan(pg_loss), f"pg_loss is NaN, probably a bug"
                assert not torch.isnan(v_loss), f"v_loss is NaN, probably a bug"
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                assert not torch.isnan(loss), f"Loss is NaN, probably a bug"
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = values_B.cpu().numpy(), returns_B.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        if len(final_throughputs) > 0:
            final_throughput_ma = sum(final_throughputs) / len(final_throughputs)
            pbar.set_description(f"thput: {final_throughput_ma:.2f}")

    envs.close()
    writer.close()
    if len(final_throughputs) > 0 and args.num_iterations > 10_000:
        avg_throughput = float(sum(final_throughputs) / len(final_throughputs))
        # Save the model to a file
        run_name_dir_safe = run_name.replace('/', '-').replace(':', '-')
        agent_name = f"agent-{avg_throughput:.6f}-{run_name_dir_safe}"
        print(f"Saving model with MA final throughput of {avg_throughput:.8f} to artifacts/{agent_name}.pt")
        os.makedirs("artifacts", exist_ok=True)
        torch.save(agent.state_dict(), f"artifacts/{agent_name}.pt")
        if args.track:
            artifact = wandb.Artifact(name=agent_name, type="model")
            artifact.add_file(f"artifacts/{agent_name}.pt")
            wandb.log_artifact(artifact)


