# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
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

    # Algorithm specific arguments
    env_id: str = "factorion/FactorioEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    num_steps: int = 8
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.025
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    coeff_throughput: float = 0.90
    """coefficient of the throughput when calculating reward"""
    coeff_frac_reachable: float = 0.05
    """coefficient of the fraction of unreachable nodes when calculating reward"""
    coeff_frac_hallucin: float = 0.05
    """coefficient of the fraction of tiles that had to be changed after normalisation"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

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
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
            # if 'factorio' in env_id:
            #     env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

mapping = {
    # transport belt
    (2, 0): '↑',
    (2, 4): '→',
    (2, 8): '↓',
    (2, 12): '←',
    # sink
    (3, 0):  '📥',
    (3, 4):  '📥',
    (3, 8):  '📥',
    (3, 12): '📥',
    # source
    (4, 0):  '📤',
    (4, 4):  '📤',
    (4, 8):  '📤',
    (4, 12): '📤',
}

def pretty_print_tensor(tensor, entity_dir_map):
    assert isinstance(tensor, torch.Tensor), "Input must be a torch tensor"
    assert tensor.ndim == 3 and tensor.shape[0] == 3, "Tensor must have shape (3, W, H)"
    assert tensor.shape[1] == tensor.shape[2], f"Expected world to be square, but is of shape {tensor.shape}"
    # assert torch.is_integral(tensor), "Tensor must contain integers"

    _, W, H = tensor.shape
    entities = tensor[0]
    directions = tensor[2]

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
        width: int = 5,
        height: int = 5,
        channels: int = 3,
        # TODO(boyd): num_entities is actually num_directions, 4 + "none"
        num_entities: int = 5,
    ):

        # Import the functions from factorion
        outputs, functions = factorion.functions.run()
        self.fns = {}
        for func_name, func in functions.items():
            self.fns[func_name] = func

        self.width = width
        self.height = height
        self.channels = channels
        self.num_entities = num_entities
        # The size of the square grid
        # self.size = size

        self._world_CWH = torch.zeros((self.channels, self.width, self.height))

        # Define the agent and target location; randomly chosen in `reset` and
        # updated in `step`
        self._agent_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`-1}^2
        # self.observation_space = gym.spaces.Dict(
        #     {
        #         "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #         "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
        #     }
        # )
        self.action_space = gym.spaces.Box(
            low=0,
            high=self.num_entities,
            shape=(self.channels, self.width, self.height),
            dtype=np.int8,
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.num_entities,
            shape=(self.channels, self.width, self.height),
            dtype=np.int8,
        )

    def _get_obs(self):
        clamped = torch.clamp(self._world_CWH, 0, self.num_entities)
        return clamped.to(torch.int8).numpy()

    def _get_info(self):
        return {
            "distance": 0,
        }

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self._world_CWH = self.fns['get_new_world'](seed=None, n=self.width).permute(2, 0, 1)
        observation_WHC = self._get_obs()
        info = self._get_info()
        return observation_WHC, info

    def step(self, action_CWH):
        CHANNEL_DIRECTION = 2
        dir_CWH = action_CWH[CHANNEL_DIRECTION, :, :]
        mask = (dir_CWH > 0)
        dir_CWH[mask] = dir_CWH[mask] * 4 - 4
        dir_CWH[~mask] = -1
        #action_CWH = torch.tensor(action_CWH)
        action_WHC = torch.tensor(action_CWH).permute(1, 2, 0)
        world_WHC = self._world_CWH.permute(1, 2, 0)
        normalised_world_WHC = self.fns['normalise_world'](action_WHC, world_WHC)
        throughput, num_unreachable = self.fns['funge_throughput'](normalised_world_WHC, debug=False)
        throughput /= 15.0
        # Calculate a "reachable" fraction that penalises the model for leaving
        # entities disconnected from the graph (almost certainly useless)
        frac_reachable = 1.0 - float(num_unreachable) / (self.width * self.height)
        # Calculate a hallucination fraction to encourage the model to not rely
        # on the normaliser
        frac_hallucin = (
            (
                normalised_world_WHC
                == torch.tensor(action_CWH).permute(1, 2, 0)
            ).sum()
            / torch.tensor(action_CWH).numel()
        ).item()

        reward = (
            throughput * args.coeff_throughput
            + frac_reachable * args.coeff_frac_reachable
            + frac_hallucin * args.coeff_frac_hallucin
        )
        reward /= args.coeff_throughput + args.coeff_frac_hallucin + args.coeff_frac_reachable

        if np.random.rand() > 0.999:
            normalised_world_CWH = normalised_world_WHC.permute(2, 0, 1)
            print(pretty_print_tensor(normalised_world_CWH, mapping))
            print(f"{args.coeff_throughput=} {args.coeff_frac_hallucin=} {args.coeff_frac_reachable=}")
            print(f'!!! {throughput}i/s, {num_unreachable} unreachable, {reward:.6f} reward!!!')
            print('--------------')

        # Terminate after every step
        terminated = True
        # never truncate
        truncated = False

        # Map the action (element of {0,1,2,3}) to the direction we walk in
        # direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid bounds
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )

        # An environment is completed if and only if the agent has reached the target
        # terminated = np.array_equal(self._agent_location, self._target_location)
        # truncated = False
        # reward = 1 if terminated else 0  # the agent is only reached at the end of the episode
        observation = self._get_obs()
        info = self._get_info()
        info.update({
            'throughput': throughput,
            'frac_reachable': frac_reachable,
            'frac_hallucin': frac_hallucin,
        })

        # (observation, reward, done, truncated, info)
        return observation, reward, terminated, truncated, info

class AgentCNN(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.num_envs = len(envs.envs)
        self.width = envs.envs[0].width
        self.height = envs.envs[0].height
        self.channels = envs.envs[0].channels
        self.num_entities = envs.envs[0].num_entities
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(self.channels, 32, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, self.channels * self.num_entities, kernel_size=1)),
            nn.ReLU(),
        )

        self.critic = layer_init(nn.Conv2d(
            self.channels * self.num_entities,
            1,
            kernel_size=1
        ), std=1)

        self.actor = layer_init(nn.Conv2d(
            self.channels * self.num_entities,
            self.channels * self.num_entities,
            kernel_size=1
        ), std=0.01)

    def get_value(self, x_BCWH):
        assert len(x_BCWH.shape) == 4, f'Expected 4 dimensions, got {x.shape}'
        hidden_BCWH = self.network(x_BCWH)
        value_B1WH = self.critic(hidden_BCWH)
        value_BWH = value_B1WH.squeeze(1)
        value_B = value_BWH.mean(dim=(-1, -2))
        return value_B


    def get_action_and_value(self, x_BCWH, action_BCWH=None):
        B, C, W, H = x_BCWH.shape
        hidden_BCWH = self.network(x_BCWH)
        logits_BCWH = self.actor(hidden_BCWH)

        logits_BCDWH = logits_BCWH.view(B, C, self.num_entities, W, H)
        logits_BCWHD = logits_BCDWH.permute(0, 1, 3, 4, 2)

        probs = Categorical(logits=logits_BCWHD)
        if action_BCWH is None:
            action_BCWH = probs.sample()

        value_B = self.get_value(x_BCWH)
        logprobs_BCWH = probs.log_prob(action_BCWH)
        return action_BCWH, logprobs_BCWH, probs.entropy(), value_B

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    # get the default size from the FactorioEnv
    args.size = FactorioEnv().width
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
    # assert isinstance(envs.single_action_space, gym.spaces.Discrete), f"only discrete action space is supported, not {envs.single_action_space}"

    agent = AgentCNN(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs_SECWH = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions_SECWH = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs_SECWH = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values_SE = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs_ECWH, _ = envs.reset(seed=args.seed)
    next_obs_ECWH = torch.Tensor(next_obs_ECWH).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in tqdm.trange(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs_SECWH[step] = next_obs_ECWH # !dimensions!
            dones_SE[step] = next_done

            # ALGO LOGIC: action logic
            # TODO the shapes here are wrong becuase we're using a Box space
            # and not a simple scalar Discrete space, so nothing's expecting
            # tensors, everything expects scalars
            with torch.no_grad():
                action_ECWH, logprobs_ECWH, _, value_E = agent.get_action_and_value(next_obs_ECWH)
                values_SE[step] = value_E
            actions_SECWH[step] = action_ECWH
            logprobs_SECWH[step] = logprobs_ECWH

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs_ECWH, reward, terminations, truncations, infos = envs.step(action_ECWH.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards_SE[step] = torch.tensor(reward).to(device).view(-1)
            next_obs_ECWH, next_done = torch.Tensor(next_obs_ECWH).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        if (global_step + 1) % 100 == 0:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")

                        episodic_returns.append(info["episode"]["r"])
                        avg_return = sum(episodic_returns) / len(episodic_returns)
                        writer.add_scalar("charts/episodic_return_ma", avg_return, global_step)

                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

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
                advantages_SE[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns_SE = advantages_SE + values_SE

        # flatten the batch
        obs_B = obs_SECWH.reshape((-1,) + envs.single_observation_space.shape)
        # CHECK: kinda just taking the mean in order to get one logprob for each
        # step+environment's action. This might not be mathematically valid
        logprobs_B = logprobs_SECWH.mean(dim=(-3, -2, -1)).reshape(-1)
        actions_B = actions_SECWH.reshape((-1,) + envs.single_action_space.shape)
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

                _action_ECWH, newlogprobs_ECWH, entropy, newvalue_E = agent.get_action_and_value(
                    obs_B[idxs],
                    actions_B.long()[idxs]
                )
                newlogprobs_B = newlogprobs_ECWH.mean(dim=(-3, -2, -1)).reshape(-1)
                logratio_B = newlogprobs_B - logprobs_B[idxs].reshape(-1)
                ratio_B = logratio_B.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio_B).mean()
                    approx_kl = ((ratio_B - 1) - logratio_B).mean()
                    clipfracs += [((ratio_B - 1.0).abs() > args.clip_coef).float().mean().item()]

                advantages_mB = advantages_B[idxs]
                if args.norm_adv:
                    advantages_mB = (advantages_mB - advantages_mB.mean()) / (advantages_mB.std() + 1e-8)
                # Policy loss
                pg_loss1 = -advantages_mB * ratio_B
                pg_loss2 = -advantages_mB * torch.clamp(ratio_B, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue_E.view(-1)
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

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = values_B.cpu().numpy(), returns_B.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        preds = (y_pred > 0.5).astype(int)
        truth = y_true.astype(int)

        tp = ((preds == 1) & (truth == 1)).sum() / len(preds)
        tn = ((preds == 0) & (truth == 0)).sum() / len(preds)
        fp = ((preds == 1) & (truth == 0)).sum() / len(preds)
        fn = ((preds == 0) & (truth == 1)).sum() / len(preds)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        writer.add_scalar("metrics/precision", precision, global_step)
        writer.add_scalar("metrics/recall", recall, global_step)
        writer.add_scalar("metrics/f1_score", f1, global_step)
        writer.add_scalar("metrics/true_positives", tp, global_step)
        writer.add_scalar("metrics/true_negatives", tn, global_step)
        writer.add_scalar("metrics/false_positives", fp, global_step)
        writer.add_scalar("metrics/false_negatives", fn, global_step)

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

        if (iteration + 1) % 100 == 0:
            avg_return = sum(episodic_returns) / len(episodic_returns)
            print(f"avg_return: {avg_return}\n")

    envs.close()
    writer.close()
    if len(episodic_returns) > 0: # and sum(episodic_returns) / len(episodic_returns) > 0.90:
        avg_return = float(sum(episodic_returns) / len(episodic_returns))
        # Save the model to a file
        run_name_dir_safe = run_name.replace('/', '-').replace(':', '-')
        agent_name = f"agent-{avg_return:.6f}-{run_name_dir_safe}"
        print(f"Saving model with average return of {avg_return:.8f} to artifacts/{agent_name}.pt")
        torch.save(agent.state_dict(), f"artifacts/{agent_name}.pt")
        artifact = wandb.Artifact(name=agent_name, type="model")
        artifact.add_file(f"artifacts/{agent_name}.pt")
        wandb.log_artifact(artifact)


