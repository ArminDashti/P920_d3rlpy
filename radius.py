import argparse
import torch
import d3rlpy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from d3rlpy.datasets import get_minari
from d3rlpy.algos import SACConfig, CQLConfig, BEARConfig
from d3rlpy.metrics.evaluators import EnvironmentEvaluator
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.algos.qlearning.torch import sac_impl
from d3rlpy.models.torch import get_parameter, build_squashed_gaussian_distribution
from d3rlpy.algos.qlearning.torch.ddpg_impl import DDPGBaseCriticLoss
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from d3rlpy.preprocessing import StandardRewardScaler, MinMaxRewardScaler
from d3rlpy.dataclass_utils import asdict_as_float
import sys
import inspect
device = "cuda" if torch.cuda.is_available() else "cpu"


def initialize_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class RadiusNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RadiusNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 39)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.normalize(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.sigmoid(x) * 0.09 + 0.01  # Scale output to range [0.01, 0.1]
        return x

    def normalize(self, x):
        return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-5)
    
    
def sample_in_radius_per_dim(point, radii):
    direction = torch.randn_like(point)
    direction = direction / direction.abs()
    distances = torch.rand_like(point) * radii
    sampled_point = point + direction * distances
    return sampled_point


def compute_critic_loss(self, batch, q_tpn):
    loss = self._q_func_forwarder.compute_error(
        observations=batch.observations,
        actions=batch.actions,
        rewards=batch.rewards,
        target=q_tpn,
        terminals=batch.terminals,
        gamma=self._gamma**batch.intervals,
    )  + batch.new_observations.mean()
    print(batch.new_observations.mean())
    return DDPGBaseCriticLoss(loss)


@dataclass
class ExtendedTorchMiniBatch:
    torch_batch: TorchMiniBatch
    new_observations: torch.Tensor = field(default=None)
    
    @property
    def observations(self):
        return self.torch_batch.observations

    @property
    def actions(self):
        return self.torch_batch.actions

    @property
    def rewards(self):
        return self.torch_batch.rewards

    @property
    def next_observations(self):
        return self.torch_batch.next_observations
    
    @property
    def terminals(self):
        return self.torch_batch.terminals

    @property
    def next_actions(self):
        return self.torch_batch.next_actions

    @property
    def returns_to_go(self):
        return self.torch_batch.returns_to_go

    @property
    def intervals(self):
        return self.torch_batch.intervals

    @property
    def device(self):
        return self.torch_batch.device
    
    def update_new_observations(self, new_observations):
        self.new_observations = new_observations
    

def refresh_torch_batch(torch_batch, new_observations):
    extended_batch = ExtendedTorchMiniBatch(torch_batch=torch_batch)
    extended_batch.update_new_observations(new_observations)
    return extended_batch


def update(self, batch):
    torch_batch = TorchMiniBatch.from_batch(
        batch=batch,
        gamma=self._config.gamma,
        compute_returns_to_go=self.need_returns_to_go,
        device=self._device,
        observation_scaler=self._config.observation_scaler,
        action_scaler=self._config.action_scaler,
        reward_scaler=self._config.reward_scaler)
    
    with torch.no_grad():
        rand_value = torch.rand(1).item()
        if rand_value < 0.5:
            radius_net_input = torch_batch.observations
            radii = self.radius_net(radius_net_input)
            new_observations = sample_in_radius_per_dim(torch_batch.observations, radii).to(torch_batch.device)
        else:
            new_observations = torch.ones_like(torch_batch.observations)

        extended_torch_batch = refresh_torch_batch(torch_batch, new_observations)
    
    loss_dict = self._impl.update(extended_torch_batch, self._grad_step)
    self._grad_step += 1
    return loss_dict


def update_critic(self, batch):
    self._modules.critic_optim.zero_grad()
    self.radius_optim.zero_grad()
    q_tpn = self.compute_target(batch)
    loss = self.compute_critic_loss(batch, q_tpn)
    loss.critic_loss.backward()
    self._modules.critic_optim.step()
    self.radius_optim.step()
    return asdict_as_float(loss)


def compute_custom_target(self, batch):
    with torch.no_grad():
        dist = build_squashed_gaussian_distribution(
            self._modules.policy(batch.next_observations)
        )
        action, log_prob = dist.sample_with_log_prob()
        entropy = get_parameter(self._modules.log_temp).exp() * log_prob
        target = self._targ_q_func_forwarder.compute_target(
            batch.next_observations,
            action,
            reduction="min",
        )
        return target - entropy
    

def customize_SAC(args, radius_optim):
    QLearningAlgoBase.update = update
    sac_impl.SACImpl.compute_target = compute_custom_target
    sac_impl.SACImpl.update_critic = update_critic
    sac_impl.SACImpl.compute_critic_loss = compute_critic_loss
    sac_impl.SACImpl.radius_optim = radius_optim


def fetch_dataset(dataset_name):
    return get_minari(dataset_name)


def create_algorithm_instance(args):
    if args.base_algo == 'SA':
        
        sac = SACConfig().create(device=device)
        radius_net = RadiusNetwork(input_size=args.state_dim, hidden_size=256).to(device)
        sac.radius_net = radius_net
        radius_optim = optim.Adam(radius_net.parameters(), lr=0.0001)
        customize_SAC(args, radius_optim)
        return sac
    
    elif args.base_algo == 'SAC':
        return SACConfig(reward_scaler=MinMaxRewardScaler(minimum=0.0, maximum=1.0)).create(device=device)
    
    elif args.base_algo == 'CQL':
        return CQLConfig().create(device=device)
    
    elif args.base_algo == 'BEAR':
        return BEARConfig().create(device=device)
    
    else:
        raise ValueError(f"Unsupported algorithm: {args.base_algo}")


def train_rl_model(gym_env, algorithm_instance, dataset, n_steps, video_folder):
    env = gym.make(gym_env, render_mode='rgb_array')
    env = RecordVideo(env, video_folder=video_folder)
    d3rlpy.envs.utility.seed_env(env, 42)
    algorithm_instance.build_with_env(env)
    algorithm_instance.fit(dataset, n_steps=n_steps, evaluators={"environment": EnvironmentEvaluator(env)})


def predict_agent_action(algorithm_instance, env):
    observation = env.reset()
    return algorithm_instance.predict(observation)


def run(args):
    input_size = 4
    hidden_size = 16
    
    dataset, gym_env = fetch_dataset(args.minari_env)
    algorithm_instance = create_algorithm_instance(args)
    train_rl_model(args.gym_env, algorithm_instance, dataset, args.n_steps, args.video_folder)
    actions = predict_agent_action(algorithm_instance, gym.make(args.gym_env))
    print(actions)


def main():
    parser = argparse.ArgumentParser(description='Train and predict with SAC using a safe action model.')
    parser.add_argument('--assets_dir', type=str, default=r'C:\Users\armin\P920_output')
    parser.add_argument('--minari_env', type=str, default='D4RL/door/expert-v2', help='Minari dataset environment name')
    parser.add_argument('--base_algo', type=str, default='SA', help='Base algorithm to use (SAC, CQL, BEAR)')
    parser.add_argument('--base_algo_bs', type=int, default=1024, help='Batch size for the base algorithm')
    parser.add_argument('--gym_env', type=str, default='AdroitHandDoor-v1', help='The name of the Gym environment to train on')
    parser.add_argument('--video_folder', type=str, default='c:/users/armin/v/sac', help='Folder to save recorded videos')
    parser.add_argument('--n_steps', type=int, default=1000000, help='Number of training steps')
    parser.add_argument('--state_dim', type=int, default=39, help='Number of training steps')

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()