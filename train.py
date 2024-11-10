import argparse
import torch
import d3rlpy

import numpy as np
from dataclasses import dataclass
from d3rlpy.datasets import get_minari, get_pendulum
from d3rlpy.algos import SACConfig, CQLConfig, BEARConfig
from d3rlpy.metrics.evaluators import EnvironmentEvaluator
from d3rlpy.algos.qlearning.base import QLearningAlgoBase
from d3rlpy.torch_utility import TorchMiniBatch
from d3rlpy.algos.qlearning.torch import sac_impl
from d3rlpy.models.torch import get_parameter, build_squashed_gaussian_distribution
from d3rlpy.algos.qlearning.torch.ddpg_impl import DDPGBaseCriticLoss 
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation, RecordVideo
from d3rlpy.preprocessing import StandardRewardScaler, MinMaxRewardScaler
reward_scaler = StandardRewardScaler(mean=0.0, std=1.0)
reward_scaler = MinMaxRewardScaler(minimum=0.0, maximum=1.0)
from d3rlpy.dataclass_utils import asdict_as_float
device = "cuda" if torch.cuda.is_available() else "cpu"
# d3rlpy.seed(42)

@dataclass
class ExtendedTorchMiniBatch:
    torch_batch: TorchMiniBatch
    noise: torch.Tensor

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

def initialize_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        
def compute_critic_loss(self, batch, q_tpn):
        loss = self._q_func_forwarder.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.intervals,
        ) + torch.norm(batch.noise)
        return DDPGBaseCriticLoss(loss)
    



def update_with_noise(self, batch):
    torch_batch = TorchMiniBatch.from_batch(
        batch=batch,
        gamma=self._config.gamma,
        compute_returns_to_go=self.need_returns_to_go,
        device=self._device,
        observation_scaler=self._config.observation_scaler,
        action_scaler=self._config.action_scaler,
        reward_scaler=self._config.reward_scaler,
    )
    with torch.no_grad():
        rand_value = torch.rand(1).item()
        noise, modified_actions = create_modified_actions(torch_batch, rand_value)
        torch_batch = refresh_torch_batch(torch_batch, modified_actions)
        extended_torch_batch = ExtendedTorchMiniBatch(torch_batch=torch_batch, noise=noise)
    
    loss_dict = self._impl.update(extended_torch_batch, self._grad_step)
    noise_norm = torch.norm(extended_torch_batch.noise)
    loss_dict['critic_loss']
    self._grad_step += 1
    return loss_dict


def update_critic(self, batch):
    self._modules.critic_optim.zero_grad()
    q_tpn = self.compute_target(batch)
    noise_norm = torch.norm(batch.noise)
    loss = self.compute_critic_loss(batch, q_tpn)
    # noise_norm = torch.norm(batch.noise) / torch.numel(batch.noise)
    
    loss.critic_loss.backward()
    self._modules.critic_optim.step()
    return asdict_as_float(loss)
    
    
def create_modified_actions(torch_batch, rand_value):
    if rand_value < 0.4:
        mean = torch.mean(torch_batch.actions, dim=0)
        std = torch.std(torch_batch.actions, dim=0)
        modified_actions = torch.normal(mean=mean, std=std).expand_as(torch_batch.actions).to(torch_batch.device)
        noise = torch.abs(modified_actions - torch_batch.actions)  # Ensure noise is positive
    elif rand_value < 0.5:
        shuffled_indices = torch.randperm(torch_batch.actions.size(0))
        modified_actions = torch_batch.actions[shuffled_indices]
        noise = (torch_batch.actions - modified_actions).abs()
    else:
        noise = torch.zeros_like(torch_batch.actions)
        modified_actions = torch_batch.actions
    return noise, modified_actions

def refresh_torch_batch(torch_batch, modified_actions):
    return TorchMiniBatch(
        observations=torch_batch.observations,
        actions=modified_actions,
        rewards=torch_batch.rewards,
        next_observations=torch_batch.next_observations,
        terminals=torch_batch.terminals,
        next_actions=torch_batch.next_actions,
        returns_to_go=torch_batch.returns_to_go,
        intervals=torch_batch.intervals,
        device=torch_batch.device
    )

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

def customize_SAC():
    
    QLearningAlgoBase.update = update_with_noise
    sac_impl.SACImpl.compute_target = compute_custom_target
    sac_impl.SACImpl.update_critic = update_critic
    # d3rlpy.models.torch.q_functions.ensemble_q_function = compute_ensemble_q_function_error
    sac_impl.SACImpl.compute_critic_loss = compute_critic_loss
    

def fetch_dataset(dataset_name):
    return get_minari(dataset_name)

def create_algorithm_instance(args):
    if args.base_algo == 'SA':
        customize_SAC()
        return SACConfig().create(device=device)
    elif args.base_algo == 'SAC':
        return SACConfig(reward_scaler=reward_scaler).create(device=device)
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

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    # initialize_seed()
    main()
