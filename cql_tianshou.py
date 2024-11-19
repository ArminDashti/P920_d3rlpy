import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass
import gymnasium as gym
import minari
import argparse
from tianshou.data import Batch, ReplayBuffer, Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import SACPolicy, CQLPolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger, BaseLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
from tianshou.utils.conversion import to_optional_float


def download(dataset_id='D4RL/door/expert-v2'):
    dataset = minari.load_dataset(dataset_id, True)
    observations, actions, rewards, terminations, truncations, next_observations = [], [], [], [], [], []
    for episode in dataset.iterate_episodes():
        episode_length = len(episode.observations)
        for i in range(min(episode_length, 200)):
            observations.append(episode.observations[i])
            actions.append(episode.actions[i])
            rewards.append(episode.rewards[i])
            terminations.append(episode.terminations[i])
            truncations.append(episode.truncations[i])
            next_obs = episode.observations[i + 1] if i < episode_length - 1 else np.zeros_like(episode.observations[i])
            next_observations.append(next_obs)
    
    return {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminations': np.array(terminations),
        'truncations': np.array(truncations),
        'next_observations': np.array(next_observations),
        'dones': np.logical_or(terminations, truncations).astype(int),
        'in_dist': np.ones(len(observations))
    }


minari_ds = download()
min_reward = np.min(minari_ds['rewards'])
max_reward = np.max(minari_ds['rewards'])
normalized_rewards = (minari_ds['rewards'] - min_reward) / (max_reward - min_reward)
minari_ds['rewards'] = normalized_rewards


dataset = Batch(
    obs=minari_ds['observations'],
    act=minari_ds['actions'],
    rew=minari_ds['rewards'],
    obs_next=minari_ds['next_observations'],
    terminated=minari_ds['terminations'],
    truncated=minari_ds['truncations'],
    done=minari_ds['dones'].astype(bool),
)

buffer = ReplayBuffer(size=len(dataset))
for i in range(len(minari_ds['observations'])):
    buffer.add(
        Batch(
            obs=minari_ds['observations'][i],
            act=minari_ds['actions'][i],
            rew=minari_ds['rewards'][i],
            obs_next=minari_ds['next_observations'][i],
            terminated=minari_ds['terminations'][i],
            truncated=minari_ds['truncations'][i],
            done=bool(minari_ds['dones'][i]),
        )
    )

#%%
class RadiusNoise(nn.Module):
    def __init__(self, state_dim: int):
        super(RadiusNoise, self).__init__()
        self.state_dim = state_dim
        self.radius_network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Softplus()
        )
        # self.constrain = nn.Parameter(torch.rand(256, 39))

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.radius_network(states)

    def sample(self, states):
        radii = self.forward(states)
        directions = torch.randn_like(states)
        directions_norm = torch.norm(directions, dim=1, keepdim=True).clamp(min=1e-8)
        unit_directions = (directions / directions_norm)
        uniform_samples = torch.rand(states.size(0), 1, device=states.device)
        # log_uniform = torch.log(uniform_samples).clamp(max=0)
        # random_distances = uniform_samples.pow(1.0 / self.state_dim)
        random_distances = torch.exp(torch.log(uniform_samples) / self.state_dim)
        scaled_distances = random_distances * radii
        sampled_points = states + unit_directions * scaled_distances
        return sampled_points

    def sample_with_separate_outputs(self, states: torch.Tensor) -> tuple:
        return states, self.sample(states, include_original=False)


@dataclass(kw_only=True)
class SACTrainingStats(TrainingStats):
    actor_loss: float
    critic1_loss: float
    critic2_loss: float
    alpha: float | None = None
    alpha_loss: float | None = None


class RadiusNoisePolicy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.radius_noise_model = RadiusNoise(state_dim=39)
        self.radius_noise_optim = Adam(self.radius_noise_model.parameters(), lr=3e-4)
        self.radius_noise_model.train()
    
    @staticmethod
    def _mse_optimizer(
        batch,
        critic,
        optimizer):
        """A simple wrapper script for updating critic network."""
        weight = getattr(batch, "weight", 1.0)
        current_q = critic(batch.obs, batch.act).flatten()
        target_q = batch.returns.flatten()
        td = current_q - target_q
        # critic_loss = F.mse_loss(current_q1, target_q)
        critic_loss = (td.pow(2) * weight).mean()
        optimizer.zero_grad()
        critic_loss.backward()
        optimizer.step()
        return td, critic_loss

    def learn(self, batch, *args, **kwargs):
        # critic 1&2
        td1, critic1_loss = self._mse_optimizer(batch, self.critic, self.critic_optim)
        td2, critic2_loss = self._mse_optimizer(batch, self.critic2, self.critic2_optim)
        batch.weight = (td1 + td2) / 2.0  # prio-buffer

        if torch.rand(1).item() < 0.25:
            batch.obs = torch.tensor(batch.obs, dtype=torch.float32)
            with torch.no_grad():
                noisy_states = self.radius_noise_model.sample(batch.obs)
            batch.obs = noisy_states

        # actor
        obs_result = self(batch)
        act = obs_result.act
        current_q1a = self.critic(batch.obs, act).flatten()
        current_q2a = self.critic2(batch.obs, act).flatten()
        actor_loss = (
            self.alpha * obs_result.log_prob.flatten() - torch.min(current_q1a, current_q2a)
        ).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        alpha_loss = None

        self.update_noise_net(batch)

        if self.is_auto_alpha:
            log_prob = obs_result.log_prob.detach() + self.target_entropy
            # please take a look at issue #258 if you'd like to change this line
            alpha_loss = -(self.log_alpha * log_prob).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.detach().exp()

        self.sync_weight()

        return SACTrainingStats(  # type: ignore[return-value]
            actor_loss=actor_loss.item(),
            critic1_loss=critic1_loss.item(),
            critic2_loss=critic2_loss.item(),
            alpha=to_optional_float(self.alpha),
            alpha_loss=to_optional_float(alpha_loss),
        )


    def update_noise_net(self, batch):
        # self.actor_optim.zero_grad()
        self.critic_optim.zero_grad()
        self.critic2_optim.zero_grad()
        self.radius_noise_optim.zero_grad()
        
        batch.obs = torch.tensor(batch.obs, dtype=torch.float32)
        noisy_states = self.radius_noise_model.sample(batch.obs)
        # actor_output = self.actor(noisy_states)
        critic1_output = self.critic(noisy_states, batch.act)
        target_q = batch.returns.flatten()
        td = critic1_output - target_q
        critic1_loss = (td.pow(2)).mean()

        current_q = self.critic2(noisy_states, batch.act)
        target_q = batch.returns.flatten()
        td = current_q - target_q
        critic2_loss = (td.pow(2)).mean()

        loss = critic1_loss + critic2_loss
        loss.backward()
        self.radius_noise_optim.step()

        return loss



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate for optimizers')
    parser.add_argument('--tau', type=float, default=0.005, help='Soft update factor for target network')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--alpha', type=float, default=0.2, help='Entropy regularization coefficient')
    parser.add_argument('--max_epoch', type=int, default=200, help='Maximum number of epochs')
    parser.add_argument('--step_per_epoch', type=int, default=1000, help='Number of steps per epoch')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--update_per_step', type=int, default=1, help='Number of updates per step')
    parser.add_argument('--episode_per_test', type=int, default=1, help='Number of episodes per test')
    parser.add_argument('--log_path', type=str, default='c:/users/armin/v/v1', help='Path for tensorboard logs')
    args = parser.parse_args()

    env = gym.make('AdroitHandDoor-v1', render_mode='rgb_array')
    state_shape = dataset.obs.shape[1:]
    action_shape = dataset.act.shape[1:]
    max_action = 1.0

    net = Net(state_shape, hidden_sizes=[256, 256], device='cpu')

    actor = ActorProb(net, action_shape, max_action=1, device='cpu').to('cpu')
    critic = Critic(
        Net(state_shape, action_shape, hidden_sizes=[256, 256], concat=True, device='cpu'),
        device='cpu'
    ).to('cpu')
    critic2 = Critic(
        Net(state_shape, action_shape, hidden_sizes=[256, 256], concat=True, device='cpu'), 
        device='cpu'
    ).to('cpu')

    actor_optim = Adam(actor.parameters(), lr=args.lr)
    critic_optim = Adam(critic.parameters(), lr=args.lr)
    critic2_optim = Adam(critic2.parameters(), lr=args.lr)

    policy = CQLPolicy(
        actor=actor,
        actor_optim=actor_optim,
        critic=critic,
        critic_optim=critic_optim,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        action_space=env.action_space,
    )

    def make_env():
        return gym.make('AdroitHandDoor-v1', render_mode='rgb_array')

    test_envs = DummyVectorEnv([make_env for _ in range(1)])

    test_collector = Collector(policy, test_envs)

    writer = SummaryWriter(args.log_path)
    logger = TensorboardLogger(writer)

    trainer = OfflineTrainer(
        policy=policy,
        buffer=buffer,
        test_collector=test_collector,
        max_epoch=args.max_epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        update_per_step=args.update_per_step,
        episode_per_test=args.episode_per_test,
        logger=logger
    )

    trainer.run()
