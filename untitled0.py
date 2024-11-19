from tianshou.data import Batch
from tianshou.policy import DQNPolicy
import numpy as np
import minari
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.data import ReplayBuffer, Batch
import gymnasium as gym


def download(dataset_id='D4RL/door/expert-v2'):
    dataset = minari.load_dataset(dataset_id, True)

    observations = []
    actions = []
    rewards = []
    terminations = []
    truncations = []
    next_observations = []
    
    for episode in dataset.iterate_episodes():
        episode_length = len(episode.observations)
        for i in range(0, min(episode_length, 200)):
            observations.append(episode.observations[i])
            actions.append(episode.actions[i])
            rewards.append(episode.rewards[i])
            terminations.append(episode.terminations[i]) 
            truncations.append(episode.truncations[i])
            if i < episode_length - 1:
                next_observations.append(episode.observations[i + 1])
            else:
                next_observations.append(np.zeros_like(episode.observations[i]))
            
    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    terminations = np.array(terminations)
    truncations = np.array(truncations)
    next_observations = np.array(next_observations)
    
    dones = np.logical_or(terminations, truncations).astype(int)
    dataset_dict = {
        'observations': observations,
        'actions': actions,
        'rewards': rewards,
        'terminations': terminations,
        'truncations': truncations,
        'dones': dones,
        'next_observations': next_observations,
        'in_dist': np.ones(len(observations))  # Initialize in_dist for existing data
    }

    return dataset_dict


minari_ds = download()
#%%
dataset = Batch(
    obs=minari_ds['observations'],        # list or array of observations
    act=minari_ds['actions'],        # list or array of actions
    rew=minari_ds['rewards'],        # list or array of rewards
    obs_next=minari_ds['next_observations'],   # list or array of next observations
    terminated=minari_ds['terminations'],
    truncated=minari_ds['truncations'],
    done=minari_ds['dones'].astype(bool),)


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
env = gym.make('AdroitHandDoor-v1', render_mode='rgb_array')



# Define environment parameters
state_shape = dataset.obs.shape[1:]
action_shape = dataset.act.shape[1:]
max_action = 1.0  # Adjust based on your action space

# Define networks
net = Net(state_shape, hidden_sizes=[128, 128], device='cpu')

    
actor = ActorProb(net, action_shape, max_action=1, device='cpu').to('cpu')
critic = Critic(
    Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device='cpu'),
    device='cpu'
).to('cpu')
critic2 = Critic(
    Net(state_shape, action_shape, hidden_sizes=[128, 128], concat=True, device='cpu'), 
    device='cpu').to('cpu')

# Define optimizers
actor_optim = Adam(actor.parameters(), lr=3e-4)
critic_optim = Adam(critic.parameters(), lr=3e-4)
critic2_optim = Adam(critic2.parameters(), lr=3e-4)

    
# Define SAC policy
policy = SACPolicy(
    actor=actor,
    actor_optim=actor_optim,
    critic=critic,
    critic_optim=critic_optim,
    critic2=critic2,
    critic2_optim=critic2_optim,
    tau=0.005,
    gamma=0.99,
    alpha=0.2,
    # reward_normalization=True,
    action_space=env.action_space  # Set to your environment's action space if available
)
# from tianshou.utils.torch_utils import policy_within_training_step

# # Training loop
# for epoch in range(1000):  # Adjust number of epochs as needed
#     # for batch in buffer.sample(batch_size=256):  # Adjust batch size as needed
#     with policy_within_training_step(policy, enabled=True):
#         loss = policy.update(sample_size=256, buffer=buffer, repeat=1)
#         print(loss)
#         print('=============================')
        
#%%
from tianshou.trainer import OfflineTrainer
from torch.utils.tensorboard import SummaryWriter
from tianshou.utils import TensorboardLogger
from tianshou.utils import BaseLogger
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
import gymnasium as gym
import gymnasium_robotics
from tianshou.policy.base import TLearningRateScheduler, TrainingStats
# gym.register_envs(gymnasium_robotics)
# env = gym.make('AdroitHandDoor-v1', render_mode='rgb_array')
def make_env():
    env = gym.make('AdroitHandDoor-v1', render_mode='rgb_array')
    # env = gym.wrappers.RecordVideo(
    #     env,
    #     video_folder='c:/users/armin/v',  # Specify your video directory
    #     episode_trigger=lambda episode_id: True  # Record every episode
    # )
    return env

# Create test environments
test_envs = DummyVectorEnv([make_env for _ in range(1)])
# test_envs = DummyVectorEnv([lambda: gym.make('AdroitHandDoor-v1', render_mode='rgb_array') for _ in range(1)])

test_collector = Collector(policy, test_envs)

    
log_path = 'c:/users/armin/v'  # Specify your log directory
writer = SummaryWriter(log_path)
logger = TensorboardLogger(writer)
# logger = PrintLogger()


trainer = OfflineTrainer(
    policy=policy,
    buffer=buffer,
    test_collector=test_collector,
    max_epoch=200,
    step_per_epoch=1000,
    batch_size=256,
    update_per_step=1,
    episode_per_test=1,
    logger=logger  # Pass the logger to the trainer
)
def hi1():
    print('good')
    
trainer.hi = hi1

trainer
#%%
trainer.run()