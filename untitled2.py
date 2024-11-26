import minari
import numpy as np

def download(dataset_id='D4RL/door/expert-v2'):
    dataset = minari.load_dataset(dataset_id, True)
    # return dataset
    accumulated_rewards = []
    observations, actions, rewards, terminations, truncations, next_observations = [], [], [], [], [], []
    for episode in dataset.iterate_episodes():
        episode_length = len(episode.observations)
        episode_rewards = np.sum(episode.rewards)
        accumulated_rewards.append(episode_rewards)
        for i in range(200):
            
            observations.append(episode.observations[i])
            actions.append(episode.actions[i])
            rewards.append(episode.rewards[i]) 
            terminations.append(episode.terminations[i])
            truncations.append(episode.truncations[i])
            # next_obs = episode.observations[i + 1] if i != 199 else np.zeros_like(episode.observations[i])
            next_obs = episode.observations[i + 1]
            next_observations.append(next_obs)
    
    return dataset, {
        'observations': np.array(observations),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'terminations': np.array(terminations),
        'truncations': np.array(truncations),
        'next_observations': np.array(next_observations),
        'dones': np.logical_or(terminations, truncations).astype(int),
        'accumulated_rewards': np.array(accumulated_rewards),
    }



ds, pds = download()
#%%
import matplotlib.pyplot as plt
accumulated_rewards = pds['accumulated_rewards']
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.histplot(accumulated_rewards, kde=True, bins=20, color='b')
plt.xlabel('Accumulated Reward')
plt.ylabel('Frequency')
plt.title('Distribution of Accumulated Rewards per Episode')
plt.grid(True)
plt.tight_layout()
plt.show()
