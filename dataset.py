import os
import minari
import numpy as np
import pickle
from torch.utils.data import DataLoader, TensorDataset, random_split
import random
import torch


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


def add_synthetic_data(dataset_dict):
    # Generate synthetic data by perturbing the actions based on their real actions in the dataset
    observations = dataset_dict['observations']
    actions = dataset_dict['actions']

    # Perturb the actions such that each action may deviate up to 30% from its original action based on L2 distance
    perturbation = np.random.uniform(-0.3, 0.3, size=actions.shape) * np.linalg.norm(actions, ord=2, axis=1, keepdims=True)
    perturbation = perturbation / (np.linalg.norm(perturbation, ord=2, axis=1, keepdims=True) + 1e-6) * 0.3 * np.linalg.norm(actions, ord=2, axis=1, keepdims=True)
    act_synthetic = actions + perturbation
    act_synthetic = np.clip(act_synthetic, -1, 1)

    # Calculate and print average L2 distance between original and synthetic actions
    distances = np.linalg.norm(actions - act_synthetic, ord=2, axis=1)
    avg_distance = np.mean(distances)
    print(f'Average L2 distance between original and synthetic actions: {avg_distance:.4f}')

    # Synthetic data indicator
    in_dist_synthetic = np.zeros(len(observations))

    # Set zeros for other values
    rew_zeros = np.zeros(len(observations))
    next_obs_zeros = np.zeros((len(observations), observations.shape[1]))
    term_zeros = np.zeros(len(observations), dtype=int)
    trunc_zeros = np.zeros(len(observations), dtype=int)
    dones_zeros = np.zeros(len(observations), dtype=int)

    # Combine all synthetic data
    dataset_dict['observations'] = np.vstack([dataset_dict['observations'], observations])
    dataset_dict['actions'] = np.vstack([dataset_dict['actions'], act_synthetic])
    dataset_dict['rewards'] = np.concatenate([dataset_dict['rewards'], rew_zeros])
    dataset_dict['next_observations'] = np.vstack([dataset_dict['next_observations'], next_obs_zeros])
    dataset_dict['terminations'] = np.concatenate([dataset_dict['terminations'], term_zeros])
    dataset_dict['truncations'] = np.concatenate([dataset_dict['truncations'], trunc_zeros])
    dataset_dict['dones'] = np.concatenate([dataset_dict['dones'], dones_zeros])
    dataset_dict['in_dist'] = np.concatenate([dataset_dict['in_dist'], in_dist_synthetic])

    return dataset_dict


def get_dataset_info(args, dataset_dict):
    args.state_dim = dataset_dict['observations'].shape[1]
    args.action_dim = dataset_dict['actions'].shape[1]
    args.reward_range = (np.min(dataset_dict['rewards']), np.max(dataset_dict['rewards']))
    return args


def split_dataset_dict(dataset_dict, train_ratio=0.8):
    total_size = len(dataset_dict['observations']) if isinstance(dataset_dict, dict) else len(dataset_dict.observations)
    train_size = int(train_ratio * total_size)
    test_size = total_size - train_size

    indices = list(range(total_size))
    random.shuffle(indices)
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    train_dataset = {
        key: dataset_dict[key][train_indices]
        for key in ['observations', 'actions', 'rewards', 'dones', 'next_observations', 'in_dist']
    }
    test_dataset = {
        key: dataset_dict[key][test_indices]
        for key in ['observations', 'actions', 'rewards', 'dones', 'next_observations', 'in_dist']
    }

    return train_dataset, test_dataset


# The saved dataset is stored as a pickle file, containing dictionaries with keys such as 'observations', 'actions', 'rewards', 'dones', and 'next_observations'.
# Each key holds a NumPy array representing different components of the dataset, allowing easy access for training and evaluation purposes.

# The saved dataset is stored as a pickle file in the format of a dictionary.
# It contains keys such as 'observations', 'actions', 'rewards', 'dones', and 'next_observations', each of which holds a NumPy array.
# The format is as follows:
# {
#   'observations': np.ndarray of shape (num_samples, observation_dim),
#   'actions': np.ndarray of shape (num_samples, action_dim),
#   'rewards': np.ndarray of shape (num_samples,),
#   'dones': np.ndarray of shape (num_samples,),
#   'next_observations': np.ndarray of shape (num_samples, observation_dim),
#   ... (other keys if present)
# }
# This format allows for easy access to different components of the dataset for training and evaluation purposes.

def save_dataset(dataset, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)


def run(args):
    # Merged save_safe_action_dataset functionality into run
    dataset_dict = download()
    dataset_dict_synt = add_synthetic_data(dataset_dict)
    train_ds, test_ds = split_dataset_dict(dataset_dict_synt)

    train_output_path = os.path.join(args.outputs_dir, 'datasets', 'safe_action_train_dataset.pkl')
    test_output_path = os.path.join(args.outputs_dir, 'datasets', 'safe_action_test_dataset.pkl')

    save_dataset(train_ds, train_output_path)
    save_dataset(test_ds, test_output_path)
    
    # Merged save_actor_critic_dataset functionality into run
    dataset_dict_output_path = os.path.join(args.outputs_dir, 'datasets', 'actor_critic_datasets.pkl')
    save_dataset(dataset_dict, dataset_dict_output_path)
