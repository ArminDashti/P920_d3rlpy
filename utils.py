import numpy as np
import os
import torch
import pickle
from torch.utils.data import TensorDataset


def load_checkpoint(args, policy, qf1, qf2, qf1_target, qf2_target, actor_optimizer, qf1_optimizer, qf2_optimizer, 
                    alpha_optimizer, log_alpha, epoch_counter):
    checkpoint_dir = os.path.join(args.output_dir, 'check_points')
    start_epoch = 0
    if os.path.exists(checkpoint_dir):
        checkpoint_folders = [
            f for f in os.listdir(checkpoint_dir)
            if f.startswith('epoch_') and os.path.isdir(os.path.join(checkpoint_dir, f))]
        if checkpoint_folders:
            latest_epoch_folder = max(
                checkpoint_folders,
                key=lambda x: int(x.split('_')[1]))
            checkpoint_path = os.path.join(checkpoint_dir, latest_epoch_folder, 'checkpoint.pth')
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            policy.load_state_dict(checkpoint['policy_state_dict'])
            qf1.load_state_dict(checkpoint['qf1_state_dict'])
            qf2.load_state_dict(checkpoint['qf2_state_dict'])
            qf1_target.load_state_dict(checkpoint['qf1_target_state_dict'])
            qf2_target.load_state_dict(checkpoint['qf2_target_state_dict'])
            actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            qf1_optimizer.load_state_dict(checkpoint['qf1_optimizer_state_dict'])
            qf2_optimizer.load_state_dict(checkpoint['qf2_optimizer_state_dict'])
            alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
            log_alpha.data.copy_(checkpoint['log_alpha'])
            start_epoch = checkpoint['epoch']
            epoch_counter[0] = start_epoch
    return start_epoch


def soft_update(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def compute_grad_norm(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def get_tensor_values(obs, actions, network=None):
    action_shape = actions.shape[0]
    obs_shape = obs.shape[0]
    num_repeat = int (action_shape / obs_shape)
    obs_temp = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
    preds = network(obs_temp, actions)
    preds = preds.view(obs.shape[0], num_repeat, 1)
    return preds


def get_policy_actions(obs, num_actions, network=None):
    obs_temp = obs.unsqueeze(1).repeat(1, num_actions, 1).view(obs.shape[0] * num_actions, obs.shape[1])
    new_obs_actions, new_obs_log_pi = network.select_action(obs_temp)
    return new_obs_actions, new_obs_log_pi.view(obs.shape[0], num_actions, 1)


def load_dataset(dataset_path):
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)
    return TensorDataset(
        torch.FloatTensor(data['observations']),
        torch.FloatTensor(data['actions']),
        torch.FloatTensor(data['rewards']).unsqueeze(1),
        torch.FloatTensor(data['next_observations']),
        torch.FloatTensor(data['dones']).unsqueeze(1))


def normalize_rewards(reward_batch):
    mean_reward = reward_batch.mean()
    std_reward = reward_batch.std()
    return (reward_batch - mean_reward) / (std_reward + 1e-8)


def soft_update(source_model, target_model, tau):
    for source_param, target_param in zip(source_model.parameters(), target_model.parameters()):
        target_param.data.copy_(tau * source_param.data + (1 - tau) * target_param.data)


def load_safe_action_pth(checkpoint_path, model):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])