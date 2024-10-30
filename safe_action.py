import time
import torch
import logging
import networks
from tqdm import tqdm
import os
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import style
import re

# Set matplotlib backend explicitly
plt.switch_backend('Agg')

# Set a fancy background style
style.use('seaborn-poster')
plt.style.use('dark_background')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_dataset(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def parse_logs(log_file):
    metrics = {'losses': [], 'grad_norms': [], 'accuracies': []}
    with open(log_file, 'r') as file:
        for line in file:
            if "Loss:" in line:
                metrics['losses'].append(float(re.search(r"Loss: ([0-9]*\.[0-9]+)", line).group(1)))
            elif "Grad Norm:" in line:
                metrics['grad_norms'].append(float(re.search(r"Grad Norm: ([0-9]*\.[0-9]+)", line).group(1)))
            elif "Correct Predictions in Percent:" in line:
                metrics['accuracies'].append(float(re.search(r"Correct Predictions in Percent: ([0-9]*\.[0-9]+)%", line).group(1)))
    return metrics

def initialize_logging(log_file, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        open(log_file, 'w').close()
    logging.basicConfig(filename=log_file, level=logging.INFO, format='%(message)s')

def load_checkpoint(checkpoint_path, model, optimizer):
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print("Loading checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")
    return start_epoch

def save_checkpoint(checkpoint_dir, model, optimizer, epoch, epoch_loss):
    current_time = time.strftime("%H_%M", time.localtime())
    checkpoint_name = f'epoch_{epoch+1}_{current_time}.pth'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': epoch_loss
    }
    torch.save(checkpoint, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))

def plot_metrics(plots_dir, losses, grad_norms, accuracies):
    os.makedirs(plots_dir, exist_ok=True)
    plt.figure(figsize=(15, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(range(1, len(losses) + 1), losses, label='Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss per Epoch')

    # Plot Grad Norm
    plt.subplot(1, 3, 2)
    plt.plot(range(1, len(grad_norms) + 1), grad_norms, label='Grad Norm', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Gradient Norm per Epoch')

    # Plot Accuracy
    plt.subplot(1, 3, 3)
    plt.plot(range(1, len(accuracies) + 1), accuracies, label='Accuracy (%)', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy per Epoch')

    # Save the aggregated plot
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'training_metrics.png'))
    print(f"Saved aggregated plot at: {os.path.join(plots_dir, 'training_metrics.png')}")
    plt.close()

def train(args):
    log_dir = os.path.join(args.outputs_dir, 'logs')
    checkpoint_dir = os.path.join(args.outputs_dir, 'check_points', 'safe_Action')
    log_file = os.path.join(log_dir, 'safe_action_epoch_logs.log')
    checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    initialize_logging(log_file, checkpoint_path)
    metrics = parse_logs(log_file)

    model = networks.SafeAction(args).to(device)
    loss_func = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), args.safe_action_lr)

    train_dataset = load_dataset(os.path.join(args.outputs_dir, 'datasets', 'safe_action_train_dataset.pkl'))
    test_dataset = load_dataset(os.path.join(args.outputs_dir, 'datasets', 'safe_action_test_dataset.pkl'))

    train_tensor_dataset = TensorDataset(
        torch.tensor(train_dataset['observations'], dtype=torch.float32),
        torch.tensor(train_dataset['actions'], dtype=torch.float32),
        torch.tensor(train_dataset['in_dist'], dtype=torch.float32).unsqueeze(1)
    )
    train_loader = DataLoader(train_tensor_dataset, batch_size=args.safe_action_train_bs)

    test_tensor_dataset = TensorDataset(
        torch.tensor(test_dataset['observations'], dtype=torch.float32),
        torch.tensor(test_dataset['actions'], dtype=torch.float32),
        torch.tensor(test_dataset['in_dist'], dtype=torch.float32).unsqueeze(1)
    )
    test_loader = DataLoader(test_tensor_dataset, batch_size=args.safe_action_train_bs)

    start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
    num_epochs = args.safe_action_num_epochs
    elapsed_times = []
    correct_predictions_list = []

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        with tqdm(train_loader, unit='batch', mininterval=1) as t:
            for batch in t:
                t.set_description(f"Epoch {epoch+1}/{num_epochs}")
                state, action, in_dist = (b.to(device) for b in batch)
                optimizer.zero_grad()
                pred_in_dist = model(state, action)
                loss = loss_func(pred_in_dist, in_dist)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

        elapsed_time = time.time() - start_time
        total_norm = torch.sqrt(torch.tensor(sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None)))

        # Log information
        log_message = (
            f"====================  EPOCH {epoch+1} ====================\n"
            f"Start Time: {time.strftime('%H:%M:%S', time.localtime(start_time))}\n"
            f"End Time: {time.strftime('%H:%M:%S', time.localtime(start_time + elapsed_time))}\n"
            f"Elapsed Time: {elapsed_time:.2f} seconds\n"
            f"Loss: {epoch_loss:.4f}\n"
            f"Grad Norm: {total_norm:.4f}"
        )
        print(log_message)
        logging.info(log_message)

        # Save metrics for plotting
        metrics['losses'].append(epoch_loss)
        metrics['grad_norms'].append(total_norm)
        elapsed_times.append(elapsed_time)

        # Log the range for this epoch
        epoch_final_log_message = (
            f"Loss range (Up to Epoch {epoch+1}) = [{min(metrics['losses']):.4f}, {max(metrics['losses']):.4f}]\n"
            f"Grad Norm range (Up to Epoch {epoch+1}) = [{min(metrics['grad_norms']):.4f}, {max(metrics['grad_norms']):.4f}]\n"
            f"Correct Predictions range (Up to Epoch {epoch+1}) = [{min(correct_predictions_list) if correct_predictions_list else 0}, {max(correct_predictions_list) if correct_predictions_list else 0}]\n"
            f"Elapsed Time range (Up to Epoch {epoch+1}) = [{min(elapsed_times):.2f} seconds, {max(elapsed_times):.2f} seconds]"
        )
        print(epoch_final_log_message)
        logging.info(epoch_final_log_message)

        save_checkpoint(checkpoint_dir, model, optimizer, epoch, epoch_loss)

        # Evaluate the model after each epoch
        test_accuracy, correct_predictions = evaluate_model(model, test_loader)
        eval_message = (
            f"Correct Predictions: {correct_predictions}\n"
            f"Correct Predictions in Percent: {test_accuracy * 100:.2f}%"
        )
        print(eval_message)
        logging.info(eval_message)
        metrics['accuracies'].append(test_accuracy * 100)
        correct_predictions_list.append(correct_predictions)

        # Plot metrics after each epoch
        plot_metrics(os.path.join(args.outputs_dir, 'plots'), metrics['losses'], metrics['grad_norms'], metrics['accuracies'])

def evaluate_model(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for state, action, in_dist in tqdm(dataloader, desc='Evaluating', unit='batch'):
            state = state.to(device)
            action = action.to(device)
            in_dist = in_dist.to(device)

            pred_in_dist = model(state, action)
            correct_predictions += (pred_in_dist.round() == in_dist).sum().item()
            total_samples += in_dist.size(0)

    accuracy = correct_predictions / total_samples
    return accuracy, correct_predictions
