import argparse
import os
import torch
import d3rlpy
from d3rlpy.datasets import get_minari
from d3rlpy.algos import SACConfig
from d3rlpy.models.torch.policies import build_squashed_gaussian_distribution
from d3rlpy.models.torch.parameters import get_parameter
from d3rlpy.algos.qlearning.torch.sac_impl import SACActorLoss
from d3rlpy.torch_utility import convert_to_torch 
from d3rlpy.metrics import evaluate_qlearning_with_environment
from d3rlpy.algos import SAC
from networks import SafeAction
from d3rlpy.metrics import EnvironmentEvaluator
from d3rlpy.metrics.evaluators import EnvironmentEvaluator


device = "cuda" if torch.cuda.is_available() else "cpu"


def compute_actor_loss(self, batch, action):
    sa_obs = convert_to_torch(batch.observations, device=device)
    sa_act = convert_to_torch(batch.actions, device=device)
    if torch.rand(1).item() < 0.7:
        result = safe_action(sa_obs, sa_act).round()
    else:
        result = torch.full((256,), 1.0)

    dist = build_squashed_gaussian_distribution(action)
    sampled_action, log_prob = dist.sample_with_log_prob()

    temp_loss = self.update_temp(log_prob) if self._modules.temp_optim else torch.tensor(
        0.0, dtype=torch.float32, device=sampled_action.device
    )
    entropy = get_parameter(self._modules.log_temp).exp() * log_prob
    q_t = self._q_func_forwarder.compute_expected_q(
        batch.observations, sampled_action, "min"
    )
    return SACActorLoss(
        actor_loss=(entropy*result - q_t).mean(),
        temp_loss=temp_loss,
        temp=get_parameter(self._modules.log_temp).exp()[0][0],
    )


def load_safe_action():
    parser = argparse.ArgumentParser(description='Load the safe action model.')
    parser.add_argument('--state_dim', type=int, default=39, help='State dimension of the model.')
    parser.add_argument('--action_dim', type=int, default=28, help='Action dimension of the model.')
    parser.add_argument('--safe_action_hidden_dim', type=int, default=256, help='Hidden dimension of the model.')
    load_args = parser.parse_args()

    model = SafeAction(load_args)
    checkpoint = torch.load(r"C:\Users\armin\P920_output\check_points\safe_action\latest_checkpoint.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
    
safe_action = load_safe_action()
d3rlpy.algos.qlearning.torch.SACImpl.compute_actor_loss = compute_actor_loss
d3rlpy.algos.qlearning.torch.SACImpl = safe_action


def load_dataset(dataset_name="D4RL/door/expert-v2"):
    return get_minari(dataset_name)

    
def get_sac(device=device):
    sac_config = SACConfig()
    sac = sac_config.create(device=device)
    return sac

def evaluate_after_epoch(algo, epoch, total_step):
    eval_score = evaluate_qlearning_with_environment(algo)
    print(f"Epoch {epoch}: Evaluation Score = {eval_score}")


def train_sac(env, sac, dataset, n_steps=1000000):
    sac.build_with_env(env)
    sac.fit(
    dataset,
    n_steps=n_steps,
    evaluators={
        "environment": EnvironmentEvaluator(env)  # Add environment evaluator
    },)


def predict_action(sac, env):
    observation = env.reset()
    return sac.predict(observation)


def run(args):
    dataset, env = load_dataset()
    checkpoint_path = os.path.join(args.assets_dir, 'check_points', 'safe_action', 'safe_action_checkpoint.pth')
    sac = get_sac(device=device)
    
    # Ensure SACImpl is properly initialized before patching
    if sac._impl is not None:
        sac._impl.compute_actor_loss = lambda batch, action: compute_actor_loss(sac._impl, batch, action, checkpoint_path=checkpoint_path)
    
    train_sac(env, sac, dataset)
    actions = predict_action(sac, env)
    print(actions)


def main():
    parser = argparse.ArgumentParser(description='Train and predict with SAC using a safe action model.')
    parser.add_argument('--assets_dir', type=str, required=True, help='Directory containing assets like checkpoints.')
    parser.add_argument('--checkpoint_path', type=str, required=False, help='Path to the safe action checkpoint.')
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
