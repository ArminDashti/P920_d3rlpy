import torch
import torch.nn as nn
from torch.distributions import Normal
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SafeAction(nn.Module):
    def __init__(self, args):
        super(SafeAction, self).__init__()
        state_dim = args.state_dim
        action_dim = args.action_dim
        hidden_dim = args.safe_action_hidden_dim

        self.state_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU())
        
        self.action_layer = nn.Sequential(
            nn.Linear(action_dim, hidden_dim),
            nn.ReLU())
        
        self.combined_layer = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())


    def forward(self, state, action):
        state_processed = self.state_layer(state)
        action_processed = self.action_layer(action)
        combined_input = torch.cat([state_processed, action_processed], dim=-1)
        output = self.combined_layer(combined_input)
        return output


