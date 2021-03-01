import torch
import torch.nn as nn
from torch.distributions import Categorical


class Discriminator(nn.Module):
    """
    GAIL's Discriminator neural network
    """
    def __init__(self, state_dim, action_dim, device):
        super(Discriminator, self).__init__()
        
        self.l1 = nn.Linear(state_dim+action_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, 1)
        self.device = device
    def forward(self, state, action):
        state_action = torch.cat([state, action.double()], 1)
        x = torch.tanh(self.l1(state_action))
        x = torch.tanh(self.l2(x))
        x = torch.sigmoid(self.l3(x))
        return x

