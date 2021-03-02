import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import numpy as np
from utils import *
import math
from expert_dataset import *
    
class BehavioralCloning(nn.Module):
    """
    Behavioral Cloning class.
    """
    def __init__(self, input_size, output_size):
        """
        Args
        - input_size: dimension of the states space.
        - output_size: number of possible actions.
        """
        super(BehavioralCloning, self).__init__()
        self.act_n = output_size
        
        self.obs_n = input_size
        self.pol = nn.Sequential(*[nn.Linear(self.obs_n, 64), 
                                   nn.Tanh(), 
                                   nn.Linear(64, 32), 
                                   nn.Tanh(), 
                                   nn.Linear(32, self.act_n),
                                   nn.Softmax(dim=-1)])
        
    
    def forward(self, obs):
        return self.pol(obs)
    
    def act(self, ob):
        with torch.no_grad():
            probs = self.forward(ob)
            m = Categorical(probs) 
            action = m.sample().item()
          
        return action, probs[action]
    
    
   