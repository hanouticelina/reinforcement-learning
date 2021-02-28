from collections import deque
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd 
import torch.nn.functional as F
import random 
class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, goal, action, reward, next_state, done):
        state = state.detach().cpu()
        next_state = next_state.detach().cpu()
        goal = goal.detach().cpu()
        self.buffer.append((state, goal,action, reward, next_state, done))
    
    def sample(self, batch_size):
        state, goal, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        
        return np.concatenate(state), np.concatenate(goal), action, reward, np.concatenate(next_state), done
    
    def __len__(self):
        return len(self.buffer)
    
    
    