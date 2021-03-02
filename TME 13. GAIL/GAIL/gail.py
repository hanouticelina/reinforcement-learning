import pickle
import argparse
import sys
import matplotlib
import gym
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import torch
from torch.distributions import Categorical
from random import random
from pathlib import Path
import numpy as np
import os
from torch.autograd import Variable
from discriminator import *
from expert_dataset import *
from torch.distributions import Categorical

beta = 1e-3
gamma = 0.98
delta = 0.8
h = 300
K = 10
eps = 0.2
lam_ent = 1e-3

class Memory:
    """
    Memory Buffer
    """
    def __init__(self):
        self.memory = []
        
    def __len__(self):
        return len(self.memory)
    
    def add(self, ob, action, prob_action, reward, new_ob , done):
        self.memory.append([ob, action, prob_action, reward, new_ob , done])
        
    def clear(self, device):
        obs, actions, prob_actions, rewards, new_obs, done = (
        torch.tensor([transition[0] for transition in self.memory], device=device, dtype=float),
        torch.tensor([transition[1] for transition in self.memory], device=device, dtype=int),
        torch.tensor([transition[2] for transition in self.memory], device=device, dtype=float),
        torch.tensor([transition[3] for transition in self.memory], device=device, dtype=float).view(-1, 1),
        torch.tensor([transition[4] for transition in self.memory], device=device, dtype=float),
        torch.tensor([transition[5] for transition in self.memory], device=device, dtype=float).view(-1, 1),
        )

        self.memory = []
        return obs, actions, prob_actions, rewards, new_obs, done 

class GAIL(nn.Module):
    """
    GAIL agent.
    """
    def __init__(self, env, opt, device, path_to_expert_data, mini_batch_size = 32,  eta = 1e-2):
        super(GAIL, self).__init__()
        self.epoch, self.iteration = 0, 0
        self.feature_extractor = opt.featExtractor(env)
        self.mini_batch_size = mini_batch_size
        self.eta = eta
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.nb_actions = env.action_space.n
        self.discriminator = Discriminator(self.state_dim, self.nb_actions,device).to(device).double()
        self.criterion = nn.BCELoss()
        self.v = nn.Sequential(nn.Linear(self.state_dim, h), nn.Tanh(), nn.Linear(h, 1)).to(device).double()
        self.pi = nn.Sequential(
            nn.Linear(self.state_dim, h),
            nn.Tanh(),
            nn.Linear(h, self.nb_actions),
            nn.Softmax(dim=-1),
        ).to(device).double()
        
        self.optim_discriminator = torch.optim.Adam(
            params=self.discriminator.parameters(), lr=3e-4
        )
        self.optimizer_v = torch.optim.Adam(params=self.v.parameters(), lr=3e-4)
        self.optimizer_pi = torch.optim.Adam(params=self.pi.parameters(), lr=3e-4)
        # expert data
        self.expert_dataset = ExpertDataset(env, path_to_expert_data, device)
        self.expert_states, self.expert_actions = self.expert_dataset.get_expert_data()
        
        # create a memory 
        self.buffer = Memory()
        
    def get_action(self, observation, rewrd, done):
        """
        Action with respect to the current policy
        """
        with torch.no_grad():
            features = self.feature_extractor.getFeatures(observation)
            prob_on_states = self.pi(torch.tensor(features, dtype=float).to(device))
            prob = Categorical(prob_on_states)
            action = prob.sample().item()
        return action, prob_on_states[action]
    
        
    def sample_from_expert(self, n_samples):
        """
        Get n samples from the expert's trajectory
        """
        ids = torch.randperm(self.expert_states.size(0))[: min(self.mini_batch_size, n_samples)]
        states = self.expert_states[ids]
        actions = self.expert_actions[ids]
        return states, actions

    def sample_from_agent(self):
        """
        Get n samples from the expert's trajectory
        """
        states, actions = (
            torch.tensor([transition[0] for transition in self.buffer.memory], device=device, dtype=float),
            torch.tensor([transition[1] for transition in self.buffer.memory], device=device, dtype=float),
        )
        ids = torch.randperm(states.size(0))
        states = states[ids]
        actions = actions[ids]
        actions = self.expert_dataset.toOneHot(actions)
        return states, actions

    def update(self):
        mem_size = len(self.buffer)
        for k in range(K):
            x_expert, x_agent = self.sample_from_expert(mem_size), self.sample_from_agent()
            batch_size = x_expert[0].size(0)

            exp_label= torch.full((batch_size,1), 1, device=self.device)
            policy_label = torch.full((batch_size,1), 0, device=self.device)

            self.optim_discriminator.zero_grad()
            prob_exp = self.discriminator(x_expert[0], x_expert[1])
            loss = self.criterion(prob_exp, exp_label.double())

            prob_policy = self.discriminator(x_agent[0], x_agent[1])
            loss += self.criterion(prob_policy, policy_label.double())
            loss.backward()

            self.optim_discriminator.step()
            with torch.no_grad():
                for param in self.discriminator.parameters():
                    param.add_(torch.randn(param.size(), device = self.device) * self.eta)

        state, action, prob_actions, rewards, new_obs, done = self.buffer.clear(self.device)
        # critic
        v = self.v(state).flatten()
         
        # v's loss 
        r_t = torch.log(self.discriminator(state, self.expert_dataset.toOneHot(action)).flatten())
        
        for i in range(r_t.size(0)): # clip [-100,]
            if r_t[i] <= -100:
                r_t[i] = -100
        R = [r_t[i:].mean() for i in range(mem_size)] # reward-to-go
        adv = torch.stack(R).to(device=self.device, dtype=float)
        
        v_loss = F.smooth_l1_loss(v, adv.detach()) 
        
        # gradient step for the v network
        self.optimizer_v.zero_grad()
        v_loss.backward()
        self.optimizer_v.step()
        # PPO update
        for _ in range(K):
            pi = self.pi(state)
            prob = pi[np.arange(len(pi)),action.detach().cpu().numpy()]

            ratio = torch.exp(
                torch.log(prob) - torch.log(prob_actions.float())
            )
            surr1 = ratio * adv.detach()
            surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * adv.detach()
            L = -torch.min(surr1, surr2)
            L = L.mean()
            H = (pi * torch.log(pi)).mean(dim=-1)
            H = H.mean()
            self.optimizer_pi.zero_grad()
            (L - lam_ent * H).backward()
            self.optimizer_pi.step()
        self.iteration += 1

        with torch.no_grad():
            prob_new_actions = pi[np.arange(len(pi)),action.detach().cpu().numpy()]
            d_kl = (
                prob_new_actions
                * (torch.log(prob_new_actions) / torch.log(prob_actions.float()))
            ).mean() 
        return loss.item(), L.item(), H.item(), v_loss.item(), d_kl.item()
    
        