import argparse
import sys
import matplotlib
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter
from gym import wrappers, logger
import numpy as np
import copy
from collections import deque # we use a deque as a memory ((st,at,st1,rt,done))
from torch import nn
import time
from memory import *

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
class DQNAgent(object):
    """Using DQN algorithm"""

    def __init__(self):

        self.action_space = 3 # right or left
        self.learning_rate = 0.001
        self.batchsize = 100
        self.discount = 0.999
        self.epsilon = 1
        self.capacity= 2000

        self.activation = torch.nn.Tanh()
        self.Q = NN_Q(2,outSize=self.action_space,activation=self.activation,layers=[200]).to(device)
        self.optim = torch.optim.Adam(self.Q.parameters(),lr=self.learning_rate)   
        self.Q_hat_update_step = 50 # Step before Q_hat = Q 
        #self.D = deque(maxlen=opt.capacity) # transition history 
        self.D = Memory(self.capacity) # with prior
        self.Q_hat = NN_Q(2,outSize=self.action_space,activation=self.activation,layers=[200]).to(device)
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.loss = nn.SmoothL1Loss()
      
        self.count=0
        self.episode=0
        self.lastobs = None 
        self.lasta = None
        
        self.epsilonGreedyDecay =  EpsilonGreedyDecay(self.epsilon,0.01,0.01)
        #self.memory_intialization()
        

    def act(self, obs):
        if obs.step == 0 :
            observation = np.array([0,obs.step])
        else:

            observation = np.array([obs.lastOpponentAction,obs.step])
        reward = obs.reward
        done = False
        self.episode+=1
        self.obs=observation # we save the current observation
        self.Q_hat.eval() # set the module on evaluation mode
        self.Q.train() # set the module on training mode
        self.reward = reward # we save the current reward 

        #reply ( training the NN using batch from memory)
        if self.lasta is not None: # in the first time we do not train the module 
            self.replay(observation, reward, done)
        
        # choosing action
        with torch.no_grad():
            observation = torch.from_numpy(self.obs) 
            q_values = self.Q(observation.float().to(device)) # we get the q_values for all the actions   
            action = self.epsilonGreedyDecay.act(self.episode,q_values)

        # updates
        self.lastobs = self.obs 
        self.lasta = action
        self.count+=1
        if self.count==self.Q_hat_update_step: 
            self.count=0
            self.Q_hat.load_state_dict(self.Q.state_dict()) # each Q_hat_update_step we set Q_hat to Q

        return action
    
    def play(self, obs, configuration):
    
        return self.act(obs)
    
    # didn't use it
    def memory_intialization(self):
        ob0 = self.env.reset()
        for i in range(self.capacity):
            print(i)
            action = np.random.randint(self.action_space) 
            ob1, reward, done, _ = self.env.step(action)
            self.D.append((ob0,action,reward,ob1,done)) # we fill the memory
            ob0=ob1
            if done:
                self.env.reset()
        self.env.reset()
    
    def replay(self,observation, reward, done):
        self.D.store((self.lastobs,self.lasta,self.reward,self.obs,done)) # we fill the memory
        #indices = np.random.choice(range(len(self.D)),self.batchsize) # we take #batchsize random tuples from the memory 
        #tuples = [self.D[i] for i in indices] 
        tuples=self.D.sample(self.batchsize)
        idx,w,tuples=tuples
        #print(tuples)
        so,a,r,s1,d = map(list,zip(*tuples)) 
        r=torch.Tensor(r).to(device) # rewards tensor
        d=torch.BoolTensor(d).to(device) # done's tensor
        so=torch.Tensor(so).to(device) # initial states tensor
        s1=torch.Tensor(s1).to(device) # next states tensor

        with torch.no_grad():
            q_values = self.Q_hat(s1.to(device)) # we get the Q_values for all the possible actions from the next state
            val,ind = torch.max(q_values,1) # we take only the max q-values for each element of our batch
            target_hat=torch.where(d,r,r+self.discount*val) # if done we take only r otherwise we take r+ discount * max Q(s1,a)
        
        target=self.Q(so.to(device))  # we get the Q_values for all the possible actions from the next state
        q_values=target[range(self.batchsize),a] # we take all the q_values correspending to the actions token
        loss=self.loss(target_hat,q_values) # we compute the loss between Q(s0,a) and (r + discount * max Q(s1,a))
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        
            
