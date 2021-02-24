import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
import gym
import torch
from utils import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from gym import wrappers, logger
import numpy as np
import copy
from collections import deque 
from torch import nn
import time
from memory import *
import copy
from torch.optim import SGD, Adam
import torch.autograd
import torch.optim as optim
from memory import *
from utils import *
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DDPGagent:
    def __init__(self,state_dim,action_dim,hidden_size=256, actor_learning_rate=1e-4, critic_learning_rate=1e-3, gamma=0.99, tau=1e-2, max_memory_size=50000):
        # Params
        self.num_states = state_dim
        self.num_actions = action_dim
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, hidden_size, self.num_actions)

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        
        # Training
        self.memory = Memory(max_memory_size)        
        self.critic_criterion  = nn.MSELoss()
        self.actor_optimizer  = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)
    
    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0,0]
        return action
    
    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
    
        # Critic loss        
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states, self.actor.forward(states)).mean()
        
        # update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward() 
        self.critic_optimizer.step()

        # update target networks 
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
       
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    def save(self,outputDir):
        pass

if __name__ == '__main__':
    config = load_yaml('./configs/config_Pendulum.yaml')
    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env = gym.make(config["env"])

    # DDPGagent
    agent = DDPGagent(env)


    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])
    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/DDPG"


    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    episode_count = config["nbEpisodes"]

    #Initailisation
    state = env.reset()
    noise = OUNoise(env.action_space)
    noise.reset()
   
    log = False
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    if log:
        logger = LogMe(SummaryWriter(outdir))
        loadTensorBoard(outdir)

    batch_size = 128
    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    cur_frame = 0
    train_reward = []
    test_reward = []
    verb = True
    for i in range(episode_count):
        if i % int(config["freqVerbose"]) == 0 and i >= config["freqVerbose"]:
            verbose = verb
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            #print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            itest += 1
            if log:
                logger.direct_write("rewardTest", mean / nbTest, itest)
            test_reward.append(mean / nbTest)
            agent.test = False

        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()
            action = agent.get_action(state)
            action = noise.get_action(action, j)
            
            new_state, reward, done, _ = env.step(action) 

            agent.memory.push(state, action, reward, new_state, done)
            
            if len(agent.memory) > batch_size:
                agent.update(batch_size)  
            cur_frame+=1
            j+=1

            rsum += reward
            state = new_state
        
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                if log:
                    logger.direct_write("reward", rsum, i)
                train_reward.append(rsum)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                state = env.reset()
                break

    env.close()
    train_rg = np.arange(len(train_reward))
    test_rg = np.arange(len(test_reward))
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(train_rg,train_reward)
    ax[1].plot(test_rg,test_reward)
    fig.savefig("reward.png")
    plt.close(fig)

    