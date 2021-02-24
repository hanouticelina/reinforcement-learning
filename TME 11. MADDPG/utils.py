import time
import subprocess
from collections import namedtuple,defaultdict
import logging
import json
import os
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import threading
import numpy as np
import gym
from collections import deque
import random
import torch.autograd
from torch.autograd import Variable
import torch.optim as optim
import copy



class CriticNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):

        super(CriticNetwork, self).__init__()
        self.state_fc = nn.Linear(state_dim, 64)
        self.fc1 = nn.Linear(action_dim+64, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.reset_parameters()

    def reset_parameters(self):

        self.state_fc.weight.data.uniform_(*hidden_init(self.state_fc))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):

        state, action = state.squeeze(), action.squeeze()
        x = F.relu(self.state_fc(state))
        x = torch.cat((x, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class ActorNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):


        super(ActorNetwork, self).__init__()
    
        self.fc1 = nn.Linear(state_dim, 64)
        
        self.fc2 = nn.Linear(64, 128)
        
        self.fc3 = nn.Linear(128, action_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters
        """
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, x):
        """
        Maps a state to actions
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)



def loadTensorBoard(outdir):
    t = threading.Thread(target=launchTensorBoard, args=([outdir]))
    t.start()

def launchTensorBoard(tensorBoardPath):
    print('tensorboard --logdir=' + tensorBoardPath)
    ret=os.system('tensorboard --logdir='  + tensorBoardPath)
    if ret!=0:
        syspath = os.path.dirname(sys.executable)
        print(os.path.dirname(sys.executable))
        ret = os.system(syspath+"/"+'tensorboard --logdir=' + tensorBoardPath)
    return

class Orn_Uhlen:
    def __init__(self, n_actions, mu=0, theta=0.15, sigma=0.2):
        self.n_actions = n_actions
        self.X = np.ones(n_actions) * mu
        self.mu = mu
        self.sigma = sigma
        self.theta = theta

    def reset(self):
        self.X = np.ones(self.n_actions) * self.mu

    def sample(self):
        dX = self.theta * (self.mu - self.X)
        dX += self.sigma * np.random.randn(self.n_actions)
        self.X += dX
        return torch.FloatTensor(self.X)

class FeatureExtractor(object):
    def __init__(self):
        super().__init__()

    def getFeatures(self,obs):
        pass

class NothingToDo(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        ob=env.reset()
        self.outSize=len(ob)

    def getFeatures(self,obs):
        return obs

######  Pour Gridworld #############################"

class MapFromDumpExtractor(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        outSize = env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize=outSize

    def getFeatures(self, obs):
        #prs(obs)
        return obs.reshape(1,-1)

class MapFromDumpExtractor2(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        outSize=env.start_grid_map.reshape(1, -1).shape[1]
        self.outSize=outSize*3

    def getFeatures(self, obs):
        state=np.zeros((3,np.shape(obs)[0],np.shape(obs)[1]))
        state[0]=np.where(obs == 2,1,state[0])
        state[1] = np.where(obs == 4, 1, state[1])
        state[2] = np.where(obs == 6, 1, state[2])
        return state.reshape(1,-1)




class DistsFromStates(FeatureExtractor):
    def __init__(self,env):
        super().__init__()
        self.outSize=16

    def getFeatures(self, obs):
        #prs(obs)
        #x=np.loads(obs)
        x=obs
        #print(x)
        astate = list(map(
            lambda x: x[0] if len(x) > 0 else None,
            np.where(x == 2)
        ))
        astate=np.array(astate)
        a3=np.where(x == 3)
        d3=np.array([0])
        if len(a3[0])>0:
            astate3 = np.concatenate(a3).reshape(2,-1).T
            d3=np.power(astate-astate3,2).sum(1).min().reshape(1)

            #d3 = np.array(d3).reshape(1)
        a4 = np.where(x == 4)
        d4 = np.array([0])
        if len(a4[0]) > 0:
            astate4 = np.concatenate(a4).reshape(2,-1).T
            d4 = np.power(astate - astate4, 2).sum(1).min().reshape(1)
            #d4 = np.array(d4)
        a5 = np.where(x == 5)
        d5 = np.array([0])
        #prs(a5)
        if len(a5[0]) > 0:
            astate5 = np.concatenate(a5).reshape(2,-1).T
            d5 = np.power(astate - astate5, 2).sum(1).min().reshape(1)
            #d5 = np.array(d5)
        a6 = np.where(x == 6)
        d6 = np.array([0])
        if len(a6[0]) > 0:
            astate6 = np.concatenate(a6).reshape(2,-1).T
            d6 = np.power(astate - astate6, 2).sum(1).min().reshape(1)
            #d6=np.array(d6)

        #prs("::",d3,d4,d5,d6)
        ret=np.concatenate((d3,d4,d5,d6)).reshape(1,-1)
        ret=np.dot(ret.T,ret)
        return ret.reshape(1,-1)








class convMDP(nn.Module):
    def __init__(self, inSize, outSize, layers=[], convs=None, finalActivation=None, batchNorm=False,init_batchNorm=False,activation=torch.tanh):
        super(convMDP, self).__init__()
        #print(inSize,outSize)

        self.inSize=inSize
        self.outSize=outSize
        self.batchNorm=batchNorm
        self.init_batchNorm = init_batchNorm
        self.activation=activation

        self.convs=None
        if convs is not None:
            self.convs = nn.ModuleList([])
            for x in convs:
                self.convs.append(nn.Conv2d(x[0], x[1], x[2], stride=x[3]))
                inSize = np.sqrt(inSize / x[0])
                inSize=((inSize-x[2])/x[3])+1
                inSize=inSize*inSize*x[1]
        #print(inSize)

        self.layers = nn.ModuleList([])
        self.bn = nn.ModuleList([])
        i=0
        if batchNorm or init_batchNorm:
            self.bn.append(nn.BatchNorm1d(num_features=inSize))
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            if batchNorm:
                self.bn.append(nn.BatchNorm1d(num_features=x))

            #nn.init.xavier_uniform_(self.layers[i].weight)
            nn.init.normal_(self.layers[i].weight.data, 0.0, 0.02)
            nn.init.normal_(self.layers[i].bias.data,0.0,0.02)
            i+=1
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))

        #nn.init.uniform_(self.layers[-1].weight)
        nn.init.normal_(self.layers[-1].weight.data, 0.0, 0.02)
        nn.init.normal_(self.layers[-1].bias.data, 0.0, 0.02)
        self.finalActivation=finalActivation





    def forward(self, x):
        #print("d", x.size(),self.inSize)
        x=x.view(-1,self.inSize)

        if self.convs is not None:

            n=x.size()[0]
            i=0
            for c in self.convs:
                if i==0:
                    w=np.sqrt(x.size()[1])
                    x=x.view(n,c.in_channels,w,w)
                x=c(x)
                x=self.activation(x)
                i+=1
            x=x.view(n,-1)

        #print(x.size())
        if self.batchNorm or self.init_batchNorm:
            x=self.bn[0](x)
        x = self.layers[0](x)


        for i in range(1, len(self.layers)):
            x = self.activation(x)
            #if self.drop is not None:
            #    x = nn.drop(x)
            if self.batchNorm:
                x = self.bn[i](x)
            x = self.layers[i](x)


        if self.finalActivation is not None:
            x=self.finalActivation(x)
        #print("f",x.size())
        return x

class NN(nn.Module):
    def __init__(self, inSize, outSize, layers=[]):
        super(NN, self).__init__()
        self.layers = nn.ModuleList([])
        for x in layers:
            self.layers.append(nn.Linear(inSize, x))
            inSize = x
        self.layers.append(nn.Linear(inSize, outSize))



    def forward(self, x):
        x = self.layers[0](x)
        for i in range(1, len(self.layers)):
            x = torch.tanh(x)
            x = self.layers[i](x)

        return x


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x

class LogMe(dict):
    def __init__(self,writer,term=True):
        self.writer = writer
        self.dic = defaultdict(list)
        self.term = term
    def write(self,i):
        if len(self.dic)==0: return
        s=f"Epoch {i} : "
        for k,v in self.dic.items():
            self.writer.add_scalar(k,sum(v)*1./len(v),i)
            s+=f"{k}:{sum(v)*1./len(v)} -- "
        self.dic.clear()
        if self.term: logging.info(s)
    def update(self,l):
        for k,v in l:
            self.add(k,v)
    def direct_write(self,k,v,i):
        self.writer.add_scalar(k,v,i)
    def add(self,k,v):
        self.dic[k].append(v)

def save_src(path):
    current_dir = os.getcwd()
    package_dir = current_dir.split('RL', 1)[0]
    #path = os.path.abspath(path)
    os.chdir(package_dir)
    #print(package_dir)
    src_files = subprocess.Popen(('find', 'RL', '-name', '*.py', '-o', '-name', '*.yaml'),
                                 stdout=subprocess.PIPE)
    #print(package_dir,path)
    #path=os.path.abspath(path)


    #print(str(src_files))

    subprocess.check_output(('tar', '-zcf', path+"/arch.tar", '-T', '-'), stdin=src_files.stdout, stderr=subprocess.STDOUT)
    src_files.wait()
    os.chdir(current_dir)


def draw(scores, path="fig.png", title="Performance", xlabel="Episode #", ylabel="Score"):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title(title)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(path) 


def prs(*args):
    st = ""
    for s in args:
        st += str(s)
    print(st)


class DotDict(dict):
    """dot.notation access to dictionary attributes (Thomas Robert)"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_yaml(path):
    with open(path, 'r') as stream:
        opt = yaml.load(stream,Loader=yaml.Loader)
    return DotDict(opt)

def write_yaml(file,dotdict):
    d=dict(dotdict)
    with open(file, 'w', encoding='utf8') as outfile:
        yaml.dump(d, outfile, default_flow_style=False, allow_unicode=True)


class EpsilonGreedyDecay:
    def __init__(self, epsilon, eta, epsilon_min):
        self.eta = eta
        self.epsilon=epsilon
        self.epsilon_min=epsilon_min
    def act(self, episode, q_values):
        decay = self.epsilon / (1 + (self.eta * episode))
        if decay<self.epsilon_min:
            decay=self.epsilon_min
        if np.random.random() > decay:
            _,action = torch.max(q_values,0) # we take the action that maximize the q_value
            return action.item()
        return  np.random.randint(len(q_values))




class DDPGAgent:
 
    def __init__(self,
                 state_dim,
                 action_dim,
                 lr_actor = 1e-4,
                 lr_critic = 1e-4,
                 lr_decay = .95,
                 replay_buff_size = 10000,
                 gamma = .9,
                 batch_size = 128,
                 random_seed = 42,
                 soft_update_tau = 1e-3
                 ):

        self.lr_actor = lr_actor
        self.gamma = gamma
        self.lr_critic = lr_critic
        self.lr_decay = lr_decay
        self.tau = soft_update_tau
        self.actor_local = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)     
        self.critic_local = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.lr_critic)     
        self.noise = OUNoise(action_dim, random_seed)
        self.memory = ReplayBuffer(action_dim, replay_buff_size, batch_size, random_seed)
  
        
    def update_model(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        if not self.memory.is_ready():
            return
       
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones)).detach()
        Q_expected = self.critic_local(states, actions)
        y = Q_targets.mean().item()
        critic_loss = F.smooth_l1_loss(Q_expected, Q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)   
        
    def act(self, state, noise_t=0.0):
        if random.random()<0.1:
          v=random.random()
          return np.array([v,1-v])
        if len(np.shape(state)) == 1:
            state = state.reshape(1,-1)
        state = torch.from_numpy(state).float()
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        action += self.noise.sample() * noise_t
        return np.clip(action, -1, 1).squeeze()
    
    def reset(self):
        self.noise.reset()
        
    def soft_update(self, local_model, target_model, tau):

        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            

class OUNoise:

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):

        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):

        self.state = copy.copy(self.mu)

    def sample(self):

        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:


    def __init__(self, action_size, buffer_size, batch_size, seed):

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):

        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
  
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)
    
    def is_ready(self):
        return len(self.memory) > self.batch_size

    def __len__(self):

        return len(self.memory)
