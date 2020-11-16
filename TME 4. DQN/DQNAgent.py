import argparse
import sys
import matplotlib
import gym
import gridworld
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

    def __init__(self, env, opt):
        self.opt=opt
        self.env=env
        self.featureExtractor = opt.featExtractor(self.env)
        self.action_space = self.env.action_space # right or left
        self.learning_rate = opt.lr 
        self.batchsize = opt.batchsize
        self.discount = opt.discount
        self.epsilon = opt.epsilon
        self.capacity= opt.capacity

        self.activation = opt.activation
        self.Q = NN_Q(self.featureExtractor.outSize,outSize=self.action_space.n,activation=self.activation,layers=opt.layers).to(device)
        self.optim = torch.optim.Adam(self.Q.parameters(),lr=self.learning_rate)   
        self.Q_hat_update_step = opt.Q_hat_update_step # Step before Q_hat = Q 
        #self.D = deque(maxlen=opt.capacity) # transition history 
        self.D = Memory(opt.capacity,prior=opt.prior) # with prior
        self.Q_hat = NN_Q(self.featureExtractor.outSize,outSize=self.action_space.n,activation=self.activation,layers=opt.layers).to(device)
        self.Q_hat.load_state_dict(self.Q.state_dict())
        self.loss = nn.SmoothL1Loss()
      
        self.count=0
        self.episode=0
        self.lastobs = None 
        self.lasta = None
        
        self.epsilonGreedyDecay =  EpsilonGreedyDecay(self.epsilon,opt.eta,opt.epsilon_min)
        #self.memory_intialization()
        
        
    def act(self, observation, reward, done):
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
    
    # didn't use it
    def memory_intialization(self):
        ob0 = self.env.reset()
        for i in range(self.capacity):
            print(i)
            action = np.random.randint(self.action_space.n) 
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
        
            


if __name__ == '__main__':
    #config = load_yaml('./configs/config_random_gridworld.yaml')
    config = load_yaml('./configs/config_random_cartpole.yaml')
    #config = load_yaml('./configs/config_random_lunar.yaml')

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]

    env = gym.make(config["env"])
    if hasattr(env, 'setPlan'):
        env.setPlan(config["map"], config["rewards"])

    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/random_" + "-" + tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()

    agent = DQNAgent(env,config)
   
    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    logger = LogMe(SummaryWriter(outdir))
    loadTensorBoard(outdir)

    rsum = 0
    mean = 0
    verbose = False
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        if i % int(config["freqVerbose"]) == 0 and i >= config["freqVerbose"]:
            verbose = True
        else:
            verbose = False

        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False



        j = 0
        if verbose:
            env.render()

        while True:
            if verbose:
                env.render()

            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            j+=1

            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break

    env.close()
