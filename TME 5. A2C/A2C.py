import argparse
import sys
import matplotlib
import matplotlib.pyplot as plt
import gym
import gridworld
import torch
from utils import *
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from gym import wrappers, logger
import numpy as np
import copy
from collections import deque # we use a deque as a memory ((st,at,st1,rt,done))
from torch import nn
import time
from memory import *
import copy
from torch.optim import SGD, Adam
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class A2C(object):
    def __init__(self, env, opt, buffer_size, target_hat, actor_lr, critic_lr, n_layers, dim_layers):
        self.action_space = env.action_space
        self.buffer_size = buffer_size
        self.loss = torch.nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(self.buffer_size)
        self.D = Memory(opt.capacity,prior=opt.prior) # with prior
        self.discount_factor = 0.99
        self.featureExtractor = opt.featExtractor(env)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.last_state = None
        self.last_action = None
        self.Actor = NN_Actor(self.featureExtractor.outSize, self.action_space.n, [dim_layers for i in range(n_layers)])
        self.Critic = NN_Critic(self.featureExtractor.outSize, 1, [dim_layers for i in range(n_layers)])
        self.Critic_target = copy.deepcopy(self.Critic)
        self.target_hat = target_hat
        self.optim_A = Adam(self.Actor.parameters(), lr=self.actor_lr, weight_decay=1e-1)
        self.optim_C = Adam(self.Critic.parameters(), lr=self.critic_lr)
        self.batchsize = opt.batchsize
        self.count = 0
        self.device = "cpu"
    

    def act(self, observation, reward, done):
        observation = torch.tensor(self.featureExtractor.getFeatures(observation), dtype = torch.float)
        action_scores = self.Actor(observation)
        
        if self.last_state == None:
            action = self.action_space.sample()
            self.last_state = observation
            self.last_action = action
            return self.last_action

        action = np.argmax(action_scores.detach().numpy())
        self.D.store((self.last_state, self.last_action, reward, observation, done))
        if self.D.mem_ptr == self.buffer_size :
            _, _, tuples = self.D.sample(self.buffer_size)
            states,actions,rewards,next_states,dones = map(list,zip(*tuples)) 
            rewards=torch.Tensor(rewards).to(device) # rewards tensor
            dones=torch.BoolTensor(dones).to(device) # done's tensor
            states=torch.Tensor(states).to(device) # initial states tensor
            next_states=torch.Tensor(next_states).to(device) # next states tensor
            tuples = states, actions, rewards, next_states, dones
            critic_loss = self.update_critic(*tuples)
            actor_loss = self.update_actor(*tuples)
            self.optim_A.zero_grad()
            self.optim_C.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.optim_A.step()
            self.optim_C.step()
            self.Critic_target = copy.deepcopy(self.Critic)
            self.count += 1
        self.last_state = observation
        self.last_action = action
        return action
    def update_actor(self, states, actions, rewards, next_states, dones):
        advantages = rewards + (1.0 - dones) * self.discount_factor * self.Critic(next_states).squeeze() - self.Critic(states).squeeze()
        action_masks = F.one_hot(actions, self.action_space.n)
        probas = self.Actor(states)
        masked_log_proba = (action_masks * torch.log(probas)).sum(dim=-1)
        actor_loss = torch.sum(masked_log_proba*advantages.detach())
        return actor_loss
    def update_critic(self,states, actions, rewards, next_states, dones):
        V = self.Critic(states).squeeze()
        with torch.no_grad():
            next_V = self.Critic_target(next_states).squeeze()
            target = rewards + (1 - dones)*self.discount_factor * next_V
        loss = self.loss(V, target.detach())
        return loss
    def replay(self,)
    def save(self,outputDir):
        pass
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
    buffer_size = 100
    dim_layers = 64
    n_layers = 3
    actor_lr, critic_lr = 1e-3, 1e-3
    target_hat = 1
    loss_num = 1
    tstart = str(time.time())
    tstart = tstart.replace(".", "_")
    outdir = "./XP/" + config["env"] + "/Grid_actor_critic/" + "_bs" + str(buffer_size) + "_dim" +str(dim_layers) +"_num" +str(n_layers)+"_actorlr"+str(actor_lr)+"_criticlr"+str(critic_lr)+"_loss"+str(loss_num)+tstart


    env.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])

    episode_count = config["nbEpisodes"]
    ob = env.reset()
   
    agent = A2C(env,config,buffer_size, target_hat, actor_lr, critic_lr, n_layers,dim_layers)
    log = False

    print("Saving in " + outdir)
    os.makedirs(outdir, exist_ok=True)
    save_src(os.path.abspath(outdir))
    write_yaml(os.path.join(outdir, 'info.yaml'), config)
    if log:
        logger = LogMe(SummaryWriter(outdir))
        loadTensorBoard(outdir)

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
            #print("End of test, mean reward=", mean / nbTest)
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

            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            cur_frame+=1
            j+=1

            rsum += reward
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                if log:
                    logger.direct_write("reward", rsum, i)
                train_reward.append(rsum)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break

    env.close()
    train_rg = np.arange(len(train_reward))
    test_rg = np.arange(len(test_reward))
    fig, ax = plt.subplots(1,2, figsize=(10,5))
    ax[0].plot(train_rg,train_reward)
    ax[1].plot(test_rg,test_reward)
    fig.savefig("reward.png")
    plt.close(fig)

    