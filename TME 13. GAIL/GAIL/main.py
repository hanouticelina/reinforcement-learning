import pickle
import argparse
import sys
import matplotlib
import datetime
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
from gail import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = "./expert_data/"
file = "expert.pkl"
path_to_expert_data  = path+file
n_rollout = 20
gamma = 0.99
delta = 0.8


# load the environment
config = load_yaml("./configs/config_random_lunar.yaml")

env = gym.make(config.env)
if hasattr(env, "setPlan"):
    env.setPlan(config.map, config.rewards)
tstart = str(time.time()).replace(".", "_")
env.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
episode_count = config.nbEpisodes
ob = env.reset()

# define the agent
agent_id = f"_g{gamma}_delta{delta}_beta{beta}"
agent_dir = f'models/{config["env"]}/'
os.makedirs(agent_dir, exist_ok=True)
savepath = Path(f"{agent_dir}{agent_id}.pch")
agent = GAIL(env, config, device, path_to_expert_data)
featureExtractor = config.featExtractor(env)
# ---yaml and tensorboard---#
outdir = "./runs/" + config.env + "/ppo/" + agent_id + "_" + tstart
print("Saving in " + outdir)
os.makedirs(outdir, exist_ok=True)
save_src(os.path.abspath(outdir))
write_yaml(os.path.join(outdir, "info.yaml"), config)
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
rsum = 0
mean = 0
verbose = True
itest = 0
reward = 0
done = False


for i in range(episode_count):
    if i % int(config.freqVerbose) == 0 and i >= config.freqVerbose:
        verbose = False
    else:
        verbose = False

    if i % config.freqTest == 0 and i >= config.freqTest:
        print("Test time! ")
        mean = 0
        agent.test = True

    if i % config.freqTest == config.nbTest and i > config.freqTest:
        print("End of test, mean reward=", mean / config.nbTest)
        itest += 1
        writer.add_scalar("rewardTest", mean / config.nbTest, itest)
        agent.test = False

    if i % config.freqSave == 0:
        with open(savepath, "wb") as f:
            torch.save(agent, f)
    j = 0
    if verbose:
        env.render()

    done = False
    while not (done):
        for _ in range(n_rollout):
            if verbose:
                env.render()
            action, prob = agent.get_action(
                ob, reward, done
            )  
            ob_new, reward, done, _ = env.step(action)  
            agent.buffer.add(
                featureExtractor.getFeatures(ob).reshape(-1), 
                action, float(prob), reward, 
                featureExtractor.getFeatures(ob_new).reshape(-1), done
            )   
            ob = ob_new
            j += 1
            rsum += reward
            if done:
                print(f"{i} rsum={rsum}, {j} actions")
                writer.add_scalar("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0
                ob = env.reset()
                break
        adv_loss, L, H, critic_loss, d_kl = agent.update()
    
    agent.epoch += 1
env.close()