import pickle
import sys
import os
import kaggle_environments as kg
import numpy as np
import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter

try:
    """ For kaggle environment """
    sys.path.append("/kaggle_simulations/agent")
except:
    pass

from agents.randomagent import RandomAgent
from agents.mirrorOpponentAgent import  MirrorOpponentAgent
from agents.greedyAgent import  GreedyAgent
from agents.DQNAgent import DQNAgent
from agents.QAgent import QAgent

env = kg.make("rps",debug=True, 
        configuration = { 
            "actTimeout" : 1,
            "agentTimeout": 60,
            "runTimeout" : 1200 })

agent1 =  DQNAgent()
agent2 =  GreedyAgent()
env.train([None,GreedyAgent])
### Do the training

episode_count = 300

info = env.reset()



outdir="XP/"
logger = LogMe(SummaryWriter(outdir))
loadTensorBoard(outdir)

rsum = 0
radvsum = 0
mean = 0
itest = 0
reward = 0
done = False
for i in range(episode_count):
    j=0
    while True:

        action_1 = agent1.act(info[0].observation)
        action_2 =  agent2.act(info)
        info = env.step([action_1,action_2])
        radvsum += info[1].observation.reward
        rsum += info[0].observation.reward
        logger.direct_write("reward", rsum, i)
        j+=1
        done = (info[0].status == 'DONE')
        if done:
            print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions , " +str(radvsum))
            logger.direct_write("reward", rsum, i)
            rsum = 0
            radvsum =0
            info = env.reset()
            j=0
            break

with open("agents/DQNAgent.pkl","wb") as f:
        pickle.dump(agent1, f)
