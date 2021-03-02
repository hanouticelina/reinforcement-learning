import sys
import os
import kaggle_environments as kg

# Authorized libraries : Python Standard Library, gym, numpy, scipy, pytorch (cpu only)

import numpy as np
import torch
import matplotlib
import datetime
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from agents.mirrorOpponentAgent import  MirrorOpponentAgent
from agents.DQNAgent import DQNAgent
from agents.QAgent import QAgent
from utils import *

# Symbols :
# 0 : Rock
# 1 : Paper
# 2: Scissors

# Make the environment 
# * episodeSteps 	Maximum number of steps in the episode.
# * agentTimeout 	Maximum runtime(seconds) to initialize an agent.
# * actTimeout 	Maximum runtime(seconds) to obtain an action from an agent.
# * runTimeout 	Maximum runtime(seconds) of an episode(not necessarily DONE).


env = kg.make("rps",debug=True, 
        configuration = { 
            "actTimeout" : 1,
            "agentTimeout": 60,
            "runTimeout" : 1200 })


# Agents from kaggle:  rock, paper, scissors, copy_opponent, reactionary, counter_reactionary, statistical
print(*env.agents )

# Agent definition : function of two variabless :
# *  observation:  dict : 
# **      'remainingOverageTime': int
# **      'lastOpponentAction': 0
# **      'step': int
# * configuration : 
# **      'episodeSteps': int
# **      'agentTimeout': int
# **      'actTimeout': int
# **      'runTimeout': int
# **      'isProduction': boolean
# **      'signs': int 
# **      'tieRewardThreshold': int

def random_agent(observation, configuration):
   return np.random.randint(3)

# Loading agent from file 
agent1 = "main.py"
adv = "copy_opponent" #copy_opponent reactionary counter_reactionary statistical
# my_agent vs other agent

env.reset()

# Evaluate 
<<<<<<< Updated upstream
agents = [agent1, adv]
=======
agents = ["reactionary", "statistical"]
>>>>>>> Stashed changes
configuration = None
steps = 1000
num_episodes = 100
results = kg.evaluate('rps', agents,configuration,  num_episodes= num_episodes)
my_agent_rewards = np.array(results)[:,0]
adv_agent_rewards = np.array(results)[:,1]

############# ploting result
<<<<<<< Updated upstream
myagent = my_agent_rewards.cumsum()   
other = adv_agent_rewards.cumsum()
draw(myagent,other,agentName="Q Agent"\
   ,advAgentName="random Agent",title = "Rewards cumulés en evaluation")
=======
       
draw(my_agent_rewards.cumsum(),adv_agent_rewards.cumsum(),agentName="reactionary Agent"\
   ,advAgentName="statistical Agent",title = "Les rewards cumulés en evalutaion")
>>>>>>> Stashed changes
