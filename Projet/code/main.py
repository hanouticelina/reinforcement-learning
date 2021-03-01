import pickle
import sys
import random
import logging

try:
    """ For kaggle environment, add to module path the local directory """
    sys.path.append("/kaggle_simulations/agent")
except:
    pass


## Load the local model
try:
    with open("agents/QAgent.pkl","rb") as f:
        model = pickle.load(f)
except:
    """ If played in kaggle environment """
    with open("/kaggle_simulations/agent/agents/QAgent.pkl","rb") as f:
        model = pickle.load(f)
    
import time
def agent(obs,config):
    return model.play(obs,config)


