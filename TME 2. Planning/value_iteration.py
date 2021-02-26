import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt

class ValueIterationAgent(object):
    """agent that folows value ietration policy"""
    def __init__(self, statedic, mdp, epsilon = 0.01, gamma= 0.95):
        self.policy, self.value,_ = valueIteration(statedic, mdp, epsilon, gamma)
    def act(self, observation, reward, done):
        return self.policy[str(observation.tolist())]

def valueIteration(stateDic, mdp, epsilon = 0.01, gamma= 0.95):
    V = {state: np.random.rand() if state in mdp.keys() else 0 for state in stateDic.keys()}
    optimal = False
    nb_iter = 0
    while(not optimal):
        nb_iter+=1
        v = V.copy()
        for s in mdp.keys():
            V[s] = max([sum([proba * (reward + gamma * v[futureState]) for proba, futureState, reward, done in mdp[s][a]]) for a in mdp[s]])
        if np.linalg.norm(np.array(list(v.values()))-np.array(list(V.values()))) < epsilon:
            optimal = True
    pi = {}
    for state in mdp.keys():
        pi[state] = np.argmax( [ sum([proba * (reward + gamma * V[futureState]) for proba, futureState, reward, done in mdp[state][a]]) for a in mdp[state]])
    return pi, V, nb_iter
