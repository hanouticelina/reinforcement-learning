import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
import copy
import sys
import matplotlib.pyplot as plt


class PolicyIterationAgent(object):
    def __init__(self, statedic, mdp, epsilon = 0.01, gamma= 0.95):
        self.policy, self.value, _ = policyIteration(statedic, mdp, epsilon, gamma)
    def act(self, observation, reward, done):
        return self.policy[str(observation.tolist())]



def policyIteration(statedic, mdp, epsilon= 0.01, gamma = 0.95):
    PI ={ state : np.random.randint(0,4) for state in mdp.keys()}
    optimal = False
    nb_iter= 0
    while(not optimal):
        V = {state: np.random.rand() if state in mdp.keys() else 0  for state in statedic.keys()}
        converge = False
        while(not converge):
            nb_iter +=1
            v = V.copy()
            for state in mdp:
                V[state] = sum( [ proba * (reward + gamma * v[futureState]) for proba, futureState, reward, done in mdp[state][PI[state]] ] )
            if (np.linalg.norm( np.array(list(v.values())) -np.array(list(V.values())) ) < epsilon):
                converge = True
        pi = PI.copy()
        for state in mdp:
            PI[state] = np.argmax( [ sum([proba * (reward + gamma * V[futureState]) for proba, futureState, reward, done in mdp[state][a]]) for a in mdp[state]])
        if pi == PI:
            optimal = True
    return PI, V, nb_iter
