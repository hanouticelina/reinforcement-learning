


import matplotlib
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
from value_iteration import *
from policy_iteration import *
import copy
import sys
import matplotlib.pyplot as plt


MAX_ITER = 10000


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()


def run(plan, episode_count,mode, alpha, gamma,rewards_dst = {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1}):


    env = gym.make("gridworld-v0")
    env.seed(0)  # Initialise le seed du pseudo-random

    outdir = 'gridworld-v0/random-agent-results'
    envm = wrappers.Monitor(env, directory=outdir, force=True, video_callable=False)
    env.setPlan("gridworldPlans/plan" +plan+  ".txt", rewards_dst)

    statedic, mdp = env.getMDP()  # recupere le mdp , statedic
    # policy : { state : action}
    if mode == "PolicyIteration":
        agent = PolicyIterationAgent(statedic, mdp,  alpha, gamma)
    else:
        agent = ValueIterationAgent(statedic, mdp,  alpha, gamma )


    env.seed()  # Initialiser le pseudo aleatoire
    episode_count = episode_count
    reward = 0
    done = False
    rewards = []
    iterations = []
    FPS = 0.0001

    for i in range(episode_count):
        obs = envm.reset()
        env.verbose = False#(i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
        if env.verbose:
            env.render(FPS)
        j = 0
        rsum = 0
        while True:
            action = agent.act(obs, reward, done)
            obs, reward, done, _ = envm.step(action)
            rsum += reward
            j += 1
            if env.verbose:
                env.render(FPS)
            if done:
                print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
                break
        iterations.append(j)
        rewards.append(rsum)
    print("done")
    env.close()
    return np.arange(1, episode_count+1),np.array(rewards),np.array(iterations)

rewards_dst = {0: -0.001, 3: 1, 4: 0.5, 5: -1, 6: -1}
episodes, rewards_value, iterations_value = run(plan= "6", episode_count= 700,mode="PolicyIteration", alpha=1e-3, gamma= 0.99,rewards_dst=rewards_dst)
