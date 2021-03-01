import random
from kaggle_environments.envs.rps.utils import get_score
import numpy as np

   

class GreedyAgent:
    def __init__(self):
        self.actionsCount = {}

    def act(self,info):
        if info[0].observation.step == 0:
            return int(np.random.choice(2,1)[0])
        action = info[0].observation.lastOpponentAction
        if action not in self.actionsCount:
            self.actionsCount[action] = 0
        self.actionsCount[action] += 1
        mode_action = None
        mode_action_count = None
        for k, v in self.actionsCount.items():
            if mode_action_count is None or v > mode_action_count:
                mode_action = k
                mode_action_count = v
                continue

        return mode_action

    def play(self, obs, configuration):
        return act(self,info)