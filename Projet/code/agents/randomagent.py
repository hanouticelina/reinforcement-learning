import torch
from torch import nn
import numpy as np


class RandomAgent():
    """ A random agent """
    def __init__(self):
        pass

    def act(self,info):
        return int(np.random.choice(2,1)[0])

    def play(self, obs, configuration):
        print(obs)
        with torch.no_grad():
            return self.model(torch.tensor(obs).view(1,2).float()).argmax().item()
