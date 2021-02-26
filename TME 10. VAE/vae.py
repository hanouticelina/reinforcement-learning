import os
import yaml
import time
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import make_grid
from torchvision import datasets, transforms
from torch.utils.data import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile





class Encoder(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, h_dim)
        self.fc21 = nn.Linear(h_dim, z_dim)
        self.fc22 = nn.Linear(h_dim, z_dim)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc21(h)
        log_var = self.fc22(h)
        return mu, log_var
    
class Decoder(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        super().__init__()
        self.fc1 = nn.Linear(z_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, input_dim)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(h))
        return x  
    
    
class VAE(nn.Module):
    def __init__(self, input_dim, h_dim, z_dim):
        super().__init__()
        # encoder
        self.encoder = Encoder(input_dim, h_dim, z_dim)
        # decoder
        self.decoder = Decoder(input_dim, h_dim, z_dim)

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        distr = Normal(mu, std)
        z = distr.rsample()
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.sample(mu, log_var)
        return self.decoder(z), mu, log_var


def loss_function(x_hat, x, mu, log_var, mode="FC"):
    if mode != "FC":
        x_hat = x_hat.view(-1, 784)
        x = x.view(-1, 784)
    bce = F.binary_cross_entropy(x_hat, x, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return bce + kl