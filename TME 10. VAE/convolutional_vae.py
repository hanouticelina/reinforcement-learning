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
    def __init__(self,in_channels, n_filters, latent_dims):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=n_filters, out_channels=n_filters*2, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc_mu = nn.Linear(in_features=n_filters*2*7*7, out_features=latent_dims)
        self.fc_logvar = nn.Linear(in_features=n_filters*2*7*7, out_features=latent_dims)
            
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        return x_mu, x_logvar

class Decoder(nn.Module):
    def __init__(self, in_channels, n_filters, latent_dims):
        super(Decoder, self).__init__()
        self.n_filters = n_filters
        self.fc = nn.Linear(in_features=latent_dims, out_features=n_filters*2*7*7)
        self.conv2 = nn.ConvTranspose2d(in_channels=n_filters*2, out_channels=n_filters, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=n_filters, out_channels=1, kernel_size=4, stride=2, padding=1)
            
    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), self.n_filters*2, 7, 7) # unflatten batch of feature vectors to a batch of multi-channel feature maps
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv1(x)) # last layer before output is sigmoid, since we are using BCE as reconstruction loss
        return x
    
class ConvVAE(nn.Module):
    def __init__(self, in_channels=1, n_filters=64,latent_dims=2):
        super(ConvVAE, self).__init__()
        self.encoder = Encoder(in_channels, n_filters, latent_dims)
        self.decoder = Decoder(in_channels, n_filters, latent_dims)
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent)
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
    
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss +  kldivergence