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
from utils import *
from convolutional_vae import *
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model):

    with torch.no_grad():
    
        images = images.to(device)
        images, _, _ = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[1:50], 10, 5).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()

BATCH_SIZE = 128
TRANSFORMS = transforms.Compose([
transforms.ToTensor(),
])
mnist_trainset = torchvision.datasets.MNIST("/tmp/mnist", train=True, transform=TRANSFORMS, target_transform=None, download=True)
mnist_trainset, mnist_valset = random_split(mnist_trainset,
                                            (int(0.9 * len(mnist_trainset)),
                                             int(0.1 * len(mnist_trainset))))
mnist_testset = torchvision.datasets.MNIST("/tmp/mnist", train=False, transform=TRANSFORMS, target_transform=None, download=True)


train_loader = torch.utils.data.DataLoader(mnist_trainset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_testset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

latent_dims = 2
num_epochs = 20
batch_size = 128
capacity = 64
learning_rate = 0.001
vae = ConvVAE()
vae = vae.to(device)

num_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
print('Number of parameters: %d' % num_params)
optimizer = torch.optim.Adam(params=vae.parameters())

vae.train()

train_loss_avg = []

print('Training ...')
for epoch in range(num_epochs):
    train_loss_avg.append(0)
    num_batches = 0
    for image_batch, _ in train_loader:
        image_batch = image_batch.to(device)
        image_batch_recon, latent_mu, latent_logvar = vae(image_batch)
        loss = vae_loss(image_batch_recon, image_batch, latent_mu, latent_logvar)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_avg[-1] += loss.item()
        num_batches += 1
        
    train_loss_avg[-1] /= num_batches
    print('Epoch [%d / %d] average reconstruction error: %f' % (epoch+1, num_epochs, train_loss_avg[-1]))
    
    
vae.eval()

images, labels = iter(test_loader).next()
print('Original images')
show_image(torchvision.utils.make_grid(images[1:50],10,5))
plt.show()
print('VAE reconstruction:')
visualise_output(images, vae)