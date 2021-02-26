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
from vae import *
from convolutional_vae import *
from train import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import *
with open('./configs/vae.yaml', 'r') as stream:
    args  = yaml.load(stream,Loader=yaml.Loader)

Z_DIM = args['Z_DIM']
H_DIM = args['H_DIM']
LOSS_PATH = "losses/{}".format(Z_DIM)
IMAGES_PATH = "images/{}".format(Z_DIM)
EMBEDDING_PATH = "embeddings"
CHECKPOINT_PATH = "checkpoint/h={}_z={}".format(H_DIM, Z_DIM)
ngpu = torch.cuda.device_count()
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
savepath = Path('models/checkpt.pt')
writer_embed = SummaryWriter(EMBEDDING_PATH)

train_loader, valid_loader, test_loader = get_dataloaders(args['BATCH_SIZE'])
inputs, _ = next(iter(train_loader))
_, _, width, height = inputs.shape
input_dim = width * height
writer_loss = SummaryWriter(LOSS_PATH)
writer_images = SummaryWriter(IMAGES_PATH + '/train')
writer_test = SummaryWriter(IMAGES_PATH + '/test')
writers = {'loss': writer_loss, 'images': writer_images}

vae = VAE(input_dim=input_dim, h_dim=H_DIM, z_dim=Z_DIM).to(device)
optimizer = optim.Adam(vae.parameters())
checkpoint = CheckpointState(vae, optimizer, savepath=savepath)

fit(checkpoint, train_loader, valid_loader, device,args['epochs'], writers=writers, mode="FC")
writer_loss.close()
writer_images.close()
embedding_visu(checkpoint.model, train_loader,device, writer_embed)