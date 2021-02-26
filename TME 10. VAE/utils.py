import itertools
import logging
from tqdm import tqdm
from pathlib import Path
import os
import yaml
from datamaestro import prepare_dataset
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from typing import List
import time
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

from vae import *


class CheckpointState():
    """A model checkpoint state."""
    def __init__(self, model, optimizer=None, epoch=1, savepath='./checkpt.pt'):

        self.model = model
        self.optimizer = optimizer
        self.epoch = epoch
        self.savepath = Path(savepath)

    def state_dict(self):
        """Checkpoint's state dict, to save and load"""
        dict_ = dict()
        dict_['model'] = self.model.state_dict()
        if self.optimizer:
            dict_['optimizer'] = self.optimizer.state_dict()
        dict_['epoch'] = self.epoch
        return dict_

    def save(self, suffix=''):
        """Serializes the checkpoint.
        Args:
            suffix (str): if provided, a suffix will be prepended before the extension
                of the object's savepath attribute.
        """
        if suffix:
            savepath = self.savepath.parent / Path(self.savepath.stem + suffix +
                                                   self.savepath.suffix)
        else:
            savepath = self.savepath
        with savepath.open('wb') as fp:
            torch.save(self.state_dict(), fp)

    def load(self):
        """Deserializes and map the checkpoint to the available device."""
        with self.savepath.open('rb') as fp:
            state_dict = torch.load(
                fp, map_location=torch.device('cuda' if torch.cuda.is_available()
                                              else 'cpu'))
            self.update(state_dict)

    def update(self, state_dict):
        """Updates the object with a dictionary
        Args:
            state_dict (dict): a dictionary with keys:
                - 'model' containing a state dict for the checkpoint's model
                - 'optimizer' containing a state for the checkpoint's optimizer
                  (optional)
                - 'epoch' containing the associated epoch number
        """
        self.model.load_state_dict(state_dict['model'])
        if self.optimizer is not None and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        self.epoch = state_dict['epoch']
        
        
        
def create_grid(network, im, writer=None, epoch=None, mode="FC"):
    with torch.no_grad():
        if mode =="FC":
            img = im.view((im.shape[0], -1))
        else:
            img = im
        reconstructed_im, _, _ = network(img)
        reconstructed_im = reconstructed_im.reshape(im.shape)
        im = im.repeat(1, 3, 1, 1)
        reconstructed_im = reconstructed_im.repeat(1, 3, 1, 1)

        images = torch.cat((im, reconstructed_im), 0)
        grid_img = make_grid(images, nrow=len(im))

    if writer:
        assert epoch is not None
        writer.add_image('Epoch {}'.format(epoch),
                     grid_img, epoch)
    else:
        return grid_img

def get_dataloaders(batch_size):
    
    TRANSFORMS = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
    ])
    mnist_trainset = torchvision.datasets.MNIST("/tmp/mnist", train=True, transform=TRANSFORMS, target_transform=None, download=True)
    mnist_trainset, mnist_valset = random_split(mnist_trainset,
                                                (int(0.9 * len(mnist_trainset)),
                                                 int(0.1 * len(mnist_trainset))))
    mnist_testset = torchvision.datasets.MNIST("/tmp/mnist", train=False, transform=TRANSFORMS, target_transform=None, download=True)

    

    train_loader = torch.utils.data.DataLoader(mnist_trainset,
                                               batch_size=batch_size,
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(mnist_valset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_testset,
                                              batch_size=batch_size,
                                              shuffle=True)

    return train_loader, valid_loader, test_loader

def main():
    train_loader, valid_loader, test_loader = get_dataloaders()
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
    fit(checkpoint, train_loader, valid_loader, device,args['epochs'], writers=writers)
    writer_loss.close()
    writer_images.close()
    
    
def embedding_visu(model, loader,device, writer):
    
    pbar = tqdm(loader, dynamic_ncols=True,  position=0, leave=True)  # progress bar
    with torch.no_grad():
        for i, (batch_x,batch_y) in enumerate(pbar):
            x = batch_x
            batch_x = batch_x.view((batch_x.shape[0], -1)).to(device)
            mu, log_var = model.encoder(batch_x)
            z = model.sample(mu, log_var)
            class_labels = [classes[label] for label in batch_y]
            writer_embed.add_embedding(z, metadata=class_labels, label_img=x, global_step=i)
    writer_embed.close()