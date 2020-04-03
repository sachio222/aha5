# Imports
import argparse
from pathlib2 import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms
from torchvision.datasets import Omniglot

import matplotlib.pyplot as plt
from tqdm import tqdm

import wandb

# User modules
from model import modules  # pylint: disable=no-name-in-module
from utils import utils, exp  # pylint: disable=RP0003, F0401


def make_dataset():
    """"""
    tsfm = transforms.Compose([
        transforms.Resize(params.resize_dim),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])

    dataset = Omniglot(data_path,
                       background=True,
                       transform=tsfm,
                       download=True)

    dataloader = DataLoader(dataset,
                            params.batch_size[0],
                            shuffle=True,
                            num_workers=params.num_workers,
                            drop_last=True)
    
    if not params.silent:
        print('OK: Data loaded successfully.')
    
    return dataloader


def load_model():
    model = modules.ECToCA3(D_in=1, D_out=121)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate[0])

    if params.load:
        # Get last trained weights.
        try:
            utils.load_checkpoint(params.model_path, model, optimizer, name="pre_train")
            if not params.silent:
                print('OK: Loaded weights successfully.')
        except Exception:
            print('WARNING: --load request failed. Continue without pre-trained weights?', end=' ')
            choice = input('y/n: ')
            if choice.lower() == 'y':
                print('Wise choice')
            elif choice.lower() == 'n':
                print('Sucka! Try running again without the --load flag.')
                exit()
            else:
                print('try again later.')
                exit()
            
    return model, loss_fn, optimizer

def train(model, dataloader, optimizer, loss_fn):
    print(params.autosave)

utils.clear_terminal()
aha = exp.Experiment()
params = aha.get_params()
json_path, data_path, model_path = aha.get_paths()

def main():
    
    # If GPU
    params.cuda = torch.cuda.is_available()

    # Set random seed
    seed = params.seed
    torch.manual_seed(seed)
    if params.cuda:
        torch.cuda.manual_seed(seed)
        params.num_workers = 2

    dataloader = make_dataset()
    model, loss_fn, optimizer = load_model()
    wandb.watch(model)

    train(model, dataloader, optimizer, loss_fn)


if __name__ == '__main__':
    main()
