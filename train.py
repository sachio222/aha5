
# Imports
from pathlib2 import Path
import logging
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
# wandb.init(entity="redtailedhawk", project="aha")

# User modules
from model import modules  # pylint: disable=no-name-in-module
from utils import utils  # pylint: disable=RP0003, F0401

# Clear terminal
utils.clear_terminal()

# pylint: disable=no-member

# Constants
my_system = utils.check_os()

# Instantiate
aha = utils.Experiment()
params = aha.get_params()
json_path, data_path, model_path = aha.get_paths()


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
    # model = modules.ECToCA3(D_in=1, D_out=121)
    model = modules.ECPretrain(D_in=1,
                               D_out=121,
                               KERNEL_SIZE=9,
                               STRIDE=5,
                               PADDING=1)

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate[2])

    if params.load:
        # Get last trained weights.
        try:
            utils.load_checkpoint(params.model_path,
                                  model,
                                  optimizer,
                                  name="pre_train")
            if not params.silent:
                print('OK: Loaded weights successfully.')
        except Exception:
            print(
                'WARNING: --load request failed. Continuing without pre-trained weights.'
            )
            pass

    return model, loss_fn, optimizer


def train(model, dataloader, optimizer, loss_fn):
    if not params.silent:
        print('\n[---TRAINING START---]')
    model.train()

    for epoch in range(params.num_epochs):

        loss_avg = utils.RunningAverage()

        desc = "Epoch: {}".format(epoch)  # Informational only, used in tqdm.

        with tqdm(desc=desc, total=len(dataloader)) as t:
            for i, (x, _) in enumerate(dataloader):

                if params.cuda:  #if GPU
                    x, _ = x.cuda(non_blocking=True)

                y_pred = model(x, k=1)

                # Set loss comparison to input x
                loss = loss_fn(y_pred, x)

                optimizer.zero_grad()
                loss.backward()

                #=====MONITORING=====#

                enc_weights = model.encoder.weight.data

                if my_system.lower() != 'windows':
                    utils.animate_weights(enc_weights, label=i, auto=False)
                    for s in range(len(x)):
                        utils.animate_weights(y_pred[s].detach(),
                                              label=i,
                                              auto=True)

                #=====END MONIT.=====#

                optimizer.step()
                loss_avg.update(loss.item())

                # Update tqdm progress bar.
                t.set_postfix(loss="{:05.8f}".format(loss_avg()))
                t.update()

            # Show one last time
            if my_system.lower() != 'windows':
                utils.animate_weights(enc_weights, auto=False)

        wandb.log({"Train Loss": loss_avg()})

        if params.autosave:
            # Autosaves latest state after each epoch (overwrites previous state)
            state = utils.get_save_state(epoch, model, optimizer)
            utils.save_checkpoint(state,
                                  model_path,
                                  name="pre_train",
                                  silent=False)


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

    # wandb.watch(model)

    if not params.silent:
        print(f'AUTOSAVE: {params.autosave}')

    # Run training
    train(model, dataloader, optimizer, loss_fn)


if __name__ == '__main__':
    logging.info('testing logger main')
    main()
