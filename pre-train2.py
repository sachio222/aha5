# Copyright 2020 Jacob Krajewski
#
#
"""Description"""
# Imports
import argparse
import platform
from pathlib2 import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchvision import transforms
from torchvision.datasets import Omniglot
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from tqdm import tqdm

# User modules
from model import modules  # pylint: disable=no-name-in-module
from utils import utils  # pylint: disable=RP0003, F0401

# Constants


def make_dataset():
    """
    """
    # Define transforms
    tsfm = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize(params.resize_dim),
        transforms.ToTensor()
    ])

    # Import from torchvision.datasets Omniglot
    dataset = Omniglot(data_path,
                       background=True,
                       transform=tsfm,
                       download=True)

    dataloader = DataLoader(dataset,
                            params.batch_size,
                            shuffle=True,
                            num_workers=params.num_workers,
                            drop_last=True)


def load_model():
    # Load visual cortex model here.
    model = modules.ECPretrain(D_in=1,
                               D_out=121,
                               KERNEL_SIZE=9,
                               STRIDE=5,
                               PADDING=1)

    # Set loss_fn to Binary cross entropy for Autoencoder.
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    if load_pretrained:
        # Get last trained weights. COMMENT OUT if not wanted
        utils.load_checkpoint(model_path, model, optimizer, name="pre_train")


def train(model, dataloader, optimizer, loss_fn, params, autosave=True):
    # Set model to train or eval.
    model.train()

    for epoch in range(params.num_epochs):

        loss_avg = utils.RunningAverage()
        desc = "Epoch: {}".format(epoch)  # Informational only, used in tqdm.

        with tqdm(desc=desc, total=len(dataloader)) as t:
            for i, (x, _) in enumerate(dataloader):
                if params.cuda:
                    x, _ = x.cuda(non_blocking=True)

                y_pred = model(x, k=1)

                # Set loss comparison to input x
                loss = loss_fn(y_pred, x)

                optimizer.zero_grad()
                loss.backward()

                #=====MONITORING=====#

                enc_weights = model.encoder.weight.data
                # utils.animate_weights(enc_weights, label=i, auto=False)
                # for s in range(len(x)):
                #     utils.animate_weights(y_pred[s].detach(), label=i, auto=True)

                #=====END MONIT.=====#

                optimizer.step()
                loss_avg.update(loss.item())

                # Update tqdm progress bar.
                t.set_postfix(loss="{:05.8f}".format(loss_avg()))
                t.update()

            # Show one last time
            # utils.animate_weights(enc_weights, auto=False)

        if autosave:
            # Autosaves latest state after each epoch (overwrites previous state)
            state = utils.get_save_state(epoch, model, optimizer)
            utils.save_checkpoint(state,
                                  model_path,
                                  name="pre_train",
                                  silent=False)

        # grid_img = torchvision.utils.make_grid(y_pred, nrow=8)
        # plt.imshow(grid_img.detach().numpy()[0])
        # plt.show()


def main():
    # check_os()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", help="relative path to data")
    parser.add_argument("--model", help="")
    parser.add_argument("--json", help="")
    parser.add_argument("--seed", help="")

    args = parser.parse_args()

    # Initialize paths to json parameters
    data_path = Path().absolute() / "data"
    model_path = Path().absolute() / "experiments/pretrain/"
    json_path = model_path / "params.json"

    assert json_path.is_file(
    ), f"\n\nERROR: No params.json file found at {json_path}\n"
    params = utils.Params(json_path)

    # If GPU, write to params file
    params.cuda = torch.cuda.is_available()

    seed = args.seed or params.seed
    # Set random seed
    torch.manual_seed(seed)
    if params.cuda:
        torch.cuda.manual_seed(seed)
        # Update num_workers to 2 if running on GPU
        params.num_workers = 2

    train(model, dataloader, optimizer, loss_fn, params, autosave=True)


if __name__ == '__main__':
    clear_terminal()
    main()
