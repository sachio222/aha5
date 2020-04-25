"""aha_v0.5"""

# Imports
import sys
from pathlib2 import Path
import logging
import numpy as np
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

# Buggy, I think it's them, not me.
try:
    import wandb
except:
    pass

# User modules
from evaluate import evaluate
from model import modules  # pylint: disable=no-name-in-module
from utils import utils  # pylint: disable=RP0003, F0401
"""Todo: store, access multisession train loss.
"""

# Clear terminal & Set logger variable
utils.clear_terminal()
logger = logging.getLogger(__name__)
utils.set_logger(logger)

# pylint: disable=no-member

# Constants
# Check OS
my_system = utils.check_os()


def make_dataset(params):
    """Returns DataLoader object"""

    tsfm = transforms.Compose([
        transforms.Resize(params.resize_dim),
        transforms.Grayscale(1),
        transforms.ToTensor()
    ])

    dataset = Omniglot(params.data_path,
                       background=params.background_set,
                       transform=tsfm,
                       download=True)

    # For individual module training.

    # TODO: Switch dataloaders depending on what tests are being run.

    # if params.pretrain:
    dataloader = DataLoader(dataset,
                            params.batch_size,
                            shuffle=True,
                            num_workers=params.num_workers,
                            drop_last=True)
    # else:
    #     train_dataloader, test_dataloader = train_test_split(dataset, params)

    if not params.silent:
        logger.info('Data loaded successfully.')

    return dataloader


def train_test_split(dataset, params):
    """Grabs random Omniglot samples and generates test samples from same class.

    The random seed is taken from params.sampler_seed, the test_shift is which sample
            to grab as a test. If it ends up being a different class, the sampler is
            walked back until the class is same, and the sample is different. 
        
    Args:
        dataset: (Dataset) Sampler from Omniglot dataset.
        params: (json dict) Params.json file. 

    Returns:
        train_dataloader, test_dataloader: (tuple) Containing matched train test pairs.
    """
    train_dataset = []
    test_dataset = []

    # Random seed from params file.
    torch.manual_seed(params.sampler_seed)

    # Create batch_size random indices from dataset.
    #       Subtract params.test_shift so that we don't pick a random sample
    #       so close to the end of the set that it looks for a test pair in
    #       the blackness of 'index out of range'.
    idxs = torch.randint(
        len(dataset) - params.test_shift, (1, params.batch_size))

    # Make sure one of them is our control.
    idxs[0, 0] = 19

    for i, idx in enumerate(idxs[0]):
        shift_idx = params.test_shift
        train_sample, train_lbl = dataset[idx]
        test_sample, test_lbl = dataset[idx + shift_idx]

        # Make sure labels are the same, and it is not the same sample.
        while (train_lbl != test_lbl) or (torch.equal(train_sample,
                                                      test_sample)):
            test_sample, test_lbl = dataset[idx + shift_idx]
            shift_idx -= 1

        train_dataset.append(train_sample)
        test_dataset.append(test_sample)
        #=====MONITORING=====#

        # Uncomment to see train_samples or change selection to test_sample.
        # utils.animate_weights(train_sample, auto=True)

        #=====END MONITORING=====#

    train_dataloader = torch.stack(train_dataset)
    train_dataloader.unsqueeze_(1)

    test_dataloader = torch.stack(test_dataset)
    test_dataloader.unsqueeze_(1)

    return train_dataloader, test_dataloader


def load_model(params):
    """Returns model, loss function and optimizer for training"""

    model = 'what'

    if model == 'pretrain':
        pass
    if model == 'train':
        pass
    if model == 'dg':
        pass
    if model == 'ecca3':
        pass
    if model == 'ca3':
        pass
    if model == 'ca1':
        pass
    else:
        print('what is it')

    model = modules.ECPretrain(D_in=1,
                               D_out=121,
                               KERNEL_SIZE=9,
                               STRIDE=5,
                               PADDING=1)

    # model = modules.ECToCA3

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # LOAD WEIGHTS
    # --------------------------

    if params.load:
        # Get last trained weights.
        try:
            utils.load_checkpoint(params.model_path,
                                  model,
                                  optimizer,
                                  name="pre_train")
            # if not params.silent:
            #     logger.info('Loaded weights successfully.')
        except Exception:
            logger.warning(
                '--load request failed. Continuing without pre-trained weights.'
            )
            pass

    # --------------------------

    return model, loss_fn, optimizer


def train(model, dataloader, optimizer, loss_fn, metrics, params):
    """
    Args:
        model: (nn.Module) the neural network.
        dataloader: (DataLoader) torch.utils.data.Dataloader object
        optimizer: (optim) optimizer for model params
        loss_fn: function. 
        metrics: (dict) of functions that compute metrics from output and labels
        params: (Params) hyperparameters
    """

    if not params.silent:
        print('\n[---TRAINING START---]')

    model.train()  # Set model to train mode

    for epoch in range(params.num_epochs):

        # Summary for this training loop, as well as avg loss.
        summ = []
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

                if params.animate:

                    if my_system.lower() != 'windows':
                        # For mac only
                        # Uncomment 1 of the following at a time to view kernels
                        #       while training:

                        # FULL VIEW
                        # --------------------------

                        utils.animate_weights(enc_weights, label=i, auto=True)

                        # --------------------------

                        # SINGLE VIEW
                        # --------------------------

                        # for s in range(len(x)):
                        #     utils.animate_weights(y_pred[s].detach(),
                        #                           label=i,
                        #                           auto=True)

                        # --------------------------

                    else:
                        '''Show full kernels on windows

                        Animation does not work with Windows, but each step can be
                            displayed manually.

                        '''

                        # FULL VIEW
                        # ------------------------- -

                        utils.animate_weights(enc_weights, label=i, auto=False)

                        # --------------------------

                    #=====END MONIT.=====#

                optimizer.step()

                # Evaluate summaries periodically (Should be own function)
                if i % params.save_summary_steps == 0:
                    y_pred = y_pred.detach().numpy()
                    x = x.detach().numpy()

                    # Compute all metrics on this batch
                    batch_summary = {
                        metric: metrics[metric](y_pred, x) for metric in metrics
                    }

                    batch_summary['loss'] = loss.item()
                    summ.append(batch_summary)

                # Update avg. loss after batch.
                loss_avg.update(loss.item())

                # Update tqdm progress bar.
                t.set_postfix(loss="{:05.8f}".format(loss_avg()))
                t.update()

            # Show one last time

            if params.showlast:
                utils.animate_weights(enc_weights, auto=False)

        # Compute mean of all metrics in summary.
        metrics_mean = {
            metric: np.mean([x[metric] for x in summ]) for metric in summ[0]
        }
        metrics_string = ' ; '.join(
            '{}: {:05.3f}'.format(k, v) for k, v in metrics_mean.items())
        logger.info('- Train metrics: ' + metrics_string)

        logger.info(f'Epoch: {epoch} - Train Loss: {loss_avg()}')

        if params.wandb:
            try:
                wandb.log({"Train Loss": loss_avg()})
            except:
                pass

        # SAVE WEIGHTS
        # --------------------------

        if params.autosave:
            # Autosaves latest state after each epoch (overwrites previous state)
            state = utils.get_save_state(epoch, model, optimizer)
            utils.save_checkpoint(state,
                                  params.model_path,
                                  name="pre_train",
                                  silent=False)

        # --------------------------


def main():

    # Create Experiment object
    aha = utils.Experiment()

    # Get params file updated from custom args
    params = aha.get_params()

    # Wandb Credentials
    if params.wandb:
        try:
            wandb.init(entity="redtailedhawk", project="aha")
        except:
            pass

    # If GPU
    params.cuda = torch.cuda.is_available()

    # Set random seed
    seed = params.seed
    torch.manual_seed(seed)
    if params.cuda:
        torch.cuda.manual_seed(seed)
        params.num_workers = 2

    dataloader = make_dataset(params)

    model, loss_fn, optimizer = load_model(params)

    metrics = modules.metrics

    if params.wandb:
        try:
            wandb.watch(model)
        except:
            pass

    if not params.silent:
        logger.info(
            f"Epochs: {params.num_epochs}, lr: {params.learning_rate}, batch_size: {params.batch_size}"
        )

    # Run training
    train(model, dataloader, optimizer, loss_fn, metrics, params)

    # test_metrics = evaluate(model, loss_fn, dataloader, metrics, params)
    
if __name__ == '__main__':
    main()
