"""Evaluate.py for pretraining."""
"""Evaluates the model, paired with train.py"""

# Imports
import pathlib2 as Path
import logging

import numpy as np
import torch

from utils import utils
from model import modules

def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on num_steps batches

    Args:
        model: (nn.Module) the neural network.
        loss_fn: function that takes y_pred  and labels and computs loss.
        dataloader: (torch.utils.data.Dataloader), contains samples and lbls. 
        metrics: (dict) dict of functions to compute metrics. 
        params: (Params) dict of hyperparams.
        num_steps: (int) number of batches to train on. 

    """

    # Set model to eval mode. 
    model.eval()

    # Summary for current eval loop
    summ = []

    # Compute metrics over dataset
    for x, y in dataloader:

        # Move to GPU if available.
        if params.cuda:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

        # Output
        y_pred = model(x, k=4)
        loss = loss_fn(y_pred, y)

        # Detach for metrics. 
        x = x.detach().numpy()
        y = y.detach().numpy()

        # Compute metrics.
        batch_summary = {metric: metrics[metric](x, y) for metric in metrics}
        batch_summary['loss'] = loss.item()
        summ.append(batch_summary)

    metrics_mean = {metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metric_string = ' ; '.join(f'{k}: {v:05.3f}'for k, v in metrics_mean.items())

    logging.info('- Eval metrics: '+ metric_string)

    return metrics_mean

if __name__ == '__main__':
    print('\nINFO: Please run via train.py until standalone functionality is added.\n')

