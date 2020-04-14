"""Performs hyperparameter search"""

import argparse
import os
import subprocess import check_call

# User modules
from utils import utils

PYTHON = sys.executable
parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir',
                    default='experiments/learning_rate',
                    help='Directory containing params.json')

parser.add_argument('--data_dir',
                    default='data/omniglot-py',
                    help='Directory containing dataset')


def launch_training_job(parent_dir, data_dir, job_name, params):
    """Launch training of model with set of hyperparams in parent_dir/job_name

    Args:
        model_dir: (string) directory containing config, weights, log
        data_dir: (string) directory containing dataset
        params: (dict) containing hyperparams
    """

    # Make new folder in parent_dir with job_name
    model_dir = os.path.join(parent_dir, job_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Write params in json file
    json_path = os.path.join(model_dir, 'params.json')
    params.save(json_path)

    #Launch trainingwith this config
      