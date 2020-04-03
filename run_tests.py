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
from utils import utils  # pylint: disable=RP0003, F0401


class NewExperiment():
    """An NewExperiment class sets up a new experiment.

    Consists of an argparser, loads params.json, initializes read/write paths.

    Methods:
        _set_args
        get_args
        _load_params
        _init_paths

    """
    def __init__(self):
        """Initialize new experiment. Run argparser, grab params file, init paths.
        """
        super(NewExperiment, self).__init__()
        self.args = self._set_args()
        self.params = self._load_params(path=self.args.json)
        self._set_params()
        self._init_paths(params=self.params, args=self.args)

    def _set_args(self):
        """Set path variables, settings, etc.

        Args:
            --json: (str, required) Path to params.json.
            --data: (str) Override params.json model path.
            --model: (str) Override params.json model path.
            --show_paths: (bool) Print loaded paths to console. 
        """

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            '--json',
            help='params.json filename. eg. "experiments/params.json".',
            default='experiments/train/params.json',
            type=str)

        parser.add_argument('--data',
                            help='(str) Data folder path. Eg. "data_folder".',
                            default=None,
                            type=str)

        parser.add_argument('--model',
                            help='(str) Model path, eg."pretrained_folder".',
                            default=None,
                            type=str)

        parser.add_argument('--show_paths',
                            nargs='?',
                            const=True,
                            help='(boolean) Print file paths in console.',
                            default=False,
                            type=bool)

        parser.add_argument('--seed',
                            help='(int) Set manual seed for randomization.',
                            default=None,
                            type=int)
        
        parser.add_argument('--silent',
                            nargs='?',
                            const=True,
                            default=False,
                            type=bool,
                            help='(bool) Turns off loading info.')

        return parser.parse_args()

    def _load_params(self, path):
        """Loads parameters from json file."""

        self.json_path = Path().absolute() / path

        try:
            _params = utils.Params(self.json_path)
            if not self.args.silent:
                print('OK: Params file loaded successfully.')
        except:
            print(f'\n\nERROR: No params.json file found at {self.json_path}\n')

        return _params
    
    def _set_params(self):
        if self.args.seed:
            self.params.seed = self.args.seed
        if self.args.data:
            self.params.data_path = self.args.data
        if self.args.model:
            self.params.model_path = self.args.model
        
        print(self.params.data_path)

    def get_params(self):
        return self.params

    def _init_paths(self, params, args):
        """Initialize paths relative to __main__.

        Checks for params.json file. Uses paths from file. If path is provided
            as argument, uses argument instead.

        Returns:
            data_path: (path) Path to data file(s).
            model_path: (path) Path to model weights.
            json_path: (path) Path to params json file.
        """

        # Load from args if present, else params file.
        
        self.data_path = Path().absolute() / self.params.data_path
        self.model_path = Path().absolute() / self.params.model_path

        if not self.args.silent:
            print('OK: Paths initialized successfully.')

        if args.show_paths:
            print('Paths:')
            print(f'- json path: {self.json_path}')
            print(f'- data path: {self.data_path}')
            print(f'- model path: {self.model_path}')

    def get_paths(self):
        """
        Returns:
            self.json_path
            self.data_path
            self.model_path
        """
        return self.json_path, self.data_path, self.model_path

aha = NewExperiment()
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
                            params.batch_size,
                            shuffle=True,
                            num_workers=params.num_workers,
                            drop_last=True)


def main():
    # utils.clear_terminal()

    # If GPU
    params.cuda = torch.cuda.is_available()
    
    # Set random seed
    seed = params.seed
    torch.manual_seed(seed)
    if params.cuda:
        torch.cuda.manual_seed(seed)
        params.num_workers = 2

    # make_dataset()


if __name__ == '__main__':
    main()


