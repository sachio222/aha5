# utils.py
# Jacob Krajewski, 2020
#
# Large portions inspired from Stanford CS2300 best practices guidelines
# for deep learning projects. Original repository available below:
# https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

from pathlib2 import Path
import argparse
import json
import logging
import platform
import torch
import torchvision
import matplotlib.pyplot as plt


class Params():

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by 'params.dict['learning_rate]."""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a qty.

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)

    reset = __init__


def show_sample_img(dataset, idx):
    sample = dataset.__getitem__(idx)
    plt.imshow(sample[0].numpy()[0])
    plt.show()


def print_full_tensor(tensor):
    """You know how it only shows part of the tensor when you print?

    Well use this to show the whole thing.
    """

    torch.set_printoptions(profile="full")
    print(tensor)
    torch.set_printoptions(profile="default")


def get_save_state(epoch, model, optimizer):
    state = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optim_dict': optimizer.state_dict()
    }
    return state


def save_checkpoint(state, checkpoint, name="last", silent=True):
    """Saves state dict to file. 

    Args:
        state: (dict) contains epoch, state dict and optimizer dict
        checkpoint: (Path) directory name to store saved states
        name: (string) previx to '.pth.tar' eg: name.pth.tar
        silent: (bool) if True, bypass output messages

    Todo:
        Simplify the silent checks so I don't need 4 if statements
    """
    filepath = checkpoint / "{}.pth.tar".format(name)
    if not Path(checkpoint).exists():
        if not silent:
            print("Creating checkpoint directory {}".format(checkpoint))
        Path(checkpoint).mkdir()
    else:
        if not silent:
            print("\nGetting checkpoint directory...")
    if not silent:
        print("Saving file...")
    # Remember to convert filepath to str or it flips out when trying to save
    torch.save(state, str(filepath))
    if not silent:
        print("File saved successfully.")


def load_checkpoint(checkpoint, model, optimizer=None, name="last"):
    """Loads parameters dict from checkpoint file to model, and optimizer.

    Args:
        checkpoint: (string) filename to load
        model: (torch.nn.Module) model to load parameters
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        name: (string) previx to '.pth.tar' eg: name.pth.tar
    """
    filepath = checkpoint / "{}.pth.tar".format(name)

    print("Looking for saved files...", end=" ")

    if not Path(checkpoint).exists():
        raise ("File does not exist at {}".format(checkpoint))
    checkpoint = torch.load(str(filepath))

    print("Found.")

    model.load_state_dict(checkpoint.get("state_dict"), strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint.get("optim_dict"))

    print("OK. Loading saved weights complete.")
    return checkpoint


def set_logger(log_path):
    """Logs info in terminal and file at log_path.

    Example:
    ```
    logging.info('Starting Training...')

    Args:
        log_path: (string) where to log.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def showme(tnsr,
           size_dim0=10,
           size_dim1=10,
           title=None,
           full=False,
           detach=False,
           grid=False):
    """Does all the nasty matplotlib stuff for free. 
    """
    if detach:
        tnsr = tnsr.detach().numpy()

    if not grid:
        if len(tnsr.shape) > 2:
            tnsr = tnsr.view(tnsr.shape[0], -1)

        fig, ax = plt.subplots(figsize=(size_dim0, size_dim1))
        ax.set_title(title, color="blue", loc="left", pad=20)
        ax.matshow(tnsr)
        plt.show()
        print(tnsr.shape)
        if full:
            print(tnsr)
    else:
        grid_img = torchvision.utils.make_grid(tnsr, nrow=5)
        plt.imshow(grid_img.permute(1, 2, 0))
        plt.show()
        print(tnsr.shape)


def animate_weights(t, nrow=11, label=None, auto=False):
    """Animates weights during training. Only works on Mac.

    Press ctrl + C in terminal to escape. Change auto to True if you are 
    running on a mac. It is pretty good. 

        Usage example:
            >>> animate_weights(conv1_weights, i)

    Args:
        t: (tensor) from "model.layer_name.weight.data" on layer
        iter: (scalar, string) Optional. Shows label for each pass
    """

    grid_img = torchvision.utils.make_grid(t, nrow)
    # plt.ion()
    plt.title(label, color="blue", loc="left", pad=20)
    plt.imshow(grid_img.numpy()[0])
    if not auto:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(0.0001)
        plt.close()


def check_os():
    my_system = platform.system()
    return my_system


def clear_terminal(system=None):
    """Clear the terminal on load with ANSI <ESC>"""

    system = system or check_os()
    system = system.lower()

    if system != "windows":
        print("\033c", end="")
    elif system == "windows":
        print("\033[H\033[2J", end="")


class Experiment():
    """An Experiment class sets up a new experiment.

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
        super(Experiment, self).__init__()
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
            --seed: (int) Manual seed for stochasticity
            --paths: (bool) Print loaded paths to console. 
            --silent: (bool) Do not print status.
            --load: (bool) Load pretrained weights.
            -a, --autosave: (bool)
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

        parser.add_argument('--paths',
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

        parser.add_argument(
            '--load',
            nargs='?',
            const=True,
            default=False,
            type=bool,
            help='(bool) Load pretrained weights from model path.')

        parser.add_argument('-a',
                            '--autosave',
                            nargs='?',
                            const=True,
                            default=False,
                            type=bool,
                            help='(bool) Autosave.')
        return parser.parse_args()

    def _load_params(self, path):
        """Loads parameters from json file."""

        self.json_path = Path().absolute() / path

        try:
            _params = Params(self.json_path)

            if not self.args.silent:
                print('OK: Params file loaded successfully.')

        except:
            print(f'\nERROR: No params.json file found at {self.json_path}\n')
            exit()

        return _params

    def _set_params(self):
        if self.args.seed:
            self.params.seed = self.args.seed
        if self.args.data:
            self.params.data_path = self.args.data
        if self.args.model:
            self.params.model_path = self.args.model

        self.params.load = self.args.load
        self.params.silent = self.args.silent
        self.params.autosave = self.args.autosave

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

        if args.paths:
            print('PATHS:')
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
