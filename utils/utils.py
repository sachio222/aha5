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
import matplotlib.pyplot as plt
import platform
import torch
import torchvision

# Set global logger
logger = logging.getLogger('__main__.' + __name__)


class Params():
    """Loads params file from params.json

        
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
            # self.build_experiment(params)

    def save(self, json_path):
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)

    # def build_experiment(self, params):
    #     for key, value in params.items():
    #         if isinstance(value, list):
    #             # Create subfolder
    #             for v in value:
    #                 subfolder = create_subfolder(f'{key}_{v}')

    #         else:
    #             print(value)
    #     exit()

    def update(self, json_path):
        """Loads parameters from json file."""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by 'params.dict['learning_rate]."""
        return self.__dict__


class Experiment():
    """An Experiment class sets up a new experiment.

    Consists of an argparser, loads params.json, initializes read/write paths.

    Methods:
        _set_args: (Private) Define all arg parsers here.
        _load_params: (Private) Load parameters file from json path.
        _set_params: (Private) Write custom args to params file when applicable.
        get_params: (Public) Get params object.
        _init_paths: (Private) Create absolute paths from args or params.
        get_paths: (Public) Returns tuple of json path, model path and data paths.
    """

    def __init__(self):
        super(Experiment, self).__init__()

        # Parse args
        self.args = self._set_args()

        # Check if pretrain for custom json path.
        if self.args.pretrain:
            json_path = 'experiments/pretrain/params.json'
        else:
            json_path = self.args.json

        # Load params file with class method.
        self.params = self._load_params(path=json_path)

        # Rewrite params file with user args.
        self._set_params()

        # Convert relative paths to absolute paths.
        self._init_paths()

    def _set_args(self):
        """Set path variables, settings, etc.

        Args:
            --json: (str, required) Path to params.json.
            --data: (str) Override params.json model path.
            --model: (str) Save/load model weights.
            --seed: (int) Manual seed for stochasticity
            --paths: (bool) Print loaded paths to console. 
            --silent: (bool) Do not print status.
            --wandb: (bool) Upload results to wandb.
            --showlast: (bool) Show image after each epoch.
            --animate: (bool) Shows image after each step.
            --load: (bool) Load pretrained weights.
            --pretrain: (bool) pretrain VC, background set=True, preset json path. 
            -a, --autosave: (bool)
        """

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument(
            '--json',
            help='params.json filename. eg. "experiments/params.json".',
            default='experiments/train/params.json',
            type=str)

        parser.add_argument(
            '--data',
            help='(str) Dataset folder path. Eg. "data_folder".',
            default=None,
            type=str)

        parser.add_argument(
            '--model',
            help='(str) Model weights path, eg."pretrained_folder".',
            default=None,
            type=str)

        parser.add_argument('--paths',
                            nargs='?',
                            const=True,
                            help='(bool) Print file paths in console.',
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

        parser.add_argument('--wandb',
                            nargs='?',
                            const=True,
                            default=False,
                            type=bool,
                            help='(bool) Uploads data to wandb.')

        parser.add_argument('--showlast',
                            nargs='?',
                            const=True,
                            default=False,
                            type=bool,
                            help='(bool) Shows output img after each epoch.')

        parser.add_argument('--animate',
                            nargs='?',
                            const=True,
                            default=False,
                            type=bool,
                            help='(bool) Shows weights after eacg step.')

        parser.add_argument('-a',
                            '--autosave',
                            nargs='?',
                            const=True,
                            default=False,
                            type=bool,
                            help='(bool) Autosave.')

        parser.add_argument('--pretrain',
                            nargs='?',
                            const=True,
                            default=False,
                            type=bool,
                            help='(bool) Pretrain weights.')

        return parser.parse_args()

    def _load_params(self, path):
        """Loads parameters from json file.
        
        Args:
            path: (string) relative path to params.json file.
        
        Returns:
            _params: (Params object) contains default parameters from json file.
        """

        self.json_path = Path().absolute() / path

        try:
            _params = Params(self.json_path)

            if not self.args.silent:
                logger.info(f'Reading params from {self.json_path}.')

        except:
            logger.error(f'No params.json file found at {self.json_path}.\n')
            exit()

        return _params

    def _set_params(self):
        """Adds/overwrites params.json with user args if applicable.
        
        Todo:
            Make the conversion automatic and flexible for every entry in args.
        """
        # json path already exists in order to have loaded json file.
        self.params.json_path = self.json_path

        # Check if user supplied args.
        if self.args.seed:
            self.params.seed = self.args.seed
        if self.args.data:
            self.params.data_path = self.args.data
        if self.args.model:
            self.params.model_path = self.args.model

        # Bool flags get written to params.
        self.params.load = self.args.load
        self.params.silent = self.args.silent
        self.params.animate = self.args.animate
        self.params.showlast = self.args.showlast

        if not self.params.silent:
            logger.info(f'SHOW LAST: {self.params.showlast}')

        self.params.autosave = self.args.autosave
        if not self.params.silent:
            logger.info(f'AUTOSAVE: {self.params.autosave}')

        self.params.wandb = self.args.wandb
        if self.params.wandb:
            if not self.params.silent:
                logger.info('Uploading to W&B')

        self.params.pretrain = self.args.pretrain
        if self.params.pretrain:
            if not self.params.silent:
                logger.info('ATTENTION: Pretraining Mode...')

    def get_params(self):
        return self.params

    def _init_paths(self):
        """Creates absolute paths from inputs relative to __main__.

        Checks params.json file for default paths. If path is provided
            as argument, uses argument instead. Shows output if paths flag
            present.
        """

        # Write full path to params.
        self.params.data_path = Path().absolute() / self.params.data_path
        self.params.model_path = Path().absolute() / self.params.model_path

        if not self.args.silent:
            logger.info('Paths initialized successfully.')

        # Output paths if arg --paths (bool) applied.
        if self.args.paths:
            logger.info('PATHS:')
            logger.info(f'- json path: {self.params.json_path}')
            logger.info(f'- data path: {self.params.data_path}')
            logger.info(f'- model path: {self.params.model_path}')

    def get_paths(self):
        return self.params.json_path, self.params.data_path, self.params.model_path


class RunningAverage():
    """A simple class that maintains the running average of a qty.

    Example:
    ```
    >>> loss_avg = RunningAverage()
    >>> loss_avg.update(2)
    >>> loss_avg() = 3
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

    Example:

        >>> print_full_tensor(tensor_name)
        # Prints all in tensor without elipses.
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
            logger.info("Creating checkpoint directory {}".format(checkpoint))
        Path(checkpoint).mkdir()
    else:
        if not silent:
            logger.info("Getting checkpoint directory...")
    if not silent:
        logger.info("Saving file...")
    # Remember to convert filepath to str or it flips out when trying to save
    torch.save(state, str(filepath))
    if not silent:
        logger.info("File saved successfully.")


def load_checkpoint(checkpoint, model, optimizer=None, name="last"):
    """Loads parameters dict from checkpoint file to model, and optimizer.

    Args:
        checkpoint: (string) filename to load
        model: (torch.nn.Module) model to load parameters
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
        name: (string) previx to '.pth.tar' eg: name.pth.tar
    """
    filepath = checkpoint / "{}.pth.tar".format(name)

    logger.info("Looking for saved files...")

    if not Path(checkpoint).exists():
        raise ("File does not exist at {}".format(checkpoint))
    checkpoint = torch.load(str(filepath))

    logger.info("Weights found.")

    model.load_state_dict(checkpoint.get("state_dict"), strict=False)

    if optimizer:
        optimizer.load_state_dict(checkpoint.get("optim_dict"))

    logger.info("Loading saved weights complete.")
    return checkpoint


def set_logger(logger):
    """Creates logger object for logging to log. 

    Stores log relative to project root. 

    Todo:
        customize log path from args.

    """

    log_path = './session'

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)

    # Output to console
    c_handler = logging.StreamHandler()
    c_handler.setLevel(logging.DEBUG)
    c_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s: %(message)s | (%(name)s)')
    c_handler.setFormatter(c_formatter)
    logger.addHandler(c_handler)

    # Output to file
    # Set mode to 'a' for append, 'w' for overwrite.
    from datetime import datetime
    time = str(datetime.utcnow().strftime('%I.%M.%S_%d%m%y'))
    f_handler = logging.FileHandler(filename=f'{log_path}_{time}.log', mode='a')
    f_handler.setLevel(logging.DEBUG)
    f_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s: %(message)s | %(name)s.py')
    f_handler.setFormatter(f_formatter)
    logger.addHandler(f_handler)


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
    """Returns string for os. Used for os specific performance differences. 
    """
    my_system = platform.system()
    return my_system


def clear_terminal(system=None):
    """Clear the terminal on load with ANSI <ESC>
    
    Gets rid of some of the noise for a freshs start in terminal.
    """

    system = system or check_os()
    system = system.lower()

    if system != "windows":
        print("\033c", end="")
    elif system == "windows":
        print("\033[H\033[2J", end="")
