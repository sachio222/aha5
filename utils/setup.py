from pathlib2 import Path
import argparse

from utils import utils


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
