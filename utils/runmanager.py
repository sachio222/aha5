from collections import namedtuple, OrderedDict
from itertools import product

class RunBuilder():
    """Kinda cool thing that turns params lists into lists of params"""
    @staticmethod
    def get_runs(params)
        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values())
            runs.append(Run(*v))

        return runs