# Artificial Hippocampal Algorithm in Pytorch

ATTN: This is a work in progress.

Pytorch implementation of AHA! an ‘Artificial Hippocampal Algorithm’ for Episodic Machine Learning by Kowadlo, Rawlinson and Ahmed (2019). 

## Getting Started

Use Pipenv to install dependencies and create compatible virtual environment. (https://thoughtbot.com/blog/how-to-manage-your-python-projects-with-pipenv)

 - Requires Python version >= 3.6

### Pretraining weights
To pretrain the visual cortex modules, run the following:

```python train.py --model=experiments/pretrain --json=experiments/pretrain/params.json```

To enable saving of weights after each epoch add the autosave flag:

```--autosave or -a```

To load pre-trained weights, include:

```--load```

Sending metrics to wandb.ai:

```--wandb```

Display image after each epoch:
```--showlast```

Animate training(mac only)
```--animate```

Customize relative paths with:

weights: ```--model```

datafolder: ```--data```

params.json: ```--json```

### Running
Coming...

Weights should already be trained for the VC module, so to run predictions, do the following:

1. python train.py

### Logs
Logs are stored in session.log in the project root dir. They are currently set to overwrite (```w```) and can be set to append with an ```a``` in the ```set_logs```

## Built With

* [Pytorch](https://pytorch.org/) - Artificial neural network framework
* [Pipenv](https://pypi.org/project/pipenv/) - Dependency Management


## Authors

* **Jacob Krajewski** - *Pytorch implementation*


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks to **Gideon Kowadlo** and **David Rawlinson** for guiding me through some of the more difficult elements of the ANN. 
* Thanks to @ptrblk on the Pytorch forums for walking me through some of the more confusing aspects of Pytorch. 

