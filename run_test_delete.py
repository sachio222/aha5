import torch
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning import Trainer

from model import modules

from torch.utils.data import DataLoader
from torchvision.datasets import Omniglot
from torchvision import datasets, transforms

from pathlib2 import Path

data_path = Path().absolute() / "data"

# test_x = Omniglot(data_path, background=False, transform=tsfm, download=False)
# test_x = DataLoader(test_x, 32, shuffle=True)


class LitECToCA3(pl.LightningModule):
    """Lightning EC to CA3 class"""

    def __init__(self, D_in, D_out):
        super(LitECToCA3, self).__init__()

        self.fc1 = nn.Linear(D_in, 800)
        self.fc2 = nn.Linear(800, D_out)

    def forward(self, x):
        x = x.view(32, -1)
        x = F.leaky_relu(self.fc1(x), 0.1618)
        x = torch.sigmoid(self.fc2(x))
        return x

    def prepare_data(self):
        # Download only
        Omniglot(data_path, background=True, download=True)

    def train_dataloader(self):
        tsfm = transforms.Compose([
            transforms.Resize(52),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        train_x = Omniglot(data_path,
                           background=True,
                           transform=tsfm,
                           download=True)
        return DataLoader(train_x, 32, shuffle=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        # Add logging
        logs = {'loss': loss}
        return {'loss': loss, 'log:': logs}


model = LitECToCA3(52, 1500)
trainer = Trainer()

if __name__ == '__main__':

    trainer.fit(model)
