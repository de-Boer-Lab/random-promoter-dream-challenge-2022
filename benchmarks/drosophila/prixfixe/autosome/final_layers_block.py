from typing import Any 
import torch
import torch.nn.functional as F
from torch import nn, Generator

from ..prixfixe import FinalLayersBlock
from .utils import initialize_weights

class AutosomeFinalLayersBlock(FinalLayersBlock):
    def __init__(self, in_channels=64, seqsize = 230):
        super(AutosomeFinalLayersBlock, self).__init__(
            in_channels=in_channels,
            seqsize=seqsize)
        self.mapper1 = nn.Conv1d(
            in_channels=in_channels,  # Assuming the input channels to be the same as output
            out_channels=256,
            kernel_size=1,
            padding='same'
        )
        self.flatten1 = nn.Flatten()
        self.linear1 = nn.Sequential(
            nn.Linear(256, 1)
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, 1)
        )
        self.mapper2 = nn.Conv1d(
            in_channels=in_channels,  # Assuming the input channels to be the same as output
            out_channels=256,
            kernel_size=1,
            padding='same'
        )

        self.flatten2 = nn.Flatten()
        self.linear2 = nn.Sequential(
            nn.Linear(256, 1)
            # nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim, 1)
        )
        # self.activation = nn.SiLU()
        # self.predictions = nn.Linear(256, 1)
        self.regression_criterion = nn.MSELoss()

    def forward(self, x):
        x1 = self.mapper1(x)
        x1 = F.adaptive_avg_pool1d(x1, 1)
        x1 = x1.squeeze(2) 
        y1 = self.linear1(x1)

        x2 = self.mapper2(x)
        x2 = F.adaptive_avg_pool1d(x2, 1)
        x2 = x2.squeeze(2) 
        y2 = self.linear2(x2)

        return y1, y2

    def train_step(self, batch: dict[str, Any]):
        x = batch["x"].to(self.device)
        # print(x.shape, 'x')
       
        x1 = self.mapper1(x)
        x1 = F.adaptive_avg_pool1d(x1, 1)
        x1 = x1.squeeze(2) 
        y1 = self.linear1(x1)
        y1 = y1.squeeze(-1)
        # print(y1.shape, 'y1')

        x2 = self.mapper2(x)
        x2 = F.adaptive_avg_pool1d(x2, 1)
        x2 = x2.squeeze(2) 
        y2 = self.linear2(x2)
        y2 = y2.squeeze(-1)
        # print(y2.shape, 'y2')
        y = batch["y"].to(self.device)
        # print(y.shape, 'y')
        # print(y[:, 0].shape, 'y[:, 0]')
        # print(y[:, 1].shape, 'y[:, 1]')
        loss = 1.0 * self.regression_criterion(y1, y[:, 0].squeeze(-1)) + 1.0 * self.regression_criterion(y2, y[:, 1].squeeze(-1)) # weights are 1.0 for now (same as DeepSTARR)
        # print(y.shape, 'y')
        # print(loss.shape, 'loss')
        return x, loss
    
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))
