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
        self.mapper = nn.Conv1d(
            in_channels=in_channels,  # Assuming the input channels to be the same as output
            out_channels=256,
            kernel_size=1,
            padding='same'
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
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
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2) 
        x = self.linear(x)
        # x = self.activation(x)
        # x = self.predictions(x)
        return x

    def train_step(self, batch: dict[str, Any]):
        x = batch["x"].to(self.device)
       
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2) 
        x = self.linear(x)

        x = x.squeeze(-1)
        y = batch["y"].to(self.device).squeeze(-1)

        loss = self.regression_criterion(x, y)
        # print(x.shape, 'x')
        # print(y.shape, 'y')
        # print(loss, 'loss')
        return x, loss
    
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))
