from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch import nn, Generator

from ..prixfixe import FinalLayersBlock


class BHIFinalLayersBlock(FinalLayersBlock):
    def __init__(
        self,
        in_channels: int = 320,
        seqsize: int = 110,
        hidden_dim: int = 64,
    ):
        super().__init__(in_channels, seqsize)
        self.flat = nn.Flatten()
        self.main = nn.Sequential(
            nn.Linear(in_channels * seqsize, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.criterion = nn.HuberLoss()

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = self.flat(x)  # (batch_size, in_channels * seq_len)
        x = self.main(x)  # (batch_size, output_dim)
        return x.squeeze(-1)  # (batch_size,)

    def train_step(self, batch: Dict[str, Any]):
        x = batch["x"].to(self.device)
        pred = self.forward(x)
       
        y = batch["y"].to(self.device).to(torch.float32)
        loss = self.criterion(pred, y.squeeze(-1))
            
        return y, loss
    
