from typing import Any 
import torch
import torch.nn as nn 
import torch.nn.functional as F
#from torch import nn, Generator

from ..prixfixe import FinalLayersBlock
#from .utils import initialize_weights

class UnlockDNAFinalLayersBlock(FinalLayersBlock):
    def __init__(
        self, 
        in_channels: int = 512,
        seqsize: int = 200,
        num_projectors: int = 8,
        input_dim = 5, 
        rate = 0.2
        ):
        super().__init__(in_channels=in_channels,
                         seqsize=seqsize)        
        self.expression_dense = nn.Linear(in_channels,1)
        self.dropout = nn.Dropout(rate)
        self.nucleotide_dense = nn.Linear(in_channels,input_dim)
        self.in_channels = in_channels
        self.seqsize = seqsize
        self.num_projectors = num_projectors
        self.input_dim = input_dim
        self.rate = rate
        self.criterion=nn.MSELoss()

    def forward(self, x):

        expression = x[:,:, :self.num_projectors].permute(0,2,1)
        x = x[:, :, -self.seqsize:].permute(0,2,1)

        expression = self.dropout(expression)
        expression = self.expression_dense(expression)
        expression = torch.mean(expression, 1).squeeze(-1)

        x = self.nucleotide_dense(x)
        x = x.permute(0,2,1)

        return expression, x
    
    def train_step(self, batch: dict[str, Any]):
        x = batch["x"].to(self.device)
        expression, seq = self.forward(x)

        y = batch["y"].to(self.device)
        loss_expression = self.criterion(expression.squeeze(-1), y.squeeze(-1))
            
        return [expression, seq], loss_expression
    
    @property
    def dummy(self) -> torch.Tensor:
        """
        return dummy input data to test model correctness
        """
        return torch.zeros(size=(1, self.in_channels, self.seqsize + self.num_projectors), dtype=torch.float32)
