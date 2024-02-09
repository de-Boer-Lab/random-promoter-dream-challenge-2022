from typing import Any 
import torch
import torch.nn.functional as F
from torch import nn, Generator

from ..prixfixe import FinalLayersBlock
from .utils import initialize_weights

class AutosomeFinalLayersBlock(FinalLayersBlock):
    def __init__(
        self,
        in_channels: int,
        seqsize: int # for compatibity. Isn't used by block itself
    ):
        super().__init__(in_channels=in_channels,
                         seqsize=seqsize)
        out_channels=18
        self.mapper =  nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding='same',
            ),
            #nn.SiLU(),
        )
        self.register_buffer('bins', torch.arange(start=0, end=out_channels, step=1, requires_grad=False))

        # used if "y_probs" has been provided in batch
        self.classification_criterion=nn.KLDivLoss(reduction= "batchmean")
        
        # used if only "y" has been provided ib batch dict
        self.regression_criterion=nn.MSELoss()

    def forward(self, x):
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2) 
        x = F.softmax(x, dim=1)
        score = (x * self.bins).sum(dim=1)
        return score

    def train_step(self, batch: dict[str, Any]):
        x = batch["x"].to(self.device)
       
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)
        logprobs = F.log_softmax(x, dim=1) 
        x = F.softmax(x, dim=1)
        score = (x * self.bins).sum(dim=1)
        
        if "y_probs" in batch: # classification
            y_probs = batch["y_probs"].to(self.device)
            loss = self.classification_criterion(logprobs, y_probs)
        else: # regression
            y = batch["y"].to(self.device)
            loss = self.regression_criterion(score, y.squeeze(-1))
            
        return score, loss
    
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))
