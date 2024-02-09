import torch
import torch.nn as nn

from torch import Generator

from typing import Any
from abc import ABCMeta, abstractmethod


class CoreBlock(nn.Module, metaclass=ABCMeta):
    """
    Network core layers performing complex feature extraction
    """
    @abstractmethod
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 seqsize: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seqsize = seqsize

    @abstractmethod
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Usual forward pass of torch nn.Module
        """
        ...
        
    def train_step(self, 
                   batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Modification of the forward pass. Required to train properly different combinations of blocks
        Receives batch with required "x" and "y" keys and optional keys, required for blocks from some teams
        Returns tuple containing:
            1. modified "x"
            2. auxiliary loss if it is computed by the block or `None` otherwise  

        Default realization simply call forward and return None as an auxiliary loss
        """
        return self.forward(batch["x"].to(self.device)), None
    
    def weights_init(self, generator: Generator) -> None:
        """
        Weight initializations for block. Should use provided generator to generate new weights
        By default do nothing
        """
        pass
        
        
    @property
    def dummy(self) -> torch.Tensor:
        """
        return dummy input data to test model correctness and infer output seqsize
        """
        return torch.zeros(size=(1, self.in_channels, self.seqsize), dtype=torch.float32)
    
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration: # model has no parameters
            return torch.device("cpu") # it safe to return cpu in such case
    
    def infer_outseqsize(self) -> int:
        """
        return output seqsize by running model
        """
        x = self.forward(self.dummy.to(self.device))
        return x.shape[-1]
    
    def check(self) -> None:
        """
        Run model on dummy object
        """
        self.forward(self.dummy.to(self.device))