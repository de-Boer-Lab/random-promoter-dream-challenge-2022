import torch
import torch.nn as nn
from torch import Generator

from typing import Any

from abc import ABCMeta, abstractmethod


class FinalLayersBlock(nn.Module, metaclass=ABCMeta):
    """
    Network final layers performing final prediction and (optionally) loss calculation
    """
    
    @abstractmethod
    def __init__(self,
                 in_channels: int,
                 seqsize: int):
        super().__init__()
        self.in_channels = in_channels
        self.seqsize = seqsize
    
    @abstractmethod
    def forward(self, 
                x: torch.Tensor) -> torch.Tensor:
        """
        Usual forward pass of torch nn.Module
        """
        ...
    
    @abstractmethod
    def train_step(self,
                   batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Final block must return predicted y value and loss to be used by Trainer instance
        This block MUST work in case batch contains only 'x' and 'y' keys. 
        It, however, can change behaviour (e.g loss calculated) if batch contains additional keys,
        beneficial for team implementation (see autosome final block)
        """
        ...
        
    @property
    def dummy(self) -> torch.Tensor:
        """
        return dummy input data to test model correctness
        """
        return torch.zeros(size=(1, self.in_channels, self.seqsize), dtype=torch.float32)
    
    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration: # model has no parameters
            return torch.device("cpu") # it safe to return cpu in such case
        
    def check(self) -> None:
        """
        Run model on dummy object
        """
        self.forward(self.dummy.to(self.device))
        
    def weights_init(self, generator: Generator) -> None:
        """
        Weight initializations for block. Should use provided generator to generate new weights
        By default do nothing
        """
        pass