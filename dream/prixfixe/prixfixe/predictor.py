import torch
import torch.nn as nn
from pathlib import Path 

from abc import ABCMeta, abstractmethod

class Predictor(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self,
                 model: nn.Module, 
                 model_pth: str | Path, 
                 device: torch.device):
        ...
    
    @abstractmethod
    def predict(self, 
                seq: str) -> float:
        ...