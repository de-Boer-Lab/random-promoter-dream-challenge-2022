
import torch 
import torch.nn as nn
from torch import Generator 
from typing import Any
from .coreblock import CoreBlock
from .final_layers_block import FinalLayersBlock
from .first_layers_block import FirstLayersBlock

class PrixFixeNet(nn.Module):
    """
    Main model
    Can contain up to three blocks:
    1. First block, performing initial data processing 
    2. Core block - main model part 
    3. Final block - block responsible for the final prediction and loss calculation
    
    It is possible to provide only Final block in case of very simple models
    """
    def __init__(self, 
                 first: FirstLayersBlock | None,
                 core: CoreBlock | None,
                 final: FinalLayersBlock,
                 generator: Generator):
        
        super().__init__()
        self.first = first
        self.core = core
        self.final = final
        self.generator = generator
        self._weights_init()
            
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the model
           x - batch of sequences encoded in appropriate format (batch, channel, seq_size)
        """
        if self.first is not None:
            x = self.first.forward(x)
        if self.core is not None:
            x = self.core.forward(x)
        y = self.final.forward(x)
        return y

    def train_step(self, 
                   batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Training step of the model
        As elements of batch for loss calculation can differ between models
        we propose to pass `batch` to `train_step` and use train_step while training
        phase, not `forward`
        """

        loss = 0
        if self.first is not None:
           x, ax_loss = self.first.train_step(batch)
           batch["x"] = x
           if ax_loss is not None:
               loss += ax_loss
        
        if self.core is not None:
            x, ax_loss = self.core.train_step(batch)
            batch["x"] = x
            if ax_loss is not None:
                loss += ax_loss
        
        y, pred_loss = self.final.train_step(batch)
        loss += pred_loss

        return y, loss

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration: # model has no parameters
            return torch.device("cpu") # it safe to return cpu in such case
        
    @property
    def dummy(self) -> torch.Tensor:
        """
        return dummy input data to test model correctness
        """
        if self.first is not None:
            return self.first.dummy
        if self.core is not None:
            return self.core.dummy
        return self.final.dummy
    
    @property
    def dummy_expression(self) -> torch.Tensor:
        return torch.FloatTensor([[10]])
        
    def check(self) -> None:
        """
        Run model on dummy object
        """
        print("Checking forward pass")
        self.forward(self.dummy.to(self.device))
        print("Forward is OK")
        print("Checking training step")
        batch = {"x": self.dummy, "y": self.dummy_expression}
        self.train_step(batch)
        print("Training step is OK")
        
    def _weights_init(self) -> None:
        if self.first is not None:
            self.first.weights_init(self.generator)
        if self.core is not None:
            self.core.weights_init(self.generator)
        self.final.weights_init(self.generator)