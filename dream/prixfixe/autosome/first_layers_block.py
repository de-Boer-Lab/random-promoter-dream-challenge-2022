import torch 
import torch.nn as nn 
from torch import Generator

from typing import Any
from ..prixfixe import FirstLayersBlock
from .utils import initialize_weights
import torch.nn.functional as F

class AutosomeFirstLayersBlock(FirstLayersBlock):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        seqsize: int  # for compatibity. Isn't used by block itself
    ):
        super().__init__(in_channels=in_channels, 
                         out_channels=out_channels, 
                         seqsize=seqsize)
        ks = 7
        activation = nn.SiLU
        self.bn_momentum = .1
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=ks,
                padding='same',
                bias=False
            ),
            nn.BatchNorm1d(out_channels,
                            momentum=self.bn_momentum),
            activation()
        )

    def forward(self, x) -> torch.Tensor:
        if len(x.shape) < 3:
            x = F.one_hot(x.to(torch.int64), self.in_channels)
            x = x.float().permute(0,2,1)
        x = self.block(x)
        return x
    
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))