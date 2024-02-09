import torch
import torch.nn as nn 
from torch import Generator

from typing import Any
from collections import OrderedDict

from ..prixfixe import CoreBlock
from .add_blocks import SELayerSimple
from .utils import initialize_weights

class AutosomeCoreBlock(CoreBlock):
    def __init__(
        self,
        in_channels: int=256,
        out_channels: int=64,
        seqsize: int=150, # for compatibity. Isn't used by block itself
    ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         seqsize=seqsize)

        self.resize_factor = 4
        self.se_reduction = 4
        self.bn_momentum = .1
        self.filter_per_group=2
        seqextblocks = OrderedDict()
        activation=nn.SiLU
        ks=7
        block_sizes=[128, 128, 64, 64, 64] 
        #block_sizes=[128, 128, 64, 64, 64, 64] 
        self.block_sizes = [in_channels] + block_sizes + [out_channels]
        for ind, (prev_sz, sz) in enumerate(zip(self.block_sizes[:-1], self.block_sizes[1:])):
            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=prev_sz,
                    out_channels=sz * self.resize_factor,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * self.resize_factor, 
                                momentum=self.bn_momentum),
                activation(),
                
                
                nn.Conv1d(
                    in_channels=sz * self.resize_factor,
                    out_channels=sz * self.resize_factor,
                    kernel_size=ks,
                    groups=sz * self.resize_factor // self.filter_per_group,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz * self.resize_factor, 
                                momentum=self.bn_momentum),
                activation(),
                SELayerSimple(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                nn.Conv1d(
                    in_channels=sz * self.resize_factor,
                    out_channels=prev_sz,
                    kernel_size=1,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(prev_sz,
                                momentum=self.bn_momentum),
                activation(),
            
            )
            seqextblocks[f'inv_res_blc{ind}'] = block

            block = nn.Sequential(
                nn.Conv1d(
                    in_channels=2 * prev_sz,
                    out_channels=sz,
                    kernel_size=ks,
                    padding='same',
                    bias=False
                ),
                nn.BatchNorm1d(sz, 
                                momentum=self.bn_momentum),
                activation(),
            )
            seqextblocks[f'resize_blc{ind}'] = block

        self.seqextractor = nn.ModuleDict(seqextblocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.block_sizes) - 1):
            x = torch.cat([x, self.seqextractor[f'inv_res_blc{i}'](x)], dim=1)
            x = self.seqextractor[f'resize_blc{i}'](x)
        return x 
    
    def weights_init(self, generator: Generator) -> None:
        self.apply(lambda x: initialize_weights(x, generator))
