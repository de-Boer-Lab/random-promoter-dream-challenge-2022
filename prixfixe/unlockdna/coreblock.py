import torch
import torch.nn as nn 
import torch.nn.functional as F
#from torch import Generator

from typing import Any
#from collections import OrderedDict

from ..prixfixe import CoreBlock
from .add_blocks import ConformerSASwiGLULayer
#from .utils import initialize_weights

class UnlockDNACoreBlock(CoreBlock):
    def __init__(
        self,
        in_channels: int=512,
        out_channels: int=64,
        seqsize: int = 200,
        num_heads: int=4,
        kernel_size = 15,
        rate = 0.1,
        n_blocks = 4

    ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         seqsize=seqsize)     


        self.blocks = nn.ModuleList([ConformerSASwiGLULayer(embedding_dim = in_channels,
                                    kernel_size = kernel_size, rate = rate, num_heads = num_heads) for _ in range(n_blocks)])
        self.n_blocks = n_blocks
        self.out_channels = out_channels
        self.pos_embedding = nn.Embedding(self.seqsize, out_channels)
        
    def forward(self, x):
        x = x.transpose(1,2)
        
        pos = torch.arange(start=0, end = self.seqsize, step=1).to(self.device)
        pos = pos.unsqueeze(0)
        pos = self.pos_embedding(pos.long())
        x = x + pos
        x = x.transpose(1,2)

        for i in range(self.n_blocks) :
            x = self.blocks[i](x)
        return x
    
#    def weights_init(self, generator: Generator) -> None:
#        self.apply(lambda x: initialize_weights(x, generator))
    
