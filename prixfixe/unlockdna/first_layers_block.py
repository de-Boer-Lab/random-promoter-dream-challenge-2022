import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
#from torch import Generator

from typing import Any
from ..prixfixe import FirstLayersBlock
#from .utils import initialize_weights

class UnlockDNAFirstLayersBlock(FirstLayersBlock):
    def __init__(self, 
        in_channels: int = 6,
        out_channels: int = 512,
        seqsize: int = 200, 
        kmer = 3, 
        strides = 2,
        num_projectors = 8):
        
        super().__init__(in_channels, out_channels, seqsize)
        self.pos_embedding = nn.Embedding(seqsize, out_channels)
        self.strand_embedding = nn.Embedding(2, out_channels) # plus/minus strands
        self.expression_embedding = nn.Linear(1,out_channels)
        self.kmer_dense = nn.Linear(in_channels*kmer,out_channels)
        self.in_channels = in_channels
        self.seqsize = int(seqsize / strides)
        self.kmer = kmer
        self.num_projectors = num_projectors
        
        
    def forward(self, x): # input = (batch, seq)

        x = F.pad(x, (self.kmer//2-1, self.kmer//2, 0, 0))

        x = x.unfold(2, self.kmer, 1)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.transpose(2,3)
        x = x.reshape(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])
        x = self.kmer_dense(x)

        strand = torch.tensor(np.repeat([0,1], repeats = int(self.seqsize / 2))).to(self.device)
        strand = strand.unsqueeze(0)
        strand = self.strand_embedding(strand.long())

        x = x + strand  # 채널 dim=2
        x = x.transpose(1,2)

        return x
    @property
    def dummy(self) -> torch.Tensor:
        """
        return dummy input data to test model correctness and infer output seqsize
        """
        return torch.zeros(size=(1, self.in_channels, self.seqsize), dtype=torch.float32)