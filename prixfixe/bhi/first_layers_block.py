import torch 
import torch.nn as nn 

from typing import List
from ..prixfixe import FirstLayersBlock
from .add_blocks import ConvBlock
import torch.nn.functional as F



class BHIFirstLayersBlock(FirstLayersBlock):
    """
    The firstLayersBlock of the BHI model.
    Consists of multiple ConvBlocks with different kernel sizes.
    Output of each ConvBlock is concatenated along the channel dimension.
    """
    def __init__(
        self, 
        in_channels: int = 4,
        out_channels: int = 512,
        seqsize: int = 110,
        kernel_sizes: List[int] = [9, 15],
        pool_size: int = 1,
        dropout: float = 0.2
    ):
        super().__init__(in_channels, out_channels, seqsize)
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each_out_channels = out_channels // len(kernel_sizes)

        self.conv_list = nn.ModuleList([
            ConvBlock(in_channels, each_out_channels, k, pool_size, dropout) for k in kernel_sizes
        ])

    
    def forward(self, x):
        # x: (batch_size, 4, seq_len), 4 channels: A, C, G, T
        if len(x.shape) < 3:
            x = F.one_hot(x.to(torch.int64), self.in_channels)
            x = x.float().permute(0,2,1)

        # get the output of each convolutional layer
        conv_outputs = [conv(x) for conv in self.conv_list]  # [(batch_size, each_out_channels, seq_len // pool_size), ...]

        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, out_channels, seq_len // pool_size)

        return x
