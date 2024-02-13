import torch
import torch.nn as nn 
from torch import Generator

from typing import List
from collections import OrderedDict

from ..prixfixe import CoreBlock
from .add_blocks import ConvBlock




class BHICoreBlock(CoreBlock):
    """
    The coreBlock of the BHI model.
    Consists of a bidirectional LSTM layer, multiple ConvBlocks with different kernel sizes and a dropout layer.
    LSTM layer is used for capturing long-range dependencies.
    ConvBlocks consolidate the soft-dependencies into hard-dependencies.
    Output of each ConvBlock is concatenated along the channel dimension same as the FirstCNNBlock.
    """
    def __init__(
        self, 
        in_channels: int = 512,
        out_channels: int = 320,
        seqsize: int = 110,
        lstm_hidden_channels: int = 320,
        kernel_sizes: List[int] = [9, 15],
        pool_size: int = 1,
        dropout1: float = 0.2,
        dropout2: float = 0.5
    ):
        super().__init__(in_channels, out_channels, seqsize)
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each_conv_out_channels = out_channels // len(kernel_sizes)

        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_channels, batch_first=True, bidirectional=True)
        self.conv_list = nn.ModuleList([
            ConvBlock(2 * lstm_hidden_channels, each_conv_out_channels, k, pool_size, dropout1) for k in kernel_sizes
        ])
        self.do = nn.Dropout(dropout2)

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, in_channels)
        x, _ = self.lstm(x)  # (batch_size, seq_len, 2 * lstm_hidden_channels)
        x = x.permute(0, 2, 1)  # (batch_size, 2 * lstm_hidden_channels, seq_len)
        
        # get the output of each convolutional layer
        conv_outputs = [conv(x) for conv in self.conv_list]  # [(batch_size, each_conv_out_channels, seq_len // pool_size), ...]

        # concatenate the outputs along the channel dimension
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, conv_out_channels, seq_len // pool_size)

        x = self.do(x)  # (batch_size, conv_out_channels, seq_len // pool_size)

        return x
