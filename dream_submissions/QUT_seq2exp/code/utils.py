import torch
import torch.nn as nn 
import math

class PositionalEncoding(nn.Module):
    ''' 
    positional encoding module changed around for the tensor shapes and order
    required for our special type of "Transformer"  
    
    Needed to do a special version as the way we flatten at the end requires
    [batch, seq, embed] as the axis orders (collapse last two)
    
    '''
    
    def __init__(self, d_model, max_len =  110):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(pos * div_term) 
        pe[:, 0, 1::2] = torch.cos(pos * div_term) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = torch.swapaxes(torch.swapaxes(x, 0,1)  + self.pe[:x.size(1)], 0,1)
        return x   