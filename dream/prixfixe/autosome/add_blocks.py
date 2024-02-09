import torch
import torch.nn as nn 

class Concater(nn.Module):
    """
    Concatenates an output of some module with its input alongside some dimension.
    Parameters
    ----------
    module : nn.Module
        Module.
    dim : int, optional
        Dimension to concatenate along. The default is -1.
    """
    def __init__(self, 
                 module: 
                     nn.Module, dim=-1):        
        super().__init__()
        self.mod = module
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.concat((x, self.mod(x)), dim=self.dim)

class SELayer(nn.Module):
    """
    Squeeze-and-Excite layer.

    Parameters
    ----------
    inp : int
        Middle layer size.
    oup : int
        Input and ouput size.
    reduction : int, optional
        Reduction parameter. The default is 4.
    """
    def __init__(self, 
                 inp, 
                 oup, 
                 reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), int(inp // reduction)),
                Concater(Bilinear(int(inp // reduction), 
                                  int(inp // reduction // 2),
                                  rank=0.5, 
                                  bias=True)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction) +  int(inp // reduction // 2), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
class SELayerSimple(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y