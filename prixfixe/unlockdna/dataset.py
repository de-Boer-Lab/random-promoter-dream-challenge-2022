import torch 
import scipy 
import numpy as np 
import pandas as pd 

from typing import ClassVar
from torch.utils.data import Dataset

class UnlockDNASeqDatasetProb(Dataset):
    
    """ Sequence dataset. """
    
    def __init__(self, 
                 ds: pd.DataFrame):
        self.ds = ds
    
    def __getitem__(self, i: int):
        """
        Output
        ----------
        X: torch.Tensor int     
            tensor with necloetide sequence coded as integers
        expression: torch.Tensor float 
            expression values
        """

        seq = self.ds.seq.values[i]
        X = torch.from_numpy(seq)

        expression = self.ds.expression.values[i]
        expression = torch.from_numpy(np.asarray(expression)).float()
 
        return {"x": X,
                "y": expression
                }
    
    def __len__(self) -> int:
        return len(self.ds.seq)