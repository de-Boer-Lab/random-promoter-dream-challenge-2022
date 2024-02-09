import json

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path 

from ..prixfixe.predictor import Predictor
from .utils import n2id, revcomp

from scipy import stats
import pandas as pd

'''
230407 BHIPredictor added
- flatten (optional)
    train_txt_file: txt file containing sequences and bin number of train data
    
- TTA has been used for BHI original model yet has not been included
  in this module for general usage
'''


class BHIPredictor(Predictor):
    def __init__(self,
                 model: nn.Module, 
                 model_pth: str | Path, 
                 device: torch.device,
                 train_df: pd.DataFrame,
                 flatten=False):

        self.model = model.to(device)
        self.model.load_state_dict(torch.load(model_pth))
        self.model.eval()
        
        self.device = device
        self.flatten = flatten
        
        self.train_loc = train_df['bin'].mean(axis = 0)

    def flatten_(self, x):
        return stats.norm.cdf(x, loc=self.train_loc, scale=1)
    
    def flip_list_batch(self, seq_batch):
        seq_batch_rc = []
        
        for seq in seq_batch:
            seq_rc = seq.flip([0, 1])
            seq_batch_rc.append(seq_rc)
            
        return torch.stack(seq_batch_rc, dim = 0)
    
    def predict(self, batch: dict[str, Any]) -> float:
        
        
        batch_rc = batch.copy()
        batch_rc['x'] = self.flip_list_batch(batch_rc['x'])
        
        
        y = self.model(batch['x'].to(self.device))
        
        y_rc = self.model(batch_rc['x'].to(self.device))
        
        outs = [y, y_rc]
        
        out = torch.stack(outs).mean(dim = 0)
        
        out = out.detach().cpu().numpy()
        
        if self.flatten:
            out = self.flatten_(out)
        
        return out
