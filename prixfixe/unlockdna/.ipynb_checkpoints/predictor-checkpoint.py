import json

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path 

from ..prixfixe.predictor import Predictor

from scipy import stats
import pandas as pd

class UnlockDNAPredictor(Predictor):
    def __init__(self,
                 model: nn.Module, 
                 model_pth: str | Path, 
                 device: torch.device):

        self.model = model.to(device)
        self.model.load_state_dict(torch.load(model_pth))
        self.model.eval()
        self.device = device
    
    def predict(self, batch: dict[str, Any]) -> float:

        expression, _ = self.model(batch.to(self.device))  
        expression = expression.detach().cpu().numpy()

        return expression