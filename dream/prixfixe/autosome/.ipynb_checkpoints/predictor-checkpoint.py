import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path 

from ..prixfixe.predictor import Predictor
from .utils import n2id, revcomp

class AutosomePredictor(Predictor):
    def __init__(self,
                 model: nn.Module, 
                 model_pth: str | Path, 
                 device: torch.device):

        self.model = model.to(device)
        self.model.load_state_dict(torch.load(model_pth))
        self.model = self.model.eval()

        self.use_reverse_channel = True
        self.use_single_channel = True
        self.seqsize = 150
        self.device = device

    def _add_plasmid(self, seq: str):
        left_adapter = "TGCATTTTTTTCACATC"

        with open("data/plasmid.json") as json_file:
            plasmid = json.load(json_file)

        INSERT_START = plasmid.find('N' * 80)
    
        #take the left part of the plasmid
        add_part = plasmid[INSERT_START-150:INSERT_START]
        
        # cut left adapter and append the plasmid part
        seq = add_part + seq[len(left_adapter):]
        
        # reduce sequence size to seqsize
        seq = seq[-150:]
        return seq

    def _preprocess_sequence(self, seq: str):
        seq_i = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq_i))
        code = F.one_hot(code, num_classes=5)
        
        code[code[:, 4] == 1] = 0.25
        code = code[:, :4].float()
        return code.transpose(0, 1)

    def _add_channels(self, seq: torch.Tensor, rev_value: int):
        to_concat = [seq]
        
        # add reverse augmentation channel
        if self.use_reverse_channel:
            rev = torch.full( (1, self.seqsize), rev_value, dtype=torch.float32)
            to_concat.append(rev)
            
        # add singleton channel
        if self.use_single_channel:
            single = torch.full( (1, self.seqsize) , 0, dtype=torch.float32)
            to_concat.append(single)

        # create final tensor
        if len(to_concat) > 1:
            x = torch.concat(to_concat, dim=0)
        else:
            x = seq

        return x

    def predict(self, sequence: str) -> float:
        x = self._add_channels(
            self._preprocess_sequence(self._add_plasmid(sequence)),
            rev_value=0
        )
        x_rev = self._add_channels(
            self._preprocess_sequence(revcomp(self._add_plasmid(sequence))),
            rev_value=1
        )
        x = x.to(self.device)
        y = self.model(x[None])[-1].detach().cpu().flatten().tolist()
        
        x_rev = x_rev.to(self.device)
        y_rev = self.model(x_rev[None])[-1].detach().cpu().flatten().tolist()
        
        y, y_rev = np.array(y), np.array(y_rev)

        y = (y + y_rev) / 2
        return y.item()