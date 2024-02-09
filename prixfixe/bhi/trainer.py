import json
import torch
import tqdm

import numpy as np
import torch.nn as nn

from pathlib import Path
from typing import Any

from ..prixfixe import Trainer, PrixFixeNet, DataProcessor, DEFAULT_METRICS
from .dataprocessor import BHIDataProcessor

"""
230406 BHITrainer added
- RC equivariant
    - loss for mean of forward & RC sequences
"""

class BHITrainer(Trainer):
    def __init__(
        self,
        model: PrixFixeNet, 
        dataprocessor: BHIDataProcessor,   # BHI ver.
        model_dir: str | Path,
        num_epochs: int,
        device: torch.device = torch.device("cpu")):
        
        lr = 0.0015
        weight_decay = 0.01
        model = model.to(device)
        
        super().__init__(model=model,
                         dataprocessor=dataprocessor,
                         model_dir=model_dir,
                         num_epochs=num_epochs,
                         device=device)
         
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr = lr, 
            weight_decay=weight_decay
            )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = len(dataprocessor.train) * num_epochs
            )
        
        self.optimizer, self.scheduler = optimizer, scheduler

    def train_step(self, batch):   
        """
        230406 Edited by BHI
        """
        ### get RC strands' results with given batch ###
        if not self.model.training:
            self.model = self.model.train()

        batch_rc = batch.copy()
        # Original strand
        _, loss_1 = self.model.train_step(batch)
        # RC strand
        batch_rc["x"] = self.flip_list_batch(batch_rc["x"])
        _, loss_2 = self.model.train_step(batch_rc)
        
        loss = (loss_1+loss_2)/2
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()   
        return loss.item()

        """
        230406 Added flip_list_batch by BHI
        """
    def flip_list_batch(self, seq_batch):
        seq_batch_rc = []
        for seq in seq_batch:
            seq_rc = seq.flip([0, 1])
            seq_batch_rc.append(seq_rc)
        return torch.stack(seq_batch_rc, dim = 0)
    
    def on_epoch_end(self):
        """
        Autosome sheduler is called during training steps, not on each epoch end
        Nothing to do at epoch end 
        """
        pass
    
    def deduce_max_lr(self):
        # TODO: for now no solution to search for maximum lr automatically, learning rate range test should be analysed manually
        # MAX_LR=0.005 seems OK for most models 
        return 0.005
