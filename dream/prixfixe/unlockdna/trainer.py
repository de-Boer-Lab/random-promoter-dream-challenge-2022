import json
import torch
import tqdm
import numpy as np
import torch.nn as nn
from pathlib import Path
from typing import Any
from ..prixfixe import Trainer, PrixFixeNet, DataProcessor, DEFAULT_METRICS
from .dataprocessor import UnlockDNADataProcessor
from .add_blocks import SequenceMaskLayer

class ScheduledOptim():
    '''pytorch implementation of transformer scheduler taken from  github.com/jadore801120/attention-is-all-you-need-pytorch'''

    def __init__(self, optimizer, d_model, n_warmup_steps, lr_mul = 1.0, n_steps = 0):
        self._optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.lr_mul = lr_mul
        self.n_steps = n_steps

    def step_and_update_lr(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()

        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

    def state_dict(self):
        return self._optimizer.state_dict()

class UnlockDNATrainer(Trainer):

    def __init__(
        self,
        model: PrixFixeNet,
        dataprocessor: UnlockDNADataProcessor,
        model_dir: str | Path,
        num_epochs: int,
        initial_lr: float,
        embedding_dim: int,
        warmup_steps: int,
        beta_1: float,
        beta_2: float,
        eps: float,
        clip_norm: float,
        clip_value: float,
        n_positions: int,
        N: int,
        M: int,
        device: torch.device = torch.device("cpu")):

        super().__init__(model=model,
                         dataprocessor=dataprocessor,
                         model_dir=model_dir,
                         num_epochs=num_epochs,
                         device=device)

        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr = initial_lr,
            betas = (beta_1, beta_2),
            eps=eps
            )
        
        scheduler = ScheduledOptim(optimizer, d_model=embedding_dim, n_warmup_steps=warmup_steps)
        self.model = model.to(device)
        self.optimizer, self.scheduler = optimizer, scheduler
        self.clip_norm, self.clip_value = clip_norm, clip_value
        self.device = device
        self.scc_loss = nn.CrossEntropyLoss(reduction='none').to(device)
        self.masking = SequenceMaskLayer(n_positions, N, M, ratio = 0.2)
    
    def train_step(self, batch):   
        if not self.model.training:
            self.model = self.model.train()
        unmasked = batch["x"]
        # print('unmasked', unmasked.is_cuda)
        data, mask = self.masking(unmasked)
        # print('data', data.is_cuda)
        # print('mask', mask.is_cuda)
        batch["x"] = data
        pred, loss_expression = self.model.train_step(batch)
        _, seq_pred = pred
        # print('loss expression', loss_expression.is_cuda)
        # print('seq_pred', seq_pred.is_cuda)
        loss_seq = mask.to(self.device) * self.scc_loss(seq_pred.to(self.device), unmasked.long().to(self.device))
        loss_seq = torch.sum(loss_seq) / (torch.sum(mask.to(self.device)) + 1)
        loss = (loss_expression.to(torch.float32) + loss_seq.to(torch.float32)).mean().to(torch.float32)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
        torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step_and_update_lr()

        return loss.item()

    def on_epoch_end(self):
        pass

    def deduce_max_lr(self):
        # TODO: for now no solution to search for maximum lr automatically, learning rate range test should be analysed manually
        max_lr = 0.005
        return max_lr
    
    def _evaluate(self, batch: dict[str, Any]):
        with torch.no_grad():
            X = batch["x"]
            y = batch["y"]
            X = X.to(self.device)
            y = y.float().to(self.device)
            y_pred, _ = self.model.forward(X)
        return y_pred.cpu(), y.cpu()