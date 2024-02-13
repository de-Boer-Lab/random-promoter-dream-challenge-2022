
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

#TODO: add the lr scheduler
class Trainner:
    def __init__(self, model, optimizer, criterion, device, trainloader, validloader, lr_scheduler=None, epochs=30):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.trainloader = trainloader
        self.validloader = validloader
        self.lr_scheduler = lr_scheduler
        self.epochs = epochs
        self.history = {"loss": [], "val_loss": []}
        self.best_loss = np.inf
        self.best_model = None
        self.best_epoch = 0

    def _train_step(self):
        self.model = self.model.to(self.device)
        self.model.train()
        train_loss = 0
        for seq, exp in tqdm(self.trainloader):
            seq = seq.to(self.device).float()
            exp = exp.to(self.device).float()
            self.optimizer.zero_grad()
            pred = self.model(seq)
            loss = self.criterion(pred, exp)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
        return train_loss/len(self.trainloader)

    def _valid_step(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        valid_loss = 0
        for seq, exp in tqdm(self.validloader):
            seq = seq.to(self.device).float()
            exp = exp.to(self.device).float()
            pred = self.model(seq)
            loss = self.criterion(pred, exp)
            valid_loss += loss.item()
        return valid_loss/len(self.validloader)

    def train(self):
        for epoch in range(self.epochs):
            train_loss = self._train_step()
            valid_loss = self._valid_step()
            self.history["loss"].append(train_loss)
            self.history["val_loss"].append(valid_loss)
            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.best_model = self.model.state_dict()
                self.best_epoch = epoch
            # if self.lr_scheduler is not None:
            #     self.lr_scheduler.step(valid_loss)
            print(f"Epoch: {epoch}/{self.epochs}")
            print(f"Train loss: {train_loss}")
            print(f"Valid loss: {valid_loss}")

    def evaluate(self):
        self.model.load_state_dict(self.best_model)
        self.model = self.model.to(self.device)
        self.model.eval()
        valid_loss = 0
        list_pred = []
        list_exp = []
        with torch.no_grad():
            for seq, exp in tqdm(self.validloader):
                seq = seq.to(self.device).float()
                exp = exp.to(self.device).float()
                pred = self.model(seq)
                loss = self.criterion(pred, exp)
                list_pred.append(pred.cpu().numpy())
                list_exp.append(exp.cpu().numpy())
                valid_loss += loss.item()
        np_pred = np.concatenate(list_pred, axis=0)
        np_exp = np.concatenate(list_exp, axis=0)
        return valid_loss/len(self.validloader), np_pred, np_exp

    def _save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)

    def _load_model(self, file_path):
        self.model.load_state_dict(torch.load(file_path))
