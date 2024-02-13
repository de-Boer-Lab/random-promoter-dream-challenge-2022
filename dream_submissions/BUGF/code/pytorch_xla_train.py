#!/usr/bin/python3

# BUGF-Dream 2020 Traning Script by David Jones - (C) University College London 2022

import sys
import os
os.environ["XRT_TPU_CONFIG"] = "localservice;0;localhost:51011"
import time

import gc

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from nndef_exp_alibi import TransformerNet

from focal_loss import FocalLoss

# RESTART training (i.e. if after preemption)
RESTART_FLAG = False
BATCH_SIZE = 32


# ################## Download and prepare the dataset ##################

def load_dataset():

    train_seq_list = []
    train_target_list = []
    val_seq_list = []
    val_target_list = []
    tnum = 0

    nt_trans = str.maketrans('ATGCN', 'ABCDE')

    with open('training_set.txt', 'r') as trainfile:
        for line in trainfile:
            seq, expval = line.rstrip().split()
            # Crop out invariant flanking sequences:
            ntcodes = np.frombuffer(seq[17:-13].translate(nt_trans).encode('latin-1'), dtype=np.uint8) - ord('A')
            ntcodes = torch.from_numpy(ntcodes).byte()
            ntcodes = F.pad(ntcodes, (0,112-ntcodes.size(0)), value=5)
            expval = torch.tensor([float(expval)], dtype=torch.float)
            train_seq_list.append(ntcodes)
            train_target_list.append(expval)

    with open('validation_set.txt', 'r') as valfile:
        for line in valfile:
            seq, expval = line.rstrip().split()
            ntcodes = np.frombuffer(seq[17:-13].translate(nt_trans).encode('latin-1'), dtype=np.uint8) - ord('A')
            ntcodes = torch.from_numpy(ntcodes).byte()
            ntcodes = F.pad(ntcodes, (0,112-ntcodes.size(0)), value=5)
            expval = torch.tensor([float(expval)], dtype=torch.float)
            val_seq_list.append(ntcodes)
            val_target_list.append(expval)

    train_seqs = torch.stack(train_seq_list)
    train_targets = torch.stack(train_target_list).squeeze(-1)
    val_seqs = torch.stack(val_seq_list)
    val_targets = torch.stack(val_target_list).squeeze(-1)
    
    return train_seqs, train_targets, val_seqs, val_targets


class ExpDataset(Dataset):

    def __init__(self, seqs, targets, augment=False):
        self.seqs = seqs
        self.targets = targets
        self.augment = augment

    def __len__(self):
        return self.seqs.size(0)

    def __getitem__(self, tn):

        ntseq = self.seqs[tn]

        if self.augment:
            rw = torch.rand_like(ntseq, dtype=torch.float)
            ntseq = torch.where(rw < 0.9, ntseq, 4)
        expval = self.targets[tn]

        return ntseq, expval


def pearson_correlation(x: torch.Tensor, y: torch.Tensor):

    vx = x - x.mean()
    vy = y - y.mean()

    return F.cosine_similarity(vx.flatten(), vy.flatten(), dim=0)


# ############################## Main program ################################

def _mp_fn(index, train_seqs, train_targets, val_seqs, val_targets):

    device = xm.xla_device()

    network = wrapped_network.to(device)
    
    max_lr = 5e-4

    optimizer = optim.RAdam(network.parameters(), lr=max_lr)

    if RESTART_FLAG:
        # Restart values after preemtpion
        # Final submitted model training was preempted after 6 epochs...
        start_epoch = 7
        val_err_min = 0.258207
    else:
        start_epoch = 1
        val_err_min = 1e32

    lossfn = FocalLoss(gamma = 2.0, reduction = 'none')
    lossfn2 = nn.BCELoss()
    
    exp_train_data = ExpDataset(train_seqs, train_targets, augment=False)
    exp_val_data = ExpDataset(val_seqs, val_targets, augment=False)

    train_sampler = torch.utils.data.distributed.DistributedSampler(exp_train_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True)

    data_loader = DataLoader(dataset=exp_train_data,
                             sampler=train_sampler,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             drop_last=True,
                             num_workers=0)
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(exp_val_data, num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False)
    
    val_data_loader = DataLoader(dataset=exp_val_data,
                                 sampler=val_sampler,
                                 batch_size=BATCH_SIZE,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=0)

    mp_data_loader = pl.MpDeviceLoader(data_loader, device)

    mp_val_data_loader = pl.MpDeviceLoader(val_data_loader, device)
    
    # Finally, launch the training loop.
    print("Starting training...")

    # Set to 15 as overfitting starts at around epoch 7 or 8 usually. Can be increased with more training data
    num_epochs = 15

    binvals = torch.arange(18.0, device=device).unsqueeze(0)

    for epoch in range(start_epoch, num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_samples = 0
        start_time = time.time()

        sys.stdout.flush()

        network.train()
        
        gc.collect()

        for inputs, targets in mp_data_loader:
            optimizer.zero_grad()
            predclasses = network(inputs)[0]
            # Weight down cases where sequence length is not the expected 80
            with torch.no_grad():
                weights = (inputs < 5).sum(dim=1).to(predclasses)
                weights = 1 - (weights - 80).abs() / 64
            loss1 = (lossfn(predclasses, (targets + torch.rand_like(targets)).long()) * weights).sum() / weights.sum()
            with torch.no_grad():
                # Generate sequences with random mutations
                pmut = 0.15
                selv = torch.rand_like(inputs, dtype=torch.float)
                selv = selv + (inputs > 3)
                # Mutation can be +1,2,3 but not 0 or 4 obviously!
                mutations = torch.randint_like(inputs, 1, 4)
                mutinputs = torch.where(selv < pmut, torch.remainder(inputs+mutations, 4), inputs)
            mutpred = torch.sigmoid(network(mutinputs)[1])
            loss2 = lossfn2(mutpred, torch.ne(inputs, mutinputs).float())
            #print(loss1.item(), loss2.item())
            loss = loss1 + loss2
            train_err += loss.item()
            train_samples += 1
            loss.backward()
            xm.optimizer_step(optimizer)

        gc.collect()

        val_err = 0.0
        val_samples = 0
        network.eval()
        # And a full pass over the validation data:
        with torch.no_grad():
            for inputs, targets in mp_val_data_loader:
                predclasses = network(inputs)[0]
                outvals = (predclasses.softmax(-1) * binvals).sum(-1)
                loss = 1 - pearson_correlation(outvals, targets)
                val_err += loss.item()
                val_samples += 1

        # Then we xm.master_print the results for this epoch:
        xm.master_print("Epoch {} of {} in {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        #xm.master_print(train_err, train_samples)
        xm.master_print("  training loss:\t\t{:.6f}".format(train_err / train_samples))
        xm.master_print("  validation loss:\t\t{:.6f}".format(val_err / val_samples))

        if val_err / val_samples < val_err_min: 
            # Save "best" model snapshot
            val_err_min = val_err / val_samples
            fsufx = ""
        else:
            # Save "sub-optimal" model snapshot
            fsufx = "_train"

        xm.save(network.state_dict(), 'exp_model' + fsufx + '.pt')
        xm.master_print("Saving model...")

if __name__=="__main__":

    torch.manual_seed(12345)
    
    def_network = TransformerNet(512, 16, 16)

    if RESTART_FLAG:
        def_network.load_state_dict(torch.load('exp_model.pt', map_location=lambda storage, loc: storage), strict=True)

    # Create neural network model
    wrapped_network = xmp.MpModelWrapper(def_network)

    print("Loading data...")
    train_seqs, train_targets, val_seqs, val_targets = load_dataset()

    xmp.spawn(_mp_fn, args=(train_seqs, train_targets, val_seqs, val_targets), nprocs=8, start_method='fork')
