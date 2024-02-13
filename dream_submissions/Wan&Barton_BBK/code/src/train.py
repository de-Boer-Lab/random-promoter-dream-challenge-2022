'''
This code is modefied based on a TCN impelemtation by https://github.com/locuslab/TCN
'''

import os
import csv
import math
import joblib
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader as DL


# Global variables
DICTIONARY = {'A': 0, 'C': 1, 'G': 2, 'N': 3, 'T': 4}
NUM_CHAR = len(DICTIONARY)


'''
Character-level Temporal Convolutional Network (TCN) https://arxiv.org/abs/1803.01271
'''
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        
        self.chomp_size = chomp_size


    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.padding = padding
        self.dropout = dropout
        
        self.relu = nn.ReLU()
        self.temp_block = nn.Sequential(
            weight_norm(nn.Conv1d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation)),
            Chomp1d(self.padding),
            self.relu,
            nn.Dropout(self.dropout),
            weight_norm(nn.Conv1d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=self.stride,
                                  padding=self.padding, dilation=self.dilation)),
            Chomp1d(self.padding),
            self.relu,
            nn.Dropout(self.dropout)
        )
        self.downsample = nn.Conv1d(self.in_channels, self.out_channels, 1) if self.in_channels != self.out_channels else None
        

    def forward(self, x):
        out = self.temp_block(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, em_dim, output_size, channel_size, max_seq_len, kernel_size, stride, 
                 dropout, em_dropout, num_char):
        super().__init__()
        
        # The number of levels is defined based on the longest sequence        
        levels = int(math.log(max_seq_len / (kernel_size - 1), 2))
        out_channels = [channel_size] * levels
        layers = []
        for i in range(levels):
            dilation = 2 ** i
            in_c = em_dim if i == 0 else out_channels[i - 1]
            out_c = out_channels[i]
            layers += [TemporalBlock(in_c, out_c, kernel_size, stride, dilation,
                                     padding=(kernel_size - 1) * dilation, dropout=dropout)]

        self.tcn = nn.Sequential(*layers)
        self.em = nn.Embedding(num_char, em_dim)
        self.em_dropout = nn.Dropout(em_dropout)
        self.linear = nn.Linear(channel_size, output_size)


    def forward(self, x):
        x = self.em_dropout(self.em(x))
        x = self.tcn(x.transpose(1, 2))
        x = self.linear(x[:, :, -1])
        return x.view(-1)


class CCLoss(nn.Module):
    '''
    The loss function is a combination of MSE and both
    correlation coefficients (PearsonR^2 & SpearmanR).
    MSE is multiplied by a small weight to contribute less
    to the final loss.    
    '''
    def __init__(self, w=0.1):
        super().__init__()
        
        self.mse = nn.MSELoss()
        self.w = w        

    def forward(self, out, labels):
        mse = self.mse(out, labels)
        labels = labels.cpu().detach().numpy()
        out = out.cpu().detach().numpy()
        r1, _ = pearsonr(labels, out)
        r2, _ = spearmanr(labels, out)
        loss = mse * self.w - ((r1 ** 2) + r2)
        
        return loss, r1, r2
        

def weights_init(net):
    if type(net) == nn.Conv1d:
        net.weight.data.normal_(0.0, 0.01)
        
    if type(net) == nn.Embedding:
        net.weight.data.uniform_(-0.1, 0.1)
        
    if type(net) == nn.Linear:
        net.weight.data.normal_(0.0, 0.01)


def reader(file):
    '''
    This function reads the train sequences, binning it by creating a bin 
    for each sequence length. Then, encoding the sequences in each bin using the
    dictionary and return a Python dictionary where the keys are the sequence 
    lengths and the values are the encoded sequences.
    All bins holding a sequence length above the maximum length and below the
    minimum length are discarded.
    '''
    X = {}
    y = {}
    with open(file, 'r') as data:
        reader = csv.reader(data, delimiter='\t')
        for row in reader:
            if len(row[0]) > args.max_seq_len:
                continue
                
            elif len(row[0]) < args.min_seq_len:
                continue
                
            else:
                if len(row[0]) not in X.keys():
                    X[len(row[0])] = []
                    y[len(row[0])] = []
                    
                X[len(row[0])].append([DICTIONARY[char] for char in row[0]])
                y[len(row[0])].append(float(row[1]))
                
    return X, y


def get_data_loaders(X, y):
    '''
    This function creates a data loader for each sequence length (bin).
    It returns a list of data loaders.
    '''
    train_loaders = []
    for seq_length in X.keys():
        train_X, train_y = torch.LongTensor(np.array(X[seq_length])), torch.tensor(np.array(y[seq_length]), dtype=torch.float32)
        train_data = torch.utils.data.TensorDataset(train_X, train_y)
        train_loaders.append(DL(train_data, batch_size=args.bs, drop_last=True))
        
    return train_loaders


def main(args):
    torch.manual_seed(args.seed)
    X, y = reader(Path(args.path, args.file))
    train_loaders = get_data_loaders(X, y)

    device = xm.xla_device()
    model = TCN(em_dim=args.em_dim, output_size=args.os, channel_size=args.c_size, max_seq_len=args.max_seq_len, 
                kernel_size=args.ks, stride=args.stride, dropout=args.dropout, em_dropout=args.em_dropout, 
                num_char=NUM_CHAR)
    model.apply(weights_init)
    model = model.to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr)
    loss_func = CCLoss()

    results = {
        'Training': {}
    }    
    
    # Training loop
    model.train()
    for epoch in range(args.epochs + 1):
        l, m1, m2, m3, count = 0, 0, 0, 0, 0

        for train_loader in train_loaders:
            for i, (x, labels) in enumerate(train_loader):
                opt.zero_grad()

                x = x.to(device)
                labels = labels.to(device)

                out = model(x)
                loss, r1, r2 = loss_func(out, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                
                opt.step()
                xm.mark_step()
                
                if i % args.step == 0:                   
                    l += loss.item()
                    m1 += r1
                    m2 += r1 ** 2
                    m3 += r2
                    count += 1
                
        results['Training'][epoch] = {
            'Loss': l / count,
            'Pearson_r': m1 / count,
            'Pearson_r^2': m2 / count,
            'Spearman_r': m3 / count
        }

        # Saving the model
        if epoch != 0 and epoch % 10 == 0:
            xm.save(model.state_dict(), Path(args.path, f'{epoch}-model.pt'))
            joblib.dump(results, Path(args.path, 'results.joblib'))


if __name__ == '__main__':
    os.environ['XRT_TPU_CONFIG'] = 'localservice;0;localhost:51011'
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default='.', help='the dataset path')
    parser.add_argument('--file', default='train_sequences.txt', help='trainset file')
    parser.add_argument('--min_seq_len', type=int, default=89, help='minimum sequence length')
    parser.add_argument('--max_seq_len', type=int, default=111, help='maximum sequence length')
    parser.add_argument('--seed', type=int, default=7, help='seed')
    parser.add_argument('--bs', type=int, default=64, help='batch size')
    parser.add_argument('--em_dim', type=int, default=100, help='embeddings dimension')
    parser.add_argument('--c_size', type=int, default=384, help='channel size')
    parser.add_argument('--ks', type=int, default=3, help='kernel size')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--dropout', type=float, default=0.1, help='TCN dropout')
    parser.add_argument('--em_dropout', type=float, default=0.1, help='embeddings dropout')
    parser.add_argument('--clip', type=float, default=0.15, help='gradient clip')
    parser.add_argument('--os', type=int, default=1, help='output size')
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--epochs', type=int, default=130, help='number of epochs')
    parser.add_argument('--step', type=int, default=2, help='number of steps to measure performance')

    args = parser.parse_args()
    main(args)
