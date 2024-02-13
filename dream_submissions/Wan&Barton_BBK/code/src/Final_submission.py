'''
This code is modefied based on a TCN impelemtation by https://github.com/locuslab/TCN
'''

import csv
import json
import math
import joblib
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


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
        
        # Network depth is defined based on the longest sequence       
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
    

def reader(file):
    '''
    This function reads the test sequences, encoding it using the
    dictionary and return a tensor of shape (71103, 110).
    '''
    X = []
    with open(file, 'r') as data:
        reader_ = csv.reader(data, delimiter='\t')
        for row in reader_:
            X.append([DICTIONARY[char] for char in row[0]])
                
    return torch.LongTensor(X)


def main(args):
    test_X = reader(Path(args.path, args.file))
    params = torch.load(Path(args.path, 'model.pt'))
    model = TCN(em_dim=args.em_dim, output_size=args.os, channel_size=args.c_size, max_seq_len=args.max_seq_len, 
                kernel_size=args.ks, stride=args.stride, dropout=args.dropout, em_dropout=args.em_dropout, 
                num_char=NUM_CHAR)
    model.load_state_dict(params)
    model.eval()
    
    with torch.no_grad():
        y_pred = model(test_X).detach().numpy()

    sequences = []
    with open(Path(args.path, args.file), 'r') as file:
        reader_ = csv.reader(file, delimiter='\t')
        for seq, _ in reader_:
            sequences.append(seq)

    with open('submission.txt', 'w') as file:
        for seq, pred in zip(sequences, y_pred):
            file.write(seq + '\t' + str(pred) + '\n')
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--path', default='.', help='the dataset path')
    parser.add_argument('--file', default='test_sequences.txt', help='testset file')
    parser.add_argument('--max_seq_len', type=int, default=111, help='maximum sequence length')
    parser.add_argument('--em_dim', type=int, default=100, help='embeddings dimension')
    parser.add_argument('--c_size', type=int, default=384, help='channel size')
    parser.add_argument('--ks', type=int, default=3, help='kernel size')
    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--dropout', type=float, default=0.1, help='TCN dropout')
    parser.add_argument('--em_dropout', type=float, default=0.1, help='embeddings dropout')
    parser.add_argument('--os', type=int, default=1, help='output size')

    args = parser.parse_args()
    main(args)
