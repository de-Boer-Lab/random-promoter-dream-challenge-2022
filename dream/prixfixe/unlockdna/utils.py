import json
from collections import Counter
import warnings

from tqdm.auto import tqdm

import random
import math
import numpy as np
import pandas as pd
from Bio.Seq import Seq
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.set_option('mode.chained_assignment',  None) 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Generator
from pathlib import Path 

ALPHABETS = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, 'M': 5}

class UnlockDNADataloaderWrapper:
    def __init__(self,
                 dataloader: DataLoader,
                 batch_per_epoch: int):
        self.batch_per_epoch = batch_per_epoch
        self.dataloader = dataloader
        self.iterator = iter(dataloader)

    def __len__(self):
        return self.batch_per_epoch
    
    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)

    def __iter__(self):
        for _ in range(self.batch_per_epoch):
            try:
                yield next(self.iterator)
            except StopIteration:
                self.iterator = iter(self.dataloader)

def UnlockDNA_preprocess_data(df: pd.DataFrame, 
                    head_len: int = 20,
                    tail_len: int = 20,
                    max_width: int = 100,
                    mode: str = "train"):
    
    df['len'] = df['seq'].str.len()
    df['seq'] = df['seq'].str[head_len:]
    df['seq'] = df['seq'].str[:-tail_len]
    df['len'] = df['seq'].str.len()
    df = df[df['len'] <= max_width]
    df['seq'] = df['seq'].str.pad(width = max_width, side = 'both', fillchar = 'N')
    df['seq'] = df['seq'] + df['seq'].apply(lambda x: str(Seq(x).reverse_complement())).astype('string')

    dna = np.empty((0, max_width * 2), np.uint8)

    for x in np.array_split(df['seq'], 10): # split data into chunks
        # print('splitting data')
        y = np.array(x.apply(list))
        y = np.vstack(y)
        y = np.vectorize(ALPHABETS.get)(y)
        y = y.astype(np.uint8)
        dna = np.append(dna, y, axis = 0)

    df['seq'] = list(map(list, np.array(dna)))
    df["seq"] = df["seq"].apply(lambda x: np.array(x, dtype=np.uint8))
    df['seq'] = df["seq"].to_numpy()

    expression = df['expression'].to_numpy()

    if mode == "train":
        expression_std = np.std(expression)
        expression_mean = np.mean(expression)
        expression = (expression - expression_mean) / expression_std

    df['expression'] = expression

    return df

def UnlockDNA_preprocess_df(path: str | Path,
                    head_len: int = 20,
                    tail_len: int = 20,
                    max_width: int = 100,
                    mode: str = "train"
                  ):
    
    df = pd.read_table(path, sep='\t', header=None)
    df.columns = ['seq', 'expression']
    
    df = UnlockDNA_preprocess_data(df,
                            head_len=head_len,
                            tail_len=tail_len,
                            max_width=max_width,
                            mode = mode)
    return df

def initialize_weights(m: nn.Module, generator: Generator):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n), generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001, generator=generator)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)