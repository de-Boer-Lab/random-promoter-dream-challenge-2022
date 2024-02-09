import json
from collections import Counter

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import Generator
from pathlib import Path 

CODES: dict[str, int] = {
    "A": 0,
    "T": 3,
    "G": 1,
    "C": 2,
    'N': 4
}

def n2id(n: str) -> int:
    return CODES[n.upper()]


class DataloaderWrapper:
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

def preprocess_data(data: pd.DataFrame, 
                    seqsize: int,
                    plasmid_path: str | Path):
    '''
    Training sequences are padded on the 5-end with nucleotides 
    from the corresponding plasmid to the uniform total length.
    '''
    left_adapter = "TGCATTTTTTTCACATC"
    with open(plasmid_path) as json_file:
        plasmid = json.load(json_file)

    data = data.copy()
    INSERT_START = plasmid.find('N' * 80)
    
    #take the left part of the plasmid
    add_part = plasmid[INSERT_START-seqsize:INSERT_START]
    
    # cut left adapter and append the plasmid part
    data.seq = data.seq.apply(lambda x:  add_part + x[len(left_adapter):])
    
    # reduce sequence size to seqsize
    data.seq = data.seq.str.slice(-seqsize, None)
    return data

class Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, seq: str) -> torch.Tensor:
        seq_i = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq_i))
        code = F.one_hot(code, num_classes=5) # 5th class is N
        
        code = code[:, :4].float() 
        return code.transpose(0, 1)

def infer_singleton(arr, method: str="integer"):
    if method == "integer":
        return np.array([x.is_integer() for x in arr])
    elif method.startswith("threshold"):
        th = float(method.replace("threshold", ""))
        cnt = Counter(arr)
        return np.array([cnt[x] >= th for x in arr])
    else:
        raise Exception("Wrong method")

def add_singleton_column(df: pd.DataFrame, method: str="integer") -> pd.DataFrame:
    df = df.copy()
    df["is_singleton"] = infer_singleton(df.bin.values,method)
    return df 

COMPL = {
    'A': 'T',
    'T': 'A',
    'G': 'C',
    'C': 'G',
    'N': 'N'
}

def n2compl(n: str) -> str:
    return COMPL[n.upper()]

def revcomp(seq: str):
    return "".join((n2compl(x) for x in reversed(seq)))

def add_rev(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    revdf = df.copy()
    revdf['seq'] = df.seq.apply(revcomp)
    df['rev'] = 0
    revdf['rev'] = 1
    df = pd.concat([df, revdf]).reset_index(drop=True)
    return df

def preprocess_df(path: str | Path,  
                  seqsize: int, 
                  plasmid_path: str |  Path):
    df = pd.read_table(path, sep='\t', header=None) 
    df.columns = ['seq', 'bin', 'fold'][:len(df.columns)]
    df = preprocess_data(df, 
                         seqsize=seqsize, 
                         plasmid_path=plasmid_path)
    df = add_singleton_column(df)
    df = add_rev(df)
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