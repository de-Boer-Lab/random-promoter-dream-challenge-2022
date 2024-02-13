import json
from collections import Counter
import warnings

from tqdm.auto import tqdm

import random
import math
import numpy as np
import pandas as pd
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.set_option('mode.chained_assignment',  None) 

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


"""
230406 BHIDataloaderWrapper, BHI_preprocess_df, BHI_preprocess_data, BHI_Seq2Tensor edited
- Long sequences trimming from right
- Short sequences padding at left/right/both sides
- one-hot encoding for N with 0 0 0 0

"""

class BHIDataloaderWrapper:
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


def BHI_preprocess_data(data: pd.DataFrame, 
                    seqsize: int,
                    plasmid_path: str | Path):
    """
    230407 BHI edited
    - Short training sequences are padded on the 3-end, 5-end, and both with nucleotides from the vector sequence to the uniform total length. (110bp)
        - 3:4:2 ratio
    - Long training sequences are trimmed from right
    """    
    vector_left = 'GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC'
    vector_right = 'TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA'
    seq_idx = data.columns.get_loc('seq')
    for i in tqdm(range(0, len(data))):
        # Trim from right (BHI ver.)
        if len(data.iloc[i,seq_idx]) > 110 :
            data.iloc[i,seq_idx] = data.iloc[i,seq_idx][:110]

        #### lr+sym random shift padding (BHI ver.) #### 
        # 3:4:2 ratio for right, left, both(symmetric) padding          
        elif len(data.iloc[i,seq_idx]) < 110 :
            # pad right 
            if random.random() < 0.333:
                data.iloc[i,seq_idx] = data.iloc[i,seq_idx] + vector_right[:110-len(data.iloc[i,seq_idx])]
            # pad left
            elif random.random() > 0.666:
                data.iloc[i,seq_idx] = vector_left[-(110-len(data.iloc[i,seq_idx])):] + data.iloc[i,seq_idx]
            # pad left right both, symmetrically
            else:
                data_len = len(data.iloc[i,seq_idx])
                pad_len = 110 - data_len
                left_pad = pad_len//2
                right_pad = pad_len - left_pad    # right_pad >= left_pad

                # pad right
                if right_pad > 0:
                    data.iloc[i,seq_idx] = data.iloc[i,seq_idx] + vector_right[:right_pad]
                # pad left
                if left_pad > 0:
                    data.iloc[i,seq_idx] = vector_left[-left_pad:] + data.iloc[i,seq_idx] 

    return data

class BHI_Seq2Tensor(nn.Module):
    '''
    Encode sequences using one-hot encoding after preprocessing.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, seq: str) -> torch.Tensor:
        seq_i = [n2id(x) for x in seq]
        code = torch.from_numpy(np.array(seq_i))
        code = F.one_hot(code, num_classes=5) # 5th class is N
        
        """
        230407 BHI edited
        one-hot encode N with 0 0 0 0
        """
        code[code[:, 4] == 1] = 0 # 0.25 => 0 # encode Ns with .25
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

def BHI_preprocess_df(path: str | Path,  
                  seqsize: int, 
                  plasmid_path: str |  Path):
    df = pd.read_table(path, sep='\t', header=None) 
    df.columns = ['seq', 'bin', 'fold'][:len(df.columns)]
    df = BHI_preprocess_data(df, 
                         seqsize=seqsize, 
                         plasmid_path=plasmid_path)
    """
    230407 BHI edited df
    """
    # df = add_singleton_column(df)
    # df = add_rev(df)
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