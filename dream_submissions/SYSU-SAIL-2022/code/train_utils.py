#!/home/dingml/anaconda3/envs/expressBert/bin/python3

import argparse
import shutil
from argparse import Namespace
from collections import OrderedDict
import time
import os
import torch
import torch.nn as nn
import sys
import gzip
import numpy as np
import logging
import warnings
from io import TextIOWrapper
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple, Union
import hashlib

import tqdm


def load_fasta(fn: str, ordered: bool=False) -> Dict[str, str]:
    r"""
    load fasta as sequence dict
    Input
    -----
    fn : path to fasta file
    ordered : False - dict, True - OrderedDict

    Return
    -------
    seq_dict : Dict[str, str] or OrderedDict[str, str]
    """
    if ordered:
        fasta = OrderedDict()
    else:
        fasta = dict()
    name, seq = None, list()
    with copen(fn) as infile:
        for l in infile:
            if l.startswith('>'):
                if name is not None:
                    # print("{}\n{}".format(name, ''.join(seq)))
                    fasta[name] = ''.join(seq)
                name = l.strip().lstrip('>')
                seq = list()
            else:
                seq.append(l.strip())
    fasta[name] = ''.join(seq)
    return fasta

def hash_string(s):
    return hashlib.sha256(s.encode()).hexdigest()

## PyTorch utils
def set_seed(seed: int, use_deterministic=True):
    if float(torch.version.cuda) >= 10.2:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_deterministic:
        torch.use_deterministic_algorithms(True)

def get_device(model: nn.Module):
    return next(model.parameters()).device

def model_summary(model):
    """
    model: pytorch model
    """
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}

# pad up and down flank for a seq
REF_SEQ = "../../data/yeast_pTpA_GPRA_vector_with_N80.txt"
def pad_updownseq(seq: str, flank: int):
    with open(REF_SEQ, mode='r') as reffile:
        seq_updown = reffile.readline().lstrip().rstrip("\n").upper()
        begin = seq_updown.find('N') - flank
        end = seq_updown.find('N') + len(seq) + flank
        seq_updown = seq_updown[begin: end]
        begin = seq_updown.find('N')
        end = begin + len(seq)
        return seq_updown[:begin] + seq + seq_updown[end:]


# set therehold for expresstion
def settherehold(expr_file:str, percent: float):
    with copen(expr_file) as infile:
        expr_list = list()
        for l in tqdm(infile):
            if l.startswith(">"):
                _, expr = l.lstrip(">").rstrip("\n").split("|")[0], float(l.lstrip(">").rstrip("\n").split("|")[1])
                expr_list.append(expr)
        expr_list.sort(reverse=True)
        return expr_list[int(len(expr_list)*percent)]



## machine learning utils
def split_train_valid_test_randomly(sample_number: int, val_ratio: float, test_ratio: float, stratify: Iterable=None) -> Tuple[Iterable[int], Iterable[int], Iterable[int]]:
    r"""
    Description
    ------------
    Randomly split train/validation/test data

    Arguments
    ---------
    stratify: split by groups

    Return
    -------
    train_inds : List[int]
    val_inds : List[int]
    test_inds : List[int]
    """
    val_ratio = 0 if val_ratio is None else val_ratio
    test_ratio = 0 if test_ratio is None else test_ratio
    assert val_ratio + test_ratio > 0 and val_ratio + test_ratio < 1, "{},{}".format(val_ratio, test_ratio)
    all_inds = np.arange(sample_number)
    from sklearn.model_selection import train_test_split

    train_val_inds, test_inds = train_test_split(all_inds, test_size=test_ratio, stratify=stratify)
    val_ratio = val_ratio / (1 - test_ratio)
    if stratify is not None:
        stratify = np.asarray(stratify)[train_val_inds]
    train_inds, val_inds = train_test_split(train_val_inds, test_size=val_ratio, stratify=stratify)

    return train_inds, val_inds, test_inds

def split_train_val_test_by_group(groups: List[Any], n_splits: int, val_folds: int, test_folds: int) -> Tuple[List, List, List]:
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=n_splits)
    train_inds, val_inds, test_inds = list(), list(), list()
    for i, (_, inds) in enumerate(splitter.split(groups, groups=groups)):
        if i < val_folds:
            val_inds.append(inds)
        elif i >= val_folds and i < test_folds + val_folds:
            test_inds.append(inds)
        else:
            train_inds.append(inds)
    train_inds = np.concatenate(train_inds)
    if val_folds > 0:
        val_inds = np.concatenate(val_inds)
    if test_folds:
        test_inds = np.concatenate(test_inds)
    return train_inds, val_inds, test_inds
        

def save_checkpoint(model, optimizer, scheduler, best_acc):
    pass



## other utils
def copen(fn: str, mode='rt') -> TextIOWrapper:
    if fn.endswith(".gz"):
        return gzip.open(fn, mode=mode)
    else:
        return open(fn, mode=mode)

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir

def get_run_info(argv: List[str], args: Namespace=None) -> str:
    s = list()
    s.append("")
    s.append("##time: {}".format(time.asctime()))
    s.append("##cwd: {}".format(os.getcwd()))
    s.append("##cmd: {}".format(' '.join(argv)))
    if args is not None:
        s.append("##args: {}".format(args))
    return '\n'.join(s)

def make_logger(
        title: Optional[str]="", 
        filename: Optional[str]=None, 
        level: Literal["INFO", "DEBUG"]="INFO", 
        mode: Literal['w', 'a']='w',
        trace: bool=True, 
        **kwargs):
    if isinstance(level, str):
        level = getattr(logging, level)
    logger = logging.getLogger(title)
    logger.setLevel(level)
    sh = logging.StreamHandler()
    sh.setLevel(level)

    if trace is True or ("show_line" in kwargs and kwargs["show_line"] is True):
        formatter = logging.Formatter(
                '%(levelname)s(%(asctime)s) [%(filename)s:%(lineno)d]:%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(levelname)s(%(asctime)s):%(message)s', datefmt='%Y%m%d %H:%M:%S'
        )
    # formatter = logging.Formatter(
    #     '%(message)s\t%(levelname)s(%(asctime)s)', datefmt='%Y%m%d %H:%M:%S'
    # )

    sh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(sh)

    if filename is not None:
        if os.path.exists(filename):
            suffix = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(filename)))
            while os.path.exists("{}.conflict_{}".format(filename, suffix)):
                suffix = "{}_1".format(suffix)
            shutil.move(filename, "{}.conflict_{}.log".format(filename, suffix))
            warnings.warn("log {} exists, moved to to {}.conflict_{}.log".format(filename, filename, suffix))
        fh = logging.FileHandler(filename=filename, mode=mode)
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class LabelEncoder(object):
    def __init__(self, predefined_mapping: Dict[str, int]=dict()) -> None:
        self.mapping = predefined_mapping.copy()
        if len(self.mapping) == 0:
            self._next = 0
        else:
            self._next = max(self.mapping.values()) + 1
        self.reverse_mapping = {v:k for k, v in self.mapping.items()}
    
    def __call__(self, label) -> int:
        if label not in self.mapping:
            self.mapping[label] = self._next
            self.reverse_mapping[self._next] = label
            self._next += 1
        return self.mapping[label]

    def id2label(self, id) -> str:
        return self.reverse_mapping[id]

## DNA utils
_CHROM2INT = {
    "chr1": 1,   "1": 1,   1: 1,
    "chr2": 2,   "2": 2,   2: 2,
    "chr3": 3,   "3": 3,   3: 3,
    "chr4": 4,   "4": 4,   4: 4,
    "chr5": 5,   "5": 5,   5: 5,
    "chr6": 6,   "6": 6,   6: 6,
    "chr7": 7,   "7": 7,   7: 7,
    "chr8": 8,   "8": 8,   8: 8,
    "chr9": 9,   "9": 9,   9: 9,
    "chr10": 10, "10": 10, 10: 10,
    "chr11": 11, "11": 11, 11: 11,
    "chr12": 12, "12": 12, 12: 12,
    "chr13": 13, "13": 13, 13: 13,
    "chr14": 14, "14": 14, 14: 14,
    "chr15": 15, "15": 15, 15: 15,
    "chr16": 16, "16": 16, 16: 16,
    "chr17": 17, "17": 17, 17: 17,
    "chr18": 18, "18": 18, 18: 18,
    "chr19": 19, "19": 19, 19: 19,
    "chr20": 20, "20": 20, 20: 20,
    "chr21": 21, "21": 21, 21: 21,
    "chr22": 22, "22": 22, 22: 22,
    "chrX": 23,  "X": 23,
    "chrY": 24,  "Y": 24,
    "chrM": 25,  "M": 25}

class Chrom2Int(LabelEncoder):
    def __init__(self, predefined_mapping=_CHROM2INT) -> None:
        super().__init__(predefined_mapping)

NN_COMPLEMENT = {
    'A': 'T', 'a': 't',
    'C': 'G', 'c': 'g',
    'G': 'C', 'g': 'c',
    'T': 'A', 't': 'a',
    'R': 'Y', 'r': 'y',
    'Y': 'R', 'y': 'r',
    'S': 'S', 's': 's',
    'W': 'W', 'w': 'w',
    'K': 'M', 'k': 'm',
    'M': 'K', 'm': 'k',
    'B': 'V', 'b': 'v',
    'D': 'H', 'd': 'h',
    'H': 'D', 'h': 'd',
    'V': 'B', 'v': 'b',
    'N': 'N', 'n': 'n',
    '.': '.', '-': '-'  
}
def get_reverse_strand(seq):
    seq = seq[::-1]
    seq = ''.join([NN_COMPLEMENT[n] for n in seq])
    return seq

def split_chrom_start_end(chrom_start_end):
    """
    deal with chrom:start-end format
    """
    chrom, start_end = chrom_start_end.split(':')
    start, end = start_end.split('-')
    return chrom, int(start), int(end)


## 根据极差确定采样序列
from math import ceil
from tqdm import tqdm
def cal_sampler_prob(fafile):
    with copen(fafile) as infile:
        groups = dict()
        for l in infile:
            if l.startswith('>'):
                groups[int(l.rstrip('\n').split('|')[2])] = list()
        infile.seek(0)
        for l in tqdm(infile):
            if l.startswith('>'):
                groups[int(l.rstrip('\n').split('|')[2])].append(float(l.strip().split('|')[1]))
        prob_dic = dict()
        for key in tqdm(groups.keys()):
            if len(groups[key]) >= 2:
                prob_dic[key] = int(ceil(np.array(groups[key]).ptp())**2)
            else:
                prob_dic[key] = 1
        prob = list()
        infile.seek(0)
        for l in tqdm(infile, desc="produce sampler prob:"):
            if l.startswith('>'):
                if l.startswith('>'):
                    group = int(l.split('|')[2])
                    prob.append(prob_dic[group])
    return prob

import random
def set_rand_seed(seed=1, backends=True):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = backends
    torch.backends.cudnn.benchmark = backends
    torch.backends.cudnn.deterministic = not backends

def _split_train_valid_test(groups, train_keys, valid_keys, test_keys=None):
    """
    groups: length N, the number of samples
    train
    """
    assert isinstance(train_keys, list)
    assert isinstance(valid_keys, list)
    assert test_keys is None or isinstance(test_keys, list)
    index = np.arange(len(groups))
    train_idx = index[np.isin(groups, train_keys)]
    valid_idx = index[np.isin(groups, valid_keys)]
    if test_keys is not None:
        test_idx = index[np.isin(groups, test_keys)]
        return train_idx, valid_idx, test_idx
    else:
        return train_idx, valid_idx


def split_train_valid_test(sample_number: int, val_ratio: float, test_ratio: float, stratify: Iterable = None) -> Tuple[
    Iterable[int], Iterable[int], Iterable[int]]:
    r"""
    Description
    ------------
    Randomly split train/validation/test data

    Arguments
    ---------
    stratify: split by groups

    Return
    -------
    train_inds : List[int]
    val_inds : List[int]
    test_inds : List[int]
    """
    val_ratio = 0 if val_ratio is None else val_ratio
    test_ratio = 0 if test_ratio is None else test_ratio
    assert val_ratio + test_ratio > 0 and val_ratio + test_ratio < 1, "{},{}".format(val_ratio, test_ratio)
    all_inds = np.arange(sample_number)
    from sklearn.model_selection import train_test_split

    train_val_inds, test_inds = train_test_split(all_inds, test_size=test_ratio, stratify=stratify)
    val_ratio = val_ratio / (1 - test_ratio)
    if stratify is not None:
        stratify = np.asarray(stratify)[train_val_inds]
    train_inds, val_inds = train_test_split(train_val_inds, test_size=val_ratio, stratify=stratify)

    return train_inds, val_inds, test_inds


def split_train_val_test_by_group(groups: List[Any], n_splits: int, val_folds: int, test_folds: int) -> Tuple[
    List, List, List]:
    from sklearn.model_selection import GroupKFold
    splitter = GroupKFold(n_splits=n_splits)
    train_inds, val_inds, test_inds = list(), list(), list()
    for i, (_, inds) in enumerate(splitter.split(groups, groups=groups)):
        if i < val_folds:
            train_inds.append(inds)
        elif i >= val_folds and i < test_folds:
            val_inds.append(inds)
        else:
            test_inds.append(inds)
    train_inds = np.concatenate(train_inds)
    if val_folds > 0:
        val_inds = np.concatenate(val_inds)
    if test_folds:
        test_inds = np.concatenate(test_inds)
    return train_inds, val_inds, test_inds


