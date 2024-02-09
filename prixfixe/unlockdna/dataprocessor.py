from torch import Generator
from torch.utils.data import DataLoader
from pathlib import Path
import pandas as pd

from ..prixfixe import DataProcessor
from .utils import UnlockDNA_preprocess_df, UnlockDNADataloaderWrapper
from .dataset import UnlockDNASeqDatasetProb

class UnlockDNADataProcessor(DataProcessor):
    def __init__(
        self,
        path_to_training_data: str | Path, 
        path_to_validation_data: str | Path,
        generator: Generator,
        head_len: int = 20,
        tail_len: int = 20,
        max_width: int = 100,
        train_batch_size: int = 1024,
        train_workers: int=8,
        shuffle_train: bool=True,
        valid_batch_size: int=4096,
        valid_workers: int=8,
        shuffle_val: bool=False
    ):
        # if train, valid path is given, use this
        self.train = UnlockDNA_preprocess_df(path=path_to_training_data,
                                   head_len= head_len,
                                   tail_len= tail_len,
                                   max_width= max_width,
                                   mode = "train")
                
        self.valid = UnlockDNA_preprocess_df(path=path_to_validation_data,
                                   head_len= head_len,
                                   tail_len= tail_len,
                                   max_width= max_width,
                                   mode = "valid")

        self.train_batch_size=train_batch_size
        self.batch_per_epoch= len(self.train)//train_batch_size + 1
        self.train_workers=train_workers
        self.shuffle_train=shuffle_train

        self.valid_batch_size=valid_batch_size
        self.valid_workers=valid_workers
        self.shuffle_val=shuffle_val
        self.batch_per_valid=len(self.valid)//valid_batch_size + 1
        self.generator = generator
        self.max_width = max_width

    def prepare_train_dataloader(self):
        train_ds = UnlockDNASeqDatasetProb(
            self.train)
        train_dl = DataLoader(
            train_ds, 
            batch_size=self.train_batch_size,
            num_workers=self.train_workers,
            shuffle=self.shuffle_train,
            generator=self.generator
        )
        train_dl = UnlockDNADataloaderWrapper(train_dl, self.batch_per_epoch)
        return train_dl

    def prepare_valid_dataloader(self):
        valid_ds = UnlockDNASeqDatasetProb(
            self.valid)
        valid_dl = DataLoader(
            valid_ds, 
            batch_size=self.valid_batch_size,
            num_workers=self.valid_workers,
            shuffle=self.shuffle_val
        ) 
        valid_dl = UnlockDNADataloaderWrapper(valid_dl, self.batch_per_valid)
        return valid_dl
    
    def train_epoch_size(self) -> int:
        return self.batch_per_epoch
    
    def data_channels(self) -> int:
        return 6  # A, C, G, T, N, M
    
    def data_seqsize(self) -> int:
        return self.max_width*2