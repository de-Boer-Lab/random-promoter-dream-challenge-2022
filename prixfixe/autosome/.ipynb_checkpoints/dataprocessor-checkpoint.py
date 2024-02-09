from torch import Generator
from torch.utils.data import DataLoader
from pathlib import Path

from ..prixfixe import DataProcessor
from .utils import (preprocess_df,
                            DataloaderWrapper)
from .dataset import SeqDatasetProb


class AutosomeDataProcessor(DataProcessor):
    def __init__(
        self, 
        seqsize: int, 
        path_to_training_data: str | Path, 
        path_to_validation_data: str | Path | None,
        plasmid_path: str | Path,
        generator: Generator,
        train_batch_size: int = 1024,
        batch_per_epoch: int=1000,
        train_workers: int=8,
        shuffle_train: bool=True,
        valid_batch_size: int=4096,
        valid_workers: int=8,
        shuffle_val: bool=False
    ):
        self.train = preprocess_df(path=path_to_training_data,
                                   seqsize=seqsize,
                                   plasmid_path=plasmid_path)
        if path_to_validation_data is not None:
            self.valid = preprocess_df(path=path_to_validation_data,
                                   seqsize=seqsize,
                                   plasmid_path=plasmid_path)
        else:
            self.valid = None

        self.train_batch_size=train_batch_size
        self.batch_per_epoch=batch_per_epoch
        self.train_workers=train_workers
        self.shuffle_train=shuffle_train

        self.valid_batch_size=valid_batch_size
        self.valid_workers=valid_workers
        self.shuffle_val=shuffle_val
        #self.batch_per_valid=batch_per_valid
        
        self.seqsize = seqsize
        self.plasmid_path = plasmid_path
        self.generator = generator

    def prepare_train_dataloader(self):
        train_ds = SeqDatasetProb(
            self.train, 
            seqsize=self.seqsize,
        )
        train_dl = DataLoader(
            train_ds, 
            batch_size=self.train_batch_size,
            num_workers=self.train_workers,
            shuffle=self.shuffle_train,
            generator=self.generator
        ) 
        train_dl = DataloaderWrapper(train_dl, self.batch_per_epoch)
        return train_dl

    def prepare_valid_dataloader(self):
        if self.valid is None:
            return None
        valid_ds = SeqDatasetProb(
            self.valid, 
            seqsize=self.seqsize,
        )
        valid_dl = DataLoader(
            valid_ds, 
            batch_size=self.valid_batch_size,
            num_workers=self.valid_workers,
            shuffle=self.shuffle_val
        ) 
        return valid_dl
    
    def train_epoch_size(self) -> int:
        return self.batch_per_epoch
    
    def data_channels(self) -> int:
        return 6 # 4 - onehot, 1 - singleton, 1 - is_reverse
    
    def data_seqsize(self) -> int:
        return self.seqsize
