
from typing import Iterable, Any 
from abc import ABCMeta, abstractmethod

class DataProcessor(metaclass=ABCMeta):
    @abstractmethod
    def prepare_train_dataloader(self) -> Iterable[dict[str, Any]]:
        """
        Method must return Iterable (e.g. Dataloader)
        to be used during training. 
        Each batch should be returned in the form of dict
        containing required keys "x" and "y" and some optional
        keys (if needed)
        """
        ...
    
    @abstractmethod
    def prepare_valid_dataloader(self) -> Iterable[dict[str, Any]] | None:
        """
        Method must return Iterable (e.g. Dataloader)
        to be used during validation
        Each batch should be returned in the form of dict
        containing required keys "x" and "y" and some optional
        keys (if needed)
        """
        ...
    
    @abstractmethod    
    def train_epoch_size(self) -> int:
        """
        Method must return the size of each training epoch
        """
        ...
        
    @abstractmethod
    def data_channels(self) -> int:
        """
        Method must return the number of channels in input tensor
        """
        ...
        
    @abstractmethod
    def data_seqsize(self) -> int:
        """
        Method must return the sequence size used by data processor
        """
        ...