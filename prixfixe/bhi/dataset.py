import torch 
import scipy 
import numpy as np 
import pandas as pd 

from typing import ClassVar
from torch.utils.data import Dataset

from .utils import BHI_Seq2Tensor

"""
230407 BHISeqDatasetProb edited
- use_single_channel, use_reverse_channel, use_multisubstate_channel => False
- Target standardization
"""
class BHISeqDatasetProb(Dataset):
    POINTS: ClassVar[np.ndarray] =  np.array([-np.inf, *range(1, 18, 1), np.inf])

    """ Sequence dataset. """
    
    def __init__(self, 
                 ds: pd.DataFrame, 
                 seqsize: int, 
                 use_single_channel: bool=False,
                 use_reverse_channel: bool=False, 
                 use_multisubstate_channel: bool=False, 
                 shift: float=0.5, 
                 scale: float=0.5):
        """
        Parameters
        ----------
        ds : pd.DataFrame
            Training dataset.
        seqsize : int
            Constant sequence length.
        use_single_channel : bool
            If True, additional binary channel with singleton information is used.
        use_reverse_channel : bool
            If True, additional reverse augmentation is used.
        use_multisubstate_channel : bool
            If True, additional substrate channel is used.
        shift : float, optional
            Assumed sd of real expression normal distribution.
        scale : float, optional
            Assumed scale of real expression normal distribution.
        """
        self.ds = ds
        self.seqsize = seqsize
        self.totensor = BHI_Seq2Tensor()
        self.shift = shift
        self.scale = scale
        self.use_single_channel = use_single_channel
        self.use_reverse_channel = use_reverse_channel
        self.use_multisubstate_channel = use_multisubstate_channel
        
        ## added std for train y label
        self.std = np.std(self.ds.bin.values)
        
    def transform(self, x: str) -> torch.Tensor:
        assert len(x) == self.seqsize
        return self.totensor(x)
    
    def __getitem__(self, i: int):
        """
        Output
        ----------
        X: torch.Tensor    
            Create one-hot encoding tensor with reverse and singleton channels if required.
        probs: np.ndarray
            Given a measured expression, we assume that the real expression is normally distributed
            with mean=`bin` and sd=`shift`. 
            Resulting `probs` vector contains probabilities that correspond to each class (bin).     
        bin: float 
            Training expression value
        """
        if len(self.ds.seq.values[i]) != 110:
            print(f"ds.seq.values {self.ds.seq.values[i]}")
        seq = self.transform(self.ds.seq.values[i]) # type: ignore
        to_concat = [seq]
        
        # add reverse augmentation channel
        if self.use_reverse_channel:
            rev = torch.full( (1, self.seqsize), self.ds.rev.values[i], dtype=torch.float32) # type: ignore
            to_concat.append(rev)
            
        # add singleton channel
        if self.use_single_channel:
            single = torch.full( (1, self.seqsize) , self.ds.is_singleton.values[i], dtype=torch.float32) # type: ignore
            to_concat.append(single)
            
        # add multiclass channel
        if self.use_multisubstate_channel:
            substrate = torch.full( (1, self.seqsize) , self.ds.substrate.values[i], dtype=torch.float32) # type: ignore
            to_concat.append(substrate)
        
        # create final tensor
        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq
            
        bin = self.ds.bin.values[i]
        
        # generate probabilities corresponding to each class
        norm = scipy.stats.norm(loc=bin + self.shift, #type: ignore
                                scale=self.scale)
        """
        230407 BHI edited
        - bin for y labels
        - bin = (bin - 11.0)/self.std
        """
        bin = (bin - 11.0) / self.std
        
        cumprobs = norm.cdf(self.POINTS)
        probs = cumprobs[1:] - cumprobs[:-1]
        return {"x": X.float(), 
                "y_probs": np.asarray(probs, dtype=np.float32),
                ## y => used for loss calculation
                "y": bin.astype(np.float32) # type: ignore
                }
    
    def __len__(self) -> int:
        return len(self.ds.seq)
