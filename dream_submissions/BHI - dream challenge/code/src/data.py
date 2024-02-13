import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

import numpy as np
import random

# write down train data path
data_path = '../../data/dream/train_sequences.txt'

# Globals.
base2int = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
one_hot_matrix = torch.tensor([
    [1., 0., 0., 0.],
    [0., 1., 0., 0.],
    [0., 0., 1., 0.],
    [0., 0., 0., 1.],
    [0., 0., 0., 0.],
])

class DDPDatasetShift(Dataset):

    def __init__(self, df, max_length=110, infer=False, tta=1):
        """
        df: dataframe with column `sequence` and `measured_expression`.
        `sequence` will be used as input data, and `measured_expression` is used as target value.
        max_length: Any sequences shorter than this length will be padded with 'N'.
        Any sequences longer than this length will be trimmed from the last bases of the sequence.
        """
        super().__init__()
        self.records = df.to_records()
        self.max_length = max_length
        self.infer = infer
        self.std = np.std(self.records.measured_expression)
        self.tta = tta
        assert self.tta % 2 == 1

    def _get_shifted_sequence(self, seq, shift):
        if shift == 0:
            return seq
        elif shift < 0:
            shift = -shift
            return 'CGATTCGAAC'[-shift:] + seq[:-shift]
        else:
            return seq[shift:] + 'TCTTAATTAA'[:shift] # 10bp scaffold + seq + 10bp scaffold

    def _pad(self, seq):
        # Make sure that sequence length is exactly `max_length`.
        vector_left = 'GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC'
        vector_right = 'TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA'

        #### lr+sym random shift padding #### 
        # 3:4:2 ratio for right, left, both(symmetric) padding
        if len(seq) < self.max_length:
            # pad right
            if random.random() < 0.333: 
                seq = seq + vector_right[:self.max_length - len(seq)]
            # pad left
            elif random.random() > 0.666:
                seq = vector_left[-(self.max_length - len(seq)):] + seq 
            # Pad left right both
            else:                                                       
                seq_len = len(seq)
                pad_len = self.max_length-seq_len
                left_pad = pad_len//2
                right_pad = pad_len - left_pad  # right pad >= left pad

                # Right padding
                if right_pad > 0:
                    seq = seq + vector_right[:right_pad]

                # Left padding
                if left_pad > 0:
                    seq = vector_left[-left_pad:] + seq  

        elif len(seq) > self.max_length: # Trim it from the right.
            seq = seq[:self.max_length]

        return seq
    
    def __getitem__(self, i):
        """
        """
        seq, exp = self.records[i].sequence, self.records[i].measured_expression

        # Target standardization.
        exp = (exp - 11.0) / self.std

        seqs = []
        shift_range = [i - self.tta // 2 for i in range(self.tta)]
        for shift in shift_range:
            seq_shifted = self._get_shifted_sequence(seq, shift)
            seq_shifted = self._pad(seq_shifted)

            seq_shifted_int = [base2int[base] for base in seq_shifted]
            seq_shifted = one_hot_matrix[seq_shifted_int].T
            seqs.append(seq_shifted)

            seq_shifted_rc = seq_shifted.flip([0, 1])
            seqs.append(seq_shifted_rc)

        return {
            'seqs': seqs,
            'target': torch.tensor([exp]).float(), # scalar
        }

    def __len__(self):
        return len(self.records)


class DDPDataset(Dataset):

    def __init__(self, df, max_length=110, train=True):
        """
        df: dataframe with column `sequence` and `measured_expression`.
        `sequence` will be used as input data, and `measured_expression` is used as target value.
        max_length: Any sequences shorter than this length will be padded with 'N'.
        Any sequences longer than this length will be trimmed from the last bases of the sequence.
        """
        super().__init__()
        self.records = df.to_records()
        self.max_length = max_length
        self.train = train
        self.std = np.std(self.records.measured_expression)
    
    def __getitem__(self, i):
        """
        """
        seq, exp = self.records[i].sequence, self.records[i].measured_expression

        # Target standardization.
        exp = (exp - 11.0) / self.std

        # Make sure that sequence length is exactly `max_length`.
        vector_left = 'GCTAGCAGGAATGATGCAAAAGGTTCCCGATTCGAAC'
        vector_right = 'TCTTAATTAAAAAAAGATAGAAAACATTAGGAGTGTAACACAAGACTTTCGGATCCTGAGCAGGCAAGATAAACGA'
        
        #### lr+sym random shift padding #### 
        # 3:4:2 ratio for right, left, both(symmetric) padding
        if len(seq) < self.max_length:
            # Pad Right
            if random.random() < 0.333: 
                seq = seq + vector_right[:self.max_length - len(seq)] 
            # Pad Left
            elif random.random() > 0.666:
                seq = vector_left[-(self.max_length - len(seq)):] + seq 
            # Pad left right both   
            else:                                                       
                seq_len = len(seq)
                pad_len = self.max_length-seq_len
                left_pad = pad_len//2
                right_pad = pad_len - left_pad  # right pad >= left pad

                # Right padding
                if right_pad > 0:
                    seq = seq + vector_right[:right_pad]

                # Left padding
                if left_pad > 0:
                    seq = vector_left[-left_pad:] + seq          
                            
            
        elif len(seq) > self.max_length: # Trim it from the right.
            seq = seq[:self.max_length]

        # One-hot encode sequence.
        seq_int = [base2int[base] for base in seq]
        seq = one_hot_matrix[seq_int].T # Produces 4 x max_length one-hot tensor. (in fact it's one-hot except columns for 'N')

        seq_rc = seq.flip([0, 1])

        return {
            'seq': seq, # 4 x max_length
            'seq_rc': seq_rc, # 4 x max_length
            'target': torch.tensor([exp]).float(), # scalar
        }

    def __len__(self):
        return len(self.records)

if  __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv(data_path, sep='\t', names=['sequence', 'measured_expression'])
    print('Read data with shape:', df.shape)

    dataset = DDPDataset(df, max_length=110)
    loader = DataLoader(dataset, batch_size=1024)

    for d in loader:
        print(d['seq'].shape)
        print(d['seq'].dtype)

        print(d['target'].shape)
        print(d['target'].dtype)

        break
