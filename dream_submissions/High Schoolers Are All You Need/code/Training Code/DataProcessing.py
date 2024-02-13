import numpy as np
import pandas as pd
import h5py
import os
import matplotlib.pyplot as plt




with open('train_sequences.txt') as handle:
    raw_data = pd.read_csv(handle, sep='\n', header=None).to_numpy()
    
length = []
seqs = []
labels = []
for i in range(len(raw_data)):    
    seq, label = raw_data[i][0].split('\t')
    seqs.append(seq)
    labels.append(float(label))
    length.append(len(seq))
    
seqs = np.array(seqs)
length = np.array(length)
labels = np.array(labels)




with open('test_sequences.txt') as handle:
    raw_data = pd.read_csv(handle, sep='\n', header=None).to_numpy()
    
test_length = []
test_seqs = []
test_labels = []
for i in range(len(raw_data)):    
    seq, label = raw_data[i][0].split('\t')
    test_seqs.append(seq)
    test_labels.append(float(label))
    test_length.append(len(seq))
    

test_seqs = np.array(test_seqs)
test_length = np.array(test_length)
test_labels = np.array(test_labels)




min_length = 0
max_length = 111
index = np.where((length >= min_length)&(length <= max_length))[0]
filter_seqs = seqs[index]
filter_labels = labels[index]





def convert_one_hot(sequences, max_len, alphabet="ACGTN") -> np.ndarray:

    a_dict = {}
    for i, a in enumerate(alphabet):
        a_dict[a] = i

    # Make an integer array from the string array.
    one_hot = np.zeros((len(sequences), max_len, len(alphabet)), dtype=np.uint8)
    
    for n, seq in enumerate(sequences):
        remainder = max_len - len(seq)
        if remainder:
            seq += 'N'*remainder
        for l, s in enumerate(seq):
            one_hot[n,l,a_dict[s]] = 1

    return one_hot

one_hot = convert_one_hot(filter_seqs, max_len=max_length)
test_one_hot = convert_one_hot(test_seqs, max_len=max_length)




with h5py.File('compressed_dataset_111.h5', 'w') as hf:
    hf.create_dataset('x_train', data=one_hot, dtype='int8', compression="gzip")
    hf.create_dataset('y_train', data=filter_labels.astype(np.float32), dtype='float32', compression="gzip")
    hf.create_dataset('x_test', data=test_one_hot, dtype='int8', compression="gzip")
    











