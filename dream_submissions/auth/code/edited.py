import pandas as pd
import numpy as np
from tqdm import tqdm
import h5py

df = pd.read_csv('train_sequences.txt', delimiter = "\t", header=None)
df.head()
df.columns = ['sequences', 'expression']
df['sequences'] = df['sequences'].astype('str')
df['expression'] = df['expression'].astype('float')
mask = (df['sequences'].str.len() == 110)
df = df.loc[mask]
df = df[~df["sequences"].str.contains('N')]
df['sequences'] = df['sequences'].str[len('TGCATTTTTTTCACATC'):]
df['sequences'] = df['sequences'].str[:-len('GGTTACGGCTGTT')]

df.to_csv('train_sequences_edited.txt', sep = "\t", header=None, index=False)

sequences = df['sequences'].tolist()

def seq2feature(data):
    A_onehot = np.array([1, 0, 0, 0])
    C_onehot = np.array([0, 1, 0, 0])
    G_onehot = np.array([0, 0, 1, 0])
    T_onehot = np.array([0, 0, 0, 1])

    mapper = {'A': A_onehot, 'C': C_onehot, 'G': G_onehot, 'T': T_onehot}
    worddim = len(mapper['A'])

    # data = np.asarray(data)
    # transformed = np.zeros([len(data),len(data[0]),4] , dtype=np.bool )
    # for i in (range(len(data))) :
    #    for j,k in enumerate(data[i]):
    #        transformed[i,j] = mapper[k]
    transformed = np.asarray(([[mapper[k] for k in (data[i])] for i in tqdm(range(len(data)))]))
    return transformed

seq_onehot = seq2feature(sequences)
print(seq_onehot.shape)
with h5py.File(('seq_onehot.h5'), 'w') as hf:
    hf.create_dataset("seq_onehot",  data=seq_onehot)

expression = df['expression'].tolist()
expression = np.asarray(expression)
expression = np.reshape(expression, (np.shape(expression)[0], 1))
print(expression.shape)
with h5py.File(('expression.h5'), 'w') as hf:
    hf.create_dataset("expression", data=expression)
