import pandas as pd
import numpy as np
from multiprocessing import Pool


def transform_data_5(seq):
    tmp = [np.array(list(seq[(17 + i):(-13 + i)])).reshape(-1, 1) for i in range(-2, 3)]
    tmp = np.concatenate(tmp, axis=1)
    tmp = ' '.join([''.join(x) for x in tmp])
    return tmp


data = pd.read_csv('./data/train_sequences.txt', header=None, sep='\t')
data = data[data.iloc[:, 0].apply(len) == 110]
data = data[data.iloc[:, 0].apply(lambda x: "N" not in x)]


data.iloc[:, 1] = data.iloc[:, 1].apply(round)
with Pool(16) as p:
    tmp = np.array(p.map(transform_data_5, data.iloc[:, 0].values))
data.iloc[:, 0] = tmp
data.iloc[:, 0].to_csv('./glove/text8', index=False, header=False)


