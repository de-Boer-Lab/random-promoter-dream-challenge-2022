import pandas as pd
import numpy as np
import h5py

df = pd.read_csv('test_sequences.txt', delimiter = "\t", header=None)
print(df.head())
df.columns = ['sequences', 'expression']
df = df.drop(['expression'], axis=1)

with h5py.File(('test_seq_onehot.h5'), 'r') as hf:
    X = hf['test_seq_onehot'][:]

from tensorflow import keras
model = keras.models.load_model('model.h5')

predictions = model.predict(X)
predictions = pd.DataFrame(predictions)

df = pd.concat([df, predictions], axis=1)
print(df.head())

df.to_csv('test_predictions.txt', sep = "\t", header=None, index=False)



