import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
import janggu
import pandas as pd
import numpy as np
import sonnet as snt
from tqdm import tqdm
import time
import os

import glob
import json
import functools
import tensorflow_probability as tfp
import h5py
import sys

train = pd.read_csv("train_sequences.txt",sep="\t",header=None)
train.shape


def write_fasta(file_name,myDict):
	out = open(file_name,"wt")
	for k in myDict:
		out.write(">"+str(k)+"\n")
		out.write(myDict[k]+"\n")
	out.close()
def resize_DNA(janggu_X,n,seq_len):
	train_X=np.reshape(janggu_X,(n,seq_len,4))
	return train_X
seq_length=110
f1=f"train_sequences.fa"
write_fasta(f1,train[0].to_dict())
train_X = janggu.data.Bioseq.create_from_seq(f1,f1,fixedlen=seq_length)
train_X = resize_DNA(train_X,train.shape[0],seq_length)
train_Y = np.reshape(train[1].tolist(),(train.shape[0],1))



f1 = h5py.File(f"all_train_data.h5", "w")
dset1 = f1.create_dataset("train_X", train_X.shape, data=train_X)
dset2 = f1.create_dataset("train_Y", train_Y.shape, data=train_Y)
f1.close()

test = pd.read_csv("test_sequences.txt",sep="\t",header=None)
test.shape

seq_length=110
f1=f"test_sequences.fa"
write_fasta(f1,test[0].to_dict())
test_X = janggu.data.Bioseq.create_from_seq(f1,f1,fixedlen=seq_length)
test_X = resize_DNA(test_X,test.shape[0],seq_length)

f1 = h5py.File(f"test_sequences.h5", "w")
dset1 = f1.create_dataset("test_X", test_X.shape, data=test_X)
f1.close()



