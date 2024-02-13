import tensorflow as tf
import sonnet as snt
import enformer5
import importlib
importlib.reload(enformer5)
import tensorflow_probability as tfp
import pandas as pd
import numpy as np
import h5py
import sys

# exec(open("final_submission.py").read())
try:
	final_model_dir = sys.argv[1]
except:
	final_model_dir="epoch_14_model"

# load model
trained_model = tf.saved_model.load(final_model_dir)
f=trained_model.signatures["serving_default"]

# load test data
f2 = h5py.File('test_sequences.h5', 'r')
test_X = f2['test_X'][:]
# test_X = test_X[:,15:100,:]
test_X = tf.convert_to_tensor(test_X,dtype=tf.float32)
f2.close()

# load test DNA sequence
df = pd.read_csv("test_sequences.txt",sep="\t",header=None)

# make prediction
# out = f(args_0=test_X)['yeast']
# out = f(test_X)['human']
# out_list = np.reshape(out.numpy(),(71103,))
out_list = []
n=10000
for i in range(8):
    # print (i*n,(i+1)*n,test_X[i*n:(i+1)*n,:,:].shape)
    tmp = f(test_X[i*n:(i+1)*n,:,:])['human']
    tmp = np.reshape(tmp.numpy(),(tmp.shape[0],))
    out_list.append(tmp)
out_list = np.concatenate(out_list)

# combine DNA sequence and prediction
df[1] = out_list

# save to final submission
df.to_csv("final_submission.txt",sep="\t",header=False,index=False)

