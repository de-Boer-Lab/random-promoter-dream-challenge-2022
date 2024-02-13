# STAGE-1: Read, Preprocess & Save The Data
# Imports
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

#print('Completed importing necessary libraries and utilities.')

# Start Timer
#start_time = time.time()

# Variables to be defined
ALLOWED_ALPHABETS = 'ATGCN'
OPTLEN = 110

#print('Completed defining global variables.')

# Functions to be defined
def padding(seq, scope=OPTLEN):
    l = len(seq)
    diff = l - scope
    if diff > 0:
        trunc = int(diff/2)
        if diff %2==0:
            newseq = seq[trunc:-trunc]
        else:
          if diff==1:
            newseq = seq[trunc+1:]
          else:
            newseq = seq[trunc+1:-trunc]
    else:
        diff = abs(diff)
        pad = int(diff/2)
        if diff %2==0:
            pad_str = 'N'*pad
            newseq = pad_str + seq + pad_str
        else:
            l_str = 'N'*(pad+1)
            r_str = 'N'*pad
            newseq = l_str + seq + r_str
    return newseq

def OneHotv2(seq):
  ALLOWED_CATS = ['A','T','G','C','N']
  seq_array = np.array(list(seq))
  categories = [ALLOWED_CATS]
  onehot_encoder = OneHotEncoder(categories = categories, sparse=False)
  seq_array = seq_array.reshape(len(seq_array), 1)
  onehot_encoded_seq = onehot_encoder.fit_transform(seq_array)
  return onehot_encoded_seq

#print('Completed defining necessary functions.')

# Read Input Train Sequences --> Perform Padding & OneHot Encoding --> Prepare X and Y data
input_file = 'train_sequences_wHeader.txt'

df = pd.read_csv(input_file, sep = '\t')

#print('Completed reading the input train sequences.')

#print("--- %s seconds ---" % (time.time() - start_time))

processed_seqs = []

for seq in tqdm( list(df['Sequence']) ):
  processed_seqs.append( OneHotv2(padding(seq)) )

#print('Completed padding and OneHot encoding the training sequences.')

#print("--- %s seconds ---" % (time.time() - start_time))

X = np.stack(processed_seqs, axis=0)

#print('Completed stacking.')

#print("--- %s seconds ---" % (time.time() - start_time))

Y = np.array(df['Exp'].values)

#print('Y data is ready!')

#print("--- %s seconds ---" % (time.time() - start_time))

X = np.expand_dims(X, axis=3)

#print('Completed expanding the dims.')

#print('X data is ready!')

with open('Preprocesed_X_data.npy', 'wb') as f:
  np.save(f, X)

#print('X data saved successfully!')

#print("--- %s seconds ---" % (time.time() - start_time))

with open('Y_data.npy', 'wb') as f:
  np.save(f, Y)

#print('Y data saved successfully!')

#print("--- %s seconds ---" % (time.time() - start_time))

# STAGE-2: Load Preprocessed Data, Model Building, Training & Saving

# Imports
import time
import copy
import numpy as np
import tensorflow as tf
import keras
from random import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers.core import Activation, Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution1D, Convolution2D
from tensorflow.keras.layers import ConvLSTM1D, ConvLSTM2D, TimeDistributed
from keras.layers.pooling import MaxPooling1D, MaxPooling2D
from keras.regularizers import l2, L1L2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import BatchNormalization
from keras.layers import InputLayer
from keras import backend as K
from keras.initializers import RandomNormal, HeNormal
from tensorflow.python.ops import math_ops
from tensorflow.keras import initializers
from keras.layers import InputLayer, LSTM, Bidirectional
from tensorflow.keras.constraints import MaxNorm

#print('Completed importing necessary libraries and utilities.')

# Start Timer
#start_time = time.time()

# Variables to be defined
REG = 1e-4
num_outputs = 1
#VALIDATION_SPLIT = 0.1
UNITS = (32, 64, 128)

#print('Completed defining global variables.')

# Functions to be defined
def correlationMetric(x, y, axis=-2):
  """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
  x = tf.convert_to_tensor(x)
  y = math_ops.cast(y, x.dtype)
  n = tf.cast(tf.shape(x)[axis], x.dtype)
  xsum = tf.reduce_sum(x, axis=axis)
  ysum = tf.reduce_sum(y, axis=axis)
  xmean = xsum / n
  ymean = ysum / n
  xvar = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
  yvar = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
  cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
  corr = cov / tf.sqrt(xvar * yvar)
  return tf.constant(0.0, dtype=x.dtype) + corr

def correlationLoss(x, y, axis=-2):
  """Loss function that maximizes the pearson correlation coefficient between the predicted values and the labels, while trying to have the same mean and variance"""
  x = tf.convert_to_tensor(x)
  y = math_ops.cast(y, x.dtype)
  n = tf.cast(tf.shape(x)[axis], x.dtype)
  xsum = tf.reduce_sum(x, axis=axis)
  ysum = tf.reduce_sum(y, axis=axis)
  xmean = xsum / n
  ymean = ysum / n
  xsqsum = tf.reduce_sum( tf.math.squared_difference(x, xmean), axis=axis)
  ysqsum = tf.reduce_sum( tf.math.squared_difference(y, ymean), axis=axis)
  cov = tf.reduce_sum( (x - xmean) * (y - ymean), axis=axis)
  corr = cov / tf.sqrt(xsqsum * ysqsum)
  # absdif = tmean(tf.abs(x - y), axis=axis) / tf.sqrt(yvar)
  sqdif = tf.reduce_sum(tf.math.squared_difference(x, y), axis=axis) / n / tf.sqrt(ysqsum / n)
  # meandif = tf.abs(xmean - ymean) / tf.abs(ymean)
  # vardif = tf.abs(xvar - yvar) / yvar
  # return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (meandif * 0.01) + (vardif * 0.01)) , dtype=tf.float32 )
  return tf.convert_to_tensor( K.mean(tf.constant(1.0, dtype=x.dtype) - corr + (0.01 * sqdif)) , dtype=tf.float32 )

def huber_loss(y_true, y_pred):
  d = 0.15
  x = K.abs(y_true - y_pred)
  d_t = d*K.ones_like(x)
  quad = K.min(K.stack([x, d_t], axis = -1), axis = -1)
  return( 0.5*K.square(quad) + d*(x - quad) )

def gelu(x: tf.Tensor) -> tf.Tensor:
    return tf.nn.sigmoid(1.702 * x) * x

def add_conv1(model):
    model.add(ConvLSTM1D(filters=UNITS[0], kernel_size=3, activation='tanh',strides=1,padding='same', data_format='channels_last', recurrent_activation='hard_sigmoid', return_sequences=True, kernel_initializer=initializers.HeNormal(seed=1), bias_initializer=initializers.RandomUniform(seed=1), kernel_regularizer = l2(REG), bias_regularizer = l2(REG)))
    return(model)

def add_conv2(model):
    model.add(ConvLSTM1D(filters=UNITS[1], kernel_size=5, activation='tanh',strides=1,padding='same', data_format='channels_last', recurrent_activation='hard_sigmoid', return_sequences=True, kernel_initializer=initializers.HeNormal(seed=1), bias_initializer=initializers.RandomUniform(seed=1), kernel_regularizer = l2(REG), bias_regularizer = l2(REG)))
    return(model)

#print('Completed defining necessary functions.')

with open('Preprocesed_X_data.npy', 'rb') as f:
  X = np.load(f)

#print('Preprocesed X data loaded successfully!')

#print("--- %s seconds ---" % (time.time() - start_time))

with open('Y_data.npy', 'rb') as f:
  Y = np.load(f)

#Sample weighting
lst=[]

with open('train_sequences_exp.txt', 'rb') as f:
    for line in f.readlines():
        num = float(line.split()[0])
        lst.append(num)

arr= np.array(lst)
arr_m = np.mean(arr)
arr_std = np.std(arr)

lst_trans=[]
for i in range(len(lst)):
    val = abs(float(lst[i]-arr_m)/arr_std)
    lst_trans.append(val)
    #print(val)

arr_trans = np.array(lst_trans)

with open('sample_weights_f.npy', 'wb') as f:
    np.save(f, arr_trans)

with open('sample_weights_f.npy', 'rb') as f:
    sw = np.load(f)

#print('Y data loaded successfully!')

#print("--- %s seconds ---" % (time.time() - start_time))

from keras.layers import *
from keras.models import *
from keras import backend as K

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()
    
    def get_config(self):
        config = super().get_config()
        config.update({"return_sequences":self.return_sequences})
        return config

    def build(self, input_shape):
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),initializer="zeros")
        super(attention,self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if epoch:  # or save after some epoch, each k-th epoch etc.
            self.model.save("model_10_{}.h5".format(epoch))

# detect TPUs
tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu='local')
tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

with tpu_strategy.scope():
  # Model Building
  model = Sequential()
  input_shape = X[0].shape
  model.add(InputLayer(input_shape = input_shape))

  for i in range(2):
    model = add_conv1(model)

  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.15))
  model.add(MaxPooling2D(pool_size=(2,2)))

  for i in range(1):
    model = add_conv2(model)

  model.add(BatchNormalization())
  model.add(Activation('relu'))
  model.add(Dropout(0.15))
  model.add(MaxPooling2D(pool_size=(2,2)))

  model.add(TimeDistributed(Flatten()))

  model.add(Bidirectional(LSTM(128, return_sequences=True)))
  model.add(attention(return_sequences=True)) 
  model.add(Dropout(0.1))
  model.add(Bidirectional(LSTM(32, return_sequences=True)))
  model.add(attention(return_sequences=False)) 
  model.add(Dropout(0.1))

  model.add(Dense(units=128, kernel_initializer=initializers.HeNormal(seed=1), bias_initializer=initializers.RandomUniform(seed=1), kernel_regularizer = l2(REG), bias_regularizer = l2(REG)))
  model.add(BatchNormalization())
  model.add(Activation('gelu'))
  model.add(Dropout(0.1))

  model.add(Dense(units=16, kernel_initializer=initializers.HeNormal(seed=1), bias_initializer=initializers.RandomUniform(seed=1), kernel_regularizer = l2(REG), bias_regularizer = l2(REG)))
  model.add(BatchNormalization())
  model.add(Activation('gelu'))
  model.add(Dropout(0.1))

  model.add(Dense(units=num_outputs))

  # Model Compilation
  model.compile(optimizer = Adam(lr=3e-4), loss=correlationLoss, metrics = [correlationMetric, huber_loss, 'mae'])

  #earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
  saver = CustomSaver()
    
  print(model.summary())

  # Model Taining
  history = model.fit(X, Y, epochs=5, batch_size=256,sample_weight=sw, shuffle=True, callbacks=[saver])

#print('Hooray, Training Completed!')

model.save('TrainedModel.h5')

#print('Saved the model successfully.')

# STAGE-3: Predict & Prepare Submission File

# Read test data
with open('Preprocessed_X_test-data.npy', 'rb') as f:
  X = np.load(f)

test_y = model.predict(X, verbose=1, workers=8, use_multiprocessing=True)       # Numpy Array of Predictions

with open('Y_test_data.npy', 'wb') as f:
  np.save(f, test_y)

#print("Successfully saved model predictions.")


# Preparing submission file --> "pred.json"
import json
from collections import OrderedDict

with open('sample_submission.json', 'r') as f:
  ground = json.load(f)

indices = np.array([int(index) for index in list(ground.keys())])

PRED_DATA = OrderedDict()

for i in indices:
  PRED_DATA[str(i)] = float(test_y[i])

def dump_predictions(prediction_dict, prediction_file):
  with open(prediction_file, 'w') as f:
    json.dump(prediction_dict, f)

dump_predictions(PRED_DATA, 'pred.json')

#print("Successfully prepared submission file.")
