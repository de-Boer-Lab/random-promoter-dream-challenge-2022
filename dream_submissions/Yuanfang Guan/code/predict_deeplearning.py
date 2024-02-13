from __future__ import print_function
import glob
import os
import sys
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv1D, MaxPooling1D, Conv2DTranspose,Lambda
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import tensorflow as tf
import keras
#import cv2
import scipy.io
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import re
import random

import cnn_sigmoid_1024 as cnn

from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
size= 128

def split(word):
    return [char for char in word]

channel=4
batch_size=128

model = cnn.cnn1d(size,channel)

### load data
map_ACTG={}
map_ACTG['A']=[1,0,0,0]
map_ACTG['C']=[0,1,0,0]
map_ACTG['T']=[0,0,1,0]
map_ACTG['G']=[0,0,0,1]

TRAIN=open('../../data/test_sequences.txt','r')
train_map={}
train_onehot={}
for line in TRAIN:
    line=line.strip()
    table=line.split('\t')
    table[0]=table[0].upper()
    onehot=np.zeros((128,4))
    #print(len(table[0]))
    ttt=split(table[0])#.split()
    i=0
    while(i<10):
        ttt.pop()
        i=i+1
    i=0
    while(i<10):
        ttt.pop(0)
        i=i+1
    
    i=0
    for actg in ttt:
        try:
            onehot[i,:]=map_ACTG[actg]
        except:
            onehot[i,:]=[0,0,0,0]
        i=i+1
    train_map[table[0]]=float(table[1])
    train_onehot[table[0]]=onehot
TRAIN.close()

all_models=glob.glob('weights*.h5')
for the_model in all_models:
    model = cnn.cnn1d(size,4)
    model.load_weights(the_model)
    all_test_files=open('../../data/test_sequences.txt','r')
    PRED=open(('prediction.dat.final'+the_model),'w')
    iii=1
    image_batch=[]
    for test_line in all_test_files:
        sample = test_line
        table=sample.split('\t')
        table[0]=table[0].upper()
        onehot=np.zeros((128,4))
        ttt=split(table[0])#.split()
        i=0
        while(i<10):
            ttt.pop()
            i=i+1
        i=0
        while(i<10):
            ttt.pop(0)
            i=i+1
    
        i=0
        for actg in ttt:
            try:
                onehot[i,:]=map_ACTG[actg]
            except:
                onehot[i,:]=[0,0,0,0]
            i=i+1
        image_batch.append(onehot)
        if (iii%100000==0):
            image_batch=np.asarray(image_batch)
            output = model.predict(image_batch)
            output=np.asarray(output)
            output=output[:,:,0]
            output=np.mean(output,axis=0)
            for aaa in output:
                PRED.write('%.6f\n' % aaa)
            image_batch=[]
        iii=iii+1
    image_batch=np.asarray(image_batch)
    print(image_batch.shape)
    output = model.predict(image_batch)
    output=np.asarray(output)
    output=output[:,:,0]
    output=np.mean(output,axis=0)
    for aaa in output:
        PRED.write('%.6f\n' % aaa)
    image_batch=[]
    PRED.close()

