from __future__ import print_function
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
config.gpu_options.per_process_gpu_memory_fraction = 0.95
set_session(tf.Session(config=config))

K.set_image_data_format('channels_last')  # TF dimension ordering in this code
size= 128

def split(word):
    return [char for char in word]

channel=4
batch_size=64

model = cnn.cnn1d(size,channel)

### load data
map_ACTG={}
map_ACTG['A']=[1,0,0,0]
map_ACTG['C']=[0,1,0,0]
map_ACTG['T']=[0,0,1,0]
map_ACTG['G']=[0,0,0,1]

TRAIN=open('train.txt','r')
train_map={}
train_onehot={}
the_max=-10000
the_min=10000
for line in TRAIN:
    line=line.strip()
    table=line.split('\t')
    table[0]=table[0].upper()
    if (len(table[0])==110):
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
        table[1]=float(table[1])
        if (float(table[1])==0):
            pass
        elif (float(table[1])==17):
            pass
        else:
            train_map[table[0]]=float(table[1])
            train_onehot[table[0]]=onehot
        if (table[1]>the_max):
            the_max=table[1]
        if (table[1]<the_min):
            the_min=table[1]

TRAIN.close()


all_samples=train_map.keys()
train_set_0=[]
train_set_1=[]
train_set_2=[]
train_set_3=[]
train_set_4=[]
test_set_0=[]
test_set_1=[]
test_set_2=[]
test_set_3=[]
test_set_4=[]
for the_sample in all_samples:
    rrr=random.random()
    if (rrr<0.2):
        train_set_1.append(the_sample)
        train_set_2.append(the_sample)
        train_set_3.append(the_sample)
        train_set_4.append(the_sample)
        test_set_0.append(the_sample)
    elif (rrr<0.4):
        train_set_1.append(the_sample)
        train_set_2.append(the_sample)
        train_set_3.append(the_sample)
        test_set_4.append(the_sample)
        train_set_0.append(the_sample)
    elif (rrr<0.6):
        train_set_1.append(the_sample)
        train_set_2.append(the_sample)
        test_set_3.append(the_sample)
        train_set_4.append(the_sample)
        train_set_0.append(the_sample)
    elif (rrr<0.8):
        train_set_1.append(the_sample)
        test_set_2.append(the_sample)
        train_set_3.append(the_sample)
        train_set_4.append(the_sample)
        train_set_0.append(the_sample)
    else:
        test_set_1.append(the_sample)
        train_set_2.append(the_sample)
        train_set_3.append(the_sample)
        train_set_4.append(the_sample)
        train_set_0.append(the_sample)



def generate_data(train_set, batch_size, if_train):
    i = 0
    while True:
        image_batch = []
        label_batch = []
        for b in range(batch_size):
            if i == len(train_set):
                i = 0
                random.shuffle(train_set)
            #print(train_set[i])
            image= train_onehot[train_set[i]]
            label=(train_map[train_set[i]]-the_min)/(the_max-the_min)
            image_batch.append(image)
            label_batch.append(label)
            i += 1
        image_batch=np.array(image_batch)
        label_batch=np.array(label_batch)
        yield image_batch, [label_batch,label_batch,label_batch,label_batch,label_batch]


#model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=False)
name_model='weights.h5'
callbacks = [
#    keras.callbacks.TensorBoard(log_dir='./',
#    histogram_freq=0, write_graph=True, write_images=False),
    keras.callbacks.ModelCheckpoint(os.path.join('./', name_model),
    verbose=0, save_weights_only=True,save_best_only=True,monitor='val_loss')
    ]

model.fit_generator(
    generate_data(train_set_1, batch_size,True),
    steps_per_epoch=int(len(train_set_1) // batch_size), epochs=5,
    validation_data=generate_data(test_set_1,batch_size,False),
    validation_steps=int(len(test_set_1) // batch_size),callbacks=callbacks)

del model
model = cnn.cnn1d(size,channel)
model.load_weights('weights.h5')
model.fit_generator(
    generate_data(train_set_2, batch_size,True),
    steps_per_epoch=int(len(train_set_2) // batch_size), epochs=5,
    validation_data=generate_data(test_set_2,batch_size,False),
    validation_steps=int(len(test_set_2) // batch_size),callbacks=callbacks)

del model
model = cnn.cnn1d(size,channel)
model.load_weights('weights.h5')
model.fit_generator(
    generate_data(train_set_3, batch_size,True),
    steps_per_epoch=int(len(train_set_3) // batch_size), epochs=5,
    validation_data=generate_data(test_set_3,batch_size,False),
    validation_steps=int(len(test_set_3) // batch_size),callbacks=callbacks)

del model
model = cnn.cnn1d(size,channel)
model.load_weights('weights.h5')
model.fit_generator(
    generate_data(train_set_4, batch_size,True),
    steps_per_epoch=int(len(train_set_4) // batch_size), epochs=5,
    validation_data=generate_data(test_set_4,batch_size,False),
    validation_steps=int(len(test_set_4) // batch_size),callbacks=callbacks)

del model
model = cnn.cnn1d(size,channel)
model.load_weights('weights.h5')
model.fit_generator(
    generate_data(train_set_0, batch_size,True),
    steps_per_epoch=int(len(train_set_0) // batch_size), epochs=5,
    validation_data=generate_data(test_set_0,batch_size,False),
    validation_steps=int(len(test_set_0) // batch_size),callbacks=callbacks)
