# -*- coding: utf-8 -*-

## Code for DREAM COMPETITION

import pandas as pd
import tensorflow as tf
import numpy as np
import pickle
import datetime
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm 
from  model import *

# setup TPU
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='local')
tf.config.experimental_connect_to_cluster(resolver)
# this is the TPU initialization code that has to be at the beginning.
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

strategy = tf.distribute.TPUStrategy(resolver)

# tensorboard
tf.profiler.experimental.server.start(6000)

# read data
data_path="./data/train_sequences.txt"
df = pd.read_csv(data_path, sep="\t", header=None, names=["Seq", "Expression"])

dataset = tf.data.TextLineDataset(data_path)

seq_list=[]
exp_list=[]
pbar = tqdm(total=len(df))
   
for i in dataset.as_numpy_iterator():
 seq, exp = i.decode("utf-8").split("\t")
 seq_list.append(list(seq))
 exp_list.append(float(exp))
 pbar.update(1)

pbar.close()

# pad sequence to 112
pad_seq_list = tf.keras.preprocessing.sequence.pad_sequences(seq_list, maxlen=112, padding="post", truncating='post', dtype="str", value="N")

# save the data
pickle.dump(pad_seq_list, open("./pad_seq_112_list", "wb"))
pickle.dump(exp_list, open("./exp_seq_112_list", "wb"))

# load the data
pad_seq_list = pickle.load(open("./pad_seq_112_list", "rb"))
exp_list = pickle.load(open("./exp_seq_112_list", "rb"))

new_dataset = tf.data.Dataset.from_tensor_slices((pad_seq_list,exp_list))

# clear memory
del pad_seq_list
del exp_list

# one hot encoding
vocab = ['A','C','G','T']
lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
ps_dataset = new_dataset.map(lambda x, y: (lookup(x), y))
ps_dataset = ps_dataset.map(lambda x, y: (tf.cast(x, dtype=tf.float32), y))

## agumentation: reverse complement (not used in the final model)              
#rc_vocab = ['T','G','C','A']
#rc_lookup = tf.keras.layers.StringLookup(vocabulary=rc_vocab, output_mode='one_hot')
#rc_dataset = new_dataset.map(lambda x, y: (rc_lookup(x), y))
#rc_dataset = rc_dataset.map(lambda x, y: (tf.cast(x, dtype=tf.float32), y))                                    
#new_dataset = ps_dataset.concatenate(rc_dataset)

new_dataset = ps_dataset
data_size=len(new_dataset)
train_size = int(data_size*0.95) # 95% for training
val_size = int(data_size*0.05) # 5% for validating
new_dataset = new_dataset.shuffle(50000, reshuffle_each_iteration=False)
train_dataset = new_dataset.take(train_size)
val_dataset = new_dataset.skip(train_size)
val_dataset = val_dataset.take(val_size)

batch_size=128*8 # global batch size = local batch size * number of TPU cores
train_dataset = train_dataset.shuffle(1000).batch(batch_size, drop_remainder=True)
val_dataset = val_dataset.shuffle(1000).batch(batch_size, drop_remainder=True)

train_dataset.prefetch(tf.data.AUTOTUNE)
val_dataset.prefetch(tf.data.AUTOTUNE)

with strategy.scope():
    r_square = tfa.metrics.r_square.RSquare(dtype=tf.float32, y_shape=(1,))
    rmse = tf.keras.metrics.RootMeanSquaredError()
    model = return_model("trans_unet")
    model.compile(optimizer=Adam(), steps_per_execution = 50, loss = tf.keras.losses.Huber(), metrics=[r_square,rmse])

model.summary()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=20, restore_best_weights=True)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(x=train_dataset, epochs=100, batch_size=batch_size,
                  validation_data=val_dataset, callbacks=[tensorboard_callback,early_stop])

result_dic = model.evaluate(val_dataset, return_dict=True)
print(result_dic)

model.save("trans_unet")
print("model saved")
