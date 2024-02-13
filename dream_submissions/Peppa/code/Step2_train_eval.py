import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
# for i in physical_devices:
	# tf.config.experimental.set_memory_growth(i, True)
# export CUDA_VISIBLE_DEVICES="0"
import pandas as pd
import numpy as np
import sonnet as snt
from tqdm import tqdm
import time
import os
import enformer5
import importlib
importlib.reload(enformer5)
import glob
import json
import functools
import tensorflow_probability as tfp
import h5py
import sys

try:
	opt = sys.argv[1]
	batch_size = int(sys.argv[2])
	learning_rate = float(sys.argv[3])
except:
	opt = "Momentum"
	batch_size = 10000
	learning_rate = 0.01

# Input
seq_length=85
label=f"main_v5-{opt}-{batch_size}-{learning_rate}"
test_sample = 10000

# read data
f2 = h5py.File('all_train_data.h5', 'r')
train_X = f2['train_X'][:]
train_Y = f2['train_Y'][:]
f2.close()
# train eval
random_index = np.random.permutation(train_X.shape[0])
eval_index = random_index[:test_sample]
train_index = random_index[test_sample:]
eval_X = train_X[eval_index,15:100,:]
eval_Y = train_Y[eval_index,:]
# trucation # data augmentation
print (train_X.shape)
print ("creating augmented data")
train_X_n1 = train_X[train_index,14:99,:]
print ("shift -1 created")
# train_X_n2 = train_X[train_index,13:98,:]
train_X_p1 = train_X[train_index,16:101,:]
print ("shift +1 created")
# train_X_p2 = train_X[train_index,17:102,:]
train_X = train_X[train_index,15:100,:]
train_Y = train_Y[train_index,:]
print ("concat new train data")
# train_X = np.concatenate((train_X_n1,train_X_n2,train_X,train_X_p1,train_X_p2),axis=0)
# train_Y = np.concatenate((train_Y,train_Y,train_Y,train_Y,train_Y),axis=0)
train_X = np.concatenate((train_X_n1,train_X,train_X_p1),axis=0)
train_Y = np.concatenate((train_Y,train_Y,train_Y),axis=0)
print ("finish concat")
# convert to tensor, consume too much memory, not using it
# train_data = {'sequence':tf.convert_to_tensor(train_X,dtype=tf.float32),'target':tf.convert_to_tensor(train_Y,dtype=tf.float32)}
eval_data = {'sequence':tf.convert_to_tensor(eval_X,dtype=tf.float32),'target':tf.convert_to_tensor(eval_Y,dtype=tf.float32)}

# define model

def create_step_function(model, optimizer):
	@tf.function
	def train_step(seq,target, head='yeast', optimizer_clip_norm_global=0.2):
		with tf.GradientTape() as tape:
			outputs = model(seq, is_training=True)[head]
			outputs = tf.reshape(outputs,target.shape)
			MSE,MAE,pearsonR,poisson_loss = evalate_prediction(target, outputs)
			gradients = tape.gradient(poisson_loss, model.trainable_variables)
			# gradients = tape.gradient(MSE, model.trainable_variables)
			optimizer.apply(gradients, model.trainable_variables)
		return MSE,MAE,pearsonR,poisson_loss
	return train_step
def evalate_prediction(y_true, y_pred):
	# MSE
	MSE = tf.reduce_mean(tf.keras.losses.mean_squared_error(y_true, y_pred))
	# MAE
	MAE = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y_true, y_pred))
	# pearsonR
	pearsonR = tf.reduce_mean(tfp.stats.correlation(y_true, y_pred))
	# poisson loss
	poisson_loss = tf.reduce_mean(tf.keras.losses.poisson(y_true, y_pred))
	return MSE,MAE,pearsonR,poisson_loss
	

if opt == "Adam":
	optimizer = snt.optimizers.Adam(learning_rate=learning_rate)
if opt == "RMSProp":
	optimizer = snt.optimizers.RMSProp(learning_rate=learning_rate)
if opt == "Momentum":
	optimizer = snt.optimizers.Momentum(learning_rate=learning_rate,momentum=0.9)
else:
	optimizer = snt.optimizers.SGD(learning_rate=learning_rate)


model = enformer5.Enformer(channels=1536//8,
						  num_heads=8,
						  num_transformer_layers=4,
						  pooling_type='max')

train_step = create_step_function(model, optimizer)
eval_pred = model(eval_data['sequence'], is_training=False)['yeast']
print (eval_pred[:3])

num_epochs=2000
# train model
steps_per_epoch = train_X.shape[0]//batch_size
# steps_per_epoch = 30 # for debug purpose
outdir = f"{label}_models"
os.system("mkdir -p %s"%(outdir))
global_step = 0
out_metric = []
batch_train_index = list(range(train_X.shape[0]))
import random
random.shuffle(batch_train_index)


import json
from collections import OrderedDict
with open('sample_submission.json', 'r') as f:
    ground = json.load(f)
indices = np.array([int(indice) for indice in list(ground.keys())])

def eval_submission(indices,out_list,label):
	PRED_DATA = OrderedDict()
	for i in indices:
		PRED_DATA[str(i)] = float(out_list[i])
	def dump_predictions(prediction_dict, prediction_file):
		with open(prediction_file, 'w') as f:
			json.dump(prediction_dict, f)
	dump_predictions(PRED_DATA, f'{label}_enformer.pred.json')
f2 = h5py.File('test_sequences.h5', 'r')
test_X = f2['test_X'][:]
test_X = test_X[:,15:100,:]
test_X = tf.convert_to_tensor(test_X,dtype=tf.float32)
f2.close()

p=0
for i in model.trainable_variables:
    p = p+tf.reduce_prod(i.shape).numpy()
print (p)
for epoch_i in range(num_epochs):
	for i in tqdm(range(steps_per_epoch)):
		batch_indices = batch_train_index[batch_size * i: batch_size * (i + 1)]
		seq=tf.convert_to_tensor(train_X[batch_indices,:,:],dtype=tf.float32)
		target=tf.convert_to_tensor(train_Y[batch_indices,:],dtype=tf.float32)
		MSE_train,MAE_train,pearsonR_train,poisson_loss_train = train_step(seq,target)
	# eval 
	eval_pred = model(eval_data['sequence'], is_training=False)['yeast']
	eval_pred = tf.reshape(eval_pred,eval_data['target'].shape)
	MSE_eval,MAE_eval,pearsonR_eval,poisson_loss_eval = evalate_prediction(eval_data['target'], eval_pred)
	print (label,"Epoch",epoch_i)
	train_metrics = [MSE_train.numpy(),MAE_train.numpy(),pearsonR_train.numpy(),poisson_loss_train.numpy()]
	eval_metrics = [MSE_eval.numpy(),MAE_eval.numpy(),pearsonR_eval.numpy(),poisson_loss_eval.numpy()]
	print (train_metrics)
	print (eval_metrics)
	out_metric.append([epoch_i]+train_metrics+eval_metrics)
	tmp = pd.DataFrame(out_metric)
	tmp.to_csv(f"{outdir}/{label}.tmp.out_metric.csv",index=False)
	if epoch_i>3 and pearsonR_eval > 0.73:
		tf.saved_model.save(model,"%s/epoch_%s_model"%(outdir,epoch_i))
		out = model(test_X,is_training=False)['yeast']
		out_list = np.reshape(out.numpy(),(71103,))
		eval_submission(indices,out_list,f"{outdir}/{label}_epoch{epoch_i}")
out_metric = pd.DataFrame(out_metric)
out_metric.to_csv(f"{label}.out_metric.csv",index=False)


