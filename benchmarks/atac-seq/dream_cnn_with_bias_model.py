import numpy as np
from tensorflow.keras.backend import int_shape
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Add, Concatenate, Lambda, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from chrombpnet.training.utils.losses import multinomial_nll
import tensorflow as tf
import random as rn
import os
from chrombpnet.training.models.dream_cnn import *

os.environ['PYTHONHASHSEED'] = '0'

def load_pretrained_bias(model_hdf5):
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import get_custom_objects
    custom_objects={"multinomial_nll":multinomial_nll, "tf":tf}
    get_custom_objects().update(custom_objects)
    pretrained_bias_model=load_model(model_hdf5)
    #freeze the model
    num_layers=len(pretrained_bias_model.layers)
    for i in range(num_layers):
        pretrained_bias_model.layers[i].trainable=False
    return pretrained_bias_model


def getModelGivenModelOptionsAndWeightInits(args, model_params):   
    
    assert("bias_model_path" in model_params.keys()) # bias model path not specfied for model
    filters=int(model_params['filters'])
    n_dil_layers=int(model_params['n_dil_layers'])
    counts_loss_weight=float(model_params['counts_loss_weight'])
    bias_model_path=model_params['bias_model_path']
    sequence_len=int(model_params['inputlen'])
    out_pred_len=int(model_params['outputlen'])


    bias_model = load_pretrained_bias(bias_model_path)

    #read in arguments
    seed=args.seed
    np.random.seed(seed)    
    tf.random.set_seed(seed)
    rn.seed(seed)

    class MyModel(tf.keras.Model):
        def __init__(self, sequence_len, out_pred_len):
            super(MyModel, self).__init__()

            self.bias_model = bias_model  # replace with actual model class and arguments
            self.main_model_wo_bias = SeqNN(seqsize=sequence_len, target_bins = out_pred_len)  # replace with actual model class and arguments
            self.sequence_len = sequence_len
            self.out_pred_len = out_pred_len

        def call(self, inputs):
            # Get bias output
            inputs = tf.cast(inputs, dtype=tf.float32)

            bias_output = self.bias_model(inputs)

            # Get output without bias
            output_wo_bias = self.main_model_wo_bias(inputs)
            wo_bias_profile_out = output_wo_bias['profile_out']
            wo_bias_count_out = output_wo_bias['count_out']

            profile_out = tf.add(wo_bias_profile_out, bias_output[0], name="logits_profile_predictions")
            concat_counts = tf.concat([wo_bias_count_out, bias_output[1]], axis=-1)
            count_out = tf.math.reduce_logsumexp(concat_counts, axis=-1, keepdims=True, name="logcount_predictions")

            return [profile_out, count_out]

    model = MyModel(sequence_len=sequence_len, out_pred_len=out_pred_len)

    model.compile(optimizer=Adam(learning_rate=args.learning_rate),
                    loss=[multinomial_nll,'mse'],
                    loss_weights=[1,counts_loss_weight])

    return model 


def save_model_without_bias(model, output_prefix):
    print('save model without bias')
    model.get_layer("seq_nn").save_weights(output_prefix+"_nobias.ckpt")