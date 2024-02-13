# In[ ]:


# pip install tensorflow-addons


# In[ ]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import h5py
from scipy import stats
import matplotlib.pyplot as plt
import os


# In[ ]:


with h5py.File('compressed_dataset_111.h5', 'r') as hf:
    x_train = np.array(hf['x_train'][:,:,:4]).astype(np.int8)
    y_train = np.array(hf['y_train']).astype(np.float32)
    
x_valid = x_train[-104448:]
x_train = x_train[:-104489]
y_valid = y_train[-104448:]
y_train = y_train[:-104489]


# In[ ]:


class MultiHeadAttention2(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, *args, embedding_size=None, **kwargs):
        super(MultiHeadAttention2, self).__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_size = d_model if embedding_size == None else embedding_size

        assert d_model % self.num_heads == 0 and d_model % 6 == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wk = tf.keras.layers.Dense(d_model, use_bias=False)
        self.wv = tf.keras.layers.Dense(d_model, use_bias=False)
        
        self.r_k_layer = tf.keras.layers.Dense(d_model, use_bias=False)
        self.r_w = tf.Variable(tf.random_normal_initializer(0, 0.5)(shape=[1, self.num_heads, 1, self.depth]), trainable=True, name=f'{self.name}-r_w')
        self.r_r = tf.Variable(tf.random_normal_initializer(0, 0.5)(shape=[1, self.num_heads, 1, self.depth]), trainable=True, name=f'{self.name}-r_r')

        self.dense = tf.keras.layers.Dense(d_model)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            'embedding_size': self.embedding_size
        })
        return config
        
    def split_heads(self, x, batch_size, seq_len):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        seq_len = tf.constant(q.shape[1])

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, seq_len)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, seq_len)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, seq_len)  # (batch_size, num_heads, seq_len_v, depth)
        q = q / tf.math.sqrt(tf.cast(self.depth, dtype=tf.float32))
        
        pos = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
        feature_size=self.embedding_size//6

        seq_length = tf.cast(seq_len, dtype=tf.float32)
        exp1 = f_exponential(tf.abs(pos), feature_size, seq_length=seq_length)
        exp2 = tf.multiply(exp1, tf.sign(pos)[..., tf.newaxis])
        cm1 = f_central_mask(tf.abs(pos), feature_size, seq_length=seq_length)
        cm2 = tf.multiply(cm1, tf.sign(pos)[..., tf.newaxis])
        gam1 = f_gamma(tf.abs(pos), feature_size, seq_length=seq_length)
        gam2 = tf.multiply(gam1, tf.sign(pos)[..., tf.newaxis])

        # [1, 2seq_len - 1, embedding_size]
        positional_encodings = tf.concat([exp1, exp2, cm1, cm2, gam1, gam2], axis=-1)
        positional_encodings = tf.keras.layers.Dropout(0.1)(positional_encodings)
        
        # [1, 2seq_len - 1, d_model]
        r_k = self.r_k_layer(positional_encodings)
        
        # [1, 2seq_len - 1, num_heads, depth]
        r_k = tf.reshape(r_k, [r_k.shape[0], r_k.shape[1], self.num_heads, self.depth])
        r_k = tf.transpose(r_k, perm=[0, 2, 1, 3])
        # [1, num_heads, 2seq_len - 1, depth]
        
        # [batch_size, num_heads, seq_len, seq_len]
        content_logits = tf.matmul(q + self.r_w, k, transpose_b=True)
        
        # [batch_size, num_heads, seq_len, 2seq_len - 1]
        relative_logits = tf.matmul(q + self.r_r, r_k, transpose_b=True)
        # [batch_size, num_heads, seq_len, seq_len]
        relative_logits = relative_shift(relative_logits)
        
        # [batch_size, num_heads, seq_len, seq_len]
        logits = content_logits + relative_logits
        attention_map = tf.nn.softmax(logits)
        
        # [batch_size, num_heads, seq_len, depth]
        attended_values = tf.matmul(attention_map, v)
        # [batch_size, seq_len, num_heads, depth]
        attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])
        
        concat_attention = tf.reshape(attended_values, [batch_size, seq_len, self.d_model])
        
        output = self.dense(concat_attention)
        
        return output, attention_map


def f_exponential(positions, feature_size, seq_length=None, min_half_life=3.0):
    if seq_length is None:
        seq_length = tf.cast(tf.reduce_max(tf.abs(positions)) + 1, dtype=tf.float32)
    max_range = tf.math.log(seq_length) / tf.math.log(2.0)
    half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
    half_life = tf.reshape(half_life, shape=[1]*positions.shape.rank + half_life.shape)
    positions = tf.abs(positions)
    outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
    return outputs

def f_central_mask(positions, feature_size, seq_length=None):
    center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32)) - 1
    center_widths = tf.reshape(center_widths, shape=[1]*positions.shape.rank + center_widths.shape)
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis], tf.float32)
    return outputs

def f_gamma(positions, feature_size, seq_length=None):
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    stdv = seq_length / (2*feature_size)
    start_mean = seq_length / feature_size
    mean = tf.linspace(start_mean, seq_length, num=feature_size)
    mean = tf.reshape(mean, shape=[1]*positions.shape.rank + mean.shape)
    concentration = (mean / stdv) ** 2
    rate = mean / stdv**2
    def gamma_pdf(x, conc, rt):
        log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
        log_normalization = (tf.math.lgamma(concentration) - concentration * tf.math.log(rate))
        return tf.exp(log_unnormalized_prob - log_normalization)
    probabilities = gamma_pdf(tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis], concentration, rate)
    outputs = probabilities / tf.reduce_max(probabilities)
    return outputs
    
def relative_shift(x):
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
    return x


# In[ ]:


class RevCompConv1D(tf.keras.layers.Conv1D):
    """
    Implement forward and reverse-complement filter convolutions
    for 1D signals. It takes as input either a single input or two inputs 
    (where the second input is the reverse complement scan). If a single input, 
    this performs both forward and reverse complement scans and either merges it 
    (if concat=True) or returns a separate scan for forward and reverse comp. 
    """
    def __init__(self, *args, concat=True, **kwargs):
        super(RevCompConv1D, self).__init__(*args, **kwargs)
        self.concat = concat


    def call(self, inputs, inputs2=None):

        if inputs2 is not None:
            # create rc_kernels
            rc_kernel = self.kernel[::-1,::-1,:]

            # convolution 1D
            outputs = self.convolution_op(inputs, self.kernel)
            rc_outputs = self.convolution_op(inputs2, rc_kernel)

        else:
            # create rc_kernels
            rc_kernel = tf.concat([self.kernel, self.kernel[::-1,:,:][:,::-1,:]], axis=-1)

            # convolution 1D
            outputs = self.convolution_op(inputs, rc_kernel)

            # unstack to forward and reverse strands
            outputs = tf.unstack(outputs, axis=2)
            rc_outputs = tf.stack(outputs[self.filters:], axis=2)
            outputs = tf.stack(outputs[:self.filters], axis=2)

        # add bias
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)
            rc_outputs = tf.nn.bias_add(rc_outputs, self.bias)

        # add activations
        if self.activation is not None:
            outputs = self.activation(outputs)
            rc_outputs = self.activation(rc_outputs)

        if self.concat:
            return tf.concat([outputs, rc_outputs], axis=-1)
        else:
            return outputs, rc_outputs


# In[ ]:


def residual_block(input_layer, kernel_size=3, activation='relu', num_layers=5, dropout=0.1):

    filters = input_layer.shape.as_list()[-1]  

    nn = tf.keras.layers.Conv1D(filters=filters,
                           kernel_size=kernel_size,
                           activation=None,
                           use_bias=False,
                           padding='same',
                           dilation_rate=1)(input_layer) 
    nn = tf.keras.layers.BatchNormalization()(nn)

    base_rate = 2
    for i in range(1,num_layers):
        nn = tf.keras.layers.Activation('relu')(nn)
        nn = tf.keras.layers.Dropout(dropout)(nn)
        nn = tf.keras.layers.Conv1D(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 dilation_rate=base_rate**i)(nn) 
        nn = tf.keras.layers.BatchNormalization()(nn)
    nn = tf.keras.layers.Add()([input_layer, nn])
    return tf.keras.layers.Activation(activation)(nn)


# In[ ]:


class ProfileModel(tf.keras.Model):
    
    def __init__(self, bin_size=1, rc_prob=0.5, *args, **kwargs):
        super(ProfileModel, self).__init__(*args, **kwargs)
        self.bin_size = bin_size
        self.rc_prob = rc_prob

    def get_config(self):
        config = super(ProfileModel, self).get_config()
        config.update({"bin_size": self.bin_size})
        config.update({'rc_prob': self.rc_prob})
        return config

    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        # online target resolution calculation
        y = bin_resolution(y, self.bin_size)

        # stochastic reverse complement
        x, y = reverse_complement(x, y, p=self.rc_prob)

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}


    def test_step(self, data):
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data


        # online target resolution calculation
        y = bin_resolution(y, self.bin_size)

        y_pred = self(x, training=False)
        x_RC, y_RC = reverse_complement(x, y, p=1.0)
        y_pred_RC = self(x_RC, training=False)
        y_pred = tf.math.reduce_mean([y_pred, y_pred_RC], axis=0)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        return {m.name: m.result() for m in self.metrics}
    
"""    def compute_loss(self, x, y, y_pred, sample_weight):
        del x  # The default implementation does not use `x`.
        return self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)"""
    

def reverse_complement(x, y, p=0.5):
    # x_rc = tf.gather(x, [3, 2, 1, 0], axis=-1)
    x_rc = tf.reverse(x, axis=[2])
    x_rc = tf.reverse(x_rc, axis=[1])
    # y_rc = tf.reverse(y, axis=[1])
    switch = tf.random.uniform(shape=[]) > (1 - p)
    x_new = tf.cond(switch, lambda: x_rc, lambda: x)
    # y_new = tf.cond(switch, lambda: y_rc, lambda: y)
    return x_new, y

def bin_resolution(y, bin_size):
    if bin_size > 1:
        y_dim = tf.shape(y)
        num_bins = tf.cast(y_dim[1] / bin_size, 'int32')
        y_reshape = tf.reshape(y, (y_dim[0], num_bins, bin_size, y_dim[2]))
        y_bin = tf.math.reduce_mean(y_reshape, axis=2)
        return y_bin
    else:
        return y


# In[ ]:


class AttentionPooling(keras.layers.Layer):

    def __init__(self, pool_size, *args, **kwargs):
        super(AttentionPooling, self).__init__(*args, **kwargs)
        self.pool_size = pool_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size,
        })
        return config

    def build(self, input_shape):
        self.dense = keras.layers.Conv1D(filters=input_shape[-1], kernel_size=1, activation=None, use_bias=False)

    def call(self, inputs):
        N, L, F = inputs.shape
        inputs = tf.keras.layers.Cropping1D((0, L % self.pool_size))(inputs)
        inputs = tf.reshape(inputs, (-1, L//self.pool_size, self.pool_size, F))

        raw_weights = self.dense(inputs)
        att_weights = tf.nn.softmax(raw_weights, axis=-2)
        
        return tf.math.reduce_sum(inputs * att_weights, axis=-2)


# In[ ]:


inputs = keras.layers.Input(shape=(111,4))

nn = RevCompConv1D(filters=192, kernel_size=19, padding='same', concat=True)(inputs)
nn = keras.layers.BatchNormalization()(nn)
nn = keras.layers.Activation('exponential')(nn)
nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=4, dropout=0.1)
nn = keras.layers.Dropout(0.2)(nn)
nn = AttentionPooling(4)(nn)

mod = ProfileModel(inputs=inputs, outputs=nn)
nn = mod(inputs)
hold = nn

nn = keras.layers.Conv1D(filters=256, kernel_size=9, use_bias=True, padding='same', activation='relu')(nn)
nn = keras.layers.BatchNormalization()(nn)
nn = keras.layers.Activation('relu')(nn)
nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=2, dropout=0.1)
nn = AttentionPooling(4)(nn)
nn = keras.layers.Dropout(0.3)(nn)

mod = ProfileModel(inputs=hold, outputs=nn)
nn = mod(hold)
hold = nn

nn = keras.layers.Conv1D(filters=192, kernel_size=7, padding='same')(nn)
nn = keras.layers.BatchNormalization()(nn)
nn = keras.layers.Activation('relu')(nn)
nn = residual_block(nn, kernel_size=3, activation='relu', num_layers=2, dropout=0.1)
nn = AttentionPooling(2)(nn)
nn = keras.layers.Dropout(0.1)(nn)

mod = ProfileModel(inputs=hold, outputs=nn)
nn = mod(hold)
hold = nn

nn1, att = MultiHeadAttention2(num_heads=8, d_model=192)(nn, nn, nn)
nn1 = keras.layers.Dropout(0.1)(nn1)
nn = tf.add(nn, nn1)
nn = keras.layers.LayerNormalization()(nn)
nn1 = keras.layers.Conv1D(filters=1024, kernel_size=1)(nn)
nn1 = keras.layers.Dropout(0.1)(nn1)
nn1 = keras.layers.Activation('relu')(nn1)
nn1 = keras.layers.Conv1D(filters=192, kernel_size=1)(nn1)
nn1 = keras.layers.Dropout(0.1)(nn1)
nn = tf.add(nn, nn1)
nn = keras.layers.LayerNormalization()(nn)

nn = keras.layers.Flatten()(nn)

nn = keras.layers.Dense(256)(nn)
nn = keras.layers.BatchNormalization()(nn)
nn = keras.layers.Activation('relu')(nn)
nn = keras.layers.Dropout(0.4)(nn)

nn = keras.layers.Dense(256)(nn)
nn = keras.layers.BatchNormalization()(nn)
nn = keras.layers.Activation('relu')(nn)
nn = keras.layers.Dropout(0.4)(nn)

outputs = keras.layers.Dense(1, activation='linear')(nn)

model = keras.Model(inputs=inputs, outputs=outputs)

opt = tfa.optimizers.SWA(tf.keras.optimizers.Adam(1e-4), start_averaging=150, average_period=30)
model.compile(optimizer=opt, loss='MSE', metrics=[])

prs = [0]


# In[ ]:


probabilities, bins, _ = plt.hist(y_train, bins=10)
plt.close()
bins = bins[:-1]
probabilities = probabilities/np.sum(probabilities)
p_dist = np.array([bins, probabilities])

loss_func = 'MSE'
lr = 0.0001
opt = tfa.optimizers.SWA(tf.keras.optimizers.Adam(lr), start_averaging=150, average_period=30)
model.compile(optimizer=opt, loss=loss_func, metrics=[])


# In[ ]:


# If training in more than one run -----------------------------------------------------------
model = tf.keras.models.load_model('../model.h5', custom_objects={
    'ProfileModel' : ProfileModel,
    'MultiHeadAttention2' : MultiHeadAttention2,
    'RevCompConv1D' : RevCompConv1D,
    'AttentionPooling' : AttentionPooling
    })
# --------------------------------------------------------------------------------------------


# In[ ]:


batch_size = 1024
dataset_size = x_train.shape[0]
slice_size = 307200

for i in range(750):
    print('Epoch %d' % (i+1))
    
    perm = np.random.choice(dataset_size, size=slice_size, replace=False)
    model.fit(x_train[perm], y_train[perm], epochs=1, verbose=1, batch_size=batch_size)

    if i % 3 == 0:
        y_pred = np.squeeze(model.predict(x_valid, batch_size=1024))
        pr = stats.pearsonr(y_pred, y_valid)[0]
        sr = stats.spearmanr(y_pred, y_valid)[0]
        ls = tf.keras.losses.MSE(y_pred, y_valid).numpy()
        
        if pr >= np.max(prs) - 1e-7:
            model.save('../model.h5')
            
        prs.append(pr)
        
        print('pearson r:', pr)
        print('spearman r:', sr)
        print('valid MSE:', ls)


# In[ ]:


from keras import backend as K
K.set_value(model.optimizer.learning_rate, 3e-5)


# In[ ]:


batch_size = 1024
dataset_size = x_train.shape[0]
slice_size = 307200

for i in range(750, 1250):
    print('Epoch %d' % (i+1))
    
    perm = np.random.choice(dataset_size, size=slice_size, replace=False)
    model.fit(x_train[perm], y_train[perm], epochs=1, verbose=1, batch_size=batch_size)

    if i % 3 == 0:
        y_pred = np.squeeze(model.predict(x_valid, batch_size=1024))
        pr = stats.pearsonr(y_pred, y_valid)[0]
        sr = stats.spearmanr(y_pred, y_valid)[0]
        ls = tf.keras.losses.MSE(y_pred, y_valid).numpy()
        
        if pr >= np.max(prs) - 1e-7:
            model.save('../model.h5')
            
        prs.append(pr)
        
        print('pearson r:', pr)
        print('spearman r:', sr)
        print('valid MSE:', ls)

