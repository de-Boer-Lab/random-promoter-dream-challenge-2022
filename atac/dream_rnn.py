import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Concatenate, AveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid, linear
from tensorflow import transpose, matmul, expand_dims
from tensorflow import keras
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Flatten

import tensorflow as tf
from tensorflow.keras import layers as layers
from tensorflow.keras import Model
from tensorflow.keras.activations import swish
    
class TargetLengthCrop(Model):
    def __init__(self, target_length):
        super(TargetLengthCrop, self).__init__()
        self.target_length = target_length

    def call(self, inputs):
        # print(inputs.shape)
        seq_len, target_len = inputs.shape[-2], self.target_length

        if target_len == -1:
            return inputs

        if seq_len < target_len:
            raise ValueError(f'sequence length {seq_len} is less than target length {target_len}')

        trim = (target_len - seq_len) // 2

        if trim == 0:
            return inputs
        # print(inputs[:, -trim:trim, :].shape)
        return inputs[:, -trim:trim, :]

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, out_channels, kernel_size, pool_size, dropout_rate):
        super(ConvBlock, self).__init__()
        self.conv = tf.keras.layers.Conv1D(filters=out_channels, kernel_size=kernel_size,
                                           padding='same')
        self.pool = tf.keras.layers.MaxPooling1D(pool_size=pool_size, strides = pool_size)
        self.act = tf.keras.activations.relu
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.act(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


class BHIFirstLayersBlock(tf.keras.Model):
    def __init__(self, out_channels=320, seqsize=110, kernel_sizes=[9, 15], 
                 pool_size=1, dropout=0.2):
        super(BHIFirstLayersBlock, self).__init__()
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each_out_channels = out_channels // len(kernel_sizes)
        self.conv_blocks = [ConvBlock(each_out_channels, kernel_size, pool_size, dropout) 
                            for kernel_size in kernel_sizes]

    def call(self, inputs):
        conv_outputs = [conv_block(inputs) for conv_block in self.conv_blocks]
        # print('conv block 1', conv_outputs[0].shape)
        # print('conv block 2', conv_outputs[1].shape)
        return tf.concat(conv_outputs, axis=-1)

class BHICoreBlock(tf.keras.Model):
    def __init__(self, out_channels=320, seqsize=110, 
                 lstm_hidden_channels=320, kernel_sizes=[9, 15], 
                 pool_size=1, dropout1=0.2, dropout2=0.5):
        super(BHICoreBlock, self).__init__()

        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each_conv_out_channels = out_channels // len(kernel_sizes)

        self.lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_hidden_channels, return_sequences=True, time_major = False)
        )
        
        self.conv_blocks = [ConvBlock(each_conv_out_channels, 
                                      kernel_size, pool_size, dropout1) for kernel_size in kernel_sizes]
        
        self.dropout = tf.keras.layers.Dropout(dropout2)

    def call(self, inputs):
        x = self.lstm(inputs)
        conv_outputs = [conv_block(x) for conv_block in self.conv_blocks] 
        x = tf.concat(conv_outputs, axis=-1)  # concatenate the outputs along the channel dimension
        x = self.dropout(x)
        return x
    
class SeqNN(Model):
    def __init__(self, seqsize, in_ch=4, stem_ch=256, stem_ks = 7, target_bins=64, num_tasks = 1):
        super(SeqNN, self).__init__()
        
        self.in_ch = in_ch
        self.seqsize = seqsize
        self.target_bins = target_bins
        self.num_tasks = num_tasks

        self.stem = BHIFirstLayersBlock()        
        self.lstm = BHICoreBlock()

        # Branch 1. Profile prediction
        # Step 1.1 - 1D convolution with a very large kernel
        self.prof_out_precrop = Conv1D(filters=num_tasks,
                            kernel_size=75,
                            padding='valid',
                            name='wo_bias_bpnet_prof_out_precrop')

        self.prof = TargetLengthCrop(self.target_bins)
        
        self.profile_out = Flatten(name="wo_bias_bpnet_logits_profile_predictions")

        # Branch 2. Counts prediction
        # Step 2.1 - Global average pooling along the "length", the result
        #            size is same as "filters" parameter to the BPNet function
        self.gap_combined_conv = GlobalAvgPool1D(name='gap') # acronym - gapcc

        # Step 2.3 Dense layer to predict final counts
        self.count_out = Dense(num_tasks, name="wo_bias_bpnet_logcount_predictions")

    def call(self, inputs):
        x = self.stem(inputs)
        
        x = self.lstm(x)
        
        prof_out_precrop = self.prof_out_precrop(x)
        prof = self.prof(prof_out_precrop)
        profile_out = self.profile_out(prof)

        # print(profile_out.shape, 'profile_out')
        
        gap_combined_conv = self.gap_combined_conv(x)
        count_out = self.count_out(gap_combined_conv)

        # print('count_out', count_out.shape)
        
        # return [profile_out, count_out]
        return {'profile_out': profile_out, 'count_out': count_out}