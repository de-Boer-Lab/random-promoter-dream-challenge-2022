import tensorflow as tf
from tensorflow.keras.layers import Dense, Layer, Concatenate, AveragePooling1D
from tensorflow.keras import Model
from tensorflow.keras.activations import sigmoid, linear
from tensorflow import transpose, matmul, expand_dims
from tensorflow import keras
from tensorflow import reshape
from tensorflow.keras import layers as layers
from tensorflow.keras import Model
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Input, Cropping1D, add, Conv1D, GlobalAvgPool1D, Dense, Flatten


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
    def __init__(self, out_channels=320, kernel_sizes=[9, 15],
                 pool_size=1, dropout=0.2):
        super(BHIFirstLayersBlock, self).__init__()
        assert out_channels % len(kernel_sizes) == 0, "out_channels must be divisible by the number of kernel sizes"
        each_out_channels = out_channels // len(kernel_sizes)
        self.conv_blocks = [ConvBlock(each_out_channels, kernel_size, pool_size, dropout) 
                            for kernel_size in kernel_sizes]

    def call(self, inputs):
        conv_outputs = [conv_block(inputs) for conv_block in self.conv_blocks]
        return tf.concat(conv_outputs, axis=-1)
    
class SELayer(tf.keras.layers.Layer):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.squeeze = tf.keras.layers.Dense(inp // reduction, activation='swish')
        self.excite = tf.keras.layers.Dense(oup, activation='sigmoid')
    def call(self, inputs):
        b, _, c = tf.unstack(tf.shape(inputs))
        y = tf.reduce_mean(tf.reshape(inputs, (b, -1, c)), axis=1)
        y = self.squeeze(y)
        y = self.excite(y)
        y = tf.reshape(y, (b, 1, c))
        return inputs * y
    
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
    
class AutosomeCoreBlock(tf.keras.Model):
    def __init__(
        self,
        in_channels: int = 320,
        out_channels: int = 64,
    ):
        super().__init__()

        self.resize_factor = 4
        self.se_reduction = 4
        self.bn_momentum = 0.1
        self.filter_per_group = 2
        activation = tf.keras.activations.swish
        ks = 7
        block_sizes = [128, 128, 64, 64, 64]
        self.block_sizes = [in_channels] + block_sizes + [out_channels]

        self.blocks = []
        
        for ind, (prev_sz, sz) in enumerate(zip(self.block_sizes[:-1], self.block_sizes[1:])):
            inv_res_blc = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    filters=sz * self.resize_factor,
                    kernel_size=1,
                    padding='same',
                    use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=self.bn_momentum),
                tf.keras.layers.Activation(activation),
                
                tf.keras.layers.Conv1D(
                    filters=sz * self.resize_factor,
                    kernel_size=ks,
                    groups=sz * self.resize_factor // self.filter_per_group,
                    padding='same',
                    use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=self.bn_momentum),
                tf.keras.layers.Activation(activation),
                SELayer(prev_sz, sz * self.resize_factor, reduction=self.se_reduction),
                
                tf.keras.layers.Conv1D(
                    filters=prev_sz,
                    kernel_size=1,
                    padding='same',
                    use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=self.bn_momentum),
                tf.keras.layers.Activation(activation),
            ])

            resize_blc = tf.keras.Sequential([
                tf.keras.layers.Conv1D(
                    filters=sz,
                    kernel_size=ks,
                    padding='same',
                    use_bias=False
                ),
                tf.keras.layers.BatchNormalization(momentum=self.bn_momentum),
                tf.keras.layers.Activation(activation),
            ])
            
            self.blocks.append(inv_res_blc)
            self.blocks.append(resize_blc)
            
    def call(self, x):
        for i in range(0, len(self.blocks), 2):
            x = tf.concat([x, self.blocks[i](x)], axis=-1)
            x = self.blocks[i + 1](x)
        return x


class SeqNN(Model):
    def __init__(self, seqsize, in_ch=4, stem_ch=320, stem_ks = 7, target_bins=64, num_tasks = 1):
        super(SeqNN, self).__init__()
        self.in_ch = in_ch
        self.seqsize = seqsize
        self.target_bins = target_bins
        self.num_tasks = num_tasks
        
        in_ch = stem_ch
        out_ch = stem_ch
        
        self.stem = BHIFirstLayersBlock(
                    out_channels = 320,
                    kernel_sizes = [9, 15],
                    pool_size = 1,
                    dropout = 0.2
                    )

        self.legnet = AutosomeCoreBlock()
                
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
        # print(x.shape, 'input')
        
        x = self.legnet(x)
        # print(x.shape, 'legnet')
        prof_out_precrop = self.prof_out_precrop(x)
        prof = self.prof(prof_out_precrop)
        profile_out = self.profile_out(prof)

        # print(profile_out.shape, 'profile_out')
        
        gap_combined_conv = self.gap_combined_conv(x)
        count_out = self.count_out(gap_combined_conv)
        # return [profile_out, count_out]
        return {'profile_out': profile_out, 'count_out': count_out}