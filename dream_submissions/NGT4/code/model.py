
from builtins import super
from function import *
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

def call_model(x):
    dic = {'model_12':Model_12(),
           'model_7' :Model_7(),
           'xception_1d':Xception_1d(),
           'xception_1d_2': Xception_1d_2()}
    return dic[x]
class Model_12(keras.Model):
    def __init__(self):
        super().__init__()
        weight_decay = 1e-4
        n = 1
        kernel_size = 5
        # 440
        
        self.conv1d_1 = keras.layers.Conv1D(filters= 64 * n,
                                            kernel_size= 80,
                                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                                            activation='relu',
                                            )
        
        self.conv1d_2 = keras.layers.Conv1D(filters= 64 * n,
                                            kernel_size= 80,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                                            activation='relu',
                                            )
        
        # 220
        self.pool1d_1 = keras.layers.MaxPool1D(pool_size= 2)
        
        self.conv1d_3 = keras.layers.Conv1D(filters= 128 * n,
                                            kernel_size= 40,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                                            activation='relu',
                                            )
        self.conv1d_4 = keras.layers.Conv1D(filters= 128 * n,
                                            kernel_size= 40,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                                            activation='relu',
                                            )
        
        # 110
        self.pool1d_2 = keras.layers.MaxPool1D(pool_size= 2)
        
        self.conv1d_5 = keras.layers.Conv1D(filters= 512 * n,
                                            kernel_size= 20,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                                            activation='relu',
                                            )
        self.conv1d_6 = keras.layers.Conv1D(filters= 512 * n,
                                            kernel_size= 20,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(weight_decay),
                                            activation='relu',
                                            )
        
        # 55
        self.pool1d_3 = keras.layers.MaxPool1D(pool_size=2)

        self.conv1d_7 = keras.layers.Conv1D(filters=512 * n,
                                            kernel_size= 10,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )
        self.conv1d_8 = keras.layers.Conv1D(filters=512 * n,
                                            kernel_size= 10,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )
        # 28
        self.pool1d_4 = keras.layers.MaxPool1D(pool_size=2)

        self.conv1d_9 = keras.layers.Conv1D(filters=512 * n,
                                            kernel_size=5,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )
        self.conv1d_10 = keras.layers.Conv1D(filters=512 * n,
                                            kernel_size=5,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )
        # フラット化
        self.flat = keras.layers.Flatten()
        
        # ドロップアウト
        self.drop = keras.layers.Dropout(0.4)
        
        self.dense_1 = keras.layers.Dense(512,
            activation='relu'
        )
        self.dense_2 = keras.layers.Dense(512,
            activation='relu'
        )
        self.dense1 = keras.layers.Dense(1)
        
        # 全ての層をリスト化
        self.ls = [self.conv1d_1,
                   self.conv1d_2,
                   self.pool1d_1,
                   self.conv1d_3,
                   self.conv1d_4,
                   self.pool1d_2,
                   self.conv1d_5,
                   self.conv1d_6,
                   self.pool1d_3,
                   self.conv1d_7,
                   self.conv1d_8,
                   self.pool1d_4,
                   self.conv1d_9,
                   self.conv1d_10,
                   self.flat,
                   self.drop,
                   self.dense_1,
                   self.dense_2,
                   self.dense1,
                   ]
    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        #x = keras.Input(shape=(440, 1))
        for layer in self.ls:
            x = layer(x)
        x = tf.reshape(x,[-1])
        return x


class Model_7(keras.Model):
    def __init__(self):
        super().__init__()
        weight_decay = 1e-4
        n = 1
        kernel_size = 5
        # 440

        self.conv1d_1 = keras.layers.Conv1D(filters=64 * n,
                                            kernel_size=80,
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )


        # 220
        self.pool1d_1 = keras.layers.MaxPool1D(pool_size=2)

        self.conv1d_3 = keras.layers.Conv1D(filters=128 * n,
                                            kernel_size=40,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )

        # 110
        self.pool1d_2 = keras.layers.MaxPool1D(pool_size=2)

        self.conv1d_5 = keras.layers.Conv1D(filters=512 * n,
                                            kernel_size=20,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )

        # 55
        self.pool1d_3 = keras.layers.MaxPool1D(pool_size=2)

        self.conv1d_7 = keras.layers.Conv1D(filters=512 * n,
                                            kernel_size=10,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )

        # 28
        self.pool1d_4 = keras.layers.MaxPool1D(pool_size=2)

        self.conv1d_9 = keras.layers.Conv1D(filters=512 * n,
                                            kernel_size=5,
                                            padding='same',
                                            kernel_regularizer=keras.regularizers.l2(
                                                weight_decay),
                                            activation='relu',
                                            )

        # フラット化
        self.flat = keras.layers.Flatten()

        # ドロップアウト
        self.drop = keras.layers.Dropout(0.4)

        self.dense_1 = keras.layers.Dense(512,
                                          activation='relu'
                                          )
        self.dense_2 = keras.layers.Dense(512,
                                          activation='relu'
                                          )
        self.dense1 = keras.layers.Dense(1)

        # 全ての層をリスト化
        
        self.ls = [self.conv1d_1,
                   self.pool1d_1,
                   self.conv1d_3,
                   self.pool1d_2,
                   self.conv1d_5,
                   self.pool1d_3,
                   self.conv1d_7,
                   self.pool1d_4,
                   self.conv1d_9,
                   self.flat,
                   self.drop,
                   self.dense_1,
                   self.dense_2,
                   self.dense1,
                   ]


    def call(self, x):
        x = tf.expand_dims(x, axis=-1)
        #x = keras.Input(shape=(440, 1))
        for layer in self.ls:
            x = layer(x)
        x = tf.reshape(x,[-1])
        print(x)
        return x

class Inception_1d(keras.Model):
    def __init__(self):
        super().__init__()
        self.model = self.main_model()
        
    def __call__(self):
        return self.model
    class lay():
        def __call__(self,inputs):
            x = keras.layers.Convolution2D(32, (1, 4), padding='same',)(inputs)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            return x 
    def main_model(self):
        inputs = keras.Input(shape=(100,4,1))
        lay1 = self.lay()
        lay2 = self.lay()
        x = lay1(inputs)
        x = lay2(x)
        model = keras.models.Model(inputs=inputs, outputs=x)
        return model

class Xception_1d():
    def __init__(self):
        super().__init__()
        self.entry_main_branch1 = self.Entry_main_branch(64,(3,4),(1,2))
        self.entry_sub_branch1 = self.Entry_sub_branch(64,(3,4),(1,2))
        self.entry_main_branch2 = self.Entry_main_branch(128,(3,2),(1,2))
        self.entry_sub_branch2 = self.Entry_sub_branch(128,(3,2),(1,2))
        self.entry_main_branch3 = self.Entry_main_branch(256,(3,1),(2,1))
        self.entry_sub_branch3 = self.Entry_sub_branch(256,(3,1),(2,1))
        self.entry_main_branch4 = self.Entry_main_branch(512,(3,1),(2,1))
        self.entry_sub_branch4 = self.Entry_sub_branch(512,(3,1),(2,1))
        self.middle = self.Middle(8)
        self.end_main_branch = self.Entry_main_branch(512,(3,1),(2,1))
        self.end_sub_branch = self.Entry_sub_branch(512,(3,1),(2,1))
        
    def __call__(self,weight = False):
        inputs = keras.Input(shape=(100,4,1))
        # entry flow
        x = keras.layers.Activation('relu')(inputs)
        x = keras.layers.Convolution2D(64, (3,4),padding= 'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Convolution2D(64, (3,4),padding= 'same')(x)
        x = keras.layers.BatchNormalization()(x)
        
        x1 = self.entry_main_branch1(x)
        x2 = keras.layers.Activation('relu')(x)
        x2 = self.entry_sub_branch1(x2)
        x = keras.layers.add([x1,x2])

        x1 = self.entry_main_branch2(x)
        x2 = self.entry_sub_branch2(x)
        x = keras.layers.add([x1,x2])
        
        x1 = self.entry_main_branch3(x)
        x2 = self.entry_sub_branch3(x)
        x = keras.layers.add([x1,x2])
        
        x1 = self.entry_main_branch4(x)
        x2 = self.entry_sub_branch4(x)
        x = keras.layers.add([x1,x2])
        # middle flow
        x = self.middle(x)

        # exit flow
        x1 = self.end_main_branch(x)
        x2 = self.end_sub_branch(x)
        x = keras.layers.add([x1,x2])

        x = keras.layers.SeparableConvolution2D(1024, (3,1),padding= 'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConvolution2D(1024, (3,1),padding= 'same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1, kernel_initializer='he_normal')(x)
        x = tf.reshape(x,[-1])
        model = keras.models.Model(inputs=inputs, outputs=x)
        if weight:
            model.load_weights(weight)
        return model
    def get_summary(self):
        model = self.__call__()
        model.compile(optimizer=keras.optimizers.Adam())
        model.summary()
    class Entry_main_branch():
        def __init__(self,filters,conv_shape,pool_shape):
            super().__init__()
            self.filters = filters
            self.conv_shape = conv_shape
            self.pool_shape = pool_shape
        def __call__(self,x):
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.SeparableConvolution2D(self.filters,
                                                    self.conv_shape,padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.SeparableConvolution2D(self.filters,
                                                    self.conv_shape,padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D(self.conv_shape,strides= self.pool_shape,padding = 'same')(x)
            
            
            return x
        
    class Entry_sub_branch():
        def __init__(self,filters,conv_shape,pool_shape):
            super().__init__()
            self.filters = filters
            self.conv_shape = conv_shape
            self.pool_shape = pool_shape
        def __call__(self,x):
            x = keras.layers.Convolution2D(self.filters, self.conv_shape, strides= self.pool_shape, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            return x
    class Middle():
        def __init__(self,n):
            super().__init__()
            self.n = n
        def __call__(self,x):
            for i in range(self.n):
                x1 = keras.layers.Activation('relu')(x)
                x1 = keras.layers.SeparableConv2D(512, (3,1), padding='same')(x1)
                x1 = keras.layers.BatchNormalization()(x1)
                x1 = keras.layers.Activation('relu')(x1)
                x1 = keras.layers.SeparableConv2D(512, (3,1), padding='same')(x1)
                x1 = keras.layers.BatchNormalization()(x1)
                x1 = keras.layers.Activation('relu')(x1)
                x1 = keras.layers.SeparableConv2D(512, (3,1), padding='same')(x1)
                x1 = keras.layers.BatchNormalization()(x1)
                x = keras.layers.add([x1,x])
            return x


class Xception_1d_2():
    def __init__(self):
        super().__init__()
        self.entry_main_branch1 = self.Entry_main_branch(128, (10, 4), (1, 2))
        self.entry_sub_branch1 = self.Entry_sub_branch(128, (10, 4), (1, 2))
        self.entry_main_branch2 = self.Entry_main_branch(128, (5, 2), (1, 2))
        self.entry_sub_branch2 = self.Entry_sub_branch(128, (5, 2), (1, 2))
        self.entry_main_branch3 = self.Entry_main_branch(256, (3, 1), (2, 1))
        self.entry_sub_branch3 = self.Entry_sub_branch(256, (3, 1), (2, 1))
        self.entry_main_branch4 = self.Entry_main_branch(512, (3, 1), (2, 1))
        self.entry_sub_branch4 = self.Entry_sub_branch(512, (3, 1), (2, 1))
        self.middle = self.Middle(8)
        self.end_main_branch = self.Entry_main_branch(512, (3, 1), (2, 1))
        self.end_sub_branch = self.Entry_sub_branch(512, (3, 1), (2, 1))

    def __call__(self, weight=False):
        inputs = keras.Input(shape=(100, 4, 1))

        # entry flow
        x = keras.layers.Activation('relu')(inputs)
        x = keras.layers.Convolution2D(128, (25, 4), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.Convolution2D(128, (25, 4), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)

        x1 = self.entry_main_branch1(x)
        x2 = keras.layers.Activation('relu')(x)
        x2 = self.entry_sub_branch1(x2)
        x = keras.layers.add([x1, x2])

        x1 = self.entry_main_branch2(x)
        x2 = self.entry_sub_branch2(x)
        x = keras.layers.add([x1, x2])

        x1 = self.entry_main_branch3(x)
        x2 = self.entry_sub_branch3(x)
        x = keras.layers.add([x1, x2])

        x1 = self.entry_main_branch4(x)
        x2 = self.entry_sub_branch4(x)
        x = keras.layers.add([x1, x2])
        # middle flow
        x = self.middle(x)

        # exit flow
        x1 = self.end_main_branch(x)
        x2 = self.end_sub_branch(x)
        x = keras.layers.add([x1, x2])

        x = keras.layers.SeparableConvolution2D(
            1024, (3, 1), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)
        x = keras.layers.SeparableConvolution2D(
            1024, (3, 1), padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dense(1, kernel_initializer='he_normal')(x)
        x = tf.reshape(x, [-1])
        model = keras.models.Model(inputs=inputs, outputs=x)
        if weight:
            model.load_weights(weight)
        return model

    def get_summary(self):
        model = self.__call__()
        model.compile(optimizer=keras.optimizers.Adam())
        model.summary()

    class Entry_main_branch():
        def __init__(self, filters, conv_shape, pool_shape):
            super().__init__()
            self.filters = filters
            self.conv_shape = conv_shape
            self.pool_shape = pool_shape

        def __call__(self, x):
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.SeparableConvolution2D(self.filters,
                                                    self.conv_shape, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation('relu')(x)
            x = keras.layers.SeparableConvolution2D(self.filters,
                                                    self.conv_shape, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.MaxPooling2D(
                self.conv_shape, strides=self.pool_shape, padding='same')(x)

            return x

    class Entry_sub_branch():
        def __init__(self, filters, conv_shape, pool_shape):
            super().__init__()
            self.filters = filters
            self.conv_shape = conv_shape
            self.pool_shape = pool_shape

        def __call__(self, x):
            x = keras.layers.Convolution2D(
                self.filters, self.conv_shape, strides=self.pool_shape, padding='same')(x)
            x = keras.layers.BatchNormalization()(x)
            return x

    class Middle():
        def __init__(self, n):
            super().__init__()
            self.n = n

        def __call__(self, x):
            for i in range(self.n):
                x1 = keras.layers.Activation('relu')(x)
                x1 = keras.layers.SeparableConv2D(
                    512, (3, 1), padding='same')(x1)
                x1 = keras.layers.BatchNormalization()(x1)
                x1 = keras.layers.Activation('relu')(x1)
                x1 = keras.layers.SeparableConv2D(
                    512, (3, 1), padding='same')(x1)
                x1 = keras.layers.BatchNormalization()(x1)
                x1 = keras.layers.Activation('relu')(x1)
                x1 = keras.layers.SeparableConv2D(
                    512, (3, 1), padding='same')(x1)
                x1 = keras.layers.BatchNormalization()(x1)
                x = keras.layers.add([x1, x])
            return x
