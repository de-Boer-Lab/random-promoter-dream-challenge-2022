import datetime
import json
import os

import keras
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import tensorflow.keras as keras  # # important to make sure non tf.keras is hidden
from scipy.stats import *
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.backend import conv1d
from tensorflow.keras.layers import (LSTM, Activation, Add, BatchNormalization,
                                     Bidirectional, Concatenate, Conv1D,
                                     Conv2D, Dense, Dropout, Flatten, Lambda,
                                     LeakyReLU, MaxPooling1D, MaxPooling2D,
                                     Permute, Reshape, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from tensorflow.python.client import device_lib
from tensorflow.python.keras.utils import conv_utils
from tqdm import tqdm_notebook as tqdm

def build_model(model_params):
    input_layer = Input(shape=(110,4)) 
    x_f, x_rc = rc_Conv1D(
        filters=model_params["motif_conv_hidden"], 
        kernel_size=model_params["conv_width_motif"], 
        padding='same' ,
        kernel_regularizer = l1_l2(
            l1=model_params["l1_weight"], 
            l2=model_params["l2_weight"],
        ), 
        kernel_initializer='he_normal',
        data_format = 'channels_last',
        use_bias=False,
        trainable=model_params["trainable_layers"]["block1"]["rc_conv1d"]
    )(input_layer)

    x_f = BatchNormalization()(x_f)
    x_rc = BatchNormalization()(x_rc)

    x_f = Activation('relu')(x_f)
    x_rc = Activation('relu')(x_rc)


    mha_f_input = x_f
    x_f = MultiHeadAttention(
        head_num=model_params["n_heads"],
        name='Multi-Head-forward',
        kernel_regularizer = l1_l2(
            l1=model_params["l1_weight"], 
            l2=model_params["l2_weight"],
        ), 
    )(x_f)

    mha_rc_input = x_rc
    x_rc = MultiHeadAttention(
        head_num=model_params["n_heads"],
        name='Multi-Head-reverse-comp',
        kernel_regularizer = l1_l2(
            l1=model_params["l1_weight"], 
            l2=model_params["l2_weight"],
        ), 
    )(x_rc)

    if model_params["dropout_rate"] > 0.0:

        x_f = Dropout(rate=model_params["attention_dropout_rate"])(x_f)
        x_rc = Dropout(rate=model_params["attention_dropout_rate"])(x_rc)

    x_f = Add()([mha_f_input, x_f])
    x_rc = Add()([mha_rc_input, x_rc])

    x_f = LayerNormalization()(x_f)
    x_rc = LayerNormalization()(x_rc)

    ff_f_input = x_f
    ff_rc_input = x_rc

    x_f  = FeedForward(
        units= model_params["n_heads"], 
        kernel_regularizer = l1_l2(
            l1=model_params["l1_weight"], 
            l2=model_params["l2_weight"],
        ), 
    )(x_f)
    x_rc  = FeedForward(
        units= model_params["n_heads"], 
        kernel_regularizer = l1_l2(
            l1=model_params["l1_weight"], 
            l2=model_params["l2_weight"],
        ), 
    )(x_rc)
    if model_params["dropout_rate"] > 0.0:
        x_f = Dropout(rate=model_params["attention_dropout_rate"])(x_f)
        x_rc = Dropout(rate=model_params["attention_dropout_rate"])(x_rc)

    x_f = Add()([ff_f_input, x_f])
    x_rc = Add()([ff_rc_input, x_rc])

    x_f = LayerNormalization()(x_f)    
    x_rc = LayerNormalization()(x_rc)    

    x_f = Bidirectional(
            LSTM(
                model_params["n_heads"], 
                return_sequences=True,
                kernel_regularizer  = l1_l2(
                    l1=model_params["l1_weight"], 
                    l2=model_params["l2_weight"],
                ),
                kernel_initializer='he_normal' , 
                dropout = model_params["dropout_rate"]
            )
    )(x_f)
    x_f = Dropout(model_params["dropout_rate"])(x_f)

    x_rc = Bidirectional(
            LSTM(
                model_params["n_heads"], 
                return_sequences=True,
                kernel_regularizer  = l1_l2(
                    l1=model_params["l1_weight"], 
                    l2=model_params["l2_weight"],
                ),
                kernel_initializer='he_normal' , 
                dropout = model_params["dropout_rate"]
            )
    )(x_rc)
    x_rc = Dropout(model_params["dropout_rate"])(x_rc)

    x_f = Lambda(lambda x : K.expand_dims(x,axis=1))(x_f)
    x_rc = Lambda(lambda x : K.expand_dims(x,axis=1))(x_rc)

    x =Concatenate(axis=1)([x_f, x_rc])

    x = keras.layers.ZeroPadding2D(
        padding = (
            (0,0 ),
            (int(model_params["conv_hidden"]/2)-1, int(model_params["conv_hidden"]/2))
        ), 
        data_format = 'channels_last'
    )(x)

    x = Conv2D(
            model_params["conv_hidden"], 
            (2, model_params["conv_hidden"]), 
            padding='valid',
            kernel_regularizer  = l1_l2(
                l1=model_params["l1_weight"], 
                l2=model_params["l2_weight"],
            ),
            kernel_initializer='he_normal',
            data_format = 'channels_last' , 
            use_bias=False
    )(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Lambda(lambda x : K.squeeze(x,axis=1))(x)

    x = Conv1D(
            model_params["conv_hidden"], 
            (model_params["conv_hidden"]), 
            padding='same',
            kernel_regularizer  = l1_l2(
                l1=model_params["l1_weight"], 
                l2=model_params["l2_weight"],
            ),
            kernel_initializer='he_normal' ,
            data_format = 'channels_last' , 
            use_bias=False
    )(x) 

    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    mha_input = x
    x = MultiHeadAttention(
        head_num=model_params["n_heads"],
        name='Multi-Head-concat',
        kernel_regularizer = l1_l2(
            l1=model_params["l1_weight"], 
            l2=model_params["l2_weight"],
        ), 
    )(x)


    if model_params["dropout_rate"] > 0.0:
        x = Dropout(rate=model_params["attention_dropout_rate"])(x)
    else:
        x = x
    x = Add()([mha_input, x])
    x = LayerNormalization()(x)

    ff_input = x
    x  = FeedForward(
        units= model_params["n_heads"], 
        kernel_regularizer = l1_l2(
            l1=model_params["l1_weight"], 
            l2=model_params["l2_weight"],
        ), 
    )(x)

    if model_params["dropout_rate"] > 0.0:
        x = Dropout(rate=model_params["attention_dropout_rate"])(x)
    else:
        x = x
    x = Add()([ff_input, x])
    x = LayerNormalization()(x)    



    x = Bidirectional(
            LSTM(
                model_params["n_heads"], 
                return_sequences=True,
                kernel_regularizer = l1_l2(
                    l1=model_params["l1_weight"], 
                    l2=model_params["l2_weight"],
                ), 
                kernel_initializer='he_normal' , 
                dropout = model_params["dropout_rate"]
            )
    )(x)
    x = Dropout(model_params["dropout_rate"])(x)


    if(len(x.get_shape())>2):
        x = Flatten()(x) 


    x = Dense(
            int(model_params["n_hidden"]), 
            kernel_regularizer = l1_l2(
                l1=model_params["l1_weight"], 
                l2=model_params["l2_weight"],
            ), 
            kernel_initializer='he_normal', 
            use_bias=True
    )(x)

    x = Activation('relu')(x) 

    x = Dropout(model_params["dropout_rate"])(x)

    x = Dense(
            int(model_params["n_hidden"]), 
            kernel_regularizer = l1_l2(
                l1=model_params["l1_weight"], 
                l2=model_params["l2_weight"],
            ),
            kernel_initializer='he_normal', 
            use_bias=True
    )(x)
    x = Activation('relu')(x)
    x = Dropout(model_params["dropout_rate"])(x) #https://arxiv.org/pdf/1801.05134.pdf

    output_layer = Dense(
        1, 
        kernel_regularizer = l1_l2(
            l1=model_params["l1_weight"], 
            l2=model_params["l2_weight"],
        ), 
        activation='linear', 
        kernel_initializer='he_normal', 
        use_bias=True
    )(x) 

    model = Model(input_layer, output_layer)
    lr = model_params["lr"]
    optimizer = model_params["optimizer"]
    loss = model_params["loss"]
    
    if optimizer == "RMSprop":
        opt = tf.keras.optimizers.RMSprop(lr)
    elif optimizer == "Adam":
        opt = tf.keras.optimizers.Adam(learning_rate=lr)
    elif optimizer == "Nadam":
        opt = tf.keras.optimizers.Nadam(learning_rate=lr)
    else:
        assert False, f"Please specify a parameter in 'model_params'."
    
    model.compile(
        optimizer=opt, 
        loss=loss,
        metrics=["mean_squared_error", r_square]
    )
    print(model.summary())
    return model


def _preprocess_x(list_of_sequences):
    def seq2feature(data,mapper,worddim):
        transformed = np.zeros([data.shape[0],1,len(data[0]),4] , dtype=np.bool )
        for i in tqdm(range(data.shape[0])):
            for j,k in enumerate(data[i]):
                #print j,k
                transformed[i,0,j] = mapper[k] 
                #print mapper[k]
        return transformed

    # Add 'N' to sequences that are not of length 110.
    for i in range(0,len(list_of_sequences)) : 
        if (len(list_of_sequences[i]) > 110) :
            list_of_sequences[i] = list_of_sequences[i][-110:]
        if (len(list_of_sequences[i]) < 110) : 
            while (len(list_of_sequences[i]) < 110) :
                list_of_sequences[i] = 'N'+list_of_sequences[i]
                
    A_onehot = np.array([1,0,0,0] ,  dtype=np.bool)
    C_onehot = np.array([0,1,0,0] ,  dtype=np.bool)
    G_onehot = np.array([0,0,1,0] ,  dtype=np.bool)
    T_onehot = np.array([0,0,0,1] ,  dtype=np.bool)
    N_onehot = np.array([0,0,0,0] ,  dtype=np.bool)
    
    mapper = {'A':A_onehot,'C':C_onehot,'G':G_onehot,'T':T_onehot,'N':N_onehot}
    worddim = len(mapper['A'])
    seqdata = np.asarray(list_of_sequences)
    
    seqdata_transformed = seq2feature(seqdata, mapper, worddim)
    
    return np.squeeze(seqdata_transformed)

def _preprocess_y(list_of_expressions):
    return np.asarray(list_of_expressions).astype('float')

def _get_datasets(
    path_to_train_sequences: str="/home/b330-admin/data/train_sequences.txt",
    random_state=420
):
    data = pd.read_csv(
        path_to_train_sequences,
        delimiter="\t",
        names=["sequence", "expression"]
    )

    number_of_remaining_data_points = data.shape[0] % 1024

    data = data.iloc[:data.shape[0] - number_of_remaining_data_points]

    X_train, X_test, y_train, y_test = train_test_split(
        data["sequence"], 
        data["expression"],
        test_size=(1500)/(data.shape[0] / 1024),
        random_state=random_state
    )

    validation_train_size = X_train.shape[0] - (1024 * 50)
    validation_train_size = validation_train_size/X_train.shape[0]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, 
        y_train, 
        train_size=validation_train_size,
        random_state=random_state
    )

    assert X_train.shape[0] % 1024 == 0
    assert X_valid.shape[0] % 1024 == 0

    preprocessed_xtrain = _preprocess_x(list(X_train))
    print('...done with xtrain')
    preprocessed_ytrain = _preprocess_y(list(y_train))
    print('...done with ytrain')

    preprocessed_xtest = _preprocess_x(list(X_test))
    print('...done with xtest')
    preprocessed_ytest = _preprocess_y(list(y_test))
    print('...done with ytest')

    preprocessed_xval = _preprocess_x(list(X_valid))
    print('...done with xvalid')
    preprocessed_yval = _preprocess_y(list(y_valid))
    print('...done with yvalid')

    return (
        preprocessed_xtrain,
        preprocessed_ytrain,
        preprocessed_xtest,
        preprocessed_ytest,
        preprocessed_xval,
        preprocessed_yval
    )

def _create_weights_folder(
    output_folder: str="/home/b330-admin/data",
    model_name: str="DeepAtt",
    additional_info: str=""
):
    path_to_model_weights_folder = os.path.join(
        output_folder,
        model_name,
        
    )

    now = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")
    checkpoint_path = os.path.join(
        path_to_model_weights_folder,
        f"{additional_info}_" + now
    )

    if not os.path.exists(path_to_model_weights_folder):
        os.mkdir(path_to_model_weights_folder)

    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
        
    return checkpoint_path

def _create_folders_and_callbacks(
    model_params: dict,
    model_name: str="DeepAtt",
    additional_info: str="",
):
    path_to_weights_folder = _create_weights_folder(
        model_name=model_name,
        additional_info=additional_info
    )

    path_to_logs = os.path.join(path_to_weights_folder, "logs")

    if not os.path.exists(path_to_logs):
        os.mkdir(path_to_logs)
        
    folder_of_best_weights = os.path.join(path_to_weights_folder, "best_weights")
    if not os.path.exists(folder_of_best_weights):
        os.mkdir(folder_of_best_weights)

    # path_to_weights_folder
    #   model_name
    #       configuration_file.json
    #       logs/       <--- for tensorboard
    #           train/
    #           validation/
    #       best_weights/ <--- best weights according to validation loss
    #           best_weights.h5
    #       model.epoch-01-loss-0.01-val-loss-0.01.h5 <--- all weights



    callbacks = [
        # Save all weights.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                path_to_weights_folder,
                'model.epoch-{epoch:02d}_train-loss-{loss:.5f}_val-loss-{val_loss:.5f}.h5'),
            
        ),
        
        # Save best weights.
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(
                folder_of_best_weights,
                'best-weights.h5'
            ),
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            save_best_only=True
        ),

        # Tensorboard
        keras.callbacks.TensorBoard(
            log_dir=path_to_logs,
            histogram_freq=1
        )
    
    ]

    print(f"Path to weights and config: {path_to_weights_folder}")
    with open(os.path.join(path_to_weights_folder, "configuration_file.json"), "w") as f:
        json.dump(model_params, f)
    
    return callbacks

def _get_augmented_datasets(random_state: int=420, normal_dist_variance: float=0.3):
    """
    Rather than keeping the original dataset with a lot of integer expression values,
    replace integer values with those from a Normal(integer, normal_dist_variance) distribution.
    """
    data = pd.read_csv(
        "/home/b330-admin/data/train_sequences.txt",
        delimiter="\t",
        names=["sequence", "expression"]
    )

    number_of_remaining_data_points = data.shape[0] % 1024

    data = data.iloc[:data.shape[0] - number_of_remaining_data_points]

    X_train, X_test, y_train, y_test = train_test_split(
        data["sequence"], 
        data["expression"],
        test_size=(1500)/(data.shape[0] / 1024),
        random_state=random_state
    )

    validation_train_size = X_train.shape[0] - (1024 * 50)
    validation_train_size = validation_train_size/X_train.shape[0]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, 
        y_train, 
        train_size=validation_train_size,
        random_state=random_state
    )
    
    y_train_df = pd.DataFrame()
    y_train_df["expression"] = y_train
    
    y_train_df["new_expression"] = y_train_df["expression"]
    for integer_expression in tqdm(range(18)):
        subset = y_train_df[y_train_df["expression"] == integer_expression]
        new_expressions = np.random.normal(
            integer_expression,
            normal_dist_variance,
            size=subset.shape[0]
        )
        y_train_df.loc[subset.index, "new_expression"] = new_expressions
        
    y_train = y_train_df["new_expression"]
    
    preprocessed_xtrain = _preprocess_x(list(X_train))
    print('...done with xtrain')
    preprocessed_ytrain = _preprocess_y(list(y_train))
    print('...done with ytrain')

    preprocessed_xtest = _preprocess_x(list(X_test))
    print('...done with xtest')
    preprocessed_ytest = _preprocess_y(list(y_test))
    print('...done with ytest')

    preprocessed_xval = _preprocess_x(list(X_valid))
    print('...done with xvalid')
    preprocessed_yval = _preprocess_y(list(y_valid))
    print('...done with yvalid')

    return (
        preprocessed_xtrain,
        preprocessed_ytrain,
        preprocessed_xtest,
        preprocessed_ytest,
        preprocessed_xval,
        preprocessed_yval
    )

class FeedForward(keras.layers.Layer):
    """Position-wise feed-forward layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 units,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """Initialize the layer.
        :param units: Dimension of hidden units.
        :param activation: Activation for the first linear transformation.
        :param use_bias: Whether to use the bias term.
        :param kernel_initializer: Initializer for kernels.
        :param bias_initializer: Initializer for kernels.
        :param kernel_regularizer: Regularizer for kernels.
        :param bias_regularizer: Regularizer for kernels.
        :param kernel_constraint: Constraint for kernels.
        :param bias_constraint: Constraint for kernels.
        :param kwargs:
        """
        self.supports_masking = True
        self.units = int(units)
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.W1, self.b1 = None, None
        self.W2, self.b2 = None, None
        super(FeedForward, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'units': self.units,
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
        }
        base_config = super(FeedForward, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])
        self.W1 = self.add_weight(
            shape=(feature_dim, self.units),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W1'.format(self.name),
        )
        if self.use_bias:
            self.b1 = self.add_weight(
                shape=(self.units,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b1'.format(self.name),
            )
        self.W2 = self.add_weight(
            shape=(self.units, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='{}_W2'.format(self.name),
        )
        if self.use_bias:
            self.b2 = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='{}_b2'.format(self.name),
            )
        super(FeedForward, self).build(input_shape)

    def call(self, x, mask=None):
        h = K.dot(x, self.W1)
        if self.use_bias:
            h = K.bias_add(h, self.b1)
        if self.activation is not None:
            h = self.activation(h)
        y = K.dot(h, self.W2)
        if self.use_bias:
            y = K.bias_add(y, self.b2)
        return y
    
class MultiHead(keras.layers.Wrapper):

    def __init__(self,
                 layer,
                 layer_num=1,
                 hidden_dim=None,
                 use_bias=True,
                 reg_index=None,
                 reg_slice=None,
                 reg_factor=0.0,
                 **kwargs):
        """Initialize the wrapper layer.
        :param layer: The layer to be duplicated or a list of layers.
        :param layer_num: The number of duplicated layers.
        :param hidden_dim: A linear transformation will be applied to the input data if provided, otherwise the original
                           data will be feed to the sub-layers.
        :param use_bias: Whether to use bias in the linear transformation.
        :param reg_index: The index of weights to be regularized.
        :param reg_slice: The slice indicates which part of the weight to be regularized.
        :param reg_factor: The weights of the regularization.
        :param kwargs: Arguments for parent.
        """
        if type(layer) is list:
            self.layer = layer[0]
            self.layers = layer
            self.layer_num = int(len(self.layers))
            self.rename = False
        else:
            self.layer = layer
            self.layers = []
            self.layer_num = layer_num
            self.rename = True
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias
        if reg_index is None or type(reg_index) is list:
            self.reg_index = reg_index
        else:
            self.reg_index = [reg_index]
        if type(reg_slice) is list or reg_index is None:
            self.reg_slice = reg_slice
        else:
            self.reg_slice = [reg_slice] * len(self.reg_index)
        if reg_factor is None or type(reg_factor) is list or reg_index is None:
            self.reg_weight = reg_factor
        else:
            self.reg_weight = [reg_factor] * len(self.reg_index)

        self.W, self.b = None, None
        self.supports_masking = self.layer.supports_masking
        super(MultiHead, self).__init__(self.layer, **kwargs)

    def get_config(self):
        slices = None
        if self.reg_slice:
            slices = []
            for interval in self.reg_slice:
                if interval is None:
                    slices.append(None)
                elif type(interval) is slice:
                    slices.append([interval.start, interval.stop, interval.step])
                else:
                    slices.append([])
                    for sub in interval:
                        slices[-1].append([sub.start, sub.stop, sub.step])
        config = {
            'layers': [],
            'hidden_dim': self.hidden_dim,
            'use_bias': self.use_bias,
            'reg_index': self.reg_index,
            'reg_slice': slices,
            'reg_factor': self.reg_weight,
        }
        for layer in self.layers:
            config['layers'].append({
                'class_name': layer.__class__.__name__,
                'config': layer.get_config(),
            })
        base_config = super(MultiHead, self).get_config()
        base_config.pop('layer')
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        reg_slice = config.pop('reg_slice')
        if reg_slice is not None:
            slices = []
            for interval in reg_slice:
                if interval is None:
                    slices.append(None)
                elif type(interval[0]) is list:
                    slices.append([])
                    for sub in interval:
                        slices[-1].append(slice(sub[0], sub[1], sub[2]))
                    slices[-1] = tuple(slices[-1])
                else:
                    slices.append(slice(interval[0], interval[1], interval[2]))
            reg_slice = slices
        layers = [keras.layers.deserialize(layer, custom_objects=custom_objects) for layer in config.pop('layers')]
        return cls(layers, reg_slice=reg_slice, **config)

    def build(self, input_shape):
        if type(input_shape) == list:
            self.input_spec = list(map(lambda x: keras.engine.InputSpec(shape=x), input_shape))
        else:
            self.input_spec = keras.engine.InputSpec(shape=input_shape)
        if not self.layers:
            self.layers = [copy.deepcopy(self.layer) for _ in range(self.layer_num)]
        if self.hidden_dim is not None:
            self.W = self.add_weight(
                shape=(int(input_shape[-1]), int(self.hidden_dim * self.layer_num)),
                name='{}_W'.format(self.name),
                initializer=keras.initializers.get('uniform'),
            )
            if self.use_bias:
                self.b = self.add_weight(
                    shape=(int(self.hidden_dim * self.layer_num),),
                    name='{}_b'.format(self.name),
                    initializer=keras.initializers.get('zeros'),
                )
            input_shape = input_shape[:-1] + (self.hidden_dim,)
        for i, layer in enumerate(self.layers):
            if not layer.built:
                if self.rename:
                    layer.name = layer.name + '_%d' % (i + 1)
                layer.build(input_shape)
        if self.reg_index:
            for i, (index, interval, weight) in enumerate(zip(self.reg_index, self.reg_slice, self.reg_weight)):
                weights = []
                if type(interval) is slice:
                    interval = (interval,)
                for layer in self.layers:
                    if interval is None:
                        weights.append(K.flatten(layer.get_weights()[index]))
                    else:
                        weights.append(K.flatten(layer.get_weights()[index][interval]))
                weights = K.stack(weights)
                self.add_loss(weight * K.sum(K.square(K.dot(weights, K.transpose(weights)) - K.eye(len(self.layers)))))
        super(MultiHead, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.hidden_dim is not None:
            input_shape = input_shape[:-1] + (self.hidden_dim,)
        child_output_shape = self.layers[0].compute_output_shape(input_shape)
        return child_output_shape + (self.layer_num,)

    def compute_mask(self, inputs, mask=None):
        return self.layers[0].compute_mask(inputs, mask)

    def call(self, inputs, training=None, mask=None):
        kwargs = {}
        if keras.utils.generic_utils.has_arg(self.layer.call, 'training'):
            kwargs['training'] = training
        if keras.utils.generic_utils.has_arg(self.layer.call, 'mask') and mask is not None:
            kwargs['mask'] = mask
        if self.hidden_dim is None:
            outputs = [K.expand_dims(layer.call(inputs, **kwargs)) for layer in self.layers]
        else:
            outputs = []
            for i, layer in enumerate(self.layers):
                begin = i * self.hidden_dim
                end = begin + self.hidden_dim
                transformed = K.dot(inputs, self.W[:, begin:end])
                if self.use_bias:
                    transformed += self.b[begin:end]
                outputs.append(K.expand_dims(layer.call(transformed, **kwargs)))
        return K.concatenate(outputs, axis=-1)

    @property
    def trainable_weights(self):
        weights = self._trainable_weights[:]
        for layer in self.layers:
            weights += layer.trainable_weights
        return weights

    @property
    def non_trainable_weights(self):
        weights = self._non_trainable_weights[:]
        for layer in self.layers:
            weights += layer.non_trainable_weights
        return weights

    @property
    def updates(self):
        updates = self._updates
        for layer in self.layers:
            if hasattr(layer, 'updates'):
                updates += layer.updates
        return []

    def get_updates_for(self, inputs=None):
        inner_inputs = inputs
        if inputs is not None:
            uid = keras.utils.generic_utils.object_list_uid(inputs)
            if uid in self._input_map:
                inner_inputs = self._input_map[uid]

        updates = self._updates
        for layer in self.layers:
            layer_updates = layer.get_updates_for(inner_inputs)
            layer_updates += super(MultiHead, self).get_updates_for(inputs)
            updates += layer_updates
        return updates

    @property
    def losses(self):
        losses = self._losses
        for layer in self.layers:
            if hasattr(layer, 'losses'):
                losses += layer.losses
        return losses

    def get_losses_for(self, inputs=None):
        if inputs is None:
            losses = []
            for layer in self.layers:
                losses = layer.get_losses_for(None)
            return losses + super(MultiHead, self).get_losses_for(None)
        return super(MultiHead, self).get_losses_for(inputs)
    
class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention layer.
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 head_num,
                 activation='relu',
                 use_bias=True,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param head_num: Number of heads.
        :param activation: Activations for linear mappings.
        :param use_bias: Whether to use bias term.
        :param kernel_initializer: Initializer for linear mappings.
        :param bias_initializer: Initializer for linear mappings.
        :param kernel_regularizer: Regularizer for linear mappings.
        :param bias_regularizer: Regularizer for linear mappings.
        :param kernel_constraint: Constraints for linear mappings.
        :param bias_constraint: Constraints for linear mappings.
        :param history_only: Whether to only use history in attention layer.
        """

        self.supports_masking = True
        self.head_num = head_num
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.history_only = history_only
        self.Wq, self.Wk, self.Wv, self.Wo = None, None, None, None
        self.bq, self.bk, self.bv, self.bo = None, None, None, None
        super(MultiHeadAttention, self).__init__(**kwargs)

    def get_config(self):
        config = {
            'head_num': int(self.head_num),
            'activation': keras.activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': keras.initializers.serialize(self.kernel_initializer),
            'bias_initializer': keras.initializers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'history_only': self.history_only,
        }
        base_config = super(MultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
            return q[:-1] + (v[-1],)
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        if isinstance(input_mask, list):
            return input_mask[0]
        return input_mask

    def build(self, input_shape):
        if isinstance(input_shape, list):
            q, k, v = input_shape
        else:
            q = k = v = input_shape
        feature_dim = int(v[-1])
        if feature_dim % self.head_num != 0:
            raise IndexError('Invalid head number %d with the given input dim %d' % (self.head_num, feature_dim))
        self.Wq = self.add_weight(
            shape=(int(q[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wq' % self.name,
        )
        if self.use_bias:
            self.bq = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bq' % self.name,
            )
        self.Wk = self.add_weight(
            shape=(int(k[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wk' % self.name,
        )
        if self.use_bias:
            self.bk = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bk' % self.name,
            )
        self.Wv = self.add_weight(
            shape=(int(v[-1]), feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wv' % self.name,
        )
        if self.use_bias:
            self.bv = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bv' % self.name,
            )
        self.Wo = self.add_weight(
            shape=(feature_dim, feature_dim),
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='%s_Wo' % self.name,
        )
        if self.use_bias:
            self.bo = self.add_weight(
                shape=(feature_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='%s_bo' % self.name,
            )
        super(MultiHeadAttention, self).build(input_shape)

    @staticmethod
    def _reshape_to_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        head_dim = feature_dim // head_num
        x = K.reshape(x, (batch_size, seq_len, head_num, head_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size * head_num, seq_len, head_dim))

    @staticmethod
    def _reshape_from_batches(x, head_num):
        input_shape = K.shape(x)
        batch_size, seq_len, feature_dim = input_shape[0], input_shape[1], input_shape[2]
        x = K.reshape(x, (batch_size // head_num, head_num, seq_len, feature_dim))
        x = K.permute_dimensions(x, [0, 2, 1, 3])
        return K.reshape(x, (batch_size // head_num, seq_len, feature_dim * head_num))

    @staticmethod
    def _reshape_mask(mask, head_num):
        if mask is None:
            return mask
        seq_len = K.shape(mask)[1]
        mask = K.expand_dims(mask, axis=1)
        mask = K.tile(mask, [1, head_num, 1])
        return K.reshape(mask, (-1, seq_len))

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            q, k, v = inputs
        else:
            q = k = v = inputs
        if isinstance(mask, list):
            q_mask, k_mask, v_mask = mask
        else:
            q_mask = k_mask = v_mask = mask
        q = K.dot(q, self.Wq)
        k = K.dot(k, self.Wk)
        v = K.dot(v, self.Wv)
        if self.use_bias:
            q += self.bq
            k += self.bk
            v += self.bv
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)
        y = ScaledDotProductAttention(
            history_only=self.history_only,
            name='%s-Attention' % self.name,
        )(
            inputs=[
                self._reshape_to_batches(q, self.head_num),
                self._reshape_to_batches(k, self.head_num),
                self._reshape_to_batches(v, self.head_num),
            ],
            mask=[
                self._reshape_mask(q_mask, self.head_num),
                self._reshape_mask(k_mask, self.head_num),
                self._reshape_mask(v_mask, self.head_num),
            ],
        )
        y = self._reshape_from_batches(y, self.head_num)
        y = K.dot(y, self.Wo)
        if self.use_bias:
            y += self.bo
        if self.activation is not None:
            y = self.activation(y)
        return y
    
class LayerNormalization(keras.layers.Layer):

    def __init__(self,
                 center=True,
                 scale=True,
                 epsilon=None,
                 gamma_initializer='ones',
                 beta_initializer='zeros',
                 gamma_regularizer=None,
                 beta_regularizer=None,
                 gamma_constraint=None,
                 beta_constraint=None,
                 **kwargs):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param center: Add an offset parameter if it is True.
        :param scale: Add a scale parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        :param gamma_initializer: Initializer for the gamma weight.
        :param beta_initializer: Initializer for the beta weight.
        :param gamma_regularizer: Optional regularizer for the gamma weight.
        :param beta_regularizer: Optional regularizer for the beta weight.
        :param gamma_constraint: Optional constraint for the gamma weight.
        :param beta_constraint: Optional constraint for the beta weight.
        :param kwargs:
        """
        super(LayerNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.center = center
        self.scale = scale
        if epsilon is None:
            epsilon = K.epsilon() * K.epsilon()
        self.epsilon = epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.beta_initializer = keras.initializers.get(beta_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.beta_regularizer = keras.regularizers.get(beta_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)
        self.beta_constraint = keras.constraints.get(beta_constraint)
        self.gamma, self.beta = None, None

    def get_config(self):
        config = {
            'center': self.center,
            'scale': self.scale,
            'epsilon': self.epsilon,
            'gamma_initializer': keras.initializers.serialize(self.gamma_initializer),
            'beta_initializer': keras.initializers.serialize(self.beta_initializer),
            'gamma_regularizer': keras.regularizers.serialize(self.gamma_regularizer),
            'beta_regularizer': keras.regularizers.serialize(self.beta_regularizer),
            'gamma_constraint': keras.constraints.serialize(self.gamma_constraint),
            'beta_constraint': keras.constraints.serialize(self.beta_constraint),
        }
        base_config = super(LayerNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

    def compute_mask(self, inputs, input_mask=None):
        return input_mask

    def build(self, input_shape):
        self.input_spec = keras.layers.InputSpec(shape=input_shape)
        shape = input_shape[-1:]
        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint,
                name='gamma',
            )
        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint,
                name='beta',
            )
        super(LayerNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        std = K.sqrt(variance + self.epsilon)
        outputs = (inputs - mean) / std
        if self.scale:
            outputs *= self.gamma
        if self.center:
            outputs += self.beta
        return outputs
    
class rc_Conv1D(tf.keras.layers.Conv1D):

    def compute_output_shape(self, input_shape):
        length = conv_utils.conv_output_length(input_shape[1],
                                               self.kernel_size[0],
                                               padding=self.padding,
                                               stride=self.strides[0])
        return [(1024, int(length), int(self.filters)),
                (1024, int(length), int(self.filters))]

    def call(self, inputs):
        #create a rev-comped kernel.
        #kernel shape is (width, input_channels, filters)
        #Rev comp is along both the length (dim 0) and input channel (dim 1)
        #axes; that is the reason for ::-1, ::-1 in the first and second dims.
        #The rev-comp of channel at index i should be at index i
        
        revcomp_kernel =\
            K.concatenate([self.kernel,
                           self.kernel[::-1,::-1,:]],axis=-1)
        
        if (self.use_bias):
            revcomp_bias = K.concatenate([self.bias,
                                          self.bias], axis=-1)

        outputs = K.conv1d(inputs, revcomp_kernel,
                           strides=self.strides[0],
                           padding=self.padding,
                           data_format=self.data_format,
                           dilation_rate=self.dilation_rate[0])

        if self.use_bias:
            outputs += K.bias_add(outputs,
                                  revcomp_bias,
                                  data_format=self.data_format)

        if (self.activation is not None):
            outputs = self.activation(outputs)

        x_f = outputs[:,:,:int(outputs.get_shape().as_list()[-1]/2)]
        x_rc = outputs[:,:,int(outputs.get_shape().as_list()[-1]/2):]

        return [x_f,x_rc]
    
class ScaledDotProductAttention(keras.layers.Layer):
    r"""The attention layer that takes three inputs representing queries, keys and values.
    \text{Attention}(Q, K, V) = \text{softmax}(\frac{Q K^T}{\sqrt{d_k}}) V
    See: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self,
                 return_attention=False,
                 history_only=False,
                 **kwargs):
        """Initialize the layer.
        :param return_attention: Whether to return attention weights.
        :param history_only: Whether to only use history data.
        :param kwargs: Arguments for parent class.
        """
        super(ScaledDotProductAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.return_attention = return_attention
        self.history_only = history_only

    def get_config(self):
        config = {
            'return_attention': self.return_attention,
            'history_only': self.history_only,
        }
        base_config = super(ScaledDotProductAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            query_shape, key_shape, value_shape = input_shape
        else:
            query_shape = key_shape = value_shape = input_shape
        output_shape = query_shape[:-1] + value_shape[-1:]
        if self.return_attention:
            attention_shape = query_shape[:2] + (key_shape[1],)
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if isinstance(mask, list):
            mask = mask[0]
        if self.return_attention:
            return [mask, None]
        return mask

    def call(self, inputs, mask=None, **kwargs):
        if isinstance(inputs, list):
            query, key, value = inputs
        else:
            query = key = value = inputs
        if isinstance(mask, list):
            mask = mask[1]
        feature_dim = K.shape(query)[-1]
        e = K.batch_dot(query, key, axes=2) / K.sqrt(K.cast(feature_dim, dtype=K.floatx()))
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.history_only:
            query_len, key_len = K.shape(query)[1], K.shape(key)[1]
            indices = K.expand_dims(K.arange(key_len), axis=0)
            upper = K.expand_dims(K.arange(query_len), axis=-1)
            e *= K.expand_dims(K.cast(indices <= upper, K.floatx()), axis=0)
        if mask is not None:
            e *= K.cast(K.expand_dims(mask, axis=-2), K.floatx())
        a = e / (K.sum(e, axis=-1, keepdims=True) + K.epsilon())
        v = K.batch_dot(a, value)
        if self.return_attention:
            return [v, a]
        return v
        
class SeqSelfAttention(keras.layers.Layer):

    ATTENTION_TYPE_ADD = 'additive'
    ATTENTION_TYPE_MUL = 'multiplicative'

    def __init__(self,
                 units=32,
                 attention_width=None,
                 attention_type=ATTENTION_TYPE_ADD,
                 return_attention=False,
                 history_only=False,
                 kernel_initializer='glorot_normal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 use_additive_bias=True,
                 use_attention_bias=True,
                 attention_activation=None,
                 attention_regularizer_weight=0.0,
                 **kwargs):
        """Layer initialization.
        For additive attention, see: https://arxiv.org/pdf/1806.01264.pdf
        :param units: The dimension of the vectors that used to calculate the attention weights.
        :param attention_width: The width of local attention.
        :param attention_type: 'additive' or 'multiplicative'.
        :param return_attention: Whether to return the attention weights for visualization.
        :param history_only: Only use historical pieces of data.
        :param kernel_initializer: The initializer for weight matrices.
        :param bias_initializer: The initializer for biases.
        :param kernel_regularizer: The regularization for weight matrices.
        :param bias_regularizer: The regularization for biases.
        :param kernel_constraint: The constraint for weight matrices.
        :param bias_constraint: The constraint for biases.
        :param use_additive_bias: Whether to use bias while calculating the relevance of inputs features
                                  in additive mode.
        :param use_attention_bias: Whether to use bias while calculating the weights of attention.
        :param attention_activation: The activation used for calculating the weights of attention.
        :param attention_regularizer_weight: The weights of attention regularizer.
        :param kwargs: Parameters for parent class.
        """
        super(SeqSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.units = units
        self.attention_width = attention_width
        self.attention_type = attention_type
        self.return_attention = return_attention
        self.history_only = history_only
        if history_only and attention_width is None:
            self.attention_width = int(1e9)

        self.use_additive_bias = use_additive_bias
        self.use_attention_bias = use_attention_bias
        self.kernel_initializer = keras.initializers.get(kernel_initializer)
        self.bias_initializer = keras.initializers.get(bias_initializer)
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = keras.regularizers.get(bias_regularizer)
        self.kernel_constraint = keras.constraints.get(kernel_constraint)
        self.bias_constraint = keras.constraints.get(bias_constraint)
        self.attention_activation = keras.activations.get(attention_activation)
        self.attention_regularizer_weight = attention_regularizer_weight
        self._backend = keras.backend.backend()

        if attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self.Wx, self.Wt, self.bh = None, None, None
            self.Wa, self.ba = None, None
        elif attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self.Wa, self.ba = None, None
        else:
            raise NotImplementedError('No implementation for attention type : ' + attention_type)

    def get_config(self):
        config = {
            'units': int(self.units),
            'attention_width': self.attention_width,
            'attention_type': self.attention_type,
            'return_attention': self.return_attention,
            'history_only': self.history_only,
            'use_additive_bias': self.use_additive_bias,
            'use_attention_bias': self.use_attention_bias,
            'kernel_initializer': keras.regularizers.serialize(self.kernel_initializer),
            'bias_initializer': keras.regularizers.serialize(self.bias_initializer),
            'kernel_regularizer': keras.regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': keras.regularizers.serialize(self.bias_regularizer),
            'kernel_constraint': keras.constraints.serialize(self.kernel_constraint),
            'bias_constraint': keras.constraints.serialize(self.bias_constraint),
            'attention_activation': keras.activations.serialize(self.attention_activation),
            'attention_regularizer_weight': self.attention_regularizer_weight,
        }
        base_config = super(SeqSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            self._build_additive_attention(input_shape)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            self._build_multiplicative_attention(input_shape)
        super(SeqSelfAttention, self).build(input_shape)

    def _build_additive_attention(self, input_shape):
        feature_dim = int(input_shape[2])

        self.Wt = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wt'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        self.Wx = self.add_weight(shape=(feature_dim, self.units),
                                  name='{}_Add_Wx'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_additive_bias:
            self.bh = self.add_weight(shape=(self.units,),
                                      name='{}_Add_bh'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

        self.Wa = self.add_weight(shape=(self.units, 1),
                                  name='{}_Add_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Add_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def _build_multiplicative_attention(self, input_shape):
        feature_dim = input_shape[2]

        self.Wa = self.add_weight(shape=(feature_dim, feature_dim),
                                  name='{}_Mul_Wa'.format(self.name),
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
        if self.use_attention_bias:
            self.ba = self.add_weight(shape=(1,),
                                      name='{}_Mul_ba'.format(self.name),
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer,
                                      constraint=self.bias_constraint)

    def call(self, inputs, mask=None, **kwargs):
        input_len = K.shape(inputs)[1]

        if self.attention_type == SeqSelfAttention.ATTENTION_TYPE_ADD:
            e = self._call_additive_emission(inputs)
        elif self.attention_type == SeqSelfAttention.ATTENTION_TYPE_MUL:
            e = self._call_multiplicative_emission(inputs)

        if self.attention_activation is not None:
            e = self.attention_activation(e)
        e = K.exp(e - K.max(e, axis=-1, keepdims=True))
        if self.attention_width is not None:
            if self.history_only:
                lower = K.arange(input_len) - (self.attention_width - 1)
            else:
                lower = K.arange(input_len) - self.attention_width // 2
            lower = K.expand_dims(lower, axis=-1)
            upper = lower + self.attention_width
            indices = K.expand_dims(K.arange(input_len), axis=0)
            e = e * K.cast(lower <= indices, K.floatx()) * K.cast(indices < upper, K.floatx())
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            mask = K.expand_dims(mask)
            e = K.permute_dimensions(K.permute_dimensions(e * mask, (0, 2, 1)) * mask, (0, 2, 1))

        # a_{t} = \text{softmax}(e_t)
        s = K.sum(e, axis=-1, keepdims=True)
        a = e / (s + K.epsilon())

        # l_t = \sum_{t'} a_{t, t'} x_{t'}
        v = K.batch_dot(a, inputs)
        if self.attention_regularizer_weight > 0.0:
            self.add_loss(self._attention_regularizer(a))

        if self.return_attention:
            return [v, a]
        return v

    def _call_additive_emission(self, inputs):
        input_shape = K.shape(inputs)
        batch_size, input_len = input_shape[0], input_shape[1]

        # h_{t, t'} = \tanh(x_t^T W_t + x_{t'}^T W_x + b_h)
        q = K.expand_dims(K.dot(inputs, self.Wt), 2)
        k = K.expand_dims(K.dot(inputs, self.Wx), 1)
        if self.use_additive_bias:
            h = K.tanh(q + k + self.bh)
        else:
            h = K.tanh(q + k)

        # e_{t, t'} = W_a h_{t, t'} + b_a
        if self.use_attention_bias:
            e = K.reshape(K.dot(h, self.Wa) + self.ba, (batch_size, input_len, input_len))
        else:
            e = K.reshape(K.dot(h, self.Wa), (batch_size, input_len, input_len))
        return e

    def _call_multiplicative_emission(self, inputs):
        # e_{t, t'} = x_t^T W_a x_{t'} + b_a
        e = K.batch_dot(K.dot(inputs, self.Wa), K.permute_dimensions(inputs, (0, 2, 1)))
        if self.use_attention_bias:
            e += self.ba[0]
        return e

    def compute_output_shape(self, input_shape):
        output_shape = input_shape
        if self.return_attention:
            attention_shape = (input_shape[0], output_shape[1], input_shape[1])
            return [output_shape, attention_shape]
        return output_shape

    def compute_mask(self, inputs, mask=None):
        if self.return_attention:
            return [mask, None]
        return mask

    def _attention_regularizer(self, attention):
        batch_size = K.cast(K.shape(attention)[0], K.floatx())
        input_len = K.shape(attention)[-1]
        indices = K.tile(K.expand_dims(K.arange(input_len), axis=0), [input_len, 1])
        diagonal = K.expand_dims(K.arange(input_len), axis=-1)
        eye = K.cast(K.equal(indices, diagonal), K.floatx())
        return self.attention_regularizer_weight * K.sum(K.square(K.batch_dot(
            attention,
            K.permute_dimensions(attention, (0, 2, 1))) - eye)) / batch_size

    @staticmethod
    def get_custom_objects():
        return {'SeqSelfAttention': SeqSelfAttention}
    
class SeqWeightedAttention(keras.layers.Layer):
    r"""Y = \text{softmax}(XW + b) X
    See: https://arxiv.org/pdf/1708.00524.pdf
    """

    def __init__(self, use_bias=True, return_attention=False, **kwargs):
        super(SeqWeightedAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.use_bias = use_bias
        self.return_attention = return_attention
        self.W, self.b = None, None

    def get_config(self):
        config = {
            'use_bias': self.use_bias,
            'return_attention': self.return_attention,
        }
        base_config = super(SeqWeightedAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=keras.initializers.get('uniform'))
        if self.use_bias:
            self.b = self.add_weight(shape=(1,),
                                     name='{}_b'.format(self.name),
                                     initializer=keras.initializers.get('zeros'))
        super(SeqWeightedAttention, self).build(input_shape)

    def call(self, x, mask=None):
        logits = K.dot(x, self.W)
        if self.use_bias:
            logits += self.b
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        weighted_input = x * K.expand_dims(att_weights)
        result = K.sum(weighted_input, axis=1)
        if self.return_attention:
            return [result, att_weights]
        return result

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return input_shape[0], output_len

    def compute_mask(self, _, input_mask=None):
        if self.return_attention:
            return [None, None]
        return None

    @staticmethod
    def get_custom_objects():
        return {'SeqWeightedAttention': SeqWeightedAttention}
    
def r_square(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def correlation_coefficient(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return K.square(r)
