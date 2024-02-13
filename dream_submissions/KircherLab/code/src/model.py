from tensorflow.keras import Model, activations, layers
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling1D, add
from tensorflow.keras.regularizers import l2

def standard(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(250, kernel_size=7, strides=1, activation='relu', name="conv1")(inputs) # 250 7 relu
    layer = BatchNormalization()(layer)
    layer = Conv1D(250, 8, strides=1, activation='softmax', name="conv2")(layer) # 250 8 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = Dropout(0.1)(layer)
    layer = Conv1D(250, 3, strides=1, activation='softmax', name="conv3")(layer) # 250 3 softmax
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 2, strides=1, activation='softmax', name="conv4")(layer) # 100 3 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(300, activation='sigmoid')(layer) # 300
    layer = Dropout(0.3)(layer)
    layer = Dense(300, activation='sigmoid')(layer) # 300
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)


def simplified(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(100, kernel_size=8, strides=1, activation='softmax', name="conv1")(inputs) # 250 7 relu
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 8, strides=1, activation='softmax', name="conv2")(layer) # 250 8 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=2, strides=None, name="maxpool1")(layer)
    layer = Dropout(0.1)(layer)
    layer = Conv1D(100, 3, strides=1, activation='softmax', name="conv3")(layer) # 250 3 softmax
    layer = BatchNormalization()(layer)
    layer = Conv1D(100, 3, strides=1, activation='softmax', name="conv4")(layer) # 100 3 softmax
    layer = BatchNormalization()(layer)
    layer = MaxPooling1D(pool_size=1, strides=None, name="maxpool2")(layer)
    layer = Dropout(0.1)(layer)
    layer = Flatten()(layer)
    layer = Dense(110, activation='sigmoid')(layer) # 300
    layer = Dropout(0.3)(layer)
    layer = Dense(110, activation='sigmoid')(layer) # 300
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)

## ResNet like models

class ResidualUnit1D_BN(layers.Layer):
    def __init__(self, filters, strides=1, activation="relu", first_layer_of_first_block=False, **kwargs):
        super().__init__(**kwargs)
        self.activation = activations.get(activation)
        self.skip_layers = []
        self.filters = filters
        self.strides = strides
        self.first_layer_of_first_block = first_layer_of_first_block
        
        if not first_layer_of_first_block:
            # All other blocks
            self.main_layers = [
                BatchNormalization(),
                self.activation    ]
        else:
            #FIRST BLOCK:
            ## We need to add a Conv skip connection to get the same number of filters, even though strides are 1
            self.skip_layers = [
                Conv1D(filters*4, 1, strides=strides,
                                    padding="same", use_bias=False),
                BatchNormalization()]
            ## don't repeat bn->relu since we just did bn->relu->maxpool
            self.main_layers = []
        
        self.main_layers.extend([
            Conv1D(filters, 1, strides=strides,
                                padding="same", use_bias=False,
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            self.activation,
            Conv1D(filters, 3, strides=1,
                                padding="same", use_bias=False,
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(0.0001)),
            BatchNormalization(),
            self.activation,
            Conv1D(filters*4, 1, strides=1,
                                padding="same", use_bias=False,
                                kernel_initializer="he_normal",
                                kernel_regularizer=l2(0.0001)),])

        
        if strides > 1:
        # 1 X 1 conv if stride > 1. Else identity.
            self.skip_layers = [
                Conv1D(filters*4, 1, strides=strides,
                                    padding="same", use_bias=False),
                BatchNormalization()]
    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)
        return add([Z, skip_Z])
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'filters':self.filters,
            'strides':self.strides,
            'activation':self.activation,
            'first_layer_of_first_block':self.first_layer_of_first_block,
        })
        return config


def ResNet1D_Classification(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(128, kernel_size=7, strides=2, activation='relu', name="conv1",use_bias=False, padding="same")(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = MaxPooling1D(pool_size=2, strides=2, padding="same", name="maxpool1")(layer)
    prev_filters = 128
    for i, filters in enumerate([128] * 3 + [256] * 4 + [512] * 6 + [1024] * 3):
        strides = 1 if filters == prev_filters else 2
        layer = ResidualUnit1D_BN(filters, strides=strides, first_layer_of_first_block=(i == 0))(layer)
        prev_filters = filters
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Flatten()(layer)
    predictions = Dense(output_shape[1], activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)


def ResNet1D_Regression(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(128, kernel_size=7, strides=2, activation='relu', name="conv1",use_bias=False, padding="same")(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = MaxPooling1D(pool_size=2, strides=2, padding="same", name="maxpool1")(layer)
    prev_filters = 128
    for i, filters in enumerate([128] * 3 + [256] * 4 + [512] * 6 + [1024] * 3):
        strides = 1 if filters == prev_filters else 2
        layer = ResidualUnit1D_BN(filters, strides=strides, first_layer_of_first_block=(i == 0))(layer)
        prev_filters = filters
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Flatten()(layer)
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)

def ResNet1D_Classification_medium(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(256, kernel_size=7, strides=2, activation='relu', name="conv1",use_bias=False, padding="same")(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = MaxPooling1D(pool_size=2, strides=2, padding="same", name="maxpool1")(layer)
    prev_filters = 256
    for i, filters in enumerate([256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3):
        strides = 1 if filters == prev_filters else 2
        layer = ResidualUnit1D_BN(filters, strides=strides, first_layer_of_first_block=(i == 0))(layer)
        prev_filters = filters
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Flatten()(layer)
    predictions = Dense(output_shape[1], activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)


def ResNet1D_Regression_medium(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(256, kernel_size=7, strides=2, activation='relu', name="conv1",use_bias=False, padding="same")(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = MaxPooling1D(pool_size=2, strides=2, padding="same", name="maxpool1")(layer)
    prev_filters = 256
    for i, filters in enumerate([256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3):
        strides = 1 if filters == prev_filters else 2
        layer = ResidualUnit1D_BN(filters, strides=strides, first_layer_of_first_block=(i == 0))(layer)
        prev_filters = filters
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Flatten()(layer)
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)

def ResNet1D_Classification_dropout(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(256, kernel_size=7, strides=2, activation='relu', name="conv1",use_bias=False, padding="same")(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = MaxPooling1D(pool_size=2, strides=2, padding="same", name="maxpool1")(layer)
    prev_filters = 256
    for i, filters in enumerate([256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3):
        strides = 1 if filters == prev_filters else 2
        layer = ResidualUnit1D_BN(filters, strides=strides, first_layer_of_first_block=(i == 0))(layer)
        prev_filters = filters
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = Dropout(0.3)(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Flatten()(layer)
    predictions = Dense(output_shape[1], activation='softmax')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)


def ResNet1D_Regression_dropout(input_shape, output_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]), name="input")
    layer = Conv1D(256, kernel_size=7, strides=2, activation='relu', name="conv1",use_bias=False, padding="same")(inputs)
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = MaxPooling1D(pool_size=2, strides=2, padding="same", name="maxpool1")(layer)
    prev_filters = 256
    for i, filters in enumerate([256] * 3 + [512] * 4 + [1024] * 6 + [2048] * 3):
        strides = 1 if filters == prev_filters else 2
        layer = ResidualUnit1D_BN(filters, strides=strides, first_layer_of_first_block=(i == 0))(layer)
        prev_filters = filters
    layer = BatchNormalization()(layer)
    layer = Activation("relu")(layer)
    layer = Dropout(0.3)(layer)
    layer = GlobalAveragePooling1D()(layer)
    layer = Flatten()(layer)
    predictions = Dense(output_shape[1], activation='linear')(layer)
    model = Model(inputs=inputs, outputs=predictions)
    return(model)