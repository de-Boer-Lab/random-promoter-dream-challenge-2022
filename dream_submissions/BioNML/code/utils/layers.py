import tensorflow as tf
import tensorflow_addons as tfa

tfa.options.TF_ADDONS_PY_OPS = True


@tf.keras.utils.register_keras_serializable()
class Token(tf.keras.layers.Layer):
    """Append a class token to an input layer."""

    def build(self, input_shape):
        cls_init = tf.zeros_initializer()
        self.hidden_size = input_shape[-1]
        self.cls = tf.Variable(
            name="repr",
            initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
            trainable=True,
        )

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        cls_broadcasted = tf.cast(
            tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
            dtype=inputs.dtype,
        )
        return tf.concat([cls_broadcasted, inputs], 1)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class AddPositionEmbs(tf.keras.layers.Layer):
    """Adds (optionally learned) positional embeddings to the inputs."""

    def build(self, input_shape):
        assert (
            len(input_shape) == 3
        ), f"Number of dimensions should be 3, got {len(input_shape)}"
        self.pe = tf.Variable(
            name="pos_embedding",
            initial_value=tf.random_normal_initializer(stddev=0.06)(
                shape=(1, input_shape[1], input_shape[2])
            ),
            dtype="float32",
            trainable=True,
        )

    def call(self, inputs):
        return inputs + tf.cast(self.pe, dtype=inputs.dtype)

    def get_config(self):
        config = super().get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                "embedding dimension = {} should be divisible by number of heads = {}".format(
                    hidden_size, num_heads
                )
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(
            hidden_size, name="query"
        )  # potentially sigmoid activation and with regularization
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update({"num_heads": self.num_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-instance-attributes
@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# pylint: disable=too-many-instance-attributes
@tf.keras.utils.register_keras_serializable()
class TransformerBlock_swiglu(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(self, *args, num_heads, mlp_dim, dropout, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout

    def build(self, input_shape):
        self.att = MultiHeadSelfAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
        )
        self.mlpblock = tf.keras.Sequential(
            [
                SwiGlu(name=f"{self.name}/swiglu"),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        x = self.layernorm1(inputs)
        x, weights = self.att(x)
        x = self.dropout_layer(x, training=training)
        x = x + inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class MultiHeadCrossAttention(tf.keras.layers.Layer):
    def __init__(self, *args, num_heads, query_dense_activation="linear", **kwargs):
        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.query_dense_activation = query_dense_activation

    def build(self, input_shape):
        hidden_size = input_shape[1][-1]  # query, key/value
        num_heads = self.num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(
                "embedding dimension = {} should be divisible by number of heads = {}".format(
                    hidden_size, num_heads
                )
            )
        self.hidden_size = hidden_size
        self.projection_dim = hidden_size // num_heads
        self.query_dense = tf.keras.layers.Dense(
            hidden_size, name="query", activation=self.query_dense_activation
        )  # potentially sigmoid activation and with regularization
        self.key_dense = tf.keras.layers.Dense(hidden_size, name="key")
        self.value_dense = tf.keras.layers.Dense(hidden_size, name="value")
        self.combine_heads = tf.keras.layers.Dense(hidden_size, name="out")

    # pylint: disable=no-self-use
    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], score.dtype)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        [query_inputs, key_value_inputs] = inputs
        batch_size = tf.shape(query_inputs)[0]
        query = self.query_dense(query_inputs)
        key = self.key_dense(key_value_inputs)
        value = self.value_dense(key_value_inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.hidden_size))
        output = self.combine_heads(concat_attention)
        return output, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "query_dense_activation": self.query_dense_activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class TransformerCrossAttBlock(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(
        self,
        *args,
        num_heads,
        mlp_dim,
        dropout,
        query_dense_activation="linear",
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.query_dense_activation = query_dense_activation

    def build(self, input_shape):
        self.att = MultiHeadCrossAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
            query_dense_activation=self.query_dense_activation,
        )
        self.mlpblock = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    self.mlp_dim,
                    activation="linear",
                    name=f"{self.name}/Dense_0",
                ),
                tf.keras.layers.Lambda(
                    lambda x: tf.keras.activations.gelu(x, approximate=False)
                )
                if hasattr(tf.keras.activations, "gelu")
                else tf.keras.layers.Lambda(
                    lambda x: tfa.activations.gelu(x, approximate=False)
                ),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[0][-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        [query_inputs, key_value_inputs] = inputs
        x = self.layernorm1(key_value_inputs)
        x_query_inputs = self.layernorm1(query_inputs)
        x, weights = self.att([x_query_inputs, x])
        x = self.dropout_layer(x, training=training)
        x = x + query_inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
                "query_dense_activation": self.query_dense_activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class TransformerCrossAttBlock_swiglu(tf.keras.layers.Layer):
    """Implements a Transformer block."""

    def __init__(
        self,
        *args,
        num_heads,
        mlp_dim,
        dropout,
        query_dense_activation="linear",
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.query_dense_activation = query_dense_activation

    def build(self, input_shape):
        self.att = MultiHeadCrossAttention(
            num_heads=self.num_heads,
            name="MultiHeadDotProductAttention_1",
            query_dense_activation=self.query_dense_activation,
        )
        self.mlpblock = tf.keras.Sequential(
            [
                SwiGlu(name=f"{self.name}/swiglu"),
                tf.keras.layers.Dropout(self.dropout),
                tf.keras.layers.Dense(input_shape[0][-1], name=f"{self.name}/Dense_1"),
                tf.keras.layers.Dropout(self.dropout),
            ],
            name="MlpBlock_3",
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_0"
        )
        self.layernorm2 = tf.keras.layers.LayerNormalization(
            epsilon=1e-6, name="LayerNorm_2"
        )
        self.dropout_layer = tf.keras.layers.Dropout(self.dropout)

    def call(self, inputs, training):
        [query_inputs, key_value_inputs] = inputs
        x = self.layernorm1(key_value_inputs)
        x_query_inputs = self.layernorm1(query_inputs)
        x, weights = self.att([x_query_inputs, x])
        x = self.dropout_layer(x, training=training)
        x = x + query_inputs
        y = self.layernorm2(x)
        y = self.mlpblock(y)
        return x + y, weights

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "mlp_dim": self.mlp_dim,
                "dropout": self.dropout,
                "query_dense_activation": self.query_dense_activation,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DenseResidualBlock(tf.keras.layers.Layer):
    def __init__(self, *args, activation, bottleneck_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim

    def build(self, input_shape):
        self.bn1 = tf.keras.layers.BatchNormalization(name="{}/bn1".format(self.name))
        self.act1 = tf.keras.layers.Activation(
            self.activation, name="{}/act1".format(self.name)
        )
        self.dense1 = tf.keras.layers.Dense(
            self.bottleneck_dim, name="{}/dense1".format(self.name)
        )
        self.bn2 = tf.keras.layers.BatchNormalization(name="{}/bn2".format(self.name))
        self.act2 = tf.keras.layers.Activation(
            self.activation, name="{}/act2".format(self.name)
        )
        self.dense2 = tf.keras.layers.Dense(
            input_shape[-1], name="{}/dense2".format(self.name)
        )

    def call(self, inputs):

        output = self.bn1(inputs)
        output = self.act1(output)
        output = self.dense1(output)
        output = self.bn2(output)
        output = self.act2(output)
        output = self.dense2(output)
        output = tf.keras.layers.Add(name="{}/add".format(self.name))([inputs, output])

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": tf.keras.activations.serialize(self.activation),
                "bottleneck_dim": self.bottleneck_dim,
                # "dropout": self.dropout
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class DenseResidualBlock_swish(tf.keras.layers.Layer):
    def __init__(self, *args, activation, bottleneck_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim

    def build(self, input_shape):
        self.swish1 = Swish(name="{}/swish1".format(self.name))
        self.swish2 = Swish(name="{}/swish2".format(self.name))
        self.bn1 = tf.keras.layers.BatchNormalization(name="{}/bn1".format(self.name))
        self.act1 = tf.keras.layers.Activation(
            self.activation, name="{}/act1".format(self.name)
        )
        self.dense1 = tf.keras.layers.Dense(
            self.bottleneck_dim, name="{}/dense1".format(self.name)
        )
        self.bn2 = tf.keras.layers.BatchNormalization(name="{}/bn2".format(self.name))
        self.act2 = tf.keras.layers.Activation(
            self.activation, name="{}/act2".format(self.name)
        )
        self.dense2 = tf.keras.layers.Dense(
            input_shape[-1], name="{}/dense2".format(self.name)
        )

    def call(self, inputs):

        output = self.bn1(inputs)
        output = self.act1(output)
        output = self.swish1(output)
        output = self.dense1(output)
        output = self.bn2(output)
        output = self.act2(output)
        output = self.swish2(output)
        output = self.dense2(output)
        output = tf.keras.layers.Add(name="{}/add".format(self.name))([inputs, output])

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": tf.keras.activations.serialize(self.activation),
                "bottleneck_dim": self.bottleneck_dim,
                # "dropout": self.dropout
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ConvNonlinearBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        *args,
        activation,
        kernel_length=[1, 20],
        bottleneck_dim,
        output_dim,
        kernel_first,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.kernel_length = kernel_length
        self.kernel_first = kernel_first

    def build(self, input_shape):
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.act1 = tf.keras.layers.Activation(self.activation, name="act1")
        self.conv1 = tf.keras.layers.Conv1D(
            self.bottleneck_dim,
            kernel_size=self.kernel_length[0],
            padding="same",
            name="conv1",
        )
        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
        self.act2 = tf.keras.layers.Activation(self.activation, name="act2")
        self.conv2 = tf.keras.layers.Conv1D(
            self.output_dim,
            kernel_size=self.kernel_length[1],
            padding="same",
            name="conv2",
        )

    def call(self, inputs, training):
        if self.kernel_first:
            output = self.conv1(inputs)
            output = self.bn1(output, training=training)
            output = self.act1(output)
            output = self.conv2(output)
            output = self.bn2(output, training=training)
            output = self.act2(output)

        else:
            output = self.bn1(inputs, training=training)
            output = self.act1(output)
            output = self.conv1(output)
            output = self.bn2(output, training=training)
            output = self.act2(output)
            output = self.conv2(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": self.activation,
                "kernel_length": self.kernel_length,
                "bottleneck_dim": self.bottleneck_dim,
                "output_dim": self.output_dim,
                "kernel_first": self.kernel_first,
                # "dropout": self.dropout
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class ConvNonlinearBlock_swish(tf.keras.layers.Layer):
    def __init__(
        self,
        *args,
        activation,
        kernel_length=[1, 20],
        bottleneck_dim,
        output_dim,
        kernel_first,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.activation = activation
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.kernel_length = kernel_length
        self.kernel_first = kernel_first

    def build(self, input_shape):
        self.swish1 = Swish()
        self.bn1 = tf.keras.layers.BatchNormalization(name="bn1")
        self.act1 = tf.keras.layers.Activation(self.activation, name="act1")
        self.conv1 = tf.keras.layers.Conv1D(
            self.bottleneck_dim,
            kernel_size=self.kernel_length[0],
            padding="same",
            name="conv1",
        )
        self.swish2 = Swish()
        self.bn2 = tf.keras.layers.BatchNormalization(name="bn2")
        self.act2 = tf.keras.layers.Activation(self.activation, name="act2")
        self.conv2 = tf.keras.layers.Conv1D(
            self.output_dim,
            kernel_size=self.kernel_length[1],
            padding="same",
            name="conv2",
        )

    def call(self, inputs, training):
        if self.kernel_first:
            output = self.conv1(inputs)
            output = self.bn1(output, training=training)
            output = self.act1(output)
            output = self.swish1(output)
            output = self.conv2(output)
            output = self.bn2(output, training=training)
            output = self.act2(output)
            output = self.swish2(output)

        else:
            output = self.bn1(inputs, training=training)
            output = self.act1(output)
            output = self.swish1(output)
            output = self.conv1(output)
            output = self.bn2(output, training=training)
            output = self.act2(output)
            output = self.swish2(output)
            output = self.conv2(output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": self.activation,
                "kernel_length": self.kernel_length,
                "bottleneck_dim": self.bottleneck_dim,
                "output_dim": self.output_dim,
                "kernel_first": self.kernel_first,
                # "dropout": self.dropout
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class InceptionBlock_type1(tf.keras.layers.Layer):
    """Main dense branch for dimension reduction with residual-like Conv side branches"""

    def __init__(
        self,
        *args,
        activation,
        combine_type,
        conv_bottleneck_dims,
        kernel_first,
        conv_kernel_length=[1, 20],
        bottleneck_dim,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.activation = activation
        self.combine_type = combine_type
        self.conv_bottleneck_dims = conv_bottleneck_dims
        self.conv_side_branch_number = len(self.conv_bottleneck_dims)
        self.kernel_first = kernel_first
        self.conv_kernel_length = conv_kernel_length
        self.bottleneck_dim = bottleneck_dim

    def build(self, input_shape):
        self.bn1 = tf.keras.layers.BatchNormalization(name="dense/bn")
        self.act1 = tf.keras.layers.Activation(self.activation, name="dense/act")
        self.dense1 = tf.keras.layers.Dense(self.bottleneck_dim, name="dense/dense")
        self.side_branches = []
        for idx, i in enumerate(self.conv_bottleneck_dims):
            if self.combine_type == "Add":
                self.side_branches.append(
                    ConvNonlinearBlock(
                        activation=self.activation,
                        kernel_length=self.conv_kernel_length,
                        bottleneck_dim=i,
                        output_dim=self.bottleneck_dim,
                        kernel_first=self.kernel_first,
                        name="conv_side_branch_{}".format(idx),
                    )
                )

            if self.combine_type == "Concat":
                self.side_branches.append(
                    ConvNonlinearBlock(
                        activation=self.activation,
                        kernel_length=self.conv_kernel_length,
                        bottleneck_dim=i,
                        output_dim=i,
                        kernel_first=self.kernel_first,
                        name="conv_side_branch_{}".format(idx),
                    )
                )

    def call(self, inputs, training):
        self.outputs = []

        if self.kernel_first:
            output = self.dense1(inputs)
            output = self.bn1(output, training=training)
            output = self.act1(output)
        else:
            output = self.bn1(inputs, training=training)
            output = self.act1(output)
            output = self.dense1(output)

        self.outputs.append(output)

        for _side_branch in self.side_branches:
            output = _side_branch(inputs, training=training)
            self.outputs.append(output)

        if self.combine_type == "Add":
            output = tf.keras.layers.Add(name="add")(self.outputs)
        if self.combine_type == "Concat":
            output = tf.keras.layers.Concatenate(axis=-1, name="concat")(self.outputs)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": tf.keras.activations.serialize(self.activation),
                "bottleneck_dim": self.bottleneck_dim,
                "combine_type": self.combine_type,
                "conv_bottleneck_dims": self.conv_bottleneck_dims,
                "kernel_first": self.kernel_first,
                "conv_kernel_length": self.conv_kernel_length
                # "dropout": self.dropout
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class InceptionBlock_type1_swish(tf.keras.layers.Layer):
    """Main dense branch for dimension reduction with residual-like Conv side branches"""

    def __init__(
        self,
        *args,
        activation,
        combine_type,
        conv_bottleneck_dims,
        kernel_first,
        conv_kernel_length=[1, 20],
        bottleneck_dim,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.activation = activation
        self.combine_type = combine_type
        self.conv_bottleneck_dims = conv_bottleneck_dims
        self.conv_side_branch_number = len(self.conv_bottleneck_dims)
        self.kernel_first = kernel_first
        self.conv_kernel_length = conv_kernel_length
        self.bottleneck_dim = bottleneck_dim

    def build(self, input_shape):
        self.swish = Swish()
        self.bn1 = tf.keras.layers.BatchNormalization(name="dense/bn")
        self.act1 = tf.keras.layers.Activation(self.activation, name="dense/act")
        self.dense1 = tf.keras.layers.Dense(self.bottleneck_dim, name="dense/dense")
        self.side_branches = []
        for idx, i in enumerate(self.conv_bottleneck_dims):
            if self.combine_type == "Add":
                self.side_branches.append(
                    ConvNonlinearBlock_swish(
                        activation=self.activation,
                        kernel_length=self.conv_kernel_length,
                        bottleneck_dim=i,
                        output_dim=self.bottleneck_dim,
                        kernel_first=self.kernel_first,
                        name="conv_side_branch_{}".format(idx),
                    )
                )

            if self.combine_type == "Concat":
                self.side_branches.append(
                    ConvNonlinearBlock_swish(
                        activation=self.activation,
                        kernel_length=self.conv_kernel_length,
                        bottleneck_dim=i,
                        output_dim=i,
                        kernel_first=self.kernel_first,
                        name="conv_side_branch_{}".format(idx),
                    )
                )

    def call(self, inputs, training):
        self.outputs = []

        if self.kernel_first:
            output = self.dense1(inputs)
            output = self.bn1(output, training=training)
            output = self.act1(output)
            output = self.swish(output)
        else:
            output = self.bn1(inputs, training=training)
            output = self.act1(output)
            output = self.swish(output)
            output = self.dense1(output)

        self.outputs.append(output)

        for _side_branch in self.side_branches:
            output = _side_branch(inputs, training=training)
            self.outputs.append(output)

        if self.combine_type == "Add":
            output = tf.keras.layers.Add(name="add")(self.outputs)
        if self.combine_type == "Concat":
            output = tf.keras.layers.Concatenate(axis=-1, name="concat")(self.outputs)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "activation": tf.keras.activations.serialize(self.activation),
                "bottleneck_dim": self.bottleneck_dim,
                "combine_type": self.combine_type,
                "conv_bottleneck_dims": self.conv_bottleneck_dims,
                "kernel_first": self.kernel_first,
                "conv_kernel_length": self.conv_kernel_length
                # "dropout": self.dropout
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable()
class Swish(tf.keras.layers.Layer):
    def __init__(self, beta=1.0, *args, **kwargs):
        super(Swish, self).__init__(*args, **kwargs)
        self._beta = beta

    def build(self, input_shape):
        self.beta = tf.Variable(
            initial_value=tf.constant(self._beta, dtype="float32"), trainable=True
        )

    def call(self, inputs):
        return inputs * tf.sigmoid(self.beta * inputs)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self._beta,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable()
class SwiGlu(tf.keras.layers.Layer):
    def __init__(self, beta=1.0, *args, **kwargs):
        super(SwiGlu, self).__init__(*args, **kwargs)
        self.beta = beta

    def build(self, input_shape):
        self.swish = Swish(beta=self.beta)
        self.W = tf.keras.layers.Dense(
            units=input_shape[-1],
            use_bias=True,
            bias_initializer="glorot_uniform",
            name=f"{self.name}/W_c",
        )
        self.V = tf.keras.layers.Dense(
            units=input_shape[-1],
            use_bias=True,
            bias_initializer="glorot_uniform",
            name=f"{self.name}/V_c",
        )

    def call(self, inputs):
        return self.V(inputs) * self.swish(self.W(inputs))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "beta": self._beta,
            }
        )
        return config
