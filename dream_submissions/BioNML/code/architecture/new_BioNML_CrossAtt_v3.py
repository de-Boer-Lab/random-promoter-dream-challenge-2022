import tensorflow as tf
import sys

sys.path.append("../utils")
from utils import layers


def build(
    model_name: str = "new_BioNML_CrossAtt_v2",
    sequence_length: int = 150,
    motif_size: int = 512,
    kmer_size: int = 1024,  # use 5mer
    motif_length_max: int = 30,
    kmer_length_max: int = 5,
    motif_scanning_trainable: bool = True,
    kmer_scanning_trainable: bool = False,
    motif_scanning_activation: str = "relu",
    kmer_scanning_activation: str = "relu",
    embedding_batch_norm: bool = True,
    use_kmer_embedding: bool = True,
    latent_dim: int = 512,
    embedding_activation: str = "relu",
    latent_activation: str = "gelu",
    use_repr_token: bool = True,
    use_pos_embedding: bool = True,
    encoder_layer_number: int = 3,
    num_heads: int = 8,
    mlp_dim: int = 256,
    dropout: float = 0.1,
    dropout_transformer: float = 0.1,
    representation_dim: int = 256,
    output_activations: list = ["linear", "linear"],
    query_dense_activation: str = "linear",
    pred_repr_kernel_l2: float = 0.0,
    pred_repr_activity_l2: float = 0.0,
    latent_OC_dim: int = 64,
    fwd_rc_combine_type: str = "Add",
    wide_embedding_to_latent_repr: bool = False,
    latent_inception_block_combine_type: str = "Add",
    repr_residual_expansion: bool = False,
    repr_residual_expansion_activation="linear",
    inception_block_activation="linear",
):
    with tf.name_scope("Sequence"):
        x = tf.keras.layers.Input(shape=(sequence_length, 4), name="Sequence/fwd")
    x_rc = tf.squeeze(
        tf.image.flip_up_down(
            tf.image.flip_left_right(
                tf.expand_dims(x, axis=3, name="Sequence/expand_dim")
            )
        ),
        axis=3,
        name="Sequence/rc",
    )
    motif_embedding = tf.keras.layers.Conv1D(
        filters=motif_size,
        kernel_size=motif_length_max,
        strides=1,
        padding="same",
        name="Scanning/motif",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.HeNormal(),
        activation=motif_scanning_activation,
    )
    kmer_embedding = tf.keras.layers.Conv1D(
        filters=kmer_size,
        kernel_size=kmer_length_max,
        strides=1,
        padding="same",
        name="Scanning/kmer",
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Zeros(),
        activation=kmer_scanning_activation,
    )
    motif_embedding.trainable = motif_scanning_trainable
    kmer_embedding.trainable = kmer_scanning_trainable
    motif_swish = layers.Swish(name="Scanning/motif_swish_activation")
    kmer_swish = layers.Swish(name="Scanning/kmer_swish_activation")

    if fwd_rc_combine_type == "Add":
        y1 = tf.keras.layers.Add(name="Scanning/add_motif_fwd_rc")(
            [motif_swish(motif_embedding(x)), motif_swish(motif_embedding(x_rc))]
        )
    if fwd_rc_combine_type == "Concat":
        y1 = tf.keras.layers.Concatenate(axis=-1, name="Scanning/concat_motif_fwd_rc")(
            [motif_swish(motif_embedding(x)), motif_swish(motif_embedding(x_rc))]
        )

    if use_kmer_embedding:
        y2 = kmer_swish(kmer_embedding(x))
        y = tf.keras.layers.Concatenate(axis=-1, name="Scan_to_latent/concat")([y1, y2])
    else:
        y = y1

    # whether normalize embedding output
    if embedding_batch_norm:
        y = layers.Swish(name="Scan_to_latent/embedding/swish_activation")(
            tf.keras.layers.Activation(
                embedding_activation, name="Scan_to_latent/embedding/activation"
            )(
                tf.keras.layers.BatchNormalization(
                    name="Scan_to_latent/embedding/batch_norm"
                )(y)
            )
        )
    y = tf.keras.layers.Dropout(dropout, name="Scan_to_latent/embedding/dropout")(y)

    if wide_embedding_to_latent_repr == False:
        y = tf.keras.layers.Dense(
            units=latent_dim, name="Scan_to_latent/latent/latent_repr"
        )(y)
        y = tf.keras.layers.Activation(
            latent_activation, name="Scan_to_latent/latent/activation"
        )(
            tf.keras.layers.BatchNormalization(name="Scan_to_latent/latent/batch_norm")(
                y
            )
        )
    else:
        y = layers.InceptionBlock_type1_swish(
            activation=getattr(tf.keras.activations, inception_block_activation),
            combine_type=latent_inception_block_combine_type,
            conv_bottleneck_dims=[latent_dim // 2, latent_dim // 4],
            kernel_first=True,
            conv_kernel_length=[1, 20],
            bottleneck_dim=latent_dim,
            name="Scan_to_latent/latent/latent_inception",
        )(y)

    if use_repr_token:
        y = layers.Token(name="Add_representation_token")(y)

    if use_pos_embedding:
        y = layers.AddPositionEmbs(name="Add_pos_embedding")(y)

    for n in range(encoder_layer_number):
        if n == 0:
            y1, _ = layers.TransformerCrossAttBlock_swiglu(
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout_transformer,
                query_dense_activation=query_dense_activation,
                name=f"Transformer/encoder_block_{n}",
            )([y, y])
        else:
            y1, _ = layers.TransformerCrossAttBlock_swiglu(
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                dropout=dropout_transformer,
                query_dense_activation=query_dense_activation,
                name=f"Transformer/encoder_block_{n}",
            )([y1, y])

    y = tf.keras.layers.LayerNormalization(
        epsilon=1e-6, name="Transformer/encoder_norm"
    )(y1)

    if use_repr_token:
        y = tf.keras.layers.Lambda(lambda v: v[:, 0], name="Extract_token")(y)

    if repr_residual_expansion == False:
        y = tf.keras.layers.Dense(
            representation_dim,
            name="Prediction_repr",
            activation=output_activations[0],
            kernel_regularizer=tf.keras.regularizers.l2(l2=pred_repr_kernel_l2),
            activity_regularizer=tf.keras.regularizers.l2(l2=pred_repr_activity_l2),
        )(y)
    else:
        y = layers.DenseResidualBlock_swish(
            activation=getattr(
                tf.keras.activations, repr_residual_expansion_activation
            ),
            bottleneck_dim=representation_dim // 2,
            name="Prediction_repr_residual_expansion",
        )(y)

        y = tf.keras.layers.Dense(
            representation_dim,
            name="Prediction_repr",
            activation=output_activations[0],
            kernel_regularizer=tf.keras.regularizers.l2(l2=pred_repr_kernel_l2),
            activity_regularizer=tf.keras.regularizers.l2(l2=pred_repr_activity_l2),
        )(y)

    y = tf.keras.layers.Dense(
        1, name="Prediction_head", activation=output_activations[1]
    )(y)

    return tf.keras.models.Model(inputs=x, outputs=y, name=model_name)
