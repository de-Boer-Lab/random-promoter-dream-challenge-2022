import tensorflow as tf
from tensorflow.keras import backend as K

# one-hot encoding: tensorflow genome ops
# https://github.com/tensorflow/io/blob/0b70504f1425504e1c0149ee8cb8fd55cfb9082b/tensorflow_io/python/ops/genome_ops.py 
@tf.function()
def _nucleotide_to_onehot(nucleotide):
    """Encodes a nucleotide using a one hot encoding."""
    a_onehot = tf.constant([1, 0, 0, 0])
    c_onehot = tf.constant([0, 1, 0, 0])
    g_onehot = tf.constant([0, 0, 1, 0])
    t_onehot = tf.constant([0, 0, 0, 1])
    error_onehot = tf.constant([1, 1, 1, 1])

    if tf.math.equal(nucleotide, tf.constant(b"A")):  # pylint: disable=no-else-return
        return a_onehot
    elif tf.math.equal(nucleotide, tf.constant(b"C")):
        return c_onehot
    elif tf.math.equal(nucleotide, tf.constant(b"G")):
        return g_onehot
    elif tf.math.equal(nucleotide, tf.constant(b"T")):
        return t_onehot
    else:
        return error_onehot


@tf.function()
def sequences_to_onehot(sequences):
    """Convert DNA sequences into a one hot nucleotide encoding.
    Each nucleotide in each sequence is mapped as follows:
    A -> [1, 0, 0, 0]
    C -> [0, 1, 0, 0]
    G -> [0 ,0 ,1, 0]
    T -> [0, 0, 0, 1]
    If for some reason a non (A, T, C, G) character exists in the string, it is
    currently mapped to a error one hot encoding [1, 1, 1, 1].
    Args:
        sequences: A tf.string tensor where each string represents a DNA sequence
    Returns:
        tf.RaggedTensor: The output sequences with nucleotides one hot encoded.
    """
    all_onehot_nucleotides = tf.TensorArray(
        dtype=tf.int32, size=0, dynamic_size=True)
    sequence_splits = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)

    global_nucleotide_idx = 0
    sequence_splits = sequence_splits.write(
        sequence_splits.size(), global_nucleotide_idx
    )

    for sequence in sequences:
        for nucleotide in tf.strings.bytes_split(sequence):
            all_onehot_nucleotides = all_onehot_nucleotides.write(
                global_nucleotide_idx, _nucleotide_to_onehot(nucleotide)
            )
            global_nucleotide_idx += 1
        sequence_splits = sequence_splits.write(
            sequence_splits.size(), global_nucleotide_idx
        )
    return tf.RaggedTensor.from_row_splits(
        values=all_onehot_nucleotides.stack(),
        row_splits=tf.cast(sequence_splits.stack(), tf.int64),
    ).to_tensor()


def seq_distance(a, b):
    size = len(a)
    count = 0
    i = 0

    while i < size:
        if a[i] != b[i]:
            count += 1
        i += 1
    return count


def r_square(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def reverse_complement(fwd_seq):
    return fwd_seq[::-1].translate(str.maketrans('ACTG', 'TGAC'))
