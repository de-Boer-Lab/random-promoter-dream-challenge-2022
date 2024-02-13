import copy
import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

from scipy.stats import spearmanr
from sklearn.utils import class_weight


def seq_to_one_hot(
    seq,
    model_unknown=False,
    fix_length=None,
    centering=True,
    trim_pad_unique_oligo=True,
    oligo_random_padding=True,
):
    from sklearn.preprocessing import OneHotEncoder

    encoder = OneHotEncoder(handle_unknown="ignore", dtype=np.float32)
    encoder.fit(np.array(["A", "C", "G", "T"]).reshape(-1, 1))

    seq_container = []
    encodered_seq_container = []

    unique_oligo_size = 80
    constant_region_upstream_size = 17
    constant_region_downstream_size = 13

    for _ in seq:
        _upper_case_seq = list(_.upper())
        if trim_pad_unique_oligo:
            _constant_region_upstream = _upper_case_seq[:constant_region_upstream_size]
            _constant_region_downstream = _upper_case_seq[
                -constant_region_downstream_size:
            ]
            unique_oligo = _upper_case_seq[
                constant_region_upstream_size:-constant_region_downstream_size
            ]
            # too short, add random or N padding
            if len(unique_oligo) < unique_oligo_size:
                left_padding_size = (unique_oligo_size - len(unique_oligo)) // 2
                right_padding_size = (
                    unique_oligo_size - len(unique_oligo) - left_padding_size
                )
                if oligo_random_padding:
                    unique_oligo = (
                        list(
                            np.random.choice(
                                ["A", "C", "G", "T"],
                                left_padding_size,
                            )
                        )
                        + unique_oligo
                        + list(
                            np.random.choice(
                                ["A", "C", "G", "T"],
                                right_padding_size,
                            )
                        )
                    )
                else:
                    unique_oligo = (
                        list(np.random.choice(["N"], left_padding_size))
                        + unique_oligo
                        + list(np.random.choice(["N"], right_padding_size))
                    )

            # too long, trim off by centering
            else:
                _oligo_size = len(_upper_case_seq) - 30
                _offset = (_oligo_size - 80) // 2
                unique_oligo = unique_oligo[
                    _offset : _offset + unique_oligo_size
                ]  # extract center part

            seq_container.append(
                np.array(
                    _constant_region_upstream
                    + unique_oligo
                    + _constant_region_downstream
                )
            )
        else:
            seq_container.append(_upper_case_seq)

    for i in range(len(seq_container)):
        _encoded = encoder.transform(
            np.array(seq_container[i]).reshape(-1, 1)
        ).todense()

        if fix_length != None:
            # 0 padding
            if fix_length > np.shape(_encoded)[0]:
                if centering:
                    _encoded = np.pad(
                        _encoded,
                        (
                            (
                                (fix_length - np.shape(_encoded)[0]) // 2,
                                fix_length
                                - np.shape(_encoded)[0]
                                - (fix_length - np.shape(_encoded)[0]) // 2,
                            ),
                            (0, 0),
                        ),
                        "constant",
                        constant_values=(0.0, 0.0),
                    )

        if model_unknown:
            _encoded[np.where(np.sum(_encoded, axis=1) == 0)[0]] = 0.25

        encodered_seq_container.append(_encoded.tolist())

    return np.squeeze(np.array(encodered_seq_container,dtype=np.float32))


class TrainGenerator(tf.keras.utils.Sequence):
    "Generates data for tf.keras"

    def __init__(
        self,
        df,
        batch_size=1024,
        shuffle=True,
        fix_length=150,
        trim_pad_unique_oligo=True,
        oligo_random_padding=True,
        model_unknown=False,
        centering=True,
    ):
        "Initialization"

        self.fix_length = fix_length
        self.trim_pad_unique_oligo = trim_pad_unique_oligo
        self.oligo_random_padding = oligo_random_padding
        self.model_unknown = model_unknown
        self.centering = centering
        self.df = df.copy()
        self.df.reset_index(inplace=True, drop=True)
        self.total_size = self.df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.total_size / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(self.total_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"
        # Initialization
        X = seq_to_one_hot(
            seq=self.df.iloc[indexes]["seq"].values,
            fix_length=self.fix_length,
            trim_pad_unique_oligo=self.trim_pad_unique_oligo,
            oligo_random_padding=self.oligo_random_padding,
            model_unknown=self.model_unknown,
            centering=self.centering,
        )
        y = self.df.iloc[indexes]["y"].values

        return X, y


class TrainGenerator_with_sample_weights(tf.keras.utils.Sequence):
    "Generates data for tf.keras"

    def __init__(
        self,
        df,
        batch_size=1024,
        shuffle=True,
        fix_length=150,
        trim_pad_unique_oligo=True,
        oligo_random_padding=True,
        model_unknown=False,
        centering=True,
    ):
        "Initialization"

        self.fix_length = fix_length
        self.trim_pad_unique_oligo = trim_pad_unique_oligo
        self.oligo_random_padding = oligo_random_padding
        self.model_unknown = model_unknown
        self.centering = centering
        self.df = df.copy()
        self.df.reset_index(inplace=True, drop=True)
        self.total_size = self.df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.total_size / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        # Generate data
        X, y, sample_weights = self.__data_generation(indexes)

        return X, y, sample_weights

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(self.total_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"
        # Initialization
        X = seq_to_one_hot(
            seq=self.df.iloc[indexes]["seq"].values,
            fix_length=self.fix_length,
            trim_pad_unique_oligo=self.trim_pad_unique_oligo,
            oligo_random_padding=self.oligo_random_padding,
            model_unknown=self.model_unknown,
            centering=self.centering,
        )
        y = self.df.iloc[indexes]["y"].values
        sample_weights = self.df.iloc[indexes]["sample_weights"].values
        return X, y, sample_weights


class PredictGenerator(tf.keras.utils.Sequence):
    "Generates data for tf.keras"

    def __init__(
        self,
        df,
        batch_size=1024,
        shuffle=True,
        fix_length=150,
        trim_pad_unique_oligo=True,
        oligo_random_padding=True,
        model_unknown=False,
        centering=True,
    ):
        "Initialization"

        self.fix_length = fix_length
        self.trim_pad_unique_oligo = trim_pad_unique_oligo
        self.oligo_random_padding = oligo_random_padding
        self.model_unknown = model_unknown
        self.centering = centering
        self.df = df.copy()
        self.df.reset_index(inplace=True, drop=True)
        self.total_size = self.df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        "Updates indexes after each epoch"
        self.indexes = np.arange(self.total_size)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.total_size / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        # Generate indexes of the batch
        if index < self.__len__() - 1:
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
        else:  # last batch, get all the remaining ones
            indexes = self.indexes[index * self.batch_size :]
        # Generate data
        X = self.__data_generation(indexes)

        return X

    def __data_generation(self, indexes):
        "Generates data containing batch_size samples"
        # Initialization
        X = seq_to_one_hot(
            seq=self.df.iloc[indexes]["seq"].values,
            fix_length=self.fix_length,
            trim_pad_unique_oligo=self.trim_pad_unique_oligo,
            oligo_random_padding=self.oligo_random_padding,
            model_unknown=self.model_unknown,
            centering=self.centering,
        )

        return X


class TrainDataGen:
    def __init__(
        self,
        df,
        batch_size=1024,
        shuffle=True,
        fix_length=150,
        trim_pad_unique_oligo=True,
        oligo_random_padding=True,
        model_unknown=False,
        centering=True,
    ):
        "Initialization"
        self.fix_length = fix_length
        self.trim_pad_unique_oligo = trim_pad_unique_oligo
        self.oligo_random_padding = oligo_random_padding
        self.model_unknown = model_unknown
        self.centering = centering
        self.df = df.copy()
        self.df.reset_index(inplace=True, drop=True)
        self.total_size = self.df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(self.total_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.total_size / self.batch_size))

    def get_item(self, index):
        X, y = self.__data_generation(index)
        return X, y

    def __call__(self):
        for index in np.arange(self.total_size):
            yield self.get_item(index)

    def __data_generation(self, index):
        "Generates data containing batch_size samples"
        # Initialization
        X = seq_to_one_hot(
            seq=self.df.iloc[index : index + 1]["seq"].tolist(),
            fix_length=self.fix_length,
            trim_pad_unique_oligo=self.trim_pad_unique_oligo,
            oligo_random_padding=self.oligo_random_padding,
            model_unknown=self.model_unknown,
            centering=self.centering,
        )
        y = self.df.iloc[index]["y"]

        return X, y


class TrainDataGen_with_sample_weights:
    def __init__(
        self,
        df,
        batch_size=1024,
        shuffle=True,
        fix_length=150,
        trim_pad_unique_oligo=True,
        oligo_random_padding=True,
        model_unknown=False,
        centering=True,
    ):
        "Initialization"
        self.fix_length = fix_length
        self.trim_pad_unique_oligo = trim_pad_unique_oligo
        self.oligo_random_padding = oligo_random_padding
        self.model_unknown = model_unknown
        self.centering = centering
        self.df = df.copy()
        self.df.reset_index(inplace=True, drop=True)
        self.total_size = self.df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(self.total_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.total_size / self.batch_size))

    def get_item(self, index):
        X, y, sample_weights = self.__data_generation(index)
        return X, y, sample_weights

    def __call__(self):
        for index in np.arange(self.total_size):
            yield self.get_item(index)

    def __data_generation(self, index):
        "Generates data containing batch_size samples"
        # Initialization
        X = seq_to_one_hot(
            seq=self.df.iloc[index : index + 1]["seq"].tolist(),
            fix_length=self.fix_length,
            trim_pad_unique_oligo=self.trim_pad_unique_oligo,
            oligo_random_padding=self.oligo_random_padding,
            model_unknown=self.model_unknown,
            centering=self.centering,
        )
        y = self.df.iloc[index]["y"]
        sample_weights=self.df.iloc[index]["sample_weights"]

        return X, y, sample_weights


class PredictDataGen:
    def __init__(
        self,
        df,
        batch_size=1024,
        shuffle=True,
        fix_length=150,
        trim_pad_unique_oligo=True,
        oligo_random_padding=True,
        model_unknown=False,
        centering=True,
    ):
        "Initialization"
        self.fix_length = fix_length
        self.trim_pad_unique_oligo = trim_pad_unique_oligo
        self.oligo_random_padding = oligo_random_padding
        self.model_unknown = model_unknown
        self.centering = centering
        self.df = df.copy()
        self.df.reset_index(inplace=True, drop=True)
        self.total_size = self.df.shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(self.total_size)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        "Denotes the number of batches per epoch"
        return int(np.floor(self.total_size / self.batch_size))

    def get_item(self, index):
        X = self.__data_generation(index)
        return X

    def __call__(self):
        for index in np.arange(self.total_size):
            yield self.get_item(index)

    def __data_generation(self, index):
        "Generates data containing batch_size samples"
        # Initialization
        X = seq_to_one_hot(
            seq=self.df.iloc[index : index + 1]["seq"].tolist(),
            fix_length=self.fix_length,
            trim_pad_unique_oligo=self.trim_pad_unique_oligo,
            oligo_random_padding=self.oligo_random_padding,
            model_unknown=self.model_unknown,
            centering=self.centering,
        )

        return X


def insert_kernels(model, target_layer_name, kernel):
    for layer in model.layers:
        if layer.name == target_layer_name:
            weight_shape = np.shape(layer.get_weights())
            if weight_shape == np.shape(kernel):
                layer.set_weights(kernel)
            elif weight_shape[1:] == np.shape(kernel):
                kernel = np.expand_dims(kernel, axis=0)
                layer.set_weights(kernel)
            else:
                print("kernel size doesn't match, please check!")
                raise ValueError


class motif(object):
    def __init__(self):
        self.length = 0
        self.matrix = []
        self.r_matrix = []
        self.tf = ""
        self.motif_id = ""

    def load_motif(self, motif_path, pfm_format=False):
        """
        :param motif_path:
        :return:
        """
        if pfm_format != True:
            file = os.path.join(motif_path, self.motif_id + ".txt")
            df = pd.read_csv(file, header=0, index_col=0, sep="\t")
        else:
            file = os.path.join(motif_path, self.motif_id + ".pfm")
            df = pd.read_csv(file, header=None, index_col=0, sep="\t")
            df = df.T[["A", "C", "G", "T"]]

        self.length = df.shape[0]
        mat = df.values
        mat = mat.astype("float32")
        mat2 = copy.copy(mat)
        max_value = 0.0
        for i in range(df.shape[0]):
            max_value_temp = 0.0
            for j in range(df.shape[1]):
                if mat[i, j] < 0.001:
                    mat[i, j] = 0.001
                mat2[i, j] = np.log2(mat[i, j] / 0.25)
                if mat2[i, j] >= max_value_temp:
                    max_value_temp = mat2[i, j]
            max_value += max_value_temp
        self.matrix = (
            mat2 / max_value
        )  # log2 transformed, max value normalized scoring matrix
        # print(self.motif_id, max_value, "\t")
        self.r_matrix = self.matrix.T[::-1].T[::-1]

    @staticmethod
    def load_motif_list(motif_path, motif_pair_list):
        """
        :param motif_path:
        :param motif_pair_list: [[motif_id1,tf1],[motif_id2,tf1],...]
        :return:
        """
        motif_vec = []
        for _ in motif_pair_list:
            m = motif()
            m.motif_id = _[0]
            m.tf = _[1]

            m.load_motif(motif_path=motif_path)
            motif_vec.append(m)
        return motif_vec

    @staticmethod
    def load_kmers(_kmer_list):
        motif_vec = []
        from sklearn.preprocessing import OneHotEncoder
        import numpy as np

        ohe = OneHotEncoder(handle_unknown="ignore")
        ohe.fit(np.array(["A", "C", "G", "T"]).reshape(-1, 1))

        for _ in _kmer_list:
            _input = np.array(list(_)).reshape(-1, 1)
            _matrix = ohe.transform(_input).todense()

            m = motif()
            m.motif_id = _  # e.g., ACGTACGT
            m.tf = "{}mer".format(len(_))
            df = pd.DataFrame(_matrix)
            m.length = df.shape[0]
            mat = df.values
            mat = mat.astype("float32")
            mat2 = copy.copy(mat)
            max_value = 0.0
            for i in range(df.shape[0]):
                max_value_temp = 0.0
                for j in range(df.shape[1]):
                    if mat[i, j] < 0.001:
                        mat[i, j] = 0.001
                    mat2[i, j] = np.log2(mat[i, j] / 0.25)
                    if mat2[i, j] >= max_value_temp:
                        max_value_temp = mat2[i, j]
                max_value += max_value_temp
            m.matrix = mat2 / max_value
            # print(m.motif_id, max_value, "\t")
            m.r_matrix = m.matrix.T[::-1].T[::-1]

            motif_vec.append(m)
        return motif_vec

    @staticmethod
    def motif_to_kernel(motif_vec, include_rc=False, predefined_length=0):
        """
        return zero padded motif vectors for a list of known motifs
        """

        # matrix size
        if include_rc:
            matrix_size = 2 * int(len(motif_vec))  # including forward and reverse
        else:
            matrix_size = int(len(motif_vec))

        # set max length
        max_length = 0
        for i in motif_vec:
            if i.length >= max_length:
                max_length = i.length

        if predefined_length > max_length:
            max_length = predefined_length

        print("max motif length: ", max_length)
        print("motif size: ", len(motif_vec))
        print("kernel size:", matrix_size)
        tmp_container = []
        for i in motif_vec:
            _starting_pos = max_length // 2 - i.length // 2

            # forward
            tmp_matrix = np.zeros(shape=(max_length, 4))
            for x in range(i.length):
                for y in range(4):
                    tmp_matrix[_starting_pos + x, y] = i.matrix[x, y]
            tmp_container.append(tmp_matrix)
            # reverse complement
            if include_rc:
                tmp_matrix = np.zeros(shape=(max_length, 4))
                for x in range(i.length):
                    for y in range(4):
                        tmp_matrix[_starting_pos + x, y] = i.r_matrix[x, y]
                tmp_container.append(tmp_matrix)
        motif_kernel = np.moveaxis(np.array(tmp_container), 0, 2)
        motif_kernel = motif_kernel.reshape((1, max_length, 4, matrix_size))

        return np.array(motif_kernel)

    def __eq__(self, other):
        return other == self.motif_id


def gen_kmer_list(k):
    kmer_list = []

    def gen_kmer(container, set, prefix, n, k):
        if k == 0:
            container.append(prefix)
            return

        for i in range(n):
            newPrefix = prefix + set[i]
            gen_kmer(container, set, newPrefix, n, k - 1)

    gen_kmer(kmer_list, "ACGT", "", 4, k)
    return kmer_list

# TPU compatible
def Pearson_r(y_true, y_pred):
    return tfp.stats.correlation(y_pred, y_true, name="Pearson_r")

# TPU compatible
def RSquare_CoD1(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# TPU compatible
def Weighted_metrics_tpu(y_true, y_pred):
    def true_fn(_y_true, _y_pred):
        return 0.4 * tf.sqrt(tf.abs(RSquare_CoD1(_y_true, _y_pred))) + 0.6 * Pearson_r(_y_true, _y_pred)
    def false_fn(_y_true, _y_pred):
        return Pearson_r(_y_true, _y_pred)
    return tf.cond(
           RSquare_CoD1(y_true, y_pred)>0,
           lambda: true_fn(y_true, y_pred),
           lambda: false_fn(y_true, y_pred)
    )

# not TPU compatible
def Spearman_r(y_true, y_pred):
    return tf.py_function(
        func=spearmanr,
        inp=[tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)],
        Tout=tf.float32,
        name="Spearman_r"
    )

# not TPU compatible
def Weighted_metrics(y_true, y_pred):
    return (
        0.2 * tf.sqrt(tf.abs(RSquare_CoD1(y_true, y_pred)))
        + 0.4 * Pearson_r(y_true, y_pred)
        + 0.4 * Spearman_r(y_true, y_pred)
    )

RSquare_CoD = tfa.metrics.r_square.RSquare(
    dtype=tf.float32, y_shape=(1,), name="RSquare_CoD"
)

def add_sample_weights(
    df, limit=3, class_weight_dict=None, class_weight_type="balanced"
):
    # rounded cls label
    df["cls"] = [round(x) for x in df["y"]]

    if class_weight_dict == None:
        class_weights = class_weight.compute_class_weight(
            class_weight=class_weight_type, classes=np.unique(df["cls"]), y=df["cls"]
        )
        class_weights = class_weights / np.min(class_weights)
        class_weights = [x if x < limit else limit for x in class_weights]

        class_weight_dict = dict(zip(np.unique(df["cls"]), class_weights))

    df["sample_weights"] = [class_weight_dict[x] for x in df["cls"]]

    return df, class_weight_dict
