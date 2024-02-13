import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from collections import OrderedDict


def onehot_encoding(
    sequence: str,
    length: int,
    alphabet: str = "ACGT",
    neutral_alphabet: str = "N",
    neutral_value=0.25,
    dtype=np.float32,
) -> np.ndarray:
    """One-hot encode sequence."""

    def to_uint8(string):
        return np.frombuffer(string.encode("ascii"), dtype=np.uint8)

    hash_table = np.zeros((np.iinfo(np.uint8).max, len(alphabet)), dtype=dtype)
    hash_table[to_uint8(alphabet)] = np.eye(len(alphabet), dtype=dtype)
    hash_table[to_uint8(neutral_alphabet)] = neutral_value
    hash_table = hash_table.astype(dtype)
    return hash_table[to_uint8(sequence)]


def DF2array(DF, sequence_name, length=142):
    """[summary]
    Transform sequences(DataFrame files) to np.array
    Args:
        DF ([type]): [description]
        sequence_name ([type]): [description]
        length (int, optional): [description]. Defaults to 142.

    Returns:
        [np.array]: Onehot encoding of sequences array[n_seq, seq_length, 4]
    """
    n_seq = len(DF[sequence_name])
    sequence_np = np.zeros((n_seq, length, 4))
    for i, s in tqdm(enumerate(DF[sequence_name])):
        sequence_np[i, 0:len(s), :] = onehot_encoding(s, length)
    return sequence_np


def DF2array_RC(DF, sequence_name, length=142):
    """
    Returns:
        [np.array]: Onehot encoding of sequences array with reverse compelement [n_seq, 4, seq_length*2]
    """
    n_seq = len(DF[sequence_name])
    sequence_np = np.zeros((n_seq, 4, length))
    for i, s in tqdm(enumerate(DF[sequence_name])):
        sequence_np[i, :, 0:len(s)] = np.transpose(onehot_encoding(s, length))
    reverse = np.flip(sequence_np, axis=2)
    reverse_complement = np.flip(reverse, axis=1)
    
    return np.concatenate((sequence_np, reverse_complement), axis=2)


def generate_submission(Y_test_pred: np.ndarray, sample_path=None, output_path=None):
    """[summary]
    Generate json file for submission
    Args:
        Y_test_pred: prediction output for test data in numpy array format
    """
    with open(sample_path, 'r') as f:
        ground = json.load(f)

    indices = np.array([int(indice) for indice in list(ground.keys())])

    PRED_DATA = OrderedDict()

    for i in indices:
    #Y_pred is an numpy array of dimension (71103,) that contains your
    #predictions on the test sequences
        PRED_DATA[str(i)] = float(Y_test_pred[i])
        

    def dump_predictions(prediction_dict, prediction_file):
        with open(prediction_file, 'w') as f:
            json.dump(prediction_dict, f)

    dump_predictions(PRED_DATA, output_path)



def generate_submission_txt(Y_test_pred: np.ndarray, input_path=None, output_path=None):
    """[summary]
    Generate txt file for submission
    Args:
        Y_test_pred: prediction output for test data in numpy array format
    """
    test_data = pd.read_csv(input_path, header=None, sep="\t")
    test_data.columns = ["sequence", "exp"]
    test_data["exp"] = Y_test_pred

    test_data.to_csv(output_path, sep='\t', index=False, header=False)
