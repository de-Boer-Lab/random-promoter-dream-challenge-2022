#!/home/dingml/anaconda3/envs/expressBert39/bin/python3
import json
import warnings
from collections import OrderedDict
from textwrap import indent
import numpy as np
import torch
import os
import argparse

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir

with open("sample_submission.json", mode='r') as f:
    ground = json.load(f)


indices = np.array([int(indice) for indice in list(ground.keys())])
PRED_DATA = OrderedDict()

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-o", required=True)
    p.add_argument('--test_path', type=str, required=True)
    return p

def dump_predictions(prediction_dict, prediction_file):
    with open(prediction_file, mode='w') as f:
        json.dump(prediction_dict, f)

if __name__ == "__main__":
    args = get_args().parse_args()
    outdir = make_directory(args.o)
    indices = np.array([int(indice) for indice in list(ground.keys())])
    PRED_DATA = OrderedDict()
    Y_pred = torch.load(os.path.join(args.test_path, "test.pt"))
    for i in indices:
        PRED_DATA[str(i)] = float(Y_pred[i])
    dump_predictions(PRED_DATA, os.path.join(outdir, "pred.json"))