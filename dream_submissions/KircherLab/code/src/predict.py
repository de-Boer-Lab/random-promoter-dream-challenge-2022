import click
import numpy as np
import pandas as pd

import os

from sequence import SeqDreamChallange1D

from model import ResidualUnit1D_BN

import tensorflow as tf
from tensorflow.keras import Model, models, activations, layers
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation, GlobalAveragePooling1D, add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers, callbacks

# options
@click.command()
@click.option('--test',
              'test_tsv_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Test sequences')
@click.option('--model',
              'model_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Model file')
@click.option('--weights',
              'weights_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Weights file')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
@click.option(
    "--SavedModel",
    "savedmodel_dir",
    required=False,
    type=click.Path(),
    help="Folder with saved model in tensorflow SavedModel format",
)
@click.option('--adapter-trimming/--no-adapter-trimming',
              'adapter_trimming',
              default=False,
              help='Trim adapters from the sequence')
@click.option(
    "--fit_sequence",
    "fit_seq",
    is_flag=True,
    default=None,
    help="fit sequences to sequence length (110bp)"
)


def cli(test_tsv_file, model_file, weights_file, output_file, adapter_trimming, savedmodel_dir, fit_seq):



    strategy = tf.distribute.MirroredStrategy(devices=None)


    dl_test = SeqDreamChallange1D(test_tsv_file, ignore_targets= True, trim_adapters=adapter_trimming,fit_sequence=fit_seq)

    test_data = dl_test.load_all()

    with strategy.scope():

        if savedmodel_dir:
            model = models.load_model(savedmodel_dir)
            print(f"Model loaded from: {savedmodel_dir}")
        else:
            json_file = open(model_file, 'r')
            loaded_model_json = json_file.read()
            json_file.close()

            model = tf.keras.models.model_from_json(loaded_model_json)
            model.load_weights(weights_file)
            print(f"Model loaded from: {model_file} and {weights_file}")

        print("Final prediction")
        preds = model.predict(test_data["inputs"])
        pd.DataFrame(preds).to_csv(output_file, sep='\t', header=None, index = False)

if __name__ == '__main__':
    cli()
