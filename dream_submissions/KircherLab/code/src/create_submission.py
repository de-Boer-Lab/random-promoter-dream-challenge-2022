import click
import pandas as pd
import json
import numpy as np
from collections import OrderedDict
from prediction import max_value, weighted_mean

transformation = {
        "maxvalue" : max_value,
        "weightedmean" : weighted_mean
    }

# options
@click.command()
@click.option('--predicted',
              'predicted_tsv_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='predicted file')
@click.option('--original',
              'original_tsv_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Original file')
@click.option('--sample-submission',
              'sample_submission',
              required=False,
              type=(click.Path(exists=True, readable=True), click.Path(writable=True)),
              help='sample submission file input and output')
@click.option('--method',
              'method',
              default='maxvalue',
              type=click.Choice(transformation.keys(), case_sensitive=False),
              help='Method for computing the final bucket out of predictions')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Final submission file')
@click.option(
    "--model-mode",
    "model_mode",
    type=click.Choice(
        [
            "classification",
            "regression",
        ],
        case_sensitive=False,
    ),
    default='classification',
    required=False,
    help="Choise of model type",
)

def cli(predicted_tsv_file, original_tsv_file, method, output_file, sample_submission, model_mode):

    df  = pd.read_csv(original_tsv_file, header=None, sep="\t")

    predictions = pd.read_csv(predicted_tsv_file, header=None, sep="\t")

    if model_mode == 'classification':
        df[1] = predictions.apply(lambda prediction: transformation[method](prediction),axis=1)
    elif model_mode == 'regression':
        df[1] = predictions
        
    df.to_csv(output_file, sep='\t', header=None, index = False)

    if sample_submission:
        with open(sample_submission[0], 'r') as f:
            ground = json.load(f)

        indices = np.array([int(indice) for indice in list(ground.keys())])

        PRED_DATA = OrderedDict()
        for i in indices:
        #Y_pred is an numpy array of dimension (71103,) that contains your
        #predictions on the test sequences
            PRED_DATA[str(i)] = float(df.iloc[i,1])
        
        def dump_predictions(prediction_dict, prediction_file):
            with open(prediction_file, 'w') as f:
                json.dump(prediction_dict, f)
        
        dump_predictions(PRED_DATA, sample_submission[1])

if __name__ == '__main__':
    cli()

             