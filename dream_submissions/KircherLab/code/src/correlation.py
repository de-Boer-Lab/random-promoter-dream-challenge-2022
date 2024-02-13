import click
import pandas as pd
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
@click.option('--method',
              'method',
              default='maxvalue',
              type=click.Choice(transformation.keys(), case_sensitive=False),
              help='Method for computing the final bucket out of predictions')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Spearman and pearson correlation')
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

def cli(predicted_tsv_file, original_tsv_file, method, output_file, model_mode):

    original_df  = pd.read_csv(original_tsv_file, header=None, sep="\t")

    original_df = original_df[original_df.iloc[:,0].apply(lambda x: len(str(x))) == 110]

    original_df = original_df.reset_index(drop=True)

    predictions = pd.read_csv(predicted_tsv_file, header=None, sep="\t")

    if model_mode == "classification":
        result = pd.concat([original_df, predictions.apply(lambda prediction: transformation[method](prediction),axis=1)], axis=1)
    elif model_mode == "regression":
        result = pd.concat([original_df, predictions], axis=1)
    
    result = result.iloc[:,[1,20]]

    
    cor_result = pd.DataFrame([{"spearman":result.corr(method="spearman").iloc[0,1],"pearson": result.corr(method="pearson").iloc[0,1]}])

    print(cor_result)

    cor_result.to_csv(output_file, sep='\t', index = False)

if __name__ == '__main__':
    cli()

             
