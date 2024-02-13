import pandas as pd
import click
from sklearn.model_selection import train_test_split
#from src.set_seed import set_global_determinism

import numpy as np
import os
import random
import pandas as pd

SEED = 0

def set_seeds(seed=SEED):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)



# options
@click.command()
@click.option('--input',
              'input_file',
              required=True,
              type=click.Path(exists=True, readable=True),
              help='Input file for splitting data. ')
@click.option('--split-validation',
              'split_validation',
              required=False,
              default=0.1,
              type=float,
              help='Proportion of data for validation set.')
@click.option('--split-test',
              'split_test',
              required=False,
              default = 0.2,
              type=float,
              help='Proportion of data for test set.')
@click.option('--output-train',
              'output_file_train',
              required=True,
              type=click.Path(writable=True),
              help='Output training file after split.')
@click.option('--output-test',
              'output_file_test',
              required=True,
              type=click.Path(writable=True),
              help='Output test file after split.')

@click.option('--output-val',
              'output_file_validation',
              required=True,
              type=click.Path(writable=True),
              help='Output test file after split.')
@click.option('--seed',
              'seed',
              required=False,
              type=int,
              default=None,
              help='seed for randomness.')

def cli(input_file, split_validation, split_test, output_file_train, output_file_test, output_file_validation, seed):
    if seed:
        set_global_determinism(seed)

    # input data, add columns and round all effect values
    data_ori = pd.read_csv(input_file, sep='\t', header=None)
    data_ori.columns = ["raw_sequence","effect"]
    
    print('rounding [effect] column...')
    # round effect values
    data_ori['effect_round'] = data_ori['effect'].astype(int) 
    
    
    # IMPLEMENT: distribution-based sampling
    #
    #
    
    print('splitting data...')
    # split validation data, set aside 20%
    X_train, X_test, y_train, y_test = train_test_split(data_ori.iloc[:,0:2], data_ori.iloc[:,2],
        test_size=split_test, shuffle = True, random_state = 1, stratify=data_ori.iloc[:,2])

    # split train and test data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
        test_size=split_validation, shuffle = True, random_state = 1, stratify=y_train)
    
    # function to combine dataframe arrays to a single dataframe
    def combine_cols(data_x, data_y):
        data_combined = pd.concat([data_x, data_y], axis=1)
        return data_combined
    
    print('combining columns...train')
    data_train = combine_cols(X_train, y_train)
    print('combining columns...test')
    data_test =  combine_cols(X_test, y_test)
    print('combining columns...validation')
    data_validation = combine_cols(X_val, y_val)
    
    # function to binarize labels // requires column names  
    def binarize_effect(data_file):
        bin_file = pd.get_dummies(data_file['effect_round'])
        data_file = combine_cols(data_file, bin_file)
        return data_file
    
    print('binarizing files...')
    data_train = binarize_effect(data_train)
    data_test = binarize_effect(data_test)
    data_validation = binarize_effect(data_validation)
    
    print('dropping extra data')
    data_train.drop('effect_round', axis=1, inplace=True)
    data_test.drop('effect_round', axis=1, inplace=True)
    data_validation.drop('effect_round', axis=1, inplace=True)
    
    print('saving files...')
    data_train.to_csv(output_file_train, sep='\t', index=False, header=None, compression='gzip')
    data_test.to_csv(output_file_test, sep='\t', index=False, header=None, compression='gzip')
    data_validation.to_csv(output_file_validation, sep='\t', index=False, header=None, compression='gzip')

    
if __name__ == '__main__':
    cli()
