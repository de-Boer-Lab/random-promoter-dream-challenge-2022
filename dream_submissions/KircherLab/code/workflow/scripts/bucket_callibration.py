import click
import numpy as np
import pandas as pd
import random
import os
#from set_seed import set_global_determinism

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
              help='input sequence file')
@click.option('--max-bucket-size',
              'bucket_size',
              required=False,
              type=int,
              help='Maximum size of the bucket')
@click.option('--bucket-frac',
              'bucket_frac',
              required=False,
              type=float,
              help='Fraction of the bucket')
@click.option('--round-digits',
              'round_digits',
              required=False,
              default=2,
              type=int,
              help='Rounding GC content using x digets')
@click.option('--gc-correction/--no-gc-correction',
              'gc_correction',
              default=False,
              help='Use GC correction per bucket')
@click.option('--with-replacement/--without-replacement',
              'with_replacement',
              default=False,
              help='Sample with or without replacement')
@click.option('--output',
              'output_file',
              required=True,
              type=click.Path(writable=True),
              help='Prediction output file')
@click.option('--seed',
              'seed',
              required=False,
              type=int,
              default=None,
              help='seed for randomness.')              



def cli(input_file, bucket_size, bucket_frac, with_replacement, gc_correction, output_file, round_digits, seed):
    if seed:
        set_global_determinism(seed)

    print("loading data")
    df  = pd.read_csv(input_file, header=None, sep="\t")

    if gc_correction:
        df["GC"] = round((df[0].str.count("G") + df[0].str.count("C"))/df[0].str.len(), round_digits)

        gc_dist = df.groupby('GC')['GC'].agg(['count'])
        gc_dist['gc_weight'] = gc_dist['count']/df.shape[0]
        print(gc_dist)
        gc_dist = gc_dist.reset_index()[['GC','gc_weight']]

        
        
        df = df.join(gc_dist.set_index('GC'),on='GC')
    

    if (bucket_frac):
        df = df.groupby(list(range(2,20))).sample(frac=bucket_frac,replace=with_replacement, weights="gc_weight" if gc_correction else None).reset_index()
    if (bucket_size):
        # do only sampling when bucket size is >= bucket_size
        buckets = df.groupby(list(range(2,20)))[0].agg(['count'])

        print(buckets)
        if (buckets["count"] >= bucket_size).any():
            print("yay")
            df_sample = df.set_index(list(range(2,20))).join(buckets[(buckets['count']>=bucket_size)],how='inner')
            df_sample = df_sample.reset_index().groupby(list(range(2,20))).sample(n=bucket_size,replace=with_replacement, weights="gc_weight" if gc_correction else None)
            # join sampled and buckets without sampling
            df = pd.concat([df_sample,df.set_index(list(range(2,20))).join(buckets[(buckets['count']<bucket_size)],how='inner').reset_index()]).reset_index()

    df_final = df[list(range(0,20))]
        
    df_final.to_csv(output_file, sep='\t', header=None, index = False)
        

if __name__ == '__main__':
    cli()
