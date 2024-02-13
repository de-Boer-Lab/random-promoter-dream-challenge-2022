import argparse
import numpy as np
import pandas as pd
from scipy import stats

# For whole data prediction

# loc : mean of standardized expression from train data
# scale : 1
def postprocess(x):
    return stats.norm.cdf(x, loc=0.06201038511068657, scale=1) 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    with open(args.input, 'r') as inFile, open(args.output, 'w') as outFile:
        df = pd.read_csv(inFile, sep="\t", header=None, names=['sequence', 'measured_expression'])

        predictions = np.array(list(df['measured_expression']))
        df['flattened'] = np.array(list(map(postprocess, predictions)))
        df = df.drop(columns='measured_expression')
        df.to_csv(outFile, sep='\t', header=False, index=False)
